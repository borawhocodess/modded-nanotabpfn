#!/usr/bin/env python3
"""
scripts/attention.py

Visualise per-layer per-head row-axis attention maps for a NanoTabPFN checkpoint.

Usage:
    python scripts/attention.py path/to/ckpt.pth
    python scripts/attention.py path/to/ckpt.pth --task 363629 --train-size 80 --test-size 30
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openml
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OrdinalEncoder
from torch import nn


# ---------------------------------------------------------------------------
# Model (inlined from train_nano.py to avoid training side-effects)
# ---------------------------------------------------------------------------

class LowerPrecisionRMSNorm(nn.RMSNorm):
    def forward(self, x):
        if x.dtype in (torch.float16, torch.bfloat16):
            with torch.amp.autocast("cuda", enabled=False):
                return super().forward(x)
        return super().forward(x)


class ThinkingRows(nn.Module):
    def __init__(self, num_thinking_rows: int, e: int):
        super().__init__()
        self.num_thinking_rows = num_thinking_rows
        self.row_tokens = nn.Parameter(torch.empty(num_thinking_rows, e))
        nn.init.normal_(self.row_tokens)

    def forward(self, x, sep):
        b, r, c, e = x.shape
        thinking = self.row_tokens.unsqueeze(0).unsqueeze(2).expand(b, -1, c, -1)
        x = torch.cat([thinking, x], dim=1)
        sep = sep + self.num_thinking_rows
        return x, sep


class FeatureEncoder(nn.Module):
    def __init__(self, e):
        super().__init__()
        self.linear_layer = nn.Linear(1, e)

    def forward(self, x, sep):
        x = x.unsqueeze(-1)
        mean = x[:, :sep].mean(dim=1, keepdim=True)
        std = x[:, :sep].std(dim=1, keepdim=True) + 1e-8
        x = (x - mean) / std
        x = torch.clip(x, min=-100, max=100)
        return self.linear_layer(x)


class TargetEncoder(nn.Module):
    def __init__(self, e):
        super().__init__()
        self.linear_layer = nn.Linear(1, e)

    def forward(self, y_train, num_rows):
        mean = y_train.mean(dim=1, keepdim=True)
        padding = mean.repeat(1, num_rows - y_train.shape[1], 1)
        y = torch.cat([y_train, padding], dim=1)
        y = y.unsqueeze(-1)
        return self.linear_layer(y)


class TransformerEncoderLayer(nn.Module):
    """Same as train_nano.py but row-axis attention is computed manually
    (no torch.compile) so we can capture the attention weights."""

    def __init__(self, a, e, h, eps=1e-5):
        super().__init__()
        self.num_heads = a
        self.head_dim = e // a
        assert e % a == 0

        self.qkv_datapoints = nn.Linear(e, 3 * e)
        self.qkv_features = nn.Linear(e, 3 * e)

        self.linear1 = nn.Linear(e, h)
        self.linear2 = nn.Linear(h, e)

        self.norm1 = LowerPrecisionRMSNorm(e, eps=eps)
        self.norm2 = LowerPrecisionRMSNorm(e, eps=eps)
        self.norm3 = LowerPrecisionRMSNorm(e, eps=eps)

        self._captured_attn = None  # (num_heads, r, r) filled after forward

    def forward(self, src, sep):
        b, r, c, e = src.shape

        # --- feature attention (fused, no capture needed) ---
        x = src.reshape(b * r, c, e)
        res = x
        x = self.norm1(x)
        qkv = self.qkv_features(x)
        qkv = qkv.reshape(b * r, c, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(b * r, c, e)
        src = (res + x).reshape(b, r, c, e)

        # --- row attention (manual so we can read the weights) ---
        x = src.transpose(1, 2).reshape(b * c, r, e)
        res = x
        x = self.norm2(x)
        qkv = self.qkv_datapoints(x)
        qkv = qkv.reshape(b * c, r, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # q/k/v: (b*c, heads, r, head_dim)

        q_left, q_right = q.split([sep, r - sep], dim=2)
        k_train = k[:, :, :sep, :]
        v_train = v[:, :, :sep, :]

        scale = self.head_dim ** -0.5
        # (b*c, heads, sep,   sep)
        attn_left  = (q_left  @ k_train.transpose(-2, -1) * scale).softmax(-1)
        # (b*c, heads, r-sep, sep)
        attn_right = (q_right @ k_train.transpose(-2, -1) * scale).softmax(-1)

        # Build full (r, r) map averaged over b*c — right columns stay zero (masked)
        full = torch.zeros(self.num_heads, r, r, device="cpu")
        full[:, :sep,  :sep] = attn_left.float().mean(0).cpu()
        full[:, sep:,  :sep] = attn_right.float().mean(0).cpu()
        self._captured_attn = full.detach()

        x_left  = attn_left  @ v_train
        x_right = attn_right @ v_train
        x = torch.cat([x_left, x_right], dim=2)
        x = x.transpose(1, 2).reshape(b * c, r, e)

        src = (res + x).reshape(b, c, r, e).transpose(2, 1)
        x = self.norm3(src)
        x = self.linear2(F.gelu(self.linear1(x)))
        return src + x


class TransformerEncoderStack(nn.Module):
    def __init__(self, l, a, e, h, residual_decay=1.0):
        super().__init__()
        self.residual_decay = residual_decay
        self.transformer_blocks = nn.ModuleList(
            [TransformerEncoderLayer(a, e, h) for _ in range(l)]
        )

    def forward(self, x, sep):
        for i, block in enumerate(self.transformer_blocks):
            x = x * (self.residual_decay ** i)
            x = block(x, sep=sep)
        return x


class Decoder(nn.Module):
    def __init__(self, e, h, o):
        super().__init__()
        self.linear1 = nn.Linear(e, h)
        self.linear2 = nn.Linear(h, o)

    def forward(self, x):
        return self.linear2(F.gelu(self.linear1(x)))


class NanoTabPFNModel(nn.Module):
    def __init__(self, l, a, e, h, o, residual_decay=1.0, thinking_rows=16):
        super().__init__()
        self.l = l; self.a = a; self.e = e; self.h = h; self.o = o
        self.feature_encoder = FeatureEncoder(e)
        self.target_encoder = TargetEncoder(e)
        self.transformer_encoder = TransformerEncoderStack(l, a, e, h, residual_decay=residual_decay)
        self.decoder = Decoder(e, h, o)
        self.thinking_rows = ThinkingRows(num_thinking_rows=thinking_rows, e=e)
        self.register_buffer("borders", None, persistent=True)

    def forward(self, src, sep):
        x_src, y_src = src
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src.unsqueeze(-1)
        x_src = self.feature_encoder(x_src, sep)
        num_rows = x_src.shape[1]
        y_src = self.target_encoder(y_src, num_rows)
        src = torch.cat([x_src, y_src], dim=2)
        src, sep = self.thinking_rows(src, sep)
        output = self.transformer_encoder(src, sep)
        output = output[:, sep:, -1, :]
        return self.decoder(output)


def init_model_from_ckpt(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    thinking_rows = ckpt["model"]["thinking_rows.row_tokens"].shape[0]
    model = NanoTabPFNModel(
        l=ckpt["arch"]["l"],
        a=ckpt["arch"]["a"],
        e=ckpt["arch"]["e"],
        h=ckpt["arch"]["h"],
        o=ckpt["arch"]["o"],
        thinking_rows=thinking_rows,
    )
    model.load_state_dict(ckpt["model"])
    return model, thinking_rows


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

TABARENA_CLASSIFICATION_TASKS = [
    363613, 363614, 363616, 363618, 363619, 363620, 363621, 363623, 363624,
    363626, 363627, 363628, 363629, 363630, 363632, 363671, 363673, 363674,
    363676, 363677, 363679, 363681, 363682, 363683, 363684, 363685, 363689,
    363691, 363694, 363696, 363699, 363700, 363702, 363704, 363706, 363707,
    363711, 363712,
]


def to_pandas(x):
    return pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x


def to_numeric(x):
    return x.apply(pd.to_numeric, errors="coerce").to_numpy()


def get_feature_preprocessor(X):
    X = pd.DataFrame(X)
    num_mask, cat_mask = [], []
    for col in X:
        unique = X[col].dropna().unique()
        if len(unique) <= 1:
            num_mask.append(False); cat_mask.append(False); continue
        non_nan = X[col].notna().sum()
        numeric  = pd.to_numeric(X[col], errors="coerce").notna().sum()
        num_mask.append(non_nan == numeric)
        cat_mask.append(non_nan != numeric)
    num_t = Pipeline([
        ("to_pandas", FunctionTransformer(to_pandas)),
        ("to_numeric", FunctionTransformer(to_numeric)),
        ("imputer", SimpleImputer(strategy="mean")),
    ])
    cat_t = Pipeline([
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)),
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])
    return ColumnTransformer(transformers=[
        ("num", num_t, np.array(num_mask)),
        ("cat", cat_t, np.array(cat_mask)),
    ])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def draw_heatmap(ax, data, title):
    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0, vmax=data.max() or 1)
    ax.set_title(title, fontsize=7, pad=2)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    return im


def save_individual(data, layer_idx, head_idx, out_dir, dt, ckpt_name, task_id, dataset_name):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = draw_heatmap(ax, data, title=f"Layer {layer_idx+1}  Head {head_idx+1}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xlabel("key row →", fontsize=7)
    ax.set_ylabel("← query row", fontsize=7)

    fname = f"{dt}-attention-plot-{ckpt_name}-{task_id}-{dataset_name}-layer-{layer_idx+1}-head-{head_idx+1}.png"
    fpath = os.path.join(out_dir, fname)
    plt.tight_layout()
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    return fpath


def save_combined(attn_maps, n_think, n_train, n_test, num_layers, num_heads, out_dir, dt, ckpt_name, task_id, dataset_name):
    fig, axes = plt.subplots(
        num_layers, num_heads,
        figsize=(2.2 * num_heads, 2.2 * num_layers),
        squeeze=False,
    )

    for li in range(num_layers):
        for hi in range(num_heads):
            data = attn_maps[li][hi].numpy()
            ax = axes[li][hi]
            draw_heatmap(ax, data, title=f"L{li+1} H{hi+1}")
            if hi == 0:
                ax.set_ylabel(f"L{li+1}", fontsize=7)

    fig.suptitle(
        f"{ckpt_name}\n"
        f"rows: {n_think} think  +  {n_train} train  +  {n_test} test  =  {n_think+n_train+n_test}\n"
        f"white lines mark region boundaries",
        fontsize=8, y=1.01,
    )
    plt.tight_layout()
    fname = f"{dt}-attention-plot-{ckpt_name}-{task_id}-{dataset_name}-all.png"
    fpath = os.path.join(out_dir, fname)
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    return fpath


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot row-attention maps from a NanoTabPFN checkpoint.")
    parser.add_argument("checkpoint", help="Path to .pth checkpoint file")
    parser.add_argument("--fold",  type=int, default=0, help="Which CV fold to use (default: 0)")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds (default: 5)")
    parser.add_argument("--seed",     type=int, default=11)
    parser.add_argument("--max-rows", type=int, default=1000, help="Max samples to subsample (default: 1000)")
    parser.add_argument("--max-cols", type=int, default=100,  help="Max features to subsample (default: 100)")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: workdir/plots/attention/{dt}-{ckpt_name})")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_name = Path(args.checkpoint).stem
    dt = datetime.now().strftime("%y%m%d-%H%M%S")
    run_dir = args.out_dir or os.path.join("workdir", "plots", "attention", f"{dt}-{ckpt_name}")

    # --- load model (once) ---
    print(f"loading: {args.checkpoint}")
    model, n_think = init_model_from_ckpt(args.checkpoint)
    model = model.to(device).eval()
    num_layers = model.l
    num_heads  = model.a

    # --- loop over all TABARENA tasks ---
    for task_id in TABARENA_CLASSIFICATION_TASKS:
        print(f"\n[task {task_id}]")
        try:
            task    = openml.tasks.get_task(task_id, download_splits=False)
            dataset = task.get_dataset(download_data=False)
            dataset_name = dataset.name.lower().replace("_", "-")
            X, y, _, _ = dataset.get_data(target=task.target_name, dataset_format="dataframe")
        except Exception as e:
            print(f"  skipping: {e}")
            continue

        if X.shape[1] > args.max_cols:
            rng = np.random.default_rng(args.seed)
            X = X.iloc[:, rng.choice(X.shape[1], size=args.max_cols, replace=False)]

        if len(X) > args.max_rows:
            _, X, _, y = train_test_split(X, y, test_size=args.max_rows, stratify=y, random_state=args.seed)
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)

        le    = LabelEncoder()
        y_enc = le.fit_transform(y)

        cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        splits = list(cv.split(X, y_enc))
        train_idx, test_idx = splits[args.fold]
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y_enc[train_idx]

        prep      = get_feature_preprocessor(X_train)
        X_train_t = prep.fit_transform(X_train)
        X_test_t  = prep.transform(X_test)

        n_train = len(X_train_t)
        n_test  = len(X_test_t)
        print(f"  rows: {n_think} think + {n_train} train + {n_test} test = {n_think+n_train+n_test}")

        x_np = np.concatenate([X_train_t, X_test_t], axis=0)
        x_t  = torch.from_numpy(x_np).unsqueeze(0).float().to(device)
        y_t  = torch.from_numpy(y_train).unsqueeze(0).float().to(device)
        sep  = n_train

        with torch.no_grad():
            _ = model((x_t, y_t), sep=sep)

        attn_maps = [
            layer._captured_attn
            for layer in model.transformer_encoder.transformer_blocks
        ]

        out_dir = os.path.join(run_dir, f"{task_id}-{dataset_name}")
        ind_dir = os.path.join(out_dir, "individual")
        os.makedirs(ind_dir, exist_ok=True)

        for li in range(num_layers):
            for hi in range(num_heads):
                p = save_individual(attn_maps[li][hi].numpy(), li, hi, ind_dir, dt, ckpt_name, task_id, dataset_name)
                print(f"  {p}")

        p = save_combined(attn_maps, n_think, n_train, n_test, num_layers, num_heads, out_dir, dt, ckpt_name, task_id, dataset_name)
        print(f"  {p}")

    print("\ndone.")


if __name__ == "__main__":
    main()
