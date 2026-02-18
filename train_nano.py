import argparse
import os
import sys

with open(sys.argv[0], "r") as f:
    code = f.read()

import platform
import random
import socket
import subprocess
import time
import tomllib
import uuid
from dataclasses import dataclass, fields
from datetime import datetime

import h5py
import numpy as np
import openml
import pandas as pd
import schedulefree
import torch
import torch.nn.functional as F
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OrdinalEncoder
from torch import nn
from torch.utils.data import DataLoader


# -----------------------------------------------------------------------------
# config


@dataclass
class Config:
    type: str = "classification"
    experiments_dir: str = "workdir/experiments"
    classification_dump: str = "workdir/dumps/dump-d256000b1r1000c20-8.h5"
    # wget -c https://salihboraozturk.com/other/ufr/dump-d256000b1r1000c20-8.h5
    seed: int = 11
    batch_size: int = 1
    lr: float = 0.001
    steps: int = 64
    epochs: int = 4000
    a: int = 4
    e: int = 256
    h: int = 768
    l: int = 6
    o: int | None = None
    eval_folds: int = 5
    eval_subsample_samples: int | None = 1000
    eval_subsample_features: int | None = 100
    max_train_mins: float = 80
    jackpot: float = 0.8068462330697953


parser = argparse.ArgumentParser()
parser.add_argument("--name", default="")
args = parser.parse_args()

c = Config()

random.seed(c.seed)
np.random.seed(c.seed)
torch.manual_seed(c.seed)
torch.set_float32_matmul_precision('high')
import torch._dynamo
torch._dynamo.config.cache_size_limit = 128

assert torch.cuda.is_available()

device = "cuda"

start_ts = datetime.now()
ts = start_ts.strftime("%y%m%d-%H%M%S")
uid = uuid.uuid4().hex[:8]
e_name = args.name.strip()
e_id = f"{ts}-{uid}-{e_name}" if e_name else f"{ts}-{uid}"
e_root = os.path.join(c.experiments_dir, e_name) if e_name else c.experiments_dir
e_dir = os.path.join(e_root, e_id)
os.makedirs(e_dir, exist_ok=True)
log_path = os.path.join(e_dir, f"{e_id}-log.txt")
ckpt_path = os.path.join(e_dir, f"{e_id}-ckpt.pth")

with open("pyproject.toml", "rb") as f:
    version = tomllib.load(f)["project"]["version"]


def print0(s, console=False):
    with open(log_path, "a") as f:
        if console:
            print(s)
        print(s, file=f)


print0(code)
print0("=" * 100)
print0(f"start timestamp: {start_ts.strftime('%Y-%m-%d %H:%M:%S')}", console=True)
print0(f"host: {socket.gethostname()}")
print0(f"platform: {platform.platform()}")
print0(f"python: {sys.version}")
print0(f"torch: {torch.version.__version__}")
print0(f"cuda: {torch.version.cuda}")
print0(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout)
print0("=" * 100)

TABARENA_CLASSIFICATION_TASKS = [
    363613,  # ( 32769,   10) Amazon_employee_access
    363614,  # (   898,   39) anneal
    363616,  # ( 76000,  171) APSFailure
    363618,  # ( 45211,   14) bank-marketing
    363619,  # ( 10000,   11) Bank_Customer_Churn
    363620,  # (  3751, 1777) Bioresponse
    363621,  # (   748,    5) blood-transfusion-service-center
    363623,  # (  5000,   20) churn
    363624,  # (  9822,   86) coil2000_insurance_policies
    363626,  # (  1000,   21) credit-g
    363627,  # ( 30000,   24) credit_card_clients_default
    363628,  # (129880,   22) customer_satisfaction_in_airline
    363629,  # (   768,    9) diabetes
    363630,  # ( 71518,   48) Diabetes130US
    363632,  # ( 10999,   11) E-CommereShippingData
    363671,  # (  1500,    7) Fitness_Club
    363673,  # (150000,   11) GiveMeSomeCredit
    363674,  # (  2400,   31) hazelnut-spread-contaminant-detection
    363676,  # ( 10459,   24) heloc
    363677,  # (  3845, 1618) hiva_agnostic
    363679,  # ( 19158,   13) HR_Analytics_Job_Change_of_Data_Scientists
    363681,  # ( 12684,   25) in_vehicle_coupon_recommendation
    363682,  # (  1723,   14) Is-this-a-good-customer
    363683,  # ( 50000,  213) kddcup09_appetency
    363684,  # (  2240,   26) Marketing_Campaign
    363685,  # (  1014,    7) maternal_health_risk
    363689,  # (  7491,   87) NATICUSdroid
    363691,  # ( 12330,   18) online_shoppers_intention
    363694,  # (  5910,   65) polish_companies_bankruptcy
    363696,  # (  1054,   42) qsar-biodeg
    363699,  # ( 78053,   12) SDSS17
    363700,  # (  2584,   16) seismic-bumps
    363702,  # (  3190,   61) splice
    363704,  # (  4424,   37) students_dropout_and_academic_success
    363706,  # (  6819,   95) taiwanese_bankruptcy_prediction
    363707,  # (  1353,   10) website_phishing
    363711,  # (  1699,  112) MIC
    363712,  # ( 10885,   22) jm1
]


# -----------------------------------------------------------------------------
# muon


def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)

@torch.compile
def zeropower_via_newtonschulz5_batched(G, steps=10, eps=1e-7):
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm(dim=(1, 2), keepdim=True) + eps)
    if X.size(1) > X.size(2):
        X = X.transpose(1, 2)
    for _ in range(steps):
        A = X @ X.transpose(1, 2)
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    return X.to(G.dtype)

zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)

class Muon(torch.optim.Optimizer):
    """
    code adapted from: https://github.com/KellerJordan/modded-nanogpt/commit/b356a1f
    """
    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True, backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if group['nesterov']:
                    g = g.add(buf, alpha=momentum)
                if g.size(0) == 3 * g.size(1):
                    g_batched = g.view(3, g.size(1), g.size(1))
                    g_new = zeropower_via_newtonschulz5_batched(g_batched, steps=group['backend_steps'])
                    g = g_new.view(3 * g.size(1), g.size(1))
                    scale = g.size(1)**0.5
                else:
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    scale = max(g.size(0), g.size(1))**0.5
                p.data.add_(g, alpha=-lr * scale)


# -----------------------------------------------------------------------------
# model


class NanoTabPFNModel(nn.Module):
    def __init__(self, l, a, e, h, o):
        """
        l : num layers
        a : num attention heads
        e : embedding size
        h : mlp hidden size
        o : num outputs
        """
        super().__init__()
        self.l = l
        self.a = a
        self.e = e
        self.h = h
        self.o = o
        self.feature_encoder = FeatureEncoder(e)
        self.target_encoder = TargetEncoder(e)
        self.transformer_encoder = TransformerEncoderStack(l, a, e, h)
        self.decoder = Decoder(e, h, o)

        self.register_buffer("borders", None, persistent=True)

    def forward(self, *args, **kwargs):
        if len(args) == 3:
            x = args[0]
            if args[2] is not None:
                x = torch.cat((x, args[2]), dim=1)
            return self._forward((x, args[1]), sep=args[0].shape[1], **kwargs)
        elif len(args) == 1 and isinstance(args[0], tuple):
            return self._forward(*args, **kwargs)

    def _forward(self, src, sep):
        x_src, y_src = src
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src.unsqueeze(-1)
        x_src = self.feature_encoder(x_src, sep)
        num_rows = x_src.shape[1]
        y_src = self.target_encoder(y_src, num_rows)
        src = torch.cat([x_src, y_src], 2)
        output = self.transformer_encoder(src, sep)
        output = output[:, sep:, -1, :]
        output = self.decoder(output)
        return output


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


class TransformerEncoderStack(nn.Module):
    def __init__(self, l, a, e, h):
        super().__init__()
        self.transformer_blocks = nn.ModuleList()
        for _ in range(l):
            self.transformer_blocks.append(TransformerEncoderLayer(a, e, h))

    def forward(self, x, sep):
        for block in self.transformer_blocks:
            x = block(x, sep=sep)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, a, e, h, eps=1e-5):
        super().__init__()
        self.num_heads = a
        self.head_dim = e // a
        assert e % a == 0, "Embedding size must be divisible by heads"

        self.qkv_datapoints = nn.Linear(e, 3 * e)
        self.qkv_features = nn.Linear(e, 3 * e)

        self.linear1 = nn.Linear(e, h)
        self.linear2 = nn.Linear(h, e)

        self.norm1 = nn.LayerNorm(e, eps=eps)
        self.norm2 = nn.LayerNorm(e, eps=eps)
        self.norm3 = nn.LayerNorm(e, eps=eps)

    @torch.compile(dynamic=True)
    def forward(self, src, sep):
        b, r, c, e = src.shape

        x = src.reshape(b * r, c, e)
        res = x
        x = self.norm1(x)

        qkv = self.qkv_features(x)
        qkv = qkv.reshape(b * r, c, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(b * r, c, e)

        src = res + x
        src = src.reshape(b, r, c, e)

        x = src.transpose(1, 2).reshape(b * c, r, e)
        res = x
        x = self.norm2(x)

        qkv = self.qkv_datapoints(x)
        qkv = qkv.reshape(b * c, r, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q_left, q_right = q.split([sep, r - sep], dim=2)

        k_train = k[:, :, :sep, :]
        v_train = v[:, :, :sep, :]

        x_left = F.scaled_dot_product_attention(q_left, k_train, v_train)
        x_right = F.scaled_dot_product_attention(q_right, k_train, v_train)

        x = torch.cat([x_left, x_right], dim=2)
        x = x.transpose(1, 2).reshape(b * c, r, e)

        src = res + x
        src = src.reshape(b, c, r, e).transpose(2, 1)

        x = self.norm3(src)
        x = self.linear2(F.gelu(self.linear1(x)))
        src = src + x

        return src


class Decoder(nn.Module):
    def __init__(self, e, h, o):
        super().__init__()
        self.linear1 = nn.Linear(e, h)
        self.linear2 = nn.Linear(h, o)

    def forward(self, x):
        return self.linear2(F.gelu(self.linear1(x)))


# -----------------------------------------------------------------------------
# priors


class PriorDumpDataLoader(DataLoader):
    def __init__(self, filename, num_steps, batch_size, device):
        self.filename = filename
        self.num_steps = num_steps
        self.batch_size = batch_size
        with h5py.File(self.filename, "r") as f:
            self.max_num_classes = f["max_num_classes"][0] if "max_num_classes" in f else None
            self.problem_type = f["problem_type"][()].decode("utf-8")
            self.datasets = f["X"].shape[0]
            self.max_rows = f["X"].shape[1]
            self.max_cols = f["X"].shape[2]
        self.device = device
        self.pointer = 0

    def __iter__(self):
        with h5py.File(self.filename, "r") as f:
            for _ in range(self.num_steps):
                end = self.pointer + self.batch_size

                num_features = f["num_features"][self.pointer : end].max()
                x = torch.from_numpy(f["X"][self.pointer : end, :, :num_features])
                y = torch.from_numpy(f["y"][self.pointer : end])
                sep = f["single_eval_pos"][self.pointer : end]

                self.pointer += self.batch_size
                if self.pointer >= self.datasets:
                    print("pointer >= datasets, will reset!")
                    self.pointer = 0

                yield dict(
                    x=x.to(self.device),
                    y=y.to(self.device),
                    target_y=y.to(self.device),
                    sep=sep[0].item(),
                )

    def __len__(self):
        return self.num_steps


# -----------------------------------------------------------------------------
# interface


def init_model_from_ckpt_file(file_path):
    ckpt = torch.load(file_path, map_location="cpu")
    model = NanoTabPFNModel(
        l=ckpt["arch"]["l"],
        a=ckpt["arch"]["a"],
        e=ckpt["arch"]["e"],
        h=ckpt["arch"]["h"],
        o=ckpt["arch"]["o"],
    )
    if "borders" in ckpt["model"]:
        model.borders = ckpt["model"]["borders"]
    model.load_state_dict(ckpt["model"])
    return model


def to_pandas(x):
    return pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x


def to_numeric(x):
    return x.apply(pd.to_numeric, errors="coerce").to_numpy()


def get_feature_preprocessor(X):
    X = pd.DataFrame(X)
    num_mask = []
    cat_mask = []
    for col in X:
        unique_non_nan_entries = X[col].dropna().unique()
        if len(unique_non_nan_entries) <= 1:
            num_mask.append(False)
            cat_mask.append(False)
            continue
        non_nan_entries = X[col].notna().sum()
        numeric_entries = pd.to_numeric(X[col], errors="coerce").notna().sum()
        num_mask.append(non_nan_entries == numeric_entries)
        cat_mask.append(non_nan_entries != numeric_entries)

    num_transformer = Pipeline(
        [
            ("to_pandas", FunctionTransformer(to_pandas)),
            ("to_numeric", FunctionTransformer(to_numeric)),
            ("imputer", SimpleImputer(strategy="mean")),
        ],
    )
    cat_transformer = Pipeline(
        [
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)),
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ],
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, np.array(num_mask)),
            ("cat", cat_transformer, np.array(cat_mask)),
        ],
    )
    return preprocessor


class NanoTabPFNClassifier:
    def __init__(self, model=None):
        device = "cuda"
        if model is None:
            raise ValueError("model is None")
        if isinstance(model, str):
            model = init_model_from_ckpt_file(model)
        self.model = model.to(device)
        self.device = device

    def fit(self, X_train, y_train):
        self.feature_preprocessor = get_feature_preprocessor(X_train)
        self.X_train = self.feature_preprocessor.fit_transform(X_train)
        self.y_train = y_train
        self.num_classes = max(set(y_train)) + 1

    def predict(self, X_test):
        predicted_probabilities = self.predict_proba(X_test)
        return predicted_probabilities.argmax(axis=1)

    def predict_proba(self, X_test):
        x = np.concatenate((self.X_train, self.feature_preprocessor.transform(X_test)))
        y = self.y_train
        with torch.no_grad():
            x = torch.from_numpy(x).unsqueeze(0).to(torch.float).to(self.device)
            y = torch.from_numpy(y).unsqueeze(0).to(torch.float).to(self.device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = self.model((x, y), sep=len(self.X_train)).squeeze(0)
                out = out[:, : self.num_classes]
                probabilities = F.softmax(out, dim=1)

            return probabilities.to("cpu").numpy()


# -----------------------------------------------------------------------------
# main


prior = PriorDumpDataLoader(
    filename=c.classification_dump,
    num_steps=c.steps,
    batch_size=c.batch_size,
    device=device,
)
c.o = prior.max_num_classes

model = NanoTabPFNModel(l=c.l, a=c.a, e=c.e, h=c.h, o=c.o).to(device)

muon_params = []
adam_params = []
for name, p in model.named_parameters():
    if p.ndim != 2:
        adam_params.append(p)
    elif "transformer_encoder" in name:
        muon_params.append(p)
    else:
        adam_params.append(p)

optimizer_muon = Muon(muon_params, lr=0.1*c.lr, momentum=0.95)
optimizer_adam = schedulefree.AdamWScheduleFree(adam_params, lr=c.lr, weight_decay=0.0, warmup_steps=1000)

optimizers = [optimizer_muon, optimizer_adam]

criterion = nn.CrossEntropyLoss()

t_t = 0.0


for epoch in range(1, c.epochs + 1):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.train()
    optimizer_adam.train()
    total_loss = 0.0
    num_valid = 0
    for i, full_data in enumerate(prior):
        sep = full_data["sep"]
        x = full_data["x"]
        y = full_data["y"][:, :sep]
        targets = full_data["target_y"][:, sep:]

        if torch.isnan(x).any() or torch.isnan(y).any():
            continue
        num_valid += 1

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model((x, y), sep=sep)
            output = output.reshape(-1, output.shape[-1])
            targets = targets.reshape((-1,)).to(torch.long)
            loss = criterion(output, targets)

        loss.backward()

        total_loss += loss.detach()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in optimizers:
            opt.step()

    torch.cuda.synchronize()
    e_t = time.perf_counter() - t0
    t_t += e_t
    mu_e_t = t_t / epoch

    mean_loss = (total_loss / num_valid).cpu().item()

    if t_t > c.max_train_mins * 60:
        print0("exceeded max train time", console=True)
        sys.exit(0)

    model.eval()
    optimizer_adam.eval()

    clf = NanoTabPFNClassifier(model)
    aucs = []

    for task_id in TABARENA_CLASSIFICATION_TASKS:
        task = openml.tasks.get_task(task_id, download_splits=False)

        dataset = task.get_dataset(download_data=False)
        X, y, _, _ = dataset.get_data(target=task.target_name, dataset_format="dataframe")

        len_features = X.shape[1]
        if c.eval_subsample_features is not None and len_features > c.eval_subsample_features:
            rng = np.random.default_rng(c.seed)
            feature_choices = rng.choice(len_features, size=c.eval_subsample_features, replace=False)
            X = X.iloc[:, feature_choices]

        if c.eval_subsample_samples is not None and len(X) > c.eval_subsample_samples:
            _, X, _, y = train_test_split(X, y, test_size=c.eval_subsample_samples, stratify=y, random_state=c.seed)
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)

        cv = StratifiedKFold(n_splits=c.eval_folds, shuffle=True, random_state=c.seed)

        targets = []
        probabilities = []

        for _, (train_indices, test_indices) in enumerate(cv.split(X, y)):
            X_train = X.iloc[train_indices].to_numpy()
            y_train = y.iloc[train_indices].to_numpy()
            X_test = X.iloc[test_indices].to_numpy()
            y_test = y.iloc[test_indices].to_numpy()

            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train)
            y_test = label_encoder.transform(y_test)
            targets.append(y_test)

            clf.fit(X_train, y_train)
            y_proba = clf.predict_proba(X_test)
            if y_proba.shape[1] == 2:
                y_proba = y_proba[:, 1]
            probabilities.append(y_proba)

        y_true = np.concatenate(targets, axis=0)
        y_proba = np.concatenate(probabilities, axis=0) if len(probabilities) > 0 else None

        auc = (
            roc_auc_score(y_true, y_proba, multi_class="ovr")
            if getattr(y_proba, "ndim", 1) > 1
            else roc_auc_score(y_true, y_proba)
        )
        aucs.append(auc)

    avg_auc = (sum(aucs) / len(aucs)) if len(aucs) > 0 else float("nan")

    print0(
        f"e:{epoch}/{c.epochs} μ_l:{mean_loss:.2f} "
        f"e_t:{e_t:.2f}s μ_e_t:{mu_e_t:.2f}s t_t:{t_t:.2f}s "
        f"avg_roc_auc:{avg_auc}",
        console=True,
    )

    if avg_auc >= c.jackpot:
        ckpt = {
            "version": version,
            "timestamp": ts,
            "uid": uid,
            "type": c.type,
            "arch": {
                "e": model.e,
                "a": model.a,
                "h": model.h,
                "l": model.l,
                "o": model.o,
            },
            "model": model.state_dict(),
        }
        torch.save(ckpt, ckpt_path)
        print0("=" * 100)
        print0(f"datasets seen: {epoch * c.batch_size * c.steps}", console=True)
        print0(f"record time in mins: {t_t / 60:.2f}", console=True)
        break

print0("=" * 100)
print0("config:")
for f in fields(Config):
    print0(f"  {f.name}: {getattr(c, f.name)}")
print0("=" * 100)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print0(f"params: {total_params:,} (trainable: {trainable_params:,})")
print0("=" * 100)
print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB", console=True)
print0(f"peak memory reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
print0("=" * 100)
end_ts = datetime.now()
print0(f"end timestamp: {end_ts.strftime('%Y-%m-%d %H:%M:%S')}", console=True)
print0(f"script runtime: {(end_ts - start_ts).total_seconds() / 60:.2f} mins")
print0(f"experiment done: {e_id}", console=True)
