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
from typing import Tuple

import h5py
import numpy as np
import openml
import pandas as pd
import schedulefree
import torch
import torch.nn.functional as F
from openml.tasks import TaskType
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
    seed: int = 11
    batch_size: int = 1
    lr: float = 1e-4
    steps: int = 64  # step size
    epochs: int = 4000
    a: int = 6
    e: int = 192
    h: int = 768
    l: int = 6
    o: int | None = None
    eval_every: int = 100
    eval_folds: int = 5
    eval_subsample_samples: int | None = 1000
    eval_subsample_features: int | None = 100
    jackpot: float = 0.8  # random baseline


c = Config()

random.seed(c.seed)
np.random.seed(c.seed)
torch.manual_seed(c.seed)

assert torch.cuda.is_available()

device = "cuda"

ts = datetime.now().strftime("%y%m%d-%H%M%S")
uid = uuid.uuid4().hex[:8]
e_id = f"{ts}-{uid}"
e_dir = os.path.join(c.experiments_dir, e_id)
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
# model


class NanoTabPFNModel(nn.Module):
    def __init__(self, l: int, a: int, e: int, h: int, o: int):
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

    def forward(self, *args, **kwargs) -> torch.Tensor:
        if len(args) == 3:
            x = args[0]
            if args[2] is not None:
                x = torch.cat((x, args[2]), dim=1)
            return self._forward((x, args[1]), sep=args[0].shape[1], **kwargs)
        elif len(args) == 1 and isinstance(args[0], tuple):
            return self._forward(*args, **kwargs)

    def _forward(self, src: Tuple[torch.Tensor, torch.Tensor], sep: int) -> torch.Tensor:
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
    def __init__(self, e: int):
        super().__init__()
        self.linear_layer = nn.Linear(1, e)

    def forward(self, x: torch.Tensor, sep: int) -> torch.Tensor:
        x = x.unsqueeze(-1)
        mean = x[:, :sep].mean(dim=1, keepdim=True)
        std = x[:, :sep].std(dim=1, keepdim=True) + 1e-8
        x = (x - mean) / std
        x = torch.clip(x, min=-100, max=100)
        return self.linear_layer(x)


class TargetEncoder(nn.Module):
    def __init__(self, e: int):
        super().__init__()
        self.linear_layer = nn.Linear(1, e)

    def forward(self, y_train: torch.Tensor, num_rows: int) -> torch.Tensor:
        mean = y_train.mean(dim=1, keepdim=True)
        padding = mean.repeat(1, num_rows - y_train.shape[1], 1)
        y = torch.cat([y_train, padding], dim=1)
        y = y.unsqueeze(-1)
        return self.linear_layer(y)


class TransformerEncoderStack(nn.Module):
    def __init__(self, l: int, a: int, e: int, h: int):
        super().__init__()
        self.transformer_blocks = nn.ModuleList()
        for _ in range(l):
            self.transformer_blocks.append(TransformerEncoderLayer(a, e, h))

    def forward(self, x: torch.Tensor, sep: int) -> torch.Tensor:
        for block in self.transformer_blocks:
            x = block(x, sep=sep)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, a: int, e: int, h: int, eps: float = 1e-5, batch_first: bool = True, device=None, dtype=None):
        super().__init__()
        self.a_datapoints = nn.MultiheadAttention(e, a, batch_first=batch_first, device=device, dtype=dtype)
        self.a_features = nn.MultiheadAttention(e, a, batch_first=batch_first, device=device, dtype=dtype)

        self.linear1 = nn.Linear(e, h, device=device, dtype=dtype)
        self.linear2 = nn.Linear(h, e, device=device, dtype=dtype)

        self.norm1 = nn.LayerNorm(e, eps=eps, device=device, dtype=dtype)
        self.norm2 = nn.LayerNorm(e, eps=eps, device=device, dtype=dtype)
        self.norm3 = nn.LayerNorm(e, eps=eps, device=device, dtype=dtype)

    def forward(self, src: torch.Tensor, sep: int) -> torch.Tensor:
        batch_size, rows_size, col_size, e = src.shape
        src = src.reshape(batch_size * rows_size, col_size, e)

        src = self.a_features(src, src, src)[0] + src
        src = src.reshape(batch_size, rows_size, col_size, e)
        src = self.norm1(src)
        src = src.transpose(1, 2)
        src = src.reshape(batch_size * col_size, rows_size, e)

        x_left = self.a_datapoints(src[:, :sep], src[:, :sep], src[:, :sep])[0]
        x_right = self.a_datapoints(src[:, sep:], src[:, :sep], src[:, :sep])[0]
        src = torch.cat([x_left, x_right], dim=1) + src
        src = src.reshape(batch_size, col_size, rows_size, e)
        src = src.transpose(2, 1)
        src = self.norm2(src)
        src = src.reshape(-1, e)

        src = self.linear2(F.gelu(self.linear1(src))) + src
        src = src.reshape(batch_size, rows_size, col_size, e)
        src = self.norm3(src)
        return src


class Decoder(nn.Module):
    def __init__(self, e: int, h: int, o: int):
        super().__init__()
        self.linear1 = nn.Linear(e, h)
        self.linear2 = nn.Linear(h, o)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
            # X = (num_datasets, max_num_datapoints, max_num_features)
            self.datasets = f["X"].shape[0]
            self.max_rows = f["X"].shape[1]
            self.max_cols = f["X"].shape[2]
        self.device = device
        self.pointer = 0

    def __iter__(self):
        with h5py.File(self.filename, "r") as f:
            self.data = f

            for _ in range(self.num_steps):
                end = self.pointer + self.batch_size

                num_features = self.data["num_features"][self.pointer : end].max()
                x = torch.from_numpy(self.data["X"][self.pointer : end, :, :num_features])
                y = torch.from_numpy(self.data["y"][self.pointer : end])
                sep = self.data["single_eval_pos"][self.pointer : end]

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


def get_feature_preprocessor(X: np.ndarray | pd.DataFrame) -> ColumnTransformer:
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
    def __init__(self, model: NanoTabPFNModel | str | None = None):
        device = "cuda"
        if model is None:
            raise ValueError("model is None")
        if isinstance(model, str):
            model = init_model_from_ckpt_file(model)
        self.model = model.to(device)
        self.device = device

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.feature_preprocessor = get_feature_preprocessor(X_train)
        self.X_train = self.feature_preprocessor.fit_transform(X_train)
        self.y_train = y_train
        self.num_classes = max(set(y_train)) + 1

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        predicted_probabilities = self.predict_proba(X_test)
        return predicted_probabilities.argmax(axis=1)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        x = np.concatenate((self.X_train, self.feature_preprocessor.transform(X_test)))
        y = self.y_train
        with torch.no_grad():
            x = torch.from_numpy(x).unsqueeze(0).to(torch.float).to(self.device)
            y = torch.from_numpy(y).unsqueeze(0).to(torch.float).to(self.device)
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

optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=c.lr, weight_decay=0.0, warmup_steps=1000)

criterion = nn.CrossEntropyLoss()

t_t = 0.0

for epoch in range(1, c.epochs + 1):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.train()
    optimizer.train()
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

        optimizer.zero_grad(set_to_none=True)

        output = model((x, y), sep=sep)
        output = output.reshape(-1, output.shape[-1])
        targets = targets.reshape((-1,)).to(torch.long)

        loss = criterion(output, targets)
        loss.backward()

        total_loss += loss.detach().cpu().item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    torch.cuda.synchronize()
    e_t = time.perf_counter() - t0  # epoch time
    t_t += e_t  # train time
    mu_e_t = t_t / epoch  # mean epoch time

    mean_loss = total_loss / max(num_valid, 1)

    print0(
        f"[{datetime.now().strftime('%H:%M:%S')}] "
        f"e:{epoch}/{c.epochs} μ_l:{mean_loss:.2f} "
        f"({c.steps}-{num_valid}={c.steps - num_valid}) "
        f"e_t:{e_t:.2f}s μ_e_t:{mu_e_t:.2f}s t_t:{t_t:.2f}s",
        console=True,
    )

    model.eval()
    optimizer.eval()

    if (epoch == 1) or (epoch == c.epochs) or (epoch % c.eval_every == 0):
        clf = NanoTabPFNClassifier(model)
        aucs: list[float] = []

        for task_id in TABARENA_CLASSIFICATION_TASKS:
            task = openml.tasks.get_task(task_id, download_splits=False)

            if task.task_type_id != TaskType.SUPERVISED_CLASSIFICATION:
                continue

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
        print0(f"avg_roc_auc:{avg_auc}", console=True)

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
            break

print0("=" * 100)
print0("config:")
for f in fields(Config):
    print0(f"  {f.name}: {getattr(c, f.name)}")
print0("=" * 100)
print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB", console=True)
print0(f"peak memory reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
print0(f"experiment done: {e_id}", console=True)
