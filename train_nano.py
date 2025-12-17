import math
import os
import platform
import random
import socket
import subprocess
import sys
import time
import tomllib
import uuid
import warnings
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Callable, Tuple

import h5py
import numpy as np
import openml
import pandas as pd
import schedulefree
import torch
import torch.distributed as dist
import torch.nn.functional as F
from openml.config import set_root_cache_directory
from openml.tasks import TaskType
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OrdinalEncoder
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader


# -----------------------------------------------------------------------------
# config


@dataclass
class Config:
    type: str = "classification"
    experiments_dir: str = "workdir/experiments"
    classification_dump: str = "workdir/dumps/dump-d256000b1r1000c20-8.h5"
    seed: int = 11
    batch_size: int = 1  # per GPU
    accumulate: int = 1
    lr: float = 1e-4
    steps: int = 8  # step size
    epochs: int = 4000
    a: int = 6
    e: int = 192
    h: int = 768
    l: int = 6
    o: int | None = None
    gpus: int | None = None
    eval_every: int = 100
    jackpot: float = 0.85


c = Config()

random.seed(c.seed)
np.random.seed(c.seed)
torch.manual_seed(c.seed)

assert torch.cuda.is_available()

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
c.gpus = world_size
device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
master_process = rank == 0

if master_process:
    ts = datetime.now().strftime("%y%m%d-%H%M%S")
    uid = uuid.uuid4().hex[:8]
    e_id = f"{ts}-{uid}"
    e_dir = os.path.join(c.experiments_dir, e_id)
    os.makedirs(e_dir, exist_ok=True)
    log_path = os.path.join(e_dir, f"{e_id}-log.txt")
    ckpt_path = os.path.join(e_dir, f"{e_id}-ckpt.pth")

    with open(sys.argv[0], "r") as f:
        code = f.read()

    with open("pyproject.toml", "rb") as f:
        version = tomllib.load(f)["project"]["version"]


def print0(s, console=False):
    if master_process:
        with open(log_path, "a") as f:
            if console:
                print(s)
            print(s, file=f)


if master_process:
    print0(code)
    print0("=" * 100)
    print0(f"host: {socket.gethostname()}")
    print0(f"platform: {platform.platform()}")
    print0(f"python: {sys.version}")
    print0(f"torch: {torch.version.__version__}")
    print0(f"cuda: {torch.version.cuda}")
    print0(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout)
    print0("=" * 100)


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
        elif len(args) == 1 and isinstance(args, tuple):
            return self._forward(*args, **kwargs)

    def _forward(self, src: Tuple[torch.Tensor, torch.Tensor], sep: int, chunks: int = 1) -> torch.Tensor:
        x_src, y_src = src
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src.unsqueeze(-1)
        x_src = self.feature_encoder(x_src, sep)
        num_rows = x_src.shape[1]
        y_src = self.target_encoder(y_src, num_rows)
        src = torch.cat([x_src, y_src], 2)
        output = self.transformer_encoder(src, sep, chunks=chunks)
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

    def forward(self, x: torch.Tensor, sep: int, chunks: int = 1) -> torch.Tensor:
        for block in self.transformer_blocks:
            x = block(x, sep=sep, chunks=chunks)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, a: int, e: int, h: int, eps: float = 1e-5, batch_first: bool = True, device=None, dtype=None):
        super().__init__()
        # why does switching these two lines cause a difference in results !?!
        self.a_datapoints = nn.MultiheadAttention(e, a, batch_first=batch_first, device=device, dtype=dtype)
        self.a_features = nn.MultiheadAttention(e, a, batch_first=batch_first, device=device, dtype=dtype)

        self.linear1 = nn.Linear(e, h, device=device, dtype=dtype)
        self.linear2 = nn.Linear(h, e, device=device, dtype=dtype)

        self.norm1 = nn.LayerNorm(e, eps=eps, device=device, dtype=dtype)
        self.norm2 = nn.LayerNorm(e, eps=eps, device=device, dtype=dtype)
        self.norm3 = nn.LayerNorm(e, eps=eps, device=device, dtype=dtype)

    def forward(self, src: torch.Tensor, sep: int, chunks: int = 1) -> torch.Tensor:
        batch_size, rows_size, col_size, e = src.shape
        src = src.reshape(batch_size * rows_size, col_size, e)

        @memory_chunking(chunks)
        def feature_attention(x):
            return self.a_features(x, x, x)[0] + x

        src = feature_attention(src)
        src = src.reshape(batch_size, rows_size, col_size, e)
        src = self.norm1(src)
        src = src.transpose(1, 2)
        src = src.reshape(batch_size * col_size, rows_size, e)

        @memory_chunking(chunks)
        def datapoint_attention(x):
            x_left = self.a_datapoints(x[:, :sep], x[:, :sep], x[:, :sep])[0]
            x_right = self.a_datapoints(x[:, sep:], x[:, :sep], x[:, :sep])[0]
            return torch.cat([x_left, x_right], dim=1) + x

        src = datapoint_attention(src)
        src = src.reshape(batch_size, col_size, rows_size, e)
        src = src.transpose(2, 1)
        src = self.norm2(src)
        src = src.reshape(-1, e)

        @memory_chunking(chunks)
        def mlp(x):
            return self.linear2(F.gelu(self.linear1(x))) + x

        src = mlp(src)
        src = src.reshape(batch_size, rows_size, col_size, e)
        src = self.norm3(src)
        return src


def memory_chunking(chunks: int) -> callable:
    def decorator(func: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
        def wrapper(x: torch.Tensor) -> torch.Tensor:
            if chunks <= 1 or x.shape[0] == 0:
                return func(x)
            elif torch.is_grad_enabled():
                warnings.warn(
                    "Memory chunking is disabled since gradient computation is enabled to avoid incorrect gradients. "
                    "Please use `with torch.no_grad():` during inference to enable chunking.",
                )
                return func(x)
            chunk_size = max(1, math.ceil(x.shape[0] / chunks))
            for x_split in torch.split(x, split_size_or_sections=chunk_size, dim=0):
                x_split[:] = func(x_split)
            return x

        return wrapper

    return decorator


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
    def __init__(self, filename, num_steps, batch_size, device, rank, world_size):
        self.filename = filename
        self.num_steps = num_steps
        self.batch_size = batch_size
        with h5py.File(self.filename, "r") as f:
            self.num_datapoints_max = f["X"].shape[0]
            self.max_num_classes = f["max_num_classes"][0] if "max_num_classes" in f else None
            self.problem_type = f["problem_type"][()].decode("utf-8")
            # X = (num_datasets, max_num_datapoints, max_num_features)
            self.datasets = f["X"].shape[0]
            self.max_rows = f["X"].shape[1]
            self.max_cols = f["X"].shape[2]
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.pointer = self.rank * self.batch_size
        self.stride = self.world_size * self.batch_size

    def __iter__(self):
        with h5py.File(self.filename, "r") as f:
            self.data = f

            for _ in range(self.num_steps):
                end = self.pointer + self.batch_size

                num_features = self.data["num_features"][self.pointer : end].max()
                x = torch.from_numpy(self.data["X"][self.pointer : end, :, :num_features])
                y = torch.from_numpy(self.data["y"][self.pointer : end])
                sep = self.data["single_eval_pos"][self.pointer : end]

                self.pointer += self.stride
                if self.pointer >= self.datasets:
                    print("pointer >= datasets, will reset!")
                    self.pointer = self.rank * self.batch_size

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

    num_mask = np.array(num_mask)
    cat_mask = np.array(cat_mask)

    num_transformer = Pipeline(
        [
            ("to_pandas", FunctionTransformer(to_pandas)),
            ("to_numeric", FunctionTransformer(to_numeric)),
            ("imputer", SimpleImputer(strategy="mean")),
        ]
    )
    cat_transformer = Pipeline(
        [
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)),
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_mask),
            ("cat", cat_transformer, cat_mask),
        ],
    )
    return preprocessor


class NanoTabPFNClassifier:
    def __init__(self, model: NanoTabPFNModel | str | None = None, chunks: int = 8):
        device = "cuda"
        if model is None:
            raise ValueError("model is None")
        if isinstance(model, str):
            model = init_model_from_ckpt_file(model)
        self.model = model.to(device)
        self.device = device
        self.chunks = chunks

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
            out = self.model((x, y), sep=len(self.X_train), chunks=self.chunks).squeeze(0)
            out = out[:, : self.num_classes]
            probabilities = F.softmax(out, dim=1)
            return probabilities.to("cpu").numpy()


# -----------------------------------------------------------------------------
# evaluation


TOY_TASKS_CLASSIFICATION = [
    59,  # iris
    2382,  # wine
    9946,  # breast_cancer
]

TASKS = [
    363614,  # anneal       (  898,  39) anneal
    363619,  # bank         (10000,  11) Bank_Customer_Churn
    363621,  # blood        (  748,   5) blood-transfusion-service-center
    363623,  # churn        ( 5000,  20) churn
    363624,  # coil2000     ( 9822,  86) coil2000_insurance_policies
    363626,  # credit       ( 1000,  21) credit-g
    363629,  # diabetes     (  768,   9) diabetes
    363671,  # fitness      ( 1500,   7) Fitness_Club
    363674,  # hazelnut     ( 2400,  31) hazelnut-spread-contaminant-detection
    363682,  # is           ( 1723,  14) Is-this-a-good-customer
    363684,  # marketing    ( 2240,  26) Marketing_Campaign
    363685,  # maternal     ( 1014,   7) maternal_health_risk
    363689,  # naticusdroid ( 7491,  87) NATICUSdroid
    363694,  # polish       ( 5910,  65) polish_companies_bankruptcy
    363696,  # qsar         ( 1054,  42) qsar-biodeg
    363700,  # seismic      ( 2584,  16) seismic-bumps
    363702,  # splice       ( 3190,  61) splice
    363704,  # students     ( 4424,  37) students_dropout_and_academic_success
    363706,  # taiwanese    ( 6819,  95) taiwanese_bankruptcy_prediction
    363707,  # website      ( 1353,  10) website_phishing
    363711,  # mic          ( 1699, 112) MIC
]


@torch.no_grad()
def get_openml_predictions(
    *,
    model: NanoTabPFNClassifier,
    tasks: list[int] | str = "tabarena-v0.1",
    max_n_features: int = 500,
    max_n_samples: int = 10_000,
    cache_directory: str | None = None,
):
    if cache_directory is not None:
        set_root_cache_directory(cache_directory)

    if isinstance(tasks, str):
        benchmark_suite = openml.study.get_suite(tasks)
        task_ids = benchmark_suite.tasks
    else:
        task_ids = tasks

    dataset_predictions = {}

    for task_id in task_ids:
        task = openml.tasks.get_task(task_id, download_splits=False)

        if task.task_type_id != TaskType.SUPERVISED_CLASSIFICATION:
            continue

        dataset = task.get_dataset(download_data=False)

        n_features = dataset.qualities["NumberOfFeatures"]
        n_samples = dataset.qualities["NumberOfInstances"]
        if n_features > max_n_features or n_samples > max_n_samples:
            continue

        _, folds, _ = task.get_split_dimensions()
        tabarena_light = True
        if tabarena_light:
            folds = 1
        repeat = 0
        targets = []
        predictions = []
        probabilities = []
        for fold in range(folds):
            X, y, _, _ = dataset.get_data(target=task.target_name, dataset_format="dataframe")
            train_indices, test_indices = task.get_train_test_split_indices(fold=fold, repeat=repeat)
            X_train = X.iloc[train_indices].to_numpy()
            y_train = y.iloc[train_indices].to_numpy()
            X_test = X.iloc[test_indices].to_numpy()
            y_test = y.iloc[test_indices].to_numpy()

            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train)
            y_test = label_encoder.transform(y_test)
            targets.append(y_test)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            predictions.append(y_pred)
            y_proba = model.predict_proba(X_test)
            if y_proba.shape[1] == 2:
                y_proba = y_proba[:, 1]
            probabilities.append(y_proba)

        y_pred = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        probabilities = np.concatenate(probabilities, axis=0) if len(probabilities) > 0 else None
        dataset_predictions[str(dataset.name)] = (targets, y_pred, probabilities)
    return dataset_predictions


# -----------------------------------------------------------------------------
# main


prior = PriorDumpDataLoader(
    filename=c.classification_dump,
    num_steps=c.steps,
    batch_size=c.batch_size,
    device=device,
    rank=rank,
    world_size=world_size,
)
c.o = prior.max_num_classes

assert prior.num_steps % c.accumulate == 0, "num_steps must be divisible by accumulate_gradients"

model = NanoTabPFNModel(l=c.l, a=c.a, e=c.e, h=c.h, o=c.o).to(device)

optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=c.lr, weight_decay=0.0)

criterion = nn.CrossEntropyLoss()

model = DDP(model, device_ids=[local_rank], output_device=local_rank)

t_t = 0.0

for epoch in range(1, c.epochs + 1):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.train()
    optimizer.train()
    total_loss = 0.0
    for i, full_data in enumerate(prior):
        sep = full_data["sep"]
        data = (
            full_data["x"].to(device),
            full_data["y"][:, :sep].to(device),
        )
        # TODO: handle nans properly
        x, y = data
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        data = (x, y)

        targets = full_data["target_y"].to(device)

        output = model(data, sep=sep)
        targets = targets[:, sep:]
        targets = targets.reshape((-1,)).to(torch.long)
        output = output.view(-1, output.shape[-1])

        losses = criterion(output, targets)
        loss = losses.mean() / c.accumulate

        if (i + 1) % c.accumulate == 0:
            loss.backward()
        else:
            with model.no_sync():
                loss.backward()

        total_loss += loss.cpu().detach().item() * c.accumulate

        if (i + 1) % c.accumulate == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

    torch.cuda.synchronize()
    e_t = time.perf_counter() - t0  # epoch time
    t_t += e_t  # train time
    mu_e_t = t_t / epoch  # mean epoch time

    mean_loss = total_loss / len(prior)

    print0(f"e:{epoch}/{c.epochs} μ_l:{mean_loss:.2f} e_t:{e_t:.2f}s μ_e_t:{mu_e_t:.2f}s t_t:{t_t:.2f}s", console=True)

    model.eval()
    optimizer.eval()

    stop = torch.zeros(1, device=device, dtype=torch.int32)

    if master_process and ((epoch == 1) or (epoch == c.epochs) or (epoch % c.eval_every == 0)):
        raw_model = model.module
        clf = NanoTabPFNClassifier(raw_model, chunks=64)
        preds = get_openml_predictions(model=clf, tasks=TASKS)
        aucs: list[float] = []
        for dataset_name, (y_true, y_pred, y_proba) in preds.items():
            auc = (
                roc_auc_score(y_true, y_proba, multi_class="ovr")
                if getattr(y_proba, "ndim", 1) > 1
                else roc_auc_score(y_true, y_proba)
            )
            aucs.append(auc)
            dataset_name = dataset_name.lower().split("-")[0].split("_")[0]
            print0(f"{dataset_name}_roc_auc:{auc:.2f}", console=True)
        avg_auc = (sum(aucs) / len(aucs)) if len(aucs) > 0 else float("nan")
        print0(f"avg_roc_auc:{avg_auc}", console=True)

        ckpt = {
            "version": version,
            "timestamp": ts,
            "uid": uid,
            "type": c.type,
            "arch": {
                "e": raw_model.e,
                "a": raw_model.a,
                "h": raw_model.h,
                "l": raw_model.l,
                "o": raw_model.o,
            },
            "model": raw_model.state_dict(),
        }
        torch.save(ckpt, ckpt_path)

        if avg_auc >= c.jackpot:
            stop.fill_(1)
            print0("=" * 100)
            print0(f"datasets seen: {epoch * c.gpus * c.batch_size * c.steps}", console=True)

    dist.broadcast(stop, src=0)
    if stop.item() == 1:
        break

if master_process:
    print0("=" * 100)
    print0("config:")
    for f in fields(Config):
        print0(f"  {f.name}: {getattr(c, f.name)}")
    print0("=" * 100)
    print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB", console=True)
    print0(f"peak memory reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
    print0(f"experiment done: {e_id}", console=True)

dist.destroy_process_group()
