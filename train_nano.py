import argparse
import math
import os
import random
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, Tuple, Union

import h5py
import numpy as np
import openml
import pandas as pd
import requests
import schedulefree
import torch
import torch.nn.functional as F
from openml.config import set_root_cache_directory
from openml.tasks import TaskType
from pfns.bar_distribution import FullSupportBarDistribution, get_bucket_limits
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OrdinalEncoder
from torch import nn
from torch.nn.modules.transformer import LayerNorm, Linear, MultiheadAttention
from torch.utils.data import DataLoader


# -----------------------------------------------------------------------------
# model


class NanoTabPFNModel(nn.Module):
    def __init__(self, embedding_size: int, num_attention_heads: int, mlp_hidden_size: int, num_layers: int, num_outputs: int):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_attention_heads = num_attention_heads
        self.mlp_hidden_size = mlp_hidden_size
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        self.feature_encoder = FeatureEncoder(embedding_size)
        self.target_encoder = TargetEncoder(embedding_size)
        self.transformer_encoder = TransformerEncoderStack(num_layers, embedding_size, num_attention_heads, mlp_hidden_size)
        self.decoder = Decoder(embedding_size, mlp_hidden_size, num_outputs)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        if len(args) == 3:
            x = args[0]
            if args[2] is not None:
                x = torch.cat((x, args[2]), dim=1)
            return self._forward((x, args[1]), single_eval_pos=args[0].shape[1], **kwargs)
        elif len(args) == 1 and isinstance(args, tuple):
            return self._forward(*args, **kwargs)

    def _forward(self, src: Tuple[torch.Tensor, torch.Tensor], single_eval_pos: int, num_mem_chunks: int = 1) -> torch.Tensor:
        x_src, y_src = src
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src.unsqueeze(-1)
        x_src = self.feature_encoder(x_src, single_eval_pos)
        num_rows = x_src.shape[1]
        y_src = self.target_encoder(y_src, num_rows)
        src = torch.cat([x_src, y_src], 2)
        output = self.transformer_encoder(src, single_eval_pos, num_mem_chunks=num_mem_chunks)
        output = output[:, single_eval_pos:, -1, :]
        output = self.decoder(output)
        return output


class FeatureEncoder(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)

    def forward(self, x: torch.Tensor, single_eval_pos: int) -> torch.Tensor:
        x = x.unsqueeze(-1)
        mean = x[:, :single_eval_pos].mean(dim=1, keepdim=True)
        std = x[:, :single_eval_pos].std(dim=1, keepdim=True) + 1e-8
        x = (x - mean) / std
        x = torch.clip(x, min=-100, max=100)
        return self.linear_layer(x)


class TargetEncoder(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)

    def forward(self, y_train: torch.Tensor, num_rows: int) -> torch.Tensor:
        mean = y_train.mean(dim=1, keepdim=True)
        padding = mean.repeat(1, num_rows - y_train.shape[1], 1)
        y = torch.cat([y_train, padding], dim=1)
        y = y.unsqueeze(-1)
        return self.linear_layer(y)


class TransformerEncoderStack(nn.Module):
    def __init__(self, num_layers: int, embedding_size: int, num_attention_heads: int, mlp_hidden_size: int):
        super().__init__()
        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_blocks.append(TransformerEncoderLayer(embedding_size, num_attention_heads, mlp_hidden_size))

    def forward(self, x: torch.Tensor, single_eval_position: int, num_mem_chunks: int = 1) -> torch.Tensor:
        for block in self.transformer_blocks:
            x = block(x, single_eval_position=single_eval_position, num_mem_chunks=num_mem_chunks)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_size: int, nhead: int, mlp_hidden_size: int,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True,
                 device=None, dtype=None):
        super().__init__()
        self.self_attention_between_datapoints = MultiheadAttention(embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype)
        self.self_attention_between_features = MultiheadAttention(embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype)

        self.linear1 = Linear(embedding_size, mlp_hidden_size, device=device, dtype=dtype)
        self.linear2 = Linear(mlp_hidden_size, embedding_size, device=device, dtype=dtype)

        self.norm1 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm2 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm3 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)

    def forward(self, src: torch.Tensor, single_eval_position: int, num_mem_chunks: int = 1) -> torch.Tensor:
        batch_size, rows_size, col_size, embedding_size = src.shape
        src = src.reshape(batch_size * rows_size, col_size, embedding_size)

        @memory_chunking(num_mem_chunks)
        def feature_attention(x):
            return self.self_attention_between_features(x, x, x)[0] + x

        src = feature_attention(src)
        src = src.reshape(batch_size, rows_size, col_size, embedding_size)
        src = self.norm1(src)
        src = src.transpose(1, 2)
        src = src.reshape(batch_size * col_size, rows_size, embedding_size)

        @memory_chunking(num_mem_chunks)
        def datapoint_attention(x):
            x_left = self.self_attention_between_datapoints(x[:, :single_eval_position], x[:, :single_eval_position], x[:, :single_eval_position])[0]
            x_right = self.self_attention_between_datapoints(x[:, single_eval_position:], x[:, :single_eval_position], x[:, :single_eval_position])[0]
            return torch.cat([x_left, x_right], dim=1) + x

        src = datapoint_attention(src)
        src = src.reshape(batch_size, col_size, rows_size, embedding_size)
        src = src.transpose(2, 1)
        src = self.norm2(src)
        src = src.reshape(-1, embedding_size)

        @memory_chunking(num_mem_chunks)
        def mlp(x):
            return self.linear2(F.gelu(self.linear1(x))) + x

        src = mlp(src)
        src = src.reshape(batch_size, rows_size, col_size, embedding_size)
        src = self.norm3(src)
        return src


def memory_chunking(num_mem_chunks: int) -> callable:
    def decorator(func: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
        def wrapper(x: torch.Tensor) -> torch.Tensor:
            if num_mem_chunks <= 1 or x.shape[0] == 0:
                return func(x)
            elif torch.is_grad_enabled():
                warnings.warn(
                    "Memory chunking is disabled since gradient computation is enabled to avoid incorrect gradients. "
                    "Please use `with torch.no_grad():` during inference to enable chunking.",
                )
                return func(x)
            chunk_size = max(1, math.ceil(x.shape[0] / num_mem_chunks))
            for x_split in torch.split(x, split_size_or_sections=chunk_size, dim=0):
                x_split[:] = func(x_split)
            return x

        return wrapper

    return decorator


class Decoder(nn.Module):
    def __init__(self, embedding_size: int, mlp_hidden_size: int, num_outputs: int):
        super().__init__()
        self.linear1 = nn.Linear(embedding_size, mlp_hidden_size)
        self.linear2 = nn.Linear(mlp_hidden_size, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.gelu(self.linear1(x)))


# -----------------------------------------------------------------------------
# priors


class PriorDataLoader(DataLoader):
    def __init__(self, get_batch_function: Callable[..., Dict[str, Union[torch.Tensor, int]]], num_steps: int, batch_size: int, num_datapoints_max: int, num_features: int, device: torch.device):
        self.get_batch_function = get_batch_function
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.num_datapoints_max = num_datapoints_max
        self.num_features = num_features
        self.device = device

    def __iter__(self) -> Iterator[Dict[str, Union[torch.Tensor, int]]]:
        return iter(self.get_batch_function(self.batch_size, self.num_datapoints_max, self.num_features) for _ in range(self.num_steps))

    def __len__(self) -> int:
        return self.num_steps


class PriorDumpDataLoader(DataLoader):
    def __init__(self, filename, num_steps, batch_size, device, starting_index=0):
        self.filename = filename
        self.num_steps = num_steps
        self.batch_size = batch_size
        with h5py.File(self.filename, "r") as f:
            self.num_datapoints_max = f["X"].shape[0]
            if "max_num_classes" in f:
                self.max_num_classes = f["max_num_classes"][0]
            else:
                self.max_num_classes = None
            self.problem_type = f["problem_type"][()].decode("utf-8")
        self.device = device
        self.pointer = starting_index

    def __iter__(self):
        with h5py.File(self.filename, "r") as f:
            for _ in range(self.num_steps):
                self.data = f
                end = self.pointer + self.batch_size

                num_features = self.data["num_features"][self.pointer : end].max()
                x = torch.from_numpy(self.data["X"][self.pointer : end, :, :num_features])
                y = torch.from_numpy(self.data["y"][self.pointer : end])
                single_eval_pos = self.data["single_eval_pos"][self.pointer : end]

                self.pointer += self.batch_size
                if self.pointer >= self.data["X"].shape[0]:
                    print(
                        """Finished iteration over all stored datasets! """
                        """Will start reusing the same data with different splits now.""",
                    )
                    self.pointer = 0

                yield dict(
                    x=x.to(self.device),
                    y=y.to(self.device),
                    target_y=y.to(self.device),
                    single_eval_pos=single_eval_pos[0].item(),
                )

    def __len__(self):
        return self.num_steps


# -----------------------------------------------------------------------------
# utils


def set_randomness_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_default_device():
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    if torch.cuda.is_available():
        device = "cuda"
    return device


def make_global_bucket_edges(filename, n_buckets=100, device=get_default_device(), max_y=5_000_000):
    with h5py.File(filename, "r") as f:
        y = f["y"]
        num_tables, num_datapoints = y.shape

        num_tables_to_use = min(num_tables, max_y // num_datapoints)

        y_subset = np.array(y[:num_tables_to_use, :], dtype=np.float32)
        y_means = y_subset.mean(axis=1, keepdims=True)
        y_stds = y_subset.std(axis=1, keepdims=True, ddof=1) + 1e-8
        ys_concat = ((y_subset - y_means) / y_stds).ravel()

    if ys_concat.size < n_buckets:
        raise ValueError(f"Too few target samples ({ys_concat.size}) to compute {n_buckets} buckets.")

    ys_tensor = torch.tensor(ys_concat, dtype=torch.float32, device=device)
    global_bucket_edges = get_bucket_limits(n_buckets, ys=ys_tensor).to(device)
    return global_bucket_edges


# -----------------------------------------------------------------------------
# interface


def init_model_from_state_dict_file(file_path):
    state_dict = torch.load(file_path, map_location=torch.device("cpu"))
    model = NanoTabPFNModel(
        num_attention_heads=state_dict["architecture"]["num_attention_heads"],
        embedding_size=state_dict["architecture"]["embedding_size"],
        mlp_hidden_size=state_dict["architecture"]["mlp_hidden_size"],
        num_layers=state_dict["architecture"]["num_layers"],
        num_outputs=state_dict["architecture"]["num_outputs"],
    )
    model.load_state_dict(state_dict["model"])
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

    num_transformer = Pipeline([
        ("to_pandas", FunctionTransformer(to_pandas)), 
        ("to_numeric", FunctionTransformer(to_numeric)), 
        ("imputer", SimpleImputer(strategy="mean")),
    ])
    cat_transformer = Pipeline([
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)),
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_mask),
            ("cat", cat_transformer, cat_mask),
        ],
    )
    return preprocessor


class NanoTabPFNClassifier:
    def __init__(
        self,
        model: NanoTabPFNModel | str | None = None,
        device: None | str | torch.device = None,
        num_mem_chunks: int = 8,
    ):
        if device is None:
            device = get_default_device()
        if model is None:
            model = "checkpoints/nanotabpfn.pth"
            if not os.path.isfile(model):
                os.makedirs("checkpoints", exist_ok=True)
                print("No cached model found, downloading model checkpoint.")
                response = requests.get("https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/nanotabpfn_classifier.pth")
                with open(model, "wb") as f:
                    f.write(response.content)
        if isinstance(model, str):
            model = init_model_from_state_dict_file(model)
        self.model = model.to(device)
        self.device = device
        self.num_mem_chunks = num_mem_chunks

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
            out = self.model((x, y), single_eval_pos=len(self.X_train), num_mem_chunks=self.num_mem_chunks).squeeze(0)
            out = out[:, : self.num_classes]
            probabilities = F.softmax(out, dim=1)
            return probabilities.to("cpu").numpy()


class NanoTabPFNRegressor:
    def __init__(
        self,
        model: NanoTabPFNModel | str | None = None,
        dist: FullSupportBarDistribution | str | None = None,
        device: str | torch.device | None = None,
        num_mem_chunks: int = 8,
    ):
        if device is None:
            device = get_default_device()
        if model is None:
            os.makedirs("checkpoints", exist_ok=True)
            model = "checkpoints/nanotabpfn_regressor.pth"
            dist = "checkpoints/nanotabpfn_regressor_buckets.pth"
            if not os.path.isfile(model):
                print("No cached model found, downloading model checkpoint.")
                response = requests.get("https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/nanotabpfn_regressor.pth")
                with open(model, "wb") as f:
                    f.write(response.content)
            if not os.path.isfile(dist):
                print("No cached bucket edges found, downloading bucket edges.")
                response = requests.get("https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/nanotabpfn_regressor_buckets.pth")
                with open(dist, "wb") as f:
                    f.write(response.content)
        if isinstance(model, str):
            model = init_model_from_state_dict_file(model)

        if isinstance(dist, str):
            bucket_edges = torch.load(dist, map_location=device)
            dist = FullSupportBarDistribution(bucket_edges).float()

        self.model = model.to(device)
        self.device = device
        self.dist = dist
        self.num_mem_chunks = num_mem_chunks

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.feature_preprocessor = get_feature_preprocessor(X_train)
        self.X_train = self.feature_preprocessor.fit_transform(X_train)
        self.y_train = y_train

        self.y_train_mean = np.mean(self.y_train)
        self.y_train_std = np.std(self.y_train, ddof=1) + 1e-8
        self.y_train_n = (self.y_train - self.y_train_mean) / self.y_train_std

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X = np.concatenate((self.X_train, self.feature_preprocessor.transform(X_test)))
        y = self.y_train_n

        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device).unsqueeze(0)
            y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(0)

            logits = self.model((X_tensor, y_tensor), single_eval_pos=len(self.X_train), num_mem_chunks=self.num_mem_chunks).squeeze(0)
            preds_n = self.dist.mean(logits)
            preds = preds_n * self.y_train_std + self.y_train_mean

        return preds.cpu().numpy()


# -----------------------------------------------------------------------------
# callbacks


class Callback(ABC):
    @abstractmethod
    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        pass

    @abstractmethod
    def close(self):
        pass


class BaseLoggerCallback(Callback):
    pass


class ConsoleLoggerCallback(BaseLoggerCallback):
    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        print(
            f"Epoch {epoch:5d} | Time {epoch_time:5.2f}s | Mean Loss {loss:5.2f}",
            flush=True,
        )

    def close(self):
        pass


class TensorboardLoggerCallback(BaseLoggerCallback):
    def __init__(self, log_dir: str):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=log_dir)

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        self.writer.add_scalar("Loss/train", loss, epoch)
        self.writer.add_scalar("Time/epoch", epoch_time, epoch)

    def close(self):
        self.writer.close()


class WandbLoggerCallback(BaseLoggerCallback):
    def __init__(self, project: str, name: str = None, config: dict = None, log_dir: str = None):
        try:
            import wandb

            self.wandb = wandb 
            wandb.init(
                project=project,
                name=name,
                id=name,
                config=config,
                dir=log_dir,
                resume="allow",
            )
        except ImportError as e:
            raise ImportError("wandb is not installed. Install it with: pip install wandb") from e

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        log_dict = {"epoch": epoch, "loss": loss, " epoch_time": epoch_time}
        self.wandb.log(log_dict)

    def close(self):
        self.wandb.finish()


# -----------------------------------------------------------------------------
# evaluation


TOY_TASKS_REGRESSION = [
    362443,  # diabetes
]

TOY_TASKS_CLASSIFICATION = [
    59,  # iris
    2382,  # wine
    9946,  # breast_cancer
]

TABARENA_TASKS = [
    363612, 363613, 363614, 363615, 363616, 363618, 363619, 363620, 363621, 363623, 363624, 363625, 363626, 363627, 
    363628, 363629, 363630, 363631, 363632, 363671, 363672, 363673, 363674, 363675, 363676, 363677, 363678, 363679, 
    363681, 363682, 363683, 363684, 363685, 363686, 363689, 363691, 363693, 363694, 363696, 363697, 363698, 363699, 
    363700, 363702, 363704, 363705, 363706, 363707, 363708, 363711, 363712,
]


@torch.no_grad()
def get_openml_predictions(
    *,
    model: NanoTabPFNRegressor | NanoTabPFNClassifier,
    tasks: list[int] | str = "tabarena-v0.1",
    max_n_features: int = 500,
    max_n_samples: int = 10_000,
    classification: bool | None = None,
    cache_directory: str | None = None,
):
    if classification is None:
        classification = isinstance(model, NanoTabPFNClassifier)

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

        if classification and task.task_type_id != TaskType.SUPERVISED_CLASSIFICATION:
            continue
        if not classification and task.task_type_id != TaskType.SUPERVISED_REGRESSION:
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
            X, y, categorical_indicator, attribute_names = dataset.get_data(target=task.target_name, dataset_format="dataframe")
            train_indices, test_indices = task.get_train_test_split_indices(fold=fold, repeat=repeat)
            X_train = X.iloc[train_indices].to_numpy()
            y_train = y.iloc[train_indices].to_numpy()
            X_test = X.iloc[test_indices].to_numpy()
            y_test = y.iloc[test_indices].to_numpy()

            if classification:
                label_encoder = LabelEncoder()
                y_train = label_encoder.fit_transform(y_train)
                y_test = label_encoder.transform(y_test)
            targets.append(y_test)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            predictions.append(y_pred)
            if classification:
                y_proba = model.predict_proba(X_test)
                if y_proba.shape[1] == 2: 
                    y_proba = y_proba[:, 1]
                probabilities.append(y_proba)

        y_pred = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        probabilities = np.concatenate(probabilities, axis=0) if len(probabilities) > 0 else None
        dataset_predictions[str(dataset.name)] = (targets, y_pred, probabilities)
    return dataset_predictions


def evaluate_openml_tasks(
    *,
    model_type: str,
    checkpoint: str | None = None,
    dist_path: str | None = None,
    tasks: list[int] | str = "tabarena-v0.1",
    cache_directory: str | None = None,
    max_n_features: int = 500,
    max_n_samples: int = 10_000,
    num_mem_chunks: int = 8,
):
    if model_type not in {"classification", "regression"}:
        raise ValueError("model_type must be 'classification' or 'regression'")

    if model_type == "classification":
        model = NanoTabPFNClassifier(model=checkpoint, num_mem_chunks=num_mem_chunks)
    else:
        model = NanoTabPFNRegressor(model=checkpoint, dist=dist_path, num_mem_chunks=num_mem_chunks)
    model.model.eval()

    if tasks == "toy_tasks" and model_type == "regression":
        tasks = TOY_TASKS_REGRESSION
    elif tasks == "toy_tasks" and model_type == "classification":
        tasks = TOY_TASKS_CLASSIFICATION
    else:
        tasks = tasks

    predictions = get_openml_predictions(
        model=model,
        tasks=tasks,
        max_n_features=max_n_features,
        max_n_samples=max_n_samples,
        classification=(model_type == "classification"),
        cache_directory=cache_directory,
    )

    average_score = 0.0
    for dataset_name, (y_true, y_pred, y_proba) in predictions.items():
        if model_type == "classification":
            acc = balanced_accuracy_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr")
            average_score += auc
            print(f"Dataset: {dataset_name} | ROC AUC: {auc:.4f} | Balanced Accuracy: {acc:.4f}")
        else:
            r2 = r2_score(y_true, y_pred)
            average_score += r2
            print(f"Dataset: {dataset_name} | R2: {r2:.4f}")

    average_score /= len(predictions)
    print(f"Average {'ROC AUC' if model_type == 'classification' else 'R2'}: {average_score:.4f}")


# -----------------------------------------------------------------------------
# train


def train(
    model: NanoTabPFNModel,
    prior: DataLoader,
    criterion: nn.CrossEntropyLoss | FullSupportBarDistribution,
    epochs: int,
    accumulate_gradients: int = 1,
    lr: float = 1e-4,
    device: torch.device = None,
    callbacks: list[Callback] = None,
    ckpt: Dict[str, torch.Tensor] = None,
    multi_gpu: bool = False,
    run_name: str = "nanoTFM",
):
    if multi_gpu:
        model = nn.DataParallel(model)
    if callbacks is None:
        callbacks = []
    if not device:
        device = get_default_device()
    model.to(device)
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=0.0)
    if ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    classification_task = isinstance(criterion, nn.CrossEntropyLoss)
    regression_task = not classification_task

    assert prior.num_steps % accumulate_gradients == 0, "num_steps must be divisible by accumulate_gradients"

    try:
        for epoch in range(ckpt["epoch"] + 1 if ckpt else 1, epochs + 1):
            epoch_start_time = time.time()
            model.train()
            optimizer.train()
            total_loss = 0.0
            for i, full_data in enumerate(prior):
                single_eval_pos = full_data["single_eval_pos"]
                data = (
                    full_data["x"].to(device),
                    full_data["y"][:, :single_eval_pos].to(device),
                )
                if torch.isnan(data[0]).any() or torch.isnan(data[1]).any():
                    continue
                targets = full_data["target_y"].to(device)

                if regression_task:
                    y_mean = data[1].mean(dim=1, keepdim=True)
                    y_std = data[1].std(dim=1, keepdim=True) + 1e-8
                    y_norm = (data[1] - y_mean) / y_std
                    data = (data[0], y_norm)

                output = model(data, single_eval_pos=single_eval_pos)
                targets = targets[:, single_eval_pos:]
                if regression_task:
                    targets = (targets - y_mean) / y_std
                if classification_task:
                    targets = targets.reshape((-1,)).to(torch.long)
                    output = output.view(-1, output.shape[-1])

                losses = criterion(output, targets)
                loss = losses.mean() / accumulate_gradients
                loss.backward()
                total_loss += loss.cpu().detach().item() * accumulate_gradients

                if (i + 1) % accumulate_gradients == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            end_time = time.time()
            mean_loss = total_loss / len(prior)
            model.eval()
            optimizer.eval()

            training_state = {
                "epoch": epoch,
                "architecture": {
                    "num_layers": int((model.module if multi_gpu else model).num_layers),
                    "embedding_size": int((model.module if multi_gpu else model).embedding_size),
                    "num_attention_heads": int((model.module if multi_gpu else model).num_attention_heads),
                    "mlp_hidden_size": int((model.module if multi_gpu else model).mlp_hidden_size),
                    "num_outputs": int((model.module if multi_gpu else model).num_outputs),
                },
                "model": (model.module if multi_gpu else model).state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(training_state, "workdir/checkpoints/latest_checkpoint.pth")

            for callback in callbacks:
                if type(criterion) is FullSupportBarDistribution:
                    callback.on_epoch_end(epoch, end_time - epoch_start_time, mean_loss, (model.module if multi_gpu else model), dist=criterion)
                else:
                    callback.on_epoch_end(epoch, end_time - epoch_start_time, mean_loss, (model.module if multi_gpu else model))
    except KeyboardInterrupt:
        pass
    finally:
        for callback in callbacks:
            callback.close()

    return (model.module if multi_gpu else model), total_loss


# -----------------------------------------------------------------------------
# main


@dataclass
class Config:
    dumps_dir = "workdir/dumps"
    logs_dir = "workdir/logs"
    checkpoints_dir = "workdir/checkpoints"
    models_dir = "workdir/models"
    classification_priordump = "workdir/dumps/50x3_3_100k_classification.h5"
    regression_priordump = "workdir/dumps/50x3_1280k_regression.h5"
    classifier_ckpt = "workdir/models/nano_classifier.pth"
    regressor_ckpt = "workdir/models/nano_regressor.pth"
    regressor_buckets = "workdir/models/nano_regressor_buckets.pth"
    ckpt = None  # "workdir/checkpoints/latest_checkpoint.pth"
    multigpu = False
    seed = 2402
    batch_size = 1
    accumulate = 1
    lr = 1e-4
    steps = 100
    epochs = 4
    num_attention_heads = 6
    embedding_size = 192
    mlp_hidden_size = 768
    num_layers = 6
    num_outputs = None
    n_buckets = 100


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--type",
        type=str,
        choices=["classification", "regression"],
        default="classification",
    )
    args = p.parse_args()

    c = Config()

    os.makedirs(c.dumps_dir, exist_ok=True)
    os.makedirs(c.logs_dir, exist_ok=True)
    os.makedirs(c.checkpoints_dir, exist_ok=True)
    os.makedirs(c.models_dir, exist_ok=True)

    set_randomness_seed(c.seed)

    device = get_default_device()

    ckpt = None

    if c.ckpt:
        ckpt = torch.load(c.ckpt)

    if args.type == "classification":
        prior = PriorDumpDataLoader(
            filename=c.classification_priordump,
            num_steps=c.steps,
            batch_size=c.batch_size,
            device=device,
            starting_index=c.steps * (ckpt["epoch"] if ckpt else 0),
        )
        c.num_outputs = prior.max_num_classes
        criterion = nn.CrossEntropyLoss()
        savepath = c.classifier_ckpt

        class ToyEvaluationLoggerCallback(ConsoleLoggerCallback):
            def __init__(self, tasks):
                self.tasks = tasks

            def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
                classifier = NanoTabPFNClassifier(model, device)
                predictions = get_openml_predictions(model=classifier, tasks=self.tasks)
                scores = []
                for dataset_name, (y_true, y_pred, y_proba) in predictions.items():
                    scores.append(accuracy_score(y_true, y_pred))
                avg_score = sum(scores) / len(scores)
                print(
                    f"epoch {epoch:5d} | time {epoch_time:5.2f}s | mean loss {loss:5.2f} | avg accuracy {avg_score:.3f}",
                    flush=True,
                )

        class ProductionEvaluationLoggerCallback(WandbLoggerCallback):
            def __init__(self, project: str, name: str = None, config: dict = None, log_dir: str = None):
                super().__init__(project, name, config, log_dir)

            def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
                classifier = NanoTabPFNClassifier(model, device)
                predictions = get_openml_predictions(model=classifier, classification=True)
                scores = []
                for dataset_name, (y_true, y_pred, y_proba) in predictions.items():
                    scores.append(roc_auc_score(y_true, y_proba, multi_class="ovr"))
                avg_score = sum(scores) / len(scores)
                self.wandb.log(
                    {
                        "epoch": epoch,
                        "epoch_time": epoch_time,
                        "mean_loss": loss,
                        "tabarena_avg_roc_auc": avg_score,
                    }
                )
                print(
                    f"epoch {epoch:5d} | time {epoch_time:5.2f}s | mean loss {loss:5.2f} | avg roc auc {avg_score:.3f}",
                    flush=True,
                )

        # callbacks = [ProductionEvaluationLoggerCallback('nanoTFM', 'nanoTabPFN-1')]
        callbacks = [ToyEvaluationLoggerCallback(TOY_TASKS_CLASSIFICATION)]
    elif args.type == "regression":
        prior = PriorDumpDataLoader(
            filename=c.regression_priordump,
            num_steps=c.steps,
            batch_size=c.batch_size,
            device=device,
            starting_index=c.steps * (ckpt["epoch"] if ckpt else 0),
        )
        c.num_outputs = c.n_buckets
        bucket_edges = make_global_bucket_edges(
            filename=c.regression_priordump,
            n_buckets=c.n_buckets,
            device=device,
        )
        torch.save(bucket_edges, c.regressor_buckets)
        criterion = FullSupportBarDistribution(bucket_edges)
        savepath = c.regressor_ckpt

        class EvaluationLoggerCallback(ConsoleLoggerCallback):
            def __init__(self, tasks):
                self.tasks = tasks

            def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
                regressor = NanoTabPFNRegressor(model, criterion, device)
                predictions = get_openml_predictions(model=regressor, tasks=self.tasks)
                scores = []
                for dataset_name, (y_true, y_pred, _) in predictions.items():
                    scores.append(r2_score(y_true, y_pred))
                avg_score = sum(scores) / len(scores)
                print(
                    f"epoch {epoch:5d} | time {epoch_time:5.2f}s | mean loss {loss:5.2f} | avg r2 score {avg_score:.3f}",
                    flush=True,
                )

        callbacks = [EvaluationLoggerCallback(TOY_TASKS_REGRESSION)]

    model = NanoTabPFNModel(
        num_attention_heads=c.num_attention_heads,
        embedding_size=c.embedding_size,
        mlp_hidden_size=c.mlp_hidden_size,
        num_layers=c.num_layers,
        num_outputs=c.num_outputs,
    )

    if ckpt:
        model.load_state_dict(ckpt["model"])

    trained_model, loss = train(
        model=model,
        prior=prior,
        criterion=criterion,
        epochs=c.epochs,
        accumulate_gradients=c.accumulate,
        lr=c.lr,
        device=device,
        callbacks=callbacks,
        ckpt=ckpt,
        multi_gpu=c.multigpu,
    )

    artifact = {
        "architecture": {
            "num_layers": c.num_layers,
            "embedding_size": c.embedding_size,
            "num_attention_heads": c.num_attention_heads,
            "mlp_hidden_size": c.mlp_hidden_size,
            "num_outputs": c.num_outputs,
        },
        "model": trained_model.state_dict(),
    }

    torch.save(artifact, savepath)


if __name__ == "__main__":
    main()
