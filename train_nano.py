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
        """Initializes the feature/target encoder, transformer stack and decoder"""
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
        """
        Provides two interfaces:
        model(X_train, y_train, X_test)
            Args:
                X_train: (torch.Tensor) a tensor of shape (batch_size, num_train_datapoints, num_features)
                y_train: (torch.Tensor) a tensor of shape (batch_size, num_train_datapoints, 1)
                X_test: (torch.Tensor) a tensor of shape (batch_size, num_test_datapoints, num_features)

        model((x,y), single_eval_pos)
            Args:
                x: (torch.Tensor) a tensor of shape (batch_size, num_datapoints, num_features)
                y: (torch.Tensor) a tensor of shape (batch_size, num_train_datapoints, 1)


        The former is similar to the sklearn interface.
        In the latter x is the concatenation of X_train and X_test, y is y_train and single_eval_pos is the length of X_train.
        Our model internally works with the latter representation, so we convert the former into the latter and forward it to
        _forward.

        Returns:
            (torch.Tensor) a tensor of shape (batch_size, num_test_datapoints, num_classes),
                           which represent the predicted logits
        """
        if len(args) == 3:
            # case model(train_x, train_y, test_x)
            x = args[0]
            if args[2] is not None:
                x = torch.cat((x, args[2]), dim=1)
            return self._forward((x, args[1]), single_eval_pos=len(args[0]), **kwargs)
        elif len(args) == 1 and isinstance(args, tuple):
            # case model((x,y), single_eval_pos=None)
            return self._forward(*args, **kwargs)

    def _forward(self, src: Tuple[torch.Tensor, torch.Tensor], single_eval_pos: int, num_mem_chunks: int = 1) -> torch.Tensor:
        x_src, y_src = src
        # we expect the labels to look like (batches, num_train_datapoints, 1),
        # so we add the last dimension if it is missing
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src.unsqueeze(-1)
        # from here on B=Batches, R=Rows, C=Columns, E=embedding size
        # converts scalar values to embeddings, so (B,R,C-1) -> (B,R,C-1,E)
        x_src = self.feature_encoder(x_src, single_eval_pos)
        num_rows = x_src.shape[1]
        # padds the y_train up to y by using the mean,
        # then converts scalar values to embeddings (B,R,1,E)
        y_src = self.target_encoder(y_src, num_rows)
        # concatenates the feature embeddings with the target embeddings
        # to give us the full table of embeddings (B,R,C,E))
        src = torch.cat([x_src, y_src], 2)
        # repeatedly applies the transformer block on (B,R,C,E)
        output = self.transformer_encoder(src, single_eval_pos, num_mem_chunks=num_mem_chunks)
        # selects the target embeddings (B,num_targets,1,E)
        output = output[:, single_eval_pos:, -1, :]
        # runs the embeddings through the decoder to get
        # the logits of our predictions (B,num_targets,num_classes)
        output = self.decoder(output)
        return output


# handle variable number of features in here?
class FeatureEncoder(nn.Module):
    def __init__(self, embedding_size: int):
        """Creates the linear layer that we will use to embed our features."""
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)

    def forward(self, x: torch.Tensor, single_eval_pos: int) -> torch.Tensor:
        """
        Normalizes all the features based on the mean and std of the features of the training data,
        clips them between -100 and 100, then applies a linear layer to embed the features.

        Args:
            x: (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features)
            single_eval_pos: (int) the number of datapoints in X_train
        Returns:
            (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features, embedding_size), representing
                           the embeddings of the features
        """
        x = x.unsqueeze(-1)
        mean = torch.mean(x[:, :single_eval_pos], axis=1, keepdims=True)
        std = torch.std(x[:, :single_eval_pos], axis=1, keepdims=True) + 1e-20
        x = (x - mean) / std
        x = torch.clip(x, min=-100, max=100)
        return self.linear_layer(x)


class TargetEncoder(nn.Module):
    def __init__(self, embedding_size: int):
        """Creates the linear layer that we will use to embed our targets."""
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)

    def forward(self, y_train: torch.Tensor, num_rows: int) -> torch.Tensor:
        """
        Pads up y_train to the full length of y using the mean per dataset and then embeds it using a linear layer

        Args:
            y_train: (torch.Tensor) a tensor of shape (batch_size, num_train_datapoints, 1)
            num_rows: (int) the full length of y
        Returns:
            (torch.Tensor) a tensor of shape (batch_size, num_rows, 1, embedding_size), representing
                           the embeddings of the targets
        """
        # nan padding & nan handler instead?
        mean = torch.mean(y_train, axis=1, keepdim=True)
        padding = mean.repeat(1, num_rows - y_train.shape[1], 1)
        y = torch.cat([y_train, padding], dim=1)
        y = y.unsqueeze(-1)
        return self.linear_layer(y)


class TransformerEncoderStack(nn.Module):
    def __init__(self, num_layers: int, embedding_size: int, num_attention_heads: int, mlp_hidden_size: int):
        """Instantiates num_layers many Transformer Blocks and stores them in a list so we can use them in the forward."""
        super().__init__()
        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_blocks.append(TransformerEncoderLayer(embedding_size, num_attention_heads, mlp_hidden_size))

    def forward(self, x: torch.Tensor, single_eval_position: int, num_mem_chunks: int = 1) -> torch.Tensor:
        """
        Takes the embeddings of all the cells of the table as input and applies num_layers many Transformer blocks.

        Args:
            x: (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features, embedding_size) that contains all the embeddings
                              for all the cells in the table
            single_eval_position: (int) the length of X_train
            num_mem_chunks: (int) Number of chunks that memory-intense operations will be split into. Higher values use less memory but are slower.
                                  Needs to be set to 1 during training to get correct gradients.

        Returns
            (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features, embedding_size)
        """
        for block in self.transformer_blocks:
            x = block(x, single_eval_position=single_eval_position, num_mem_chunks=num_mem_chunks)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Modified version of older version of https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/transformer.py#L630
    """

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
        """
        Takes the embeddings of the table as input and applies self-attention between features and self-attention between datapoints
        followed by a simple 2 layer MLP.

        Args:
            src: (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features, embedding_size) that contains all the embeddings
                                for all the cells in the table
            single_eval_position: (int) the length of X_train
            num_mem_chunks: (int) Number of chunks that memory-intense operations will be split into. Higher values use less memory but are slower.
                                  Needs to be set to 1 during training to get correct gradients.
        Returns
            (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features, embedding_size)
        """
        batch_size, rows_size, col_size, embedding_size = src.shape
        # attention between features
        src = src.reshape(batch_size * rows_size, col_size, embedding_size)

        @memory_chunking(num_mem_chunks)
        def feature_attention(x):
            return self.self_attention_between_features(x, x, x)[0] + x

        src = feature_attention(src)
        src = src.reshape(batch_size, rows_size, col_size, embedding_size)
        src = self.norm1(src)
        # attention between datapoints
        src = src.transpose(1, 2)
        src = src.reshape(batch_size * col_size, rows_size, embedding_size)

        @memory_chunking(num_mem_chunks)
        def datapoint_attention(x):
            x_left = self.self_attention_between_datapoints(x[:, :single_eval_position], x[:, :single_eval_position], x[:, :single_eval_position])[0]
            # test data attends to the training data
            x_right = self.self_attention_between_datapoints(x[:, single_eval_position:], x[:, :single_eval_position], x[:, :single_eval_position])[0]
            return torch.cat([x_left, x_right], dim=1) + x

        src = datapoint_attention(src)
        src = src.reshape(batch_size, col_size, rows_size, embedding_size)
        src = src.transpose(2, 1)
        src = self.norm2(src)
        # MLP after attention
        src = src.reshape(-1, embedding_size)

        @memory_chunking(num_mem_chunks)
        def mlp(x):
            return self.linear2(F.gelu(self.linear1(x))) + x

        src = mlp(src)
        src = src.reshape(batch_size, rows_size, col_size, embedding_size)
        src = self.norm3(src)
        return src


def memory_chunking(num_mem_chunks: int) -> callable:
    """
    This decorator will split the first dimension of the input into chunks and apply the wrapped function
    to each chunk separately.
    Args:
        num_mem_chunks: (int) Number of chunks to split the input into, higher values use less memory but are slower.
                          Needs to be set to 1 during training to disable chunking and get correct gradients.
    """
    def decorator(func: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
        def wrapper(x: torch.Tensor) -> torch.Tensor:
            if num_mem_chunks <= 1 or x.shape[0] == 0:
                return func(x)
            elif torch.is_grad_enabled():
                warnings.warn("Memory chunking is disabled since gradient computation is enabled to avoid incorrect gradients. "
                              "Please use `with torch.no_grad():` during inference to enable chunking.")
                return func(x)
            chunk_size = max(1, math.ceil(x.shape[0] / num_mem_chunks))
            for x_split in torch.split(x, split_size_or_sections=chunk_size, dim=0):
                x_split[:] = func(x_split) # in-place modification to save memory, will cause wrong gradients if used during training
            return x

        return wrapper

    return decorator


class Decoder(nn.Module):
    def __init__(self, embedding_size: int, mlp_hidden_size: int, num_outputs: int):
        """Initializes the linear layers for use in the forward"""
        super().__init__()
        self.linear1 = nn.Linear(embedding_size, mlp_hidden_size)
        self.linear2 = nn.Linear(mlp_hidden_size, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies an MLP to the embeddings to get the logits

        Args:
            x: (torch.Tensor) a tensor of shape (batch_size, num_rows, embedding_size)
        Returns:
            (torch.Tensor) a tensor of shape (batch_size, num_rows, num_outputs)
        """
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
                    print("""Finished iteration over all stored datasets! """
                          """Will start reusing the same data with different splits now.""")
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
        total = num_tables * num_datapoints

        if max_y >= total:
            ys_concat = y[...].reshape(-1)
        else:
            full_rows = max_y // num_datapoints
            rem =  max_y % num_datapoints

            parts = []
            if full_rows > 0:
                parts.append(y[:full_rows, :].reshape(-1))
            if rem > 0:
                parts.append(y[full_rows, :rem].reshape(-1))
            ys_concat = np.concatenate(parts, axis=0) if parts else np.empty((0,), dtype=y.dtype)

    if ys_concat.size < n_buckets:
        raise ValueError(f"Too few target samples ({ys_concat.size}) to compute {n_buckets} buckets.")

    ys_tensor = torch.tensor(ys_concat, dtype=torch.float32, device=device)
    global_bucket_edges = get_bucket_limits(n_buckets, ys=ys_tensor).to(device)
    return global_bucket_edges


# -----------------------------------------------------------------------------
# interface

def init_model_from_state_dict_file(file_path):
    """
    reads model architecture from state dict, instantiates the architecture and loads the weights
    """
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


# doing these as lambdas would cause NanoTabPFNClassifier to not be pickle-able,
# which would cause issues if we want to run it inside the tabarena codebase
def to_pandas(x):
    return pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x


def to_numeric(x):
    return x.apply(pd.to_numeric, errors="coerce").to_numpy()


def get_feature_preprocessor(X: np.ndarray | pd.DataFrame) -> ColumnTransformer:
    """
    fits a preprocessor that imputes NaNs, encodes categorical features and removes constant features
    """
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
        numeric_entries = pd.to_numeric(X[col], errors='coerce').notna().sum() # in case numeric columns are stored as strings
        num_mask.append(non_nan_entries == numeric_entries)
        cat_mask.append(non_nan_entries != numeric_entries)
        # num_mask.append(is_numeric_dtype(X[col]))  # Assumes pandas dtype is correct

    num_mask = np.array(num_mask)
    cat_mask = np.array(cat_mask)

    num_transformer = Pipeline([
        ("to_pandas", FunctionTransformer(to_pandas)), # to apply pd.to_numeric of pandas
        ("to_numeric", FunctionTransformer(to_numeric)), # in case numeric columns are stored as strings
        ('imputer', SimpleImputer(strategy='mean')) # median might be better because of outliers
    ])
    cat_transformer = Pipeline([
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)),
        ('imputer', SimpleImputer(strategy='most_frequent')),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_mask),
            ("cat", cat_transformer, cat_mask),
        ]
    )
    return preprocessor


class NanoTabPFNClassifier:
    """scikit-learn like interface"""
    def __init__(self, model: NanoTabPFNModel|str|None = None, device: None|str|torch.device = None, num_mem_chunks: int = 8):
        if device is None:
            device = get_default_device()
        if model is None:
            model = "nanotabpfn.pth"
            if not os.path.isfile(model):
                print('No cached model found, downloading model checkpoint.')
                response = requests.get('https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/nanoTabPFN/nanotabpfn_classifier.pth')
                with open(model, 'wb') as f:
                    f.write(response.content)
        if isinstance(model, str):
            model = init_model_from_state_dict_file(model)
        self.model = model.to(device)
        self.device = device
        self.num_mem_chunks = num_mem_chunks

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """stores X_train and y_train for later use, also computes the highest class number occuring in num_classes"""
        self.feature_preprocessor = get_feature_preprocessor(X_train)
        self.X_train = self.feature_preprocessor.fit_transform(X_train)
        self.y_train = y_train
        self.num_classes = max(set(y_train)) + 1

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """calls predit_proba and picks the class with the highest probability for each datapoint"""
        predicted_probabilities = self.predict_proba(X_test)
        return predicted_probabilities.argmax(axis=1)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        creates (x,y), runs it through our PyTorch Model, cuts off the classes that didn't appear in the training data
        and applies softmax to get the probabilities
        """
        x = np.concatenate((self.X_train, self.feature_preprocessor.transform(X_test)))
        y = self.y_train
        with torch.no_grad():
            x = torch.from_numpy(x).unsqueeze(0).to(torch.float).to(self.device)  # introduce batch size 1
            y = torch.from_numpy(y).unsqueeze(0).to(torch.float).to(self.device)
            out = self.model((x, y), single_eval_pos=len(self.X_train), num_mem_chunks=self.num_mem_chunks).squeeze(0)  # remove batch size 1
            # our pretrained classifier supports up to num_outputs classes, if the dataset has less we cut off the rest
            out = out[:, : self.num_classes]
            # apply softmax to get a probability distribution
            probabilities = F.softmax(out, dim=1)
            return probabilities.to("cpu").numpy()


class NanoTabPFNRegressor:
    """scikit-learn like interface"""
    def __init__(self, model: NanoTabPFNModel|str|None = None, dist: FullSupportBarDistribution|str|None = None, device: str|torch.device|None = None, num_mem_chunks: int = 8):
        if device is None:
            device = get_default_device()
        if model is None:
            model = "nanotabpfn_regressor.pth"
            dist = "nanotabpfn_regressor_buckets.pth"
            if not os.path.isfile(model):
                print('No cached model found, downloading model checkpoint.')
                response = requests.get('https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/nanoTabPFN/nanotabpfn_regressor.pth')
                with open(model, 'wb') as f:
                    f.write(response.content)
            if not os.path.isfile(dist):
                print('No cached bucket edges found, downloading bucket edges.')
                response = requests.get('https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/nanoTabPFN/nanotabpfn_regressor_buckets.pth')
                with open(dist, 'wb') as f:
                    f.write(response.content)
        if isinstance(model, str):
            model = init_model_from_state_dict_file(model)

        if isinstance(dist, str):
            bucket_edges = torch.load(dist, map_location=device)
            dist = FullSupportBarDistribution(bucket_edges).float()

        self.model = model.to(device)
        self.device = device
        self.dist = dist
        self.normalized_dist = None  # Used after fit()
        self.num_mem_chunks = num_mem_chunks

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Stores X_train and y_train for later use. Computes target normalization. Builds normalized bar distribution from existing self.dist.
        """
        self.feature_preprocessor = get_feature_preprocessor(X_train)
        self.X_train = self.feature_preprocessor.fit_transform(X_train)
        self.y_train = y_train

        self.y_train_mean = np.mean(self.y_train)
        self.y_train_std = np.std(self.y_train) + 1e-8
        self.y_train_n = (self.y_train - self.y_train_mean) / self.y_train_std

        # Convert base distribution to original scale for output
        bucket_edges = self.dist.borders
        bucket_edges_denorm = bucket_edges * self.y_train_std + self.y_train_mean
        self.normalized_dist = FullSupportBarDistribution(bucket_edges_denorm).float()

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Performs in-context learning using X_train and y_train. Predicts the means of the output distributions for X_test.
        """
        X = np.concatenate((self.X_train, self.feature_preprocessor.transform(X_test)))
        y = self.y_train_n

        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device).unsqueeze(0)
            y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(0)

            logits = self.model((X_tensor, y_tensor), single_eval_pos=len(self.X_train), num_mem_chunks=self.num_mem_chunks).squeeze(0)
            preds = self.normalized_dist.mean(logits)

        return preds.cpu().numpy()

# -----------------------------------------------------------------------------
# callbacks

class Callback(ABC):
    """Abstract base class for callbacks."""

    @abstractmethod
    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        """
        Called at the end of each epoch.

        Args:
            epoch (int): The current epoch number.
            epoch_time (float): Time of the epoch in seconds.
            loss (float): Mean loss for the epoch.
            model: The model being trained.
            **kwargs: Additional arguments.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Called to release any resources or perform cleanup.
        """
        pass


class BaseLoggerCallback(Callback):
    """Abstract base class for logger callbacks."""

    pass


class ConsoleLoggerCallback(BaseLoggerCallback):
    """Logger callback that prints epoch information to the console."""

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        print(f'Epoch {epoch:5d} | Time {epoch_time:5.2f}s | Mean Loss {loss:5.2f}', flush=True)

    def close(self):
        """Nothing to clean up for print logger."""
        pass


class TensorboardLoggerCallback(BaseLoggerCallback):
    """Logger callback that logs epoch information to TensorBoard."""

    def __init__(self, log_dir: str):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=log_dir)

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        self.writer.add_scalar("Loss/train", loss, epoch)
        self.writer.add_scalar("Time/epoch", epoch_time, epoch)

    def close(self):
        self.writer.close()


class WandbLoggerCallback(BaseLoggerCallback):
    """Logger callback that logs epoch information to Weights & Biases."""

    def __init__(self, project: str, name: str = None, config: dict = None, log_dir: str = None):
        """
        Initializes a WandbLoggerCallback.

        Args:
            project (str): The name of the wandb project.
            name (str, optional): The name of the run. Defaults to None.
            config (dict, optional): Configuration dictionary for the run. Defaults to None.
            log_dir (str, optional): Directory to save wandb logs. Defaults to None.
        """
        try:
            import wandb

            self.wandb = wandb  # store wandb module to avoid import if not used
            wandb.init(
                project=project,
                name=name,
                config=config,
                dir=log_dir,
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

# we hardcode the list here because even if the tasks are cached
# openml.study.get_suite("tabarena-v0.1") might fail if there are connection issues
TABARENA_TASKS = [
    363612, 363613, 363614, 363615, 363616, 363618, 363619, 363620, 363621, 363623, 363624, 363625, 363626, 363627, 
    363628, 363629, 363630, 363631, 363632, 363671, 363672, 363673, 363674, 363675, 363676, 363677, 363678, 363679, 
    363681, 363682, 363683, 363684, 363685, 363686, 363689, 363691, 363693, 363694, 363696, 363697, 363698, 363699, 
    363700, 363702, 363704, 363705, 363706, 363707, 363708, 363711, 363712
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
    """
    Evaluates a model on a set of OpenML tasks and returns predictions.

    Retrieves datasets from OpenML, applies preprocessing, and evaluates the given model on each task.
    Returns true targets, predicted labels, and predicted probabilities for each dataset.

    Args:
        model (NanoTabPFNRegressor | NanoTabPFNClassifier): A scikit-learn compatible model or classifier to be evaluated.
        tasks (list[int] | str, optional): A list of OpenML task IDs or the name of a benchmark suite.
        max_n_features (int, optional): Maximum number of features allowed for a task. Tasks exceeding this limit are skipped.
        max_n_samples (int, optional): Maximum number of instances allowed for a task. Tasks exceeding this limit are skipped.
        classification (bool | None, optional): Whether the model is a classifier (True) or regressor (False). If None, it is inferred from the model type.
        cache_directory (str | None, optional): Directory to save OpenML data. If None, default cache path is used.
    Returns:
        dict: A dictionary where keys are dataset names and values are tuples of (true targets, predicted labels, predicted probabilities).
    """
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
            continue  # skip task, only classification
        if not classification and task.task_type_id != TaskType.SUPERVISED_REGRESSION:
            continue  # skip task, only regression

        dataset = task.get_dataset(download_data=False)

        n_features = dataset.qualities["NumberOfFeatures"]
        n_samples = dataset.qualities["NumberOfInstances"]
        if n_features > max_n_features or n_samples > max_n_samples:
            continue  # skip task, too big

        _, folds, _ = task.get_split_dimensions()
        tabarena_light = True
        if tabarena_light:
            folds = 1  # code supports multiple folds but tabarena_light only has one
        repeat = 0  # code only supports one repeat
        targets = []
        predictions = []
        probabilities = []
        for fold in range(folds):
            X, y, categorical_indicator, attribute_names = dataset.get_data(
                target=task.target_name, dataset_format="dataframe"
            )
            train_indices, test_indices = task.get_train_test_split_indices(
                fold=fold, repeat=repeat
            )
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
                if y_proba.shape[1] == 2:  # binary classification
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
) -> dict:
    """
    Run the OpenML evaluation previously done in the CLI block, but as a callable function.

    Returns a metrics summary dict with per-dataset scores and the overall average.
    """
    if model_type not in {"classification", "regression"}:
        raise ValueError("model_type must be 'classification' or 'regression'")

    if model_type == "classification":
        model = NanoTabPFNClassifier(model=checkpoint, num_mem_chunks=num_mem_chunks)
    else:
        model = NanoTabPFNRegressor(model=checkpoint, dist=dist_path, num_mem_chunks=num_mem_chunks)
    model.model.eval()

    if tasks == "toy_tasks" and model_type == "regression":
        resolved_tasks = TOY_TASKS_REGRESSION
    elif tasks == "toy_tasks" and model_type == "classification":
        resolved_tasks = TOY_TASKS_CLASSIFICATION
    else:
        resolved_tasks = tasks

    predictions = get_openml_predictions(
        model=model,
        tasks=resolved_tasks,
        max_n_features=max_n_features,
        max_n_samples=max_n_samples,
        classification=(model_type == "classification"),
        cache_directory=cache_directory,
    )

    per_dataset = {}
    average_score = 0.0
    for dataset_name, (y_true, y_pred, y_proba) in predictions.items():
        if model_type == "classification":
            acc = balanced_accuracy_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr")
            average_score += auc
            per_dataset[dataset_name] = {
                "roc_auc": float(auc),
                "balanced_accuracy": float(acc),
            }
        else:
            r2 = r2_score(y_true, y_pred)
            average_score += r2
            per_dataset[dataset_name] = {
                "r2": float(r2),
            }

    metric_name = "ROC AUC" if model_type == "classification" else "R2"
    average_score = float(average_score / max(1, len(per_dataset)))
    return {
        "metric_name": metric_name,
        "average_score": average_score,
        "per_dataset": per_dataset,
    }

# -----------------------------------------------------------------------------
# train

def train(model: NanoTabPFNModel, prior: DataLoader, criterion: nn.CrossEntropyLoss | FullSupportBarDistribution,
          epochs: int, accumulate_gradients: int = 1, lr: float = 1e-4, device: torch.device = None,
          callbacks: list[Callback]=None, ckpt: Dict[str, torch.Tensor] = None, multi_gpu: bool = False):
    """
    Trains our model on the given prior using the given criterion.

    Args:
        model: (NanoTabPFNModel) our PyTorch model
        prior: (DataLoader) torch-compatible dataloader
        criterion: (nn.CrossEntropyLoss | FullSupportBarDistribution) our loss criterion
        epochs: (int) the number of epochs we train for, the number of steps that constitute an epoch are decided by the prior
        accumulate_gradients: (int) the number of gradients to accumulate before updating the weights
        device: (torch.device) the device we are using
        callbacks: A list of callback instances to execute at the end of each epoch. These can be used for
            logging, validation, or other custom actions.
        ckpt (Dict[str, torch.Tensor], optional): A checkpoint dictionary containing the model and optimizer states,
            as well as the last completed epoch. If provided, training resumes from this checkpoint.

    Returns:
        (torch.Tensor) a tensor of shape (num_rows, batch_size, num_features, embedding_size)
    """
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

    assert prior.num_steps % accumulate_gradients == 0, 'num_steps must be divisible by accumulate_gradients'

    try:
        for epoch in range(ckpt["epoch"] + 1 if ckpt else 1, epochs + 1):
            epoch_start_time = time.time()
            model.train()  # Turn on the train mode
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

                output = model(data, single_eval_pos=single_eval_pos)
                targets = targets[:, single_eval_pos:]
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
                'epoch': epoch,
                'architecture': {
                    'num_layers': int((model.module if multi_gpu else model).num_layers),
                    'embedding_size': int((model.module if multi_gpu else model).embedding_size),
                    'num_attention_heads': int((model.module if multi_gpu else model).num_attention_heads),
                    'mlp_hidden_size': int((model.module if multi_gpu else model).mlp_hidden_size),
                    'num_outputs': int((model.module if multi_gpu else model).num_outputs)
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
                print(f'epoch {epoch:5d} | time {epoch_time:5.2f}s | mean loss {loss:5.2f} | avg accuracy {avg_score:.3f}',
                    flush=True)

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
                self.wandb.log({
                    'epoch': epoch,
                    'epoch_time': epoch_time,
                    'mean_loss': loss,
                    'tabarena_avg_roc_auc': avg_score
                })
                print(f'epoch {epoch:5d} | time {epoch_time:5.2f}s | mean loss {loss:5.2f} | avg roc auc {avg_score:.3f}',
                    flush=True)

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
                print(f'epoch {epoch:5d} | time {epoch_time:5.2f}s | mean loss {loss:5.2f} | avg r2 score {avg_score:.3f}',
                    flush=True)


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
