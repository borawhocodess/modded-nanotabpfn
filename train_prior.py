import argparse
import os
import platform
import random
import socket
import subprocess
import sys
import time
import tomllib
import uuid
from dataclasses import dataclass, fields
from datetime import UTC, datetime

import numpy as np
import openml
import pandas as pd
import schedulefree
import torch
import torch._dynamo
import torch.nn.functional as F
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OrdinalEncoder
from torch import nn


@dataclass
class ScriptConfig:
    type: str = "classification"
    experiments_dir: str = "workdir/experiments"
    seed: int = 11
    batch_size: int = 2
    steps: int = 10000
    eval_every: int = 100
    grad_clip: float = 2.0
    max_train_mins: float = 10
    jackpot: float = 0.8068462330697953


@dataclass
class PriorConfig:
    min_num_classes: int = 2
    max_num_classes: int = 8
    min_num_cols: int = 20
    max_num_cols: int = 20
    min_num_parent_attempts: int = 3
    max_num_parent_attempts: int = 3
    min_redirection: float = 0.5
    max_redirection: float = 0.5
    min_num_rows: int = 1000
    max_num_rows: int = 1000
    min_num_test_rows: int = 128
    max_num_test_rows: int = 128


@dataclass
class ModelConfig:
    a: int = 4
    e: int = 256
    h: int = 768
    l: int = 5
    o: int = 8
    residual_decay: float = 0.95
    thinking_rows: int = 24
    feature_group_size: int = 5


@dataclass
class OptimizerConfig:
    lr: float = 0.001
    adam_wd: float = 0.01
    adam_warmup_steps: int = 1000
    muon_wd: float = 0.1
    muon_lr_scale: float = 0.1
    muon_momentum: float = 0.96
    muon_ns_steps: int = 5
    muon_ns_abc: tuple[float, float, float] = (3.4445, -4.7750, 2.0315)


@dataclass
class EvalConfig:
    seed: int = 11
    folds: int = 5
    max_features: int = 100
    max_samples: int = 1000


parser = argparse.ArgumentParser()
parser.add_argument("--name", default="test")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--steps", type=int, default=None)
args = parser.parse_args()

sc = ScriptConfig()
pc = PriorConfig()
mc = ModelConfig()
oc = OptimizerConfig()
ec = EvalConfig()

if args.seed is not None:
    sc.seed = args.seed
if args.steps is not None:
    sc.steps = args.steps

random.seed(sc.seed)
np.random.seed(sc.seed)
torch.manual_seed(sc.seed)
torch.set_float32_matmul_precision("high")
torch._dynamo.config.cache_size_limit = 128
assert torch.cuda.is_available()
device = "cuda"

start_ts = datetime.now(tz=UTC).astimezone()

with open(sys.argv[0], "r") as f:
    code = f.read()

with open("pyproject.toml", "rb") as f:
    version = tomllib.load(f)["project"]["version"]

tabarena_classification_tasks = [
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


class Prior:
    activations = (lambda z: z, torch.tanh, torch.sin, torch.abs, torch.square, F.softplus)

    def __init__(self, config, device):
        self.config = config
        self.device = device
        assert self.config.max_num_test_rows < self.config.min_num_rows

    def hyperparameters(self):
        c = self.config
        self.num_cols = int(np.random.randint(c.min_num_cols, c.max_num_cols + 1))
        self.num_rows = int(np.random.randint(c.min_num_rows, c.max_num_rows + 1))
        self.nodes = self.num_cols + 1
        self.num_test_rows = int(np.random.randint(c.min_num_test_rows, c.max_num_test_rows + 1))
        self.sep = self.num_rows - self.num_test_rows
        self.redirection = np.random.uniform(c.min_redirection, c.max_redirection)
        self.num_classes = int(np.random.randint(c.min_num_classes, c.max_num_classes + 1))
        self.num_parent_attempts = int(np.random.randint(c.min_num_parent_attempts, c.max_num_parent_attempts + 1))

    def gnr(self):
        parents = [[] for _ in range(self.nodes)]
        for child in range(1, self.nodes):
            chosen = set()
            for _ in range(self.num_parent_attempts):
                candidate = int(np.random.randint(child))
                if np.random.rand() < self.redirection and parents[candidate]:
                    candidate = int(np.random.choice(parents[candidate]))
                chosen.add(candidate)
            parents[child] = sorted(chosen)
        return parents

    def propagate(self):
        parents = self.gnr()
        w = np.zeros((self.nodes, self.nodes), dtype=np.float32)
        for i in range(1, self.nodes):
            w[i, parents[i]] = np.random.randn(len(parents[i]))
        w = torch.from_numpy(w).to(self.device)
        acts = np.random.randint(len(self.activations), size=self.nodes)
        z = torch.randn(self.num_rows, self.nodes, device=self.device)
        for i in range(1, self.nodes):
            zi = self.activations[acts[i]](z @ w[i]) + 0.1 * z[:, i]
            std, mean = torch.std_mean(zi)
            z[:, i] = (zi - mean) / (std + 1e-6)
        return z

    def target(self, z):
        target = int(np.random.randint(1, self.nodes))
        zt = z[:, target].contiguous()
        cuts = torch.linspace(0, 1, self.num_classes + 1, device=self.device)[1:-1]
        y = torch.bucketize(zt, zt.quantile(cuts))
        x = torch.cat([z[:, :target], z[:, target + 1 :]], dim=1)
        return x, y.float()

    def postprocess(self, x):
        return x

    def dataset(self):
        z = self.propagate()
        x, y = self.target(z)
        x = self.postprocess(x)
        return x, y

    def batch(self, batch_size):
        self.hyperparameters()
        datasets = [self.dataset() for _ in range(batch_size)]
        x = torch.stack([d[0] for d in datasets])
        y = torch.stack([d[1] for d in datasets])
        sep = self.sep
        return x[:, :sep], y[:, :sep], x[:, sep:], y[:, sep:]


class PriorDataLoader:
    def __init__(self, prior, batch_size):
        self.prior = prior
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            yield self.prior.batch(self.batch_size)


class ModdedNanoTabPFNModel(nn.Module):
    def __init__(self, l, a, e, h, o, residual_decay, thinking_rows, feature_group_size):
        super().__init__()
        self.l = l
        self.a = a
        self.e = e
        self.h = h
        self.o = o
        self.feature_encoder = FeatureEncoder(e, feature_group_size=feature_group_size)
        self.target_encoder = TargetEncoder(e)
        self.transformer_encoder = TransformerEncoderStack(l, a, e, h, residual_decay=residual_decay)
        self.decoder = Decoder(e, h, o)
        self.thinking_rows = thinking_rows
        self.row_tokens = nn.Parameter(torch.empty(thinking_rows, e))
        nn.init.normal_(self.row_tokens)

        self.register_buffer("borders", None, persistent=True)

    def forward(self, X_train, y_train, X_test):
        sep = X_train.shape[1]
        x_src = torch.cat([X_train, X_test], dim=1)
        y_src = y_train.unsqueeze(-1)
        x_src = self.feature_encoder(x_src, sep)
        num_rows = x_src.shape[1]
        y_src = self.target_encoder(y_src, num_rows)
        src = torch.cat([x_src, y_src], 2)
        b, _r, c, _e = src.shape
        thinking = self.row_tokens.unsqueeze(0).unsqueeze(2).expand(b, -1, c, -1)
        src = torch.cat([thinking, src], dim=1)
        sep = sep + self.thinking_rows
        output = self.transformer_encoder(src, sep)
        output = output[:, sep:, :-1, :].mean(dim=2)
        output = self.decoder(output)
        return output


class FeatureEncoder(nn.Module):
    def __init__(self, e, feature_group_size):
        super().__init__()
        self.feature_group_size = feature_group_size
        self.linear_layer = nn.Linear(feature_group_size, e)

    def forward(self, x, sep):
        n_cols = x.shape[-1]
        idxs = torch.arange(n_cols, dtype=torch.long, device=x.device)
        x = torch.stack([x[:, :, (idxs + (2**i - 1)) % n_cols] for i in range(self.feature_group_size)], dim=-1)
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
    def __init__(self, l, a, e, h, residual_decay):
        super().__init__()
        self.residual_decay = residual_decay
        self.transformer_blocks = nn.ModuleList()
        for _ in range(l):
            self.transformer_blocks.append(TransformerEncoderLayer(a, e, h))

    def forward(self, x, sep):
        for i, block in enumerate(self.transformer_blocks):
            x = x * (self.residual_decay**i)
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

        self.norm1 = nn.RMSNorm(e, eps=eps)
        self.norm2 = nn.RMSNorm(e, eps=eps)
        self.norm3 = nn.RMSNorm(e, eps=eps)

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


@torch.compile
def zeropower_via_newtonschulz5(G, steps, abc, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = abc
    X = G.bfloat16()
    X /= X.norm() + eps
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
def zeropower_via_newtonschulz5_batched(G, steps, abc, eps=1e-7):
    a, b, c = abc
    X = G.bfloat16()
    X /= X.norm(dim=(1, 2), keepdim=True) + eps
    if X.size(1) > X.size(2):
        X = X.transpose(1, 2)
    for _ in range(steps):
        A = X @ X.transpose(1, 2)
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """
    code adapted from: https://github.com/KellerJordan/modded-nanogpt/commit/b356a1f
    """

    def __init__(self, params, lr, momentum, weight_decay, ns_steps, ns_abc):
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "ns_steps": ns_steps,
            "ns_abc": ns_abc,
        }
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum)
                if g.size(0) == 3 * g.size(1):
                    g_batched = g.view(3, g.size(1), g.size(1))
                    g_new = zeropower_via_newtonschulz5_batched(g_batched, steps=group["ns_steps"], abc=group["ns_abc"])
                    g = g_new.view(3 * g.size(1), g.size(1))
                    scale = g.size(1) ** 0.5
                else:
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"], abc=group["ns_abc"])
                    scale = max(g.size(0), g.size(1)) ** 0.5
                p.data.add_(g, alpha=-lr * scale)
                if group["weight_decay"] > 0:
                    p.data.mul_(1 - lr * group["weight_decay"])


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
            ("to_numeric", FunctionTransformer(lambda x: pd.DataFrame(x).apply(pd.to_numeric, errors="coerce"))),
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


class ModdedNanoTabPFNClassifier:
    def __init__(self, model):
        device = "cuda"
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
        x_test = self.feature_preprocessor.transform(X_test)
        with torch.no_grad():
            x_train = torch.from_numpy(self.X_train).unsqueeze(0).to(torch.float).to(self.device)
            y_train = torch.from_numpy(self.y_train).unsqueeze(0).to(torch.float).to(self.device)
            x_test = torch.from_numpy(x_test).unsqueeze(0).to(torch.float).to(self.device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = self.model(x_train, y_train, x_test).squeeze(0)
                out = out[:, : self.num_classes]
                probabilities = F.softmax(out, dim=1)

            return probabilities.to("cpu").numpy()


def evaluate(model, tasks, config):
    clf = ModdedNanoTabPFNClassifier(model)
    aucs = []

    for task_id in tasks:
        task = openml.tasks.get_task(task_id, download_splits=False)

        dataset = task.get_dataset(download_data=False)
        X, y, _, _ = dataset.get_data(target=task.target_name, dataset_format="dataframe")

        len_features = X.shape[1]
        if len_features > config.max_features:
            rng = np.random.default_rng(config.seed)
            feature_choices = rng.choice(len_features, size=config.max_features, replace=False)
            X = X.iloc[:, feature_choices]

        if len(X) > config.max_samples:
            _, X, _, y = train_test_split(X, y, test_size=config.max_samples, stratify=y, random_state=config.seed)
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)

        cv = StratifiedKFold(n_splits=config.folds, shuffle=True, random_state=config.seed)

        targets = []
        probabilities = []

        for train_indices, test_indices in cv.split(X, y):
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
        y_proba = np.concatenate(probabilities, axis=0)

        auc = roc_auc_score(y_true, y_proba, multi_class="ovr") if y_proba.ndim > 1 else roc_auc_score(y_true, y_proba)
        aucs.append(auc)

    return aucs


def print0(s, console=False):
    with open(log_path, "a") as f:
        if console:
            print(s)
        print(s, file=f)


prior = Prior(config=pc, device=device)
loader = PriorDataLoader(prior=prior, batch_size=sc.batch_size)
batches = iter(loader)

model = ModdedNanoTabPFNModel(
    l=mc.l,
    a=mc.a,
    e=mc.e,
    h=mc.h,
    o=mc.o,
    residual_decay=mc.residual_decay,
    thinking_rows=mc.thinking_rows,
    feature_group_size=mc.feature_group_size,
).to(device)

muon_params = []
adam_params = []
for name, p in model.named_parameters():
    if p.ndim != 2:
        adam_params.append(p)
    elif "transformer_encoder" in name:
        muon_params.append(p)
    else:
        adam_params.append(p)

optimizer_muon = Muon(
    muon_params,
    lr=oc.muon_lr_scale * oc.lr,
    momentum=oc.muon_momentum,
    weight_decay=oc.muon_wd,
    ns_steps=oc.muon_ns_steps,
    ns_abc=oc.muon_ns_abc,
)
optimizer_adam = schedulefree.AdamWScheduleFree(
    adam_params,
    lr=oc.lr,
    weight_decay=oc.adam_wd,
    warmup_steps=oc.adam_warmup_steps,
)

optimizers = [optimizer_muon, optimizer_adam]

criterion = nn.CrossEntropyLoss()

ts = start_ts.strftime("%y%m%d-%H%M%S")
uid = uuid.uuid4().hex[:8]
e_name = args.name.strip()
e_id = f"{ts}-{uid}-{e_name}" if e_name else f"{ts}-{uid}"
e_root = os.path.join(sc.experiments_dir, e_name) if e_name else sc.experiments_dir
e_dir = os.path.join(e_root, e_id)
os.makedirs(e_dir, exist_ok=True)
log_path = os.path.join(e_dir, f"{e_id}-log.txt")
ckpt_path = os.path.join(e_dir, f"{e_id}-ckpt.pth")

print0(code)
print0("=" * 100)
print0(f"start timestamp: {start_ts.strftime('%Y-%m-%d %H:%M:%S')}", console=True)
print0(f"host: {socket.gethostname()}")
print0(f"platform: {platform.platform()}")
print0(f"python: {sys.version}")
print0(f"torch: {torch.version.__version__}")
print0(f"cuda: {torch.version.cuda}")
print0(subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False).stdout)
print0("=" * 100)

train_time = 0.0
total_loss = 0.0

for step in range(1, sc.steps + 1):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.train()
    optimizer_adam.train()

    x_train, y_train, x_test, y_test = next(batches)

    for opt in optimizers:
        opt.zero_grad(set_to_none=True)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(x_train, y_train, x_test)
        output = output.reshape(-1, output.shape[-1])
        targets = y_test.reshape((-1,)).to(torch.long)
        loss = criterion(output, targets)

    loss.backward()

    total_loss += loss.detach()

    torch.nn.utils.clip_grad_norm_(model.parameters(), sc.grad_clip)
    for opt in optimizers:
        opt.step()

    torch.cuda.synchronize()
    train_time += time.perf_counter() - t0

    if train_time > sc.max_train_mins * 60:
        print0("exceeded max train time", console=True)
        sys.exit(0)

    if step % sc.eval_every != 0:
        continue

    mean_loss = (total_loss / sc.eval_every).cpu().item()
    total_loss = 0.0

    model.eval()
    optimizer_adam.eval()

    aucs = evaluate(model, tabarena_classification_tasks, config=ec)
    avg_auc = sum(aucs) / len(aucs)

    run_time = (datetime.now(tz=UTC) - start_ts).total_seconds() / 60

    print0(
        f"s:{step}/{sc.steps} r_t:{run_time:.2f}m t_t:{train_time:.2f}s μ_l:{mean_loss:.2f} avg_roc_auc:{avg_auc}",
        console=True,
    )

    if avg_auc >= sc.jackpot:
        ckpt = {
            "version": version,
            "timestamp": ts,
            "uid": uid,
            "type": sc.type,
            "arch": {
                "e": model.e,
                "a": model.a,
                "h": model.h,
                "l": model.l,
                "o": model.o,
                "residual_decay": model.transformer_encoder.residual_decay,
                "thinking_rows": model.thinking_rows,
                "feature_group_size": model.feature_encoder.feature_group_size,
            },
            "model": model.state_dict(),
        }
        torch.save(ckpt, ckpt_path)
        print0("=" * 100)
        print0(f"datasets seen: {step * sc.batch_size}", console=True)
        print0(f"record time in mins: {train_time / 60:.2f}", console=True)
        break

print0("=" * 100)
print0("script config:")
for f in fields(ScriptConfig):
    print0(f"  {f.name}: {getattr(sc, f.name)}")
print0("prior config:")
for f in fields(PriorConfig):
    print0(f"  {f.name}: {getattr(pc, f.name)}")
print0("model config:")
for f in fields(ModelConfig):
    print0(f"  {f.name}: {getattr(mc, f.name)}")
print0("optimizer config:")
for f in fields(OptimizerConfig):
    print0(f"  {f.name}: {getattr(oc, f.name)}")
print0("eval config:")
for f in fields(EvalConfig):
    print0(f"  {f.name}: {getattr(ec, f.name)}")
print0("=" * 100)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print0(f"params: {total_params:,} (trainable: {trainable_params:,})")
print0("=" * 100)
print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB", console=True)
print0(f"peak memory reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
print0("=" * 100)
end_ts = datetime.now(tz=UTC).astimezone()
print0(f"end timestamp: {end_ts.strftime('%Y-%m-%d %H:%M:%S')}", console=True)
print0(f"script runtime: {(end_ts - start_ts).total_seconds() / 60:.2f} mins")
print0(f"experiment done: {e_id}", console=True)
