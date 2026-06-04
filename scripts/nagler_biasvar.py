"""Reproduce the bias-variance experiment of Nagler (2023),
"Statistical Foundations of Prior-Data Fitted Networks", Section 6.5 / Figure 1.

Faithful port of the author's R gist:
https://gist.github.com/tnagler/62f6ce1f996333c799c81f1aef147e72

Data-generating process (d = 5):
    X ~ N(0, I_5),   P(Y=1 | X) = 1/2 + sin(1^T X)/2,   Y in {0,1}

For each (seed, n) we draw a fresh training set, predict P(Y=1) on a FIXED set of
quasi-random test points, and record the signed error  err = p_0(x) - p_hat(x).
Per test point i, aggregated over seeds:
    bias^2_i = ( mean_seed err )^2
    var_i    =   var_seed err
The reported curves average these over test points (gist: bias `mean(error)^2`,
variance `var(error)`, then `mean` over i).

"Localized" variant (gist): for n > 1000, predict each test point using only its
k_n nearest training neighbours, with
    k_n = round( n * min( (n/500)^(-d/(d+4)), 1 ) ).

We run two models side by side:
  * nano  : the repo's nanoTabPFN (a checkpoint)
  * tabpfn: official pretrained TabPFN (v2.5, loaded from local cache)

Output: a long-format CSV of raw errors (flushed incrementally) and a
bias/variance-vs-n plot. Global results are computed first so the main finding
survives even if the expensive localized arm is cut short.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm, qmc
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nano_model import NanoTabPFNClassifier, init_model_from_ckpt_file  # noqa: E402

D = 5  # feature dimension (fixed by the DGP)


# ---------------------------------------------------------------------------
# data-generating process


def p0(x):
    """True conditional probability P(Y=1 | x). x: (n, D)."""
    return 0.5 + 0.5 * np.sin(x.sum(axis=1))


def make_train(rng, n):
    x = rng.standard_normal((n, D))
    y = (rng.random(n) < p0(x)).astype(np.int64)
    return x, y


def make_test(n_test, seed=0):
    """Quasi-random N(0, I_D) test points (gist: qnorm(ghalton(500, 5)))."""
    sampler = qmc.Halton(d=D, scramble=True, seed=seed)
    u = sampler.random(n_test)
    u = np.clip(u, 1e-6, 1 - 1e-6)
    return norm.ppf(u)


def k_n(n):
    """Localized neighbourhood size (gist formula)."""
    return int(round(n * min((n / 500.0) ** (-D / (D + 4)), 1.0)))


# ---------------------------------------------------------------------------
# model wrappers: each exposes .fit(X, y) and .prob1(X) -> P(Y=1)


class NanoModel:
    name = "nano"

    def __init__(self, ckpt, residual_decay, device="cuda"):
        self._model = init_model_from_ckpt_file(ckpt, residual_decay=residual_decay)
        self.device = device

    def new(self):
        # fresh classifier sharing the loaded weights (cheap)
        m = NanoTabPFNClassifier.__new__(NanoTabPFNClassifier)
        m.model = self._model.to(self.device).eval()
        m.device = self.device
        return m

    def prob1(self, clf, X):
        clf.num_classes = 2  # binary task; use first two logits
        return clf.predict_proba(X)[:, 1]


class TabPFNModel:
    name = "tabpfn"

    def __init__(self, model_path, device="cuda"):
        from tabpfn import TabPFNClassifier
        self._cls = TabPFNClassifier
        self.model_path = model_path
        self.device = device

    def new(self):
        return self._cls(device=self.device, model_path=self.model_path)

    def prob1(self, clf, X):
        proba = clf.predict_proba(X)
        classes = list(clf.classes_)
        if 1 in classes:
            return proba[:, classes.index(1)]
        return np.zeros(len(X))  # train subset had no positives


def fit_predict_global(model, x_train, y_train, x_test):
    clf = model.new()
    clf.fit(x_train, y_train)
    return model.prob1(clf, x_test)


def fit_predict_localized(model, x_train, y_train, x_test, k):
    nn = NearestNeighbors(n_neighbors=k).fit(x_train)
    idx = nn.kneighbors(x_test, return_distance=False)
    preds = np.empty(len(x_test))
    for i in range(len(x_test)):
        sel = idx[i]
        clf = model.new()
        clf.fit(x_train[sel], y_train[sel])
        preds[i] = model.prob1(clf, x_test[i : i + 1])[0]
    return preds


# ---------------------------------------------------------------------------
# main study


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--ns", type=int, nargs="+", default=[200, 500, 1000, 2000, 4000])
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--localized-above", type=int, default=1000,
                    help="run the localized variant for n strictly above this")
    ap.add_argument("--nano-ckpt", default="records/20260506autohuman/"
                    "260506-163637-589ea1cf-autohuman-ckpt.pth")
    ap.add_argument("--nano-residual-decay", type=float, default=0.95)
    ap.add_argument("--tabpfn-path", default=os.path.expanduser(
        "~/.cache/tabpfn/tabpfn-v2.5-classifier-v2.5_default.ckpt"))
    ap.add_argument("--out-dir", default="workdir/nagler")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--models", nargs="+", default=["nano", "tabpfn"],
                    choices=["nano", "tabpfn"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"{args.tag}-errors.csv")

    x_test = make_test(args.n_test)
    true_p = p0(x_test)

    available = {
        "nano": lambda: NanoModel(args.nano_ckpt, args.nano_residual_decay, device=args.device),
        "tabpfn": lambda: TabPFNModel(args.tabpfn_path, device=args.device),
    }
    models = [available[m]() for m in args.models]

    # CSV holds raw signed errors, one row per (model, method, n, seed, test point).
    header_written = os.path.exists(csv_path)
    t_start = time.time()

    def flush(model_name, method, n, errs_by_seed):
        nonlocal header_written
        rows = []
        for seed, errs in errs_by_seed.items():
            for i, e in enumerate(errs):
                rows.append((model_name, method, n, seed, i, e, true_p[i]))
        df = pd.DataFrame(rows, columns=["model", "method", "n", "seed", "i", "error", "true"])
        df.to_csv(csv_path, mode="a", header=not header_written, index=False)
        header_written = True

    # ---- pass 1: global (cheap) for every model x n -------------------------
    for model in models:
        for n in args.ns:
            errs_by_seed = {}
            for seed in range(args.seeds):
                rng = np.random.default_rng(seed)
                x_tr, y_tr = make_train(rng, n)
                phat = fit_predict_global(model, x_tr, y_tr, x_test)
                errs_by_seed[seed] = true_p - phat
            flush(model.name, "tab", n, errs_by_seed)
            print(f"[{time.time()-t_start:7.1f}s] global  {model.name:7s} n={n:5d} done",
                  flush=True)

    # ---- pass 2: localized (expensive) for n above threshold ----------------
    for model in models:
        for n in args.ns:
            if n <= args.localized_above:
                continue
            k = k_n(n)
            errs_by_seed = {}
            for seed in range(args.seeds):
                rng = np.random.default_rng(seed)
                x_tr, y_tr = make_train(rng, n)
                phat = fit_predict_localized(model, x_tr, y_tr, x_test, k)
                errs_by_seed[seed] = true_p - phat
                if (seed + 1) % 20 == 0:
                    print(f"[{time.time()-t_start:7.1f}s] local   {model.name:7s} "
                          f"n={n:5d} k={k} seed {seed+1}/{args.seeds}", flush=True)
            flush(model.name, "ltab", n, errs_by_seed)
            print(f"[{time.time()-t_start:7.1f}s] local   {model.name:7s} n={n:5d} k={k} done",
                  flush=True)

    summarize(csv_path, args.out_dir, args.tag)


# ---------------------------------------------------------------------------
# aggregation + plot


def summarize(csv_path, out_dir, tag):
    df = pd.read_csv(csv_path)
    # per (model, method, n, i): squared bias and variance across seeds
    per_i = df.groupby(["model", "method", "n", "i"])["error"].agg(
        bias2=lambda e: e.mean() ** 2, var="var").reset_index()
    agg = per_i.groupby(["model", "method", "n"]).agg(
        bias2=("bias2", "mean"), variance=("var", "mean")).reset_index()
    agg = agg.sort_values(["model", "method", "n"])
    summ_path = os.path.join(out_dir, f"{tag}-summary.csv")
    agg.to_csv(summ_path, index=False)
    print("\n=== summary (avg squared bias / avg variance vs n) ===")
    print(agg.to_string(index=False))
    print(f"\nsummary -> {summ_path}")

    try:
        plot(agg, out_dir, tag)
    except Exception as e:  # plotting is optional; data is already saved
        print(f"(plot skipped: {type(e).__name__}: {e})")


def plot(agg, out_dir, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted(agg["model"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    styles = {"tab": ("-", "o"), "ltab": ("--", "s")}
    colors = {m: c for m, c in zip(models, ["#1f77b4", "#d62728", "#2ca02c"])}
    labels = {"tab": "global", "ltab": "localized"}

    for col, metric in enumerate(["bias2", "variance"]):
        ax = axes[col]
        for model in models:
            for method in ["tab", "ltab"]:
                sub = agg[(agg.model == model) & (agg.method == method)].sort_values("n")
                if sub.empty:
                    continue
                ls, mk = styles[method]
                ax.plot(sub["n"], sub[metric], ls, marker=mk, color=colors[model],
                        label=f"{model} ({labels[method]})")
        ax.set_xlabel("n (training set size)")
        ax.set_title("average squared bias" if metric == "bias2" else "average variance")
        ax.set_xscale("log")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Nagler (2023) Fig. 1 reproduction — bias/variance vs n")
    fig.tight_layout()
    out = os.path.join(out_dir, f"{tag}-bias-variance.png")
    fig.savefig(out, dpi=140)
    print(f"plot    -> {out}")


if __name__ == "__main__":
    main()
