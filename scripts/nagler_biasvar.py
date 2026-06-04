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
    """Baseline true conditional probability P(Y=1 | x) from Nagler (2023)."""
    return 0.5 + 0.5 * np.sin(x.sum(axis=1))


# Registry of target functions P(Y=1 | x), each mapping (n, D) -> (n,).
# Two sweeps share this registry:
#   * frequency sweep  : sin(w * 1^T X) for w in {0.5, 1, 2, 3, 4}  (freq1 == baseline)
#   * diverse structure: linear / highfreq / radial / interaction / xor
DGPS = {
    "freq0.5": lambda x: 0.5 + 0.5 * np.sin(0.5 * x.sum(axis=1)),
    "freq1": lambda x: 0.5 + 0.5 * np.sin(1.0 * x.sum(axis=1)),  # == baseline p0
    "freq2": lambda x: 0.5 + 0.5 * np.sin(2.0 * x.sum(axis=1)),
    "freq3": lambda x: 0.5 + 0.5 * np.sin(3.0 * x.sum(axis=1)),
    "freq4": lambda x: 0.5 + 0.5 * np.sin(4.0 * x.sum(axis=1)),
    "linear": lambda x: 1.0 / (1.0 + np.exp(-x.sum(axis=1) / np.sqrt(D))),
    "highfreq": lambda x: 0.5 + 0.5 * np.sin(2.0 * x.sum(axis=1)),  # coincides w/ freq2
    "radial": lambda x: 0.5 + 0.5 * np.sin((x ** 2).sum(axis=1) / 2.0),
    "interaction": lambda x: 0.5 + 0.5 * np.tanh(x[:, 0] * x[:, 1]),
    "xor": lambda x: (x[:, 0] * x[:, 1] > 0).astype(float),
}


def make_train(rng, n, prob_fn):
    x = rng.standard_normal((n, D))
    y = (rng.random(n) < prob_fn(x)).astype(np.int64)
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
    ap.add_argument("--dgp", default="freq1", choices=list(DGPS),
                    help="target function P(Y=1|x); freq1 == Nagler baseline")
    ap.add_argument("--plot", choices=["data", "grid", "aggregate", "all"], default=None,
                    help="instead of running, produce a figure from existing CSVs/DGPs")
    ap.add_argument("--summary", default="workdir/nagler/light100-summary.csv",
                    help="summary CSV for --plot grid")
    args = ap.parse_args()

    if args.plot:  # plotting mode: no model/GPU needed, reads CSVs + DGP registry
        run_plots(args)
        return

    prob_fn = DGPS[args.dgp]
    print(f"DGP = {args.dgp}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"{args.tag}-errors.csv")

    x_test = make_test(args.n_test)
    true_p = prob_fn(x_test)

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
                x_tr, y_tr = make_train(rng, n, prob_fn)
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
                x_tr, y_tr = make_train(rng, n, prob_fn)
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


# ---------------------------------------------------------------------------
# standalone figures (python nagler_biasvar.py --plot {data,grid,aggregate,all})


def run_plots(args):
    which = ["data", "grid", "aggregate"] if args.plot == "all" else [args.plot]
    if "data" in which:
        plot_data(args.out_dir)
    if "grid" in which:
        plot_grid(args.summary, args.out_dir)
    if "aggregate" in which:
        plot_aggregate(args.out_dir)


def plot_data(out_dir, n=600):
    """Visualize the synthetic datasets in the coordinate each target depends on."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, D))
    s, r2 = X.sum(axis=1), (X ** 2).sum(axis=1)

    def labels(fn):
        return (rng.random(n) < fn(X)).astype(int)

    def idx(ax, fn, coord, title, xlabel):
        y, p = labels(fn), fn(X)
        order = np.argsort(coord)
        ax.plot(coord[order], p[order], color="k", lw=2, label="true P(Y=1|x)", zorder=3)
        jit = (rng.random(n) - 0.5) * 0.06
        ax.scatter(coord, y + jit, s=10, alpha=0.35, c=np.where(y == 1, "#d62728", "#1f77b4"))
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("y / P(Y=1)")
        ax.set_ylim(-0.15, 1.15); ax.legend(fontsize=7, loc="center right")

    def sc2d(ax, fn, title):
        y = labels(fn)
        ax.scatter(X[:, 0], X[:, 1], s=14, alpha=0.6, c=np.where(y == 1, "#d62728", "#1f77b4"))
        ax.set_title(title); ax.set_xlabel("x1"); ax.set_ylabel("x2")
        ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    idx(ax[0, 0], DGPS["freq1"], s, "baseline:  1/2 + 1/2 sin(1^T X)", "s = 1^T X")
    gs = np.linspace(s.min(), s.max(), 500)
    for w, c in zip([0.5, 1, 2, 3, 4], plt.cm.viridis(np.linspace(0, 0.9, 5))):
        ax[0, 1].plot(gs, 0.5 + 0.5 * np.sin(w * gs), color=c, label=f"w={w}")
    ax[0, 1].set_title("frequency sweep:  sin(w * 1^T X)")
    ax[0, 1].set_xlabel("s = 1^T X"); ax[0, 1].set_ylabel("P(Y=1)"); ax[0, 1].legend(fontsize=7)
    idx(ax[0, 2], DGPS["linear"], s, "linear:  sigma(1^T X / sqrt(5))", "s = 1^T X")
    idx(ax[1, 0], DGPS["radial"], r2, "radial:  1/2 + 1/2 sin(||X||^2 / 2)", "r^2 = ||X||^2")
    sc2d(ax[1, 1], DGPS["interaction"], "interaction:  1/2 + 1/2 tanh(x1 x2)")
    sc2d(ax[1, 2], DGPS["xor"], "xor:  1{ x1 x2 > 0 }")
    fig.suptitle(f"Synthetic datasets — {n} samples each, X ~ N(0, I_5)  "
                 "(red = y=1, blue = y=0)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(out_dir, "data-samples.png")
    fig.savefig(out, dpi=130); print(f"data    -> {out}")


def plot_grid(summary_csv, out_dir):
    """3x2: nano only / TabPFN only / both; columns = bias^2 and variance vs n."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(summary_csv)
    colors = {"nano": "#1f77b4", "tabpfn": "#d62728"}
    names = {"nano": "nanoTabPFN", "tabpfn": "TabPFN v2.5"}
    style = {"tab": ("-", "o", "global"), "ltab": ("--", "s", "localized")}

    def draw(ax, models, metric):
        for model in models:
            for method, (ls, mk, ml) in style.items():
                sub = df[(df.model == model) & (df.method == method)].sort_values("n")
                if sub.empty:
                    continue
                lab = f"{names[model]} ({ml})" if len(models) > 1 else ml
                ax.plot(sub["n"], sub[metric], ls, marker=mk, color=colors[model], label=lab)
        ax.set_xscale("log"); ax.set_ylim(bottom=0); ax.grid(alpha=0.3)
        ax.set_xlabel("n (training set size)"); ax.legend(fontsize=8)

    rows = [(["nano"], "nanoTabPFN only"), (["tabpfn"], "TabPFN v2.5 only"),
            (["nano", "tabpfn"], "both together")]
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    for r, (models, title) in enumerate(rows):
        draw(axes[r, 0], models, "bias2"); draw(axes[r, 1], models, "variance")
        axes[r, 0].set_ylabel("average squared bias"); axes[r, 1].set_ylabel("average variance")
        axes[r, 0].set_title(f"{title} — squared bias"); axes[r, 1].set_title(f"{title} — variance")
    fig.suptitle("Nagler (2023) Fig. 1 — bias & variance vs n", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(out_dir, "grid-3x2-bias-variance.png")
    fig.savefig(out, dpi=140); print(f"grid    -> {out}")


def plot_aggregate(out_dir, n_ref=4000):
    """Cross-function comparison from fn-<dgp>-summary.csv: freq sweep + diverse bars."""
    import glob
    import re
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    freqs = {"freq0.5": 0.5, "freq1": 1.0, "freq2": 2.0, "freq3": 3.0, "freq4": 4.0}
    diverse = ["linear", "highfreq", "radial", "interaction", "xor"]
    rows = []
    for path in glob.glob(os.path.join(out_dir, "fn-*-summary.csv")):
        d = pd.read_csv(path)
        d["dgp"] = re.match(r"fn-(.+)-summary\.csv", os.path.basename(path)).group(1)
        rows.append(d)
    if not rows:
        print("(aggregate skipped: no fn-*-summary.csv)")
        return
    df = pd.concat(rows, ignore_index=True)

    def b(dgp, model, method):
        sub = df[(df.dgp == dgp) & (df.model == model) & (df.method == method) & (df.n == n_ref)]
        return float(sub["bias2"].iloc[0]) if len(sub) else None

    have_f = sorted([f for f in freqs if f in set(df.dgp)], key=lambda f: freqs[f])
    if len(have_f) >= 2:
        ws = [freqs[f] for f in have_f]
        series = {("nano", "tab"): ("nanoTabPFN (global)", "#1f77b4", "-", "o"),
                  ("nano", "ltab"): ("nanoTabPFN (localized)", "#1f77b4", "--", "s"),
                  ("tabpfn", "tab"): ("TabPFN (global)", "#d62728", "-", "o"),
                  ("tabpfn", "ltab"): ("TabPFN (localized)", "#d62728", "--", "s")}
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
        for (m, meth), (lab, c, ls, mk) in series.items():
            ax[0].plot(ws, [b(f, m, meth) for f in have_f], ls, marker=mk, color=c, label=lab)
        ax[0].set_xlabel("sine frequency w"); ax[0].set_ylabel("average squared bias")
        ax[0].set_title(f"bias^2 at n={n_ref} vs target frequency")
        ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
        for m, c in [("nano", "#1f77b4"), ("tabpfn", "#d62728")]:
            ax[1].plot(ws, [b(f, m, "tab") - b(f, m, "ltab") for f in have_f], "-o", color=c, label=m)
        ax[1].axhline(0, color="k", lw=0.8); ax[1].set_xlabel("sine frequency w")
        ax[1].set_title("localization gain (global - localized bias^2)")
        ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
        fig.suptitle("Frequency sweep — bias plateau and localization gain vs frequency")
        fig.tight_layout()
        out = os.path.join(out_dir, "agg-frequency-sweep.png")
        fig.savefig(out, dpi=140); print(f"agg-freq -> {out}")

    have_d = [d for d in diverse if d in set(df.dgp)]
    if have_d:
        x = np.arange(len(have_d)); w = 0.2
        bars = {("nano", "tab"): ("nano global", "#1f77b4", -1.5),
                ("nano", "ltab"): ("nano localized", "#7fb3d5", -0.5),
                ("tabpfn", "tab"): ("TabPFN global", "#d62728", 0.5),
                ("tabpfn", "ltab"): ("TabPFN localized", "#e8867f", 1.5)}
        fig, ax = plt.subplots(figsize=(1.6 * len(have_d) + 3, 4.6))
        for (m, meth), (lab, c, off) in bars.items():
            ax.bar(x + off * w, [b(d, m, meth) or 0 for d in have_d], w, label=lab, color=c)
        ax.set_xticks(x); ax.set_xticklabels(have_d); ax.set_ylabel("average squared bias")
        ax.set_title(f"Diverse target structures — bias^2 at n={n_ref}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        out = os.path.join(out_dir, "agg-diverse-structures.png")
        fig.savefig(out, dpi=140); print(f"agg-div  -> {out}")


if __name__ == "__main__":
    main()
