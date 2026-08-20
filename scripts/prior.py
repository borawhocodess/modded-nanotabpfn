import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

PLOTS = Path(__file__).resolve().parent.parent / "workdir" / "plots" / "prior"

SETTINGS = {
    "f3": {"features": 3, "rows": 100, "test_rows": 20},
    "f4": {"features": 4, "rows": 1000, "test_rows": 128},
    "f9": {"features": 9, "rows": 1000, "test_rows": 128},
    "f11": {"features": 11, "rows": 1000, "test_rows": 128},
    "f14": {"features": 14, "rows": 1000, "test_rows": 128},
    "f20": {"features": 20, "rows": 1000, "test_rows": 128},
}


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


class ModdedNanoPrior:
    activations = (lambda z: z, torch.tanh, torch.sin, torch.abs, torch.square, F.softplus)
    activation_names = ("identity", "tanh", "sin", "abs", "square", "softplus")

    def __init__(self, config, device):
        self.config = config
        self.device = device
        assert self.config.max_num_test_rows < self.config.min_num_rows

    def hyperparameters(self):
        c = self.config
        self.num_cols = int(np.random.randint(c.min_num_cols, c.max_num_cols + 1))
        self.num_rows = int(np.random.randint(c.min_num_rows, c.max_num_rows + 1))
        self.num_test_rows = int(np.random.randint(c.min_num_test_rows, c.max_num_test_rows + 1))
        self.sep = self.num_rows - self.num_test_rows
        self.nodes = self.num_cols + 1
        self.redirection = np.random.uniform(c.min_redirection, c.max_redirection)
        self.num_classes = int(np.random.randint(c.min_num_classes, c.max_num_classes + 1))
        self.num_parent_attempts = int(np.random.randint(c.min_num_parent_attempts, c.max_num_parent_attempts + 1))

    def gnr(self):
        parents = [[] for _ in range(self.nodes)]
        self.attempts = self.redirects = 0
        for child in range(1, self.nodes):
            chosen = set()
            for _ in range(self.num_parent_attempts):
                candidate = int(np.random.randint(child))
                self.attempts += 1
                if np.random.rand() < self.redirection and parents[candidate]:
                    candidate = int(np.random.choice(parents[candidate]))
                    self.redirects += 1
                chosen.add(candidate)
            parents[child] = sorted(chosen)
        return parents

    def propagate(self):
        parents = self.parents = self.gnr()
        w = np.zeros((self.nodes, self.nodes), dtype=np.float32)
        for i in range(1, self.nodes):
            w[i, parents[i]] = np.random.randn(len(parents[i]))
        w = self.w = torch.from_numpy(w).to(self.device)
        acts = np.random.randint(len(self.activations), size=self.nodes)
        z = torch.randn(self.num_rows, self.nodes, device=self.device)
        for i in range(1, self.nodes):
            zi = self.activations[acts[i]](z @ w[i]) + 0.1 * z[:, i]
            std, mean = torch.std_mean(zi)
            z[:, i] = (zi - mean) / (std + 1e-6)
        self.z, self.acts = z, acts
        return z

    def target(self, z):
        target = self.target_node = int(np.random.randint(1, self.nodes))
        zt = z[:, target].contiguous()
        cuts = torch.linspace(0, 1, self.num_classes + 1, device=self.device)[1:-1]
        self.zt, self.cuts = zt, zt.quantile(cuts)
        y = torch.bucketize(zt, self.cuts)
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


def plot_hyperparameters(config, device, path, samples=5000, ncols=4):
    fields = (
        ("num_cols", "sampled", "min_num_cols", "max_num_cols"),
        ("num_rows", "sampled", "min_num_rows", "max_num_rows"),
        ("num_test_rows", "sampled", "min_num_test_rows", "max_num_test_rows"),
        ("sep", "derived", None, None),
        ("nodes", "derived", None, None),
        ("redirection", "sampled", "min_redirection", "max_redirection"),
        ("num_classes", "sampled", "min_num_classes", "max_num_classes"),
        ("num_parent_attempts", "sampled", "min_num_parent_attempts", "max_num_parent_attempts"),
    )
    prior = ModdedNanoPrior(config, device)
    drawn = {name: [] for name, *_ in fields}
    for _ in range(samples):
        prior.hyperparameters()
        for name, *_ in fields:
            drawn[name].append(getattr(prior, name))

    colours = {"sampled": "tab:blue", "derived": "tab:orange"}
    rows = int(np.ceil(len(fields) / ncols))
    fig, axes = plt.subplots(rows, ncols, figsize=(3.1 * ncols, 2.6 * rows), squeeze=False)
    for k, (name, kind, lo_attr, hi_attr) in enumerate(fields):
        ax = axes[k // ncols][k % ncols]
        v = np.asarray(drawn[name])
        uniq = np.unique(v)
        if len(uniq) == 1:
            bins = np.array([uniq[0] - 0.5, uniq[0] + 0.5])
        elif v.dtype.kind in "iu" and len(uniq) <= 30:
            bins = np.arange(uniq.min() - 0.5, uniq.max() + 1.5, 1.0)
        else:
            bins = 40
        ax.hist(v, bins=bins, color=colours[kind], edgecolor="white", lw=0.4)
        rng = "" if lo_attr is None else f"  [{getattr(config, lo_attr)}, {getattr(config, hi_attr)}]"
        fixed = "  · fixed" if len(uniq) == 1 else ""
        ax.set_title(f"{name}{rng}{fixed}", fontsize=8)
        ax.set_ylabel("draws", fontsize=7)
        ax.tick_params(labelsize=6)
        if len(uniq) == 1:
            ax.set_xlim(uniq[0] - 1, uniq[0] + 1)
            ax.set_xticks([uniq[0]])
    for k in range(len(fields), rows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colours.values()]
    fig.legend(handles, list(colours), loc="lower center", ncol=2, fontsize=8, frameon=False)
    varying = sum(len(np.unique(drawn[n])) > 1 for n, *_ in fields)
    fig.suptitle(
        f"prior hyperparameters — {samples} draws of hyperparameters(), "
        f"{varying}/{len(fields)} non-degenerate under this config",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_growth(config, device, path, ncols=5):
    prior = ModdedNanoPrior(config, device)
    prior.hyperparameters()
    parents = prior.gnr()
    n = len(parents)

    depth = [0] * n
    for i in range(1, n):
        depth[i] = 1 + max(depth[p] for p in parents[i])
    levels, pos = {}, {}
    for i in range(n):
        d = depth[i]
        k = levels.get(d, 0)
        levels[d] = k + 1
        pos[i] = [d, k]
    for i in range(n):
        pos[i][1] -= (levels[depth[i]] - 1) / 2
    span = max(levels.values())

    marker, label = 40 + 240 / n, 5 + 10 / n
    rows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(rows, ncols, figsize=(2.5 * ncols, 1.8 * rows), squeeze=False)
    for k in range(rows * ncols):
        ax = axes[k // ncols][k % ncols]
        ax.set_axis_off()
        if k >= n:
            continue
        shown = k + 1
        edges = 0
        for i in range(1, shown):
            for p in parents[i]:
                edges += 1
                new = i == k
                ax.annotate(
                    "",
                    xy=pos[i],
                    xytext=pos[p],
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": "tab:green" if new else "0.75",
                        "lw": 1.1 if new else 0.7,
                        "shrinkA": 5,
                        "shrinkB": 6,
                    },
                )
        for i in range(shown):
            colour = "tab:green" if i == k else ("0.4" if i == 0 else "tab:blue")
            ax.scatter(*pos[i], s=marker, color=colour, zorder=3, edgecolors="w", lw=0.8)
            ax.text(*pos[i], str(i), color="w", ha="center", va="center", fontsize=label, zorder=4)
        ax.set_title(f"{shown} node{'' if shown == 1 else 's'} · {edges} edge{'' if edges == 1 else 's'}", fontsize=8)
        ax.set_xlim(-0.6, max(depth) + 0.6)
        ax.set_ylim(-span / 2 - 0.6, span / 2 + 0.6)
    fig.suptitle(
        f"gnr() building one graph node by node — grey = root, green = the node just added, "
        f"redirection {config.min_redirection}, {config.min_num_parent_attempts} parent attempts",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_dataset(x, y, sep, path, cols=4):
    classes = np.unique(y)
    cmap = plt.get_cmap("tab10")
    train = np.arange(len(y)) < sep
    data = x[:, :cols]
    names = [f"x{j}" for j in range(cols)]
    n = data.shape[1]
    fig, axes = plt.subplots(n, n, figsize=(2.1 * n, 2.1 * n))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            bins = np.histogram_bin_edges(data[:, i], bins=30)
            for k, cl in enumerate(classes):
                m = y == cl
                if i == j:
                    ax.hist(data[m, i], bins=bins, color=cmap(k), alpha=0.55, lw=0)
                else:
                    ax.scatter(data[m & train, j], data[m & train, i], s=5, color=cmap(k), lw=0, alpha=0.8)
                    ax.scatter(
                        data[m & ~train, j], data[m & ~train, i], s=18, facecolors="none", edgecolors=cmap(k), lw=0.7
                    )
            ax.set_xticks([])
            ax.set_yticks([])
            if i == n - 1:
                ax.set_xlabel(names[j], fontsize=8)
            if j == 0:
                ax.set_ylabel(names[i] if i != 0 else f"{names[0]} (count)", fontsize=8)
    handles = [plt.Line2D([], [], ls="", marker="o", ms=5, color=cmap(k)) for k in range(len(classes))]
    handles += [plt.Line2D([], [], ls="", marker="o", ms=5, mfc="none", mec="k")]
    labels = [f"class {int(cl)}" for cl in classes] + ["test row"]
    fig.legend(handles, labels, loc="lower center", fontsize=8, frameon=False, ncol=len(labels))
    fig.suptitle(
        f"one prior dataset — {x.shape[0]} rows ({sep} train / {x.shape[0] - sep} test), "
        f"{x.shape[1]} features, {len(classes)} classes",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_latents(z, acts, names, target, cuts, path, ncols=5):
    n = z.shape[1]
    rows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(rows, ncols, figsize=(2.4 * ncols, 2.1 * rows), squeeze=False)
    for k in range(rows * ncols):
        ax = axes[k // ncols][k % ncols]
        if k >= n:
            ax.axis("off")
            continue
        is_target = k == target
        ax.hist(z[:, k], bins=40, color="tab:red" if is_target else "tab:blue", alpha=0.75, lw=0)
        if is_target:
            for cut in cuts:
                ax.axvline(cut, color="k", ls="--", lw=0.8, alpha=0.6)
        label = "input" if k == 0 else names[acts[k]]
        ax.set_title(f"node {k} — {label}" + (" (target)" if is_target else ""), fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"prior latents — {n} nodes, {z.shape[0]} rows, each standardised to mean 0 / std 1", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_dag(parents, acts, names, target, path):
    n = len(parents)
    depth = [0] * n
    for i in range(1, n):
        depth[i] = 1 + max(depth[p] for p in parents[i])
    children = [0] * n
    for i in range(1, n):
        for p in parents[i]:
            children[p] += 1

    levels = {}
    pos = {}
    for i in range(n):
        d = depth[i]
        k = levels.get(d, 0)
        levels[d] = k + 1
        pos[i] = [d, k]
    for i in range(n):
        pos[i][1] -= (levels[depth[i]] - 1) / 2

    fig, ax = plt.subplots(figsize=(1.9 * (max(depth) + 1) + 2, 0.9 * max(levels.values()) + 2))
    for i in range(1, n):
        for p in parents[i]:
            ax.annotate(
                "",
                xy=pos[i],
                xytext=pos[p],
                arrowprops={"arrowstyle": "-|>", "color": "0.6", "lw": 0.8, "shrinkA": 9, "shrinkB": 11},
            )
    for i in range(n):
        c = "tab:red" if i == target else ("0.4" if i == 0 else "tab:blue")
        ax.scatter(*pos[i], s=120 + 90 * children[i], color=c, zorder=3, edgecolors="w", lw=1.2)
        ax.text(*pos[i], str(i), color="w", ha="center", va="center", fontsize=7, zorder=4)
        label = "input" if i == 0 else names[acts[i]]
        ax.annotate(
            label,
            pos[i],
            textcoords="offset points",
            xytext=(0, -16),
            ha="center",
            va="top",
            fontsize=6.5,
            color="0.3",
        )
    for d in sorted(levels):
        ax.text(
            d,
            1.0,
            f"depth {d}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            color="0.5",
        )
    span = max(levels.values())
    ax.set_xlim(-0.5, max(depth) + 0.5)
    ax.set_ylim(-span / 2 - 0.5, span / 2 + 0.5)
    ax.set_axis_off()
    fig.suptitle(
        f"prior DAG — {n} nodes, node 0 = root gaussian, red = target, marker area ∝ out-degree",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_split(config, device, path, per_class=60):
    counts = {}
    for classes in range(config.min_num_classes, config.max_num_classes + 1):
        cfg = replace(config, min_num_classes=classes, max_num_classes=classes)
        prior = ModdedNanoPrior(cfg, device)
        train, test = [], []
        for _ in range(per_class):
            _, y_train, _, y_test = prior.batch(1)
            train.append(np.bincount(y_train[0].numpy().astype(int), minlength=classes))
            test.append(np.bincount(y_test[0].numpy().astype(int), minlength=classes))
        counts[classes] = (np.array(train), np.array(test))

    n_rows, n_test = config.max_num_rows, config.max_num_test_rows
    n_train = n_rows - n_test
    classes = sorted(counts)
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes.ravel()

    show = classes[-1]
    train, test = counts[show]
    width = 0.4
    ax[0].bar(np.arange(show) - width / 2, train[0], width, color="tab:blue", label="train")
    ax[0].bar(np.arange(show) + width / 2, test[0], width, color="tab:orange", label="test")
    ax[0].axhline(n_train / show, color="tab:blue", ls="--", lw=1)
    ax[0].axhline(n_test / show, color="tab:orange", ls="--", lw=1)
    ax[0].set_yscale("log")
    ax[0].set_xticks(range(show))
    ax[0].set_xlabel("class", fontsize=8)
    ax[0].set_ylabel("rows", fontsize=8)
    ax[0].set_title(f"one draw with {show} classes — dashed = perfectly even", fontsize=9)
    ax[0].legend(fontsize=7, frameon=False)

    ratios = []
    for k, c in enumerate(classes):
        _, test = counts[c]
        expected = n_test / c
        sd = np.sqrt(n_test * (1 / c) * (1 - 1 / c) * (n_rows - n_test) / (n_rows - 1))
        v = np.sort((test.ravel() - expected) / sd)
        ratios.append(float(v.std()))
        ax[1].step(v, np.arange(1, len(v) + 1) / len(v), color=cmap(k), lw=1.1, label=f"{c} classes")
    grid = np.linspace(-4, 4, 200)
    ax[1].plot(
        grid,
        0.5 * (1 + torch.erf(torch.from_numpy(grid) / np.sqrt(2))).numpy(),
        color="k",
        ls="--",
        lw=1.4,
        label="N(0,1)",
    )
    ax[1].set_xlabel("(test count − expected) / hypergeometric sd", fontsize=8)
    ax[1].set_ylabel("empirical CDF", fontsize=8)
    ax[1].set_title(f"test split is a plain random subsample — sd ratio {np.mean(ratios):.2f}", fontsize=9)
    ax[1].legend(fontsize=6, frameon=False, ncol=2)

    ratios_test = [counts[c][1].max(axis=1) / np.maximum(counts[c][1].min(axis=1), 1) for c in classes]
    ratios_train = [counts[c][0].max(axis=1) / np.maximum(counts[c][0].min(axis=1), 1) for c in classes]
    pos = np.arange(len(classes))
    b0 = ax[2].boxplot(ratios_train, positions=pos - 0.18, widths=0.3, patch_artist=True, showfliers=False)
    b1 = ax[2].boxplot(ratios_test, positions=pos + 0.18, widths=0.3, patch_artist=True, showfliers=False)
    for box, colour in ((b0, "tab:blue"), (b1, "tab:orange")):
        for patch in box["boxes"]:
            patch.set_facecolor(colour)
        for median in box["medians"]:
            median.set_color("k")
    ax[2].set_xticks(pos)
    ax[2].set_xticklabels(classes)
    ax[2].set_xlabel("num_classes", fontsize=8)
    ax[2].set_ylabel("largest class / smallest class", fontsize=8)
    ax[2].set_title("imbalance ratio — blue train, orange test", fontsize=9)

    missing = [(counts[c][1].min(axis=1) == 0).mean() for c in classes]
    smallest = [counts[c][1].min(axis=1).mean() for c in classes]
    ax[3].bar(pos, smallest, color="tab:orange", width=0.6)
    ax[3].plot(pos, [n_test / c for c in classes], "k--", lw=1, label="expected if even")
    for k, c in enumerate(classes):
        ax[3].text(
            k, smallest[k], f"{missing[k]:.0%} empty" if missing[k] else "", ha="center", va="bottom", fontsize=7
        )
    ax[3].set_xticks(pos)
    ax[3].set_xticklabels(classes)
    ax[3].set_xlabel("num_classes", fontsize=8)
    ax[3].set_ylabel("rows in the rarest test class", fontsize=8)
    ax[3].set_title(f"rarest class in the {n_test} test rows", fontsize=9)
    ax[3].legend(fontsize=7, frameon=False)

    for a in ax:
        a.tick_params(labelsize=7)
    fig.suptitle(
        f"train / test split — {n_train} train + {n_test} test rows, "
        f"{per_class} datasets per class count, balancing is over all {n_rows} rows",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_diversity(draws, path, ncols=4):
    cmap = plt.get_cmap("tab10")
    rows = int(np.ceil(len(draws) / ncols))
    fig, axes = plt.subplots(rows, ncols, figsize=(2.2 * ncols, 2.3 * rows), squeeze=False)
    for k in range(rows * ncols):
        ax = axes[k // ncols][k % ncols]
        if k >= len(draws):
            ax.axis("off")
            continue
        x, y = draws[k]
        a, b = np.random.choice(x.shape[1], size=2, replace=False)
        for c, cl in enumerate(np.unique(y)):
            m = y == cl
            ax.scatter(x[m, a], x[m, b], s=4, color=cmap(c), lw=0, alpha=0.8)
        ax.set_title(f"x{a} vs x{b} — {len(np.unique(y))} classes", fontsize=7, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{len(draws)} independent draws from the prior — random feature pair, coloured by class", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_weights(w, acts, names, target, path):
    n = w.shape[0]
    vmax = np.abs(w).max()
    fig, ax = plt.subplots(figsize=(0.42 * n + 4.0, 0.42 * n + 3.0))
    cmap = plt.get_cmap("RdBu_r").with_extremes(bad="black")
    im = ax.imshow(np.ma.masked_where(w == 0, w), cmap=cmap, vmin=-vmax, vmax=vmax)
    labels = [f"{i} · {'input' if i == 0 else names[acts[i]]}" for i in range(n)]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=6, rotation=90)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=6)
    for group in (ax.get_xticklabels(), ax.get_yticklabels()):
        for i, lbl in enumerate(group):
            if i == target:
                lbl.set_color("tab:red")
                lbl.set_fontweight("bold")
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="0.5", lw=0.5)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("parent node (column) · its activation", fontsize=8)
    ax.set_ylabel("child node (row) · its activation", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03).ax.tick_params(labelsize=6)
    density = (w != 0).sum() / (n * (n - 1) / 2)
    fig.suptitle(
        f"prior weights — {n}×{n}, strictly lower-triangular, "
        f"{(w != 0).sum()} edges ({density:.0%} of the triangle), black = no edge, red = target",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(name, features, rows, test_rows, seed, draws, out):
    np.random.seed(seed)
    torch.manual_seed(seed)

    config = PriorConfig(
        min_num_cols=features,
        max_num_cols=features,
        min_num_rows=rows,
        max_num_rows=rows,
        min_num_test_rows=test_rows,
        max_num_test_rows=test_rows,
    )
    prior = ModdedNanoPrior(config, device="cpu")
    x_train, y_train, x_test, y_test = prior.batch(1)
    x = torch.cat([x_train, x_test], dim=1)[0].numpy()
    y = torch.cat([y_train, y_test], dim=1)[0].numpy().astype(int)

    out.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] {features} features, {rows} rows, {test_rows} test rows, seed {seed} -> {out}")
    plot_dataset(x, y, prior.sep, out / "prior_dataset.png", cols=x.shape[1])
    plot_latents(
        prior.z.numpy(),
        prior.acts,
        prior.activation_names,
        prior.target_node,
        prior.cuts.numpy(),
        out / "prior_latents.png",
    )
    plot_dag(prior.parents, prior.acts, prior.activation_names, prior.target_node, out / "prior_dag.png")
    plot_weights(prior.w.numpy(), prior.acts, prior.activation_names, prior.target_node, out / "prior_weights.png")

    batches = []
    for _ in range(draws):
        x_train, y_train, x_test, y_test = prior.batch(1)
        batches.append((torch.cat([x_train, x_test], dim=1)[0].numpy(), torch.cat([y_train, y_test], dim=1)[0].numpy()))
    plot_diversity(batches, out / "prior_diversity.png")
    plot_hyperparameters(config, "cpu", out / "prior_hyperparameters.png")
    plot_growth(config, "cpu", out / "prior_growth.png")
    plot_split(config, "cpu", out / "prior_split.png")
    for path in sorted(out.glob("prior_*.png")):
        print(f"  wrote {path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", nargs="+", default=list(SETTINGS), choices=list(SETTINGS))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--draws", type=int, default=16)
    parser.add_argument("--out", default=None, help="default: workdir/plots/prior/<setting>")
    args = parser.parse_args()

    assert args.out is None or len(args.settings) == 1, "--out only makes sense for a single setting"
    for name in args.settings:
        out = Path(args.out) if args.out else PLOTS / name
        run(name, seed=args.seed, draws=args.draws, out=out, **SETTINGS[name])


if __name__ == "__main__":
    main()
