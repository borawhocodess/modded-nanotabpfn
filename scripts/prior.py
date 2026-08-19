import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

PLOTS = Path(__file__).resolve().parent.parent / "workdir" / "plots" / "prior"


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


class Prior:
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


def plot_dataset(x, y, sep, path, cols=4):
    """Scatter grid of the first `cols` features, coloured by class; test rows drawn hollow."""
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
    """One marginal per node of the SCM, in topological order, labelled with its activation.

    Node 0 is the raw gaussian input and never gets an activation. Every later node is a
    standardised nonlinearity of its parents, so the shapes drift away from gaussian as depth
    grows -- `abs`/`square` go one-sided, `sin` goes multimodal, `tanh` saturates. The target
    node is highlighted and its class cuts drawn, since that column is what `y` is binned from.
    """
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
    """The sampled causal graph, laid out left-to-right by depth.

    Node 0 is the root gaussian; every other node is a function of its parents, so depth is
    1 + max(depth of parents). Marker area scales with out-degree, which is where `redirection`
    shows up: redirection resamples a candidate parent as *its* parent, so attachment is
    preferential and a few nodes become hubs instead of parenthood spreading evenly.
    """
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


def plot_diversity(draws, path, ncols=4):
    """Many independent draws from the prior, one panel each, coloured by class.

    Every panel re-samples the graph, the weights, the activations, the target node and the class
    count, so the spread across panels is the thing the prior actually is. The single-dataset
    figures show one sample; this shows the distribution those samples come from.

    The feature pair is drawn at random per panel rather than fixed at (x0, x1): `gnr()` builds
    node 1's parent with `randint(1)`, which can only return 0, so node 1 is *always* a direct
    child of the root and a fixed (x0, x1) pair would show the most degenerate pair every time.
    """
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
    """The sampled weight matrix: row i holds the coefficients node i puts on its parents.

    Strictly lower-triangular by construction -- `gnr()` only ever picks a parent with a lower
    index than the child, which is what makes the graph acyclic and lets `propagate()` fill the
    columns in one forward pass. Each row has at most `num_parent_attempts` non-zeros, so the
    matrix is very sparse; the column sums are what the DAG figure draws as out-degree.
    """
    n = w.shape[0]
    vmax = np.abs(w).max()
    fig, ax = plt.subplots(figsize=(0.42 * n + 3.5, 0.42 * n + 2))
    im = ax.imshow(w, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n))
    ax.set_xticklabels(range(n), fontsize=6)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{i} · {'input' if i == 0 else names[acts[i]]}" for i in range(n)], fontsize=6)
    for i, lbl in enumerate(ax.get_yticklabels()):
        if i == target:
            lbl.set_color("tab:red")
            lbl.set_fontweight("bold")
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="w", lw=0.5)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("parent node (column)", fontsize=8)
    ax.set_ylabel("child node (row) · its activation", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03).ax.tick_params(labelsize=6)
    density = (w != 0).sum() / (n * (n - 1) / 2)
    fig.suptitle(
        f"prior weights — {n}×{n}, strictly lower-triangular, "
        f"{(w != 0).sum()} non-zeros ({density:.0%} of the triangle), red = target row",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--features", type=int, default=3)
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--test-rows", type=int, default=20)
    parser.add_argument("--draws", type=int, default=16)
    parser.add_argument("--out", default=str(PLOTS))
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = PriorConfig(
        min_num_cols=args.features,
        max_num_cols=args.features,
        min_num_rows=args.rows,
        max_num_rows=args.rows,
        min_num_test_rows=args.test_rows,
        max_num_test_rows=args.test_rows,
    )
    prior = Prior(config, device="cpu")
    x_train, y_train, x_test, y_test = prior.batch(1)
    x = torch.cat([x_train, x_test], dim=1)[0].numpy()
    y = torch.cat([y_train, y_test], dim=1)[0].numpy().astype(int)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plot_dataset(x, y, prior.sep, out / "prior_dataset.png", cols=x.shape[1])
    print(f"wrote {out / 'prior_dataset.png'}")

    plot_latents(
        prior.z.numpy(),
        prior.acts,
        prior.activation_names,
        prior.target_node,
        prior.cuts.numpy(),
        out / "prior_latents.png",
    )
    print(f"wrote {out / 'prior_latents.png'}")

    plot_dag(prior.parents, prior.acts, prior.activation_names, prior.target_node, out / "prior_dag.png")
    print(f"wrote {out / 'prior_dag.png'}")

    plot_weights(prior.w.numpy(), prior.acts, prior.activation_names, prior.target_node, out / "prior_weights.png")
    print(f"wrote {out / 'prior_weights.png'}")

    draws = []
    for _ in range(args.draws):
        x_train, y_train, x_test, y_test = prior.batch(1)
        draws.append((torch.cat([x_train, x_test], dim=1)[0].numpy(), torch.cat([y_train, y_test], dim=1)[0].numpy()))
    plot_diversity(draws, out / "prior_diversity.png")
    print(f"wrote {out / 'prior_diversity.png'}")


if __name__ == "__main__":
    main()
