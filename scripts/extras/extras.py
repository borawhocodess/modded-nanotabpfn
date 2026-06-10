"""Plot the diffeval retime sweep: 9-rung ablation ladder × 5 reps, evaluated on
the TabPFN-paper classification suite. Parses the per-epoch logs under logs/ and
writes the figures under plots/.

Each log line looks like:
  e:49/4000 μ_l:0.94 g:2.15 g_max:5.68 clip:0.75 d_mu:4.68 d_ad:0.29
  d0_mu:35.97 d0_ad:7.45 e_t:1.62s μ_e_t:1.73s t_t:84.77s avg_roc_auc:0.82003

Runs stop the moment they cross the jackpot (RF-on-paper baseline), so each curve
ends at its own crossing and reps differ slightly in length.
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

# ladder order (baseline -> full stack); colour reads as a gradient down the ladder
ORDER = ["baseline", "muon", "carter", "batchedmuon", "residualdecay",
         "rmsthink", "lawa1wd", "featuregroup", "autohuman"]

NUM = r"([0-9]+(?:\.[0-9]+)?|nan)"
FIELDS = {
    "epoch":   re.compile(r"\be:(\d+)/"),
    "loss":    re.compile(r"μ_l:" + NUM),
    "g":       re.compile(r"\bg:" + NUM),
    "g_max":   re.compile(r"\bg_max:" + NUM),
    "clip":    re.compile(r"\bclip:" + NUM),
    "d_mu":    re.compile(r"\bd_mu:" + NUM),
    "d_ad":    re.compile(r"\bd_ad:" + NUM),
    "d0_mu":   re.compile(r"\bd0_mu:" + NUM),
    "d0_ad":   re.compile(r"\bd0_ad:" + NUM),
    "mu_e_t":  re.compile(r"μ_e_t:" + NUM + "s"),
    "t_t":     re.compile(r"\bt_t:" + NUM + "s"),
    "auc":     re.compile(r"avg_roc_auc:" + NUM),
}


def _f(m):
    if m is None:
        return np.nan
    v = m.group(1)
    return np.nan if v == "nan" else float(v)


def parse_rep(path):
    """Return dict of per-epoch arrays for one run (only lines with an AUC)."""
    cols = {k: [] for k in FIELDS}
    for line in path.read_text(errors="ignore").splitlines():
        if "avg_roc_auc:" not in line or "e:" not in line:
            continue
        if FIELDS["auc"].search(line) is None:
            continue
        for k, rx in FIELDS.items():
            cols[k].append(_f(rx.search(line)))
    out = {k: np.asarray(v, float) for k, v in cols.items()}
    out["t_min"] = out["t_t"] / 60.0
    return out


def load(logs):
    """variant -> list of rep dicts, in ladder order.

    Each top-level subdir of `logs` is one rung; its name may carry a leading date
    (records/ uses '20260131baseline', diffeval/ uses 'baseline'), so the variant
    is the name with any leading digits/dashes stripped. Logs may sit directly in
    the rung dir (diffeval) or one level down per run (archive, records)."""
    data = {v: [] for v in ORDER}
    for sub in sorted(p for p in logs.iterdir() if p.is_dir()):
        v = re.sub(r"^[\d-]+", "", sub.name)
        if v not in data:
            continue
        reps = [parse_rep(p) for p in sorted(sub.rglob("*-log.txt"))]
        data[v] = [r for r in reps if r["epoch"].size > 1]
    return data


def median_band(reps, xkey, ykey, xstop, n=240):
    """Interpolate each rep's y(x) onto a shared grid [0, xstop]; return grid and
    the (25, 50, 75) percentiles, masking each rep beyond its own last sample so a
    short rep doesn't get clamp-extended into the band."""
    grid = np.linspace(0, xstop, n)
    stack = []
    for r in reps:
        x, y = r[xkey], r[ykey]
        yi = np.interp(grid, x, y)
        yi[grid > x.max()] = np.nan
        stack.append(yi)
    stack = np.vstack(stack)
    lo, mid, hi = np.nanpercentile(stack, [25, 50, 75], axis=0)
    return grid, lo, mid, hi


def colors():
    cmap = plt.get_cmap("viridis")
    return {v: cmap(i / (len(ORDER) - 1)) for i, v in enumerate(ORDER)}


def ttj(reps, key):
    """time/epochs to jackpot per rep = value at the final (crossing) epoch."""
    return np.array([r[key][-1] for r in reps])


# ----------------------------------------------------------------------------- plots

def out(name):
    """output path, with the optional --tag suffixed before the extension."""
    stem, _, ext = name.rpartition(".")
    return PLOTS / (f"{stem}_{TAG}.{ext}" if TAG else name)


def plot_curves(data, c, xkey, xlabel, fname, title, logx=True):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for v in ORDER:
        reps = data[v]
        xstop = np.median(ttj(reps, xkey if xkey != "t_min" else "t_min"))
        grid, lo, mid, hi = median_band(reps, xkey, "auc", xstop)
        ax.plot(grid, mid, color=c[v], lw=2, label=v, zorder=3)
        ax.fill_between(grid, lo, hi, color=c[v], alpha=0.15, lw=0, zorder=2)
        # star where the median curve meets the jackpot line
        ax.scatter([grid[-1]], [JACKPOT], color=c[v], marker="*", s=120,
                   edgecolor="k", lw=0.4, zorder=4)
    ax.axhline(JACKPOT, color="k", ls="--", lw=1, alpha=0.7)
    ax.text(ax.get_xlim()[0], JACKPOT, f" jackpot (RF={JACKPOT:.3f})", va="bottom",
            ha="left", fontsize=8, alpha=0.8)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("avg_roc_auc")
    ax.set_ylim(0.45, 0.90)
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out(fname), dpi=150)
    plt.close(fig)


def plot_ttj_bar(data, c):
    fig, ax = plt.subplots(figsize=(8, 5))
    base = np.median(ttj(data["baseline"], "t_min"))
    counts = {len(data[v]) for v in ORDER}
    rep_txt = f"{counts.pop()} reps" if len(counts) == 1 else "individual reps"
    y = np.arange(len(ORDER))[::-1]  # baseline on top
    xmin = min(ttj(data[v], "t_min").min() for v in ORDER)
    ax.set_xlim(xmin * 0.6, base * 3)  # set before bars so the log base draws from here
    for yi, v in zip(y, ORDER):
        pts = ttj(data[v], "t_min")
        med = np.median(pts)
        ax.barh(yi, med, color=c[v], alpha=0.9, edgecolor="k", lw=0.6, zorder=2)
        ax.scatter(pts, np.full_like(pts, yi), color="k", s=6, zorder=3)
        # label past the rightmost rep-dot so it never sits on the points
        ax.text(pts.max() * 1.18, yi, f"{med:.1f} min  ({base/med:.1f}×)",
                va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(ORDER)
    ax.set_xscale("log")
    ax.set_xlabel(f"time to jackpot (min, log) — median bar, dots = {rep_txt}")
    ax.set_title(f"Time to cross the RF baseline ({JACKPOT:.3f} roc_auc)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out("ttj_bar.png"), dpi=150)
    plt.close(fig)


def plot_dynamics(data, c):
    """grad-norm and clip-rate trajectories (baseline has no Muon -> skip it)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for v in ORDER:
        if v == "baseline":
            continue
        reps = data[v]
        xstop = np.median(ttj(reps, "epoch"))
        for ax, key in zip(axes, ["g", "clip"]):
            grid, _, mid, _ = median_band(reps, "epoch", key, xstop)
            ax.plot(grid, mid, color=c[v], lw=1.8, label=v)
    axes[0].set_ylabel("mean grad norm")
    axes[1].set_ylabel("clip rate (frac steps clipped @1.0)")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.set_xscale("log")
        ax.grid(alpha=0.25)
    axes[0].set_title("Gradient magnitude over training")
    axes[1].set_title("Clipping rate over training")
    axes[1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out("dynamics_gradnorm_clip.png"), dpi=150)
    plt.close(fig)


def plot_drift(data, c):
    """cumulative param drift from init: Muon-params vs Adam-params."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for v in ORDER:
        if v == "baseline":
            continue
        reps = data[v]
        xstop = np.median(ttj(reps, "epoch"))
        for ax, key in zip(axes, ["d0_mu", "d0_ad"]):
            grid, _, mid, _ = median_band(reps, "epoch", key, xstop)
            ax.plot(grid, mid, color=c[v], lw=1.8, label=v)
    axes[0].set_title("Muon-param drift from init  ‖θ−θ₀‖")
    axes[1].set_title("Adam-param drift from init  ‖θ−θ₀‖")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.set_xscale("log")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("L2 distance from init")
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out("param_drift.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs", type=Path, default=HERE / "diffeval",
                    help="dir with <variant>/*-log.txt subfolders (default: extras/diffeval)")
    ap.add_argument("--plots", type=Path, default=None,
                    help="output dir (default: <logs>/../plots, i.e. extras/plots)")
    ap.add_argument("--jackpot", type=float, default=0.8760712024651358,
                    help="RF baseline line (default: TabPFN-paper suite)")
    ap.add_argument("--tag", default="",
                    help="suffix added to every figure name (e.g. ttj_bar_<tag>.png)")
    ap.add_argument("--only-ttj", action="store_true",
                    help="produce only the time-to-jackpot bar")
    args = ap.parse_args()

    global PLOTS, JACKPOT, TAG
    PLOTS = args.plots or args.logs.parent / "plots"
    JACKPOT = args.jackpot
    TAG = args.tag
    PLOTS.mkdir(parents=True, exist_ok=True)

    data = load(args.logs)
    n = {len(data[v]) for v in ORDER}
    nlabel = f"{n.pop()}" if len(n) == 1 else "varying"
    for v in ORDER:
        print(f"{v:14s} {len(data[v])} reps")

    c = colors()
    plot_ttj_bar(data, c)
    made = 1
    if not args.only_ttj:
        plot_curves(data, c, "t_min", "wall-clock time (min, log)",
                    "auc_vs_time.png", f"AUC vs wall-clock time (median of {nlabel} reps)")
        plot_curves(data, c, "epoch", "epoch (log)",
                    "auc_vs_epochs.png", "AUC vs epochs (sample efficiency)")
        made = 3
        # the grad-norm/clip and drift plots need fields only the diffeval logs carry
        if any(np.isfinite(r["g"]).any() for r in data["muon"]):
            plot_dynamics(data, c)
            plot_drift(data, c)
            made = 5
        else:
            print("note: logs lack grad-norm/drift fields -> skipping dynamics & drift")
    print(f"wrote {made} figure(s) to {PLOTS}")


if __name__ == "__main__":
    main()
