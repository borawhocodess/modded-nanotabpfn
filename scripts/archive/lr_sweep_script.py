#!/usr/bin/env python3
"""
Parse LR sweep experiment logs and print a table of train times (seconds).
With --timeplot --heatmap, save a heatmap (activation vs LR, value = train time s) to workdir/plots/.
"""
import argparse
import re
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev


def _plot_out_path(base: Path, seedlines: bool = False, ts: str | None = None) -> Path:
    """Return path like TS-basename[-seedlines].ext (TS = yymmdd-HHMMSS), using '-' instead of '_'.
    If ts is None, use current time; pass the same ts for all saves in one run so filenames match."""
    stem = base.stem
    suffix = base.suffix
    parent = base.parent
    if ts is None:
        ts = datetime.now().strftime("%y%m%d-%H%M%S")
    name = stem
    if seedlines:
        name += "_seedlines"
    filename = f"{ts}_{name}{suffix}"
    # Replace all underscores with hyphens in the saved filename
    filename = filename.replace("_", "-")
    return parent / filename

TT_RE = re.compile(r"\bt_t:([0-9]+(?:\.[0-9]+)?)s\b")
# "datasets seen: 2688" when jackpot hit (epoch * batch_size * steps = steps with batch_size=1)
DATASETS_SEEN_RE = re.compile(r"\bdatasets seen\s*:\s*(\d+)\b", re.IGNORECASE)
# "e:42/4000" on each epoch line; use last one when run didn't finish (no datasets seen)
EPOCH_RE = re.compile(r"\be:(\d+)(?:/\d+)?\b")
# "exceeded max train time" when run hit time limit without jackpot
EXCEEDED_RE = re.compile(r"\bexceeded max train time\b", re.IGNORECASE)
# "experiment done" when run finished the loop (all epochs or time limit) without jackpot
EXPERIMENT_DONE_RE = re.compile(r"\bexperiment done\b", re.IGNORECASE)
# steps per epoch in train_nano (c.steps); used when falling back to last epoch
STEPS_PER_EPOCH = 64
# Experiment name like gelu-lr8e-4-s11 or golu-lr1e-3-s22
NAME_RE = re.compile(r"^(.+?)-lr([\d.e+-]+)-s(\d+)$", re.IGNORECASE)


def parse_train_time(log_path: Path) -> float | None:
    last_t_t = None
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = TT_RE.search(line)
                if m:
                    try:
                        last_t_t = float(m.group(1))
                    except ValueError:
                        pass
    except OSError:
        return None
    return last_t_t


def parse_mean_epoch_time(log_path: Path) -> float | None:
    """Mean epoch time (s) = last t_t / last epoch. Returns None if either missing or epoch 0."""
    last_t_t = None
    last_epoch = None
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = TT_RE.search(line)
                if m:
                    try:
                        last_t_t = float(m.group(1))
                    except ValueError:
                        pass
                m = EPOCH_RE.search(line)
                if m:
                    try:
                        last_epoch = int(m.group(1))
                    except ValueError:
                        pass
    except OSError:
        return None
    if last_t_t is not None and last_epoch is not None and last_epoch > 0:
        return last_t_t / last_epoch
    return None


def parse_steps_to_target(log_path: Path) -> float | None:
    """Steps to target: 'datasets seen' if run hit jackpot, else last_epoch * STEPS_PER_EPOCH (run didn't finish)."""
    last_datasets_seen = None
    last_epoch = None
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = DATASETS_SEEN_RE.search(line)
                if m:
                    try:
                        last_datasets_seen = float(m.group(1))
                    except ValueError:
                        pass
                m = EPOCH_RE.search(line)
                if m:
                    try:
                        last_epoch = int(m.group(1))
                    except ValueError:
                        pass
    except OSError:
        return None
    if last_datasets_seen is not None:
        return last_datasets_seen
    if last_epoch is not None:
        return float(last_epoch * STEPS_PER_EPOCH)
    return None


def parse_run_state(log_path: Path, running_secs: float = 600) -> str:
    """Return run state: d=done (jackpot), e=exceeded, r=running (recent log), x=crashed/incomplete."""
    if not log_path.exists():
        return "x"
    try:
        mtime = log_path.stat().st_mtime
    except OSError:
        return "x"
    has_datasets_seen = False
    has_exceeded = False
    has_experiment_done = False
    has_epoch = False
    has_progress = False
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                # Ignore lines from dumped source (print0(code)) so we only count runtime output
                if "print0(" in line or "console=" in line:
                    continue
                if DATASETS_SEEN_RE.search(line):
                    has_datasets_seen = True
                    break
                if EXCEEDED_RE.search(line):
                    has_exceeded = True
                if EXPERIMENT_DONE_RE.search(line):
                    has_experiment_done = True
                if EPOCH_RE.search(line):
                    has_epoch = True
                if TT_RE.search(line) or EPOCH_RE.search(line):
                    has_progress = True
    except OSError:
        return "x"
    if has_datasets_seen:
        return "d"
    if has_exceeded or has_experiment_done:
        return "e"
    if not has_epoch:
        return "x"
    if has_progress and (time.time() - mtime) <= running_secs:
        return "r"
    return "x"


def iter_run_dirs(exp_dir: Path):
    """Yield directories that contain *-log.txt (one per run)."""
    if not exp_dir.is_dir():
        return
    for sub in exp_dir.iterdir():
        if not sub.is_dir():
            continue
        logs = list(sub.glob("*-log.txt"))
        if logs:
            yield sub


def parse_experiment_name(name: str) -> tuple[str | None, float | None, int | None]:
    """Return (activation, lr, seed) or (None, None, None)."""
    m = NAME_RE.match(name.strip())
    if not m:
        return None, None, None
    act, lr_s, seed_s = m.group(1), m.group(2), m.group(3)
    try:
        lr = float(lr_s)
        seed = int(seed_s)
    except ValueError:
        return None, None, None
    return act.lower(), lr, seed


def _only_lr_sweep(data: dict[str, list]) -> dict[str, list]:
    """Keep only experiment names that match act-lrX-sY (exclude e.g. newlr)."""
    return {k: v for k, v in data.items() if parse_experiment_name(k)[0] is not None}


def collect_steps(experiments_dir: Path) -> dict[str, list[float | None]]:
    """experiment_name -> list of steps to target (datasets seen when jackpot hit), one per run dir; None if run failed."""
    experiments_dir = Path(experiments_dir)
    out = {}
    for exp_dir in sorted(experiments_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        name = exp_dir.name
        run_dirs = sorted(iter_run_dirs(exp_dir), key=lambda p: p.name)
        steps_list: list[float | None] = []
        for run_dir in run_dirs:
            logs = list(run_dir.glob("*-log.txt"))
            log_file = max(logs, key=lambda p: p.stat().st_mtime) if logs else None
            if log_file is None:
                steps_list.append(None)
                continue
            s = parse_steps_to_target(log_file)
            steps_list.append(s)
        if steps_list:
            out[name] = steps_list
    return out


def collect_times(experiments_dir: Path) -> dict[str, list[float | None]]:
    """experiment_name -> list of train times (seconds), one per run dir in sorted order; None if that run failed / no t_t."""
    experiments_dir = Path(experiments_dir)
    out = {}
    for exp_dir in sorted(experiments_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        name = exp_dir.name
        run_dirs = sorted(iter_run_dirs(exp_dir), key=lambda p: p.name)
        times: list[float | None] = []
        for run_dir in run_dirs:
            logs = list(run_dir.glob("*-log.txt"))
            log_file = max(logs, key=lambda p: p.stat().st_mtime) if logs else None
            if log_file is None:
                times.append(None)
                continue
            t = parse_train_time(log_file)
            times.append(t)
        if times:
            out[name] = times
    return out


def collect_run_states(experiments_dir: Path, running_secs: float = 600) -> dict[str, list[str]]:
    """experiment_name -> list of state chars (d/e/x/r) per run dir, same order as collect_times."""
    experiments_dir = Path(experiments_dir)
    out = {}
    for exp_dir in sorted(experiments_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        name = exp_dir.name
        run_dirs = sorted(iter_run_dirs(exp_dir), key=lambda p: p.name)
        states: list[str] = []
        for run_dir in run_dirs:
            logs = list(run_dir.glob("*-log.txt"))
            log_file = max(logs, key=lambda p: p.stat().st_mtime) if logs else None
            if log_file is None:
                states.append("x")
                continue
            states.append(parse_run_state(log_file, running_secs=running_secs))
        if states:
            out[name] = states
    return out


def collect_mean_epoch_times(experiments_dir: Path) -> dict[str, list[float | None]]:
    """experiment_name -> list of mean epoch time (s), one per run dir; None if run failed or no epoch."""
    experiments_dir = Path(experiments_dir)
    out = {}
    for exp_dir in sorted(experiments_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        name = exp_dir.name
        run_dirs = sorted(iter_run_dirs(exp_dir), key=lambda p: p.name)
        vals: list[float | None] = []
        for run_dir in run_dirs:
            logs = list(run_dir.glob("*-log.txt"))
            log_file = max(logs, key=lambda p: p.stat().st_mtime) if logs else None
            if log_file is None:
                vals.append(None)
                continue
            v = parse_mean_epoch_time(log_file)
            vals.append(v)
        if vals:
            out[name] = vals
    return out


def make_table(data: dict[str, list[float | None]], states: dict[str, list[str]] | None = None) -> str:
    """Format table: rows = experiment, cols = r1..rN, [st], n, avg, std, median. st = d/e/x/r/- per run."""
    max_runs = max((len(times) for times in data.values()), default=0)
    if states is not None:
        headers = ["experiment"] + [f"r{i + 1}" for i in range(max_runs)] + ["st", "n", "avg", "std", "median"]
    else:
        headers = ["experiment"] + [f"r{i + 1}" for i in range(max_runs)] + ["n", "avg", "std", "median"]
    rows = []
    for name in sorted(data.keys()):
        times = data[name]
        st_list = states.get(name, []) if states is not None else []
        def _run_display(i: int) -> str:
            if i < len(times) and times[i] is not None:
                return f"{times[i]:.1f}"
            if i < len(st_list) and st_list[i] == "x":
                return "-"  # crashed
            return " "  # no run
        run_cols = [_run_display(i) for i in range(max_runs)]
        valid = [t for t in times if t is not None]
        n_str = str(len(valid))
        avg = f"{mean(valid):.1f}" if valid else " "
        std = f"{pstdev(valid):.1f}" if len(valid) > 1 else " "
        med = f"{median(valid):.1f}" if valid else " "
        if states is not None:
            st_list = states.get(name, [])
            def _st_display(c: str) -> str:
                if c == "-":
                    return " "
                if c == "x":
                    return "-"
                return c
            st_str = " ".join(
                _st_display(st_list[i] if i < len(st_list) else "-") for i in range(max_runs)
            )
            rows.append([name] + run_cols + [st_str, n_str, avg, std, med])
        else:
            rows.append([name] + run_cols + [n_str, avg, std, med])
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    lines = [
        "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)),
        "  ".join("-" * widths[i] for i in range(len(widths))),
    ]
    for row in rows:
        lines.append("  ".join(row[i].ljust(widths[i]) for i in range(len(row))))
    return "\n".join(lines)


def build_heatmap_data(data: dict[str, list[float | None]]) -> tuple[list[str], list[float], dict[tuple[str, float], float], dict[tuple[str, float], float]]:
    """Parse names to (activations, lrs, (act, lr) -> mean_time, (act, lr) -> std). Uses only valid (non-None) times."""
    activations = set()
    lrs = set()
    cell_values: dict[tuple[str, float], list[float]] = {}
    for name, times in data.items():
        valid = [t for t in times if t is not None]
        if not valid:
            continue
        act, lr, _ = parse_experiment_name(name)
        if act is None or lr is None:
            continue
        act_display = act.upper() if act == "gelu" else act.capitalize()
        if act_display == "Golu":
            act_display = "GoLU"
        activations.add(act_display)
        lrs.add(lr)
        key = (act_display, lr)
        cell_values.setdefault(key, []).extend(valid)
    means = {}
    stds = {}
    for key, vals in cell_values.items():
        means[key] = sum(vals) / len(vals)
        stds[key] = pstdev(vals) if len(vals) > 1 else 0.0
    return sorted(activations), sorted(lrs), means, stds


def save_heatmap(experiments_dir: Path, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    data = collect_times(experiments_dir)
    activations, lrs, means, stds = build_heatmap_data(data)
    if not activations or not lrs:
        print("No data for heatmap (need experiment names like gelu-lr8e-4-s11).")
        return
    import numpy as np
    matrix = np.full((len(activations), len(lrs)), np.nan)
    annot_matrix = np.empty((len(activations), len(lrs)), dtype=object)
    for i, act in enumerate(activations):
        for j, lr in enumerate(lrs):
            m = means.get((act, lr), np.nan)
            s = stds.get((act, lr), 0.0)
            matrix[i, j] = m
            if np.isnan(m):
                annot_matrix[i, j] = ""
            else:
                annot_matrix[i, j] = f"{m:.1f}\n±{s:.1f}"
    lr_labels = [str(lr) for lr in lrs]
    fig, ax = plt.subplots(figsize=(max(6, len(lrs) * 1.5), max(4, len(activations) * 1.2)))
    sns.heatmap(
        matrix,
        ax=ax,
        annot=annot_matrix,
        fmt="",
        cmap="coolwarm",
        xticklabels=lr_labels,
        yticklabels=activations,
        cbar_kws={"label": "Train time (s)"},
    )
    ax.set_xlabel("Learning rate", fontsize=12)
    ax.set_ylabel("Activation", fontsize=12)
    ax.set_title("Train time (s) by activation and learning rate")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved heatmap to {out_path}")


def save_lineplot(experiments_dir: Path, out_path: Path, seedlines: bool = False) -> None:
    """Save a line plot: train time vs LR, one line per activation (with std error bars).

    If seedlines is True, also draw faint per-seed lines in the background.
    """
    import matplotlib.pyplot as plt

    data = collect_times(experiments_dir)
    activations, lrs, means, stds = build_heatmap_data(data)
    if not activations or not lrs:
        print("No data for line plot (need experiment names like gelu-lr8e-4-s11).")
        return

    # Optional: per-seed mean curves
    seeds: list[int] = []
    seed_means: dict[tuple[str, int], dict[float, float]] = {}
    if seedlines:
        seed_set = set()
        for name, times in data.items():
            act, lr, seed = parse_experiment_name(name)
            if act is None or lr is None or seed is None:
                continue
            valid = [t for t in times if t is not None]
            if not valid:
                continue
            act_display = act.upper() if act == "gelu" else act.capitalize()
            if act_display == "Golu":
                act_display = "GoLU"
            key = (act_display, seed)
            seed_means.setdefault(key, {})[lr] = sum(valid) / len(valid)
            seed_set.add(seed)
        seeds = sorted(seed_set)

    fig, ax = plt.subplots(figsize=(8, 5))
    seed_styles = ["--", "-.", ":"]
    seed_labeled: set[tuple[str, int]] = set()
    for act in activations:
        # Main activation line (mean ± std)
        y = [means.get((act, lr), float("nan")) for lr in lrs]
        yerr = [stds.get((act, lr), 0.0) for lr in lrs]
        main = ax.errorbar(
            lrs,
            y,
            yerr=yerr,
            marker="o",
            linewidth=1.5,
            capsize=3,
            label=act,
        )
        # Get the color matplotlib assigned to this activation line
        try:
            main_color = main.lines[0].get_color()
            main_zorder = main.lines[0].get_zorder()
        except Exception:
            # Fallback in case API differs
            main_color = None
            main_zorder = 2

        # Optional seed lines (per-seed means), faint in the background, same color as activation
        if seedlines and seeds:
            for idx, seed in enumerate(seeds):
                key = (act, seed)
                per_lr = seed_means.get(key, {})
                if not per_lr:
                    continue
                y_seed = [per_lr.get(lr, float("nan")) for lr in lrs]
                # Add legend entry once per (activation, seed)
                seed_label = None
                if key not in seed_labeled:
                    seed_label = f"{act} seed {seed}"
                    seed_labeled.add(key)
                ax.plot(
                    lrs,
                    y_seed,
                    linestyle=seed_styles[idx % len(seed_styles)],
                    linewidth=1.0,
                    alpha=0.3,
                    color=main_color,
                    zorder=main_zorder - 1,
                    label=seed_label,
                )

    ax.set_xscale("log")
    ax.set_xticks(lrs)
    ax.set_xticklabels([str(lr) for lr in lrs], rotation=75, ha="right")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Train time (s)")
    ax.set_title("Train time vs learning rate")
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved line plot to {out_path}")


def save_stepplot(experiments_dir: Path, out_path: Path, seedlines: bool = False, data: dict[str, list[float | None]] | None = None) -> None:
    """Save a line plot: steps to target vs LR, one line per activation (with std error bars).

    If seedlines is True, also draw faint per-seed lines in the background.
    If data is provided, use it; otherwise collect_steps(experiments_dir).
    """
    import matplotlib.pyplot as plt

    if data is None:
        data = collect_steps(experiments_dir)
    activations, lrs, means, stds = build_heatmap_data(data)
    if not activations or not lrs:
        print("No data for step plot (need experiment names like gelu-lr8e-4-s11 and logs with 'datasets seen').")
        return

    seeds: list[int] = []
    seed_means: dict[tuple[str, int], dict[float, float]] = {}
    if seedlines:
        seed_set = set()
        for name, steps_list in data.items():
            act, lr, seed = parse_experiment_name(name)
            if act is None or lr is None or seed is None:
                continue
            valid = [s for s in steps_list if s is not None]
            if not valid:
                continue
            act_display = act.upper() if act == "gelu" else act.capitalize()
            if act_display == "Golu":
                act_display = "GoLU"
            key = (act_display, seed)
            seed_means.setdefault(key, {})[lr] = sum(valid) / len(valid)
            seed_set.add(seed)
        seeds = sorted(seed_set)

    fig, ax = plt.subplots(figsize=(8, 5))
    seed_styles = ["--", "-.", ":"]
    seed_labeled: set[tuple[str, int]] = set()
    for act in activations:
        y = [means.get((act, lr), float("nan")) for lr in lrs]
        yerr = [stds.get((act, lr), 0.0) for lr in lrs]
        main = ax.errorbar(
            lrs,
            y,
            yerr=yerr,
            marker="o",
            linewidth=1.5,
            capsize=3,
            label=act,
        )
        try:
            main_color = main.lines[0].get_color()
            main_zorder = main.lines[0].get_zorder()
        except Exception:
            main_color = None
            main_zorder = 2

        if seedlines and seeds:
            for idx, seed in enumerate(seeds):
                key = (act, seed)
                per_lr = seed_means.get(key, {})
                if not per_lr:
                    continue
                y_seed = [per_lr.get(lr, float("nan")) for lr in lrs]
                seed_label = None
                if key not in seed_labeled:
                    seed_label = f"{act} seed {seed}"
                    seed_labeled.add(key)
                ax.plot(
                    lrs,
                    y_seed,
                    linestyle=seed_styles[idx % len(seed_styles)],
                    linewidth=1.0,
                    alpha=0.3,
                    color=main_color,
                    zorder=main_zorder - 1,
                    label=seed_label,
                )

    ax.set_xscale("log")
    ax.set_xticks(lrs)
    ax.set_xticklabels([str(lr) for lr in lrs], rotation=45, ha="right")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Steps to target")
    ax.set_title("Steps to target vs learning rate")
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved step plot to {out_path}")


def save_meanepochtimeplot(experiments_dir: Path, out_path: Path, seedlines: bool = False, data: dict[str, list[float | None]] | None = None) -> None:
    """Save a line plot: mean epoch time (s) vs LR, one line per activation (with std error bars).

    If seedlines is True, also draw faint per-seed lines. If data is provided, use it; else collect_mean_epoch_times.
    """
    import matplotlib.pyplot as plt

    if data is None:
        data = collect_mean_epoch_times(experiments_dir)
    activations, lrs, means, stds = build_heatmap_data(data)
    if not activations or not lrs:
        print("No data for mean epoch time plot (need experiment names like gelu-lr8e-4-s11 and logs with t_t and e:N).")
        return

    seeds: list[int] = []
    seed_means: dict[tuple[str, int], dict[float, float]] = {}
    if seedlines:
        seed_set = set()
        for name, vals in data.items():
            act, lr, seed = parse_experiment_name(name)
            if act is None or lr is None or seed is None:
                continue
            valid = [v for v in vals if v is not None]
            if not valid:
                continue
            act_display = act.upper() if act == "gelu" else act.capitalize()
            if act_display == "Golu":
                act_display = "GoLU"
            key = (act_display, seed)
            seed_means.setdefault(key, {})[lr] = sum(valid) / len(valid)
            seed_set.add(seed)
        seeds = sorted(seed_set)

    fig, ax = plt.subplots(figsize=(8, 5))
    seed_styles = ["--", "-.", ":"]
    seed_labeled: set[tuple[str, int]] = set()
    for act in activations:
        y = [means.get((act, lr), float("nan")) for lr in lrs]
        yerr = [stds.get((act, lr), 0.0) for lr in lrs]
        main = ax.errorbar(
            lrs,
            y,
            yerr=yerr,
            marker="o",
            linewidth=1.5,
            capsize=3,
            label=act,
        )
        try:
            main_color = main.lines[0].get_color()
            main_zorder = main.lines[0].get_zorder()
        except Exception:
            main_color = None
            main_zorder = 2

        if seedlines and seeds:
            for idx, seed in enumerate(seeds):
                key = (act, seed)
                per_lr = seed_means.get(key, {})
                if not per_lr:
                    continue
                y_seed = [per_lr.get(lr, float("nan")) for lr in lrs]
                seed_label = None
                if key not in seed_labeled:
                    seed_label = f"{act} seed {seed}"
                    seed_labeled.add(key)
                ax.plot(
                    lrs,
                    y_seed,
                    linestyle=seed_styles[idx % len(seed_styles)],
                    linewidth=1.0,
                    alpha=0.3,
                    color=main_color,
                    zorder=main_zorder - 1,
                    label=seed_label,
                )

    ax.set_xscale("log")
    ax.set_xticks(lrs)
    ax.set_xticklabels([str(lr) for lr in lrs], rotation=45, ha="right")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Mean epoch time (s)")
    ax.set_title("Mean epoch time vs learning rate")
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved mean epoch time plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Summarize LR sweep experiment logs (train time in seconds).")
    parser.add_argument(
        "--experiments-dir", "-d",
        default="workdir/experiments",
        type=Path,
        help="Experiments root (e.g. workdir/experiments)",
    )
    parser.add_argument("--allplots", action="store_true", help="Produce all plots (lineplot, stepplot, meanepochtimeplot, heatmap) at once")
    parser.add_argument("--lineplot", action="store_true", help="Save activation vs LR line plot to workdir/plots/")
    parser.add_argument("--stepplot", action="store_true", help="Save steps-to-target vs LR line plot to workdir/plots/")
    parser.add_argument("--meanepochtimeplot", action="store_true", help="Save mean epoch time (s) vs LR line plot to workdir/plots/")
    parser.add_argument("--seedlines", action="store_true", help="When using --lineplot, --stepplot, or --meanepochtimeplot, also draw per-seed lines")
    parser.add_argument("--heatmap", action="store_true", help="Save activation vs LR heatmap to workdir/plots/")
    parser.add_argument(
        "--out",
        default="workdir/plots/lr_sweep_train_time_heatmap.png",
        type=Path,
        help="Heatmap output path (default: workdir/plots/lr_sweep_train_time_heatmap.png)",
    )
    parser.add_argument(
        "--lineplot-out",
        default="workdir/plots/lr_sweep_train_time_lineplot.png",
        type=Path,
        help="Line plot output path (default: workdir/plots/lr_sweep_train_time_lineplot.png)",
    )
    parser.add_argument(
        "--stepplot-out",
        default="workdir/plots/lr_sweep_steps_to_target_lineplot.png",
        type=Path,
        help="Step plot output path (default: workdir/plots/lr_sweep_steps_to_target_lineplot.png)",
    )
    parser.add_argument(
        "--meanepochtimeplot-out",
        default="workdir/plots/lr_sweep_mean_epoch_time_lineplot.png",
        type=Path,
        help="Mean epoch time plot output path (default: workdir/plots/lr_sweep_mean_epoch_time_lineplot.png)",
    )
    parser.add_argument(
        "--running-secs",
        type=float,
        default=600,
        help="Log mtime within this many seconds → state 'r' (running). Default 600.",
    )
    args = parser.parse_args()
    if args.allplots:
        args.lineplot = args.stepplot = args.meanepochtimeplot = args.heatmap = True

    experiments_dir = Path(args.experiments_dir)
    data_times = _only_lr_sweep(collect_times(experiments_dir))
    data_steps = _only_lr_sweep(collect_steps(experiments_dir)) if args.stepplot else None
    run_states = collect_run_states(experiments_dir, running_secs=args.running_secs)
    run_states = _only_lr_sweep(run_states)

    if args.stepplot and data_steps:
        print(make_table(data_steps, states=run_states))
    elif data_times:
        print(make_table(data_times, states=run_states))
    else:
        print("No experiment logs found.")
        return
    print("st: d=done e=exceeded -=crashed r=running  =no run")

    # Single timestamp for all plot filenames in this run
    plot_ts = datetime.now().strftime("%y%m%d-%H%M%S")

    if args.heatmap:
        save_heatmap(experiments_dir, _plot_out_path(Path(args.out), ts=plot_ts))
    if args.lineplot:
        if args.allplots:
            save_lineplot(experiments_dir, _plot_out_path(Path(args.lineplot_out), seedlines=False, ts=plot_ts), seedlines=False)
            save_lineplot(experiments_dir, _plot_out_path(Path(args.lineplot_out), seedlines=True, ts=plot_ts), seedlines=True)
        else:
            save_lineplot(experiments_dir, _plot_out_path(Path(args.lineplot_out), seedlines=args.seedlines, ts=plot_ts), seedlines=args.seedlines)
    if args.stepplot:
        if args.allplots:
            save_stepplot(experiments_dir, _plot_out_path(Path(args.stepplot_out), seedlines=False, ts=plot_ts), seedlines=False, data=data_steps)
            save_stepplot(experiments_dir, _plot_out_path(Path(args.stepplot_out), seedlines=True, ts=plot_ts), seedlines=True, data=data_steps)
        else:
            save_stepplot(experiments_dir, _plot_out_path(Path(args.stepplot_out), seedlines=args.seedlines, ts=plot_ts), seedlines=args.seedlines, data=data_steps)
    if args.meanepochtimeplot:
        data_met = _only_lr_sweep(collect_mean_epoch_times(experiments_dir))
        if args.allplots:
            save_meanepochtimeplot(experiments_dir, _plot_out_path(Path(args.meanepochtimeplot_out), seedlines=False, ts=plot_ts), seedlines=False, data=data_met)
            save_meanepochtimeplot(experiments_dir, _plot_out_path(Path(args.meanepochtimeplot_out), seedlines=True, ts=plot_ts), seedlines=True, data=data_met)
        else:
            save_meanepochtimeplot(experiments_dir, _plot_out_path(Path(args.meanepochtimeplot_out), seedlines=args.seedlines, ts=plot_ts), seedlines=args.seedlines, data=data_met)


if __name__ == "__main__":
    main()
