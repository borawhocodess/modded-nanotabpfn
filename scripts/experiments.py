import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev

HOST_RE = re.compile(r"^\s*host:\s*(.+?)\s*$")
TT_RE = re.compile(r"\bt_t:([0-9]+(?:\.[0-9]+)?)s\b")
TOTAL_RE = re.compile(r"\b(total time|total_time|elapsed|duration)\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)s\b", re.IGNORECASE)
TIME_BRACKET_RE = re.compile(r"\[(\d{2}):(\d{2}):(\d{2})\]")
ROC_RE = re.compile(r"\bavg_roc_auc\s*:\s*([0-9]+(?:\.[0-9]+)?)\b")
MU_E_T_RE = re.compile(r"μ_e_t:([0-9]+(?:\.[0-9]+)?)s")
EPOCH_RE = re.compile(r"\be:(\d+)(?:/\d+)?\b")
DATASETS_RE = re.compile(r"\bdatasets seen\s*:\s*(\d+)\b", re.IGNORECASE)
SCRIPT_RUNTIME_RE = re.compile(r"script runtime\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*mins", re.IGNORECASE)

# per-epoch diagnostic fields (for --trace). value may be a float or "nan".
_NUM = r"(nan|[0-9]+(?:\.[0-9]+)?)"
TRACE_FIELDS = {
    "loss": re.compile(r"μ_l:" + _NUM),
    "g": re.compile(r"\bg:" + _NUM),
    "g_max": re.compile(r"\bg_max:" + _NUM),
    "clip": re.compile(r"\bclip:" + _NUM),
    "d_mu": re.compile(r"\bd_mu:" + _NUM),
    "d_ad": re.compile(r"\bd_ad:" + _NUM),
    "d0_mu": re.compile(r"\bd0_mu:" + _NUM),
    "d0_ad": re.compile(r"\bd0_ad:" + _NUM),
    "roc_auc": ROC_RE,
}

COLUMNS = [
    {
        "key": "experiment_id",
        "flag": "--eid",
        "attr": "eid",
        "header": "experiment id",
    },
    {
        "key": "date",
        "flag": "--date",
        "attr": "date",
        "header": "date",
    },
    {
        "key": "time",
        "flag": "--time",
        "attr": "time",
        "header": "time",
    },
    {
        "key": "hostname",
        "flag": "--hostname",
        "attr": "hostname",
        "header": "hostname",
    },
    {
        "key": "total_time",
        "flag": "--total-time",
        "attr": "total_time",
        "header": "total time",
    },
    {
        "key": "total_time_min",
        "flag": "--mins",
        "attr": "mins",
        "header": "in mins",
    },
    {
        "key": "roc_auc",
        "flag": "--roc-auc",
        "attr": "roc_auc",
        "header": "roc auc",
    },
    {
        "key": "epoch",
        "flag": "--epoch",
        "attr": "epoch",
        "header": "epoch",
    },
    {
        "key": "mean_epoch_time",
        "flag": "--mean-epoch-time",
        "attr": "mean_epoch_time",
        "header": "μ epoch t",
    },
    {
        "key": "datasets",
        "flag": "--datasets",
        "attr": "datasets",
        "header": "datasets",
    },
    {
        "key": "script_runtime",
        "flag": "--script-runtime",
        "attr": "script_runtime",
        "header": "runtime",
    },
    {
        "key": "id_name0",
        "flag": "--id-name0",
        "attr": "id_name0",
        "header": "id-name",
    },
    {
        "key": "id_name",
        "flag": "--id-name",
        "attr": "id_name",
        "header": "id-name",
    },
]

COLUMN_BY_KEY = {col["key"]: col for col in COLUMNS}
FLAG_TO_KEY = {col["flag"]: col["key"] for col in COLUMNS}
DIR_FLAGS = {"--experiments-dir", "--dir", "-d"}

# Named output presets. Add more entries as needed.
SETTINGS_PRESETS = {
    "xyz": {
        "short_id": True,
        "columns": ["experiment_id", "hostname", "total_time_min", "epoch", "mean_epoch_time", "datasets"],
    },
    "alpha": {
        "columns": ["hostname", "total_time_min", "epoch", "mean_epoch_time", "datasets", "script_runtime", "id_name0"],
    },
    "beta": {
        "columns": ["date", "hostname", "total_time_min", "epoch", "mean_epoch_time", "datasets", "script_runtime", "id_name"],
    },
}


def parse_log(log_path):
    hostname = None
    total_time = None
    roc_auc = None
    mean_epoch_time = None
    epoch = None
    datasets = None
    script_runtime = None
    last_t_t = None
    first_clock = None
    last_clock = None

    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if hostname is None:
                    match = HOST_RE.match(line)
                    if match:
                        hostname = match.group(1).strip()

                match = TT_RE.search(line)
                if match:
                    try:
                        last_t_t = float(match.group(1))
                    except ValueError:
                        pass

                if total_time is None:
                    match = TOTAL_RE.search(line)
                    if match:
                        try:
                            total_time = float(match.group(2))
                        except ValueError:
                            pass

                match = ROC_RE.search(line)
                if match:
                    try:
                        roc_auc = float(match.group(1))
                    except ValueError:
                        pass

                match = MU_E_T_RE.search(line)
                if match:
                    try:
                        mean_epoch_time = float(match.group(1))
                    except ValueError:
                        pass

                match = EPOCH_RE.search(line)
                if match:
                    try:
                        epoch = int(match.group(1))
                    except ValueError:
                        pass

                match = DATASETS_RE.search(line)
                if match:
                    try:
                        datasets = int(match.group(1))
                    except ValueError:
                        pass

                match = SCRIPT_RUNTIME_RE.search(line)
                if match:
                    try:
                        script_runtime = float(match.group(1))
                    except ValueError:
                        pass

                match = TIME_BRACKET_RE.search(line)
                if match:
                    h = int(match.group(1))
                    m = int(match.group(2))
                    s = int(match.group(3))
                    seconds = h * 3600 + m * 60 + s
                    if first_clock is None:
                        first_clock = seconds
                    last_clock = seconds
    except FileNotFoundError:
        return None, None, None, None, None, None, None

    if total_time is None:
        total_time = last_t_t

    if total_time is None and first_clock is not None and last_clock is not None:
        if last_clock < first_clock:
            last_clock += 24 * 3600
        total_time = float(last_clock - first_clock)

    return hostname, total_time, roc_auc, mean_epoch_time, epoch, datasets, script_runtime


def pick_log_file(exp_dir):
    logs = list(exp_dir.glob("*-log.txt"))
    return max(logs, key=lambda p: p.stat().st_mtime) if logs else None


def iter_experiments(experiments_dir):
    if not experiments_dir.exists():
        return []
    return sorted([p for p in experiments_dir.iterdir() if p.is_dir()])


def make_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    header_line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep_line = "  ".join("-" * widths[i] for i in range(len(headers)))
    lines = [header_line, sep_line]
    for row in rows:
        lines.append("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(lines)


def extract_order(argv):
    order = []
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token in DIR_FLAGS:
            skip_next = True
            continue
        if any(token.startswith(flag + "=") for flag in DIR_FLAGS if flag.startswith("--")):
            continue
        key = FLAG_TO_KEY.get(token)
        if key:
            order.append(key)
    return order


def unique(items):
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def column_values(exp_id, hostname, total_time, roc_auc, mean_epoch_time, epoch, datasets, script_runtime=None):
    total_time_min = "-" if total_time is None else f"{total_time / 60:.2f}m"
    return {
        "experiment_id": exp_id,
        "date": exp_date(exp_id),
        "time": exp_time(exp_id),
        "hostname": hostname or "-",
        "total_time": "-" if total_time is None else f"{total_time:.2f}s",
        "total_time_min": total_time_min,
        "roc_auc": "-" if roc_auc is None else f"{roc_auc:.6f}",
        "epoch": "-" if epoch is None else str(epoch),
        "mean_epoch_time": "-" if mean_epoch_time is None else f"{mean_epoch_time:.2f}s",
        "datasets": "-" if datasets is None else str(datasets),
        "script_runtime": "-" if script_runtime is None else f"{script_runtime:.2f}m",
        "id_name0": id_name0(exp_id),
        "id_name": id_name(exp_id),
    }


def exp_date(exp_id: str) -> str:
    """From '260420-022941-...' return '26-04-20'."""
    parts = exp_id.split("-")
    if parts and len(parts[0]) == 6 and parts[0].isdigit():
        d = parts[0]
        return f"{d[0:2]}-{d[2:4]}-{d[4:6]}"
    return "-"


def exp_time(exp_id: str) -> str:
    """From '260420-022941-...' return '02:29:41'."""
    parts = exp_id.split("-")
    if len(parts) >= 2 and len(parts[1]) == 6 and parts[1].isdigit():
        t = parts[1]
        return f"{t[0:2]}:{t[2:4]}:{t[4:6]}"
    return "-"


def short_experiment_id(exp_id: str) -> str:
    """
    For ids like:
      260311-154259-9fcfcbb7-new2-s11
    return:
      9fcfcbb7
    """
    parts = exp_id.split("-")
    if len(parts) >= 3:
        return parts[2]
    return exp_id


def id_name0(exp_id: str) -> str:
    """
    For ids like:
      260311-154259-9fcfcbb7-new2-s11
    return:
      9fcfcbb7-new2
    (short hash + first segment of the experiment name)
    """
    parts = exp_id.split("-")
    if len(parts) >= 4:
        return f"{parts[2]}-{parts[3]}"
    if len(parts) >= 3:
        return parts[2]
    return exp_id


def id_name(exp_id: str) -> str:
    """
    For ids like:
      260311-154259-9fcfcbb7-new2-s11
    return:
      9fcfcbb7-new2-s11
    (short hash + all remaining name segments)
    """
    parts = exp_id.split("-")
    if len(parts) >= 3:
        return "-".join(parts[2:])
    return exp_id


def save_plot(values, out_path):
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    x = list(range(1, len(values) + 1))
    plt.figure(figsize=(8, 4))
    plt.plot(x, values, marker="o", linewidth=1.5)
    plt.xticks(x)
    plt.xlabel("experiment")
    plt.ylabel("total_time (s)")
    add_stat_lines(plt, values)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    return out_path


def save_plot_grouped(series_by_host, out_path):
    if "__scatter__" in series_by_host:
        points = [(h, v) for h, v in series_by_host["__scatter__"] if v is not None]
        if len(points) < 2:
            return None
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return None
        hosts = [h for h, _ in points]
        vals = [v for _, v in points]
        plt.figure(figsize=(9, 4.5))
        plt.scatter(hosts, vals, s=20, alpha=0.7)
        plt.xticks(rotation=80, ha="right")
        plt.xlabel("hostname")
        plt.ylabel("total_time (s)")
        add_stat_lines(plt, vals)
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
        plt.close()
        return out_path

    series_by_host = {k: [v for v in vs if v is not None] for k, vs in series_by_host.items()}
    series_by_host = {k: v for k, v in series_by_host.items() if len(v) >= 2}
    if not series_by_host:
        return None
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    plt.figure(figsize=(9, 4.5))
    max_len = 0
    for host, values in sorted(series_by_host.items()):
        x = list(range(1, len(values) + 1))
        max_len = max(max_len, len(values))
        plt.plot(x, values, marker="o", linewidth=1.2, label=host)
    if max_len:
        plt.xticks(list(range(1, max_len + 1)))
    plt.xlabel("experiment")
    plt.ylabel("total_time (s)")
    all_vals = [v for series in series_by_host.values() for v in series if v is not None]
    add_stat_lines(plt, all_vals)
    plt.legend(loc="best", fontsize="small")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    return out_path


def add_stat_lines(plt, values):
    if len(values) < 2:
        return
    mu = mean(values)
    sigma = pstdev(values)
    med = median(values)
    plt.axhline(y=mu, linestyle="-", linewidth=1.4, alpha=0.8)
    plt.axhline(y=med, linestyle=":", linewidth=1.1, alpha=0.7, color="purple")
    plt.axhline(y=mu + sigma, linestyle="--", linewidth=0.9, alpha=0.6)
    plt.axhline(y=mu - sigma, linestyle="--", linewidth=0.9, alpha=0.6)
    plt.axhline(y=mu + 2 * sigma, linestyle="--", linewidth=0.8, alpha=0.5)
    plt.axhline(y=mu - 2 * sigma, linestyle="--", linewidth=0.8, alpha=0.5)
    ticks = [mu, mu + sigma, mu - sigma, mu + 2 * sigma, mu - 2 * sigma]
    ticks = sorted(ticks)
    plt.yticks(ticks)


def print_group(experiments_dir, columns, args, exclude_ids):
    """Print stats block + runs table for a single experiments directory."""
    rows = []
    times = []
    times_by_host = {}
    stats_by_col = {}

    for idx, exp_dir in enumerate(iter_experiments(experiments_dir), start=1):
        exp_id = exp_dir.name
        if exp_id in exclude_ids:
            continue
        log_path = pick_log_file(exp_dir)
        hostname = None
        total_time = None
        roc_auc = None
        mean_epoch_time = None
        epoch = None
        datasets = None
        script_runtime = None
        if log_path is not None:
            hostname, total_time, roc_auc, mean_epoch_time, epoch, datasets, script_runtime = parse_log(log_path)

        display_exp_id = short_experiment_id(exp_id) if args.short_id else exp_id
        values = column_values(display_exp_id, hostname, total_time, roc_auc, mean_epoch_time, epoch, datasets, script_runtime)
        rows.append((total_time, [str(idx)] + [values[key] for key in columns]))
        times.append(total_time)

        if "total_time" in columns and total_time is not None:
            stats_by_col.setdefault("total_time", []).append(total_time)
        if "total_time_min" in columns and total_time is not None:
            stats_by_col.setdefault("total_time_min", []).append(total_time / 60.0)
        if "roc_auc" in columns and roc_auc is not None:
            stats_by_col.setdefault("roc_auc", []).append(roc_auc)
        if "epoch" in columns and epoch is not None:
            stats_by_col.setdefault("epoch", []).append(epoch)
        if "mean_epoch_time" in columns and mean_epoch_time is not None:
            stats_by_col.setdefault("mean_epoch_time", []).append(mean_epoch_time)
        if "datasets" in columns and datasets is not None:
            stats_by_col.setdefault("datasets", []).append(datasets)
        if "script_runtime" in columns and script_runtime is not None:
            stats_by_col.setdefault("script_runtime", []).append(script_runtime)
        if hostname:
            times_by_host.setdefault(hostname, []).append(total_time)

    if args.sort:
        rows.sort(key=lambda r: float("inf") if r[0] is None else r[0])
        rows = [[str(i + 1)] + row for i, (_, row) in enumerate(rows)]
        headers = ["##", "#"] + [
            ("id" if args.short_id and key == "experiment_id" else COLUMN_BY_KEY[key]["header"])
            for key in columns
        ]
    else:
        rows = [row for _, row in rows]
        headers = ["#"] + [
            ("id" if args.short_id and key == "experiment_id" else COLUMN_BY_KEY[key]["header"])
            for key in columns
        ]

    if not rows:
        return times_by_host

    stats_summary = {}
    for key, vals in stats_by_col.items():
        if not vals:
            continue
        stats_summary[key] = {
            "mean": mean(vals),
            "std": pstdev(vals),
            "median": median(vals),
        }

    if stats_summary:
        widths = [len(h) for h in headers]
        for row in rows:
            for idx, cell in enumerate(row):
                widths[idx] = max(widths[idx], len(cell))

        def fmt_stat_value(col_key, value):
            if col_key == "total_time":
                return f"{value:.2f}s"
            if col_key == "total_time_min":
                return f"{value:.2f}m"
            if col_key == "roc_auc":
                return f"{value:.6f}"
            if col_key in ("epoch", "datasets"):
                return f"{int(round(value))}"
            if col_key == "mean_epoch_time":
                return f"{value:.2f}s"
            if col_key == "script_runtime":
                return f"{value:.2f}m"
            return ""

        offset = 2 if args.sort else 1
        stat_col_indices = []
        for header_idx in range(offset, len(headers)):
            col_key = columns[header_idx - offset]
            if col_key in stats_summary:
                stat_col_indices.append(header_idx)
        first_calc_idx = min(stat_col_indices) if stat_col_indices else len(headers)
        value_prefix_width = sum(widths[:first_calc_idx]) + 2 * first_calc_idx
        dash_prefix_width = sum(widths[:first_calc_idx]) + 2 * max(first_calc_idx - 1, 0)

        for stat_type in ("mean", "std", "median"):
            pad = max(0, value_prefix_width - len(stat_type))
            prefix = stat_type + (" " * pad)
            rest_cells = []
            for header_idx in range(first_calc_idx, len(headers)):
                col_key = columns[header_idx - offset] if header_idx >= offset else None
                if col_key is None or col_key not in stats_summary:
                    rest_cells.append("".ljust(widths[header_idx]))
                    continue
                value = stats_summary[col_key].get(stat_type)
                if value is None:
                    rest_cells.append("".ljust(widths[header_idx]))
                    continue
                rest_cells.append(fmt_stat_value(col_key, value).ljust(widths[header_idx]))
            print(prefix + "  ".join(rest_cells))

        if first_calc_idx < len(headers):
            prefix_dashes = "-" * dash_prefix_width
            rest_dashes = "  ".join("-" * widths[i] for i in range(first_calc_idx, len(headers)))
            print(prefix_dashes + "  " + rest_dashes)

    print(make_table(headers, rows))
    return times_by_host


def collect_group_stats(experiments_dir, exclude_ids):
    """Aggregate stats for one group folder. Returns (n_runs, stats_by_col)."""
    stats_by_col = {}
    n = 0
    for exp_dir in iter_experiments(experiments_dir):
        exp_id = exp_dir.name
        if exp_id in exclude_ids:
            continue
        log_path = pick_log_file(exp_dir)
        if log_path is None:
            continue
        hostname, total_time, roc_auc, mean_epoch_time, epoch, datasets, script_runtime = parse_log(log_path)
        n += 1
        if total_time is not None:
            stats_by_col.setdefault("total_time_min", []).append(total_time / 60.0)
        if epoch is not None:
            stats_by_col.setdefault("epoch", []).append(epoch)
        if mean_epoch_time is not None:
            stats_by_col.setdefault("mean_epoch_time", []).append(mean_epoch_time)
        if datasets is not None:
            stats_by_col.setdefault("datasets", []).append(datasets)
        if script_runtime is not None:
            stats_by_col.setdefault("script_runtime", []).append(script_runtime)
        if roc_auc is not None:
            stats_by_col.setdefault("roc_auc", []).append(roc_auc)
    return n, stats_by_col


SUMMARY_COLUMNS = [
    ("total_time_min", "med mins", lambda v: f"{v:.2f}m"),
    ("epoch", "med epoch", lambda v: f"{int(round(v))}"),
    ("mean_epoch_time", "med μ ep t", lambda v: f"{v:.2f}s"),
    ("datasets", "med datasets", lambda v: f"{int(round(v))}"),
    ("script_runtime", "med runtime", lambda v: f"{v:.2f}m"),
]


def print_summary(top_dirs, args, exclude_ids):
    rows = []
    for group_dir in top_dirs:
        sub_run_dirs = sorted([p for p in group_dir.iterdir() if p.is_dir()])
        if not sub_run_dirs:
            continue
        if not any(list(d.glob("*-log.txt")) for d in sub_run_dirs):
            continue
        n, stats = collect_group_stats(group_dir, exclude_ids)
        if n == 0:
            continue
        cells = [group_dir.name, str(n)]
        for key, _, fmt in SUMMARY_COLUMNS:
            vals = stats.get(key, [])
            cells.append(fmt(median(vals)) if vals else "-")
        rows.append(cells)

    if not rows:
        print("No experiments found.")
        return 1

    if args.sort:
        def _key(r):
            v = r[2]
            if v == "-":
                return float("inf")
            return float(v.rstrip("m"))
        rows.sort(key=_key)

    headers = ["group", "n"] + [h for _, h, _ in SUMMARY_COLUMNS]
    print(make_table(headers, rows))
    return 0


def _to_float(m):
    if not m:
        return None
    v = m.group(1)
    return float("nan") if v == "nan" else float(v)


def resolve_trace_log(arg, experiments_dir):
    """Accept a log file, a run dir, or an experiment name; return the log path."""
    p = Path(arg)
    if p.is_file():
        return p
    if p.is_dir():
        logs = sorted(p.glob("**/*-log.txt"))
        return logs[-1] if logs else None
    cand = experiments_dir / arg
    if cand.is_dir():
        logs = sorted(cand.glob("**/*-log.txt"))
        return logs[-1] if logs else None
    logs = sorted(experiments_dir.glob(f"**/*{arg}*-log.txt"))
    return logs[-1] if logs else None


def parse_run_trace(log_path):
    """Pull per-epoch diagnostic series out of one run's log."""
    series = {"epoch": []}
    series.update({k: [] for k in TRACE_FIELDS})
    with open(log_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            em = EPOCH_RE.search(line)
            # only real metric lines: have an epoch count AND a numeric g:/μ_l:
            if not em or not (TRACE_FIELDS["g"].search(line) or TRACE_FIELDS["loss"].search(line)):
                continue
            series["epoch"].append(float(em.group(1)))
            for key, rx in TRACE_FIELDS.items():
                series[key].append(_to_float(rx.search(line)))
    return series


def plot_run_trace(series, out_path, title=""):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    x = series["epoch"]
    if len(x) < 2:
        return None

    def clean(vals):
        return [float("nan") if v is None else v for v in vals]

    def has(vals):
        return any(v is not None and v == v for v in vals)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle(title or "run trace")

    ax = axes[0][0]
    ax.plot(x, clean(series["loss"]), color="tab:blue", label="μ_l")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss", color="tab:blue")
    axr = ax.twinx()
    axr.plot(x, clean(series["roc_auc"]), color="tab:green", label="roc_auc")
    axr.set_ylabel("roc_auc", color="tab:green")
    ax.set_title("loss & roc_auc")

    ax = axes[0][1]
    ax.plot(x, clean(series["g"]), color="tab:blue", label="g")
    ax.plot(x, clean(series["g_max"]), color="tab:red", label="g_max")
    ax.set_xlabel("epoch")
    ax.set_ylabel("grad norm")
    ax.legend(loc="upper right", fontsize="small")
    if has(series["clip"]):
        axr = ax.twinx()
        axr.plot(x, clean(series["clip"]), color="tab:orange", linestyle="--", label="clip")
        axr.set_ylabel("clip rate", color="tab:orange")
        axr.set_ylim(0, 1.05)
    ax.set_title("grad norm & clip rate")

    ax = axes[1][0]
    ax.plot(x, clean(series["d_mu"]), color="tab:purple", label="d_mu")
    ax.plot(x, clean(series["d_ad"]), color="tab:brown", label="d_ad")
    ax.set_xlabel("epoch")
    ax.set_ylabel("consecutive drift")
    ax.legend(fontsize="small")
    ax.set_title("per-epoch drift (muon vs adam)")

    ax = axes[1][1]
    ax.plot(x, clean(series["d0_mu"]), color="tab:purple", label="d0_mu")
    ax.plot(x, clean(series["d0_ad"]), color="tab:brown", label="d0_ad")
    ax.set_xlabel("epoch")
    ax.set_ylabel("drift from init")
    ax.legend(fontsize="small")
    ax.set_title("from-init drift (muon vs adam)")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Summarize experiment logs.")
    parser.add_argument("--experiments-dir", "--dir", "-d", default="workdir/experiments")
    for col in COLUMNS:
        parser.add_argument(col["flag"], action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--group-host", action="store_true")
    parser.add_argument("--x-host", action="store_true", help="use hostname on x-axis (default with --plot --group-host)")
    parser.add_argument("--sort", action="store_true", help="sort by total_time ascending")
    parser.add_argument("--short-id", action="store_true", help="show only the short id part")
    parser.add_argument(
        "--setting",
        choices=sorted(SETTINGS_PRESETS.keys()),
        help="use a predefined output setting preset",
    )
    parser.add_argument("--exclude", default="", help="comma-separated experiment ids to exclude")
    parser.add_argument("--summary", action="store_true", help="one-row-per-group summary table")
    parser.add_argument(
        "--trace",
        metavar="RUN",
        help="plot per-epoch diagnostics (loss/gnorm/clip/drift) for a single run: "
        "a log path, a run dir, or an experiment name under --experiments-dir",
    )
    args = parser.parse_args()

    if args.trace:
        log_path = resolve_trace_log(args.trace, Path(args.experiments_dir))
        if not log_path:
            print(f"trace: no log found for {args.trace!r}")
            return 1
        series = parse_run_trace(log_path)
        ts = datetime.now().strftime("%y%m%d-%H%M%S")
        out_path = Path("workdir/plots") / f"{ts}-trace.png"
        saved = plot_run_trace(series, out_path, title=log_path.stem)
        if saved:
            print(f"trace: {saved}  ({len(series['epoch'])} epochs from {log_path})")
            return 0
        print("trace: matplotlib missing or not enough data")
        return 1

    if args.plot and args.group_host and not args.x_host:
        args.x_host = True

    preset = SETTINGS_PRESETS.get(args.setting) if args.setting else None
    requested = [col["key"] for col in COLUMNS if getattr(args, col["attr"])]
    if preset:
        if not requested:
            requested = preset["columns"]
        if preset.get("short_id"):
            args.short_id = True
    elif not requested:
        requested = [col["key"] for col in COLUMNS]

    order_tokens = extract_order(sys.argv[1:])
    columns = unique(order_tokens + [key for key in requested if key not in order_tokens]) if order_tokens else requested

    experiments_dir = Path(args.experiments_dir)
    exclude_ids = {e.strip() for e in args.exclude.split(",") if e.strip()}

    # Detect whether we're at the top-level grouping dir (subdirs contain runs, not logs)
    top_dirs = sorted([p for p in experiments_dir.iterdir() if p.is_dir()]) if experiments_dir.exists() else []
    is_grouped = top_dirs and not any(list(d.glob("*-log.txt")) for d in top_dirs)

    if args.summary and is_grouped:
        return print_summary(top_dirs, args, exclude_ids)

    times_by_host = {}
    if is_grouped:
        # Print each subfolder as its own block
        found_any = False
        for group_dir in top_dirs:
            sub_run_dirs = sorted([p for p in group_dir.iterdir() if p.is_dir()])
            if not sub_run_dirs:
                continue
            has_logs = any(list(d.glob("*-log.txt")) for d in sub_run_dirs)
            if not has_logs:
                continue
            found_any = True
            print()
            tbh = print_group(group_dir, columns, args, exclude_ids)
            for host, ts in tbh.items():
                times_by_host.setdefault(host, []).extend(ts)
        if not found_any:
            print("No experiments found.")
            return 1
    else:
        # Single directory mode (e.g. -d workdir/experiments/normuon-s11/)
        tbh = print_group(experiments_dir, columns, args, exclude_ids)
        times_by_host = tbh

    if args.plot:
        ts = datetime.now().strftime("%y%m%d-%H%M%S")
        suffix = "hostplot" if args.group_host else "expplot"
        out_path = Path("workdir/plots") / f"{ts}-{suffix}.png"
        if args.group_host and args.x_host:
            flat_hosts = []
            flat_times = []
            for host, series in sorted(times_by_host.items()):
                for v in series:
                    if v is not None:
                        flat_hosts.append(host)
                        flat_times.append(v)
            saved = save_plot_grouped({"__scatter__": list(zip(flat_hosts, flat_times))}, out_path)
        elif args.group_host:
            saved = save_plot_grouped(times_by_host, out_path)
        else:
            flat_times = [v for vs in times_by_host.values() for v in vs if v is not None]
            saved = save_plot(flat_times, out_path)
        if saved:
            print(f"plot: {saved}")
        else:
            print("plot: matplotlib missing or not enough data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
