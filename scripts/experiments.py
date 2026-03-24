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

COLUMNS = [
    {
        "key": "experiment_id",
        "flag": "--eid",
        "attr": "eid",
        "header": "experiment_id",
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
        "header": "total_time",
    },
    {
        "key": "total_time_min",
        "flag": "--mins",
        "attr": "mins",
        "header": "in_mins",
    },
    {
        "key": "roc_auc",
        "flag": "--roc-auc",
        "attr": "roc_auc",
        "header": "roc_auc",
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
        "header": "μ_epoch_t",
    },
    {
        "key": "datasets",
        "flag": "--datasets",
        "attr": "datasets",
        "header": "datasets",
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
    }
}


def parse_log(log_path):
    hostname = None
    total_time = None
    roc_auc = None
    mean_epoch_time = None
    epoch = None
    datasets = None
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
        return None, None, None, None, None, None

    if total_time is None:
        total_time = last_t_t

    if total_time is None and first_clock is not None and last_clock is not None:
        if last_clock < first_clock:
            last_clock += 24 * 3600
        total_time = float(last_clock - first_clock)

    return hostname, total_time, roc_auc, mean_epoch_time, epoch, datasets


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


def column_values(exp_id, hostname, total_time, roc_auc, mean_epoch_time, epoch, datasets):
    total_time_min = "-" if total_time is None else f"{total_time / 60:.2f}m"
    return {
        "experiment_id": exp_id,
        "hostname": hostname or "-",
        "total_time": "-" if total_time is None else f"{total_time:.2f}s",
        "total_time_min": total_time_min,
        "roc_auc": "-" if roc_auc is None else f"{roc_auc:.6f}",
        "epoch": "-" if epoch is None else str(epoch),
        "mean_epoch_time": "-" if mean_epoch_time is None else f"{mean_epoch_time:.2f}s",
        "datasets": "-" if datasets is None else str(datasets),
    }


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
    args = parser.parse_args()

    if args.plot and args.group_host and not args.x_host:
        args.x_host = True

    preset = SETTINGS_PRESETS.get(args.setting) if args.setting else None
    requested = [col["key"] for col in COLUMNS if getattr(args, col["attr"])]
    if preset:
        # Explicit flags still win, otherwise use preset columns.
        if not requested:
            requested = preset["columns"]
        if preset.get("short_id"):
            args.short_id = True
    elif not requested:
        requested = [col["key"] for col in COLUMNS]

    order_tokens = extract_order(sys.argv[1:])
    columns = unique(order_tokens + [key for key in requested if key not in order_tokens]) if order_tokens else requested

    experiments_dir = Path(args.experiments_dir)
    rows = []
    times = []
    times_by_host = {}
    stats_by_col = {}

    exclude_ids = {e.strip() for e in args.exclude.split(",") if e.strip()}

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
        if log_path is not None:
            hostname, total_time, roc_auc, mean_epoch_time, epoch, datasets = parse_log(log_path)

        display_exp_id = short_experiment_id(exp_id) if args.short_id else exp_id
        values = column_values(display_exp_id, hostname, total_time, roc_auc, mean_epoch_time, epoch, datasets)
        rows.append((total_time, [str(idx)] + [values[key] for key in columns]))
        times.append(total_time)

        # collect per-column numeric stats for displayed columns
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
        print("No experiments found.")
        return 1

    # compute per-column statistics for numeric columns that were actually displayed
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
        # Recompute column widths (same logic as `make_table`) so stats lines line up
        # with the table columns.
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
            if col_key == "epoch" or col_key == "datasets":
                return f"{int(round(value))}"
            if col_key == "mean_epoch_time":
                return f"{value:.2f}s"
            return ""

        offset = 2 if args.sort else 1

        # Find where the first calculable/stat column appears in the printed table.
        # Everything before this index is non-calculated (e.g. experiment id / hostname),
        # so we "span" the mean/std/median label across it to keep alignment stable.
        stat_col_indices = []
        for header_idx in range(offset, len(headers)):
            col_key = columns[header_idx - offset]
            if col_key in stats_summary:
                stat_col_indices.append(header_idx)
        first_calc_idx = min(stat_col_indices) if stat_col_indices else len(headers)

        # The table uses "  " (two spaces) between columns.
        # For stats *values*, we must pad up to the exact start position of
        # `first_calc_idx` in the table line.
        value_prefix_width = sum(widths[:first_calc_idx]) + 2 * first_calc_idx

        # For the separator dashes, we want the continuous run to stop right before
        # the first calculated column's dash block. That means we fill the inter-
        # prefix column gaps with dashes (2 chars each), but do NOT include the final
        # "  " gap right before `first_calc_idx`.
        dash_prefix_width = sum(widths[:first_calc_idx]) + 2 * max(first_calc_idx - 1, 0)

        for stat_type in ("mean", "std", "median"):
            # Pad the label so that the first calculated value begins at the same
            # x-position as in the table.
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

            line = prefix + "  ".join(rest_cells)
            print(line)

        # Separator line between stats and the table: continuous dashes up to the first
        # calculable column (matches your request), then use the normal column spacing.
        if first_calc_idx < len(headers):
            prefix_dashes = "-" * dash_prefix_width
            rest_dashes = "  ".join("-" * widths[i] for i in range(first_calc_idx, len(headers)))
            # Insert the normal "  " gap between the non-calculated prefix and the first
            # calculated column's dash block.
            print(prefix_dashes + "  " + rest_dashes)

    print(make_table(headers, rows))

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
            saved = save_plot(times, out_path)
        if saved:
            print(f"plot: {saved}")
        else:
            print("plot: matplotlib missing or not enough data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
