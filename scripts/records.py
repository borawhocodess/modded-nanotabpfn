import argparse
import bisect
import fnmatch
import re
import statistics
from pathlib import Path

TT_RE = re.compile(r"\bt_t:([0-9]+(?:\.[0-9]+)?)s\b")
TOTAL_RE = re.compile(r"\b(total time|total_time|elapsed|duration)\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)s\b", re.IGNORECASE)
TIME_BRACKET_RE = re.compile(r"\[(\d{2}):(\d{2}):(\d{2})\]")
README_MEAN_RE = re.compile(r"\bmean\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*s\b", re.IGNORECASE)
VAL_RE = re.compile(r"\bavg_roc_auc[:=]\s*([0-9]+(?:\.[0-9]+)?)\b")
EPOCH_RE = re.compile(r"\be:(\d+)/\d+\b")
STEPS_RE = re.compile(r"\((\d+)-")
CONFIG_INT_RE = re.compile(r"^\s*(batch_size|steps):(?:\s*int\s*=\s*|\s*)(\d+)\s*$")
LOSS_RE = re.compile(r"\bμ_l:([0-9]+(?:\.[0-9]+)?)\b")
TABPFN_REF_LINE_SPECS = [
    {
        "id": "tabpfn_v2",
        "arg_prefix": "tabpfn-v2",
        "default_label": "TabPFNv2 baseline",
        "default_y": 0.8301771427689452,
        "default_alpha": 0.6,
    },
    {
        "id": "tabpfn_v2_nopre",
        "arg_prefix": "tabpfn-v2-nopre",
        "default_label": "TabPFNv2 nopre baseline",
        "default_y": 0.8307422813464695,
        "default_alpha": 0.4,
    },
    {
        "id": "tabpfn_v2_5",
        "arg_prefix": "tabpfn-v2-5",
        "default_label": "TabPFNv2.5 baseline",
        "default_y": 0.8320368014767934,
        "default_alpha": 1.0,
    },
    {
        "id": "tabpfn_v2_5_nopre",
        "arg_prefix": "tabpfn-v2-5-nopre",
        "default_label": "TabPFNv2.5 nopre baseline",
        "default_y": 0.8285797522680965,
        "default_alpha": 0.8,
    },
]


def parse_log_total_time(log_path):
    total_time = None
    last_t_t = None
    first_clock = None
    last_clock = None

    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
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
        return None

    if total_time is None:
        total_time = last_t_t

    if total_time is None and first_clock is not None and last_clock is not None:
        if last_clock < first_clock:
            last_clock += 24 * 3600
        total_time = float(last_clock - first_clock)

    return total_time


def parse_readme_mean(readme_path):
    if not readme_path.exists():
        return None
    try:
        content = readme_path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return None
    match = README_MEAN_RE.search(content)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def parse_metric_series(log_path, x_mode, metric_re):
    series = []
    if log_path is None:
        return series

    first_clock = None
    last_seen_clock = None
    current_clock = None
    offset = 0
    last_t_t = None
    last_epoch = None
    steps = None
    batch_size = None
    pending_dataset_points = []

    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                match = CONFIG_INT_RE.match(line)
                if match:
                    key = match.group(1)
                    try:
                        value = int(match.group(2))
                    except ValueError:
                        value = None
                    if value is not None:
                        if key == "batch_size":
                            batch_size = value
                        elif key == "steps":
                            steps = value
                    if x_mode == "datasets" and steps is not None and pending_dataset_points:
                        bs = batch_size if batch_size is not None else 1
                        for epoch, val in pending_dataset_points:
                            series.append((epoch * bs * steps, val))
                        pending_dataset_points = []

                match = EPOCH_RE.search(line)
                if match:
                    try:
                        last_epoch = int(match.group(1))
                    except ValueError:
                        pass

                if steps is None:
                    match = STEPS_RE.search(line)
                    if match:
                        try:
                            steps = int(match.group(1))
                        except ValueError:
                            pass

                match = TIME_BRACKET_RE.search(line)
                if match:
                    h = int(match.group(1))
                    m = int(match.group(2))
                    s = int(match.group(3))
                    clock = h * 3600 + m * 60 + s
                    if last_seen_clock is not None and clock < last_seen_clock:
                        offset += 24 * 3600
                    last_seen_clock = clock
                    if first_clock is None:
                        first_clock = clock
                    current_clock = clock + offset

                match = TT_RE.search(line)
                if match:
                    try:
                        last_t_t = float(match.group(1))
                    except ValueError:
                        pass

                match = metric_re.search(line)
                if match:
                    try:
                        val = float(match.group(1))
                    except ValueError:
                        continue
                    x_value = None
                    if x_mode == "datasets":
                        if last_epoch is not None:
                            if steps is not None:
                                bs = batch_size if batch_size is not None else 1
                                x_value = last_epoch * bs * steps
                            else:
                                pending_dataset_points.append((last_epoch, val))
                    elif x_mode == "wallclock":
                        if current_clock is not None and first_clock is not None:
                            x_value = current_clock - first_clock
                    else:
                        if last_t_t is not None:
                            x_value = last_t_t
                        elif current_clock is not None and first_clock is not None:
                            x_value = current_clock - first_clock
                    if x_value is not None:
                        series.append((x_value, val))
    except FileNotFoundError:
        return series

    return series


def pick_log_file(record_dir):
    logs = pick_log_files(record_dir)
    return max(logs, key=lambda p: p.stat().st_mtime) if logs else None


def pick_log_files(record_dir):
    logs = list(record_dir.glob("*-log.txt"))
    return sorted(logs, key=lambda p: p.stat().st_mtime)


def iter_records(records_dir):
    if not records_dir.exists():
        return []
    return sorted([p for p in records_dir.iterdir() if p.is_dir()])


def parse_name_patterns(pattern_args):
    patterns = []
    for value in pattern_args or []:
        for part in value.split(","):
            part = part.strip()
            if part:
                patterns.append(part)
    return patterns


def name_matches_patterns(name, patterns):
    name_l = name.lower()
    for pattern in patterns:
        pattern_l = pattern.lower()
        if fnmatch.fnmatch(name_l, pattern_l) or pattern_l in name_l:
            return True
    return False


def filter_record_dirs(record_dirs, only_patterns, exclude_patterns):
    out = []
    for record_dir in record_dirs:
        name = record_dir.name
        if only_patterns and not name_matches_patterns(name, only_patterns):
            continue
        if exclude_patterns and name_matches_patterns(name, exclude_patterns):
            continue
        out.append(record_dir)
    return out


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


def format_total_time(total_time):
    return "-" if total_time is None else f"{total_time:.2f}s"


def format_ms(total_time):
    return "-" if total_time is None else f"{total_time * 1000:.0f}ms"


def format_mins(total_time):
    return "-" if total_time is None else f"{total_time / 60:.2f}m"


def format_improvement(current, reference):
    if current is None or reference is None:
        return "-"
    delta = (reference - current) / reference * 100.0
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.2f}%"


def smooth_series(values, window):
    if window <= 1:
        return values
    out = []
    acc = 0.0
    for idx, v in enumerate(values):
        acc += v
        if idx >= window:
            acc -= values[idx - window]
            out.append(acc / window)
        else:
            out.append(acc / (idx + 1))
    return out


def parse_figsize(value):
    text = value.strip().lower().replace("x", ",")
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("figsize must be in format W,H (e.g. 14,8)")
    try:
        width = float(parts[0])
        height = float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("figsize values must be numbers") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("figsize values must be > 0")
    return (width, height)


def parse_fontsize(value):
    text = value.strip()
    if not text:
        raise argparse.ArgumentTypeError("font size cannot be empty")
    try:
        size = float(text)
    except ValueError:
        return text
    if size <= 0:
        raise argparse.ArgumentTypeError("font size must be > 0")
    return size


def smooth_series_points(series, window):
    xs = [x for x, _ in series]
    ys = smooth_series([y for _, y in series], window)
    return list(zip(xs, ys))


def interpolate_y_at_x(series, x):
    if not series:
        return None
    xs = [pt[0] for pt in series]
    ys = [pt[1] for pt in series]
    if x < xs[0] or x > xs[-1]:
        return None
    idx = bisect.bisect_left(xs, x)
    if idx < len(xs) and xs[idx] == x:
        return ys[idx]
    if idx == 0 or idx >= len(xs):
        return None
    x0, y0 = xs[idx - 1], ys[idx - 1]
    x1, y1 = xs[idx], ys[idx]
    if x1 == x0:
        return y1
    ratio = (x - x0) / (x1 - x0)
    return y0 + ratio * (y1 - y0)


def aggregate_series_mean_std(series_list):
    prepared = [sorted(series, key=lambda p: p[0]) for series in series_list if series]
    if not prepared:
        return []
    x_grid = sorted({x for series in prepared for x, _ in series})
    out = []
    for x in x_grid:
        vals = []
        for series in prepared:
            y = interpolate_y_at_x(series, x)
            if y is not None:
                vals.append(y)
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        std = statistics.stdev(vals) if len(vals) >= 2 else 0.0
        out.append((x, mean, std, len(vals)))
    return out


def add_tabpfn_ref_line_args(parser):
    parser.add_argument(
        "--tabpfn-ref-lines",
        action="store_true",
        default=False,
        help="draw all predefined TabPFN reference lines (valplot)",
    )
    parser.add_argument(
        "--tabpfn-ref-line-color",
        default="black",
        help="default color for predefined TabPFN reference lines",
    )
    parser.add_argument(
        "--tabpfn-ref-line-style",
        default="-",
        help="default linestyle for predefined TabPFN reference lines",
    )
    parser.add_argument(
        "--tabpfn-ref-line-width",
        type=float,
        default=1.0,
        help="default line width for predefined TabPFN reference lines",
    )
    for spec in TABPFN_REF_LINE_SPECS:
        line_name = spec["default_label"]
        arg_prefix = spec["arg_prefix"]
        parser.add_argument(
            f"--{arg_prefix}-line",
            action="store_true",
            default=False,
            help=f"draw {line_name} reference line",
        )
        parser.add_argument(
            f"--{arg_prefix}-line-y",
            type=float,
            default=spec["default_y"],
            help=f"y value for {line_name} reference line",
        )
        parser.add_argument(
            f"--{arg_prefix}-line-label",
            default=spec["default_label"],
            help=f"label for {line_name} reference line",
        )
        parser.add_argument(
            f"--{arg_prefix}-line-color",
            default=None,
            help=f"color override for {line_name} reference line",
        )
        parser.add_argument(
            f"--{arg_prefix}-line-style",
            default=None,
            help=f"linestyle override for {line_name} reference line",
        )
        parser.add_argument(
            f"--{arg_prefix}-line-width",
            type=float,
            default=None,
            help=f"line width override for {line_name} reference line",
        )
        parser.add_argument(
            f"--{arg_prefix}-line-alpha",
            type=float,
            default=spec["default_alpha"],
            help=f"alpha for {line_name} reference line (0-1)",
        )


def build_tabpfn_ref_lines(args):
    lines = []
    for spec in TABPFN_REF_LINE_SPECS:
        line_id = spec["id"]
        if not (args.tabpfn_ref_lines or getattr(args, f"{line_id}_line")):
            continue
        color_override = getattr(args, f"{line_id}_line_color")
        style_override = getattr(args, f"{line_id}_line_style")
        width_override = getattr(args, f"{line_id}_line_width")
        lines.append(
            {
                "label": getattr(args, f"{line_id}_line_label"),
                "y": getattr(args, f"{line_id}_line_y"),
                "color": color_override if color_override is not None else args.tabpfn_ref_line_color,
                "style": style_override if style_override is not None else args.tabpfn_ref_line_style,
                "width": width_override if width_override is not None else args.tabpfn_ref_line_width,
                "alpha": getattr(args, f"{line_id}_line_alpha"),
            }
        )
    return lines


def save_metric_plot(
    series_by_record,
    out_path,
    x_mode,
    ylabel,
    ma_window,
    point_size,
    line_width,
    y_min,
    figsize,
    legend_size,
    tick_size,
    hline_y,
    hline_label,
    hline_color,
    hline_style,
    hline_width,
    extra_hlines,
    run_alpha,
    std_alpha,
    multi_run_color,
    multi_mean_color,
    multi_std_color,
):
    series_by_record = {k: v for k, v in series_by_record.items() if v}
    if not series_by_record:
        return None
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    plt.figure(figsize=figsize)
    for idx, (record_name, series_list) in enumerate(sorted(series_by_record.items())):
        smoothed_runs = [smooth_series_points(series, ma_window) for series in series_list if series]
        if not smoothed_runs:
            continue
        base_color = f"C{idx % 10}"
        if len(smoothed_runs) == 1:
            series = smoothed_runs[0]
            if x_mode == "datasets":
                xs = [t for t, _ in series]
            else:
                xs = [t / 60 for t, _ in series]
            ys = [v for _, v in series]
            if len(series) == 1:
                plt.scatter(xs, ys, s=point_size, color=base_color, label=record_name)
            else:
                plt.plot(
                    xs,
                    ys,
                    marker="o",
                    markersize=max(1.0, point_size / 6.0),
                    linewidth=line_width,
                    color=base_color,
                    label=record_name,
                )
            continue

        run_color = multi_run_color if multi_run_color is not None else base_color
        mean_color = multi_mean_color if multi_mean_color is not None else base_color
        std_color = multi_std_color if multi_std_color is not None else mean_color

        for run_series in smoothed_runs:
            if x_mode == "datasets":
                xs_run = [t for t, _ in run_series]
            else:
                xs_run = [t / 60 for t, _ in run_series]
            ys_run = [v for _, v in run_series]
            if len(run_series) == 1:
                plt.scatter(xs_run, ys_run, s=max(1.0, point_size / 2.0), color=run_color, alpha=run_alpha)
            else:
                plt.plot(
                    xs_run,
                    ys_run,
                    linewidth=max(0.5, line_width),
                    color=run_color,
                    alpha=run_alpha,
                )

        aggregated = aggregate_series_mean_std(smoothed_runs)
        if not aggregated:
            continue
        xs_agg = [x for x, _, _, _ in aggregated]
        if x_mode != "datasets":
            xs_agg = [x / 60 for x in xs_agg]
        ys_mean = [mean for _, mean, _, _ in aggregated]
        ys_low = [mean - std for _, mean, std, _ in aggregated]
        ys_high = [mean + std for _, mean, std, _ in aggregated]
        n_runs = len(smoothed_runs)
        label = f"{record_name} mean±std (n={n_runs})"
        if len(xs_agg) == 1:
            plt.scatter(xs_agg, ys_mean, s=point_size, color=mean_color, label=label)
        else:
            plt.plot(
                xs_agg,
                ys_mean,
                linewidth=line_width + 1.0,
                color=mean_color,
                label=label,
            )
            plt.fill_between(xs_agg, ys_low, ys_high, color=std_color, alpha=std_alpha)
    if hline_y is not None:
        plt.axhline(
            y=hline_y,
            color=hline_color,
            linestyle=hline_style,
            linewidth=hline_width,
            label=hline_label,
        )
    for extra_hline in extra_hlines or []:
        plt.axhline(
            y=extra_hline["y"],
            color=extra_hline["color"],
            linestyle=extra_hline["style"],
            linewidth=extra_hline["width"],
            alpha=extra_hline.get("alpha", 1.0),
            label=extra_hline["label"],
        )
    if x_mode == "datasets":
        plt.xlabel("datasets_seen")
    elif x_mode == "wallclock":
        plt.xlabel("wallclock (mins)")
    else:
        plt.xlabel("time (mins)")
    plt.ylabel(ylabel)
    if tick_size is not None:
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
    if y_min is not None:
        plt.ylim(bottom=y_min)
    if len(series_by_record) > 1 or hline_y is not None or extra_hlines:
        plt.legend(loc="best", fontsize=legend_size)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    return out_path


def resolve_record_time(record_dir, use_readme_mean):
    if use_readme_mean:
        readme_mean = parse_readme_mean(record_dir / "README.md")
        if readme_mean is not None:
            return readme_mean
    log_path = pick_log_file(record_dir)
    if log_path is None:
        return None
    return parse_log_total_time(log_path)


def choose_baseline_name(record_names, requested_baseline):
    if requested_baseline and requested_baseline in record_names:
        return requested_baseline
    for name in record_names:
        if "baseline" in name.lower():
            return name
    return record_names[0] if record_names else None


def main():
    parser = argparse.ArgumentParser(description="Summarize record performance.")
    parser.add_argument("--records-dir", default="records")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="include only matching record names (repeatable, comma-separated, supports glob patterns)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="exclude matching record names (repeatable, comma-separated, supports glob patterns)",
    )
    parser.add_argument("--baseline", default="", help="record name to use as baseline")
    parser.add_argument("--valplot", action="store_true", default=True, help="plot validation scores vs wallclock")
    parser.add_argument("--no-valplot", action="store_false", dest="valplot", help="disable validation plot")
    parser.add_argument("--lossplot", action="store_true", help="plot training loss vs x-axis")
    parser.add_argument(
        "--valplot-x",
        choices=["total_time", "wallclock", "datasets"],
        default="datasets",
        help="x axis for valplot",
    )
    parser.add_argument(
        "--lossplot-x",
        choices=["total_time", "wallclock", "datasets"],
        default="total_time",
        help="x axis for lossplot",
    )
    parser.add_argument("--ma", type=int, default=1, help="moving average window for plots")
    parser.add_argument("--point-size", type=float, default=10.0, help="marker size for plots")
    parser.add_argument("--line-width", type=float, default=1.0, help="line width for plots")
    parser.add_argument("--start-y-axis-at", type=float, default=0.72, help="y-axis lower bound")
    parser.add_argument(
        "--figsize",
        type=parse_figsize,
        default=(18.0, 9.0),
        metavar="W,H",
        help="plot size in inches, e.g. 14,8 or 14x8",
    )
    parser.add_argument(
        "--legend-size",
        type=parse_fontsize,
        default=18.0,
        help="legend font size (e.g. small, medium, 12)",
    )
    parser.add_argument(
        "--tick-size",
        type=float,
        default=14.0,
        help="x/y tick label font size (e.g. 12); default keeps matplotlib setting",
    )
    parser.add_argument(
        "--rf-line-y",
        type=float,
        default=0.8068462330697953,
        help="draw horizontal Random Forest reference line at this y value (valplot)",
    )
    parser.add_argument(
        "--rf-line-label",
        default="RF baseline",
        help="label for Random Forest reference line",
    )
    parser.add_argument(
        "--rf-line-color",
        default="gray",
        help="color for Random Forest reference line",
    )
    parser.add_argument(
        "--rf-line-style",
        default="--",
        help="linestyle for Random Forest reference line (e.g. --, -, :, -.)",
    )
    parser.add_argument(
        "--rf-line-width",
        type=float,
        default=2.0,
        help="line width for Random Forest reference line",
    )
    parser.add_argument(
        "--multi-log-alpha",
        type=float,
        default=0.18,
        help="alpha for individual runs when a record has multiple logs",
    )
    parser.add_argument(
        "--multi-log-std-alpha",
        type=float,
        default=0.14,
        help="alpha for mean±std shaded band when a record has multiple logs",
    )
    parser.add_argument(
        "--multi-log-run-color",
        default="0.75",
        help="color for individual runs when a record has multiple logs",
    )
    parser.add_argument(
        "--multi-log-mean-color",
        default=None,
        help="color for mean line with multiple logs (default: per-record color cycle)",
    )
    parser.add_argument(
        "--multi-log-std-color",
        default=None,
        help="color for std band with multiple logs (default: mean color)",
    )
    add_tabpfn_ref_line_args(parser)
    args = parser.parse_args()

    records_dir = Path(args.records_dir)
    record_dirs = iter_records(records_dir)
    only_patterns = parse_name_patterns(args.only)
    exclude_patterns = parse_name_patterns(args.exclude)
    record_dirs = filter_record_dirs(record_dirs, only_patterns, exclude_patterns)
    if not record_dirs:
        print("No records found.")
        return 1

    record_names = [record_dir.name for record_dir in record_dirs]
    baseline_name = choose_baseline_name(record_names, args.baseline)
    records = [
        (record_dir.name, resolve_record_time(record_dir, record_dir.name == baseline_name))
        for record_dir in record_dirs
    ]
    val_series_by_record = {}
    if args.valplot:
        for record_dir in record_dirs:
            series_list = []
            for log_path in pick_log_files(record_dir):
                series = parse_metric_series(log_path, args.valplot_x, VAL_RE)
                if series:
                    series_list.append(series)
            if series_list:
                val_series_by_record[record_dir.name] = series_list
    loss_series_by_record = {}
    if args.lossplot:
        for record_dir in record_dirs:
            series_list = []
            for log_path in pick_log_files(record_dir):
                series = parse_metric_series(log_path, args.lossplot_x, LOSS_RE)
                if series:
                    series_list.append(series)
            if series_list:
                loss_series_by_record[record_dir.name] = series_list

    baseline_time = None
    for name, total_time in records:
        if name == baseline_name:
            baseline_time = total_time
            break

    headers = ["#", "record_name", "total_time", "in_mins", "impr_vs_prev", "impr_vs_baseline"]
    rows = []
    prev_time = None
    for idx, (name, total_time) in enumerate(records, start=1):
        row = [
            str(idx),
            name,
            format_total_time(total_time),
            format_mins(total_time),
            format_improvement(total_time, prev_time),
            "-" if name == baseline_name else format_improvement(total_time, baseline_time),
        ]
        rows.append(row)
        prev_time = total_time

    print(make_table(headers, rows))
    if args.valplot:
        extra_hlines = build_tabpfn_ref_lines(args)
        ts = __import__("datetime").datetime.now().strftime("%y%m%d-%H%M%S")
        x_tag = "time" if args.valplot_x == "total_time" else args.valplot_x
        out_path = Path("workdir/plots") / f"{ts}-records-plot-x-{x_tag}-y-val.png"
        saved = save_metric_plot(
            val_series_by_record,
            out_path,
            args.valplot_x,
            "avg_roc_auc",
            args.ma,
            args.point_size,
            args.line_width,
            args.start_y_axis_at,
            args.figsize,
            args.legend_size,
            args.tick_size,
            args.rf_line_y,
            args.rf_line_label,
            args.rf_line_color,
            args.rf_line_style,
            args.rf_line_width,
            extra_hlines,
            args.multi_log_alpha,
            args.multi_log_std_alpha,
            args.multi_log_run_color,
            args.multi_log_mean_color,
            args.multi_log_std_color,
        )
        if saved:
            print(f"valplot: {saved}")
        elif val_series_by_record:
            print("valplot: matplotlib missing")
        else:
            print("valplot: no validation data found")
    if args.lossplot:
        ts = __import__("datetime").datetime.now().strftime("%y%m%d-%H%M%S")
        x_tag = "time" if args.lossplot_x == "total_time" else args.lossplot_x
        out_path = Path("workdir/plots") / f"{ts}-records-plot-x-{x_tag}-y-loss.png"
        saved = save_metric_plot(
            loss_series_by_record,
            out_path,
            args.lossplot_x,
            "train_loss",
            args.ma,
            args.point_size,
            args.line_width,
            args.start_y_axis_at,
            args.figsize,
            args.legend_size,
            args.tick_size,
            None,
            args.rf_line_label,
            args.rf_line_color,
            args.rf_line_style,
            args.rf_line_width,
            [],
            args.multi_log_alpha,
            args.multi_log_std_alpha,
            args.multi_log_run_color,
            args.multi_log_mean_color,
            args.multi_log_std_color,
        )
        if saved:
            print(f"lossplot: {saved}")
        elif loss_series_by_record:
            print("lossplot: matplotlib missing")
        else:
            print("lossplot: no loss data found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
