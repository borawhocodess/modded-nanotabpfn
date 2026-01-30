import argparse
import re
import sys
from pathlib import Path

HOST_RE = re.compile(r"^\s*host:\s*(.+?)\s*$")
TT_RE = re.compile(r"\bt_t:([0-9]+(?:\.[0-9]+)?)s\b")
TOTAL_RE = re.compile(r"\b(total time|total_time|elapsed|duration)\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)s\b", re.IGNORECASE)
TIME_BRACKET_RE = re.compile(r"\[(\d{2}):(\d{2}):(\d{2})\]")

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
        "key": "status",
        "flag": "--status",
        "attr": "status",
        "header": "status",
    },
]

COLUMN_BY_KEY = {col["key"]: col for col in COLUMNS}
FLAG_TO_KEY = {col["flag"]: col["key"] for col in COLUMNS}


def parse_log(log_path):
    hostname = None
    total_time = None
    last_t_t = None
    first_clock = None
    last_clock = None
    completed = False

    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip().startswith("experiment done:"):
                    completed = True

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
        return None, None

    if total_time is None:
        total_time = last_t_t

    if total_time is None and first_clock is not None and last_clock is not None:
        if last_clock < first_clock:
            last_clock += 24 * 3600
        total_time = float(last_clock - first_clock)

    return hostname, total_time, completed


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
        if token == "--experiments-dir":
            skip_next = True
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


def column_values(exp_id, hostname, total_time, completed):
    total_time_min = "-" if total_time is None else f"{total_time / 60:.2f}m"
    return {
        "experiment_id": exp_id,
        "hostname": hostname or "-",
        "total_time": "-" if total_time is None else f"{total_time:.2f}s",
        "total_time_min": total_time_min,
        "status": "completed" if completed else "uncompleted",
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize experiment logs.")
    parser.add_argument("--experiments-dir", default="workdir/experiments")
    for col in COLUMNS:
        parser.add_argument(col["flag"], action="store_true")
    args = parser.parse_args()

    requested = [col["key"] for col in COLUMNS if getattr(args, col["attr"])]
    if not requested:
        requested = [col["key"] for col in COLUMNS]

    order_tokens = extract_order(sys.argv[1:])
    columns = unique(order_tokens + [key for key in requested if key not in order_tokens]) if order_tokens else requested

    experiments_dir = Path(args.experiments_dir)
    rows = []

    for idx, exp_dir in enumerate(iter_experiments(experiments_dir), start=1):
        exp_id = exp_dir.name
        log_path = pick_log_file(exp_dir)
        hostname = None
        total_time = None
        completed = False
        if log_path is not None:
            hostname, total_time, completed = parse_log(log_path)

        values = column_values(exp_id, hostname, total_time, completed)
        row = [str(idx)] + [values[key] for key in columns]
        rows.append(row)

    headers = ["#"] + [COLUMN_BY_KEY[key]["header"] for key in columns]

    if not rows:
        print("No experiments found.")
        return 1

    print(make_table(headers, rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
