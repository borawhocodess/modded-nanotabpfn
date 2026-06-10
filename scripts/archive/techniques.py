"""
Inspect results from technique sweeps.

Usage:
    uv run python scripts/techniques.py
    uv run python scripts/techniques.py --batch v26
    uv run python scripts/techniques.py --batch tech6
    uv run python scripts/techniques.py --sort auc
    uv run python scripts/techniques.py --all-runs
    uv run python scripts/techniques.py --dir workdir/experiments --seed 11
"""

import argparse
import re
import sys
from pathlib import Path
from statistics import mean, median

# ---------------------------------------------------------------------------
# Technique metadata (same order as sweep-techniques.sh / techniques.md)
# ---------------------------------------------------------------------------

TECHNIQUES = [
    ("staticcompile",   "P7",      "torch.compile(dynamic=False)"),
    ("layerscale",      "R2",      "per-layer learned residual scale"),
    ("normuon",         "O4",      "NorMuon (7-step NS + col norm)"),
    ("muonmom",         "O6",      "Muon momentum warmup/cooldown"),
    ("fp8",             "P4",      "FP8 matmul"),
    ("unet",            "R3",      "U-Net skip connections"),
    ("valueemb",        "A4",      "value embeddings"),
    ("gqa",             "A5",      "grouped query attention"),
    ("cautious",        "O5",      "cautious weight decay"),
    ("zeroinit",        "I2",      "zero-init output projections"),
    ("cooldown",        "H4",      "explicit LR cooldown"),
    ("ema",             "H6",      "EMA parameter averaging"),
    ("orthoinit",       "I3/I4",   "orthogonal initialization"),
    ("softcap",         "A6",      "softcap attention logits"),
    ("lrelu2",          "M4",      "LeakyReLU² activation"),
    ("x0mix",           "R4",      "residual mix with x0"),
    ("fp32weights",     "P5/P6/P8","FP32 master weights"),
    ("adafactormuon",   "O7",      "Adafactor-style factored momentum"),
    ("pairedhead",      "A7",      "paired head attention"),
    ("fusedadam",       "O8",      "fused AdamW"),
]

# ---------------------------------------------------------------------------
# v26 technique batch (v26techniques2.sh, ThinkingRows+RMSNorm base)
# ---------------------------------------------------------------------------

TECHNIQUES_V26 = [
    ("v26thinkrows32",   "T1", "ThinkingRows x32 (vs 16 baseline)"),
    ("v26thinkrows64",   "T2", "ThinkingRows x64"),
    ("v26thinkrowstest", "T3", "ThinkingRows test-side only"),
    ("v26gatedattn",     "A1", "per-head sigmoid gate on attn output"),
    ("v26qgain",         "A2", "learnable per-head Q scale"),
    ("v26vrl",           "A3", "value residual cross-layer"),
    ("v26xsa",           "A4", "subtract V-parallel from attn output"),
    ("v26dtg",           "R1", "DTG per-block bypass gate"),
    ("v26residmix",      "R2", "per-block learned x/x0 mix"),
    ("v26siglu",         "M1", "SwiGLU MLP"),
    ("v26mlp4x",         "M2", "MLP 4x wider (h=1024)"),
    ("v26muonbeta2",     "O1", "AdamW beta2=0.9"),
    ("v26adamwd",        "O2", "AdamW weight_decay=0.01"),
    ("v26muonwd",        "O3", "Muon weight_decay=0.04"),
    ("v26lawa",          "O4", "LAWA latest weight averaging"),
    ("v26labelsmooth",   "L1", "label smoothing 0.1"),
    ("v26l8",            "C1", "l=8 layers"),
    ("v26e512",          "C2", "e=512 wider model"),
]

# Baseline: ThinkingRows+RMSNorm (rmsthink-s11, 23-run median)
BASELINE_V26_MINS = 3.88

# ---------------------------------------------------------------------------
# v27 technique batch (techniques3.sh, LAWA freq=1 + AdamW WD=0.01 base)
# ---------------------------------------------------------------------------

TECHNIQUES_V27 = [
    ("muoneqr",        "O1", "row-normalized MuonEq-R style update"),
    ("recur23late",    "R1", "delayed middle-layer recurrence, layers 2/3"),
    ("recur34late",    "R1", "delayed middle-layer recurrence, layers 3/4"),
    ("recur23prog",    "R2", "progressive recurrence (phase 1→2)"),
    ("parallel45",     "R3", "late parallel residuals, last 2 blocks"),
    ("parallel345",    "R3", "late parallel residuals, last 3 blocks"),
    ("qgainfix1",      "A1", "vectorized Q gain, init 1.0"),
    ("qgainfix4",      "A1", "vectorized Q gain, init 4.0"),
    ("siglulawa",      "M1", "SiGLU/SwiGLU on LAWA base"),
    ("think32lawa",    "T1", "ThinkingRows x32 on LAWA base"),
    ("lawa_microgrid", "O2", "LAWA/WD microgrid, default point"),
    ("curriculum",     "S1", "ramp prior steps per epoch"),
    ("featuregroup",   "I1", "NanoTabICL-style feature grouping"),
    ("rowcls",         "I2", "NanoTabICL-style row CLS decoder column"),
    ("inducedcol",     "I3", "induced feature/column attention"),
    ("qassmax",        "A2", "QASSMax-style row query scaling"),
    ("cautiousadamwd", "O3", "cautious Adam weight decay approximation"),
    ("muonwarmup",     "O4", "Muon momentum warmup-only"),
]

# Baseline: LAWA freq=1 + AdamW WD=0.01 (20260402lawa1wd, mean over runs)
BASELINE_V27_MINS = 3.48

# ---------------------------------------------------------------------------
# v28 technique batch (train_nano_automod_*.py, automod20.sh, techniques5.md)
# Base script: train_nano_automod_.py (l=5, fgs=5, rows=24, mean-pool-nolast,
# ns=5 default, Muon on transformer_encoder, ScheduleFree Adam warmup=1000,
# LAWA k=10 freq=1). Each variant applies a single candidate on top.
# ---------------------------------------------------------------------------

TECHNIQUES_V28 = [
    ("automod_k1_fusedmuon",      "K1", "fused Muon: Polar Express + NorMuon + cautious WD"),
    ("automod_k2_cautiouswd",     "K2", "cautious WD mask on Muon step"),
    ("automod_a1_lambdas",        "A1", "per-layer resid_lambdas (1.15→1.05) + x0_lambdas (0.20→0.05)"),
    ("automod_a2_backout",        "A2", "mid-layer residual backout (l//2 cache, subtract)"),
    ("automod_a3_hyperconn",      "A3", "saved mid-layer activation hyperconnection"),
    ("automod_a4_softcap",        "A4", "decoder tanh softcap at 15.0"),
    ("automod_o1_normuon",        "O1", "NorMuon factored per-row/col variance post-NS"),
    ("automod_o2_muoneqr",        "O2", "MuonEq-R row normalization pre-NS"),
    ("automod_q1_qkgain",         "Q1", "per-head Q-gain parameters (init 5.25)"),
    ("automod_r1_recur",          "R1", "depth recurrence [0,1,2,3,4,1,2,3]"),
    ("automod_r2_recurprog",      "R2", "progressive two-phase recurrence"),
    ("automod_r3_parallel",       "R3", "GPT-J parallel residuals (attn_f + attn_d + MLP)"),
    ("automod_s1_multigroup",     "S1", "multi-group AdamW (norms/embed/decoder/other)"),
    ("automod_s2_dmodellr",       "S2", "Adam LR scaled by sqrt(768/e)"),
    ("automod_s3_schedule",       "S3", "adam_wd=0.03 + lawa_freq=2 + warmup=1500"),
    ("automod_x1_rows24only",     "X1", "revert l=6/fgs=3/last-col decoder, keep rows=24"),
    ("automod_w1_warmup750",      "W1", "Adam warmup_steps 1000 → 750"),
    ("automod_v1_deterministic",  "V1", "deterministic CUDA ops"),
    ("automod_d1_curriculum",     "D1", "easy-first dataset curriculum (num_features asc)"),
    ("automod_f1_fgmask",         "F1", "cyclic-distance feature-attn bias (learnable slope)"),
]

# Baseline: train_nano_automod_ (confirmed best config, n=11)
BASELINE_V28_MINS = 1.35

# ---------------------------------------------------------------------------
# techniques6/autoresearch batch (techniques6_autoresearch.sh, techniques6.md)
# Base script: train_nano_autoresearch.py. Each variant was copied from that
# base and named train_nano_autoresearch_<technique>.py; Slurm run names are
# tech6-<technique>-s<seed>.
# ---------------------------------------------------------------------------

TECHNIQUES_V29 = [
    ("tech6-rowclsdecoder",    "T6-01", "row CLS decoder column"),
    ("tech6-inducedcolattn",   "T6-02", "learned inducing-token feature attention"),
    ("tech6-qassmax",          "T6-03", "QASSMax row/ICL attention scaling"),
    ("tech6-rowrope",          "T6-04", "cached RoPE on row attention"),
    ("tech6-classembedding",   "T6-05", "ClassEmbedding target encoder"),
    ("tech6-dualtarget",       "T6-06", "dual target embeddings"),
    ("tech6-popstd",           "T6-07", "train-only std with unbiased=False"),
    ("tech6-fgattnmask",       "T6-08", "sparse feature-group attention mask"),
    ("tech6-qgain",            "T6-09", "per-head Q gain"),
    ("tech6-sparseattngate",   "T6-10", "sparse attention output gate"),
    ("tech6-denseattngate",    "T6-11", "dense attention output gate"),
    ("tech6-rowsmear",         "T6-12", "row smear lookback"),
    ("tech6-residx0lambdas",   "T6-13", "learnable residual/x0 lambdas"),
    ("tech6-backout",          "T6-14", "backout before decoder"),
    ("tech6-lateparallel",     "T6-15", "late parallel residual lane"),
    ("tech6-depthrecurrence",  "T6-16", "middle block depth recurrence"),
    ("tech6-polarexpress",     "T6-17", "Polar Express Muon"),
    ("tech6-normuon",          "T6-18", "NorMuon variance reduction"),
    ("tech6-cautiouswd",       "T6-19", "cautious weight decay"),
    ("tech6-multigroupadam",   "T6-20", "multi-group AdamW"),
]

# Baseline: train_nano_autoresearch.py base/automod confirmed best config.
BASELINE_V29_MINS = 1.35

# Current record to compare against
BASELINE_MINS = 7.57
BASELINE_AUC  = 0.8068462330697953

# ---------------------------------------------------------------------------
# Log parsing (reusing patterns from experiments.py)
# ---------------------------------------------------------------------------

TT_RE       = re.compile(r"\bt_t:([0-9]+(?:\.[0-9]+)?)s\b")
ROC_RE      = re.compile(r"\bavg_roc_auc\s*:\s*([0-9]+(?:\.[0-9]+)?)\b")
MU_E_T_RE   = re.compile(r"μ_e_t:([0-9]+(?:\.[0-9]+)?)s")
EPOCH_RE    = re.compile(r"\be:(\d+)(?:/\d+)?\b")
RECORD_RE   = re.compile(r"record time in mins\s*:\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
DATASETS_RE = re.compile(r"datasets seen\s*:\s*(\d+)", re.IGNORECASE)
RUNTIME_RE  = re.compile(r"script runtime\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*mins", re.IGNORECASE)


def parse_log(log_path: Path) -> dict:
    result = {
        "total_time_s": None,   # wall-clock training time (t_t)
        "record_mins":  None,   # time when jackpot was hit
        "final_auc":    None,   # last reported AUC
        "best_auc":     None,   # highest AUC seen
        "mean_epoch_t": None,
        "epoch":        None,
        "datasets":     None,
        "hit_jackpot":  False,
        "script_runtime_mins": None,
    }
    last_t_t = None
    best_auc = None

    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = TT_RE.search(line)
                if m:
                    last_t_t = float(m.group(1))

                m = ROC_RE.search(line)
                if m:
                    auc = float(m.group(1))
                    result["final_auc"] = auc
                    if best_auc is None or auc > best_auc:
                        best_auc = auc

                m = MU_E_T_RE.search(line)
                if m:
                    result["mean_epoch_t"] = float(m.group(1))

                m = EPOCH_RE.search(line)
                if m:
                    result["epoch"] = int(m.group(1))

                m = DATASETS_RE.search(line)
                if m:
                    result["datasets"] = int(m.group(1))
                    result["hit_jackpot"] = True

                m = RECORD_RE.search(line)
                if m:
                    result["record_mins"] = float(m.group(1))

                m = RUNTIME_RE.search(line)
                if m:
                    result["script_runtime_mins"] = float(m.group(1))

    except (FileNotFoundError, PermissionError):
        return result

    result["total_time_s"] = last_t_t
    result["best_auc"] = best_auc
    return result


def pick_log(run_dir: Path) -> Path | None:
    logs = list(run_dir.glob("*-log.txt"))
    return max(logs, key=lambda p: p.stat().st_mtime) if logs else None


def find_runs(experiments_dir: Path, technique: str, seed: int) -> list[dict]:
    """Return list of parsed results for all runs of a technique.

    Tries `{technique}-s{seed}` first (legacy naming), then falls back to
    `{technique}` (automod20.sh v28 naming, which passes --name "${NAME}"
    without a seed suffix).
    """
    for candidate in (f"{technique}-s{seed}", technique):
        technique_dir = experiments_dir / candidate
        if technique_dir.exists():
            break
    else:
        return []

    runs = []
    for run_dir in sorted(technique_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        log = pick_log(run_dir)
        if log is None:
            continue
        r = parse_log(log)
        r["run_id"] = run_dir.name
        runs.append(r)
    return runs


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_runs(runs: list[dict]) -> dict:
    """Summarise multiple runs of the same technique into one row."""
    if not runs:
        return None

    times_mins = [
        r["record_mins"] if r["hit_jackpot"] and r["record_mins"] is not None
        else (r["total_time_s"] / 60.0 if r["total_time_s"] is not None else None)
        for r in runs
    ]
    times_mins = [t for t in times_mins if t is not None]

    best_aucs = [r["best_auc"] for r in runs if r["best_auc"] is not None]
    hits = sum(1 for r in runs if r["hit_jackpot"])

    return {
        "n_runs":    len(runs),
        "n_hits":    hits,
        "best_mins": min(times_mins) if times_mins else None,
        "mean_mins": mean(times_mins) if times_mins else None,
        "best_auc":  max(best_aucs) if best_aucs else None,
        "mean_auc":  mean(best_aucs) if best_aucs else None,
        "epoch":     max(r["epoch"] for r in runs if r["epoch"] is not None) if any(r["epoch"] for r in runs) else None,
        "runs":      runs,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_mins(mins, *, delta=False):
    if mins is None:
        return "-"
    if delta:
        sign = "+" if mins >= 0 else ""
        return f"{sign}{mins:.2f}m"
    return f"{mins:.2f}m"


def fmt_pct(pct):
    if pct is None:
        return "-"
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def fmt_auc(auc):
    if auc is None:
        return "-"
    return f"{auc:.4f}"


def make_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    sep = "  ".join("-" * w for w in widths)
    header = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    lines = [header, sep]
    for row in rows:
        lines.append("  ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Inspect technique sweep results.")
    parser.add_argument("--dir", "-d", default="workdir/experiments",
                        help="experiments directory (default: workdir/experiments)")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--batch", choices=["v25", "v26", "v27", "v28", "v29", "tech6"], default="v25",
                        help="which technique batch to show (default: v25)")
    parser.add_argument("--baseline", type=float, default=None,
                        help="override baseline time in mins")
    parser.add_argument("--sort", choices=["time", "auc", "name", "priority"],
                        default="priority",
                        help="sort rows by: priority (default), time, auc, name")
    parser.add_argument("--all-runs", action="store_true",
                        help="show one row per run instead of aggregated per technique")
    args = parser.parse_args()

    if args.batch in ("v29", "tech6"):
        techniques = TECHNIQUES_V29
        default_baseline = BASELINE_V29_MINS
    elif args.batch == "v28":
        techniques = TECHNIQUES_V28
        default_baseline = BASELINE_V28_MINS
    elif args.batch == "v27":
        techniques = TECHNIQUES_V27
        default_baseline = BASELINE_V27_MINS
    elif args.batch == "v26":
        techniques = TECHNIQUES_V26
        default_baseline = BASELINE_V26_MINS
    else:
        techniques = TECHNIQUES
        default_baseline = BASELINE_MINS

    baseline_mins = args.baseline if args.baseline is not None else default_baseline
    experiments_dir = Path(args.dir)
    if not experiments_dir.exists():
        print(f"experiments dir not found: {experiments_dir}", file=sys.stderr)
        return 1

    # Collect data
    rows_data = []
    for priority, (name, code, desc) in enumerate(techniques, start=1):
        runs = find_runs(experiments_dir, name, args.seed)
        agg = aggregate_runs(runs)
        rows_data.append((priority, name, code, desc, agg, runs))

    # --all-runs: expand to one row per individual run
    if args.all_runs:
        print(f"\n{'=' * 110}")
        print(f"  All individual runs [{args.batch}]  (baseline: {baseline_mins:.2f}m  |  jackpot AUC: {BASELINE_AUC:.4f})")
        print(f"{'=' * 110}\n")

        headers = ["#", "technique", "code", "run_id", "time", "Δ vs base", "Δ%", "best_auc", "hit?", "epoch"]
        all_run_rows = []
        for priority, name, code, desc, agg, runs in rows_data:
            for r in runs:
                t = (r["record_mins"] if r["hit_jackpot"] and r["record_mins"] is not None
                     else (r["total_time_s"] / 60.0 if r["total_time_s"] is not None else None))
                delta = (t - baseline_mins) if t is not None else None
                pct = (delta / baseline_mins * 100) if delta is not None else None
                hit = "✓" if r["hit_jackpot"] else "✗"
                all_run_rows.append({
                    "priority": priority,
                    "name": name,
                    "code": code,
                    "run_id": r.get("run_id", "-"),
                    "time": t,
                    "delta": delta,
                    "pct": pct,
                    "best_auc": r["best_auc"],
                    "hit": hit,
                    "epoch": r["epoch"],
                })

        if args.sort == "time":
            all_run_rows.sort(key=lambda r: r["time"] if r["time"] is not None else float("inf"))
        elif args.sort == "auc":
            all_run_rows.sort(key=lambda r: -(r["best_auc"] or 0))
        elif args.sort == "name":
            all_run_rows.sort(key=lambda r: r["name"])
        # default: priority (already sorted)

        table_rows = []
        for i, r in enumerate(all_run_rows, start=1):
            table_rows.append([
                str(i),
                r["name"],
                r["code"],
                r["run_id"][-20:],
                fmt_mins(r["time"]),
                fmt_mins(r["delta"], delta=True),
                fmt_pct(r["pct"]),
                fmt_auc(r["best_auc"]),
                r["hit"],
                str(r["epoch"]) if r["epoch"] else "-",
            ])

        print(make_table(headers, table_rows))
        return 0

    # Default: one row per technique (aggregated)
    print(f"\n{'=' * 110}")
    print(f"  Technique sweep results [{args.batch}]  (baseline: {baseline_mins:.2f}m  |  jackpot AUC: {BASELINE_AUC:.4f})")
    print(f"{'=' * 110}\n")

    headers = [
        "#", "technique", "code", "runs", "hits",
        "best_time", "mean_time", "Δ mean", "Δ%",
        "best_auc", "mean_auc",
        "description",
    ]

    table_data = []
    for priority, name, code, desc, agg, runs in rows_data:
        if agg is None:
            table_data.append({
                "priority": priority, "name": name, "code": code, "desc": desc,
                "n_runs": 0, "n_hits": 0,
                "best_mins": None, "mean_mins": None, "delta": None, "pct": None,
                "best_auc": None, "mean_auc": None,
            })
            continue

        delta = (agg["mean_mins"] - baseline_mins) if agg["mean_mins"] is not None else None
        pct   = (delta / baseline_mins * 100) if delta is not None else None

        table_data.append({
            "priority": priority, "name": name, "code": code, "desc": desc,
            "n_runs":   agg["n_runs"],
            "n_hits":   agg["n_hits"],
            "best_mins": agg["best_mins"],
            "mean_mins": agg["mean_mins"],
            "delta":    delta,
            "pct":      pct,
            "best_auc": agg["best_auc"],
            "mean_auc": agg["mean_auc"],
        })

    # Sort
    if args.sort == "time":
        table_data.sort(key=lambda r: r["mean_mins"] if r["mean_mins"] is not None else float("inf"))
    elif args.sort == "auc":
        table_data.sort(key=lambda r: -(r["best_auc"] or 0))
    elif args.sort == "name":
        table_data.sort(key=lambda r: r["name"])
    # default: priority order (already in order)

    table_rows = []
    for i, r in enumerate(table_data, start=1):
        table_rows.append([
            str(i),
            r["name"],
            r["code"],
            str(r["n_runs"]),
            f"{r['n_hits']}/{r['n_runs']}",
            fmt_mins(r["best_mins"]),
            fmt_mins(r["mean_mins"]),
            fmt_mins(r["delta"], delta=True),
            fmt_pct(r["pct"]),
            fmt_auc(r["best_auc"]),
            fmt_auc(r["mean_auc"]),
            r["desc"],
        ])

    print(make_table(headers, table_rows))

    # Summary: which ones improved
    improved = [r for r in table_data if r["delta"] is not None and r["delta"] < 0]
    failed   = [r for r in table_data if r["n_runs"] > 0 and r["n_hits"] == 0]
    missing  = [r for r in table_data if r["n_runs"] == 0]

    print()
    if improved:
        improved.sort(key=lambda r: r["delta"])
        print(f"  ↓ improved over baseline ({baseline_mins:.2f}m) by mean time:")
        for r in improved:
            print(f"    {r['name']:20s}  mean:{fmt_mins(r['mean_mins'])}  {fmt_mins(r['delta'], delta=True)}  {fmt_pct(r['pct'])}")
    else:
        print("  no technique improved over baseline yet")

    if failed:
        print(f"\n  ✗ ran but did not hit jackpot: {', '.join(r['name'] for r in failed)}")

    if missing:
        print(f"\n  ? no results yet: {', '.join(r['name'] for r in missing)}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
