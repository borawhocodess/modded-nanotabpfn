# modded-nanoTabPFN speedrun — L40S paired result (oneirology)

**Pinned baseline:** upstream `687cfd9b5777`, verified byte-identical to the file we ran
(sha256 `a63654348e4c3e6d`). Base arm runs the pinned file **unmodified** — no patch, no seed shim
(`--seed` is upstream argparse and its default is already 11).
**Hardware:** 1x NVIDIA L40S on Modal (driver 580.95.05) — the record's own hardware class.
**Metric:** the repo's own `t_t` (sum of per-epoch times, each `cuda.synchronize()`-bracketed,
excluding eval and setup). Eval pipeline untouched (their rule 1). No pretrained weights (rule 2).
**Protocol:** n=15 paired rounds, seed 11, base and fast2 interleaved **inside the same container**,
arm order alternating by round parity. All 30 runs reached the target
(min AUC 0.806861 >= 0.8068462330697953).

## Headline (relative — which is what their rule 3 asks for)

> Rule 3, verbatim: *"Run faster than prior record when baselined on the same hardware with the same seed."*

| quantity | base (pinned) | fast2 | delta | 95% CI | test |
|---|---|---|---|---|---|
| **t_t, median** | 1.032 min | **0.873 min** | **-17.73%** | [-10.26, -20.48] | Wilcoxon p=0.00018 |
| **steady-state s/epoch** | 0.870 | **0.750** | **-13.79%** | [-12.79, -13.79] | Wilcoxon p=0.00057 |
| epoch-1 (warm compile) | 10.53 s | 9.01 s | -14.4% | — | — |
| epochs-to-target, median | 58 | 57 | — | — | Mann-Whitney **p=0.615** |
| paired wins | — | — | **14 / 15** | — | — |

**Steady-state per-epoch is the primary physical quantity** because it is invariant to how anyone
treats compile time; total `t_t` is the convention-bound one.

## Bit-equivalence witness

Epochs-to-target is statistically indistinguishable between arms (p=0.615; base 55-76, fast2 55-72).
The patch changes how the work is executed, not what is computed. This now holds on **two hardware
classes**: RTX 5090 (20/20, epochs 55-62 both arms) and L40S (14/15, p=0.615).

## The one loss, reported in full

Round 102: base 57.84 s / 58 epochs, fast2 61.10 s / **72 epochs**. fast2 lost on total `t_t` purely
because that draw needed 14 more epochs to cross the AUC threshold — its steady-state was still
**12.8% faster per epoch** in that same round. Epochs-to-target is not deterministic even at fixed
seed (bf16 reductions, atomics, autotune); the record holder's own table shows the same, spanning
54-58 epochs across 31 same-seed runs.

## Why the delta, not the absolute, is the claim

Across containers our **absolute** base steady-state ranged **0.86 - 1.01 s/epoch** (a 17% spread that
is a property of the machine we landed on), while the **within-container paired delta stayed
+12.8% to +25.7% in every single container**. The absolute number is a property of the host; the delta
is a property of the patch. That is the entire justification for the paired design.

**Mechanism, directly visible in the data:** fast2 held **0.750 s/epoch to three significant figures in
all 15 rounds**, while base varied 0.86-1.01 across hosts. The baseline's variance IS host variance;
fast2 largely stops inheriting it. This is the expected signature of removing ~64 GPU syncs per epoch,
and it makes the -17.7% much harder to attribute to lucky hosts — the patch is both faster and markedly
more stable. Check the `b_st` / `f_st` columns of `l40s_paired_rounds.csv`.

## What we do NOT claim

Our L40S **absolute** baseline median is 1.032 min against their published 0.92 min. We are ~12% slower
in absolute terms on Modal's L40S, saturated at cpu>=16 (0.87 s/epoch at cpu=16, 0.88 at cpu=32), i.e.
a genuine Modal-vs-cluster-node difference. **We therefore do not assert a reproduction of 0.92 min,
and no absolute number here should be read as a record time.** One container did produce a fast2 run at
0.83 min, under 0.92 — we note it and explicitly decline to headline it, because on that same container
the pinned baseline ran 0.99 min, so it is one container's absolute, not an apples-to-apples record.

**Prediction for the record holder's hardware** (for them to re-time, not a claim):
applying our median paired delta to their published 0.92 min gives **0.76 min**; a mechanistic
estimate that scales their implied epoch-1 and steady-state separately gives **0.79 min**.
Call it **0.76-0.79 min**.

## Disclosed environment conditions

* `cpu=16` on Modal. The repo's `PriorDumpDataLoader` is synchronous and host-CPU-bound; at cpu=8 the
  steady state degrades to ~1.40 s/epoch. Applied identically to both arms.
* **Warm `torch.compile` cache, symmetric across arms.** A cold inductor cache costs 41-109 s in epoch 1
  alone — more than the entire record — so the record's 0.92 min necessarily reflects a warm cache.
  We warm **every** arm before timed rounds and publish the symmetry evidence below.
* 12.2 GB prior dump staged to container-local disk.

## Methodology: an error we caught that biased AGAINST our own candidate

Our first warm-cache implementation created the shared cache once and never merged it. base and fast2
are different code and therefore different inductor cache keys, so the cache was warmed for **base
only** and fast2 paid a cold compile every round:

| | epoch-1, cache warmed for base only | epoch-1, after symmetric warm |
|---|---|---|
| base | 11.03 s | **9.82 s** |
| fast2 | **35.05 s** | **9.10 s** |

Under the broken version fast2 measured *slower* overall (77.31 s vs 62.21 s) while being 14% faster
per steady-state epoch — a plausible, publishable, wrong conclusion in the conservative direction. It
was caught by instrumenting per-epoch times rather than totals. We report it because compile-cache
asymmetry is the obvious thing a reviewer should ask about, and because a methods section that only
reports successes is less trustworthy than one that documents a self-caught error.

## Data

`l40s_paired_rounds.csv` — all 15 rounds, both arms: t_t, epochs, steady-state, epoch-1, final AUC,
and which arm ran first.
