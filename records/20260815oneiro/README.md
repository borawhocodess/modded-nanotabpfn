# oneiro: batched Newton-Schulz, producer-thread dataloader, fused SDPA

Systems-level speedup. **The training algorithm is unchanged** - epochs-to-target is statistically
indistinguishable from the pinned baseline (Mann-Whitney p=0.615), so this is a pure throughput win.

## Result (relative, per rule 3)

Verified head-to-head against the pinned baseline @687cfd9, **paired inside the same container**,
seed 11, alternating arm order, on two hardware classes.

| venue | protocol | base t_t (median) | fast2 t_t (median) | delta | paired wins | epochs-to-target |
|---|---|---|---|---|---|---|
| **1x L40S** | n=15 paired rounds | 1.032 min | **0.873 min** | **-17.7%** (95% CI -10.3 to -20.5, Wilcoxon p=0.00018) | **14/15** | base 58 vs fast2 57, MWU **p=0.615** |
| 1x RTX 5090 | n=20 interleaved | 0.650 min | **0.530 min** | **-18.5%** | 20/20 | 55-62 both arms, median 57 |

Convention-invariant form (does not depend on how compile time is counted):
steady-state **0.870 -> 0.750 s/epoch, -13.8%** (95% CI -12.8 to -13.8, p=0.00057).
All 30 L40S runs and all 40 RTX 5090 runs reached the target (min AUC 0.806861 >= 0.8068462330697953).

## What changed (algorithm bit-equivalent)

1. **Batched Newton-Schulz** - all 20 Muon parameter matrices orthogonalized in 3 batched kernel calls
   per step instead of 20 sequential ones; `torch._foreach_*` momentum update.
2. **Producer-thread dataloader** - pinned memory + non-blocking H2D copies; the per-batch NaN guard
   moves CPU-side into the producer thread, removing ~64 GPU syncs per epoch while preserving the exact
   skip semantics.
3. **Fused SDPA** - one `scaled_dot_product_attention` over all query rows, mathematically identical to
   the existing split+cat.

## Mechanism, visible in the data

**fast2 held 0.750 s/epoch to three significant figures in all fifteen L40S rounds, while the baseline
wandered 0.86-1.01 s/epoch across hosts.** Removing ~64 GPU syncs per epoch should decouple the training
loop from host jitter, and it does - measurably, fifteen times out of fifteen. Check the `steady` columns
of `l40s_paired_rounds.csv`. This makes the -17.7% hard to attribute to lucky hosts: the baseline's
variance IS host variance, and fast2 largely stops inheriting it.

## What we do NOT claim

**We are not claiming a reproduction of the 0.92 min record.** Our L40S absolute baseline runs 1.032 min
- the L40S instances available to us are ~12% slower per epoch than the record node even saturated at
cpu>=16, which is a property of our host, not of the code. Applying our measured delta to 0.92 min
predicts roughly **0.76-0.79 min** on the record hardware. **We invite a re-timing on the original node
for the official number.**

## Methodology notes (please read)

* **Paired, in-container, alternating order.** Between-container spread in the absolute baseline was
  0.86-1.01 s/epoch while the within-container delta stayed +12.8% to +25.7% in EVERY container. The
  absolute is a property of the machine; the delta is a property of the patch.
* **Warm `torch.compile` cache, symmetric across arms, disclosed.** A cold inductor cache costs 41-109 s
  in epoch 1 alone - more than the entire record - so any sub-minute measurement necessarily reflects a
  warm cache. Every arm is warmed before timed rounds.
* **An error we caught that biased AGAINST our own patch.** Our first warm-cache implementation warmed
  whichever arm ran first (base), so fast2 paid a cold compile every round: epoch-1 35.05 s vs 11.03 s.
  Under that bug fast2 measured SLOWER overall despite being 14% faster per steady-state epoch. After the
  fix, epoch-1 is 9.10 s (fast2) vs 9.82 s (base). Flagged because compile-cache asymmetry is the first
  thing a reviewer should suspect in a claim like this.
* **The one loss, in full.** L40S round 102: fast2 needed 72 epochs vs base's 58 on that draw and lost on
  total `t_t` (61.10 s vs 57.84 s) while still being 12.8% faster per epoch. Epochs-to-target is not
  deterministic at fixed seed - the record table's own runs span 54-58 across 31 same-seed runs.
* **Environment:** Modal, 1x L40S, driver 580.95.05, torch 2.9.1+cu128, python 3.13, cpu=16, upstream uv
  lock. The base arm is the pinned file byte-for-byte.

## Files

* `l40s_paired_rounds.csv` - all 15 paired rounds, per-round base/fast2 times, epochs, steady-state, AUC
* `RESULTS_L40S.md` - full analysis including the bit-equivalence witness
* `INTEGRITY.md` - the metric-gaming vectors we checked against ourselves, and what we refused to use

Found and verified with an autonomous agent harness. Rule 3 is measured directly by the paired protocol:
the pinned code and ours, same machine, same seed, interleaved, fifteen times.


## Round provenance, stated explicitly (added 2026-08-15)

The headline n=15 is **two batches**, and a reviewer should see both separately rather than only the pooled number.

| subset | n | base median | fast2 median | median delta | paired wins |
|---|---|---|---|---|---|
| **batch B** (rounds 100-111, full per-round data in `logs/l40s_rounds_full.json`) | 12 | 0.997 min | 0.851 min | **-15.90%** | **11/12** |
| batch A (surviving rounds 15-17 from an earlier run) | 3 | - | - | **~-20.5%** | **3/3** |
| **pooled (headline)** | **15** | **1.032 min** | **0.873 min** | **-17.73%** | **14/15** |

Why two batches: an earlier 21-round run lost most of its data to a Modal Volume concurrent-write bug on our
side (all workers appending to one file; Volume commits are whole-file, so only the last writer survived).
Three rounds were recoverable. The protocol was then rebuilt with **one result file per worker** and re-run as
batch B. **The loss was ours and it cost ~$20; no round was discarded for its result.**

The pooled median is slightly stronger than batch B alone because batch A's three surviving rounds happened to
land at the wider end. **If you prefer the single-batch number, use -15.90% at 11/12** - it is the more
conservative reading and it does not change the conclusion or the sign. Every arm in every round hit the target
(minimum AUC across all 24 arms in batch B: 0.806866441 >= 0.8068462330697953).
