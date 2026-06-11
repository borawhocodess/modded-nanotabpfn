# Extras — retime on the TabPFN-paper eval set

Time-to-jackpot (cross RF baseline `avg_roc_auc = 0.876071`), 5 reps, 1×L40S.

| variant | jackpot | median (min) | mean | sd | min–max | med epochs | speedup |
|---|---|---|---|---|---|---|---|
| baseline | 5/5 | 119.0 | 118.9 | 0.7 | 118.1–120.2 | 2254 | 1.0× |
| muon | 5/5 | 101.9 | 101.7 | 0.4 | 101.1–102.2 | 1385 | 1.17× |
| carter | 5/5 | 31.6 | 31.6 | 3.4 | 25.8–35.4 | 822 | 3.8× |
| batchedmuon | 5/5 | 22.3 | 22.5 | 3.1 | 17.8–26.2 | 866 | 5.3× |
| residualdecay | 5/5 | 15.5 | 15.9 | 1.3 | 14.2–18.2 | 602 | 7.7× |
| rmsthink | 5/5 | 13.2 | 13.1 | 0.7 | 12.0–14.2 | 528 | 9.0× |
| lawa1wd | 5/5 | 8.6 | 9.3 | 1.3 | 8.0–11.3 | 337 | 13.8× |
| featuregroup | 5/5 | 6.6 | 6.7 | 0.6 | 5.8–7.4 | 264 | 18.0× |
| autohuman | 5/5 | 2.1 | 2.0 | 0.2 | 1.7–2.3 | 139 | 56.7× |

# Residual decay study — fixed vs learned (2026-06-11)

Answers "why fix 0.95 with an exponential schedule instead of learning it?".
Current record applies `x * 0.95**i` at layer `i` (fixed base, exponential layer schedule).
Setup: current `train_nano.py` defaults (TABARENA eval, jackpot `avg_roc_auc = 0.806846`,
2-min cap), 5 reps, 1×L40S, job 29110509. Reference: autohuman record median 0.92 min.

| variant | change at layer i | jackpot | median (min) | mean | sd | min–max | med epochs |
|---|---|---|---|---|---|---|---|
| record (ref) | `x * 0.95**i` | — | 0.92 | 0.93 | 0.04 | 0.87–1.05 | 57 |
| learnedexpodecay | `x * d**i`, scalar d learned, init 0.95 | 5/5 | 0.94 | 0.97 | 0.04 | 0.94–1.01 | 56 |
| learnedlayerexpodecay | `x * s[i]`, l scalars, init 0.95**i | 5/5 | 1.00 | 0.98 | 0.03 | 0.94–1.01 | 60 |
| learnedlayerflatdecay | `x * s[i]`, l scalars, init 0.95 | 3/5¹ | 1.12 | 1.18 | 0.20 | 1.01–1.40 | 68 |
| flatdecay | `x * 0.95` for i>0 (no exponent) | 5/5 | 1.18 | 1.19 | 0.14 | 1.04–1.40 | 71 |

¹ one run exceeded the 2-min cap without jackpot, one died on a GPU ECC error (dlc2gpu05/10 batch; hardware, not code).

Where the learned values landed (final epoch, all reps):

- learnedexpodecay: base drifts 0.95 → **~0.89** very consistently (0.8893–0.8924) — gradient
  wants slightly *more* decay, it does not undo the trick.
- learnedlayerexpodecay: stays near a geometric schedule, slightly steeper than init:
  `[1.02, 0.95, 0.86, 0.78, 0.73]` (layer ratios ~0.90–0.93).
- learnedlayerflatdecay: starting flat, it *rediscovers* a monotonically decreasing schedule
  `[0.97, 0.94, 0.86, 0.85, 0.82]` — but never catches up in wall-clock.

Conclusions: (1) the exponential schedule itself matters — flat 0.95 costs ~28% time;
(2) learning the parameter(s) doesn't beat fixing them (best learned arm 0.94 vs 0.92 record),
the extra freedom only adds optimization overhead; (3) the learned values converge near the
fixed choice, slightly steeper — but a fixed-base sweep over 0.89–0.92 (job 29110598, 5 reps
each, 20/20 jackpot) found lower bases monotonically *worse* in wall-clock
(medians: 0.89 → 1.55, 0.90 → 1.20, 0.91 → 1.01, 0.92 → 0.93, vs 0.95 record 0.92), so the
learned ~0.89 optimizes training loss, not time-to-jackpot, and **0.95 stands**. Sweep tails
(1.5–1.76 min outliers) clustered by node (dlc2gpu12/16/17 vs 05), so medians are the
trustworthy number.

New parameters were routed to Adam by the existing `ndim != 2` split with the default
`adam_wd=0.01` (same treatment as every other scalar), so the ~0.89 plateau may be biased
slightly low by weight decay. Learned values were logged per epoch via an extra fragment
in the `print0` line, captured before the LAWA swap.

## Related work / what to cite

Per-layer scalar reweighting of residual paths is well explored — learned on the branch
(ReZero, LayerScale, LAuReL), learned on the stream (Admin, modded-nanogpt lambdas), or
fixed >1 on the stream for depth stability (DeepNorm) — but a fixed geometric decay of the
stream (`0.95**i`), chosen for time-to-accuracy rather than stability, appears unexplored;
our ablations show the learned variants converge near this schedule without beating it.

Must cite:

- **Admin** — Liu et al., "Understanding the Difficulty of Training Transformers", EMNLP 2020,
  arXiv:2004.08249. Only prior work scaling the *accumulated stream* per layer
  (`ω_l ⊙ x_l + f(x_l)`); ours is a fixed geometric schedule instead of an
  initialized-then-learned vector.
- **DeepNorm** — Wang et al., "DeepNet: Scaling Transformers to 1,000 Layers", 2022,
  arXiv:2203.00555. Fixed constant on the stream like ours but α > 1 for depth stability —
  same knob, opposite direction, different goal.
- **modded-nanogpt** — Jordan et al., github.com/KellerJordan/modded-nanogpt. Method lineage;
  its learned per-layer stream lambdas are the learned analog (theirs settle > 1 in LM
  pretraining, ours want ~0.89–0.95).
- **ReZero** — Bachlechner et al. 2020, arXiv:2003.04887; **LayerScale** — Touvron et al.,
  CaiT, ICCV 2021, arXiv:2103.17239. The branch-scaling family: they scale `f(x)`, we scale `x`.

Should cite:

- **Curse of Depth / LayerNorm Scaling** — Sun et al. 2025, arXiv:2502.05795. Motivation:
  residual-stream norm grows with depth and makes deep layers lazy; decay attacks this
  preemptively.
- **LAuReL** — Menghani et al., ICML 2025, arXiv:2411.07501. Learned `α·f + β·x`
  generalization; preempts "why not learn it", which the ablation above answers.

Optional (one sentence each): **DenseFormer** (arXiv:2402.02622) — our decay is the
zero-parameter special case of learned depth-weighted averaging; **nGPT** (arXiv:2410.01131)
— the norm-controlled-stream extreme.

Citation traps from an earlier LLM lit pass: CompleteP (arXiv:2505.01618) α=1 means 1/L on
the *branch*, not "no decay"; arXiv:2206.03126 is Noci et al. (rank collapse), not
Takase & Kiyono (theirs is B2T, arXiv:2206.00330); ProRes (arXiv:2603.05369) unverified.
