# trilinear

trilinear feature attention in two of the five layers, a retuned Muon and AdamWScheduleFree, and a gradient clip of 1. on top of record 13 (testtime, [PR #26](https://github.com/borawhocodess/modded-nanotabpfn/pull/26)), whose test-time inference is kept unchanged.

**stacked on PR #26, which is not merged yet.** this branch contains the #26 commit, and every number below was measured with the #26 inference (8 feature-permutation members, per-fold test-time fine-tuning). if #26 is not merged, or merged in a different form, this record has to be rebased onto whatever `main` is then and re-timed. the changes here are all on the training side and should carry over, but the epoch counts and times in this README would not. the one data point for that case: with the plain inference of record 12, the search's 5 runs of this configuration reached the target at epoch 45.2 on average (38 to 52), against 50.4 for the hand-retuned optimizer alone (7 runs, 46 to 59) and 55 to 66 for record 12 itself (13 runs during the #26 work, median 58).

measured on 1x RTX 5090 under Windows, not on an L40S, so the record time in the root README is left as `X.XX`.

the gain is in epochs: the target is reached at epoch 39 to 40 instead of 50 to 54 (-24% on the means). part of it is paid back on the clock, because the new layers make the first epoch about 1s longer and every later epoch a few percent longer, so the median training time drops by 13% (18.80s against 21.70s).

## Changes

three change groups, all on the training side. `ModdedNanoTabPFNClassifier` and the evaluation loop are identical to record 13.

### 1. trilinear feature attention (layers 3 and 4)

in two layers the feature attention gets a second key and value projection, `k2_features` and `v2_features` (`Linear(e, e)` each, split into the same 4 heads). every feature token then has a partner key `k2` and a partner value `v2` next to its key and value. two forms use them:

- **cascade**, layer 3 (`cascade_layers = (2,)`): a first attention over the partners scales the query of the second one.

  ```python
  u = attn(q, k2, v2)
  out = attn(q * u, k, v)
  ```

  the score of feature `j` is `<q * u, k_j> = sum_d q_d u_d k_jd`, a trilinear form in the query, the partner summary and the key.

- **product**, layer 4 (`product_layers = (3,)`): the product of two ordinary attentions.

  ```python
  out = attn(q, k, v) * attn(q, k2, v2)
  ```

  this is 2-simplicial attention over feature pairs `(j, l)` with additive pair scores `<q, k_j> + <q, k2_l>` and pair values `v_j * v2_l`. with additive scores the softmax over pairs factorises into the product of two softmaxes, so the `C^2` pair attention costs two ordinary attentions.

the cascade runs as one Triton kernel (forward and backward, `cascade_attn`), written after the record 12 feature attention kernel and with the same limits: `[N, H, C<=32, 64]`, which covers every training batch; wider inputs (evaluation datasets with up to 100 features) fall back to two SDPA calls. the product form is two calls of the record 12 `feat_attn`.

init: the partner projections draw `nn.Linear`'s default init from their own `torch.Generator`s (seeds 1001 to 1004). the global RNG is not touched, so every other parameter starts exactly as in record 13.

optimizer: the partner weight matrices are Muon matrices like every other matrix in the encoder, in their own shape group with the learning rate scaled by `partner_lr_scale = 0.56`. Muon's weight decay is applied as `lr * wd`, so it is scaled with it. their biases go to AdamWScheduleFree like every other bias.

parameters: 4,160,520 -> 4,423,688 (2 layers x 2 projections x (256 x 256 + 256)).

```python
# TransformerEncoderLayer.forward, feature attention
if self.form == "plain":
    x = feat_attn(q, k, v)
else:
    k2 = self.k2_features(x).reshape(b * r, c, self.num_heads, self.head_dim).transpose(1, 2)
    v2 = self.v2_features(x).reshape(b * r, c, self.num_heads, self.head_dim).transpose(1, 2)
    if self.form == "cascade":
        x = cascade_attn(q, k, v, k2, v2)
    else:
        u = feat_attn(q, k2, v2)
        x = feat_attn(q, k, v) * u
```

|                    | value  |
| ------------------ | ------ |
| `cascade_layers`   | `(2,)` |
| `product_layers`   | `(3,)` |
| `partner_lr_scale` | 0.56   |

with `cascade_layers = ()` and `product_layers = ()` the model is the record 13 model again (same parameters, same init), with the optimizer settings below.

### 2. optimizer retune

| | record 13 | this record |
| --- | --- | --- |
| Muon weight decay `muon_wd` | 0.1 | 0.5 |
| Muon learning rate scale `muon_lr_scale` | 0.1 | 0.14 |
| AdamWScheduleFree learning rate `adam_lr` | 0.001 (`lr`) | 0.0015 |
| AdamWScheduleFree `warmup_steps` | 1000 | 50 |
| AdamWScheduleFree `r` | 0 (default) | 2 |

`adam_lr` is a new `Config` field and only feeds AdamWScheduleFree. Muon's learning rate is still `muon_lr_scale * lr`.

### 3. gradient clip

`grad_clip` 2.0 -> 1.0.

## Results

3 runs of this record and 1 run of the record 13 script, same machine, seed 11, one run at a time with nothing else of mine on the GPU, every timed run preceded by a discarded `--epochs 1` run of the same script. the record 13 run is a check against the three runs published with #26 four days earlier, which are reused instead of timing record 13 again. the median run of this record is at the top level, all its runs and the record 13 run are in `logs/`, and the 3 timed runs of config 195 (ablation below) are in `logs/config195/`.

this record:

```
mean             0.31m    18.75s   40     0.47s      2539      25.51m
std              0.00m    0.24s    0      0.00s      30        0.35m
median           0.31m    18.80s   40     0.47s      2560      25.71m
---------------  -------  -------  -----  ---------  --------  -------  ------------------
##  #  date      in mins  in secs  epoch  μ epoch t  datasets  runtime  id-name
--  -  --------  -------  -------  -----  ---------  --------  -------  ------------------
1   3  26-11-28  0.31m    18.43s   39     0.47s      2496      25.02m   deefe210-trilinear
2   2  26-11-28  0.31m    18.80s   40     0.47s      2560      25.71m   b5e6af4c-trilinear
3   1  26-11-28  0.32m    19.01s   40     0.48s      2560      25.80m   b0da06d2-trilinear
```

record 13 (`record-testtime` at `08dd5d6`, the #26 branch): the 3 runs from the #26 pr and today's check run:

```
mean             0.36m    21.68s   52     0.42s      3344      32.56m
std              0.01m    0.81s    1      0.01s      95        1.64m
median           0.36m    21.70s   52     0.41s      3360      32.47m
---------------  -------  -------  -----  ---------  --------  -------  -----------------------
##  #  date      in mins  in secs  epoch  μ epoch t  datasets  runtime  id-name
--  -  --------  -------  -------  -----  ---------  --------  -------  -----------------------
1   1  26-11-23  0.34m    20.53s   50     0.41s      3200      30.35m   a59ff5f7-testtime
2   3  26-11-24  0.36m    21.52s   53     0.41s      3392      32.16m   2d56a2ac-testtime
3   2  26-11-24  0.36m    21.87s   54     0.41s      3456      32.79m   ec742d13-testtime
4   4  26-11-28  0.38m    22.78s   52     0.44s      3328      34.95m   c5a607f2-trilinear-base
```

median 21.70s against 18.80s, `-13%` (`-17%` against today's record 13 run alone, `-13%` against the median of the three #26 runs alone). epochs to target 39, 40, 40 against 50, 52, 53, 54.

the seconds are noisy, the epochs less so. per epoch after the first, today's record 13 run (midday) took 0.402s against 0.372s four days earlier, the config 195 runs (early afternoon) 0.413s to 0.436s, and this record's runs (evening) 0.400s. that is background load on a desktop machine, nothing I started. the epoch counts do not depend on it.

the timed runs of this record (mean 39.7 epochs) came out a little worse than the search's 5 runs of the same configuration (mean 37.0, 35 to 40), and equal to the timed runs of config 195 (39.7). so in timed runs the last step of the ablation (AdamW lr, Muon wd) shows no gain in epochs, and the 5% lower median time against config 195 (18.80s against 19.87s) comes from the quieter evening, not from the settings. 3 runs is a small sample, and the maintainers re-time every pr on their nodes.

discarded, not in the tables: one run of config 195 that was killed before its first epoch, because its warm-up run had been killed by mistake while I restarted the run queue.

## Ablation Study

no separate timed runs per arm. the table uses the runs that already exist: the timed runs above, the hand runs of the optimizer retune, and the runs of the search that picked this configuration (next section). everything is the first epoch at which the #26 inference reaches the target, seed 11. epoch counts do not depend on the clock, so runs from different days compare directly; the time column is not given for that reason.

| model | runs | first epoch at target | mean | vs record 13 |
| --- | --- | --- | --- | --- |
| record 13 (#26), timed runs: 3 from the #26 pr and 1 from today | 4 | 50, 52, 53, 54 | 52.3 | |
| + optimizer retune by hand (Muon wd 1.0, Muon lr scale 0.14, warmup 50, `r=2`), hand runs | 4 | 43, 46, 49, 52 | 47.5 | `-9%` |
| same, the search's reference runs | 8 | 43, 44, 45, 45, 46, 47, 47, 48 | 45.6 | `-13%` |
| + trilinear layers + grad clip 1 (config 195), the search's runs | 17 | 33 to 41 | 37.8 | `-28%` |
| same, timed runs (`logs/config195/`) | 3 | 37, 39, 43 | 39.7 | `-24%` |
| + AdamW lr 1.5e-3, Muon wd 0.5 (config 399, this record), the search's runs | 5 | 35, 36, 36, 38, 40 | 37.0 | `-29%` |
| this record, timed runs | 3 | 39, 40, 40 | 39.7 | `-24%` |

read the other way round, as record minus change: without the trilinear layers and the grad clip `+21%` epochs (45.6 against 37.8). trilinear layers and grad clip 1 come as one step because the search only tried the clip together with them (below). the last step, AdamW lr and Muon wd, is 0.8 epochs in the search, inside its noise, and 0 in the timed runs (39.7 against 39.7).

around the last step, from the search's runs (same seed, everything else as in this record):

| change | runs | mean first epoch at target |
| --- | --- | --- |
| this record (AdamW lr 1.5e-3, Muon wd 0.5) | 5 | 37.0 |
| AdamW lr 1e-3, Muon wd 1.0 (config 195) | 17 | 37.8 |
| AdamW lr 1e-3, Muon wd 0.5 | 2 | 39.0 |
| AdamW lr 1.5e-3, Muon wd 0.7 | 3 | 39.3 |
| AdamWScheduleFree `r=1` | 1 | 42 |

around config 195, from the search's runs (same seed, everything else as in config 195):

| change | runs | mean first epoch at target |
| --- | --- | --- |
| config 195 (grad clip 1, partner lr 0.56) | 17 | 37.8 |
| grad clip 0.7 | 4 | 38.5 |
| grad clip 1.4 | 4 | 38.5 |
| grad clip 2 | 1 | not measured: stopped early by the search. AUC at epoch 28 0.7981, inside the 0.7977 to 0.8026 of this record's runs |
| partner lr 0.32 | 4 | 39.2 |
| partner lr 1.0 (no multiplier) | 1 | 41 |
| only layer 3, cascade form, grad clip 2, partner lr 1.0 | 1 | 45 |
| only layer 3, full 2-simplicial attention (softmax over feature pairs, own kernel), grad clip 2, partner lr 1.0 | 2 | 47.5 |

so on the retuned optimizer the full 2-simplicial attention in one layer does not help (47.5 against 45.6 for the reference), the cascade alone is within noise of the reference (one run), and the gain shows up with the cascade and the product form in two adjacent layers plus the lower clip.

not measured:

- the optimizer retune with grad clip 1 and without the trilinear layers. every run of the search had some trilinear form in it.
- the trilinear layers on the optimizer of record 13. the search started from the retuned optimizer.
- the product form alone in layer 4.

things to know about this table:

- the search trained with the plain evaluation and saved the weights of every epoch near the target; the #26 inference was then run on those saved weights. #26 showed that such a replay reproduces the inline evaluation to the last digit for the ensemble, and to about ±0.00002 AUC with the test-time fine-tuning, which is not bit-reproducible.
- the search trained on both GPUs of the machine, the RTX 5090 and an RTX 3080, after checking that the 3080's runs fall inside the 5090's run-to-run spread of loss and AUC. all #26 replays ran on the 5090.
- the hand runs of the retune were trained with the plain evaluation and replayed the same way.
- with the seed fixed, one configuration still spreads over about 8 epochs (33 to 41 over the 17 runs of config 195): training is not bit-reproducible on this GPU. the 3- and 4-run means above are good to about ±2 epochs.

## How the settings were chosen

by looking at TabArena AUC, the metric the target is defined on, in two steps.

1. **optimizer retune, by hand.** a few single-knob sweeps on record 13 with the #26 inference: Muon weight decay, Muon learning rate scale, AdamWScheduleFree warmup and `r`. the four runs of the result are the "hand runs" row above.
2. **a multi-fidelity Bayesian search**, starting from the retuned model, over the trilinear forms, their placement and their own settings (partner learning rate among them), together with 14 other settings: number of layers, Muon weight decay, momentum and learning rate scale, AdamW learning rate, warmup and `r`, grad clip, LAWA length, residual decay, decoder readout, a feature-position embedding, label smoothing and row dropout. 646 trials of 515 configurations in about 19 hours on the RTX 5090 and an RTX 3080 of the same machine. the objective was the training time to target with the #26 inference, taken as (first epoch at target) x (seconds per epoch): the epoch count from replays of saved weights, the seconds per epoch from solo-timed runs on the 5090 (predicted for architectures that were not timed, and only timed architectures could rank as best). a random-forest surrogate proposed configurations by expected improvement per unit of compute, and a running trial was stopped when its expected improvement over the remaining compute was small. every trial used seed 11.

this record is the search's top-ranked configuration, config 399 (37.0 epochs over 5 runs). the runner-up, config 195 (37.8 epochs over 17 runs), is the same without the AdamW lr 1.5e-3 and Muon weight decay 0.5. the two are within noise of each other (0.8 ± 1.2 epochs), and 195 is in the ablation above with timed runs of its own. the search's other close configurations all share this architecture and differ in a scalar or two.

so the configuration is selected on the same metric and with the same seed as the timed runs. a different seed was not tried.

## Cost

|                                    | record 13, today | this record, 3 runs     |
| ---------------------------------- | ---------------- | ----------------------- |
| first epoch                        | 2.28s            | 3.40s, 3.21s, 3.20s     |
| later epochs, mean                 | 0.402s           | 0.400s, 0.400s, 0.401s  |
| epochs to target                   | 52               | 40, 40, 39              |
| untimed time per epoch             | 39.9s            | 38.2s, 38.1s, 38.0s     |
| script runtime                     | 34.95m           | 25.80m, 25.71m, 25.02m  |
| peak memory allocated              | 4629 MiB         | 4925 MiB                |
| parameters                         | 4,160,520        | 4,423,688               |

- the new layers add about 1s to the first epoch (the first epoch of the four record 13 runs: 2.17s to 2.28s, of the config 195 runs 3.16s to 4.32s). most likely compile and CUDA graph setup that the warm-up run does not cache away, not investigated. that is about 5% of the record time and the first thing to look at if the time on the L40S disappoints.
- after the first epoch each new layer computes one more attention and two more projections per step, forward and backward (the cascade fuses both of its attentions into one kernel, the product form calls `feat_attn` twice). the search measured this solo as 0.400s against 0.385s per epoch, +4%. here this record's evening runs (0.400s) and the midday record 13 run (0.402s) are level, because the machine's load moved more than the layers cost (see Results).
- the evaluation (untimed) costs about the same per epoch, and fewer epochs make the whole script shorter.

## Prior work

- 2-simplicial attention: Clift, Doryn, Murfet, Wallbridge, "Logic and the 2-Simplicial Transformer", https://arxiv.org/abs/1909.00668, and Roy et al., "Fast and Simplex: 2-Simplicial Attention in Triton", https://arxiv.org/abs/2507.02754. the product form (additive pair scores, so the pair softmax factorises) and the cascade form are our own simplifications of it; we know of no source for either.
- the cascade kernel follows the Triton feature attention kernel of record 12 ([launchbound](../20260928launchbound/README.md)).
- schedule-free AdamW (warmup, `r`): Defazio et al., "The Road Less Scheduled", https://arxiv.org/abs/2405.15682, https://github.com/facebookresearch/schedule_free. the record already used it, only two of its settings change.
- Muon: https://kellerjordan.github.io/posts/muon/, code adapted from modded-nanogpt as before. its weight decay came in with record 9.
- the search: random-forest surrogate as in SMAC (Hutter, Hoos, Leyton-Brown, "Sequential Model-Based Optimization for General Algorithm Configuration", LION 2011), expected improvement per unit cost (Snoek, Larochelle, Adams, https://arxiv.org/abs/1206.2944), early stopping of running trials related in spirit to Hyperband (https://arxiv.org/abs/1603.06560), BOHB (https://arxiv.org/abs/1807.01774) and freeze-thaw BO (https://arxiv.org/abs/1406.3896). the combination is our own and implements none of them exactly.

## Environment

- RTX 5090, Windows 11, Python 3.13, torch 2.9.1+cu128, triton-windows 3.5.1, the same machine and environment as record 13.
- as in record 13, the committed `train_nano.py` reads `/proc/cpuinfo` and calls `os.sched_getaffinity`, neither exists on Windows. the logs come from a local copy in which that one logging line is replaced, nothing else differs, and the code printed at the top of each log is the code that ran:

```python
# committed
cpu = next(l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
print0(f"cpu: {cpu} ({len(os.sched_getaffinity(0))} cores)")
# local copy
print0(f"cpu: {platform.processor()} ({os.cpu_count()} cores)")
```

- environment variables for the runs: `TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0` (the static launcher fails on Windows with this torch version), `PYTHONUTF8=1`, `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=0` (the machine also has an RTX 3080, which was idle).
- the committed `train_nano.py` itself has not been run on Linux by me.
- in the logs the hostname and the process rows of the `nvidia-smi` table are removed (a desktop machine, the rows listed desktop applications only, all of type C+G). nothing else in the logs is edited.
