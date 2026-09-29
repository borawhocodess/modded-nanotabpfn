# launchbound

This record is a systems-level speedup of record #10 ([#21](https://github.com/borawhocodess/modded-nanotabpfn/pull/21)). The training algorithm, the data and the evaluation are unchanged; only the execution of the training step differs.

On the L40S, the training step of #10 is limited by kernel launch overhead rather than by GPU compute: each step issues about 500 small kernels, and each launch costs roughly 45 µs of host time. The changes below therefore either reduce the number of launches or remove redundant work.

## Results

I compare against #10 (`ace9230`, unmodified) on the same machine with seed 11 and a warm `torch.compile` cache for both versions. The two versions alternate within each round, and every run is a full run with the standard evaluation. Measurements were taken on two Runpod L40S hosts with different CPUs, which I report separately.

| host | rounds | #10 median | this median | paired Δ | wins |
| --- | --- | --- | --- | --- | --- |
| A, EPYC 9374F | 8 | 0.865m | 0.574m | -32.7% | 8/8 |
| B, EPYC 9354 | 15 | 0.915m | 0.676m | -26.0% | 15/15 |

The number of epochs to reach the target is unchanged (median 59 vs. 56 on host A, 57 vs. 58 on host B), so the gain comes entirely from faster steps. The gain is larger on the host with the faster CPU, which is consistent with a host-bound step.

Both hosts I used run #10 more slowly than the reference node (0.865m and 0.915m, compared with 0.79m). I therefore do not claim an absolute time. Applying the paired differences to 0.79m gives an estimate of 0.53-0.58m on the reference node, which a re-timing there would confirm.

The table below lists every round. `first` indicates which version ran first in the round.

```
host A (EPYC 9374F, driver 580)                      host B (EPYC 9354, driver 570)
round  first  #10    epoch  this   epoch  Δ          round  first  #10    epoch  this   epoch  Δ
1      #10    0.90m  61     0.52m  55     -41.7%     1      #10    1.19m  77     0.79m  61     -33.4%
2      this   0.83m  59     0.62m  63     -25.8%     2      this   0.90m  56     0.64m  54     -29.5%
3      #10    0.91m  64     0.52m  56     -43.1%     3      #10    0.95m  60     0.70m  60     -26.2%
4      this   0.81m  56     0.55m  60     -32.0%     4      this   0.90m  57     0.80m  59     -11.6%
5      #10    0.83m  57     0.59m  56     -28.7%     5      #10    0.93m  56     0.67m  56     -27.8%
6      this   0.87m  58     0.59m  56     -31.8%     6      this   0.91m  56     0.66m  55     -27.8%
7      #10    0.87m  59     0.58m  59     -33.5%     7      #10    0.90m  54     0.68m  55     -24.0%
8      this   0.86m  60     0.57m  56     -34.2%     8      #10    0.93m  59     0.73m  60     -21.2%
                                                     9      this   0.90m  57     0.70m  56     -22.3%
median        0.865m 59     0.574m 56     -32.7%     10     #10    0.92m  58     0.69m  58     -25.2%
                                                     11     this   0.87m  56     0.67m  59     -23.9%
                                                     12     #10    1.10m  71     0.67m  59     -38.8%
                                                     13     this   0.90m  57     0.67m  57     -26.0%
                                                     14     #10    0.93m  59     0.68m  59     -27.7%
                                                     15     this   0.88m  55     0.67m  57     -23.3%

                                                     median        0.915m 57     0.676m 58     -26.0%
```

Host B was noisier than host A. In two rounds #10 needed 71 and 77 epochs, and in round 4 every epoch of this version was about 15% slower than usual. The rounds on host B were run as two consecutive batches (rounds 1-7 and 8-15), so the alternation restarts at round 8.

All 46 runs reached the target. The log at the top level of this folder is the host A run closest to the median (round 8); all logs are in `logs/hostA/` and `logs/hostB/`.

As with the other sub-minute records, these timings assume a warm compile cache. A cold cache adds 12-25 s to the first epoch.

## Changes

**Feature attention.** The attention across the columns of each row now uses a custom Triton kernel. This sequence has at most 21 tokens (20 features and the target), but FlashAttention pads it to tiles of 128 and launches two additional helper kernels. In the new kernel, one program per (row, head) computes the forward pass and another computes the backward pass. The kernel is registered as a `triton_op` with autograd, so it is compiled together with the rest of the layer. Evaluation datasets with more than 32 columns fall back to SDPA.

```python
# before x = F.scaled_dot_product_attention(q, k, v)
# after  x = feat_attn(q, k, v)
```

**Optimizer.** Gradient clipping and the Muon step are captured once as a CUDA graph and replayed at every step, which replaces about 60 launches with a single one. Muon performs the same update as in #10, but each group of equally shaped matrices is now a single compiled graph (`MuonFused`). Adam remains eager because its scheduled values change at every step.

```python
# before gnorms.append(clip_grad_norm_(model.parameters(), c.grad_clip)); for opt in optimizers: opt.step()
# after  gnorms.append(step_graph()); optimizer_adam.step()
```

**Precision.** The residual stream is kept in bf16 inside the transformer. Previously, concatenating the thinking rows promoted it to fp32.

```python
# src = src.to(torch.bfloat16)
```

**Compilation.** Three settings reduce the first epoch from 5.2 s to 2.9 s on host A. `specialize_float` prevents dynamo from restarting the compilation of the first layer, and `backed_size_oblivious` prevents the bf16 casts from triggering a recompilation for every input width. The layer is also compiled with `max_autotune` and `combo_kernels`.

```python
# torch._dynamo.config.specialize_float = True
# torch.fx.experimental._config.backed_size_oblivious = True
# @torch.compile(dynamic=True, options={"max_autotune": True, "combo_kernels": True})
```

## Ablation

The ablation measures timing only, on host B: each configuration trains for 10 epochs without evaluation, three times, and I report medians. "Epoch t" is the median time of epochs 2-10. `train_nano_abl.py` is the submitted file with one switch per change, `ablate_l40s.sh` runs all configurations, and the logs are in `logs/ablation_hostB/`.

| configuration | epoch 1 | epoch t |
| --- | --- | --- |
| #10 | 7.08 s | 0.82 s |
| all changes disabled | 6.01 s | 0.80 s |
| this record | 3.77 s | 0.57 s |

The table below gives the relative change when one component is removed from this record, and when it is added alone to the configuration with all changes disabled.

| change | removed (epoch 1 / epoch t) | added alone (epoch 1 / epoch t) |
| --- | --- | --- |
| `attention` | `-1%` / `+21%` | `+5%` / `-14%` |
| `bf16`      | `+1%` / `+16%` | `+3%` / `-8%`  |
| `graph`     | `+1%` / `+9%`  | `+7%` / `-1%`  |
| `compile`   | `+69%` / `+5%` | `-35%` / `+1%` |
| `muon`      | `+8%` / `0%`   | `-1%` / `0%`   |

The attention kernel and the bf16 stream account for most of the per-epoch gain, while the compilation settings account for the shorter first epoch. The CUDA graph has little effect on its own and helps only once the layer itself is fast; on host A it was the largest single gain. Because the components interact, the individual effects do not sum to the total. The configuration with all changes disabled still contains two `.contiguous()` calls, which explains why its first epoch is about 1 s shorter than that of #10.

## Notes

- Apart from the training step, the file is unchanged: the data loader, the preprocessing, the classifier and everything from `model.eval()` to the end of the file are identical to #10.
- I initially tuned on a workstation RTX 5000 Ada, where kernel launches are about three times cheaper. There, a batch size of 1 with 64 steps was about 50% faster, but on the L40S it was twice as slow per epoch. All tuning and timing reported here were therefore done on the L40S.
- The following did not help on the L40S: the packing from PR #20, `cpp_wrapper` (+3.4 s in the first epoch), compiling the whole model, other compilation modes, forcing SDPA backends, variants of Muon and of the learning-rate schedule, a sweep over 19 hyperparameter variants and a sweep over 12 architecture variants.
