# launchbound

launch-bound speedup by [@XingyuQu](https://github.com/XingyuQu) ([#22](https://github.com/borawhocodess/modded-nanotabpfn/pull/22)) on top of record 10, the training algorithm is unchanged, only how the training step is executed. previous record 0.76m, this is -39% on our node.

on the L40S the step is limited by kernel launch overhead and not by gpu compute: about 500 small kernels per step at roughly 45 µs of host time each (numbers from the pr). every change below cuts launches or removes redundant work.

the paired protocol is in the same pr, it estimated 0.53-0.58 min on our node, our re-timing lands at 0.46.

the feature width sorting of record 11 is taken out, on top of this record it does not help: epochs barely get faster and it needs more of them.

only the median run's log is included at the top level, all 31 runs are in `logs/`.

```
mean                         0.47m    58     0.49s      3714      7.15m
std                          0.03m    4      0.01s      276       0.52m
median                       0.46m    56     0.49s      3584      6.99m
---------------------------  -------  -----  ---------  --------  -------  -------------
##  #   date      hostname   in mins  epoch  μ epoch t  datasets  runtime  id-name
--  --  --------  ---------  -------  -----  ---------  --------  -------  -------------
1   6   26-09-30  dlc2gpu12  0.43m    54     0.48s      3456      6.58m    8fcf564d-pr22
2   3   26-09-30  dlc2gpu12  0.44m    54     0.49s      3456      6.71m    5bc13613-pr22
3   14  26-09-30  dlc2gpu12  0.44m    55     0.48s      3520      6.80m    06fd7942-pr22
4   4   26-09-30  dlc2gpu12  0.45m    55     0.49s      3520      6.74m    d29e5f49-pr22
5   20  26-09-30  dlc2gpu12  0.45m    55     0.49s      3520      6.78m    6e68bb4b-pr22
6   18  26-09-30  dlc2gpu12  0.45m    55     0.49s      3520      6.81m    12872764-pr22
7   12  26-09-30  dlc2gpu12  0.45m    55     0.49s      3520      6.77m    f6c9016c-pr22
8   1   26-09-30  dlc2gpu04  0.45m    56     0.49s      3584      6.88m    41144530-pr22
9   13  26-09-30  dlc2gpu12  0.46m    56     0.49s      3584      6.94m    1a1e624c-pr22
10  25  26-09-30  dlc2gpu12  0.46m    56     0.49s      3584      6.82m    2736358c-pr22
11  17  26-09-30  dlc2gpu12  0.46m    56     0.49s      3584      6.88m    37723255-pr22
12  30  26-09-30  dlc2gpu30  0.46m    56     0.49s      3584      7.03m    821eb3e2-pr22
13  11  26-09-30  dlc2gpu12  0.46m    56     0.49s      3584      6.89m    acae8a7e-pr22
14  23  26-09-30  dlc2gpu12  0.46m    56     0.49s      3584      6.90m    4e2e594b-pr22
15  9   26-09-30  dlc2gpu12  0.46m    56     0.49s      3584      7.01m    8be37ab5-pr22
16  8   26-09-30  dlc2gpu07  0.46m    57     0.48s      3648      7.02m    caf9862e-pr22
17  5   26-09-30  dlc2gpu12  0.46m    56     0.49s      3584      6.90m    6eb18d4c-pr22
18  31  26-09-30  dlc2gpu12  0.46m    57     0.49s      3648      6.98m    0a2ee41c-pr22
19  26  26-09-30  dlc2gpu12  0.47m    57     0.49s      3648      7.04m    a8d9235c-pr22
20  2   26-09-30  dlc2gpu04  0.47m    57     0.49s      3648      7.01m    548bfa9d-pr22
21  10  26-09-30  dlc2gpu12  0.47m    57     0.49s      3648      6.99m    1f774efa-pr22
22  15  26-09-30  dlc2gpu12  0.47m    59     0.48s      3776      7.19m    0ac8ab86-pr22
23  28  26-09-30  dlc2gpu30  0.47m    58     0.49s      3712      7.39m    9dc8c218-pr22
24  21  26-09-30  dlc2gpu12  0.48m    59     0.49s      3776      7.23m    b714449b-pr22
25  24  26-09-30  dlc2gpu12  0.48m    59     0.49s      3776      7.30m    2060df78-pr22
26  19  26-09-30  dlc2gpu12  0.49m    60     0.49s      3840      7.31m    5bacfa35-pr22
27  7   26-09-30  dlc2gpu07  0.49m    61     0.48s      3904      7.43m    51ec2876-pr22
28  27  26-09-30  dlc2gpu12  0.49m    61     0.48s      3904      7.48m    c0c11aed-pr22
29  22  26-09-30  dlc2gpu12  0.55m    69     0.48s      4416      8.50m    47d4709a-pr22
30  29  26-09-30  dlc2gpu12  0.56m    70     0.48s      4480      8.56m    4219695e-pr22
31  16  26-09-30  dlc2gpu12  0.56m    71     0.47s      4544      8.72m    1c97df6c-pr22
```

note: all sub-minute timings here assume a warm `torch.compile` cache.

## Changes

Feature attention: a custom Triton kernel replaces SDPA for the attention across columns, wide evaluation tables fall back to SDPA.

```python
# before
x = F.scaled_dot_product_attention(q, k, v)
# after
x = feat_attn(q, k, v)
```

Precision: the residual stream stays in bf16 inside the transformer.

```python
src = src.to(torch.bfloat16)
```

Compilation: two settings stop the layer from recompiling, and the layer is compiled with autotuned matmuls and combined kernels.

```python
torch._dynamo.config.specialize_float = True
torch.fx.experimental._config.backed_size_oblivious = True
LAYER_OPTS = {"max_autotune": True, "combo_kernels": True}
```

Optimizer: Muon does the same update, but each group of equally shaped matrices is one compiled graph (`MuonFused`). gradient clip and the Muon update are recorded once as a CUDA graph after 3 eager steps and replayed every step (`StepGraph`). Adam stays eager.

```python
# before
gnorms.append(torch.nn.utils.clip_grad_norm_(model.parameters(), c.grad_clip))
for opt in optimizers:
    opt.step()
# after
gnorms.append(step_graph())
optimizer_adam.step()
```

## Ablation Study

ran multiple trials per knob in one direction:
- removing the change from the launchbound record

| knob | launchbound - knob | mean | std | median |
| --- | --- | --- | --- | --- |
| `attention` | `+42%` | 40.41s (0.67m) | 6.92s | 39.08s (0.65m) |
| `bf16`      | `+22%` | 34.49s (0.57m) | 2.94s | 33.62s (0.56m) |
| `compile`   | `+12%` | 31.14s (0.52m) | 1.34s | 30.82s (0.51m) |
| `muon`      | `+1%`  | 28.75s (0.48m) | 2.55s | 27.85s (0.46m) |
| `graph`     | `-1%`  | 27.95s (0.47m) | 1.62s | 27.41s (0.46m) |

percentages compare medians, rounded to whole numbers. 17 runs per knob on 07/10, against 31 runs of the record script from the same day (median 27.58s).

`attention` and `bf16` make the epochs faster, `compile` mostly shortens the first epoch (2.83s with it, 5.04s without). `muon` and `graph` are within noise on our node when removed one at a time (+1%, -1%). the pr's ablation agrees on `muon` for epoch time and measures a gain for `graph` on both hosts. both stay in.

## Comparison


![pr19, pr19+pr20, pr22 and pr20+pr22, 31 runs each](https://github.com/user-attachments/assets/7f346f8c-d77d-40bf-b980-1b025a15183c)
