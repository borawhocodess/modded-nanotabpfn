# launchbound

launch-bound speedup by [@XingyuQu](https://github.com/XingyuQu) ([#22](https://github.com/borawhocodess/modded-nanotabpfn/pull/22)), the training algorithm is unchanged, only how the training step is executed. previous record 0.76m, this is -39% on our node.

on the L40S the step is limited by kernel launch overhead and not by gpu compute: about 500 small kernels per step at roughly 45 µs of host time each (his numbers). every change below cuts launches or removes redundant work.

his paired protocol is in the same pr: 23 paired rounds against record 10 on two runpod l40s hosts, -32.7% and -26.0% median, epochs-to-target unchanged. he estimated 0.53-0.58 min on our node, our re-timing lands at 0.46.

the feature width sorting of record 11 is taken out again: on top of this record it needed more epochs without making them faster (0.50m with it, 0.46m without, 31 runs each).

only the median run's log is included at the top level, all 31 runs are in `logs/`.

```
mean                         0.48m    59     0.49s      3788      7.43m
std                          0.07m    10     0.01s      617       1.33m
median                       0.46m    56     0.49s      3584      7.11m
---------------------------  -------  -----  ---------  --------  -------  ----------------------
##  #   date      hostname   in mins  epoch  μ epoch t  datasets  runtime  id-name
--  --  --------  ---------  -------  -----  ---------  --------  -------  ----------------------
1   3   26-10-07  dlc2gpu37  0.44m    54     0.48s      3456      6.52m    f373f082-pr22minuspr20
2   5   26-10-07  dlc2gpu37  0.44m    55     0.48s      3520      6.75m    3f93ec4b-pr22minuspr20
3   1   26-10-07  dlc2gpu37  0.44m    55     0.48s      3520      6.76m    6b8b7746-pr22minuspr20
4   8   26-10-07  dlc2gpu37  0.44m    55     0.49s      3520      6.65m    85ae4457-pr22minuspr20
5   9   26-10-07  dlc2gpu37  0.45m    56     0.48s      3584      6.85m    0dcb6135-pr22minuspr20
6   23  26-10-07  dlc2gpu11  0.45m    55     0.49s      3520      6.84m    649540a3-pr22minuspr20
7   17  26-10-07  dlc2gpu11  0.45m    54     0.50s      3456      7.33m    53562b4f-pr22minuspr20
8   12  26-10-07  dlc2gpu11  0.45m    55     0.49s      3520      6.74m    18ff1d05-pr22minuspr20
9   14  26-10-07  dlc2gpu11  0.45m    56     0.48s      3584      6.86m    f5987d38-pr22minuspr20
10  24  26-10-07  dlc2gpu11  0.46m    55     0.50s      3520      7.12m    201b14e1-pr22minuspr20
11  29  26-10-07  dlc2gpu11  0.46m    57     0.48s      3648      6.94m    23214b1c-pr22minuspr20
12  19  26-10-07  dlc2gpu11  0.46m    57     0.48s      3648      7.11m    bef7241f-pr22minuspr20
13  25  26-10-07  dlc2gpu11  0.46m    56     0.49s      3584      7.02m    ca913528-pr22minuspr20
14  30  26-10-07  dlc2gpu11  0.46m    56     0.49s      3584      7.03m    851a474a-pr22minuspr20
15  28  26-10-07  dlc2gpu11  0.46m    55     0.50s      3520      7.11m    199e3a57-pr22minuspr20
16  7   26-10-07  dlc2gpu37  0.46m    57     0.48s      3648      6.99m    1433cd61-pr22minuspr20
17  26  26-10-07  dlc2gpu33  0.46m    55     0.50s      3520      7.51m    53a31123-pr22minuspr20
18  15  26-10-07  dlc2gpu11  0.46m    56     0.50s      3584      6.99m    f960933d-pr22minuspr20
19  20  26-10-07  dlc2gpu11  0.46m    55     0.51s      3520      7.15m    536ee893-pr22minuspr20
20  27  26-10-07  dlc2gpu11  0.46m    57     0.49s      3648      6.96m    953ad497-pr22minuspr20
21  11  26-10-07  dlc2gpu11  0.47m    57     0.49s      3648      7.10m    ec475637-pr22minuspr20
22  16  26-10-07  dlc2gpu11  0.47m    56     0.51s      3584      7.24m    398e00af-pr22minuspr20
23  22  26-10-07  dlc2gpu07  0.48m    59     0.48s      3776      7.21m    efd3fb1f-pr22minuspr20
24  18  26-10-07  dlc2gpu11  0.48m    59     0.49s      3776      7.22m    61224b06-pr22minuspr20
25  4   26-10-07  dlc2gpu37  0.48m    60     0.48s      3840      7.27m    b6124a02-pr22minuspr20
26  2   26-10-07  dlc2gpu37  0.48m    60     0.48s      3840      7.32m    5dc0f50f-pr22minuspr20
27  13  26-10-07  dlc2gpu11  0.48m    60     0.48s      3840      7.87m    b8e14fdd-pr22minuspr20
28  21  26-10-07  dlc2gpu11  0.49m    60     0.49s      3840      7.36m    901d4a51-pr22minuspr20
29  6   26-10-07  dlc2gpu37  0.51m    64     0.48s      4096      7.76m    02fe71b0-pr22minuspr20
30  10  26-10-07  dlc2gpu11  0.72m    89     0.48s      5696      11.46m   ec7ac4e2-pr22minuspr20
31  31  26-10-07  dlc2gpu33  0.79m    100    0.48s      6400      13.14m   f0b64c6c-pr22minuspr20
```

note: all sub-minute timings here assume a warm `torch.compile` cache

## Changes

Feature attention: the attention across the columns of a row uses a custom Triton kernel. the sequence is at most 21 tokens (20 features and the target), one program per (row, head) does the forward pass and another the backward pass. tables with more than 32 columns (evaluation) fall back to SDPA.

```python
# before
x = F.scaled_dot_product_attention(q, k, v)
# after
x = feat_attn(q, k, v)
```

Precision: the residual stream stays in bf16 inside the transformer, before the thinking rows concat promoted it to fp32.

```python
src = src.to(torch.bfloat16)
```

Compilation: two settings stop the layer from recompiling, and the layer is compiled with autotuned matmuls and combined kernels.

```python
torch._dynamo.config.specialize_float = True
torch.fx.experimental._config.backed_size_oblivious = True
LAYER_OPTS = {"max_autotune": True, "combo_kernels": True}
```

Optimizer: Muon does the same update, but each group of equally shaped matrices is one compiled graph (`MuonFused`), and gradient clipping plus the Muon step are recorded once as a CUDA graph and replayed every step (`StepGraph`). Adam stays eager.

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

Record:
- launchbound stats: mean: 28.86s (0.48m) std: 4.46s median: 27.58s (0.46m)

Removing each change from the launchbound record:

| knob | launchbound - knob | mean | std | median |
| --- | --- | --- | --- | --- |
| `attention` | `+42%` | 40.41s (0.67m) | 6.92s | 39.08s (0.65m) |
| `bf16`      | `+22%` | 34.49s (0.57m) | 2.94s | 33.62s (0.56m) |
| `compile`   | `+12%` | 31.14s (0.52m) | 1.34s | 30.82s (0.51m) |
| `muon`      | `+1%`  | 28.75s (0.48m) | 2.55s | 27.85s (0.46m) |
| `graph`     | `-1%`  | 27.95s (0.47m) | 1.62s | 27.41s (0.46m) |

percentages compare medians, rounded to whole numbers. 17 runs per knob, 31 for the record.

`muon` (`MuonFused`) and `graph` (`StepGraph`) show no effect on our nodes when removed one at a time. his own ablation in the pr agrees on `muon` and has a gain for `graph` that depends on the host cpu.

## Comparison

31 runs each of record 10 (pr19), record 11 (pr19+pr20), this one (pr22) and this one with the feature width sorting kept (pr20+pr22), bars show 95% bootstrap confidence intervals. these are the first re-timing runs from 30/09, the table above is from a second set on 07/10.

![pr19, pr19+pr20, pr22 and pr20+pr22, 31 runs each](https://github.com/user-attachments/assets/7f346f8c-d77d-40bf-b980-1b025a15183c)
