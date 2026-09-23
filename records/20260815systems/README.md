# systems

systems-level speedup by [@tjeong117](https://github.com/tjeong117) ([#19](https://github.com/borawhocodess/modded-nanotabpfn/pull/19)), the training algorithm is unchanged, only how the training steps are executed. previous record 0.92m, this is -14% on our node.

his paired protocol and integrity notes are in the same pr: 15 paired rounds on modal l40s, -17.7% median, epochs-to-target unchanged (mwu p=0.615). he predicted 0.76-0.79 min on our node, our re-timing lands at 0.79.

found and verified with an autonomous agent harness.

only the median run's log is included at the top level, all 31 runs are in `logs/`.

```
mean                         0.81m    58     0.84s      3726      10.57m
std                          0.04m    3      0.01s      217       0.74m
median                       0.79m    57     0.84s      3648      10.68m
---------------------------  -------  -----  ---------  --------  -------  ----------------
##  #   date      hostname   in mins  epoch  μ epoch t  datasets  runtime  id-name
--  --  --------  ---------  -------  -----  ---------  --------  -------  ----------------
1   6   26-08-21  dlc2gpu34  0.75m    54     0.84s      3456      9.03m    75b2e8ea-systems
2   14  26-08-21  dlc2gpu30  0.77m    55     0.84s      3520      10.75m   b816a615-systems
3   28  26-08-21  dlc2gpu08  0.77m    55     0.84s      3520      10.18m   2f545c45-systems
4   17  26-08-21  dlc2gpu16  0.77m    55     0.84s      3520      10.45m   147ce097-systems
5   7   26-08-21  dlc2gpu05  0.77m    55     0.84s      3520      9.54m    36d614b8-systems
6   19  26-08-21  dlc2gpu17  0.77m    55     0.84s      3520      10.51m   7ea027c7-systems
7   13  26-08-21  dlc2gpu30  0.78m    56     0.83s      3584      10.84m   c8db5368-systems
8   16  26-08-21  dlc2gpu04  0.78m    56     0.83s      3584      9.68m    c6a710e9-systems
9   15  26-08-21  dlc2gpu30  0.78m    56     0.84s      3584      10.92m   ba42862a-systems
10  24  26-08-21  dlc2gpu16  0.78m    56     0.84s      3584      10.68m   7ced59da-systems
11  20  26-08-21  dlc2gpu31  0.79m    56     0.84s      3584      9.93m    db34a408-systems
12  29  26-08-21  dlc2gpu08  0.79m    56     0.85s      3584      10.43m   2f0103ce-systems
13  2   26-08-21  dlc2gpu06  0.79m    57     0.83s      3648      9.71m    a9cf1e1d-systems
14  12  26-08-21  dlc2gpu38  0.79m    57     0.83s      3648      10.08m   09e23c13-systems
15  23  26-08-21  dlc2gpu17  0.79m    56     0.85s      3584      10.76m   0a146133-systems
16  5   26-08-21  dlc2gpu06  0.79m    57     0.83s      3648      10.07m   da187cba-systems
17  11  26-08-21  dlc2gpu30  0.79m    57     0.84s      3648      11.01m   015c3811-systems
18  21  26-08-21  dlc2gpu31  0.80m    57     0.84s      3648      10.07m   fc73cd5b-systems
19  22  26-08-21  dlc2gpu17  0.81m    58     0.83s      3712      10.99m   febb89b0-systems
20  18  26-08-21  dlc2gpu16  0.81m    58     0.84s      3712      10.73m   54134ecc-systems
21  8   26-08-21  dlc2gpu05  0.82m    59     0.83s      3776      9.99m    895c199b-systems
22  25  26-08-21  dlc2gpu17  0.83m    59     0.84s      3776      11.14m   81721429-systems
23  1   26-08-21  dlc2gpu01  0.83m    60     0.83s      3840      9.53m    4b50acbe-systems
24  31  26-08-21  dlc2gpu08  0.84m    61     0.83s      3904      11.03m   f995e9f1-systems
25  3   26-08-21  dlc2gpu06  0.84m    61     0.83s      3904      10.47m   d7f68fe9-systems
26  10  26-08-21  dlc2gpu38  0.85m    61     0.84s      3904      10.91m   59041a58-systems
27  9   26-08-21  dlc2gpu38  0.87m    63     0.83s      4032      10.96m   a9de0d84-systems
28  4   26-08-21  dlc2gpu06  0.87m    63     0.83s      4032      10.91m   023f92c1-systems
29  27  26-08-21  dlc2gpu17  0.87m    63     0.83s      4032      11.94m   2aaef248-systems
30  30  26-08-21  dlc2gpu08  0.88m    64     0.83s      4096      11.56m   bc218607-systems
31  26  26-08-21  dlc2gpu16  0.95m    69     0.82s      4416      12.72m   e700f310-systems
```

note: all sub-minute timings here assume a warm `torch.compile` cache, a cold inductor cache costs 41-109 s in epoch 1 alone.

## Changes

Optimizer: Muon groups matrices by shape and orthogonalizes each group in a single batched Newton-Schulz call, with `torch._foreach_*` momentum and weight decay updates.

```python
# before: one zeropower call per parameter
# after:  stack same-shape grads -> zeropower_via_newtonschulz5_batched -> _foreach_add_
```

Dataloader: a producer thread reads the prior dump into pinned memory and a bounded queue, batches are copied to the GPU with `non_blocking=True`, and the NaN check moves to the CPU, removing the per-step GPU syncs.

```python
# before if torch.isnan(x).any() or torch.isnan(y).any(): continue
# after  if not full_data["valid"]: continue
```

Architecture: the datapoint attention runs a single SDPA over all query rows instead of splitting train and test queries, both attended to the same train keys and values.

```python
# before cat([sdpa(q_left, k_train, v_train), sdpa(q_right, k_train, v_train)])
# after  sdpa(q, k_train, v_train)
```
