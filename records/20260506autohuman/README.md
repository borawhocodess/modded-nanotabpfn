# autoresearch

changes found by adapting [autoresearch](https://github.com/karpathy/autoresearch) over multiple runs with human intervention. 

Only the median run's log is included here. The median run is the one added to the general record table.

```
mean                         0.93m    57     0.97s      3646      9.41m
std                          0.04m    2      0.02s      124       0.38m
median                       0.92m    57     0.97s      3648      9.42m
---------------------------  -------  -----  ---------  --------  -------  ------------------
##  #   date      hostname   in mins  epoch  μ epoch t  datasets  runtime  id-name
--  --  --------  ---------  -------  -----  ---------  --------  -------  ------------------
1   28  26-05-06  dlc2gpu05  0.87m    54     0.97s      3456      8.54m    c2cf2aca-autohuman
2   20  26-05-06  dlc2gpu17  0.88m    55     0.96s      3520      8.73m    de07f7cf-autohuman
3   5   26-05-06  dlc2gpu01  0.89m    55     0.97s      3520      9.09m    2f4aaa47-autohuman
4   9   26-05-06  dlc2gpu05  0.89m    54     0.99s      3456      9.03m    b7e23b18-autohuman
5   11  26-05-06  dlc2gpu01  0.89m    55     0.97s      3520      9.23m    527e280a-autohuman
6   23  26-05-06  dlc2gpu01  0.89m    55     0.97s      3520      9.22m    557e23f8-autohuman
7   25  26-05-06  dlc2gpu01  0.89m    55     0.97s      3520      9.19m    ceb3be6f-autohuman
8   26  26-05-06  dlc2gpu01  0.89m    55     0.97s      3520      9.15m    f68aff00-autohuman
9   18  26-05-06  dlc2gpu05  0.90m    56     0.96s      3584      9.02m    ca8b3be8-autohuman
10  14  26-05-06  dlc2gpu01  0.90m    56     0.97s      3584      9.22m    09ede188-autohuman
11  8   26-05-06  dlc2gpu05  0.90m    56     0.97s      3584      8.98m    67fc3f41-autohuman
12  22  26-05-06  dlc2gpu01  0.90m    56     0.97s      3584      9.30m    74fc131b-autohuman
13  2   26-05-06  dlc2gpu01  0.92m    57     0.96s      3648      9.42m    4519f068-autohuman
14  13  26-05-06  dlc2gpu01  0.92m    57     0.96s      3648      9.50m    757685bf-autohuman
15  15  26-05-06  dlc2gpu01  0.92m    57     0.97s      3648      9.49m    8c8facf1-autohuman
16  10  26-05-06  dlc2gpu01  0.92m    57     0.97s      3648      9.64m    589ea1cf-autohuman
17  6   26-05-06  dlc2gpu01  0.92m    57     0.97s      3648      9.49m    c9c55713-autohuman
18  19  26-05-06  dlc2gpu05  0.92m    57     0.97s      3648      9.20m    32e1064b-autohuman
19  1   26-05-06  dlc2gpu01  0.92m    57     0.97s      3648      9.41m    e2904ad9-autohuman
20  7   26-05-06  dlc2gpu03  0.93m    57     0.98s      3648      9.47m    7cae85f0-autohuman
21  3   26-05-06  dlc2gpu01  0.93m    57     0.98s      3648      9.43m    a620fe56-autohuman
22  16  26-05-06  dlc2gpu03  0.93m    57     0.98s      3648      9.53m    a334e9ce-autohuman
23  24  26-05-06  dlc2gpu01  0.94m    58     0.97s      3712      9.62m    a5bd799b-autohuman
24  29  26-05-06  dlc2gpu05  0.94m    58     0.97s      3712      9.49m    6747e820-autohuman
25  21  26-05-06  dlc2gpu01  0.94m    59     0.96s      3776      9.82m    6b19ed70-autohuman
26  12  26-05-06  dlc2gpu01  0.95m    59     0.96s      3776      9.96m    6f69d83f-autohuman
27  4   26-05-06  dlc2gpu01  0.95m    59     0.97s      3776      9.77m    25e61e94-autohuman
28  17  26-05-06  dlc2gpu05  0.99m    61     0.97s      3904      10.18m   9ac584cb-autohuman
29  27  26-05-06  dlc2gpu01  0.99m    62     0.96s      3968      10.25m   e68e24ae-autohuman
30  30  26-05-06  dlc2gpu01  1.03m    57     1.08s      3648      9.23m    8cb97025-autohuman
31  31  26-05-06  dlc2gpu01  1.05m    61     1.03s      3904      10.01m   3dc28f54-autohuman
```

## Changes

HPO:

|                    | before | after |
| ------------------ | ------ | ----- |
| `batch_size`       | 1      | 2     |
| `steps`            | 64     | 32    |
| `l`                | 6      | 5     |
| `thinking_rows`    | 16     | 24    |
| `feature_group_size` | 3    | 5     |
| `muon_momentum`    | 0.95   | 0.96  |
| `grad_clip`        | 1.0    | 2.0   |


Optimizer: added weight decay to Muon.

```python
# p.data.mul_(1 - lr * group['weight_decay'])
```


Architecture: the Decoder is fed the mean over feature tokens at test rows instead of target tokens.

```python
# before out[:, sep:, -1, :]
# after out[:, sep:, :-1, :].mean(dim=2)
```
