# widthsort

feature width sorted batching by [@shounakb1](https://github.com/shounakb1) ([#20](https://github.com/borawhocodess/modded-nanotabpfn/pull/20)), the model and optimizer are unchanged, only the order of the datasets within an epoch.

every batch is sliced to the widest dataset in it, and feature width is uncorrelated with position in the prior dump, so at batch size 2 a narrow dataset is routinely padded to a wide one. his numbers: mean processed width 14.04 against a true mean of 11.08 over the 3648 datasets of a record run.

epochs get faster (0.84s to 0.72s), though it needs more of them to reach the target (57 to 64).

only the median run's log is included at the top level, all 31 runs are in `logs/`.

```
mean                         0.75m    63     0.72s      4020      8.26m
std                          0.06m    5      0.01s      322       0.66m
median                       0.76m    64     0.72s      4096      8.31m
---------------------------  -------  -----  ---------  --------  -------  -------------
##  #   date      hostname   in mins  epoch  μ epoch t  datasets  runtime  id-name
--  --  --------  ---------  -------  -----  ---------  --------  -------  -------------
1   4   26-09-23  dlc2gpu26  0.62m    51     0.73s      3264      6.72m    d433aeef-pr20
2   15  26-09-23  dlc2gpu37  0.68m    57     0.72s      3648      7.38m    d2a68392-pr20
3   7   26-09-23  dlc2gpu26  0.68m    57     0.72s      3648      7.50m    9ca87a0a-pr20
4   8   26-09-23  dlc2gpu26  0.68m    57     0.72s      3648      7.45m    903b4aaa-pr20
5   12  26-09-23  dlc2gpu26  0.69m    57     0.72s      3648      7.67m    582794d5-pr20
6   16  26-09-23  dlc2gpu12  0.69m    58     0.72s      3712      7.52m    debc883d-pr20
7   20  26-09-23  dlc2gpu37  0.70m    57     0.73s      3648      7.70m    bf9c4501-pr20
8   30  26-09-23  dlc2gpu05  0.70m    58     0.72s      3712      7.60m    ef276a39-pr20
9   2   26-09-23  dlc2gpu26  0.70m    58     0.73s      3712      7.77m    9ca9d46f-pr20
10  18  26-09-23  dlc2gpu26  0.70m    59     0.72s      3776      7.67m    0e29f8d8-pr20
11  17  26-09-23  dlc2gpu37  0.71m    59     0.72s      3776      7.67m    12bd9234-pr20
12  25  26-09-23  dlc2gpu37  0.72m    60     0.72s      3840      7.80m    498594ed-pr20
13  13  26-09-23  dlc2gpu26  0.74m    61     0.72s      3904      8.15m    9107e80d-pr20
14  14  26-09-23  dlc2gpu37  0.74m    61     0.73s      3904      8.15m    8d5b7d8d-pr20
15  22  26-09-23  dlc2gpu07  0.75m    63     0.71s      4032      8.23m    aa6c31a3-pr20
16  21  26-09-23  dlc2gpu37  0.76m    64     0.71s      4096      8.31m    59cb2d39-pr20
17  11  26-09-23  dlc2gpu26  0.76m    64     0.71s      4096      8.38m    1b8f629d-pr20
18  24  26-09-23  dlc2gpu07  0.78m    66     0.71s      4224      8.69m    03db2257-pr20
19  10  26-09-23  dlc2gpu37  0.78m    66     0.71s      4224      8.54m    20a867a0-pr20
20  31  26-09-23  dlc2gpu05  0.78m    66     0.71s      4224      8.61m    3cbca8c7-pr20
21  6   26-09-23  dlc2gpu26  0.79m    66     0.71s      4224      8.59m    8eb7f014-pr20
22  23  26-09-23  dlc2gpu37  0.79m    66     0.72s      4224      8.80m    a8a420cc-pr20
23  19  26-09-23  dlc2gpu26  0.79m    66     0.72s      4224      8.81m    210e2b68-pr20
24  9   26-09-23  dlc2gpu37  0.80m    66     0.73s      4224      8.79m    8021990c-pr20
25  5   26-09-23  dlc2gpu26  0.81m    68     0.71s      4352      8.88m    3b8e3b28-pr20
26  29  26-09-23  dlc2gpu37  0.82m    69     0.71s      4416      8.98m    2fc6be54-pr20
27  26  26-09-23  dlc2gpu07  0.82m    69     0.71s      4416      8.96m    7cbce9b7-pr20
28  28  26-09-23  dlc2gpu37  0.82m    68     0.72s      4352      9.17m    59210684-pr20
29  3   26-09-23  dlc2gpu37  0.82m    69     0.71s      4416      8.96m    c4178d25-pr20
30  1   26-09-23  dlc2gpu37  0.84m    70     0.72s      4480      9.30m    68a05ab5-pr20
31  27  26-09-23  dlc2gpu05  0.84m    71     0.71s      4544      9.32m    79bd8a85-pr20
```

note: all sub-minute timings here assume a warm `torch.compile` cache.

## Changes

Dataloader: datasets are sorted by feature count inside windows of `steps * batch_size`, which is exactly one epoch. every epoch still sees the same set of datasets, but both the pairing and the batch order change: similar widths share a batch, and each epoch runs from the narrowest batches to the widest.

```python
# before: walk the dump in file order
f["X"][pointer : end, :, :num_features]
# after:  walk it in width-sorted order within each epoch's window
sel = np.sort(order[pointer : end])
f["X"][sel, :, :num_features]
```

## Comparison

31 runs each of the previous record (pr19) and this one (pr19+pr20), bars show 95% bootstrap confidence intervals.

![pr19 vs pr19+pr20, 31 runs each](https://github.com/user-attachments/assets/58cda880-f5ca-40d5-84cf-c407123d8b6a)
