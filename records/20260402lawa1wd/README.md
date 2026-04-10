# LAWA + AdamW Weight Decay

Adds two orthogonal optimizer improvements on top of the ThinkingRows+RMSNorm base:

- **LAWA (Latest Weight Averaging)**: maintain a sliding window of the last K=10 epoch checkpoints; before every eval, average them into a temporary model, evaluate, then restore training weights. If jackpot is hit on the averaged model, save the averaged weights as the checkpoint.
- **AdamW weight_decay=0.01**: small L2 regularization on the non-Muon parameters.

Only the median run's log is included here. The median run is the one added to the general record table.

```
mean               3.48m    127    1.64s      8133      24.91m
std                0.30m    11     0.11s      688       2.20m
median             3.48m    131    1.59s      8384      24.21m
-----------------  -------  -----  ---------  --------  -------  -------------------
##  #   hostname   in mins  epoch  μ epoch t  datasets  runtime  id-name
--  --  ---------  -------  -----  ---------  --------  -------  -------------------
1   4   dlc2gpu01  2.99m    116    1.55s      7424      23.50m   8e9a0ba2-v26lawa1wd
2   6   dlc2gpu01  3.00m    113    1.59s      7232      22.39m   0e80df14-v26lawa1wd
3   24  dlc2gpu05  3.05m    118    1.55s      7552      20.88m   bbba7b74-v26lawa1wd
4   8   dlc2gpu05  3.06m    118    1.56s      7552      23.60m   afc647f6-v26lawa1wd
5   9   dlc2gpu01  3.13m    119    1.58s      7616      23.86m   f8bdb197-v26lawa1wd
6   21  dlc2gpu01  3.15m    119    1.59s      7616      21.25m   901527fe-v26lawa1wd
7   1   dlc2gpu01  3.29m    126    1.57s      8064      26.56m   70496a00-v26lawa1wd
8   11  dlc2gpu12  3.38m    113    1.80s      7232      23.76m   31fbb082-v26lawa1wd
9   13  dlc2gpu09  3.39m    111    1.83s      7104      22.32m   d53fd3f4-v26lawa1wd
10  10  dlc2gpu11  3.39m    115    1.77s      7360      23.23m   1097a413-v26lawa1wd
11  26  dlc2gpu05  3.39m    131    1.55s      8384      23.12m   eba30685-v26lawa1wd
12  23  dlc2gpu09  3.44m    133    1.55s      8512      23.38m   6adb8b0b-v26lawa1wd
13  15  dlc2gpu12  3.47m    116    1.79s      7424      24.21m   18b66c01-v26lawa1wd
14  3   dlc2gpu15  3.48m    137    1.52s      8768      27.50m   edb3469e-v26lawa1wd 
15  14  dlc2gpu17  3.51m    118    1.78s      7552      25.30m   df28c463-v26lawa1wd
16  16  dlc2gpu12  3.51m    117    1.80s      7488      24.46m   360f7bc6-v26lawa1wd
17  7   dlc2gpu05  3.52m    137    1.54s      8768      27.17m   690f2c45-v26lawa1wd
18  22  dlc2gpu09  3.53m    136    1.56s      8704      23.86m   384a5165-v26lawa1wd
19  27  dlc2gpu05  3.54m    137    1.55s      8768      24.11m   90385845-v26lawa1wd
20  5   dlc2gpu07  3.57m    140    1.53s      8960      25.88m   8597d0eb-v26lawa1wd
21  2   dlc2gpu17  3.61m    135    1.60s      8640      27.23m   eda58f0c-v26lawa1wd
22  25  dlc2gpu09  3.78m    149    1.52s      9536      26.02m   d4ef3028-v26lawa1wd
23  17  dlc2gpu17  3.80m    131    1.74s      8384      27.61m   71f23f49-v26lawa1wd
24  20  dlc2gpu17  3.84m    131    1.76s      8384      27.80m   6704b82b-v26lawa1wd
25  12  dlc2gpu09  3.88m    133    1.75s      8512      26.19m   45a84c14-v26lawa1wd
26  18  dlc2gpu11  3.99m    139    1.72s      8896      27.62m   9309454f-v26lawa1wd
27  19  dlc2gpu17  4.16m    143    1.74s      9152      29.80m   4d6e9eba-v26lawa1wd
```

## LAWA

LAWA was introduced in:

1. **Kaddour (2022)** — *"Stop Wasting My Time! Saving Days of ImageNet and BERT Training with Latest Weight Averaging"* — [arXiv:2209.14981](https://arxiv.org/abs/2209.14981). Proposes the algorithm: snapshot model weights at the end of each epoch, maintain a FIFO queue of the last k, average them for evaluation while keeping the original model training.

2. **Sanyal et al. (2023)** — *"Early Weight Averaging meets High Learning Rates for LLM Pre-training"* — [arXiv:2306.03241](https://arxiv.org/abs/2306.03241). Extends LAWA to LLM pre-training, adds `k_stepsize` for spacing between selected checkpoints, shows LAWA acts as a surrogate for LR decay.

LAWA also appears as an **experimental (disabled by default)** feature in the [parameter-golf](https://github.com/openai/parameter-golf) competition records — in [`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`](https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon) and [`2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072`](https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072) records.
