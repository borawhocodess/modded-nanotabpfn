# modded-nanoTabPFN

This repository hosts the nanoTabPFN speedrun, in which we search for the fastest way to use 1 NVIDIA L40S to train a tabular foundation model (nanoTabPFN) that beats Random Forest on [TabArena](https://github.com/autogluon/tabarena) datasets.

The code is derived from [nanoTabPFN](https://github.com/automl/nanoTabPFN) with the inspiration of [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).

This repo now contains a training algorithm which attains the target performance in:
* 10.10 minutes on 1xL40S (baseline needed 74.32)
* 13184 synthetic datasets (baseline needed 80576)

with the help of the following techniques:
* Muon optimizer

also these were tried but did not lead to improvements:
* Xavier initialization


## Running the current record

To run the current record, run the following commands.
```bash
git clone https://github.com/borawhocodess/modded-nanotabpfn.git
cd modded-nanotabpfn
uv sync
./run.sh
```


## Record history

The following is the historical progression of speed records for the following task:

> *Pretrain a neural network to ≤0.8068462330697953 validation average ROC AUC on subsampled TabArena using 1 NVIDIA L40S.*

Note: The 0.8068462330697953 target was selected to match the performance of Random Forest on the same subsampled TabArena evaluation.

| # | Record time | Date | Description | Log | Contributors
| - | - | - | - | - | - |
| 1 | 74.32 minutes | 31/01/26 | Baseline | [log](records/20260131baseline/260205-223919-fad38238-baseline-log.txt) | @borawhocodess, nanotabpfn contributors |
| 2 | 54.41 minutes | 02/02/26 | Muon optimizer | [log](records/20260131baseline/260205-224113-34e556ea-muon-log.txt) | @borawhocodess |
| 3 | 10.10 minutes | 04/02/26 | SDPA, bf16, higher LR, wider embeddings, fewer heads | [log](records/20260204carter/260205-225951-d60000eb-carter-log.txt) | @carterprince |


## Evaluation details

Evaluation is on all of 38 TabArena classification tasks.

- Subsampling:
  - if >100 features, randomly select 100
  - if >1000 rows, randomly select 1000 (stratiﬁed by class labels)
- Cross-validation:
  - 5-fold StratifiedKFold with shuffling
  - class labels are encoded with integers per fold
- Preprocessing (per fold, fit on train only):
  - constant columns: dropped
  - numeric columns: numeric coercion + mean imputation
  - categorical columns: ordinal encoding + most-frequent imputation
- Metric:
  - binary or one-vs-rest ROC AUC
  - average over all tasks


## References

1. [S. Müller et al. Transformers Can Do Bayesian Inference. arXiv preprint (2021).](https://arxiv.org/abs/2112.10510)
2. [N. Hollmann et al. TabPFN: “A Transformer That Solves Small Tabular Classification Problems in a Second”. Presented at ICLR (2022)](https://arxiv.org/abs/2207.01848)
3. [N. Hollmann et al. Accurate predictions on small data with a tabular foundation model. Nature 637, 319–326 (2025).](https://doi.org/10.1038/s41586-024-08328-6)
4. [N. Erickson et al. TabArena: A Living Benchmark for Machine Learning on Tabular Data. arXiv preprint (2025).](https://arxiv.org/abs/2506.16791)
5. [A. Pfefferle et al. nanoTabPFN: A Lightweight and Educational Reimplementation of TabPFN. arXiv preprint (2025).](https://arxiv.org/abs/2511.03634)
6. [K. Jordan et al. Muon: An optimizer for hidden layers in neural networks (2024).](https://kellerjordan.github.io/posts/muon/)
