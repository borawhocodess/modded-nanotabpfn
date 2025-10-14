# modded-nanoTabPFN

This repository hosts the nanoTabPFN speedrun, in which we search for the fastest way to use ___TODO___ to train a tabular foundation model (nanoTabPFN) that attains ___TODO___ cross-entropy loss on the ___TODO___ validation set.

The code is derived from [nanoTabPFN](https://github.com/automl/nanoTabPFN) with the inspiration of [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).

This repo now contains a training algorithm which attains the target performance in:
* ___TODO___ minutes on ___TODO___ (baseline needed ___TODO___ minutes)
* ___TODO___ tokens (baseline needed ___TODO___)

with the help of the following techniques:
* ___TODO___


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

> *Pretrain a neural network to ≤___TODO___ validation loss on ___TODO___ using ___TODO___.*

| # | Record time | Date | Description | Log |
| - | - | - | - | - |
1 | ___TODO___ minutes | 14/10/25 | baseline | [log](records/...TODO...)

## Changelog

- 14/10/25: Created repository and moved the whole codebase.


## References

1. [S. Müller et al. Transformers Can Do Bayesian Inference. arXiv preprint (2021).](https://arxiv.org/abs/2112.10510)
2. [N. Hollmann et al. TabPFN: “A Transformer That Solves Small Tabular Classification Problems in a Second”. Presented at ICLR (2022)](https://arxiv.org/abs/2207.01848)
3. [N. Hollmann et al. Accurate predictions on small data with a tabular foundation model. Nature 637, 319–326 (2025).](https://doi.org/10.1038/s41586-024-08328-6)
4. [N. Erickson et al. TabArena: A Living Benchmark for Machine Learning on Tabular Data. arXiv preprint (2025).](https://arxiv.org/abs/2506.16791)
