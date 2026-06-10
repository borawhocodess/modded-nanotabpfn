# modded-nanoTabPFN

> **Disclaimer:** The training code has gotten a bit crowded. If you want a simpler version to start from, checking out commit [b0f29b7](https://github.com/borawhocodess/modded-nanotabpfn/commit/b0f29b7) is a good option, or do `git log --oneline -- train_nano.py` to pick the point that suits :)

This repository hosts the *nanoTabPFN speedrun*, in which we (collaboratively|competitively) search for the fastest way to use 1 NVIDIA L40S to train a tabular foundation model (nanoTabPFN) that beats Random Forest on [TabArena](https://github.com/autogluon/tabarena) datasets.

The code is derived from [nanoTabPFN](https://github.com/automl/nanoTabPFN) with the inspiration of [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).

This repo now contains a training algorithm which attains the target performance in:
* 0.92m minutes on 1xL40S (baseline needed 74.32)
* 3648 synthetic datasets (baseline needed 80576)

This improvement in training speed has been brought about by the following techniques:
* Muon optimizer
* Batched Muon zeropower update for grouped QKV matrices
* Scaled Dot-Product Attention rewrite with explicit QKV
* Pre-norm transformer blocks
* Compile TransformerEncoderLayer forward
* bfloat16 autocast in training and inference
* Set float32 matmul precision to high
* Increase learning rate from 1e-4 to 1e-3
* Increase embedding size from 192 to 256
* Reduce attention heads from 6 to 4
* Exponential decay of residual stream across layers
* Lower precision RMSNorm
* Prepend 24 learnable Thinking Rows
* Latest Weight Averaging of 10 checkpoints
* Set AdamWScheduleFree weight decay to 0.01
* Repeated feature grouping (group size 5)
* Reduce transformer layers from 6 to 5
* Increase batch size from 1 to 2
* Add decoupled weight decay 0.1 to Muon
* Increase Muon momentum from 0.95 to 0.96
* Increase gradient clip from 1.0 to 2.0
* Feed mean of test feature embeddings into output decoder


## Running the current record

To run the current record, run the following commands.
```bash
git clone https://github.com/borawhocodess/modded-nanotabpfn.git
cd modded-nanotabpfn
uv sync
./run.sh
```


## Record history

The following is the historical progression of speed records for the following competitive task:

> *Pretrain a neural network to ≤0.8068462330697953 validation average ROC AUC on subsampled TabArena using 1 NVIDIA L40S.*

Note: The 0.8068462330697953 target was selected to match the performance of Random Forest on the same subsampled TabArena evaluation.

| # | Record time | Date | Description | Links | Contributors
| - | - | - | - | - | - |
| 1 | 74.32 minutes | 31/01/26 | Baseline | [log](records/20260131baseline/260205-223919-fad38238-baseline-log.txt),[read](records/20260131baseline/README.md),[X](https://x.com/boratwits/status/2018694333654860275) | @borawhocodess, nanotabpfn contributors |
| 2 | 54.41 minutes | 02/02/26 | Muon optimizer | [log](records/20260202muon/260205-224113-34e556ea-muon-log.txt),[read](records/20260202muon/README.md),[PR](https://github.com/borawhocodess/modded-nanotabpfn/pull/1),[X](https://x.com/boratwits/status/2020941276946833615) | @borawhocodess |
| 3 | 10.10 minutes | 04/02/26 | SDPA, bf16, higher LR, wider embeddings, fewer heads | [log](records/20260204carter/260205-225951-d60000eb-carter-log.txt),[read](records/20260204carter/README.md),[PR](https://github.com/borawhocodess/modded-nanotabpfn/pull/4),[X](https://x.com/boratwits/status/2020943088428917240) | @carterprince |
| 4 | 9.26 minutes  | 08/02/26 | Batched Muon, compiled forward | [log](records/20260208batchedmuon/260209-204142-79d70f03-log.txt),[read](records/20260208batchedmuon/README.md),[PR](https://github.com/borawhocodess/modded-nanotabpfn/pull/7),[X](https://x.com/boratwits/status/2021388220282568828) | @carterprince |
| 5 | 7.57 minutes | 16/03/26 | Exponential decay of residual stream | [log](records/20260310residualdecay/260311-154253-a787618f-new2-s11-log.txt),[read](records/20260310residualdecay/README.md),[PR](https://github.com/borawhocodess/modded-nanotabpfn/pull/12),[X](https://x.com/boratwits/status/2034426880208593057) | @borawhocodess |
| 6 | 3.88 minutes | 28/03/26 | RMSNorm, ThinkingRows | [log](records/20260328rmsthink/260328-011343-4ba701de-rmsthink-s11-log.txt),[read](records/20260328rmsthink/README.md),[PR](https://github.com/borawhocodess/modded-nanotabpfn/pull/14),[X](https://x.com/boratwits/status/2038018033763918087) | @borawhocodess |
| 7 | 3.48 minutes | 02/04/26 | LAWA, AdamW weight decay | [log](records/20260402lawa1wd/260401-183226-edb3469e-v26lawa1wd-s11-log.txt),[read](records/20260402lawa1wd/README.md),[PR](https://github.com/borawhocodess/modded-nanotabpfn/pull/16),[X](https://x.com/boratwits/status/2042729853821022644) | @borawhocodess |
| 8 | 2.15 minutes | 11/04/26 | Repeated feature grouping | [log](records/20260411featuregroup/260411-142657-1686dd83-featuregroup-s11-log.txt),[read](records/20260411featuregroup/README.md),[PR](https://github.com/borawhocodess/modded-nanotabpfn/pull/17),[X](https://x.com/boratwits/status/2043047953502290052) | @borawhocodess |
| 9 | 0.92 minutes | 06/05/26 | autoresearch HPO, Muon weight decay, mean feature pooling | [log](records/20260506autohuman/260506-163637-589ea1cf-autohuman-log.txt),[read](records/20260506autohuman/README.md),[PR](https://github.com/borawhocodess/modded-nanotabpfn/pull/18),[X](https://x.com/boratwits/status/2052199021775647173) | @borawhocodess |


## Rules

New records must:
- Not modify the evaluation pipeline.
- Not load any pretrained weights.
- Run faster than prior record when baselined on the same hardware with the same seed.

Other than that, anything and everything is fair game!


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


## Citation

This repo now has an accompanying paper, *Speedrunning Tabular Foundation Model Pretraining* ([arXiv](https://arxiv.org/abs/2606.03681)), accepted ([OpenReview](https://openreview.net/forum?id=QT1ySCPeW3)) at the [FM4SD](https://icml-structured-fm-workshop.github.io) workshop at ICML 2026.

```bibtex
@misc{ozturk2026speedrunningtabularfoundationmodel,
      title={Speedrunning Tabular Foundation Model Pretraining},
      author={Salih Bora Ozturk and Alexander Pfefferle and Frank Hutter},
      year={2026},
      eprint={2606.03681},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.03681},
}
```


## References

1. [S. Müller et al. Transformers Can Do Bayesian Inference. arXiv preprint (2021).](https://arxiv.org/abs/2112.10510)
2. [N. Hollmann et al. TabPFN: “A Transformer That Solves Small Tabular Classification Problems in a Second”. Presented at ICLR (2022)](https://arxiv.org/abs/2207.01848)
3. [N. Hollmann et al. Accurate predictions on small data with a tabular foundation model. Nature 637, 319–326 (2025).](https://doi.org/10.1038/s41586-024-08328-6)
4. [N. Erickson et al. TabArena: A Living Benchmark for Machine Learning on Tabular Data. arXiv preprint (2025).](https://arxiv.org/abs/2506.16791)
5. [A. Pfefferle et al. nanoTabPFN: A Lightweight and Educational Reimplementation of TabPFN. arXiv preprint (2025).](https://arxiv.org/abs/2511.03634)
6. [K. Jordan et al. Muon: An optimizer for hidden layers in neural networks (2024).](https://kellerjordan.github.io/posts/muon/)
7. [R. Xiong et al. On Layer Normalization in the Transformer Architecture. ICML 2020. arXiv:2002.04745.](https://arxiv.org/abs/2002.04745)
8. [PyTorch docs: `torch.nn.functional.scaled_dot_product_attention`.](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
9. [PyTorch docs: Automatic Mixed Precision (`torch.autocast`).](https://pytorch.org/docs/stable/amp.html)
10. [PyTorch docs: `torch.set_float32_matmul_precision`.](https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html)
11. [PyTorch docs: `torch.compile`.](https://pytorch.org/docs/stable/generated/torch.compile.html)
12. [PyTorch docs: Dealing with Recompilations.](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.recompilation.html)
13. [B. Zhang & R. Sennrich. Root Mean Square Layer Normalization. NeurIPS 2019. arXiv:1910.07467.](https://arxiv.org/abs/1910.07467)
14. [L. Grinsztajn et al. TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models. arXiv:2511.08667 (2026).](https://arxiv.org/abs/2511.08667)
15. [J. Kaddour. Stop Wasting My Time! Saving Days of ImageNet and BERT Training with Latest Weight Averaging. arXiv:2209.14981 (2022).](https://arxiv.org/abs/2209.14981)
16. [S. Sanyal et al. Early Weight Averaging meets High Learning Rates for LLM Pre-training. arXiv:2306.03241 (2023).](https://arxiv.org/abs/2306.03241)
17. [I. Loshchilov & F. Hutter. Decoupled Weight Decay Regularization. ICLR 2019. arXiv:1711.05101.](https://arxiv.org/abs/1711.05101)
18. [A. Defazio et al. Schedule-Free Learning: A New Way to Train. arXiv:2405.15682 (2024).](https://arxiv.org/abs/2405.15682)
19. [J. Qu et al. TabICLv2: A better, faster, scalable, and open tabular foundation model. arXiv:2602.11139 (2026).](https://arxiv.org/abs/2602.11139)
20. [A. Karpathy. autoresearch: AI agents running research on single-GPU nanochat training automatically (2026).](https://github.com/karpathy/autoresearch)
