# x posts

every record announcement on x, in order. text is copied as posted.

## 1 - 74.32 minutes - Baseline

[03/02/26](https://x.com/boratwits/status/2018694333654860275)

> Inspired by @kellerjordan0's modded-nanogpt, I present to you speedrunning repository for PFNs (NanoTabPFN), where we search the fastest way to train a TabPFN that beats Random Forest on TabArena.
>
> To make it as accessible as possible, everything is in one short script, which makes it easy to try new ideas and also LLM-friendly :) The log file produced by the current script contains all the details one needs to reproduce the record.
>
> It is currently a small ~4M parameter setting compared to today's big models. However, I still believe in the power of experimentation on small scale.
>
> I am sharing this in the hope of collaboration/competition, and hoping that some interesting things will come out of it.
>
> Thanks,
>
> SBO
>
> https://github.com/borawhocodess/modded-nanotabpfn

## 2 - 54.41 minutes - Muon optimizer

[09/02/26](https://x.com/boratwits/status/2020941276946833615)

> Update / Correction:
> The previous speedrun used a sparse / windowed eval schedule, which could miss earlier jackpot crossings and bias timing. 
>
> Updated Record: 54.41 mins, 45824 synthetic datasets
> Baseline: 74.25 mins, 80576 synthetic datasets

![](https://pbs.twimg.com/media/HAvRqcna8AAMWpJ.png?name=orig)

## 3 - 10.10 minutes - SDPA, bf16, higher LR, wider embeddings, fewer heads

[09/02/26](https://x.com/boratwits/status/2020943088428917240)

> New NanoTabPFN training speed record: Beating RF in 10.10 minutes
>
> Previous record: 54.41 minutes
> Changelog: 
> - Scaled Dot-Product Attention rewrite with explicit QKV
> - Pre-norm transformer blocks
> - bfloat16 autocast
> - Increase learning rate
> - Increase embedding size 
> - Reduce attention heads 
>
> This record is by @carterprince03

![](https://pbs.twimg.com/media/HAvU_0YXsAAzssZ.png?name=orig)

## 4 - 9.26 minutes - Batched Muon, compiled forward

[11/02/26](https://x.com/boratwits/status/2021388220282568828)

> New NanoTabPFN speedrun record: Beating RF in 9.26 minutes
>
> Previous record: 10.10 minutes
> Changelog: 
> - Batched Muon zeropower update for grouped QKV matrices
> - Compile TransformerEncoderLayer forward
>
> This record is by @carterprince03

## 5 - 7.57 minutes - Exponential decay of residual stream

[19/03/26](https://x.com/boratwits/status/2034426880208593057)

> New NanoTabPFN speedrun record: Beating RF in 7.57 minutes
>
> Previous record: 9.26 minutes
> Changelog:  
> - Exponential decay of residual stream across layers (marked as red blocks)

![](https://pbs.twimg.com/media/HDu7V-cXQAE6lPd.jpg?name=orig)

## 6 - 3.88 minutes - RMSNorm, ThinkingRows

[28/03/26](https://x.com/boratwits/status/2038018033763918087)

> New NanoTabPFN speedrun record: Beating RF in 3.88 minutes
>
> Previous record: 7.57 minutes
> Changelog:  
> - Lower precision RMSNorm
> - Prepend 16 learnable Thinking Rows 
>
> changes inspired by TabPFN 2.6 release

![](https://pbs.twimg.com/media/HEh-PPwbsAAamgv.jpg?name=orig)

## 7 - 3.48 minutes - LAWA, AdamW weight decay

[10/04/26](https://x.com/boratwits/status/2042729853821022644)

> New NanoTabPFN speedrun record: Beating RF in 3.48 minutes
>
> Previous record: 3.88 minutes
> Changelog:
> - LAWA (Latest Weight Averaging)
> - AdamW weight decay

## 8 - 2.15 minutes - Repeated feature grouping

[11/04/26](https://x.com/boratwits/status/2043047953502290052)

> New NanoTabPFN speedrun record: Beating RF in 2.15 minutes
>
> Previous record: 3.48 minutes
> Changelog:
> - Repeated feature grouping
>
> Inspired by TabICLv2 architecture

![](https://pbs.twimg.com/media/HFpdOfQXwAAHbJp.jpg?name=orig)

## 9 - 0.92 minutes - autoresearch HPO, Muon weight decay, mean feature pooling

[07/05/26](https://x.com/boratwits/status/2052199021775647173)

> New NanoTabPFN speedrun record: Beating RF in 0.92 minutes
>
> Previous record: 2.15 minutes
> Changelog:
> - Feed mean of feature tokens to decoder
> - Add decoupled weight decay to Muon
> - Reduce transformer layers
> - Increase feature grouping size
> - Increase thinking rows
> - Increase Muon momentum  
> - Increase batch size
> - Increase gradient clipping
>
> done with autoresearch using @claudeai

quoting [@karpathy](https://x.com/karpathy/status/2030371219518931079):

> I packaged up the "autoresearch" project into a new self-contained minimal repo if people would like to play over the weekend. It's basically nanochat LLM training core stripped down to a single-GPU, one file version of ~630 lines of code, then:
>
> - the human iterates on the prompt (.md)
> - the AI agent iterates on the training code (.py)
>
> The goal is to engineer your agents to make the fastest research progress indefinitely and without any of your own involvement. In the image, every dot is a complete LLM training run that lasts exactly 5 minutes. The agent works in an autonomous loop on a git feature branch and accumulates git commits to the training script as it finds better settings (of lower validation loss by the end) of the neural network architecture, the optimizer, all the hyperparameters, etc. You can imagine comparing the research progress of different prompts, different agents, etc.
>
> https://github.com/karpathy/autoresearch
> Part code, part sci-fi, and a pinch of psychosis :)

## 10 - 0.79 minutes - Shape-grouped Newton-Schulz, producer-thread dataloader, single datapoint SDPA

[23/09/26](https://x.com/boratwits/status/2102855207675978023)

> New NanoTabPFN speedrun record: Beating RF in 0.79 minutes
>
> Previous record: 0.92 minutes
> Changelog:
> - Group Muon matrices by shape
> - Prefetch batches in a producer thread
> - Move NaN check to CPU
> - Merge datapoint SDPA calls
>
> This record is by @TommyLovesSosa

![](https://pbs.twimg.com/media/HS7XeUFWEAAvqZf.jpg?name=orig)

## 11 - 0.76 minutes - Feature width sorted batching

[07/10/26](https://x.com/boratwits/status/2107816065136767260)

New NanoTabPFN speedrun record: Beating RF in 0.76 minutes

Previous record: 0.79 minutes
Changelog:
- Sort datasets by increasing feature width within each epoch

This record is by @ShounakBanerj15

![](https://pbs.twimg.com/media/HUB3WxKXAAA6aMu.png?name=orig)
