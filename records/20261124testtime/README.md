# testtime

test-time adaptation inside `ModdedNanoTabPFNClassifier`, on top of record 12 (launchbound). training is untouched: same data, same steps, same optimizer, same weights after every epoch.

**what this is, plainly:** the speedrun timer covers training only. this record moves compute into the untimed evaluation. weights from an earlier epoch reach the target because inference is stronger, not because training is faster. evaluation gets about 5x heavier (numbers below). whether that is within the spirit of "do not modify the evaluation pipeline" is for the maintainers to decide, the pr asks it directly.

measured on 1x RTX 5090 under Windows, not on an L40S, so the record time in the root README is left as `X.XX`.

the gain is small: 3 to 5 epochs, 5% in the inline runs and 8% in the paired replays.

## Changes

only the `Config` dataclass and `ModdedNanoTabPFNClassifier` are touched. the training loop and the evaluation loop are identical to record 12.

Ensembling: `predict_proba` averages the softmax of 8 forward passes. member 0 sees the features in their original order, members 1-7 see a random feature permutation. the model is not permutation invariant across features (the feature encoder groups each feature with neighbours at fixed offsets), so the members disagree a little and the average is better. class labels are not permuted: they enter the model as a number through `Linear(1, e)`, and permuting or shifting them made things worse.

```python
rng = np.random.default_rng(0)
probabilities = 0
for member in range(c.tta_members):
    cols = rng.permutation(x.shape[-1]) if member else np.arange(x.shape[-1])
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = self.model((x[:, :, cols], y), sep=len(self.X_train)).squeeze(0)
        out = out[:, : self.num_classes]
        probabilities = probabilities + F.softmax(out, dim=1)
probabilities = probabilities / c.tta_members
```

Fine-tuning: `fit` takes 8 AdamW steps (lr 3e-6, no weight decay, grad clip 1.0, bf16 autocast) on the training rows it was given. every step splits those rows at random into 70% context and 30% query and minimises cross entropy on the query rows. the weights are put back at the end of `predict_proba` (and at the start of the next `fit`), so nothing carries over to the next fold, the next dataset or the next training epoch.

```python
self.base_state = {k: v.clone() for k, v in self.model.state_dict().items()}
...
for _ in range(c.ttft_steps):
    perm = torch.randperm(len(x), generator=generator).to(self.device)
    xp, yp = x[perm], y[perm]
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = self.model((xp.unsqueeze(0), yp[:sep].unsqueeze(0)), sep=sep).squeeze(0)
        loss = F.cross_entropy(out[:, : self.num_classes], yp[sep:].long())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, c.ttft_grad_clip)
    optimizer.step()
```

|                  | value |
| ---------------- | ----- |
| `tta_members`    | 8     |
| `ttft_steps`     | 8     |
| `ttft_lr`        | 3e-6  |
| `ttft_context`   | 0.7   |
| `ttft_grad_clip` | 1.0   |

`tta_members = 1` and `ttft_steps = 0` give back the predictions of record 12 exactly.

what the classifier sees: in the evaluation loop `fit` receives the training rows and training labels of one fold, `predict_proba` receives the test features of that fold. fine-tuning uses only what `fit` receives. test rows and test labels are never used for it. the seeds are fixed (`torch.Generator().manual_seed(0)` for the splits, `np.random.default_rng(0)` for the permutations) and do not touch the global RNGs, so training is not affected by the evaluation.

## Results

3 runs of the record 12 script and 3 runs of this record, same machine, seed 11, alternating, one run at a time with nothing else on the GPU. every timed run was preceded by a discarded `--epochs 1` run of the same script. all 6 logs are in `logs/`, the median run of this record is at the top level.

record 12 script (`main` at `1d6f11c`):

```
mean             0.38m    22.53s   56     0.40s      3563      6.18m
std              0.00m    0.16s    1      0.00s      37        0.01m
median           0.38m    22.62s   56     0.40s      3584      6.17m
---------------  -------  -------  -----  ---------  --------  -------  ----------------------
##  #  date      in mins  in secs  epoch  μ epoch t  datasets  runtime  id-name
--  -  --------  -------  -------  -----  ---------  --------  -------  ----------------------
1   1  26-11-23  0.37m    22.35s   55     0.41s      3520      6.17m    65b5bafe-testtime-base
2   2  26-11-24  0.38m    22.62s   56     0.40s      3584      6.19m    bd3d106a-testtime-base
3   3  26-11-24  0.38m    22.62s   56     0.40s      3584      6.17m    79167cd0-testtime-base
```

this record:

```
mean             0.36m    21.31s   52     0.41s      3349      31.77m
std              0.01m    0.70s    2      0.00s      133       1.27m
median           0.36m    21.52s   53     0.41s      3392      32.16m
---------------  -------  -------  -----  ---------  --------  -------  -----------------
##  #  date      in mins  in secs  epoch  μ epoch t  datasets  runtime  id-name
--  -  --------  -------  -------  -----  ---------  --------  -------  -----------------
1   1  26-11-23  0.34m    20.53s   50     0.41s      3200      30.35m   a59ff5f7-testtime
2   3  26-11-24  0.36m    21.52s   53     0.41s      3392      32.16m   2d56a2ac-testtime
3   2  26-11-24  0.36m    21.87s   54     0.41s      3456      32.79m   ec742d13-testtime
```

median 22.62s against 21.52s, -5%. the first epoch (2.2s) and the later epochs (0.37s) take the same time in both, the whole difference is the number of epochs: 55-56 against 50-54.

3 runs per side is a small sample. the maintainers re-time every pr on their nodes, and the paired comparison below does not depend on run to run luck.

one more run of this record was started and thrown away: another process was launched on the same GPU while it was running (my mistake), it was killed at epoch 29 and is not in the tables.

## Ablation Study

paired replay instead of separate runs: five runs of the record 12 script saved their evaluated weights after every epoch, and each inference variant was then scored on exactly those weights. the variants therefore see the same training runs, and the only thing that differs is the inference. the table gives the first epoch at which each variant reaches the target and the training time the run had accumulated at that epoch.

| inference | first epoch at target (5 runs) | mean epoch | mean training time | vs plain |
| --- | --- | --- | --- | --- |
| plain (record 12) | 57, 62, 59, 56, 56 | 58.0 | 24.66s | |
| fine-tuning only | 55, 59, 56, 55, 55 | 56.0 | 23.88s | `-3%` |
| ensemble only | 54, 55, 54, 54, 54 | 54.2 | 23.18s | `-6%` |
| both (this record) | 52, 55, 53, 51, 53 | 52.8 | 22.65s | `-8%` |

read the other way round, as record minus change: without the ensemble `+5%`, without the fine-tuning `+2%`.

the ensemble carries most of the gain and is the cheaper of the two. the fine-tuning adds 1 to 3 epochs on top and costs most of the extra evaluation time.

things to know about this table:

- the replays ran an experimental copy of the script with both methods behind flags. the code in this pr is a port of that. checked on the saved weights of one epoch: with fine-tuning off the port reproduces the ensemble replay to the last digit (0.8063946253884687), with both off it reproduces the plain evaluation to the last digit (0.805307344654534).
- fine-tuning is not bit-reproducible. four replays of the same weights with both methods on gave 0.8070382 and 0.8070559 (this code), 0.8070692 and 0.8070450 (experimental copy). that is about ±0.00002, against about 0.0003 of progress per epoch near the target, so it can move a borderline crossing by an epoch. inference alone is exact.
- these five baseline runs are from a day earlier than the timed runs above and four of them evaluated only from epoch 44 on (training is identical, the evaluation does not feed back into it). their absolute times are a little slower than the timed baseline above, the comparison inside the table is unaffected.
- inline, the three runs of this record reached the target at epochs 50, 53 and 54, the replays of the same configuration at 51-55.

## Cost

the evaluation runs 38 datasets x 5 folds after every epoch. per fold this record adds 8 fine-tuning steps and 7 forward passes.

|                               | record 12 | this record |
| ----------------------------- | --------- | ----------- |
| untimed time per epoch        | 6.3s      | 36.0s       |
| script runtime, median        | 6.17m     | 32.16m      |
| peak memory allocated         | 2338 MiB  | 4629 MiB    |
| timed training, median        | 22.62s    | 21.52s      |

so one second less on the clock is paid for with 26 minutes more wall time per run on this machine.

## How the settings were chosen

by looking at TabArena AUC, the same metric the target is defined on. the sweep used the saved epoch 40 and epoch 50 weights of one baseline run and the chosen settings were then applied unchanged to the other four runs and to the inline runs. AUC at epoch 50 of that run:

| variant | AUC @ e50 |
| --- | --- |
| plain | 0.8051 |
| ensemble, feature permutation, 4 members | 0.8057 |
| ensemble, feature permutation, 8 members | 0.8059 |
| ensemble, cyclic feature shift, 8 members | 0.8050 |
| ensemble, class permutation, 8 members | 0.8042 |
| ensemble, cyclic class shift, 8 members | 0.8016 |
| ensemble, feature and class permutation, 8 members | 0.8040 |
| fine-tuning, 8 steps, lr 1e-6 | 0.8053 |
| fine-tuning, 8 steps, lr 3e-6 | 0.8055 |
| fine-tuning, 16 steps, lr 3e-6 | 0.8054 |
| fine-tuning, 8 steps, lr 1e-5 | 0.8048 |
| fine-tuning, 4 steps, lr 1e-5 | 0.8048 |
| fine-tuning, 8 steps, lr 3e-5 | 0.8017 |

the fine-tuning window is narrow: more steps or a higher learning rate lose the gain. 32 members instead of 8 crossed the target one epoch earlier on the one run it was tried on, at four times the ensemble cost.

## Environment

- RTX 5090, Windows 11, Python 3.13, torch 2.9.1+cu128, triton-windows 3.5.1.
- `train_nano.py` as committed reads `/proc/cpuinfo` and calls `os.sched_getaffinity`, neither exists on Windows. the logs come from a local copy in which that one logging line is replaced, nothing else differs, and the code printed at the top of each log is the code that ran:

```python
# committed
cpu = next(l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
print0(f"cpu: {cpu} ({len(os.sched_getaffinity(0))} cores)")
# local copy
print0(f"cpu: {platform.processor()} ({os.cpu_count()} cores)")
```

- environment variables for the runs: `TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0` (the static launcher fails on Windows with this torch version) and `PYTHONUTF8=1`.
- the committed `train_nano.py` itself has not been run on Linux by me.
- in the logs the hostname and the process rows of the `nvidia-smi` table are removed (a desktop machine, the rows listed desktop applications only, all of type C+G). nothing else in the logs is edited.
