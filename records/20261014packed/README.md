# packed

column packing, last-layer query-only rows, untimed kernel warmup and a persistent prefetching dataloader on top of record 12. on our RTX PRO 6000 Blackwell node the record script needs 0.28-0.29m, this needs 0.20-0.21m, about -29%.

unlike the L40S of record 12, this GPU is busy for nearly the whole step (about 9.2 ms of GPU time per 8.7 ms step), so cutting launches does not help here (a full CUDA graph of forward + backward was slower, 12.7 ms/step, because it needs masked attention over all rows). the gains below either do less GPU work or move fixed costs out of the timed region.

epoch counts stay where the record script lands (55-57), all of the gain comes from faster epochs: 0.222s instead of about 0.275s, and a first epoch of 0.25s instead of 1.95s.

only the median run's log is included.

```
                     runs  epochs  first epoch  mean epoch  training   record
record 12 script     4     55-59   1.91-1.99s   0.273-0.278s  16.77-17.72s  0.28-0.29m
this record          3     55-57   0.24-0.26s   0.222s        12.21-12.65s  0.20-0.21m
```

all runs seed 11, RTX PRO 6000 Blackwell Server Edition, warm `torch.compile` cache. the slowest run here is 4.1s faster than the fastest run of the record script. three runs is a small sample, a re-timing with more runs on the reference node is still to do.

## Changes

Column packing: the two tables of a batch are no longer padded to the wider one's feature count. they are put side by side along the column axis as one table, and feature attention runs block diagonal with a Triton kernel (`packed_attn_fwd`), so each table only attends to its own columns. row attention is per column, so it needs no change. the padding columns used to go through every layer and into the feature pooling, so this both skips work and changes what the model computes during training (each table is now encoded exactly as it would be alone). evaluation still uses the unpacked path.

```python
# before
x_src = self.feature_encoder(x_src, sep)
# after
f0 = self.feature_encoder(x_src[:1, :, :n0], sep)
f1 = self.feature_encoder(x_src[1:2, :, :n1], sep)
src = torch.cat([f0, ys[:1], f1, ys[1:2]], 2)
```

this assumes `batch_size == 2`, as in the config.

Query-only last layer: only the test rows of the last layer reach the decoder, and the context rows enter row attention as keys and values only. the last layer computes queries, residual and MLP for the test rows alone. this is exact.

```python
if query_only:
    q = q[:, :, sep:, :]
    res = res[:, sep:, :]
    r = r - sep
```

Kernel warmup: before the timer starts, one training step per feature width on random tables loads the compiled kernels and captures the optimizer graph, then the weights, the Muon momentum and the AdamW schedule-free state go back to their initial values (`warmup_kernels`, `reset_training_state`). no training data is read and nothing learned survives. this is the warmup modded-nanogpt does, it takes the first epoch from about 1.95s to 0.25s.

Dataloader: one producer thread for the whole run instead of one per epoch, so it reads the next epoch's batches (same order) while the model is evaluated. this takes the per-epoch host wait (about 16ms) out of the timed loop.

## Things that did not help on this node

- full CUDA graph of forward + backward + optimizer (static shapes, masked row attention): 12.7 ms/step against 8.7 ms
- schedule-free AdamW inside a CUDA graph: did not capture, and Adam is about 1% of GPU time
- cuDNN attention instead of FlashAttention-2 for row attention: within noise
- computing row attention keys and values only for context rows in every layer: no faster than the record script, slicing the projection weights costs more than it saves
- batch size 1: faster epochs but needs far more of them
