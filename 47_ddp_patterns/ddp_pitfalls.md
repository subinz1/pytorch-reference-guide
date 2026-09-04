# DDP Pitfalls — Common Bugs and Fixes

Companion guide for [Module 47 — DDP Patterns](../47_ddp_patterns/).

DistributedDataParallel works well until it does not. Most production issues are
**process-group setup**, **uneven workloads**, or **incorrect accumulation** —
not exotic NCCL bugs. Use this as a debugging checklist.

## 1. Forgot `DistributedSampler` (or forgot `set_epoch`)

**Symptom:** All ranks train on the same data; metrics look “too good” or diverge oddly.

**Fix:**
```python
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
loader = DataLoader(dataset, sampler=sampler, shuffle=False)
# each epoch:
sampler.set_epoch(epoch)
```

Without `set_epoch`, shuffling is identical every epoch across ranks’ partitions.

## 2. Gradient accumulation without `no_sync()`

**Symptom:** Slower training than expected; redundant all-reduces each micro-batch.

**Fix:**
```python
for i, (x, y) in enumerate(micro_batches):
    sync_ctx = nullcontext() if i == last else model.no_sync()
    with sync_ctx:
        loss = model(x, y) / accum_steps
        loss.backward()
optimizer.step(); optimizer.zero_grad()
```

Only the last micro-batch should synchronize gradients.

## 3. Unused parameters

**Symptom:** Hang or error when some parameters skip the backward graph (conditional experts, unused heads).

**Fix:** `DDP(model, find_unused_parameters=True)` — works but adds overhead.
Better: restructure so all params participate, or use FSDP / custom hooks for MoE.

## 4. Logging / checkpointing only on rank 0… incorrectly

**Symptom:** Deadlocks — rank 0 writes while others enter a collective.

**Fix:** Every rank must hit the same collectives in the same order. Do I/O on
rank 0 **outside** collective regions, or barrier carefully:

```python
if rank == 0:
    torch.save(model.module.state_dict(), path)
dist.barrier()
```

## 5. Saving `model.state_dict()` with `module.` prefixes

**Symptom:** Load fails on single-GPU inference (`Unexpected key(s) in state_dict: module.`).

**Fix:** Save `model.module.state_dict()` (or strip the prefix on load).

## 6. NCCL timeout / hang

**Common causes:**
- One rank OOMs or crashes → others wait until timeout
- Uneven control flow (`if rank == 0: backward()`)
- Different numbers of collectives per rank
- Firewall / wrong `MASTER_ADDR` / InfiniBand misconfig

**Fix pattern:** Enable debug, shorten repro, assert equal step counts:

```bash
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun ...
```

## 7. BatchNorm in multi-GPU

**Symptom:** Accuracy drop vs single GPU — each rank sees a shard of the batch.

**Fix:** Prefer `SyncBatchNorm.convert_sync_batchnorm(model)` for small per-rank batches,
or use GroupNorm / LayerNorm when appropriate.

## 8. Mixing `DataParallel` habits with DDP

**Symptom:** Putting `.cuda()` on tensors incorrectly, or wrapping twice.

**Fix:**
- One process ↔ one GPU: `torch.cuda.set_device(local_rank)`
- Move model **before** wrapping: `model.cuda(local_rank); model = DDP(model, device_ids=[local_rank])`
- Never nest DP inside DDP

## 9. AMP scaler + DDP ordering

**Symptom:** Inf/NaN or skipped steps inconsistently across ranks.

**Fix:** Unscale / step identically on all ranks; avoid rank-only scaler updates.
BF16 often needs no scaler; FP16 does.

## 10. Evaluation without aggregating metrics

**Symptom:** Reported val accuracy is rank-local only.

**Fix:** `all_reduce` totals (correct counts / sums), then compute metrics on rank 0.

```python
metrics = torch.tensor([correct, total], device=device, dtype=torch.float64)
dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
```

## Quick Triage Order

1. Does every rank enter the same collectives each step?
2. Is data sharded (`DistributedSampler` + `set_epoch`)?
3. Does accumulation use `no_sync()` correctly?
4. Do checkpoints load without `module.` surprises?
5. Turn on `TORCH_DISTRIBUTED_DEBUG=DETAIL` and reproduce with 2 ranks.

## References

- Module script: `ddp_training.py`
- [DDP notes](https://pytorch.org/docs/stable/notes/ddp.html)
- [DDP tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- Module 10: [Distributed Training](../10_distributed/)
