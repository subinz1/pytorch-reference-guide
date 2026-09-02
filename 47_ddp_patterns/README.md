# Module 47: DistributedDataParallel Patterns

## Overview

**DistributedDataParallel (DDP)** is PyTorch's primary API for multi-GPU and
multi-node training. Unlike `DataParallel`, DDP uses one process per GPU with
NCCL/Gloo backends for efficient gradient all-reduce, avoiding the GIL bottleneck.

## Key Concepts

### Process Groups

DDP communicates via **process groups**. `init_process_group()` establishes a
default group; custom sub-groups enable overlapping communication patterns
(e.g., intra-node all-reduce + inter-node reduce-scatter).

### Gradient Synchronization

DDP hooks into `autograd` to overlap gradient all-reduce with backward computation.
Gradients are bucketed by parameter order (reverse of `model.parameters()`) for
coalesced communication. Bucket size is tunable via `bucket_cap_mb`.

### Launch Methods

- **torchrun**: Recommended launcher (`torchrun --nproc_per_node=4 train.py`)
- **mp.spawn**: Programmatic launch for testing
- **SLURM**: Multi-node via `srun` with `MASTER_ADDR`/`MASTER_PORT` env vars

### Gradient Accumulation

To accumulate gradients across micro-batches without synchronizing each step,
use `model.no_sync()` as a context manager for all but the last micro-batch.

### Mixed Precision + DDP

Combine DDP with `torch.amp.autocast` and `GradScaler` for FP16/BF16 training.
The scaler handles gradient unscaling before the all-reduce.

## Files in This Module

| File | Description |
|------|-------------|
| `ddp_training.py` | Full DDP training loop with process groups, gradient sync, mixed precision |

## Examples

```bash
# Single-node, 4 GPUs
torchrun --nproc_per_node=4 ddp_training.py

# Multi-node (2 nodes, 4 GPUs each)
torchrun --nnodes=2 --nproc_per_node=4 \
    --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \
    ddp_training.py
```

## Common Pitfalls

1. **Unused parameters**: Set `find_unused_parameters=True` if not all params
   participate in every forward pass (but this adds overhead).
2. **Non-deterministic reductions**: Floating-point all-reduce is not bitwise
   reproducible across runs. Use `torch.use_deterministic_algorithms(True)` cautiously.
3. **State dict loading**: Save/load the `module` attribute (`model.module.state_dict()`)
   to get a clean checkpoint without DDP wrapper keys.

## References

- [DDP tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [torchrun docs](https://pytorch.org/docs/stable/elastic/run.html)
- [FSDP for large models](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [DDP design notes](https://pytorch.org/docs/stable/notes/ddp.html)
