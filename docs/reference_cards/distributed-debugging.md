# Distributed debugging

Use this card when a single-process test passes but a `torchrun` job hangs,
times out, or produces inconsistent gradients.

## First checks

1. Print rank, world size, hostname, and selected device at process startup.
2. Ensure every rank enters every collective in the same order.
3. Use a small timeout while reproducing so a deadlock surfaces quickly.
4. Confirm per-rank batch sizes and loss scaling are intentional.
5. Save rank-specific logs; interleaved stdout is rarely enough.

## High-signal environment variables

```bash
TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_DEBUG=INFO torchrun --nproc-per-node=2 train.py
```

Reduce to CPU/Gloo if possible to distinguish collective ordering from a
device-transport issue. Do not use a barrier as a universal fix: a barrier only
moves the deadlock when ranks disagree about the program path.
