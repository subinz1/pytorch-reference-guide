# Distributed Environment

## Validate the launch before debugging the model

Most distributed failures are configuration mismatches: wrong ranks, device
assignment, rendezvous settings, or a collective called by only some ranks.

```python
import os
import torch
import torch.distributed as dist

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")

print(
    f"rank={dist.get_rank()} world_size={dist.get_world_size()} "
    f"device={torch.cuda.current_device()}"
)
```

Launch with `torchrun --standalone --nproc_per_node=2 train.py` for a
single-node smoke test.

## Checklist

- Set the CUDA device from `LOCAL_RANK` before allocating model tensors.
- Use the same collective sequence on every participating rank.
- Shard both data and evaluation accounting deliberately.
- Keep rank-zero-only logging and checkpointing explicit.
- Enable distributed debug logs when isolating hangs.

First make a two-rank CPU or single-node GPU test pass. Scale node count only
after ranks, devices, and rendezvous are proven correct.

