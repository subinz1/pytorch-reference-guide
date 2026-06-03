"""
Distributed Training Overview — DDP, FSDP2, DeviceMesh
=========================================================
NOTE: This file demonstrates the API patterns. Actually running distributed
code requires multiple GPUs and launching with torchrun.

Usage:
  torchrun --nproc_per_node=2 distributed_overview.py
"""

import torch
import torch.nn as nn

print("=" * 60)
print("DISTRIBUTED TRAINING API OVERVIEW")
print("=" * 60)

print("""
--- 1. DDP (DistributedDataParallel) ---

Best for: Model fits on one GPU, replicate across GPUs

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    dist.init_process_group(backend="nccl")
    model = MyModel().to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # Training loop is same as single-GPU
    for data, target in train_loader:
        loss = criterion(model(data.cuda()), target.cuda())
        loss.backward()   # Gradients are all-reduced automatically
        optimizer.step()

    Launch: torchrun --nproc_per_node=4 train.py

--- 2. FSDP2 (fully_shard) — THE NEW STANDARD ---

Best for: Model too large for one GPU

    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

    # Shard sub-modules first, then root
    for layer in model.layers:
        fully_shard(layer, mp_policy=mp)
    fully_shard(model, mp_policy=mp)

    # Training loop is same as single-GPU!
    for data, target in train_loader:
        loss = model(data).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

--- 3. DeviceMesh ---

Foundation for multi-dimensional parallelism:

    from torch.distributed.device_mesh import init_device_mesh

    # 1D mesh (data parallel)
    mesh = init_device_mesh("cuda", (world_size,))

    # 2D mesh (data parallel + tensor parallel)
    mesh_2d = init_device_mesh("cuda", (dp, tp), mesh_dim_names=("dp", "tp"))
    dp_mesh = mesh_2d["dp"]
    tp_mesh = mesh_2d["tp"]

--- 4. DTensor (Distributed Tensor) ---

    from torch.distributed.tensor import DTensor, Shard, Replicate

    # Placement types:
    # Shard(dim)  — sharded along dimension
    # Replicate() — fully replicated
    # Partial()   — needs reduction

--- 5. Tensor Parallelism ---

    from torch.distributed.tensor.parallel import (
        parallelize_module, ColwiseParallel, RowwiseParallel
    )

    parallelize_module(model.layer, tp_mesh, {
        "q_proj": ColwiseParallel(),
        "out_proj": RowwiseParallel(),
    })

--- 6. Pipeline Parallelism ---

    from torch.distributed.pipelining import (
        pipeline, SplitPoint, Schedule1F1B,
        ScheduleGPipe, ScheduleZBVZeroBubble
    )

    pipe = pipeline(model, num_chunks=4, split_spec={...})
    stage = build_stage(pipe, rank, device)
    schedule = Schedule1F1B(stage, n_microbatches=4)
    schedule.step(input_batch)

--- 7. Collective Operations ---

    import torch.distributed as dist

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    dist.all_gather(output_list, tensor)
    dist.broadcast(tensor, src=0)
    dist.reduce_scatter(output, input_list)
    dist.barrier()

--- 8. Distributed Checkpointing (DCP) ---

    from torch.distributed.checkpoint import save, load, async_save
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    state = {"model": get_model_state_dict(model)}
    save(state, checkpoint_id="epoch_5")
    # Or async: f = async_save(state, checkpoint_id="epoch_5")
""")

# Demonstrate non-distributed API parts that work on CPU
print("=" * 60)
print("DEMONSTRATING CPU-AVAILABLE APIs")
print("=" * 60)

# Meta device for memory estimation
class LargeModel(nn.Module):
    def __init__(self, hidden=4096, layers=32):
        super().__init__()
        self.embed = nn.Embedding(50000, hidden)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden, 32, hidden * 4, batch_first=True)
            for _ in range(layers)
        ])
        self.head = nn.Linear(hidden, 50000)

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)

model = LargeModel().to('meta')
total_params = sum(p.numel() for p in model.parameters())
param_gb = total_params * 4 / 1e9

print(f"LargeModel (32 layers, hidden=4096):")
print(f"  Parameters:  {total_params / 1e9:.2f}B")
print(f"  FP32 memory: {param_gb:.2f} GB")
print(f"  BF16 memory: {param_gb / 2:.2f} GB")
print(f"  GPUs needed (80GB): ~{max(1, int(param_gb * 3 / 80) + 1)} (with optimizer state)")

print("\nDone!")
