"""
FSDP2 (fully_shard) API Patterns
=================================

Demonstrates the FSDP2 API for sharding model parameters across GPUs.
Shows the code structure for:
- Basic fully_shard usage
- MixedPrecisionPolicy
- CPUOffloadPolicy
- FSDP2 with DeviceMesh
- Training loop with FSDP2

Requires multi-GPU. Run with:
    torchrun --nproc_per_node=2 fsdp2_example.py

If running on CPU/single-GPU, this script prints the API patterns
and skips actual distributed execution.
"""

import os

import torch
import torch.nn as nn


def check_multi_gpu() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.device_count() >= 2


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.act(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class SmallTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(self.norm(h))


def demo_basic_fsdp2():
    """Basic FSDP2 setup: shard submodules first, then root."""
    import torch.distributed as dist
    from torch.distributed.fsdp import fully_shard

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    model = SmallTransformer().cuda()

    if rank == 0:
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {param_count:,}")
        print(f"Sharding across {world_size} GPUs with FSDP2\n")

    for layer in model.layers:
        fully_shard(layer)
    fully_shard(model)

    if rank == 0:
        print("FSDP2 applied. Each parameter is now a DTensor sharded across ranks.\n")

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for step in range(5):
        input_ids = torch.randint(0, 1000, (4, 32), device="cuda")
        target = torch.randint(0, 1000, (4, 32), device="cuda")

        logits = model(input_ids)
        loss = loss_fn(logits.view(-1, 1000), target.view(-1))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if rank == 0:
            print(f"  Step {step + 1}: loss = {loss.item():.4f}")

    if rank == 0:
        print("\nBasic FSDP2 training complete.\n")

    dist.destroy_process_group()


def demo_mixed_precision_fsdp2():
    """FSDP2 with mixed precision policy."""
    import torch.distributed as dist
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()

    model = SmallTransformer().cuda()

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )

    for layer in model.layers:
        fully_shard(layer, mp_policy=mp_policy)
    fully_shard(model, mp_policy=mp_policy)

    if rank == 0:
        print("FSDP2 with MixedPrecisionPolicy (bf16 compute, fp32 reduce)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    input_ids = torch.randint(0, 1000, (4, 32), device="cuda")
    target = torch.randint(0, 1000, (4, 32), device="cuda")

    logits = model(input_ids)
    loss = loss_fn(logits.view(-1, 1000), target.view(-1))
    loss.backward()
    optimizer.step()

    if rank == 0:
        print(f"  Mixed precision step complete, loss = {loss.item():.4f}\n")

    dist.destroy_process_group()


def print_api_patterns():
    """Print FSDP2 API patterns for reference when multi-GPU is not available."""
    print("=" * 70)
    print("  FSDP2 (fully_shard) API Reference Patterns")
    print("  Requires multi-GPU to run. Showing code patterns below.")
    print("=" * 70)

    print("""
1. BASIC FSDP2 SETUP
====================

    from torch.distributed.fsdp import fully_shard

    model = MyModel().cuda()

    # Apply to submodules first (innermost → outermost)
    for layer in model.layers:
        fully_shard(layer)
    fully_shard(model)  # Root module LAST


2. MIXED PRECISION
==================

    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,    # Compute in bf16
        reduce_dtype=torch.float32,     # Gradient reduction in fp32
    )

    for layer in model.layers:
        fully_shard(layer, mp_policy=mp_policy)
    fully_shard(model, mp_policy=mp_policy)


3. CPU OFFLOAD
==============

    from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard

    offload = CPUOffloadPolicy(pin_memory=True)

    for layer in model.layers:
        fully_shard(layer, offload_policy=offload)
    fully_shard(model, offload_policy=offload)


4. FSDP2 + DEVICEMESH (for combining with TP)
===============================================

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    # 2D mesh: dp_size × tp_size
    mesh = init_device_mesh("cuda", (dp_size, tp_size),
                            mesh_dim_names=("dp", "tp"))
    dp_mesh = mesh["dp"]

    # Apply TP first (on tp_mesh), then FSDP (on dp_mesh)
    for layer in model.layers:
        parallelize_module(layer, mesh["tp"], tp_plan)
        fully_shard(layer, mesh=dp_mesh)
    fully_shard(model, mesh=dp_mesh)


5. TRAINING LOOP
================

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for batch in dataloader:
            input_ids, targets = batch
            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


6. GRADIENT ACCUMULATION WITH FSDP2
====================================

    from contextlib import nullcontext

    for i, batch in enumerate(dataloader):
        is_accumulating = (i + 1) % accumulation_steps != 0
        ctx = model.no_sync() if is_accumulating else nullcontext()

        with ctx:
            loss = loss_fn(model(batch.input), batch.target)
            (loss / accumulation_steps).backward()

        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()


7. CHECKPOINTING WITH FSDP2 (using DCP)
=========================================

    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict,
        set_model_state_dict,
    )

    # Save
    model_state = get_model_state_dict(model)
    dcp.save({"model": model_state}, checkpoint_id="ckpt/step_100")

    # Load
    model_state = get_model_state_dict(model)
    dcp.load({"model": model_state}, checkpoint_id="ckpt/step_100")
    set_model_state_dict(model, model_state)
""")


def main():
    if not check_multi_gpu():
        print_api_patterns()
        return

    demo_basic_fsdp2()


if __name__ == "__main__":
    main()
