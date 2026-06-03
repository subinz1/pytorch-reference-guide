"""
Parallelism Overview: TP, PP, and Combined Strategies
=====================================================

Shows API patterns for Tensor Parallelism (TP), Pipeline Parallelism (PP),
and 3D parallelism (DP + TP + PP). These patterns require multi-GPU setups.

This script prints documented API patterns and, when sufficient GPUs are
available, runs live demonstrations.

Run:
    python parallelism_overview.py              # prints patterns (no GPU needed)
    torchrun --nproc_per_node=4 parallelism_overview.py   # runs TP demo if 4+ GPUs
"""

import torch
import torch.nn as nn


def print_section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def tp_patterns():
    """Print Tensor Parallelism API patterns."""
    print_section("Tensor Parallelism (TP)")

    print("""\
HOW IT WORKS
------------
TP splits individual weight matrices across GPUs. For a linear layer Y = XW:

  ColwiseParallel: Split W along columns (output dimension)
    Each GPU computes a slice of the output → results are gathered.

  RowwiseParallel: Split W along rows (input dimension)
    Input is split, each GPU computes partial result → results are all-reduced.

Typical pattern for a transformer: ColwiseParallel for the first linear,
RowwiseParallel for the second. This avoids an all-gather between them.

    Attention:
      qkv_proj  → ColwiseParallel  (split heads across GPUs)
      out_proj  → RowwiseParallel  (each GPU has partial result, all-reduce)

    FFN:
      up_proj   → ColwiseParallel
      down_proj → RowwiseParallel


API CODE
--------

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.parallel import (
        parallelize_module,
        ColwiseParallel,
        RowwiseParallel,
        SequenceParallel,
    )

    # Create a 1D mesh for TP (typically within one node)
    tp_mesh = init_device_mesh("cuda", (tp_size,))

    # Define how each sub-module should be parallelized
    tp_plan = {
        # Attention
        "attn.q_proj": ColwiseParallel(),
        "attn.k_proj": ColwiseParallel(),
        "attn.v_proj": ColwiseParallel(),
        "attn.out_proj": RowwiseParallel(),
        # FFN
        "ffn.up_proj": ColwiseParallel(),
        "ffn.gate_proj": ColwiseParallel(),
        "ffn.down_proj": RowwiseParallel(),
    }

    # Apply TP to each transformer block
    for layer in model.layers:
        parallelize_module(layer, tp_mesh, tp_plan)


SEQUENCE PARALLEL
-----------------
Between ColwiseParallel and RowwiseParallel, activations can be split along
the sequence dimension instead of being replicated, saving activation memory:

    tp_plan = {
        "norm1": SequenceParallel(),
        "attn.q_proj": ColwiseParallel(input_layouts=Shard(0)),
        "attn.out_proj": RowwiseParallel(output_layouts=Shard(0)),
        "norm2": SequenceParallel(),
        "ffn.up_proj": ColwiseParallel(input_layouts=Shard(0)),
        "ffn.down_proj": RowwiseParallel(output_layouts=Shard(0)),
    }


WHEN TO USE TP
--------------
- Hidden dimension > 4096 (large linear layers)
- Within a node only (NVLink required for performance)
- TP degree: 2, 4, or 8 (matching node topology)
- Beyond 8-way TP, communication overhead dominates
""")


def pp_patterns():
    """Print Pipeline Parallelism API patterns."""
    print_section("Pipeline Parallelism (PP)")

    print("""\
HOW IT WORKS
------------
PP splits the model into sequential stages. Each stage runs on a different
GPU (or set of GPUs). Micro-batching pipelines data through stages.

    Stage 0 (GPU 0):  Embedding + Layers 0-5
    Stage 1 (GPU 1):  Layers 6-11
    Stage 2 (GPU 2):  Layers 12-17
    Stage 3 (GPU 3):  Layers 18-23 + LM Head

Only activations at stage boundaries are communicated (much less than
all-reduce of all parameters).


PIPELINE SCHEDULES COMPARISON
-----------------------------

    ┌──────────────────┬──────────────┬────────────┬──────────────────────────┐
    │ Schedule         │ Bubble Ratio │ Memory     │ Notes                    │
    ├──────────────────┼──────────────┼────────────┼──────────────────────────┤
    │ GPipe            │ (p-1)/m      │ High       │ Simple, all F then all B │
    │ 1F1B             │ (p-1)/m      │ Low        │ Alternating F/B          │
    │ Interleaved 1F1B │ (p-1)/(m*v)  │ Low        │ Virtual stages           │
    │ Zero Bubble      │ ~0           │ Moderate   │ Split B into Bw + Bi     │
    │ DualPipeV        │ ~0           │ Moderate   │ Bidirectional pipeline   │
    └──────────────────┴──────────────┴────────────┴──────────────────────────┘

    p = pipeline stages, m = micro-batches, v = virtual stages


API CODE
--------

    from torch.distributed.pipelining import (
        pipeline,
        SplitPoint,
        ScheduleGPipe,
        Schedule1F1B,
        ScheduleInterleaved1F1B,
    )

    # Step 1: Define split points
    # The model is split at specified module boundaries
    pipe = pipeline(
        module=model,
        mb_args=(example_input,),  # Example micro-batch for tracing
        split_spec={
            "layers.6": SplitPoint.BEGINNING,   # Split before layer 6
            "layers.12": SplitPoint.BEGINNING,   # Split before layer 12
            "layers.18": SplitPoint.BEGINNING,   # Split before layer 18
        },
    )

    # Step 2: Get this rank's stage
    stage = pipe.get_stage(pp_rank, device)

    # Step 3: Create a schedule
    schedule = Schedule1F1B(
        stage,
        n_microbatches=8,       # More micro-batches = smaller bubble
        loss_fn=loss_fn,        # Only needed on last stage
    )

    # Step 4: Run one training step
    # First stage feeds input, last stage gets loss
    if pp_rank == 0:
        schedule.step(input_batch)
    elif pp_rank == num_stages - 1:
        losses = schedule.step()
    else:
        schedule.step()  # Middle stages just process


CHOOSING A SCHEDULE
-------------------
- GPipe: Simplest, good for getting started. High memory (stores all
  micro-batch activations).
- 1F1B: Same bubble as GPipe but much lower memory. Good default.
- Interleaved 1F1B: Smaller bubble with virtual stages. Requires
  n_microbatches >= 2 * num_stages.
- Zero Bubble / DualPipeV: Near-zero bubble but more complex. Use for
  maximum throughput at scale.


WHEN TO USE PP
--------------
- Many GPUs across nodes (PP has less communication than TP)
- Very deep models that can be naturally split into stages
- Combine with TP (within node) and FSDP (across DP replicas)
""")


def combined_3d_patterns():
    """Print 3D parallelism patterns."""
    print_section("3D Parallelism: DP + TP + PP")

    print("""\
OVERVIEW
--------
For the largest models (100B+ params), combine all three strategies:

  TP: Within a node (fast NVLink)
  PP: Across nodes (less communication)
  FSDP: Across remaining GPUs (data parallelism with memory efficiency)


DEVICEMESH SETUP
-----------------

    from torch.distributed.device_mesh import init_device_mesh

    # Example: 64 GPUs = 4 DP x 8 TP x 2 PP
    mesh = init_device_mesh(
        "cuda",
        (4, 8, 2),
        mesh_dim_names=("dp", "tp", "pp"),
    )

    dp_mesh = mesh["dp"]
    tp_mesh = mesh["tp"]
    pp_mesh = mesh["pp"]


APPLICATION ORDER
-----------------
Apply from innermost to outermost:

    # 1. Tensor Parallelism (innermost — affects individual layers)
    for block in model.layers:
        parallelize_module(block, tp_mesh, {
            "attn.q_proj": ColwiseParallel(),
            "attn.k_proj": ColwiseParallel(),
            "attn.v_proj": ColwiseParallel(),
            "attn.out_proj": RowwiseParallel(),
            "ffn.up_proj": ColwiseParallel(),
            "ffn.down_proj": RowwiseParallel(),
        })

    # 2. FSDP (middle — shards remaining parameters)
    for block in model.layers:
        fully_shard(block, mesh=dp_mesh)
    fully_shard(model, mesh=dp_mesh)

    # 3. Pipeline Parallelism (outermost — splits model into stages)
    pipe = pipeline(model, mb_args=(...), split_spec={...})
    stage = pipe.get_stage(pp_mesh.get_local_rank(), device)
    schedule = Schedule1F1B(stage, n_microbatches=n_mb, loss_fn=loss_fn)


TYPICAL GPU ASSIGNMENT (64 GPUs, 8 nodes x 8 GPUs)
---------------------------------------------------

    Node 0: [GPU0..GPU7]  ← 8-way TP group, PP stage 0, DP replica 0
    Node 1: [GPU8..GPU15] ← 8-way TP group, PP stage 1, DP replica 0
    Node 2: [GPU16..GPU23] ← 8-way TP group, PP stage 0, DP replica 1
    Node 3: [GPU24..GPU31] ← 8-way TP group, PP stage 1, DP replica 1
    ...

    TP groups: within each node (8 GPUs connected by NVLink)
    PP groups: across nodes (communicate only activations)
    DP groups: across PP-stage-equivalent ranks on different replicas


CHOOSING YOUR STRATEGY
-----------------------

    ┌─────────────────────────┬───────────────────────────────────┐
    │ Scenario                │ Recommended Strategy              │
    ├─────────────────────────┼───────────────────────────────────┤
    │ Model fits on 1 GPU     │ DDP                               │
    │ 1-8 GPUs, tight memory  │ FSDP2                             │
    │ 8-32 GPUs, large model  │ FSDP2 + TP (within node)          │
    │ 32-128 GPUs             │ FSDP2 + TP + PP                   │
    │ 128+ GPUs               │ FSDP2 + TP + PP + CP (if needed)  │
    │ Long sequences (>16K)   │ Add Context Parallel              │
    └─────────────────────────┴───────────────────────────────────┘
""")


def demo_tp_live():
    """Run a live TP demo if enough GPUs are available."""
    import os

    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
        parallelize_module,
    )

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print_section("Live TP Demo")

    class SimpleMLP(nn.Module):
        def __init__(self, dim: int = 256):
            super().__init__()
            self.up = nn.Linear(dim, dim * 4, bias=False)
            self.act = nn.ReLU()
            self.down = nn.Linear(dim * 4, dim, bias=False)

        def forward(self, x):
            return self.down(self.act(self.up(x)))

    model = SimpleMLP(256).cuda()

    if rank == 0:
        orig_up_shape = model.up.weight.shape
        orig_down_shape = model.down.weight.shape
        print(f"  Before TP: up.weight={list(orig_up_shape)}, down.weight={list(orig_down_shape)}")

    tp_mesh = init_device_mesh("cuda", (world_size,))

    parallelize_module(
        model,
        tp_mesh,
        {
            "up": ColwiseParallel(),
            "down": RowwiseParallel(),
        },
    )

    tp_up_shape = model.up.weight.shape
    tp_down_shape = model.down.weight.shape
    print(
        f"  [Rank {rank}] After TP: up.weight={list(tp_up_shape)}, "
        f"down.weight={list(tp_down_shape)} (local shard)",
        flush=True,
    )
    dist.barrier()

    x = torch.randn(2, 256, device="cuda")
    y = model(x)
    if rank == 0:
        print(f"\n  Output shape: {list(y.shape)} (full output, automatically gathered)")
        print()

    dist.destroy_process_group()


def main():
    tp_patterns()
    pp_patterns()
    combined_3d_patterns()

    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        try:
            demo_tp_live()
        except Exception as e:
            print(f"\n  (Live TP demo skipped: {e})\n")
    else:
        print("\n  (Live demos require multi-GPU. Run with torchrun for live examples.)\n")


if __name__ == "__main__":
    main()
