<div align="center">

[← Previous Module](../09_attention/) | [🏠 Home](../README.md) | [Next Module →](../11_export_deploy/)

</div>

---

> **Module 10** of the PyTorch Complete Learning Guide
> **Prerequisites:** [Module 07 — Training](../07_training/), [Module 08 — torch.compile](../08_torch_compile/)
> **Time to complete:** ~4 hours

### 📁 Files in This Module

| File | Description |
|------|-------------|
| `README.md` | This guide — theory, explanations, and inline examples |
| `concepts_and_collectives.py` | Distributed concepts and collective operations (CPU/Gloo) |
| `ddp_example.py` | DistributedDataParallel (DDP) complete training example |
| `fsdp2_example.py` | FSDP2 (fully_shard) API patterns |
| `device_mesh_example.py` | DeviceMesh creation and usage |
| `parallelism_overview.py` | Parallelism overview: TP, PP, and combined strategies |

---

# Module 10: Distributed Training in PyTorch

## Table of Contents
1. [Why Distributed Training?](#why-distributed-training)
2. [Key Concepts](#key-concepts)
3. [Launching with torchrun](#launching-with-torchrun)
4. [Collective Operations](#collective-operations)
5. [DistributedDataParallel (DDP)](#distributeddataparallel-ddp)
6. [DeviceMesh](#devicemesh)
7. [DTensor (Distributed Tensor)](#dtensor-distributed-tensor)
8. [FSDP1 vs FSDP2](#fsdp1-vs-fsdp2)
9. [FSDP2 (fully_shard)](#fsdp2-fully_shard)
10. [Tensor Parallelism](#tensor-parallelism)
11. [Pipeline Parallelism](#pipeline-parallelism)
12. [Combining Strategies: 3D Parallelism](#combining-strategies-3d-parallelism)
13. [Distributed Checkpointing (DCP)](#distributed-checkpointing-dcp)
14. [SymmetricMemory](#symmetricmemory)
15. [Context Parallel](#context-parallel)
16. [Practical Advice](#practical-advice)

---

## Why Distributed Training?

As models grow larger and datasets expand, a single GPU becomes insufficient.
Distributed training addresses three fundamental bottlenecks:

**1. Model too large for one GPU (Memory)**
A model like Llama 70B requires ~140 GB just for parameters in FP16. Even the
largest GPUs (H100 with 80 GB) cannot hold this model, let alone the optimizer
states and activations needed for training. Distributed strategies split the
model across GPUs.

**2. Training too slow (Compute)**
Even when a model fits on one GPU, training can take weeks or months. By
distributing data across N GPUs, each GPU processes 1/N of the data per step,
achieving near-linear speedup. Training that takes 30 days on 1 GPU takes
~4 days on 8 GPUs.

**3. Data too large (I/O and throughput)**
With terabytes of training data, increasing throughput by processing more
samples in parallel reduces wall-clock training time proportionally.

### The Parallelism Taxonomy

| Strategy | What is split? | When to use |
|----------|---------------|-------------|
| Data Parallel (DDP) | Data (each GPU has full model copy) | Model fits on 1 GPU |
| Fully Sharded Data Parallel (FSDP) | Data + model parameters | Model barely fits or doesn't fit on 1 GPU |
| Tensor Parallel (TP) | Individual layers/tensors | Very large layers (e.g., huge linear layers) |
| Pipeline Parallel (PP) | Model stages (groups of layers) | Very deep models, many GPUs |
| Context Parallel (CP) | Sequence dimension | Very long sequences |
| 3D Parallelism | Combination of DP + TP + PP | Large-scale training (100s-1000s of GPUs) |

---

## Key Concepts

### World Size, Rank, and Local Rank

When you launch distributed training, you create multiple **processes**, each
driving one GPU. These processes form a **process group** and coordinate via
collective communication.

```
Node 0 (Machine 0)          Node 1 (Machine 1)
┌──────────────────┐        ┌──────────────────┐
│ GPU0   GPU1      │        │ GPU0   GPU1      │
│ rank=0 rank=1    │        │ rank=2 rank=3    │
│ local_rank=0     │        │ local_rank=0     │
│        local_rank=1       │        local_rank=1
└──────────────────┘        └──────────────────┘
        world_size = 4
```

- **World size**: Total number of processes across all machines
- **Rank**: Unique global identifier for each process (0 to world_size-1)
- **Local rank**: Identifier within a single machine (0 to num_local_gpus-1)

```python
import torch.distributed as dist

dist.init_process_group(backend="nccl")  # or "gloo" for CPU
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ["LOCAL_RANK"])
```

### Process Groups

A process group is a subset of all processes that can communicate. The
**default process group** includes all processes and is created by
`init_process_group`. You can create sub-groups for specialized communication:

```python
# Create a group with only ranks 0 and 1
subgroup = dist.new_group(ranks=[0, 1])

# Only ranks in the group participate in collectives on this group
if rank in [0, 1]:
    dist.all_reduce(tensor, group=subgroup)
```

### Backends

| Backend | Devices | Use Case |
|---------|---------|----------|
| **NCCL** | GPU (NVIDIA) | Default for GPU training. Highly optimized for NVIDIA hardware |
| **Gloo** | CPU, GPU | CPU training, or as a fallback. Also used for CPU collectives in GPU training |
| **UCC** | GPU | Alternative to NCCL, supports additional hardware |

NCCL (pronounced "nickel") is the standard for GPU training and provides the
best performance. Gloo is useful for CPU-based experiments and prototyping.

```python
# GPU training (most common)
dist.init_process_group(backend="nccl")

# CPU training or prototyping
dist.init_process_group(backend="gloo")

# Use NCCL for GPU ops, Gloo for CPU ops (advanced)
dist.init_process_group(backend="nccl")
cpu_group = dist.new_group(backend="gloo")
```

---

## Launching with torchrun

`torchrun` is PyTorch's built-in launcher for distributed training. It replaces
the older `torch.distributed.launch`. It sets up environment variables and
spawns processes for you.

### Single-Node Launch

```bash
# 4 GPUs on one machine
torchrun --nproc_per_node=4 train.py --arg1 val1

# Can also use for CPU (with gloo backend)
torchrun --nproc_per_node=2 train.py
```

### Multi-Node Launch

On each machine, run `torchrun` with the same `--master_addr` and
`--master_port`:

```bash
# Machine 0 (master)
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    train.py

# Machine 1
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    train.py
```

### Environment Variables Set by torchrun

| Variable | Description |
|----------|-------------|
| `RANK` | Global rank of this process |
| `LOCAL_RANK` | Local rank on this node |
| `WORLD_SIZE` | Total number of processes |
| `MASTER_ADDR` | Address of the master node |
| `MASTER_PORT` | Port for master node communication |
| `LOCAL_WORLD_SIZE` | Number of processes on this node |

Your training script reads these:

```python
import os

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
```

### Elastic Launch

torchrun supports elastic training where nodes can join or leave:

```bash
torchrun \
    --nproc_per_node=4 \
    --nnodes=2:8 \       # min 2, max 8 nodes
    --rdzv_backend=c10d \
    --rdzv_endpoint=master:29500 \
    train.py
```

---

## Collective Operations

Collective operations are communication primitives where all processes in a
group participate. Understanding these is essential because all distributed
strategies are built on top of them.

### All-Reduce

Every process starts with a tensor. After all-reduce, every process has the
element-wise sum (or other reduction) of all tensors.

```
Before:                After all_reduce(SUM):
Rank 0: [1, 2]        Rank 0: [10, 20]
Rank 1: [3, 4]   →    Rank 1: [10, 20]
Rank 2: [6, 14]       Rank 2: [10, 20]
```

This is the core operation in DDP: after each backward pass, gradients are
all-reduced so every replica has the same averaged gradients.

```python
tensor = torch.tensor([rank * 2.0, rank * 3.0])
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
# Now tensor is the same on all ranks
```

### All-Gather

Each process contributes a tensor, and every process receives the
concatenation of all tensors.

```
Before:                After all_gather:
Rank 0: [A]            Rank 0: [A, B, C]
Rank 1: [B]       →    Rank 1: [A, B, C]
Rank 2: [C]            Rank 2: [A, B, C]
```

FSDP uses all-gather to reconstruct full parameter tensors before the
forward pass.

```python
local_tensor = torch.tensor([rank])
gathered = [torch.zeros(1) for _ in range(world_size)]
dist.all_gather(gathered, local_tensor)
# gathered = [tensor([0]), tensor([1]), tensor([2])]
```

### Reduce-Scatter

The inverse of all-gather. First reduces (sums) all tensors element-wise,
then scatters the result so each rank gets a different chunk.

```
Before:                  After reduce_scatter:
Rank 0: [1, 2, 3]       Rank 0: [6]   (sum of position 0: 1+2+3)
Rank 1: [2, 3, 4]  →    Rank 1: [9]   (sum of position 1: 2+3+4+... wait)
Rank 2: [3, 4, 5]       Rank 2: [12]  (sum of position 2: 3+4+5)
```

More precisely, with 3 ranks each holding a 3-element tensor:
- Element-wise sum: [1+2+3, 2+3+4, 3+4+5] = [6, 9, 12]
- Scatter: Rank 0 gets [6], Rank 1 gets [9], Rank 2 gets [12]

FSDP uses reduce-scatter after backward to reduce gradients and distribute
shards back to their owners.

```python
output = torch.zeros(2)
input_tensor = torch.arange(world_size * 2, dtype=torch.float) + rank
dist.reduce_scatter_tensor(output, input_tensor, op=dist.ReduceOp.SUM)
```

### Broadcast

One process sends a tensor to all other processes.

```
Before:                After broadcast(src=0):
Rank 0: [42, 7]        Rank 0: [42, 7]
Rank 1: [0, 0]    →    Rank 1: [42, 7]
Rank 2: [0, 0]         Rank 2: [42, 7]
```

```python
tensor = torch.tensor([42.0, 7.0]) if rank == 0 else torch.zeros(2)
dist.broadcast(tensor, src=0)
```

### Barrier

Synchronizes all processes. Every process blocks until all processes have
reached the barrier. No data is exchanged.

```python
dist.barrier()  # All ranks wait here until everyone arrives
```

Use barriers sparingly; they are expensive and often unnecessary when
collectives already imply synchronization.

### Reduce

Like all-reduce, but the result only goes to one destination rank.

```
Before:                After reduce(dst=0, SUM):
Rank 0: [1, 2]         Rank 0: [6, 9]
Rank 1: [2, 3]    →    Rank 1: [2, 3]  (unchanged)
Rank 2: [3, 4]         Rank 2: [3, 4]  (unchanged)
```

### Scatter

One process distributes different chunks to each process.

```
Before (rank 0 has all data):     After scatter(src=0):
Rank 0: [[A], [B], [C]]          Rank 0: [A]
Rank 1: []                  →    Rank 1: [B]
Rank 2: []                       Rank 2: [C]
```

### Gather

The inverse of scatter. All processes send data to one destination.

```
Before:                After gather(dst=0):
Rank 0: [A]            Rank 0: [[A], [B], [C]]
Rank 1: [B]       →    Rank 1: [B]  (unchanged)
Rank 2: [C]            Rank 2: [C]  (unchanged)
```

---

## DistributedDataParallel (DDP)

DDP is the simplest and most commonly used distributed training strategy. Each
GPU holds a **complete copy** of the model. Training data is split across GPUs,
and gradients are synchronized via all-reduce after each backward pass.

### How DDP Works Internally

1. **Initialization**: The model is replicated on each GPU. Parameters are
   broadcast from rank 0 to ensure all replicas start identically.

2. **Forward pass**: Each rank processes its own mini-batch independently.
   No communication occurs during forward.

3. **Backward pass**: As gradients are computed, DDP groups them into
   **buckets** (default ~25 MB each). When a bucket is full, all-reduce
   starts immediately -- overlapping communication with computation for the
   remaining layers. This is called **gradient bucketing**.

4. **Optimizer step**: After all-reduce completes, every rank has identical
   averaged gradients. Each rank runs the optimizer independently, producing
   identical updated parameters.

```
                  Forward (independent)
                  ┌─────────┐
Rank 0: Data₀ →  │ Model₀  │ → Loss₀
                  └─────────┘
                  ┌─────────┐
Rank 1: Data₁ →  │ Model₁  │ → Loss₁
                  └─────────┘

                  Backward (all-reduce gradients)
                  ┌────────────────────────┐
                  │  All-Reduce Gradients  │
                  │  (bucketed, overlapped │
                  │   with backward comp)  │
                  └────────────────────────┘

                  Optimizer Step (independent, identical)
```

### Complete DDP Setup

```python
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()

def main():
    local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")

    # Create model and move to GPU
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    ).to(device)

    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank])

    # DistributedSampler ensures each rank sees different data
    dataset = MyDataset()
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10):
        sampler.set_epoch(epoch)  # Shuffle differently each epoch
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()

    cleanup()

if __name__ == "__main__":
    main()
```

### DistributedSampler

The `DistributedSampler` partitions the dataset indices so each rank gets a
different subset. It pads the dataset to make it evenly divisible by world_size.

Key: call `sampler.set_epoch(epoch)` each epoch to get different shuffling.
Without this, every epoch uses the same data order per rank.

### DDP Tips

- Access the underlying model via `model.module` (DDP wraps it).
- Save checkpoints only on rank 0 to avoid file conflicts.
- Use `torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)` before wrapping
  with DDP if your model uses BatchNorm.
- The `find_unused_parameters=True` flag is needed if some parameters don't
  receive gradients every iteration (e.g., conditional branches).

---

## DeviceMesh

`DeviceMesh` is the foundation of modern distributed training in PyTorch.
It provides a multi-dimensional abstraction over a group of devices, replacing
the manual management of process groups.

### What is a DeviceMesh?

A DeviceMesh represents a logical grid of devices. Each dimension of the mesh
corresponds to a parallelism strategy:

```python
from torch.distributed.device_mesh import init_device_mesh

# 1D mesh: all 8 GPUs in one dimension (simple data parallelism)
mesh_1d = init_device_mesh("cuda", (8,), mesh_dim_names=("dp",))

# 2D mesh: 4 data-parallel groups × 2 tensor-parallel groups
# GPUs arranged as:
#   TP=0  TP=1
#   ──────────
#   GPU0  GPU1    ← DP group 0
#   GPU2  GPU3    ← DP group 1
#   GPU4  GPU5    ← DP group 2
#   GPU6  GPU7    ← DP group 3
mesh_2d = init_device_mesh("cuda", (4, 2), mesh_dim_names=("dp", "tp"))

# 3D mesh: DP × TP × PP
mesh_3d = init_device_mesh(
    "cuda", (2, 2, 2), mesh_dim_names=("dp", "tp", "pp")
)
```

### Why DeviceMesh?

Before DeviceMesh, you had to manually create process groups for each
parallelism dimension and carefully track which ranks belonged to which
groups. DeviceMesh automates this:

```python
# Old way: manually creating groups
dp_groups = []
for i in range(0, 8, 2):
    dp_groups.append(dist.new_group([i, i+1]))

# New way: DeviceMesh handles it
mesh = init_device_mesh("cuda", (4, 2), mesh_dim_names=("dp", "tp"))
dp_mesh = mesh["dp"]  # Automatically creates the right groups
tp_mesh = mesh["tp"]
```

### Accessing Sub-Meshes

You can slice a DeviceMesh to get a sub-mesh for a specific dimension:

```python
mesh = init_device_mesh("cuda", (4, 2), mesh_dim_names=("dp", "tp"))

# Get the sub-mesh for data parallelism
dp_mesh = mesh["dp"]  # 1D mesh with 4 devices (for this rank's DP group)

# Get the sub-mesh for tensor parallelism
tp_mesh = mesh["tp"]  # 1D mesh with 2 devices (for this rank's TP group)

# These sub-meshes carry the correct process groups
# so you can pass them directly to FSDP, TP, etc.
```

### DeviceMesh for 3D Parallelism

```python
# 16 GPUs: 2 DP × 4 TP × 2 PP
mesh = init_device_mesh(
    "cuda", (2, 4, 2), mesh_dim_names=("dp", "tp", "pp")
)

# Each parallelism strategy gets its own sub-mesh
dp_mesh = mesh["dp"]
tp_mesh = mesh["tp"]
pp_mesh = mesh["pp"]

# Apply each strategy using its sub-mesh
# TP on tp_mesh, FSDP on dp_mesh, PP on pp_mesh
```

---

## DTensor (Distributed Tensor)

DTensor is a tensor abstraction that represents a tensor distributed across
multiple devices. It knows *how* the tensor is distributed and automatically
handles the communication needed for operations.

### Placement Types

DTensor uses **placements** to describe how a tensor's data is distributed
across the devices in a DeviceMesh dimension:

| Placement | Description | Example |
|-----------|-------------|---------|
| `Shard(dim)` | Tensor is sharded along dimension `dim` | A [4, 8] tensor Shard(1) across 2 GPUs → each gets [4, 4] |
| `Replicate()` | Tensor is fully replicated on each device | A [4, 8] tensor Replicate() → each GPU has [4, 8] |
| `Partial()` | Each device has a partial result; needs reduction | Intermediate matmul results before all-reduce |

### Creating DTensors

```python
from torch.distributed.tensor import DTensor, Shard, Replicate, distribute_tensor

mesh = init_device_mesh("cuda", (4,))

# Create a regular tensor and distribute it
big_tensor = torch.randn(16, 32)

# Shard along dim 0: each of 4 GPUs gets a [4, 32] chunk
sharded = distribute_tensor(big_tensor, mesh, placements=[Shard(0)])

# Replicate: each GPU gets the full [16, 32] tensor
replicated = distribute_tensor(big_tensor, mesh, placements=[Replicate()])
```

### distribute_module

Instead of manually distributing each parameter, `distribute_module`
distributes an entire module's parameters and handles input/output:

```python
from torch.distributed.tensor import distribute_module, Shard, Replicate

def input_fn(mod, inputs, mesh):
    # How to distribute inputs
    return (distribute_tensor(inputs[0], mesh, [Shard(0)]),)

def output_fn(mod, outputs, mesh):
    # How to gather outputs
    return outputs.full_tensor()

model = nn.Linear(1024, 512)
distribute_module(
    model,
    device_mesh=mesh,
    input_fn=input_fn,
    output_fn=output_fn,
)
```

### DTensor and Automatic Communication

The key insight: when you do operations on DTensors, PyTorch automatically
inserts the right collectives. If you multiply a `Shard(1)` tensor by a
`Replicate()` tensor, PyTorch knows it needs an all-reduce to get the
correct result.

```
A: [Shard(1)]  ×  B: [Replicate()]  →  C: [Partial()]  →  all_reduce → C: [Replicate()]
```

This is how Tensor Parallelism works under the hood.

---

## FSDP1 vs FSDP2

### FSDP1 (FullyShardedDataParallel - Legacy)

FSDP1 was PyTorch's first fully sharded data parallelism implementation. It
wraps the entire module:

```python
# FSDP1 (legacy)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(model, auto_wrap_policy=size_based_auto_wrap_policy)
```

Problems with FSDP1:
- **Module wrapping**: FSDP1 wraps modules, creating a new `FSDP` module
  hierarchy. This breaks `model.layer1.weight` access patterns.
- **Non-composable**: Hard to combine with other parallelism strategies.
- **Complex API**: Many constructor arguments, hard to reason about behavior.
- **Flattened parameters**: Parameters are flattened into a single 1D tensor,
  making debugging and checkpointing harder.

### FSDP2 (fully_shard - Current Standard)

FSDP2 is a ground-up rewrite with a composable, per-parameter design:

```python
# FSDP2 (current)
from torch.distributed.fsdp import fully_shard

# Apply to submodules first, then root
for layer in model.layers:
    fully_shard(layer)
fully_shard(model)
```

Advantages of FSDP2:
- **Composable**: Works seamlessly with TP, PP, and other strategies.
- **Per-parameter sharding**: Each parameter is sharded independently (via
  DTensor), preserving the original module structure.
- **Simpler API**: A single function call per module.
- **Better debugging**: Parameters remain accessible with their original names.
- **DTensor-based**: Built on DTensor, providing a clean abstraction.

**Use FSDP2 for all new code.** FSDP1 is maintained but not actively developed.

---

## FSDP2 (fully_shard)

### How Sharding Works

FSDP2 shards model parameters across data-parallel ranks. During training:

1. **Idle state**: Each rank holds only its shard (1/N) of each parameter.
   Memory usage is reduced by ~N× for parameters.

2. **Before forward**: `all-gather` reconstructs the full parameters from all
   shards. Each rank now temporarily has the full parameter for computation.

3. **Forward computation**: Runs normally with the full parameters.

4. **After forward**: Full parameters are freed (unless needed for backward).
   Memory drops back to 1/N.

5. **Before backward**: `all-gather` again to get full parameters for gradient
   computation.

6. **After backward**: `reduce-scatter` synchronizes gradients AND distributes
   gradient shards. Each rank ends up with the gradient shard corresponding
   to its parameter shard.

7. **Optimizer step**: Each rank updates only its parameter shard using its
   gradient shard. No communication needed.

```
     ┌─────────────────────────────────────────────────┐
     │              FSDP2 Training Loop                │
     │                                                 │
     │  Idle: Each rank holds 1/N of params            │
     │         ↓                                       │
     │  all-gather → Full params → Forward             │
     │         ↓                                       │
     │  Free full params (keep shards)                 │
     │         ↓                                       │
     │  all-gather → Full params → Backward            │
     │         ↓                                       │
     │  reduce-scatter → Gradient shards               │
     │         ↓                                       │
     │  Optimizer step (on shards only)                │
     │         ↓                                       │
     │  Back to idle (1/N params + 1/N grads)          │
     └─────────────────────────────────────────────────┘
```

### Basic FSDP2 Setup

```python
import torch
import torch.nn as nn
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

class TransformerBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, 8)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x

class TransformerModel(nn.Module):
    def __init__(self, d_model=512, n_layers=6):
        super().__init__()
        self.embed = nn.Embedding(10000, d_model)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model) for _ in range(n_layers)]
        )
        self.head = nn.Linear(d_model, 10000)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)

# Apply FSDP2: submodules first, then root
model = TransformerModel().cuda()
for layer in model.layers:
    fully_shard(layer)
fully_shard(model)  # Root module last
```

### MixedPrecisionPolicy

Mixed precision reduces memory and increases throughput by using lower
precision for computation while maintaining accuracy:

```python
from torch.distributed.fsdp import MixedPrecisionPolicy

# BFloat16 for compute, FP32 for parameter storage
mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,    # Cast params to bf16 for forward/backward
    reduce_dtype=torch.float32,     # Reduce gradients in fp32 for accuracy
)

for layer in model.layers:
    fully_shard(layer, mp_policy=mp_policy)
fully_shard(model, mp_policy=mp_policy)
```

### CPUOffloadPolicy

For extremely large models, offload parameters to CPU when not in use:

```python
from torch.distributed.fsdp import CPUOffloadPolicy

offload_policy = CPUOffloadPolicy(pin_memory=True)

for layer in model.layers:
    fully_shard(layer, offload_policy=offload_policy)
fully_shard(model, offload_policy=offload_policy)
```

This trades compute speed for memory: parameters are moved to CPU after
forward/backward, freeing GPU memory. `pin_memory=True` uses pinned (page-
locked) CPU memory for faster CPU-GPU transfers.

### FSDP2 + DeviceMesh

When combining FSDP with other parallelism, use DeviceMesh to specify which
dimension is for data parallelism:

```python
mesh = init_device_mesh("cuda", (4, 2), mesh_dim_names=("dp", "tp"))
dp_mesh = mesh["dp"]

# fully_shard on the DP dimension
for layer in model.layers:
    fully_shard(layer, mesh=dp_mesh)
fully_shard(model, mesh=dp_mesh)
```

---

## Tensor Parallelism

Tensor Parallelism (TP) splits individual layers across GPUs. Unlike FSDP
which shards parameters and reconstructs them for computation, TP keeps
parameters split and performs distributed computation.

### Why Tensor Parallelism?

- Reduces memory per GPU for very large individual layers
- Reduces latency per layer (each GPU does less work)
- Essential for models where single layers are too large for one GPU
- Works within a single node (requires fast interconnect like NVLink)

### Column-wise and Row-wise Parallelism

For a linear layer `Y = XW + b`, there are two ways to split:

**ColwiseParallel**: Split W along columns (output dimension)
```
         GPU 0          GPU 1
W = [w₀ | w₁]   →   W₀ = w₀     W₁ = w₁
Y = X @ W       →   Y₀ = X @ W₀  Y₁ = X @ W₁
                     Y = [Y₀ | Y₁]  (gathered)
```
Each GPU computes a portion of the output. The input X is replicated.

**RowwiseParallel**: Split W along rows (input dimension)
```
          GPU 0          GPU 1
W = [w₀]          W₀ = w₀     W₁ = w₁
    [w₁]
X = [x₀ | x₁]    X₀ = x₀     X₁ = x₁
Y = X @ W    →    Y = X₀@W₀ + X₁@W₁  (all-reduce)
```
Input is split and each GPU computes a partial result that must be summed.

### Using TP in PyTorch

```python
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
)

mesh = init_device_mesh("cuda", (world_size,))

# Parallelize specific layers
parallelize_plan = {
    "attn.qkv_proj": ColwiseParallel(),
    "attn.out_proj": RowwiseParallel(),
    "ffn.up_proj": ColwiseParallel(),
    "ffn.down_proj": RowwiseParallel(),
}

parallelize_module(model, mesh, parallelize_plan)
```

### SequenceParallel

SequenceParallel splits the sequence dimension instead of replicating
activations. Between ColwiseParallel and RowwiseParallel layers, activations
can be kept split along the sequence dimension, reducing activation memory:

```python
parallelize_plan = {
    "norm1": SequenceParallel(),
    "attn.qkv_proj": ColwiseParallel(input_layouts=Shard(0)),
    "attn.out_proj": RowwiseParallel(output_layouts=Shard(0)),
    "norm2": SequenceParallel(),
    "ffn.up_proj": ColwiseParallel(input_layouts=Shard(0)),
    "ffn.down_proj": RowwiseParallel(output_layouts=Shard(0)),
}
```

### When to Use TP

- Large models with huge linear layers (e.g., LLMs with 8192+ hidden dim)
- When you have fast intra-node interconnect (NVLink)
- Typically TP degree of 2, 4, or 8 within a single node
- Beyond 8-way TP, the communication overhead usually outweighs benefits

---

## Pipeline Parallelism

Pipeline Parallelism (PP) splits a model into sequential **stages**, each
running on a different GPU (or set of GPUs). Data flows through stages
sequentially, like an assembly line.

### The Bubble Problem

Naive pipeline parallelism has a severe inefficiency: while stage 1 processes
micro-batch 1, stages 2-N are idle. This idle time is called the **pipeline
bubble**.

```
Naive (Sequential):
Stage 0: [F1][F2][F3][F4]
Stage 1:     [F1][F2][F3][F4]
Stage 2:         [F1][F2][F3][F4]
                              ↑ huge bubble, most GPUs idle most of the time
```

### Micro-batching

The solution: split each mini-batch into multiple **micro-batches** and
pipeline them:

```
With micro-batches:
Stage 0: [F1][F2][F3][F4][B4][B3][B2][B1]
Stage 1:     [F1][F2][F3][F4][B4][B3][B2][B1]
Stage 2:         [F1][F2][F3][F4][B4][B3][B2][B1]

F = forward, B = backward, number = micro-batch id
```

### Pipeline Schedules

Different schedules offer different trade-offs between memory, bubble ratio,
and implementation complexity:

| Schedule | Bubble Ratio | Memory | Description |
|----------|-------------|--------|-------------|
| **GPipe** | (p-1)/m | High (all activations) | All forwards, then all backwards |
| **1F1B** | (p-1)/m | Low (1 activation) | Alternating forward-backward in steady state |
| **Interleaved 1F1B** | (p-1)/(m×v) | Low | Multiple virtual stages per rank, smaller bubbles |
| **Zero Bubble** | ~0 | Moderate | Overlaps weight gradient with next forward |
| **DualPipeV** | ~0 | Moderate | Bidirectional pipeline for near-zero bubble |

Where p = number of pipeline stages, m = number of micro-batches, v = number
of virtual stages (chunks).

**GPipe**: Simple but memory-hungry. All micro-batches do forward, then all
do backward. Must store activations for all micro-batches simultaneously.

**1F1B (One Forward One Backward)**: After a warm-up phase, alternates one
forward and one backward. Steady-state memory is constant (only one micro-
batch's activations at a time).

**Interleaved 1F1B**: Each rank handles multiple non-contiguous stages (e.g.,
rank 0 handles stages 0 and 4). This reduces the bubble because micro-batches
cycle through stages faster.

**Zero Bubble**: Splits backward into two parts (input gradient and weight
gradient) and overlaps weight gradient computation with the next forward pass.
Nearly eliminates the bubble.

**DualPipeV**: A bidirectional schedule where micro-batches flow both forward
and backward through the pipeline simultaneously, achieving near-zero bubble
with better memory efficiency.

### PP in PyTorch

```python
from torch.distributed.pipelining import (
    pipeline,
    SplitPoint,
    ScheduleGPipe,
    Schedule1F1B,
    ScheduleInterleaved1F1B,
)

# Split model into stages
pipe = pipeline(
    model,
    mb_args=(torch.randn(batch_size, seq_len, d_model),),
    split_spec={
        "layers.3": SplitPoint.BEGINNING,  # Split before layer 3
        "layers.6": SplitPoint.BEGINNING,  # Split before layer 6
    },
)

# Get this rank's stage
stage = pipe.get_stage(rank, device)

# Create schedule
schedule = Schedule1F1B(stage, n_microbatches=8)

# Run
if rank == 0:
    schedule.step(input_data)
elif rank == num_stages - 1:
    losses = schedule.step()
else:
    schedule.step()
```

---

## Combining Strategies: 3D Parallelism

For training the largest models (100B+ parameters), you combine multiple
parallelism strategies. The standard combination is **3D parallelism**:
Data Parallel (FSDP) × Tensor Parallel × Pipeline Parallel.

### DeviceMesh for 3D Parallelism

```python
# 64 GPUs across 8 nodes, 8 GPUs per node
# 8 DP × 4 TP × 2 PP
mesh = init_device_mesh(
    "cuda", (8, 4, 2), mesh_dim_names=("dp", "tp", "pp")
)
```

### Typical Assignment

- **TP within a node**: TP requires fast interconnect, so TP groups are
  within a single node (connected by NVLink).
- **PP across nodes**: PP has less communication (only activations at stage
  boundaries), so it can span nodes.
- **FSDP across remaining GPUs**: FSDP handles the data parallelism dimension.

```
Node 0:  [GPU0, GPU1, GPU2, GPU3]  ← TP group, PP stage 0
Node 1:  [GPU4, GPU5, GPU6, GPU7]  ← TP group, PP stage 1
...
Across nodes: FSDP groups
```

### Code Pattern for 3D Parallelism

```python
mesh = init_device_mesh(
    "cuda", (dp_size, tp_size, pp_size),
    mesh_dim_names=("dp", "tp", "pp"),
)

# 1. Apply Tensor Parallelism first (innermost)
tp_mesh = mesh["tp"]
for layer in model.layers:
    parallelize_module(layer, tp_mesh, tp_plan)

# 2. Apply FSDP (middle)
dp_mesh = mesh["dp"]
for layer in model.layers:
    fully_shard(layer, mesh=dp_mesh)
fully_shard(model, mesh=dp_mesh)

# 3. Apply Pipeline Parallelism (outermost)
pp_mesh = mesh["pp"]
# Split model into stages along pp_mesh
```

---

## Distributed Checkpointing (DCP)

When training with FSDP, TP, or other distributed strategies, each rank
holds only a shard of the model. Distributed Checkpointing (DCP) handles
saving and loading these sharded states correctly.

### Why Not torch.save?

`torch.save` requires gathering the full model to one rank, which:
- May not fit in memory for very large models
- Creates a bottleneck (one rank doing all the work)
- Produces a format tied to the original parallelism configuration

DCP saves each rank's shard independently and can reshard when loading
with a different parallelism configuration.

### Basic Save and Load

```python
import torch.distributed.checkpoint as dcp

# Save
state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
dcp.save(state_dict, checkpoint_id="checkpoints/step_1000")

# Load
state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
dcp.load(state_dict, checkpoint_id="checkpoints/step_1000")
model.load_state_dict(state_dict["model"])
optimizer.load_state_dict(state_dict["optimizer"])
```

### Async Save

For large models, checkpointing can take minutes. Async save moves the
checkpoint writing to a background thread so training can continue:

```python
# Async save returns a Future
future = dcp.async_save(state_dict, checkpoint_id="checkpoints/step_1000")

# Training continues immediately...
# Optionally wait for completion before the next checkpoint
future.result()
```

### get_model_state_dict / set_model_state_dict

These utilities handle the complexity of getting/setting state dicts for
models wrapped with FSDP, TP, etc.:

```python
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)

# Get a "clean" state dict (handles FSDP/TP unwrapping)
model_state = get_model_state_dict(model)
optim_state = get_optimizer_state_dict(model, optimizer)

# Save
dcp.save({"model": model_state, "optim": optim_state}, checkpoint_id=path)

# Load
state = {"model": model_state, "optim": optim_state}
dcp.load(state, checkpoint_id=path)
set_model_state_dict(model, state["model"])
set_optimizer_state_dict(model, optimizer, state["optim"])
```

### HuggingFace Format

DCP can save in HuggingFace-compatible format for interoperability:

```python
from torch.distributed.checkpoint import HuggingFaceLoadPlanner

# Load a HuggingFace checkpoint
dcp.load(
    state_dict,
    checkpoint_id="path/to/hf_checkpoint",
    planner=HuggingFaceLoadPlanner(),
)
```

---

## SymmetricMemory

SymmetricMemory is an intra-node optimization that provides direct GPU-to-GPU
memory access using NVLink, bypassing the traditional collective communication
libraries.

### What is SymmetricMemory?

On a multi-GPU node with NVLink, GPUs can directly read/write each other's
memory. SymmetricMemory allocates a shared memory region accessible by all
GPUs in a group, enabling custom, low-latency communication patterns.

```python
import torch.distributed.symmetric_memory as sm

# Allocate symmetric memory (same virtual address on all GPUs)
t = sm.empty_strided_p2p(
    size=(1024, 1024),
    stride=(1024, 1),
    dtype=torch.float32,
    device=torch.device(f"cuda:{local_rank}"),
)

# Direct GPU-to-GPU operations
sm.memcpy_p2p(dst=t_on_gpu1, src=t_on_gpu0)
```

### When to Use

- Custom all-reduce implementations that exploit NVLink topology
- Fine-grained producer-consumer patterns between GPUs
- Overlapping communication with computation at a granularity finer than
  what standard collectives offer
- Typically used in advanced performance optimization, not in everyday training

---

## Context Parallel

Context Parallelism (CP) addresses the challenge of training with very long
sequences. When the sequence length is so large that a single GPU cannot hold
the activations for one sequence, CP splits the sequence across GPUs.

### How It Works

CP distributes the sequence dimension across GPUs in a process group.
For attention computation, this requires specialized communication because
each position needs to attend to all other positions:

```
Sequence: [token_0, token_1, ..., token_8191]

GPU 0: [token_0 ... token_2047]
GPU 1: [token_2048 ... token_4095]
GPU 2: [token_4096 ... token_6143]
GPU 3: [token_6144 ... token_8191]

For attention: GPU 0 needs KV from all GPUs → ring attention
```

### Ring Attention

Ring attention is a common CP implementation where KV pairs are passed
around in a ring. Each GPU computes attention with its local Q and the
received KV, then passes KV to the next GPU:

```
Step 1: GPU0 attends to KV₀, GPU1 to KV₁, ...
Step 2: GPUs pass KV to neighbor: GPU0 gets KV₃, GPU1 gets KV₀, ...
Step 3: Repeat until all KV seen by all GPUs
```

### When to Use

- Sequence lengths > 8K-16K tokens (depends on model size and GPU memory)
- Long-document training, video models, genomics
- Often combined with TP and FSDP

---

## Practical Advice

### Choosing a Parallelism Strategy

```
Start: Does the model fit on 1 GPU with your batch size?
  │
  ├── YES → Use DDP
  │         Still too slow? → Increase DDP world size
  │
  └── NO  → Does the model fit on 1 GPU (batch_size=1)?
            │
            ├── YES → Use FSDP2
            │         Want even faster? → FSDP2 + TP
            │
            └── NO  → Use FSDP2 + TP
                      Still doesn't fit? → Add PP
                      Very long sequences? → Add CP
```

### Rules of Thumb

1. **Start with DDP** if your model fits on one GPU. It's the simplest and
   most efficient.

2. **Move to FSDP2** when memory is the bottleneck. FSDP reduces per-GPU
   memory at the cost of extra communication.

3. **Add TP** when individual layers are very large (hidden_dim > 4096) or
   when you need to reduce per-GPU memory further. Keep TP within a node.

4. **Add PP** when you have many GPUs across nodes and want to reduce
   cross-node communication. PP only communicates activations at stage
   boundaries.

5. **Use CP** specifically for long-sequence training where sequence
   activations dominate memory.

6. **Match TP to NVLink topology**: Use TP degree 2, 4, or 8 matching the
   NVLink connectivity within your node.

7. **More micro-batches reduce PP bubble**: For pipeline parallelism, use
   at least 2-4× as many micro-batches as pipeline stages.

### Common Pitfalls

- **Forgetting `sampler.set_epoch(epoch)`**: Causes same data order every
  epoch with DDP/FSDP.
- **Not sharding submodules before root in FSDP2**: Always apply `fully_shard`
  to submodules first, then the root module.
- **TP across nodes**: TP requires fast interconnect. Putting a TP group
  across nodes with only InfiniBand (instead of NVLink) kills performance.
- **Saving checkpoints on all ranks**: Use rank 0 for simple saves, or DCP
  for distributed saves.
- **Not using `no_sync()` for gradient accumulation**: When accumulating
  gradients across multiple steps, wrap forward/backward in `model.no_sync()`
  to skip all-reduce on intermediate steps.

### Gradient Accumulation with DDP/FSDP

```python
accumulation_steps = 4
for i, (data, target) in enumerate(dataloader):
    # Use no_sync for intermediate steps to avoid wasteful all-reduce
    context = model.no_sync() if (i + 1) % accumulation_steps != 0 else nullcontext()
    with context:
        output = model(data)
        loss = loss_fn(output, target) / accumulation_steps
        loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## Files in This Module

| File | Description | Run Command |
|------|-------------|-------------|
| `concepts_and_collectives.py` | Collective operations on CPU with Gloo | `torchrun --nproc_per_node=3 concepts_and_collectives.py` |
| `ddp_example.py` | Complete DDP training with synthetic data | `torchrun --nproc_per_node=2 ddp_example.py` |
| `fsdp2_example.py` | FSDP2 API patterns and setup | `torchrun --nproc_per_node=2 fsdp2_example.py` |
| `device_mesh_example.py` | DeviceMesh creation and sub-mesh access | `torchrun --nproc_per_node=4 device_mesh_example.py` |
| `parallelism_overview.py` | API patterns for TP and PP | Reference script showing code patterns |

---

<div align="center">

[← Previous Module](../09_attention/) | [🏠 Home](../README.md) | [Next Module →](../11_export_deploy/)

**[📓 Open Notebook](../notebooks/10_distributed_overview.ipynb)** — Interactive version of this module

</div>
