<div align="center">

[← Previous Module](../25_triton_kernels/) | [🏠 Home](../README.md) | Next Module →

</div>

---

> **Module 26** of the PyTorch Complete Learning Guide
> **Prerequisites:** [Module 07 — Training Pipelines](../07_training/), [Module 08 — torch.compile](../08_torch_compile/)
> **Time to complete:** ~2 hours

### 📁 Files in This Module

| File | Description |
|------|-------------|
| [`README.md`](README.md) | This guide — GPU memory anatomy, profiling tools, optimization techniques |
| [`memory_tools.py`](memory_tools.py) | Meta-device estimation, memory monitoring, "will it fit?" calculator |
| [`memory_optimization.py`](memory_optimization.py) | Gradient checkpointing, mixed precision, accumulation, in-place ops |

---

# GPU Memory Profiling & Optimization — Every Byte Accounted For

## Table of Contents

1. [Where Does GPU Memory Go?](#1-where-does-gpu-memory-go)
2. [torch.cuda.memory_allocated / memory_reserved](#2-torchcudamemory_allocated--memory_reserved)
3. [torch.cuda.memory_summary()](#3-torchcudamemory_summary)
4. [Peak Memory Tracking](#4-peak-memory-tracking)
5. [torch.cuda.memory_stats()](#5-torchcudamemory_stats)
6. [Memory Snapshots](#6-memory-snapshots)
7. [Finding Memory Leaks](#7-finding-memory-leaks)
8. [torch.cuda.empty_cache()](#8-torchcudaempty_cache)
9. [Memory Optimization Techniques](#9-memory-optimization-techniques)
10. [torch.profiler for Memory](#10-torchprofiler-for-memory)
11. [Memory-Efficient Attention](#11-memory-efficient-attention)
12. [Practical: Estimating Memory Before Training](#12-practical-estimating-memory-before-training)
13. [Upstream Updates (June 17–18, 2026)](#13-upstream-updates-june-17-18-2026)

---

## 1. Where Does GPU Memory Go?

Before optimizing, you need to understand what consumes GPU memory during training. Every byte falls into one of these categories:

### 1.1 CUDA Context Overhead

The CUDA runtime itself consumes memory just by being initialized — typically **~300–800 MB** depending on GPU architecture and driver version:

```python
import torch
torch.cuda.init()  # triggers context creation
print(torch.cuda.memory_reserved())  # ~300-800 MB before any tensors
```

This is unavoidable. A freshly initialized CUDA context on an A100 typically uses ~300 MB, while an H100 may use ~500 MB.

### 1.2 Model Parameters

Each parameter stores weights in the model's dtype:

```
Parameter Memory = num_params × bytes_per_element

fp32: 4 bytes/param → 1B params = 4 GB
bf16: 2 bytes/param → 1B params = 2 GB
fp16: 2 bytes/param → 1B params = 2 GB
```

### 1.3 Optimizer State

Optimizers store per-parameter state tensors. This is often the largest memory consumer:

| Optimizer | State per Parameter | Total for N params (fp32) |
|-----------|-------------------|---------------------------|
| SGD (no momentum) | 0 | 0 |
| SGD + momentum | 1× (momentum buffer) | 4N bytes |
| Adam / AdamW | 2× (first moment m, second moment v) | 8N bytes |
| Adam + master weights (bf16 training) | 2× moments + 1× master copy | 12N bytes |
| Adafactor | ~1× (row/col factors) | ~4N bytes |

**Adam dominates**: for a model with N fp32 params, Adam needs **8N bytes** of optimizer state — **twice the parameter memory**.

### 1.4 Gradients

During backward, each parameter accumulates gradients in the parameter's dtype (or fp32 for mixed precision):

```
Gradient Memory = num_params × bytes_per_element
                = same as parameter memory (1×)
```

### 1.5 Activations (Forward Pass Intermediates)

The biggest variable. Activations saved for backward scale with:

```
Activation Memory ∝ batch_size × seq_len × hidden_dim × num_layers
```

For a Transformer with L layers, hidden size H, sequence length S, batch size B:

```
Per-layer activations ≈ 2 × B × S × H × bytes_per_element  (input + output)
Attention scores      ≈ B × num_heads × S × S × bytes_per_element
Total activations     ≈ L × (per-layer + attention) activations
```

Activations grow **linearly** with batch size and number of layers, and **quadratically** with sequence length (for standard attention).

### 1.6 Fragmentation

PyTorch uses a caching allocator. Memory that has been freed but not returned to CUDA appears as "reserved but not allocated." Fragmentation occurs when freed blocks can't be coalesced into contiguous chunks for new allocations.

### 1.7 Complete Memory Budget Example: 7B Model

```
Model: 7B parameters, bf16, Adam optimizer, batch=4, seq=2048, 32 layers

Parameters:     7B × 2 bytes     =  14 GB  (bf16)
Gradients:      7B × 2 bytes     =  14 GB  (bf16)
Adam state:     7B × 4 bytes × 2 =  56 GB  (fp32 moments: m + v)
Master weights: 7B × 4 bytes     =  28 GB  (fp32 copy for mixed precision)
Activations:    ~12-20 GB        (depends on architecture details)
CUDA context:   ~0.5 GB
─────────────────────────────────────────
Total:          ~125-133 GB      → needs multiple GPUs or FSDP
```

This is why a 7B model requires at least 2× 80 GB GPUs (A100/H100) for full-parameter fine-tuning with Adam. Techniques like LoRA, 8-bit optimizers, and gradient checkpointing reduce this dramatically.

---

## 2. torch.cuda.memory_allocated / memory_reserved

These two functions tell you the current state of GPU memory:

```python
import torch

# memory_allocated: bytes actively used by tensors
allocated = torch.cuda.memory_allocated()  # in bytes

# memory_reserved: bytes held by PyTorch's caching allocator
reserved = torch.cuda.memory_reserved()    # in bytes

# The gap = cached but not actively used (fragmentation + cache)
gap = reserved - allocated
```

### What Each Means

- **`memory_allocated()`** — memory occupied by tensors that currently exist. When you `del` a tensor, this number drops.
- **`memory_reserved()`** — total memory PyTorch has claimed from CUDA. This only drops when you call `empty_cache()` or PyTorch returns blocks to CUDA under memory pressure.

```python
x = torch.randn(1000, 1000, device='cuda')
print(f"Allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
print(f"Reserved:  {torch.cuda.memory_reserved() / 1e6:.1f} MB")

del x
# Allocated drops, but reserved stays the same
print(f"After del - Allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
print(f"After del - Reserved:  {torch.cuda.memory_reserved() / 1e6:.1f} MB")

torch.cuda.empty_cache()
# Now reserved drops too
print(f"After empty_cache - Reserved: {torch.cuda.memory_reserved() / 1e6:.1f} MB")
```

### Helper for Human-Readable Output

```python
def print_memory(tag=""):
    alloc = torch.cuda.memory_allocated() / 1024**2
    res = torch.cuda.memory_reserved() / 1024**2
    print(f"[{tag}] Allocated: {alloc:.1f} MB | Reserved: {res:.1f} MB | Gap: {res-alloc:.1f} MB")
```

---

## 3. torch.cuda.memory_summary()

For a detailed breakdown, `memory_summary()` prints a formatted table:

```python
print(torch.cuda.memory_summary())
```

This prints a table with columns like:

```
|                   |   Cur Usage |   Peak Usage |   Tot Alloc  |   Tot Freed  |
|   Allocated Bytes |     4.00 MB |    16.00 MB  |   128.00 MB  |   124.00 MB  |
|   Active Bytes    |     4.00 MB |    16.00 MB  |   128.00 MB  |   124.00 MB  |
|   Reserved Bytes  |    20.00 MB |    20.00 MB  |    20.00 MB  |     0.00 MB  |
|   Inactive Split  |     0.00 MB |     ...      |     ...      |     ...      |
```

### Key Rows Explained

| Row | Meaning |
|-----|---------|
| **Allocated Bytes** | Memory occupied by live tensors |
| **Active Bytes** | Same as allocated (non-released blocks) |
| **Reserved Bytes** | Total memory held by the caching allocator |
| **Inactive Split Bytes** | Freed memory within split blocks (fragmentation indicator) |
| **Allocation count** | How many `cudaMalloc` calls were made |
| **Active allocs** | Number of currently live tensor allocations |

### Reading the Table for Fragmentation

If **Inactive Split Bytes** is large relative to **Reserved Bytes**, you have significant fragmentation. This means memory was allocated, some tensors were freed, but the freed chunks are sandwiched between live allocations and can't be coalesced.

```python
# Common usage: log memory state at key points
model = build_model().cuda()
print(torch.cuda.memory_summary(abbreviated=True))

output = model(input_batch)
print(torch.cuda.memory_summary(abbreviated=True))

loss = criterion(output, target)
loss.backward()
print(torch.cuda.memory_summary(abbreviated=True))
```

---

## 4. Peak Memory Tracking

### max_memory_allocated / max_memory_reserved

Track the high-water mark of GPU memory usage:

```python
# Reset counters before your experiment
torch.cuda.reset_peak_memory_stats()

# Run your training step
output = model(batch)
loss = criterion(output, targets)
loss.backward()
optimizer.step()

# Check peak usage during that step
peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
peak_res = torch.cuda.max_memory_reserved() / 1024**3
print(f"Peak allocated: {peak_alloc:.2f} GB")
print(f"Peak reserved:  {peak_res:.2f} GB")
```

### reset_peak_memory_stats()

Resets the peak counters without clearing any cached memory. Use this to compare memory usage between different configurations:

```python
# Experiment 1: batch_size=32
torch.cuda.reset_peak_memory_stats()
train_step(model, batch_32)
peak_bs32 = torch.cuda.max_memory_allocated()

# Experiment 2: batch_size=64
torch.cuda.reset_peak_memory_stats()
train_step(model, batch_64)
peak_bs64 = torch.cuda.max_memory_allocated()

print(f"BS=32 peak: {peak_bs32/1e9:.2f} GB")
print(f"BS=64 peak: {peak_bs64/1e9:.2f} GB")
print(f"Ratio: {peak_bs64/peak_bs32:.2f}x")
```

---

## 5. torch.cuda.memory_stats()

For programmatic access to all memory statistics (useful for logging to TensorBoard/W&B):

```python
stats = torch.cuda.memory_stats()

# Key fields
print(f"Current allocated:  {stats['allocated_bytes.all.current'] / 1e9:.2f} GB")
print(f"Peak allocated:     {stats['allocated_bytes.all.peak'] / 1e9:.2f} GB")
print(f"Current reserved:   {stats['reserved_bytes.all.current'] / 1e9:.2f} GB")
print(f"Peak reserved:      {stats['reserved_bytes.all.peak'] / 1e9:.2f} GB")
print(f"Active allocations: {stats['active.all.current']}")
print(f"Peak active allocs: {stats['active.all.peak']}")
print(f"Total alloc calls:  {stats['allocation.all.current']}")
```

### Important stat keys

| Key Pattern | Description |
|-------------|-------------|
| `allocated_bytes.all.current` | Current allocated memory (bytes) |
| `allocated_bytes.all.peak` | Peak allocated memory |
| `reserved_bytes.all.current` | Current reserved memory |
| `reserved_bytes.all.peak` | Peak reserved memory |
| `active.all.current` | Number of currently active allocations |
| `active.all.peak` | Peak number of simultaneous allocations |
| `inactive_split_bytes.all.current` | Current fragmentation (inactive splits) |
| `num_alloc_retries` | How many times allocator retried after `cudaMalloc` failure |
| `num_ooms` | Number of OOM errors caught by the allocator |

### Logging to Training Loop

```python
def log_memory_stats(step, writer):
    stats = torch.cuda.memory_stats()
    writer.add_scalar('memory/allocated_gb',
                      stats['allocated_bytes.all.current'] / 1e9, step)
    writer.add_scalar('memory/peak_allocated_gb',
                      stats['allocated_bytes.all.peak'] / 1e9, step)
    writer.add_scalar('memory/reserved_gb',
                      stats['reserved_bytes.all.current'] / 1e9, step)
    writer.add_scalar('memory/fragmentation_mb',
                      stats['inactive_split_bytes.all.current'] / 1e6, step)
    writer.add_scalar('memory/num_alloc_retries',
                      stats['num_alloc_retries'], step)
```

---

## 6. Memory Snapshots

Memory snapshots capture a complete record of every allocation and deallocation, including Python stack traces. This is the most powerful debugging tool for memory issues.

### Recording Snapshots

```python
# Start recording allocation history
torch.cuda.memory._record_memory_history(max_entries=100_000)

# Run your code
model = MyModel().cuda()
output = model(batch)
loss = criterion(output, target)
loss.backward()
optimizer.step()

# Save the snapshot
torch.cuda.memory._dump_snapshot("snapshot.pickle")

# Stop recording
torch.cuda.memory._record_memory_history(enabled=None)
```

### Visualizing with PyTorch Memory Viz

PyTorch provides an interactive HTML visualizer. There are two ways to use it:

**Option 1: Use the online tool**

Upload `snapshot.pickle` to [pytorch.org/memory_viz](https://pytorch.org/memory_viz).

**Option 2: Generate HTML locally**

```python
from torch.cuda._memory_viz import segment_plot, trace_plot

# Read the snapshot
import pickle
with open("snapshot.pickle", "rb") as f:
    snapshot = pickle.load(f)

# Generate plots
with open("segment_plot.html", "w") as f:
    f.write(segment_plot(snapshot))

with open("trace_plot.html", "w") as f:
    f.write(trace_plot(snapshot))
```

### What Snapshots Show

- **Segment plot**: shows how memory blocks are allocated over time, colored by which Python call site allocated them. Reveals fragmentation patterns.
- **Trace plot**: timeline of allocations/deallocations. Shows exactly where peak memory occurs and what's live at that point.
- **Stack traces**: for each allocation, you get the full Python stack trace, so you know exactly which line of code created each tensor.

### Best Practices for Snapshots

1. Record during a **single training step** — recording for many steps generates huge files
2. Use `max_entries` to limit snapshot size
3. Call `torch.cuda.empty_cache()` before recording to reduce noise from cached blocks
4. Compare snapshots between configurations to identify which optimization helped

---

## 7. Finding Memory Leaks

A memory leak in PyTorch means tensors that should be garbage-collected are kept alive by unintentional references. The symptom is `memory_allocated()` growing monotonically across training steps.

### Common Cause 1: Accumulating Tensors in Lists

```python
# BAD — stores computation graph for every step
all_losses = []
for batch in dataloader:
    loss = model(batch).sum()
    all_losses.append(loss)  # holds entire computation graph!

# GOOD — detach before storing
all_losses = []
for batch in dataloader:
    loss = model(batch).sum()
    all_losses.append(loss.item())  # scalar, no graph reference
```

### Common Cause 2: Not Detaching from Computation Graph

```python
# BAD — hidden_state retains the graph from the forward pass
hidden_state = model.get_hidden(batch)
# hidden_state.grad_fn exists, keeping all intermediates alive

# GOOD — detach to break the graph
hidden_state = model.get_hidden(batch).detach()
```

### Common Cause 3: Closures Holding References

```python
# BAD — closure captures `output` tensor
def make_callback(output):
    def callback():
        print(output.shape)  # keeps output alive indefinitely
    return callback

# GOOD — capture only what you need
def make_callback(shape):
    def callback():
        print(shape)
    return callback
callback = make_callback(output.shape)
del output
```

### Common Cause 4: Global Variables and Caches

```python
# BAD — module-level cache grows without bound
_cache = {}
def forward(x, key):
    result = model(x)
    _cache[key] = result  # never cleared!
    return result

# GOOD — use bounded cache or WeakRef
from weakref import WeakValueDictionary
_cache = WeakValueDictionary()
```

### Detection Pattern

```python
def detect_memory_leak(model, dataloader, num_steps=10):
    """Run a few steps and check if memory grows."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    memory_readings = []
    for i, batch in enumerate(dataloader):
        if i >= num_steps:
            break
        output = model(batch)
        loss = output.sum()
        loss.backward()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        memory_readings.append(torch.cuda.memory_allocated())

    # Check for monotonic growth
    for i in range(1, len(memory_readings)):
        if memory_readings[i] > memory_readings[0] * 1.1:
            print(f"WARNING: Memory grew from {memory_readings[0]/1e6:.1f} MB "
                  f"to {memory_readings[i]/1e6:.1f} MB in {i} steps")
            return True
    print("No memory leak detected")
    return False
```

---

## 8. torch.cuda.empty_cache()

### What It Does

`empty_cache()` releases all **unused cached memory** held by the PyTorch caching allocator back to CUDA. It does NOT free tensors that are still alive.

```python
# Before: reserved = 2 GB, allocated = 500 MB
torch.cuda.empty_cache()
# After: reserved ≈ 500 MB, allocated = 500 MB (unchanged)
```

### When to Use

1. **Between training phases** — after loading a model but before the first forward pass
2. **After deleting large tensors** — when you know you won't need that memory pattern again
3. **Before memory-critical operations** — to maximize available contiguous memory
4. **Between experiments** — when switching from one model/config to another

```python
# Good: between phases
model_1 = train_phase_1(...)
del model_1
torch.cuda.empty_cache()  # return memory to CUDA before loading phase 2
model_2 = load_phase_2_model()
```

### When NOT to Use

1. **During training** — calling `empty_cache()` every step forces reallocation overhead
2. **As a "fix" for OOM** — it won't help if your tensors genuinely don't fit
3. **Inside tight loops** — the caching allocator exists to avoid expensive `cudaMalloc`/`cudaFree` calls

```python
# BAD — defeats the purpose of the caching allocator
for batch in dataloader:
    output = model(batch)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    torch.cuda.empty_cache()  # forces realloc every step — slower!
```

---

## 9. Memory Optimization Techniques

### 9.1 Gradient Checkpointing (Trade Compute for Memory)

Instead of storing all intermediate activations during forward, recompute them during backward:

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedTransformerBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        return checkpoint(self.block, x, use_reentrant=False)
```

**Memory savings**: reduces activation memory from O(L) to O(√L) for L layers.
**Cost**: ~33% more compute (one extra forward pass per checkpointed segment).

For SAC (Selective Activation Checkpointing), see [Module 16](../16_activation_checkpointing/).

### 9.2 Mixed Precision Training (Halve Activation Memory)

Using `bf16` or `fp16` instead of `fp32` halves the memory for activations and parameters:

```python
# Using torch.autocast
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(batch)
    loss = criterion(output, target)

# Or cast model directly
model = model.to(dtype=torch.bfloat16)
```

**Memory savings**: activations and parameters cut in half. Optimizer state may still be fp32.

### 9.3 Gradient Accumulation (Smaller Per-Step Batch)

Process a large effective batch using smaller micro-batches:

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        output = model(batch)
        loss = criterion(output, target) / accumulation_steps

    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Memory savings**: peak activation memory scales with `micro_batch_size`, not `effective_batch_size`. Using 4× accumulation = 4× smaller activation footprint.

### 9.4 In-Place Operations

Some operations can be done in-place to avoid allocating new tensors:

```python
# Out-of-place: allocates new tensor
x = x + 1        # new tensor
x = F.relu(x)    # new tensor

# In-place: modifies tensor directly
x.add_(1)         # no new allocation
x = F.relu(x, inplace=True)  # reuses memory
```

**Warning**: in-place operations on tensors that require gradients can break autograd:

```python
# This will raise an error if x requires grad and is needed for backward
x.add_(1)  # RuntimeError: a leaf Variable that requires grad has been used in an in-place operation
```

Use in-place operations only on tensors that don't need gradients or that aren't saved for backward.

### 9.5 `del` + `gc.collect()` for Large Intermediates

Explicitly delete large tensors and trigger garbage collection:

```python
import gc

# After using a large intermediate
large_features = extract_features(data)  # several GB
predictions = classify(large_features)

del large_features
gc.collect()
torch.cuda.empty_cache()  # optional: return blocks to CUDA
```

### 9.6 `torch.cuda.empty_cache()` Between Phases

See [Section 8](#8-torchcudaempty_cache) above for detailed guidance.

### 9.7 CPU Offloading

FSDP2 can offload parameters and gradients to CPU between forward/backward:

```python
from torch.distributed._composable.fsdp import fully_shard, CPUOffloadPolicy

policy = CPUOffloadPolicy(pin_memory=True)
for layer in model.layers:
    fully_shard(layer, offload_policy=policy)
fully_shard(model, offload_policy=policy)
```

**Memory savings**: massive — only one layer's parameters on GPU at a time.
**Cost**: CPU↔GPU transfer overhead. `pin_memory=True` helps with async transfers.

### 9.8 Reducing Optimizer Memory

| Approach | Memory vs. Adam | Trade-off |
|----------|----------------|-----------|
| SGD + momentum | 50% less | May need different hyperparameters |
| 8-bit Adam (bitsandbytes) | 75% less | Slight accuracy impact |
| Adafactor | ~50% less | Uses row/column factorization |
| LoRA (low-rank adaptation) | 90%+ less | Only trains adapter weights |
| GaLore | ~65% less | Projects gradients to low-rank space |

```python
# 8-bit Adam with bitsandbytes
import bitsandbytes as bnb
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=1e-4)
```

### Summary Table

| Technique | Saves | Cost | Typical Reduction |
|-----------|-------|------|-------------------|
| Gradient checkpointing | Activations | ~33% more compute | 60-70% activation memory |
| Mixed precision (bf16) | Parameters + activations | None (sometimes better) | ~50% |
| Gradient accumulation | Activations | None | Proportional to accum steps |
| In-place operations | Intermediate tensors | Autograd limitations | 5-15% |
| `del` + `gc.collect()` | Named intermediates | Manual management | Variable |
| CPU offloading | Parameters + optimizer | Transfer overhead | Up to 90% GPU memory |
| 8-bit optimizers | Optimizer state | Slight accuracy impact | 75% optimizer memory |

---

## 10. torch.profiler for Memory

The PyTorch profiler can track memory allocations alongside compute:

```python
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,  # enable memory profiling
    record_shapes=True,
    with_stack=True,  # include Python stack traces
    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=tensorboard_trace_handler('./log/memory_profile'),
) as prof:
    for step, batch in enumerate(dataloader):
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        prof.step()
```

### Memory Timeline

With `profile_memory=True`, the profiler records every allocation and deallocation. In TensorBoard, this appears as a **memory timeline** showing:

- **Memory curve**: total allocated memory over time
- **Allocation spikes**: where peak memory occurs (usually during backward)
- **Memory events**: individual tensor allocations, tagged with their size and operator

### Identifying Allocation Spikes

```python
# Print the top memory-consuming operations
print(prof.key_averages().table(
    sort_by="self_cuda_memory_usage",
    row_limit=20
))
```

This table shows which operations allocate the most GPU memory — the targets for optimization.

### Export for Chrome Trace Viewer

```python
prof.export_chrome_trace("trace_with_memory.json")
# Open in chrome://tracing — memory events appear alongside compute
```

---

## 11. Memory-Efficient Attention

Standard dot-product attention has **O(N²)** memory complexity in sequence length:

```
Standard attention: stores full N×N attention matrix
Memory = batch × heads × seq_len × seq_len × bytes_per_element

Example: batch=8, heads=32, seq=4096, bf16
= 8 × 32 × 4096 × 4096 × 2 = 8.6 GB  ← just for attention scores!
```

### Flash Attention: O(N) Memory

Flash Attention (used by default in `F.scaled_dot_product_attention`) never materializes the full N×N matrix. Instead, it computes attention in tiles:

```python
import torch.nn.functional as F

# This automatically uses Flash Attention when available
output = F.scaled_dot_product_attention(query, key, value)

# Memory: O(N) instead of O(N²)
# For seq=4096: ~100× less memory for the attention computation
```

### Memory Impact on Long Sequences

| Sequence Length | Standard Attention | Flash Attention | Savings |
|----------------|-------------------|-----------------|---------|
| 512 | 16 MB | 0.25 MB | 64× |
| 2048 | 256 MB | 1 MB | 256× |
| 4096 | 1 GB | 2 MB | 512× |
| 16384 | 16 GB | 8 MB | 2048× |
| 65536 | 256 GB (impossible) | 32 MB | — |

Flash Attention is what makes long-context LLMs (100k+ tokens) feasible.

### FlexAttention

For custom attention patterns with memory efficiency:

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

block_mask = create_block_mask(causal_mask, B=1, H=1, Q_LEN=4096, KV_LEN=4096)
output = flex_attention(query, key, value, block_mask=block_mask)
```

FlexAttention maintains O(N) memory while supporting arbitrary attention patterns. See [Module 09](../09_attention/) for full coverage.

---

## 12. Practical: Estimating Memory Before Training

### Using `meta` Device

The `meta` device creates tensors with shapes and dtypes but no actual storage — perfect for estimating memory without a GPU:

```python
with torch.device('meta'):
    model = MyLargeModel(config)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"Parameters: {total_params:,} ({param_bytes / 1e9:.2f} GB)")
```

### Full Memory Estimator

```python
def estimate_training_memory(
    model_params: int,
    dtype_bytes: int = 2,     # 2 for bf16, 4 for fp32
    optimizer: str = "adam",  # "sgd", "adam", "adam_8bit"
    batch_size: int = 1,
    seq_len: int = 2048,
    hidden_dim: int = 4096,
    num_layers: int = 32,
    num_heads: int = 32,
    use_flash_attn: bool = True,
    gradient_checkpointing: bool = False,
    cuda_context_gb: float = 0.5,
) -> dict:
    """Estimate GPU memory needed for training."""

    # Parameters
    param_gb = model_params * dtype_bytes / 1e9

    # Gradients (same dtype as params)
    grad_gb = param_gb

    # Optimizer state
    if optimizer == "sgd":
        optim_gb = model_params * 4 / 1e9  # momentum buffer in fp32
    elif optimizer == "adam":
        optim_gb = model_params * 4 * 2 / 1e9  # m + v in fp32
        if dtype_bytes < 4:
            optim_gb += model_params * 4 / 1e9  # master weights
    elif optimizer == "adam_8bit":
        optim_gb = model_params * 1 * 2 / 1e9  # 8-bit m + v

    # Activations (rough estimate for Transformer)
    bytes_per_activation = dtype_bytes
    per_layer_act = 2 * batch_size * seq_len * hidden_dim * bytes_per_activation
    if use_flash_attn:
        attn_mem = batch_size * num_heads * seq_len * 64 * bytes_per_activation
    else:
        attn_mem = batch_size * num_heads * seq_len * seq_len * bytes_per_activation

    effective_layers = num_layers
    if gradient_checkpointing:
        effective_layers = int(num_layers ** 0.5)  # sqrt(L) with checkpointing

    act_gb = effective_layers * (per_layer_act + attn_mem) / 1e9

    total = param_gb + grad_gb + optim_gb + act_gb + cuda_context_gb

    return {
        "parameters_gb": param_gb,
        "gradients_gb": grad_gb,
        "optimizer_gb": optim_gb,
        "activations_gb": act_gb,
        "cuda_context_gb": cuda_context_gb,
        "total_gb": total,
    }
```

### "Will This Fit on My GPU?" Calculator

```python
def will_it_fit(total_memory_gb: float, gpu: str = "A100-80GB") -> dict:
    gpu_memory = {
        "RTX-3090": 24, "RTX-4090": 24, "A100-40GB": 40,
        "A100-80GB": 80, "H100-80GB": 80, "H200-141GB": 141,
    }
    available = gpu_memory.get(gpu, 80)
    usable = available * 0.90  # ~10% overhead for CUDA/driver
    fits = total_memory_gb <= usable
    headroom = usable - total_memory_gb

    return {
        "gpu": gpu,
        "gpu_memory_gb": available,
        "usable_gb": usable,
        "estimated_gb": total_memory_gb,
        "fits": fits,
        "headroom_gb": headroom if fits else 0,
        "gpus_needed": max(1, int(total_memory_gb / usable) + 1) if not fits else 1,
    }
```

### Example: 13B Model

```python
estimate = estimate_training_memory(
    model_params=13_000_000_000,
    dtype_bytes=2,        # bf16
    optimizer="adam",
    batch_size=4,
    seq_len=2048,
    hidden_dim=5120,
    num_layers=40,
    num_heads=40,
)
# Parameters:  26.0 GB
# Gradients:   26.0 GB
# Optimizer:  104.0 GB  (Adam fp32 moments + master weights)
# Activations: ~30 GB
# Total:      ~187 GB → needs 3× A100-80GB with FSDP
```

---

## 13. Upstream Updates (June 17–18, 2026)

Recent changes to the PyTorch codebase relevant to memory profiling, FX infrastructure, and profiler internals:

### FX Canonicalize Pass

A new canonicalization pass at `torch/fx/passes/canonicalize.py` normalizes FX graph node ordering for deterministic graph comparisons. This helps when comparing graph structures before and after memory optimization passes, ensuring that semantically equivalent graphs produce identical canonical forms regardless of insertion order.

### ShapesSpec Binding Helpers

New utilities in `torch/fx/experimental/_spec_binding.py` provide structured binding for shape specifications in FX graphs. These helpers enable cleaner expression of dynamic shape constraints, which is relevant for memory estimation — knowing tensor shapes at trace time allows accurate activation memory predictions.

### Profiler pattern_matcher Removed

The profiler's internal `pattern_matcher` module was refactored out. Pattern-based profiling analysis (identifying common patterns like conv-bn-relu fusion opportunities) has been reorganized into more targeted analysis passes, reducing profiler overhead for memory-focused profiling sessions.

### Profiler RecordFunction Drain Fix (#187483)

A fix for `RecordFunction` callback drain ordering in the profiler. Previously, when profiling memory-intensive workloads, callback cleanup could trigger additional allocations during drain, inflating peak memory measurements. The fix ensures callbacks are drained in the correct order, giving more accurate memory profiles.

### _scaled_mm_v2 Swizzled Scales Test (#186948)

New tests for `_scaled_mm_v2` with swizzled quantization scales. This is relevant to FP8 training memory optimization — swizzled scales enable more efficient memory layout for quantized matrix multiplications, reducing both memory footprint and fragmentation from scale tensors.

### Dynamo Canonicalize output_graph Node Order (#181775)

Dynamo now canonicalizes the node order in `output_graph`, ensuring deterministic compilation output. For memory profiling, this means memory allocation patterns are reproducible across runs, making it easier to isolate the impact of individual optimizations.

### Dynamic Spec Error Messages (#187143)

Improved error messages for dynamic shape specification mismatches. When memory estimation relies on symbolic shapes (e.g., via `torch.export` with dynamic shapes), clearer error messages help diagnose why estimated memory budgets differ from actual usage due to unexpected shape specialization.

---

## Best Practices Checklist

Before training a large model, work through this checklist:

1. **Estimate first**: Use meta device to compute parameter + optimizer memory. Don't guess.
2. **Pick your precision**: bf16 is free performance and memory. Use it unless you have a reason not to.
3. **Enable Flash Attention**: ensure `F.scaled_dot_product_attention` is routing to the flash kernel.
4. **Consider gradient checkpointing**: if activations dominate, checkpoint every N layers.
5. **Right-size your batch**: use gradient accumulation to decouple batch size from memory.
6. **Profile before optimizing**: use `memory_summary()` or snapshots to find the actual bottleneck.
7. **Monitor during training**: log `memory_allocated()` per step to catch leaks early.
8. **Choose your optimizer wisely**: Adam costs 2× parameters in state. SGD or 8-bit Adam costs much less.
9. **Test at scale gradually**: increase batch size / seq_len incrementally, checking peak memory each time.
10. **Use FSDP for multi-GPU**: shards parameters, gradients, and optimizer state across GPUs.

---

### Further Resources

- [PyTorch Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management) — official CUDA memory docs
- [Memory Snapshot Visualizer](https://pytorch.org/memory_viz) — interactive memory snapshot viewer
- [Training a 1T Model](https://pytorch.org/blog/) — scaling techniques for massive models
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135) — the algorithm behind O(N) attention memory
- [Module 07 — Training Pipelines](../07_training/) — mixed precision and gradient accumulation
- [Module 16 — Activation Checkpointing](../16_activation_checkpointing/) — SAC details

---

<div align="center">

[← Previous Module](../25_triton_kernels/) | [🏠 Home](../README.md) | Next Module →

**Notebook**: [`26_memory_profiling.ipynb`](../notebooks/26_memory_profiling.ipynb)

</div>
