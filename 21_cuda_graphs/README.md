<div align="center">

[← Previous Module](../20_backends_tuning/) | [🏠 Home](../README.md) | [Next Module →](../22_llm_recipes/)

</div>

---

> **Module 21** of the PyTorch Complete Learning Guide
> **Prerequisites:** [Module 07 — Training](../07_training/), [Module 08 — torch.compile](../08_torch_compile/), [Module 12 — Architectures](../12_model_architectures/)
> **Time to complete:** ~2 hours

### 📁 Files in This Module

| File | Description |
|------|-------------|
| `README.md` | This guide |
| `cuda_graphs.py` | CUDA Graphs — capture, replay, static inputs, benchmarking, torch.compile reduce-overhead |

---

# Module 21: CUDA Graphs — Eliminating CPU Launch Overhead

*Day 7 of the incremental learning series*

---

## Table of Contents

1. [What Are CUDA Graphs?](#1-what-are-cuda-graphs)
2. [Why CUDA Graphs Matter](#2-why-cuda-graphs-matter)
3. [Basic API: torch.cuda.CUDAGraph](#3-basic-api-torchcudacudagraph)
4. [The Static Inputs Requirement](#4-the-static-inputs-requirement)
5. [Warmup — Why It's Mandatory](#5-warmup--why-its-mandatory)
6. [CUDA Graph Pools](#6-cuda-graph-pools)
7. [torch.compile with CUDA Graphs](#7-torchcompile-with-cuda-graphs)
8. [Limitations — What Breaks](#8-limitations--what-breaks)
9. [CUDA Graphs with AMP](#9-cuda-graphs-with-amp)
10. [torch.cuda.make_graphed_callables](#10-torchcudamake_graphed_callables)
11. [Practical Patterns](#11-practical-patterns)
12. [When to Use / Not Use](#12-when-to-use--not-use)
13. [Upstream Updates (June 11–12, 2026)](#13-upstream-updates-june-1112-2026)
14. [Further Reading](#14-further-reading)

---

## 1. What Are CUDA Graphs?

When you run PyTorch code on a GPU, every operation — matrix multiply, activation, copy — is a **kernel launch**. Each launch requires the CPU to:

1. Prepare kernel arguments
2. Submit work to the CUDA driver
3. The driver queues it on the GPU stream
4. Return control to the CPU

For large kernels (e.g., a big GEMM), the GPU execution time dwarfs this overhead. But for small/medium operations, the **CPU launch latency** (5–15 microseconds per kernel) can dominate total runtime.

**CUDA Graphs** solve this by recording a sequence of GPU operations into a **graph** during a capture phase, then **replaying** the entire graph with a single CPU-side launch.

```
Normal execution:            CUDA Graph replay:

CPU: launch K1               CPU: launch graph ─────────────────┐
     wait...                                                     │
     launch K2               GPU: K1 → K2 → K3 → K4 → K5       │
     wait...                       (all pre-recorded, no waits)  │
     launch K3                                                   │
     wait...                 Result: 1 CPU launch instead of 5   │
     launch K4               ────────────────────────────────────┘
     wait...
     launch K5

GPU: ▓░░▓░░▓░░▓░░▓          GPU: ▓▓▓▓▓
     (gaps = idle)                 (no gaps)
```

The graph captures:
- Which kernels to run and in what order
- Memory addresses of all inputs and outputs
- Kernel launch parameters

It does **not** capture tensor values — only the operations and where they read/write.

---

## 2. Why CUDA Graphs Matter

### The CPU Bottleneck

Modern GPUs execute small kernels in microseconds. If each kernel takes 3 µs on the GPU but 10 µs for the CPU to launch, you spend **77% of your time on CPU overhead**.

```
Without CUDA Graphs (100 kernels):
  CPU overhead:  100 × 10 µs = 1,000 µs
  GPU compute:   100 ×  3 µs =   300 µs
  Total:                        1,300 µs
  GPU utilization:                  23%

With CUDA Graphs (100 kernels):
  CPU overhead:    1 × 10 µs =    10 µs
  GPU compute:   100 ×  3 µs =   300 µs
  Total:                          310 µs
  GPU utilization:                  97%
  Speedup:                        4.2x
```

### When Speedups Are Largest

| Scenario | Typical Speedup |
|----------|----------------|
| Small model inference (ResNet-18, batch=1) | 3–10x |
| Medium model inference (BERT, batch=8) | 2–5x |
| Large model inference (GPT-2, batch=32) | 1.2–2x |
| Training (full step) | 1.1–1.5x |

The pattern: **the smaller the model and the more kernels per unit of compute, the bigger the win**.

---

## 3. Basic API: torch.cuda.CUDAGraph

### Minimal Example

```python
import torch

model = torch.nn.Linear(512, 512).cuda()
static_input = torch.randn(64, 512, device="cuda")

# Step 1: Warmup (covered in Section 5)
with torch.no_grad():
    for _ in range(3):
        _ = model(static_input)

# Step 2: Capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model(static_input)

# Step 3: Replay
static_input.copy_(torch.randn(64, 512, device="cuda"))
g.replay()
print(static_output)  # Contains result for the new input
```

### What Happens During Capture

When you enter `torch.cuda.graph(g)`:

1. PyTorch switches to a **capturing stream**
2. Every CUDA operation is recorded (not executed normally)
3. Memory allocations inside the block come from a special **graph pool**
4. On exit, the graph is finalized and ready for replay

### What Happens During Replay

`g.replay()` submits the entire recorded sequence to the GPU in one shot. The GPU executes all kernels using the **same memory addresses** that were captured.

This is why inputs must be **static** — the graph hardcodes memory pointers.

---

## 4. The Static Inputs Requirement

This is the most important concept to understand. CUDA Graphs capture **memory addresses**, not tensor values. When you replay, the GPU reads from and writes to the exact same addresses.

### Correct Pattern

```python
# Pre-allocate (addresses are fixed)
static_input = torch.zeros(batch_size, features, device="cuda")
static_output = None

# Capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model(static_input)

# For each new input: copy data IN, replay, read data OUT
for batch in dataloader:
    static_input.copy_(batch.cuda())  # Copy INTO the pre-allocated tensor
    g.replay()                        # GPU reads from same address
    result = static_output.clone()    # Copy OUT (or use in-place)
```

### Wrong Pattern

```python
# DON'T DO THIS — creates new tensor each iteration
for batch in dataloader:
    new_input = batch.cuda()       # New memory address each time!
    g.replay()                     # Graph still reads from OLD address
    # Result: stale data, wrong answers
```

### Why This Design?

Recording memory addresses (not values) is what makes replay so fast. If the graph had to remap pointers each time, it would lose most of its advantage. The tradeoff: you manage input/output buffers manually.

---

## 5. Warmup — Why It's Mandatory

Before capturing a CUDA Graph, you **must** run the model at least once (usually 3 times for safety). Warmup triggers:

| Lazy Initialization | Why It Matters |
|---------------------|---------------|
| cuDNN algorithm selection | First conv/GEMM benchmarks multiple algorithms |
| CUDA context creation | First CUDA call initializes the driver context |
| Memory allocator warmup | Caching allocator builds its pool |
| JIT kernel compilation | Some ops compile PTX on first use |
| cuBLAS handle creation | First matmul creates the handle |

### Warmup Pattern

```python
model = MyModel().cuda().eval()
static_input = torch.randn(B, C, H, W, device="cuda")

# Warmup: run several forward passes
with torch.no_grad():
    for _ in range(3):
        _ = model(static_input)

# Now safe to capture
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model(static_input)
```

### What Happens Without Warmup?

If you skip warmup, the capture records the lazy initialization itself — memory allocations, algorithm searches, handle creation. This means:

1. The graph becomes bloated with one-time setup
2. Replaying re-executes setup on every call (pointless overhead)
3. Some lazy ops allocate memory dynamically, which is **forbidden** inside capture and will raise a `RuntimeError`

---

## 6. CUDA Graph Pools

When multiple graphs share the same model or intermediate buffers, you can share their memory pools to avoid redundant allocations.

### Default: Separate Pools

```python
g1 = torch.cuda.CUDAGraph()
g2 = torch.cuda.CUDAGraph()

with torch.cuda.graph(g1):
    out1 = model(input1)

with torch.cuda.graph(g2):
    out2 = model(input2)
# g1 and g2 each have their own memory pool
```

### Shared Pool

```python
g1 = torch.cuda.CUDAGraph()
g2 = torch.cuda.CUDAGraph()

# Capture both with the same pool
with torch.cuda.graph(g1):
    out1 = model(input1)

# Share g1's pool — no extra memory allocated for g2
with torch.cuda.graph(g2, pool=g1.pool()):
    out2 = model(input2)
```

Pool sharing works when graphs **don't execute concurrently**. The shared pool means they reuse the same memory, so overlapping execution would corrupt data.

### When to Share Pools

- Multiple batch-size variants of the same model
- Encoder and decoder graphs that run sequentially
- A/B model comparisons that never overlap

---

## 7. torch.compile with CUDA Graphs

The easiest way to use CUDA Graphs is through `torch.compile`:

```python
model = MyModel().cuda()

# reduce-overhead mode automatically uses CUDA Graphs
compiled = torch.compile(model, mode="reduce-overhead")

# Just use it normally — graphs are managed for you
output = compiled(input_tensor)
```

### How It Works: cudagraph_trees

Under the hood, `reduce-overhead` mode uses Inductor's **cudagraph_trees** system:

```
torch.compile(mode="reduce-overhead")
  └─ Dynamo traces the Python code
      └─ AOTAutograd generates forward/backward
          └─ Inductor generates optimized CUDA code
              └─ cudagraph_trees wraps each compiled region in a graph
```

cudagraph_trees manages:
- **Automatic warmup** — runs the compiled code once before capturing
- **Graph caching** — one graph per unique input shape
- **Memory management** — pools are handled internally
- **Fallback** — if a region can't be graphed, it runs eagerly

### Advantages Over Manual Graphs

| Manual `CUDAGraph` | `torch.compile(mode="reduce-overhead")` |
|---------------------|----------------------------------------|
| You manage static inputs | Inputs handled automatically |
| You do warmup | Warmup is automatic |
| Entire model must be graphable | Per-region graphs (partial capture) |
| No fusion | Kernel fusion + graphs combined |
| Fixed shapes only | Multiple shape variants cached |

### Checking What Got Graphed

```python
import torch._dynamo as dynamo

compiled = torch.compile(model, mode="reduce-overhead")
explanation = dynamo.explain(compiled)(sample_input)
print(explanation)
```

---

## 8. Limitations — What Breaks

CUDA Graphs capture a **fixed sequence of GPU operations**. Anything that deviates from this at replay time will fail or produce wrong results.

### Hard Failures (RuntimeError)

These operations **cannot** be captured and will raise errors:

| Operation | Why It Fails |
|-----------|-------------|
| `print(tensor)` inside graph | Requires CPU sync |
| `tensor.item()` | Transfers data to CPU |
| `tensor.cpu()` | Cross-device copy |
| `torch.tensor([1, 2, 3])` | CPU tensor creation |
| `torch.cuda.synchronize()` | Blocks the stream |
| Dynamic memory allocation | Graph can't record variable-size allocs |

### Silent Failures (Wrong Results)

These won't crash but will produce incorrect output:

| Pattern | Problem |
|---------|---------|
| Data-dependent control flow (`if x > 0`) | Condition was recorded at capture time |
| Dynamic shapes (varying batch size) | Graph hardcodes tensor dimensions |
| In-place ops on non-static tensors | Writes to wrong addresses |
| Random ops without manual seed | Same random values on every replay |

### Operations That Prevent Capture

```python
# These will fail during capture:

# 1. CPU sync
with torch.cuda.graph(g):
    out = model(x)
    print(out.sum().item())  # ERROR: .item() syncs to CPU

# 2. Dynamic allocation
with torch.cuda.graph(g):
    out = model(x)
    mask = out > 0          # OK so far
    filtered = out[mask]    # ERROR: output size depends on data

# 3. CPU tensor creation
with torch.cuda.graph(g):
    scale = torch.tensor(2.0)  # ERROR: creates CPU tensor
    out = model(x) * scale

# Fix: pre-allocate the scale on GPU
scale = torch.tensor(2.0, device="cuda")
with torch.cuda.graph(g):
    out = model(x) * scale     # OK: scale already on GPU
```

### NCCL and Distributed

Most NCCL collective operations (all-reduce, all-gather, etc.) are **not compatible** with CUDA Graphs. This is why CUDA Graphs are primarily used for inference and single-GPU training.

Exception: PyTorch's `torch.distributed` has experimental support for graphing some collectives on newer NCCL versions.

---

## 9. CUDA Graphs with AMP

Automatic Mixed Precision works inside CUDA Graph capture, but you must set up the autocast context **inside** the capture block:

```python
model = MyModel().cuda()
static_input = torch.randn(64, 512, device="cuda")

# Warmup with AMP
with torch.no_grad(), torch.cuda.amp.autocast():
    for _ in range(3):
        _ = model(static_input)

# Capture with AMP
g = torch.cuda.CUDAGraph()
with torch.cuda.amp.autocast():
    with torch.cuda.graph(g):
        static_output = model(static_input)

# Replay (autocast context not needed — types are baked into the graph)
static_input.copy_(new_data)
g.replay()
```

### Key Points

- The autocast context must be **inside** capture (or wrapping it) so the graph records the mixed-precision kernel variants
- After capture, replay uses whatever dtypes were recorded — no autocast needed at replay time
- GradScaler is harder — avoid it with CUDA Graphs. If you need training with AMP + graphs, prefer `torch.compile(mode="reduce-overhead")`

---

## 10. torch.cuda.make_graphed_callables

`make_graphed_callables` is a convenience wrapper that handles warmup, capture, and static-input management for `nn.Module` or simple callables:

```python
model = MyModel().cuda()
sample_input = torch.randn(64, 512, device="cuda")

# Wrap the model — handles warmup and capture automatically
graphed_model = torch.cuda.make_graphed_callables(
    model,
    sample_args=(sample_input,),
    num_warmup_iters=3,
)

# Use like a normal callable
output = graphed_model(sample_input)
```

### Multiple Callables

You can graph multiple modules together, sharing a pool:

```python
encoder = Encoder().cuda()
decoder = Decoder().cuda()

graphed_encoder, graphed_decoder = torch.cuda.make_graphed_callables(
    (encoder, decoder),
    sample_args=(
        (encoder_input,),
        (decoder_input,),
    ),
)
```

### When to Use

- Quick experiments where you want CUDA Graph speedups without manual buffer management
- Models with simple input/output signatures
- Inference-only paths

### Caveats

- Only supports fixed shapes (like manual graphs)
- The returned callable replaces the original forward pass
- Multiple return values need careful handling

---

## 11. Practical Patterns

### Pattern 1: Inference Server

The most common CUDA Graph use case — a model serving predictions with fixed batch size:

```python
class GraphedInferenceServer:
    def __init__(self, model, batch_size, input_dim):
        self.model = model.cuda().eval()
        self.static_input = torch.zeros(
            batch_size, input_dim, device="cuda"
        )
        self.static_output = None
        self.graph = torch.cuda.CUDAGraph()

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = self.model(self.static_input)

        # Capture
        with torch.no_grad():
            with torch.cuda.graph(self.graph):
                self.static_output = self.model(self.static_input)

    def predict(self, input_tensor):
        self.static_input.copy_(input_tensor)
        self.graph.replay()
        return self.static_output.clone()
```

### Pattern 2: Multiple Batch Sizes

For serving with variable batch sizes, capture one graph per batch size:

```python
class MultiBatchServer:
    def __init__(self, model, batch_sizes, input_dim):
        self.model = model.cuda().eval()
        self.graphs = {}

        for bs in batch_sizes:
            static_in = torch.zeros(bs, input_dim, device="cuda")

            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    _ = self.model(static_in)

            g = torch.cuda.CUDAGraph()
            with torch.no_grad():
                with torch.cuda.graph(g):
                    static_out = self.model(static_in)

            self.graphs[bs] = (g, static_in, static_out)

    def predict(self, input_tensor):
        bs = input_tensor.shape[0]
        g, static_in, static_out = self.graphs[bs]
        static_in.copy_(input_tensor)
        g.replay()
        return static_out.clone()
```

### Pattern 3: Partial Graph Capture for Training

Full training steps are hard to graph (optimizer state updates, gradient scaling). Instead, graph just the forward pass:

```python
model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters())
static_input = torch.randn(32, 512, device="cuda")
static_target = torch.randn(32, 10, device="cuda")

# Warmup
for _ in range(3):
    out = model(static_input)
    loss = torch.nn.functional.mse_loss(out, static_target)
    loss.backward()
    optimizer.zero_grad()

# Capture forward + backward (not optimizer step)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    out = model(static_input)
    loss = torch.nn.functional.mse_loss(out, static_target)
    loss.backward()

# Training loop
for batch_input, batch_target in dataloader:
    static_input.copy_(batch_input)
    static_target.copy_(batch_target)
    g.replay()                   # Forward + backward replayed
    optimizer.step()             # Optimizer runs eagerly (safe)
    optimizer.zero_grad()
```

---

## 12. When to Use / Not Use

### Decision Tree

```
Is your model running on NVIDIA GPU?
├─ No  → CUDA Graphs not applicable
└─ Yes
    │
    Are input shapes fixed (or a small set of fixed shapes)?
    ├─ No  → Use torch.compile (handles dynamic shapes)
    └─ Yes
        │
        Is CPU launch overhead significant?
        (small/medium model, many small kernels, high throughput needed)
        ├─ No  → Graphs won't help much, use torch.compile for fusion
        └─ Yes
            │
            Is it inference only?
            ├─ Yes → CUDA Graphs are ideal
            │        Use torch.compile(mode="reduce-overhead")
            │        or manual CUDAGraph API
            └─ No (training)
                │
                Can you graph forward+backward only?
                ├─ Yes → Partial graph capture (optimizer runs eagerly)
                └─ No  → torch.compile(mode="reduce-overhead") is safer
```

### Quick Reference

| Scenario | Recommendation |
|----------|---------------|
| Inference, fixed shapes, max throughput | Manual `CUDAGraph` or `reduce-overhead` |
| Inference, variable shapes | `torch.compile(mode="default")` — dynamic shape support |
| Training, single GPU | `torch.compile(mode="reduce-overhead")` — handles optimizer |
| Training, multi-GPU (DDP/FSDP) | `torch.compile(mode="default")` — NCCL compatibility |
| Model has data-dependent control flow | `torch.compile` with graph breaks |
| Quick prototyping | `make_graphed_callables` |

---

## 13. Upstream Updates (June 11–12, 2026)

Recent PyTorch development activity relevant to CUDA Graphs and the broader ecosystem:

### Version Bump to 2.14.0a0 (#187070)

The main branch has been bumped to version 2.14.0a0+, signaling the start of the next development cycle. This is the version that includes the latest CUDA Graph improvements and new features described below.

### FlexGEMM Higher-Order Op (`torch/_higher_order_ops/flex_gemm.py`)

A new higher-order operation for flexible GEMM execution has been added. FlexGEMM allows user-defined epilogues on GEMM results, similar to how FlexAttention allows custom attention score modifications. This interacts with CUDA Graphs through the `reduce-overhead` compilation path.

### c10d Window Interfaces for One-Sided Communication (#186299)

New window-based interfaces in PyTorch's distributed backend (c10d) enable one-sided MPI-style communication patterns (put, get, accumulate). While most NCCL collectives remain incompatible with CUDA Graphs, these new primitives expand the distributed toolkit.

### cuSOLVER Workspace Optimization (#181998)

Workspace allocation for cuSOLVER operations has been optimized, reducing memory overhead for linear algebra operations. This is relevant to CUDA Graphs because workspace allocations during graph capture can cause issues — the optimization makes these allocations more predictable and graph-friendly.

### Dynamo `itertools.permutations` Polyfill (#186937)

`torch._dynamo` now supports `itertools.permutations` during tracing, allowing more Python code to be captured without graph breaks. Fewer graph breaks mean larger graphable regions, which directly benefits CUDA Graph capture through `torch.compile(mode="reduce-overhead")`.

### uint16/uint32/uint64 Test Coverage Extended (#183473)

Test infrastructure now covers unsigned integer types more broadly. While not directly related to CUDA Graphs, this improves the reliability of quantized models that may be served with CUDA Graph-accelerated inference.

### FlexAttention INDEX_DTYPE for Pointer Arithmetic (#185264)

FlexAttention now uses a dedicated `INDEX_DTYPE` for pointer arithmetic in block-sparse patterns, improving numerical stability on different GPU architectures. FlexAttention kernels are common targets for CUDA Graph capture in Transformer inference.

---

## 14. Further Reading

- [CUDA Graphs documentation (PyTorch)](https://pytorch.org/docs/stable/cuda.html#cuda-graphs)
- [NVIDIA CUDA Graphs guide](https://developer.nvidia.com/blog/cuda-graphs/)
- [torch.compile reduce-overhead mode](https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html)
- [make_graphed_callables API](https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html)
- [Accelerating PyTorch with CUDA Graphs (NVIDIA blog)](https://developer.nvidia.com/blog/accelerating-pytorch-with-cuda-graphs/)

---

<div align="center">

[← Previous Module](../20_backends_tuning/) | [🏠 Home](../README.md) | [Next Module →](../22_llm_recipes/)

**Notebook**: [`21_cuda_graphs.ipynb`](../notebooks/21_cuda_graphs.ipynb)

</div>
