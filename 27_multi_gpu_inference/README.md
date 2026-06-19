<div align="center">

[← Previous Module](../26_memory_profiling/) | [🏠 Home](../README.md) | [Next Module →](../28_benchmarking/)

</div>

---

> **Module 27** of the PyTorch Complete Learning Guide
> **Prerequisites:** [Module 10 — Distributed Training](../10_distributed/), [Module 11 — Export & Deployment](../11_export_deploy/), [Module 22 — LLM Recipes](../22_llm_recipes/)
> **Time to complete:** ~3 hours

### 📁 Files in This Module

| File | Description |
|------|-------------|
| [`README.md`](README.md) | This guide — multi-GPU inference strategies, quantization, benchmarking |
| [`inference_patterns.py`](inference_patterns.py) | Model size estimation, device_map sharding, KV cache sizing, decision tree |
| [`model_sharding.py`](model_sharding.py) | Manual sharding, TP/PP patterns, continuous batching, AOTInductor workflow |

---

# Multi-GPU Inference Patterns — Serving Large Models at Scale

## Table of Contents

1. [Why Multi-GPU Inference?](#1-why-multi-gpu-inference)
2. [Strategy 1: Tensor Parallel Inference](#2-strategy-1-tensor-parallel-inference)
3. [Strategy 2: Pipeline Parallel Inference](#3-strategy-2-pipeline-parallel-inference)
4. [Strategy 3: Simple Model Sharding with device_map](#4-strategy-3-simple-model-sharding-with-device_map)
5. [KV Cache Across GPUs](#5-kv-cache-across-gpus)
6. [Continuous Batching](#6-continuous-batching)
7. [Quantized Inference](#7-quantized-inference)
8. [torch.compile for Inference](#8-torchcompile-for-inference)
9. [AOTInductor for Production](#9-aotinductor-for-production)
10. [Benchmarking Inference](#10-benchmarking-inference)
11. [Decision Tree: Choosing a Strategy](#11-decision-tree-choosing-a-strategy)
12. [Upstream Updates (June 18–19, 2026)](#12-upstream-updates-june-18-19-2026)

---

## 1. Why Multi-GPU Inference?

Modern large language models simply do not fit in the memory of a single GPU. A 70B parameter model in FP16 requires **140 GB** of memory just for the weights — exceeding even the 80 GB available on an A100 or H100:

```
Model Size (FP16 weights only):
  7B  parameters →  14 GB  (fits 1× A100-80GB)
  13B parameters →  26 GB  (fits 1× A100-80GB)
  34B parameters →  68 GB  (fits 1× A100-80GB, tight)
  70B parameters → 140 GB  (needs 2× A100-80GB minimum)
 175B parameters → 350 GB  (needs 5× A100-80GB)
 405B parameters → 810 GB  (needs 11× A100-80GB)
```

Even models that fit on one GPU can benefit from multi-GPU inference for two reasons:

### 1.1 Latency Reduction

Tensor parallelism splits each matrix multiplication across GPUs, reducing the per-operation compute time. For latency-sensitive serving (chatbots, real-time APIs), splitting a 7B model across 2 GPUs can halve per-token generation time.

### 1.2 Throughput Scaling

Pipeline parallelism and continuous batching allow you to process more requests simultaneously. While GPU 0 generates tokens for request A, GPU 1 processes request B's prefill. Throughput scales roughly linearly with GPU count.

### 1.3 Latency vs Throughput Tradeoff

```
                    ┌─────────────────────────────────┐
                    │         Inference Goals          │
                    ├────────────────┬────────────────┤
                    │   Low Latency  │ High Throughput │
                    ├────────────────┼────────────────┤
                    │ Tensor Parallel│ Pipeline Parallel│
                    │ CUDA Graphs    │ Continuous Batch │
                    │ reduce-overhead│ Larger batches   │
                    │ Single request │ Many concurrent  │
                    │   focus        │   requests       │
                    └────────────────┴────────────────┘
```

For most production LLM serving, you want **both**: TP within a node for latency, PP across nodes for throughput.

---

## 2. Strategy 1: Tensor Parallel Inference

Tensor Parallelism (TP) splits individual weight matrices across GPUs. Every GPU participates in every layer, each holding a shard of every weight matrix. Communication (all-reduce) happens once per layer.

### 2.1 How TP Works for Transformers

In a Transformer layer, the key operations are linear projections. TP splits these column-wise or row-wise:

```
Column-wise (split output dim):        Row-wise (split input dim):
┌──────────┐                           ┌──────────┐
│  W full   │                           │  W full   │
│ [d, 4d]  │                           │ [4d, d]  │
└──────────┘                           └──────────┘
      ↓ split columns                        ↓ split rows
┌─────┐ ┌─────┐                       ┌─────┐ ┌─────┐
│W_0   │ │W_1   │  ← each on          │W_0   │ │W_1   │
│[d,2d]│ │[d,2d]│     one GPU          │[2d,d]│ │[2d,d]│
└─────┘ └─────┘                       └─────┘ └─────┘
```

**ColwiseParallel**: each GPU computes a subset of the output features. No communication during the matmul. Used for QKV projections and FFN up-projections.

**RowwiseParallel**: each GPU holds a subset of input features. Requires an all-reduce after the matmul to sum partial results. Used for output projections and FFN down-projections.

### 2.2 TP with `torch.distributed.tensor.parallel`

```python
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("tp",))

# Define parallelization plan for a Transformer block
plan = {
    # QKV: split output columns across GPUs
    "attn.qkv_proj": ColwiseParallel(),
    # Output projection: split input rows, all-reduce output
    "attn.out_proj": RowwiseParallel(),
    # FFN up: split output columns
    "ffn.up_proj": ColwiseParallel(),
    "ffn.gate_proj": ColwiseParallel(),
    # FFN down: split input rows, all-reduce output
    "ffn.down_proj": RowwiseParallel(),
}

for layer in model.layers:
    parallelize_module(layer, mesh["tp"], plan)
```

### 2.3 Communication Cost

Each Transformer layer with TP requires **2 all-reduce operations** (one after attention output projection, one after FFN down projection). For L layers:

```
Total all-reduce calls = 2 × L
Per all-reduce data    = batch × seq_len × hidden_dim × dtype_bytes
```

On NVLink (900 GB/s bidirectional on H100), this overhead is small for large hidden dims. On PCIe (64 GB/s), TP across more than 2 GPUs becomes communication-bound.

### 2.4 TP Best Practices

- Use TP within a single node (NVLink connectivity)
- TP degree should divide `num_heads` and `num_kv_heads` evenly
- TP=2 for 7-13B models, TP=4 for 34-70B, TP=8 for 70B+ on a single node
- Combine with PP for models spanning multiple nodes

---

## 3. Strategy 2: Pipeline Parallel Inference

Pipeline Parallelism (PP) assigns different layers to different GPUs. GPU 0 holds layers 0-15, GPU 1 holds layers 16-31. Data flows sequentially through the pipeline.

### 3.1 How PP Works

```
Request → [GPU 0: Layers 0-15] → [GPU 1: Layers 16-31] → Output
               embed, norm          layers 16-31, head

PP=4 example (70B, 80 layers):
GPU 0: embed + layers  0-19   (20 layers)
GPU 1: layers 20-39            (20 layers)
GPU 2: layers 40-59            (20 layers)
GPU 3: layers 60-79 + head    (20 layers)
```

### 3.2 Pipeline with Micro-Batching

Without micro-batching, PP has terrible utilization — only one GPU is active at a time. Micro-batching fills the pipeline:

```
Time →
GPU 0: [batch0] [batch1] [batch2] [batch3]  idle    idle    idle    idle
GPU 1:  idle   [batch0] [batch1] [batch2] [batch3]  idle    idle    idle
GPU 2:  idle    idle   [batch0] [batch1] [batch2] [batch3]  idle    idle
GPU 3:  idle    idle    idle   [batch0] [batch1] [batch2] [batch3]  idle
```

Pipeline bubble = `(PP_degree - 1) / (PP_degree + num_microbatches - 1)`. With 4 GPUs and 8 micro-batches, the bubble is 3/11 ≈ 27%. More micro-batches shrink the bubble.

### 3.3 PP Communication

PP communicates only at pipeline stage boundaries — the hidden states between consecutive layer groups. This is **point-to-point** (send/recv), not all-reduce:

```
Communication per stage boundary:
  data = batch × seq_len × hidden_dim × dtype_bytes

For 70B (hidden=8192, bf16), batch=1, seq=4096:
  = 1 × 4096 × 8192 × 2 = 64 MB per boundary
```

Much less communication than TP, making PP suitable for cross-node distribution.

### 3.4 PP vs TP Tradeoffs

| Aspect | Tensor Parallel | Pipeline Parallel |
|--------|----------------|-------------------|
| Communication | All-reduce per layer | Point-to-point between stages |
| Latency | Lower (all GPUs active per token) | Higher (pipeline bubble) |
| Throughput | Limited by TP comm | Scales with micro-batches |
| Best interconnect | NVLink (intra-node) | Works on PCIe/InfiniBand |
| Memory balance | Even (same layers on all GPUs) | Can be uneven (first/last stage) |

---

## 4. Strategy 3: Simple Model Sharding with device_map

The simplest multi-GPU approach: manually assign model components to different devices. No distributed communication library needed.

### 4.1 Manual device_map

```python
model = MyLLM(config)

# Assign layers to GPUs
model.embed.to('cuda:0')
model.layers[:16].to('cuda:0')
model.layers[16:].to('cuda:1')
model.head.to('cuda:1')
```

### 4.2 Forward Pass with Cross-Device Transfer

The forward pass must move activations between devices at the boundary:

```python
def forward(self, input_ids):
    # Phase 1: on cuda:0
    x = self.embed(input_ids.to('cuda:0'))
    for layer in self.layers[:16]:
        x = layer(x)

    # Transfer activations to cuda:1
    x = x.to('cuda:1')

    # Phase 2: on cuda:1
    for layer in self.layers[16:]:
        x = layer(x)
    x = self.norm(x)
    logits = self.head(x)
    return logits
```

### 4.3 Limitations

- **No parallelism within a layer**: only one GPU is active at any time during a single request
- **Sequential execution**: GPU 0 sits idle while GPU 1 runs its layers
- **Good for**: fitting large models for batch inference where latency is not critical
- **Bad for**: real-time serving (doubles latency compared to TP)

### 4.4 Balancing Memory Across Devices

Not all layers are the same size. The embedding and output head can be large (vocab_size × hidden_dim). Balance by assigning more transformer layers to GPUs without the embedding/head:

```python
def compute_device_map(model, num_gpus):
    """Assign layers to GPUs, balancing parameter memory."""
    param_sizes = {}
    for name, param in model.named_parameters():
        device_key = name.split('.')[0]
        param_sizes[device_key] = param_sizes.get(device_key, 0) + param.numel() * param.element_size()

    total = sum(param_sizes.values())
    per_gpu = total / num_gpus

    device_map = {}
    current_gpu, current_load = 0, 0
    for name, size in param_sizes.items():
        device_map[name] = f'cuda:{current_gpu}'
        current_load += size
        if current_load >= per_gpu and current_gpu < num_gpus - 1:
            current_gpu += 1
            current_load = 0
    return device_map
```

---

## 5. KV Cache Across GPUs

During autoregressive generation, the KV cache stores past key and value tensors to avoid recomputation. Multi-GPU inference shards this cache differently depending on the parallelism strategy.

### 5.1 KV Cache in Tensor Parallel

With TP, each GPU holds a shard of the attention heads. The KV cache is naturally sharded — each GPU caches only its head shard:

```
TP=2, 32 heads total:
GPU 0: caches heads 0-15  → cache_size / 2
GPU 1: caches heads 16-31 → cache_size / 2

With GQA (8 KV heads):
GPU 0: caches KV heads 0-3  → cache_size / 2
GPU 1: caches KV heads 4-7  → cache_size / 2
```

### 5.2 KV Cache in Pipeline Parallel

With PP, each stage caches only its own layers' KV pairs. The cache is split by layers, not by heads:

```
PP=2, 32 layers total:
GPU 0 (layers 0-15):  caches 16 layers × full heads
GPU 1 (layers 16-31): caches 16 layers × full heads
```

### 5.3 KV Cache Memory Estimation

```
Per-token KV cache memory:
  = 2 × num_layers × num_kv_heads × head_dim × dtype_bytes

For Llama-70B (80 layers, 8 KV heads, head_dim=128, bf16):
  = 2 × 80 × 8 × 128 × 2 = 327,680 bytes ≈ 320 KB per token

For 4096 tokens:
  = 320 KB × 4096 = 1.28 GB per sequence

For batch_size=32 concurrent requests:
  = 1.28 GB × 32 = 41 GB for KV cache alone
```

### 5.4 Pre-Allocation Strategies

Pre-allocating KV cache avoids memory fragmentation during serving:

```python
def preallocate_kv_cache(num_layers, num_kv_heads, head_dim, max_seq_len,
                         max_batch, dtype=torch.bfloat16, device='cuda'):
    """Pre-allocate KV cache buffers to avoid fragmentation."""
    cache = []
    for _ in range(num_layers):
        k = torch.zeros(max_batch, num_kv_heads, max_seq_len, head_dim,
                        dtype=dtype, device=device)
        v = torch.zeros(max_batch, num_kv_heads, max_seq_len, head_dim,
                        dtype=dtype, device=device)
        cache.append((k, v))
    return cache
```

Pre-allocation trades unused memory for deterministic allocation patterns, which is critical for CUDA Graphs and production serving.

---

## 6. Continuous Batching

Traditional batching waits for all sequences in a batch to finish before starting new ones. Continuous batching (also called in-flight batching) allows new requests to join the batch as soon as any request finishes.

### 6.1 The Problem with Static Batching

```
Static batching (batch=4):
Request A: ████████████████████████████████  (128 tokens)
Request B: ████████                          (32 tokens)  ← idle after 32 tokens
Request C: ████████████████                  (64 tokens)  ← idle after 64 tokens
Request D: ████████████                      (48 tokens)  ← idle after 48 tokens

GPU utilization: only ~68% — short requests waste GPU cycles
```

### 6.2 Continuous Batching

```
Continuous batching:
Request A: ████████████████████████████████
Request B: ████████ E: ████████████████████████
Request C: ████████████████ F: ████████████████
Request D: ████████████ G: ████████████████████

GPU utilization: ~95% — new requests fill slots immediately
```

### 6.3 Implementation Concepts

The key data structures for continuous batching:

```python
class ContinuousBatchScheduler:
    """Simplified continuous batching scheduler."""

    def __init__(self, max_batch_size, max_seq_len):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.active_requests = {}  # request_id -> RequestState
        self.waiting_queue = []    # requests awaiting scheduling

    def step(self):
        """Run one generation step for all active requests."""
        finished = []
        for req_id, state in self.active_requests.items():
            if state.is_finished():
                finished.append(req_id)

        for req_id in finished:
            del self.active_requests[req_id]

        while (len(self.active_requests) < self.max_batch_size
               and self.waiting_queue):
            new_req = self.waiting_queue.pop(0)
            self.active_requests[new_req.id] = new_req
```

This is the core idea behind serving engines like vLLM, TensorRT-LLM, and SGLang. Production implementations add PagedAttention for efficient KV cache management.

---

## 7. Quantized Inference

Quantization reduces model precision to fit larger models on fewer GPUs and improve throughput.

### 7.1 Precision Comparison

```
Precision    Bits/Param   70B Model Size   Quality Impact
─────────────────────────────────────────────────────────
FP32         32 bits      280 GB           Baseline
FP16/BF16    16 bits      140 GB           Negligible
INT8         8 bits       70 GB            Minimal (<1% degradation)
INT4         4 bits       35 GB            Small (1-3% degradation)
NF4          4 bits       35 GB            Very small with double quant
```

### 7.2 Dynamic Quantization (INT8)

```python
import torch.ao.quantization as quant

model = MyLLM(config)
model.eval()

# Dynamic quantization: weights quantized statically, activations quantized dynamically
quantized_model = torch.ao.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # quantize Linear layers
    dtype=torch.qint8,
)
```

Dynamic quantization is the simplest approach — no calibration data needed. Weights are quantized to INT8 at load time, activations are quantized on-the-fly during inference.

### 7.3 Weight-Only Quantization (INT4/INT8)

Weight-only quantization keeps activations in FP16/BF16 but stores weights in lower precision:

```python
from torchao.quantization import quantize_, int4_weight_only, int8_weight_only

model = MyLLM(config).to(dtype=torch.bfloat16)

# INT4 weight-only quantization
quantize_(model, int4_weight_only(group_size=128))

# Or INT8 weight-only
quantize_(model, int8_weight_only())
```

### 7.4 Combining Quantization with TP

Quantize first, then apply tensor parallelism:

```python
# 1. Load and quantize
model = load_model(config)
quantize_(model, int4_weight_only(group_size=128))

# 2. Apply TP
mesh = init_device_mesh("cuda", (tp_size,))
for layer in model.layers:
    parallelize_module(layer, mesh, tp_plan)

# 3. Compile for peak performance
model = torch.compile(model, mode="max-autotune")
```

This combination is how production systems serve 70B models on 2 GPUs: INT4 reduces the 140 GB to 35 GB (fits on 2 × 24 GB GPUs), and TP splits computation for lower latency.

---

## 8. torch.compile for Inference

`torch.compile` applies kernel fusion and optimization for significant inference speedups.

### 8.1 Compile Modes for Inference

```python
# Maximum throughput — longer compile time, best steady-state performance
model = torch.compile(model, mode="max-autotune")

# Lowest latency — uses CUDA Graphs to eliminate kernel launch overhead
model = torch.compile(model, mode="reduce-overhead")

# Default balance
model = torch.compile(model)
```

### 8.2 mode="reduce-overhead" and CUDA Graphs

`reduce-overhead` mode wraps the compiled model in CUDA Graphs, eliminating CPU kernel launch overhead:

```python
model = model.eval().cuda()
model = torch.compile(model, mode="reduce-overhead")

# Warmup: first few calls trigger compilation + graph capture
with torch.no_grad():
    for _ in range(3):
        _ = model(warmup_input)

# Steady state: near-zero CPU overhead per forward pass
with torch.no_grad():
    output = model(real_input)  # runs via CUDA Graph replay
```

### 8.3 Static Shapes for Best Performance

CUDA Graphs require static input shapes. For inference, pad inputs to fixed lengths:

```python
def pad_to_static(input_ids, pad_id=0, max_len=2048):
    """Pad inputs to static shape for CUDA Graph compatibility."""
    batch, seq = input_ids.shape
    if seq < max_len:
        padding = torch.full((batch, max_len - seq), pad_id,
                             dtype=input_ids.dtype, device=input_ids.device)
        input_ids = torch.cat([input_ids, padding], dim=1)
    return input_ids
```

### 8.4 Combining torch.compile with TP

```python
# Apply TP first, then compile each rank's model
for layer in model.layers:
    parallelize_module(layer, mesh, tp_plan)

model = torch.compile(model, mode="max-autotune")
```

Each TP rank compiles independently. The compiled graph includes the communication operations (all-reduce), so they are fused into the overall execution plan.

---

## 9. AOTInductor for Production

AOTInductor (Ahead-of-Time Inductor) pre-compiles PyTorch models into shared libraries (.so files) that can be loaded and run from C++ without any Python dependency.

### 9.1 Export and Compile

```python
import torch
from torch._export import aot_compile

model = MyLLM(config).eval().cuda()
example_input = torch.randint(0, 32000, (1, 2048), device='cuda')

# Export to a .so file
so_path = aot_compile(
    model,
    args=(example_input,),
    options={"max_autotune": True},
)
print(f"Compiled to: {so_path}")
```

### 9.2 Load in C++ (No Python)

```cpp
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>

int main() {
    auto runner = std::make_unique<torch::inductor::AOTIModelContainerRunnerCuda>(
        "model.so"
    );

    auto input = torch::randint(0, 32000, {1, 2048},
                                torch::dtype(torch::kLong).device(torch::kCUDA));
    auto outputs = runner->run({input});
    auto logits = outputs[0];
    return 0;
}
```

### 9.3 PT2 Archive Format

The newer packaging approach bundles model + weights + metadata:

```python
import torch
from torch._inductor.package import package_aoti

model = MyLLM(config).eval()
example = (torch.randint(0, 32000, (1, 2048)),)
ep = torch.export.export(model, example)

# Package to .pt2 archive
package_aoti("model.pt2", ep)

# Load and run (Python)
runner = torch._inductor.package.load_package("model.pt2")
output = runner(input_ids)
```

### 9.4 Benefits for Production

| Aspect | Python (torch.compile) | AOTInductor (.so) |
|--------|----------------------|-------------------|
| Startup time | Compile on first call | Pre-compiled, instant |
| Python GIL | Yes, limits concurrency | No Python needed |
| Deployment | Needs Python + PyTorch | Just the .so + libtorch |
| Debugging | Full Python stack traces | C++ debugging |
| Use case | Development, prototyping | Production serving |

---

## 10. Benchmarking Inference

### 10.1 Key Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| **TTFT** (Time to First Token) | Time from request to first generated token | < 500ms |
| **ITL** (Inter-Token Latency) | Time between consecutive generated tokens | < 30ms |
| **Throughput** | Total tokens generated per second across all requests | Maximize |
| **p50 / p95 / p99 latency** | Percentile latency across requests | p99 < 2× p50 |

### 10.2 TTFT vs ITL

TTFT includes the **prefill phase** (processing the entire prompt). ITL measures the **decode phase** (generating one token at a time):

```
Request lifecycle:
  [Prompt arrives] → [Prefill: process all prompt tokens] → [First token]
                                TTFT ──────────────────────────┘
  [First token] → [Second token] → [Third token] → ... → [EOS]
                  ├─── ITL ────┤├─── ITL ────┤
```

Prefill is compute-bound (one large matmul). Decode is memory-bandwidth-bound (one token at a time, must read all weights).

### 10.3 Measuring with torch.utils.benchmark

```python
import torch.utils.benchmark as benchmark

model = model.eval().cuda()
input_ids = torch.randint(0, 32000, (1, 512), device='cuda')

# Prefill latency
timer_prefill = benchmark.Timer(
    stmt='model(input_ids)',
    globals={'model': model, 'input_ids': input_ids},
    num_threads=1,
)
result_prefill = timer_prefill.blocked_autorange(min_run_time=5.0)
print(f"Prefill (512 tokens): {result_prefill.median * 1000:.1f} ms")

# Decode latency (single token)
single_token = torch.randint(0, 32000, (1, 1), device='cuda')
timer_decode = benchmark.Timer(
    stmt='model(single_token)',
    globals={'model': model, 'single_token': single_token},
    num_threads=1,
)
result_decode = timer_decode.blocked_autorange(min_run_time=5.0)
print(f"Decode (1 token): {result_decode.median * 1000:.2f} ms")
print(f"Max tokens/sec: {1.0 / result_decode.median:.0f}")
```

### 10.4 Throughput Benchmarking

```python
import time

def benchmark_throughput(model, prompts, max_new_tokens=128):
    """Measure end-to-end throughput in tokens/second."""
    total_tokens = 0
    start = time.perf_counter()

    for prompt in prompts:
        output = generate(model, prompt, max_new_tokens=max_new_tokens)
        total_tokens += output.shape[-1]

    elapsed = time.perf_counter() - start
    throughput = total_tokens / elapsed
    print(f"Throughput: {throughput:.0f} tokens/sec")
    print(f"Total time: {elapsed:.2f}s for {total_tokens} tokens")
    return throughput
```

---

## 11. Decision Tree: Choosing a Strategy

```
Model fits on 1 GPU?
├─ YES → Use torch.compile
│        ├─ Latency-sensitive? → mode="reduce-overhead" (CUDA Graphs)
│        └─ Throughput?        → mode="max-autotune" + larger batches
│
└─ NO → How many GPUs needed?
         │
         ├─ 2-4 GPUs (within 1 node, NVLink)
         │   └─ Tensor Parallel
         │       Best single-request latency
         │       Each GPU holds a shard of every layer
         │
         ├─ 4-8 GPUs (1-2 nodes)
         │   └─ TP + PP hybrid
         │       TP within node, PP across nodes
         │       Balance latency and throughput
         │
         └─ Need maximum throughput?
             └─ Pipeline Parallel + continuous batching
                 Fill pipeline with micro-batches
                 New requests join as old ones finish

Additional considerations:
  ├─ Production deployment? → AOTInductor (.so, no Python)
  ├─ Memory constrained?   → INT4 quantization first, then TP
  └─ Simple setup needed?  → device_map sharding (no dist init)
```

### Quick Reference

| Scenario | Recommended Strategy | Why |
|----------|---------------------|-----|
| 7B model, 1 GPU | `torch.compile(mode="reduce-overhead")` | Fits easily, maximize latency |
| 7B model, high QPS | Continuous batching on 1 GPU | Maximize throughput |
| 70B model, 2 GPUs | TP=2 + INT4 quantization | Fits with INT4, TP for latency |
| 70B model, 4 GPUs | TP=4 (FP16) or TP=2 (INT8) | Full precision or quant + TP |
| 70B model, 8 GPUs | TP=4 + PP=2 | TP within node, PP across |
| 405B model, 16 GPUs | TP=8 + PP=2 + INT8 | Multi-node, hybrid parallelism |

---

## 12. Upstream Updates (June 18–19, 2026)

Recent changes to the PyTorch codebase relevant to multi-GPU inference, distributed systems, and compiler infrastructure:

### CUPTI Profiler Refactored

The CUPTI profiler has been refactored into a dedicated `torch/profiler/_cupti/` package, separating CUPTI-specific logic from the general profiler infrastructure. This improves maintainability and makes it easier to profile multi-GPU inference workloads where per-GPU profiling data needs independent collection and aggregation.

### DTensor logspace Support (#186398)

`torch.logspace` now supports DTensor, enabling logarithmically-spaced tensor creation across distributed meshes. Useful for creating learning rate schedules or quantization scales directly on sharded tensors without manual gather/scatter.

### ShapesSpec/ParamsSpec in Non-Strict Export (#187602)

New `ShapesSpec` and `ParamsSpec` support in non-strict export mode allows more flexible shape specifications when exporting models for AOTInductor deployment. This is particularly relevant for inference models with dynamic batch sizes or sequence lengths that need to be compiled ahead of time.

### Distributed Backend Accessors Exposed (#187494)

Backend accessors for distributed communication are now publicly exposed, making it easier to query and configure the communication backend (NCCL, Gloo) programmatically. Useful for inference servers that need to dynamically select backends based on available hardware.

### set_timeout on FakeProcessGroup (#187693)

`FakeProcessGroup` now supports `set_timeout`, enabling better testing of multi-GPU inference code without actual GPUs. Test timeouts for distributed operations can be configured independently, catching hangs in TP/PP initialization logic during unit tests.

### L2-Aware Two-Pass Variance Heuristic (#183661)

A new L2-cache-aware heuristic for two-pass variance computation improves kernel selection in Inductor. For inference workloads that compute layer normalization or RMS normalization across TP shards, this heuristic selects kernels that better utilize GPU L2 cache, reducing memory bandwidth pressure.

---

## Best Practices Checklist

Before deploying a multi-GPU inference system:

1. **Estimate model memory**: use meta device to compute weight + KV cache memory before allocating GPUs.
2. **Choose quantization first**: INT4/INT8 can reduce GPU count by 2-4×. Always quantize before considering more GPUs.
3. **Match parallelism to hardware**: TP within NVLink nodes, PP across nodes. Never TP across PCIe.
4. **Pre-allocate KV cache**: avoids fragmentation and enables CUDA Graphs.
5. **Use torch.compile**: `reduce-overhead` for latency, `max-autotune` for throughput.
6. **Benchmark all three metrics**: TTFT, ITL, and throughput. Optimizing one can hurt another.
7. **Consider AOTInductor for production**: eliminates Python overhead and GIL contention.
8. **Implement continuous batching**: static batching wastes 30-50% of GPU cycles.
9. **Monitor GPU utilization**: all GPUs should be >80% utilized in steady state.
10. **Test with realistic workloads**: generation length distributions affect throughput significantly.

---

### Further Resources

- [PyTorch Tensor Parallel docs](https://pytorch.org/docs/stable/distributed.tensor.parallel.html) — official TP API reference
- [PyTorch Pipeline Parallel docs](https://pytorch.org/docs/stable/pipeline.html) — official PP API reference
- [vLLM](https://github.com/vllm-project/vllm) — production LLM serving with PagedAttention
- [Module 10 — Distributed Training](../10_distributed/) — DDP, FSDP2, DeviceMesh, TP, PP
- [Module 11 — Export & Deployment](../11_export_deploy/) — torch.export, AOTInductor, NativeRT
- [Module 22 — LLM Recipes](../22_llm_recipes/) — RoPE, KV Cache, GQA, SwiGLU

---

<div align="center">

[← Previous Module](../26_memory_profiling/) | [🏠 Home](../README.md) | [Next Module →](../28_benchmarking/)

**Notebook**: [`27_multi_gpu_inference.ipynb`](../notebooks/27_multi_gpu_inference.ipynb)

</div>
