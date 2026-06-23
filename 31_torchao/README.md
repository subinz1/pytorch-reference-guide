<div align="center">

[← Previous Module (Debugging)](../30_debugging/) | [🏠 Home](../README.md) | Next Module →

</div>

---

# Module 31 — torchao: Architecture Optimization

> **Prerequisites**: [Module 07 — Training](../07_training/), [Module 08 — torch.compile](../08_torch_compile/), [Module 29 — Mixed Precision](../29_mixed_precision/)
>
> **Time**: ~3 hours | **Files**: `quantization_basics.py`, `torchao_workflows.py`

---

## Table of Contents

1. [What is torchao?](#1-what-is-torchao)
2. [torchao vs torch.ao](#2-torchao-vs-torchao)
3. [Installation](#3-installation)
4. [The quantize_() API](#4-the-quantize_-api)
5. [Weight-Only Quantization (INT8/INT4)](#5-weight-only-quantization-int8int4)
6. [Dynamic Quantization (INT8)](#6-dynamic-quantization-int8)
7. [Float8 Training and Inference](#7-float8-training-and-inference)
8. [Semi-Structured Sparsity (2:4)](#8-semi-structured-sparsity-24)
9. [PT2E Quantization Flow](#9-pt2e-quantization-flow)
10. [Integration with torch.compile](#10-integration-with-torchcompile)
11. [Practical: Quantize and Benchmark](#11-practical-quantize-and-benchmark)
12. [Choosing a Quantization Strategy](#12-choosing-a-quantization-strategy)
13. [Upstream Updates (June 2026)](#13-upstream-updates-june-2026)

---

## 1. What is torchao?

**torchao** (PyTorch Architecture Optimization) is a PyTorch-native library for making models faster and smaller through quantization, sparsity, and dtype optimization.

Repository: [github.com/pytorch/ao](https://github.com/pytorch/ao)

### Key features

- **Quantization**: INT8, INT4, FP8 weight-only and dynamic quantization
- **Sparsity**: Semi-structured (2:4) sparsity with hardware acceleration
- **Composability**: Works seamlessly with `torch.compile` for fused, optimized kernels
- **Tensor subclass-based**: Uses PyTorch's tensor subclass system — no graph rewrites needed

### Why torchao matters

Quantization and sparsity can provide:

| Technique | Memory Reduction | Speedup | Use Case |
|-----------|:----------------:|:-------:|----------|
| INT8 weight-only | ~2× | 1.2–2× | Memory-bound inference (LLM serving) |
| INT4 weight-only | ~4× | 2–4× | Extremely memory-bound inference |
| INT8 dynamic | ~2× | 1.5–3× | Compute-bound batch inference |
| FP8 | ~2× | 1.5–2× | Training on H100+ |
| 2:4 sparsity | ~2× | ~2× | Prunable models on Ampere+ |

These are approximate — actual results depend on model architecture, hardware, batch size, and workload characteristics.

---

## 2. torchao vs torch.ao

PyTorch has two quantization stories. Understanding the distinction is critical:

### torch.ao.quantization (Old — Being Deprecated)

```python
# OLD approach — FX-based graph rewriting
import torch.ao.quantization as taq

model_prepared = taq.prepare(model, qconfig_mapping, example_inputs)
# ... calibrate ...
model_quantized = taq.convert(model_prepared)
```

- Lives in-tree at `torch.ao.quantization`
- Uses FX graph tracing (fragile, breaks on dynamic control flow)
- Requires `qconfig_mapping` boilerplate
- **Not composable** with `torch.compile`
- Being deprecated in favor of torchao

### torchao (New — Recommended)

```python
# NEW approach — tensor subclass-based
from torchao import quantize_
from torchao.quantization import int8_weight_only

quantize_(model, int8_weight_only())
# That's it. Model is quantized.
```

- External library at `pytorch/ao`
- Uses **tensor subclasses** — weights become quantized tensor objects
- No graph rewriting — the model structure stays the same
- **Composable** with `torch.compile` (Inductor generates fused kernels)
- Actively developed, production-ready

### Migration path

```
torch.ao.quantization.quantize_dynamic  →  torchao.quantize_(model, int8_dynamic_activation_int8_weight())
torch.ao.quantization.prepare/convert   →  torchao.quantize_(model, int8_weight_only())
torch.ao.quantization (FX)              →  PT2E quantization flow (torch.export + quantize_pt2e)
```

---

## 3. Installation

```bash
pip install torchao
```

Verify:

```python
import torchao
print(torchao.__version__)
print(f"CUDA available: {torchao.utils.TORCH_VERSION_AT_LEAST_2_3}")
```

torchao requires PyTorch 2.3+ and works best with PyTorch 2.6+ for the latest features.

**Note**: torchao quantization kernels are optimized for CUDA. CPU execution works for development/testing but won't show the full performance benefits.

---

## 4. The quantize_() API

The `quantize_()` function is the single entry point for all torchao quantization:

```python
from torchao import quantize_
from torchao.quantization import (
    int8_weight_only,
    int4_weight_only,
    int8_dynamic_activation_int8_weight,
    float8_dynamic_activation_float8_weight,
)

# Weight-only quantization
quantize_(model, int8_weight_only())          # INT8 weights
quantize_(model, int4_weight_only())          # INT4 weights

# Dynamic quantization (weights + activations)
quantize_(model, int8_dynamic_activation_int8_weight())

# Float8 quantization (H100+)
quantize_(model, float8_dynamic_activation_float8_weight())
```

### How it works

1. `quantize_()` walks the model's `nn.Module` tree
2. For each matching module (default: `nn.Linear`), it replaces the weight tensor with a **quantized tensor subclass**
3. The module itself is unchanged — it's still `nn.Linear`
4. When the layer runs, the tensor subclass handles dequantization during matmul
5. With `torch.compile`, Inductor fuses the dequantize + matmul into a single kernel

```python
# Before quantize_():
model.layer.weight  # torch.float16, shape [1024, 512]

quantize_(model, int8_weight_only())

# After quantize_():
model.layer.weight  # AffineQuantizedTensor (int8 storage, float16 dequant)
type(model.layer)   # Still nn.Linear!
```

### Filtering which layers get quantized

```python
# Only quantize layers with > 1024 dimensions
def filter_fn(module, fqn):
    if isinstance(module, torch.nn.Linear):
        return module.in_features >= 1024
    return False

quantize_(model, int8_weight_only(), filter_fn=filter_fn)
```

---

## 5. Weight-Only Quantization (INT8/INT4)

**Weight-only** quantization stores model weights in lower precision (INT8 or INT4) while keeping activations in FP16/BF16. This is the most common approach for memory-bound inference (e.g., LLM serving at batch size 1).

### How quantization works

For a weight tensor `W` in FP16:

```
Quantize:   W_int8 = round(W / scale) + zero_point
Dequantize: W_approx = (W_int8 - zero_point) * scale
```

Where:
- `scale = (max(W) - min(W)) / (2^bits - 1)`
- `zero_point` maps the real zero to an integer value

### INT8 weight-only

```python
from torchao import quantize_
from torchao.quantization import int8_weight_only

quantize_(model, int8_weight_only())

# With group-wise quantization (more accurate, slightly more overhead)
quantize_(model, int8_weight_only(group_size=128))
```

**Per-channel** (default): One scale per output channel. Good accuracy, simple.

**Per-group** (group_size=128 or 32): One scale per group of weights. Better accuracy for large layers, slight overhead for scale storage.

### INT4 weight-only

```python
from torchao.quantization import int4_weight_only

# INT4 always uses group-wise quantization
quantize_(model, int4_weight_only(group_size=128))
quantize_(model, int4_weight_only(group_size=32))   # More accurate, more scales
```

INT4 packs two values per byte → ~4× memory reduction. This is the standard for LLM inference (used by llama.cpp, GPTQ, AWQ, etc.).

### Memory comparison

For a 7B parameter model (all Linear layers):

| Precision | Memory | Relative |
|-----------|:------:|:--------:|
| FP32 | 28 GB | 1.0× |
| FP16/BF16 | 14 GB | 0.5× |
| INT8 | 7 GB | 0.25× |
| INT4 | 3.5 GB | 0.125× |

---

## 6. Dynamic Quantization (INT8)

Dynamic quantization quantizes **both** weights and activations. Weights are quantized ahead of time; activations are quantized on-the-fly during each forward pass.

```python
from torchao import quantize_
from torchao.quantization import int8_dynamic_activation_int8_weight

quantize_(model, int8_dynamic_activation_int8_weight())
```

### When to use dynamic quantization

- **Compute-bound** workloads: Batch inference (batch_size > 1)
- INT8 matrix multiply (GEMM) is ~2× faster than FP16 on most hardware
- The activation quantization overhead is amortized over larger batches

### How it differs from weight-only

| | Weight-Only | Dynamic |
|---|---|---|
| Weights | INT8/INT4 | INT8 |
| Activations | FP16/BF16 | INT8 (computed at runtime) |
| Matmul precision | FP16 | INT8 |
| Best for | Memory-bound (batch=1) | Compute-bound (batch>1) |
| Accuracy impact | Lower | Slightly higher |
| Extra overhead | None | Per-batch activation quantization |

### Calibration-free

Unlike static quantization, dynamic quantization doesn't need calibration data — activation ranges are computed per-batch. This makes deployment simpler.

---

## 7. Float8 Training and Inference

Float8 (FP8) uses 8-bit floating-point formats for both training and inference. Unlike INT8, FP8 preserves the floating-point dynamic range.

### FP8 formats

- **E4M3** (4 exponent, 3 mantissa): Higher precision, used for weights and activations
- **E5M2** (5 exponent, 2 mantissa): Higher range, used for gradients during training

### FP8 inference with torchao

```python
from torchao import quantize_
from torchao.quantization import float8_dynamic_activation_float8_weight

# Requires H100 or newer (Hopper architecture)
quantize_(model, float8_dynamic_activation_float8_weight())
```

### FP8 training with Float8Linear

For training, torchao provides `Float8Linear` which replaces `nn.Linear` layers:

```python
from torchao.float8 import Float8LinearConfig, convert_to_float8_training

config = Float8LinearConfig()
convert_to_float8_training(model, config=config)

# Now train normally — forward/backward use FP8 matmuls
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

### Scaling strategies

FP8 has limited range, so **scaling** is critical:

- **Per-tensor scaling** (default): One scale factor per tensor. Simple but less accurate.
- **Per-row scaling**: One scale factor per row/column. Better accuracy, used in production.
- **Delayed scaling**: Use statistics from previous iteration to scale current one. Reduces overhead.

### Hardware requirements

| Feature | Minimum GPU |
|---------|-------------|
| FP8 inference | H100, L40S, MI300 |
| FP8 training | H100, MI300 |
| FP8 with per-row scaling | H100 |

---

## 8. Semi-Structured Sparsity (2:4)

NVIDIA Ampere and newer GPUs have hardware support for **semi-structured sparsity**: exactly 2 out of every 4 consecutive values must be zero (the "2:4" pattern).

```
Dense:   [1.2, 0.5, 3.1, 0.8, 2.0, 1.1, 0.3, 4.2]
2:4 :    [1.2, 0.0, 3.1, 0.0, 2.0, 0.0, 0.3, 4.2]
                ↑         ↑         ↑              
           zeroed out  zeroed   zeroed
```

The hardware stores only the non-zero values + a 2-bit index per group of 4, achieving ~2× compression and ~2× matmul speedup.

### Applying 2:4 sparsity with torchao

```python
from torchao.sparsity import sparsify_
from torchao.sparsity import semi_structured_sparsify

# Apply 2:4 sparsity to model weights
sparsify_(model, semi_structured_sparsify())
```

### The sparsification process

1. For each group of 4 consecutive values, keep the 2 with largest magnitude
2. Zero out the other 2
3. Repack into the hardware sparse format
4. CUDA sparse matmul kernel handles the rest

### Combining sparsity and quantization

You can combine 2:4 sparsity with quantization for compounding benefits:

```python
# First quantize, then sparsify
quantize_(model, int8_weight_only())
sparsify_(model, semi_structured_sparsify())

# Theoretical: 2× (quantization) × 2× (sparsity) = 4× improvement
```

### Considerations

- **Accuracy**: Forcing 50% of weights to zero degrades accuracy. Models may need fine-tuning.
- **Hardware**: Requires NVIDIA A100 or newer.
- **Not all layers benefit**: Small layers or layers already memory-bound may not see speedup.
- **Training**: For best results, use sparsity-aware training (gradually introduce the sparsity pattern during training).

---

## 9. PT2E Quantization Flow

PT2E (PyTorch 2 Export) quantization is the export-based quantization flow. It uses `torch.export` to capture the model graph, then applies quantization transformations. This is the path for deploying quantized models to specific hardware backends.

### The PT2E pipeline

```
     Model
       │
       ▼
  torch.export()         ← Capture to ExportedProgram
       │
       ▼
  prepare_pt2e()         ← Insert observers for calibration
       │
       ▼
  Calibrate (run data)   ← Collect activation statistics
       │
       ▼
  convert_pt2e()         ← Replace observers with quantize/dequantize ops
       │
       ▼
  Quantized Model        ← Ready for backend-specific compilation
```

### Code walkthrough

```python
import torch
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)

# 1. Export the model
exported = torch.export.export(model, example_inputs)

# 2. Create a backend-specific quantizer
quantizer = XNNPACKQuantizer().set_global(
    get_symmetric_quantization_config()
)

# 3. Prepare for calibration
prepared = prepare_pt2e(exported, quantizer)

# 4. Calibrate with representative data
with torch.no_grad():
    for batch in calibration_loader:
        prepared(batch)

# 5. Convert to quantized model
quantized = convert_pt2e(prepared)
```

### Available backend quantizers

| Quantizer | Target Hardware | Typical Use |
|-----------|----------------|-------------|
| `XNNPACKQuantizer` | ARM CPU (mobile) | Android/iOS inference |
| `X86InductorQuantizer` | x86 CPU | Server-side CPU inference |
| `QNNPackQuantizer` | ARM CPU | Legacy mobile path |

### When to use PT2E vs `quantize_()`

| | `quantize_()` (torchao) | PT2E |
|---|---|---|
| Ease of use | One-liner | Multi-step pipeline |
| Needs calibration | No (weight-only/dynamic) | Yes (static quant) |
| Backend-specific | No (generic) | Yes (XNNPack, x86, etc.) |
| Works with compile | Yes (primary use case) | Yes (through export) |
| Best for | GPU inference, LLM serving | Mobile/edge deployment |

---

## 10. Integration with torch.compile

The key advantage of torchao over the old torch.ao is **native composability with torch.compile**. When you compile a torchao-quantized model, Inductor generates fused kernels that combine dequantization with the actual computation.

### Basic workflow

```python
import torch
from torchao import quantize_
from torchao.quantization import int8_weight_only

# Step 1: Quantize
quantize_(model, int8_weight_only())

# Step 2: Compile
model = torch.compile(model, mode="max-autotune")

# Step 3: Run (first call triggers compilation)
output = model(input_tensor)
```

### What happens under the hood

Without `torch.compile`:
```
input → dequantize(int8_weight → fp16) → matmul(input, fp16_weight) → output
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
         Two separate operations, intermediate fp16 tensor allocated
```

With `torch.compile`:
```
input → fused_int8_matmul(input, int8_weight, scale) → output
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
         Single fused kernel, no intermediate allocation
```

The fusion eliminates:
- Memory allocation for the dequantized weight
- Memory bandwidth for reading/writing the intermediate tensor
- Kernel launch overhead for the separate dequantize op

### Performance impact

A compiled quantized model is typically faster than either quantization or compilation alone:

```
Baseline (FP16, eager):        1.0×
FP16 + torch.compile:          1.5×
INT8 quantized (eager):        1.3×
INT8 quantized + compile:      2.5×  ← Multiplicative gains
```

### Best practices

1. **Quantize before compile**: `quantize_()` first, then `torch.compile()`
2. **Use `mode="max-autotune"`** for best performance (longer compilation time)
3. **Warm up**: First forward pass triggers compilation. Time the second pass onward.
4. **Dynamic shapes**: Use `torch.compile(dynamic=True)` if input shapes vary

---

## 11. Practical: Quantize and Benchmark

A complete workflow for quantizing and evaluating a model:

```python
import torch
import torch.nn as nn
import time

class SimpleMLP(nn.Module):
    def __init__(self, dim=4096, hidden=11008):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))

def measure_model_size(model):
    """Measure total parameter memory in MB."""
    total = sum(
        p.nelement() * p.element_size() for p in model.parameters()
    )
    return total / 1024 / 1024

def benchmark_inference(model, input_tensor, warmup=10, runs=100):
    """Benchmark inference latency."""
    for _ in range(warmup):
        model(input_tensor)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(runs):
        model(input_tensor)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / runs
    return elapsed * 1000  # ms

# Create model
model = SimpleMLP().half().cuda()  # FP16 on GPU
x = torch.randn(1, 4096, dtype=torch.float16, device="cuda")

print(f"Baseline size: {measure_model_size(model):.1f} MB")
print(f"Baseline latency: {benchmark_inference(model, x):.2f} ms")

# Quantize
from torchao import quantize_
from torchao.quantization import int8_weight_only

quantize_(model, int8_weight_only())
print(f"INT8 size: {measure_model_size(model):.1f} MB")
print(f"INT8 latency: {benchmark_inference(model, x):.2f} ms")

# Compile for maximum performance
model = torch.compile(model, mode="max-autotune")
print(f"INT8+compile latency: {benchmark_inference(model, x):.2f} ms")
```

---

## 12. Choosing a Quantization Strategy

Use this decision tree to pick the right method for your workload:

```
                        What's your goal?
                              │
                 ┌────────────┴────────────┐
                 ▼                          ▼
            Inference                   Training
                 │                          │
       ┌─────────┴──────────┐         ┌────┴────┐
       ▼                    ▼         ▼         ▼
   Memory-bound?      Compute-bound?  BF16     FP8
   (batch=1, LLMs)    (batch>1)     (default) (H100+)
       │                    │
   ┌───┴───┐              INT8
   ▼       ▼             dynamic
  INT4   INT8
  (max    (good
  savings) balance)

  Also consider:
  ├─ Maximum speed on H100 → FP8
  ├─ Sparse model → 2:4 sparsity + quantization
  └─ Mobile/edge deployment → PT2E + XNNPack quantizer
```

### Quick reference

| Scenario | Method | Memory | Speedup | Accuracy |
|----------|--------|:------:|:-------:|:--------:|
| LLM serving (batch=1) | `int4_weight_only(group_size=128)` | 4× less | 2–4× | Good |
| LLM serving (batch=1, quality) | `int8_weight_only()` | 2× less | 1.5–2× | Very good |
| Batch inference (batch>8) | `int8_dynamic_activation_int8_weight()` | 2× less | 2–3× | Good |
| H100 inference | `float8_dynamic_activation_float8_weight()` | 2× less | 1.5–2× | Excellent |
| H100 training (large models) | Float8Linear | 2× less | 1.3–1.5× | Excellent |
| Mobile deployment | PT2E + XNNPack | 4× less | 2–4× | Good |
| Prunable model | 2:4 sparsity | 2× less | ~2× | Varies |

### Accuracy considerations

Quantization always trades some accuracy for efficiency. Guidelines:

1. **INT8 weight-only**: Usually <0.1% accuracy loss. Safe for most models.
2. **INT4 weight-only (group_size=128)**: ~0.5–1% loss. Test on your task.
3. **INT4 weight-only (group_size=32)**: ~0.2–0.5% loss. Better accuracy, more scales.
4. **Dynamic INT8**: ~0.1–0.5% loss. Depends on activation distribution.
5. **FP8**: Negligible loss. Closest to original precision.
6. **2:4 sparsity**: 1–3% loss without fine-tuning. Fine-tuning recovers most of it.

### Quantization-Aware Training (QAT)

When post-training quantization isn't accurate enough, use QAT to simulate quantization during training:

```python
from torchao.quantization import int8_weight_only

# During training, quantization is simulated (fake quantize)
# The model learns to be robust to quantization noise
# After training, apply real quantization with quantize_()
```

---

## 13. Upstream Updates (June 2026)

Recent PyTorch development highlights relevant to architecture optimization and production deployment:

### Gloo fault tolerance support ([#187381](https://github.com/pytorch/pytorch/pull/187381))
The Gloo collective communication backend now supports fault-tolerant operation, enabling better recovery from node failures in distributed training. This complements torchao's distributed quantization workflows where training nodes may need to recover gracefully.

### CUDAGraph execution state exposed to Python ([#187740](https://github.com/pytorch/pytorch/pull/187740))
CUDA Graph execution state is now accessible from Python, enabling better integration between CUDA Graphs and quantized model serving. This is relevant for torchao users who combine `torch.compile(mode="reduce-overhead")` with quantized models for maximum inference throughput.

### NativeRT selectScalarOverload fix ([#187059](https://github.com/pytorch/pytorch/pull/187059))
Fix for scalar overload selection in NativeRT, the C++ inference engine for exported models. This improves reliability when deploying PT2E-quantized models through the export → NativeRT path.

### TorchElastic signal-failure enrichment ([#187098](https://github.com/pytorch/pytorch/pull/187098))
TorchElastic now provides richer signal information on training failures. When running large-scale FP8 training jobs with torchao's Float8Linear, better failure diagnostics help identify whether crashes are caused by numerical issues (FP8 overflow) vs. infrastructure problems.

### MPS bucket large allocations for decode ([#187441](https://github.com/pytorch/pytorch/pull/187441))
Memory allocation improvements for Apple MPS (Metal Performance Shaders) backend during decode operations. While torchao's primary optimization targets are CUDA, this improves the experience for development and testing on Apple Silicon.

### Dynamo symbolic range propagation ([#187350](https://github.com/pytorch/pytorch/pull/187350))
Improved symbolic shape analysis in Dynamo helps `torch.compile` generate better code for quantized models with dynamic shapes. torchao's tensor subclasses benefit from more precise shape tracking during compilation.

### XPU device info in Inductor ([#187308](https://github.com/pytorch/pytorch/pull/187308))
Inductor now has access to XPU (Intel GPU) device information, enabling better code generation for Intel hardware. This lays groundwork for torchao quantization support on Intel discrete GPUs.

---

## Summary

### Core concepts

| Concept | Description |
|---------|-------------|
| **Quantization** | Reducing numerical precision (FP16→INT8/INT4) to save memory and increase speed |
| **Weight-only** | Only weights are quantized; activations stay in higher precision |
| **Dynamic** | Both weights and activations are quantized; activation scales computed at runtime |
| **Tensor subclass** | torchao's approach: quantized weights are special tensor objects that handle dequant transparently |
| **Semi-structured sparsity** | 2:4 pattern — hardware-accelerated on NVIDIA Ampere+ |
| **PT2E** | Export-based quantization for backend-specific deployment |

### The three-step workflow

```python
# 1. Choose your method
from torchao.quantization import int8_weight_only

# 2. Quantize in-place
from torchao import quantize_
quantize_(model, int8_weight_only())

# 3. Compile for maximum performance
model = torch.compile(model, mode="max-autotune")
```

### Common pitfalls

1. **Forgetting to compile**: torchao without `torch.compile` works but misses the fused-kernel speedup
2. **Wrong method for workload**: INT4 weight-only for batch inference (should use dynamic INT8)
3. **Expecting GPU speedups on CPU**: torchao kernels are optimized for CUDA
4. **Not warming up**: First inference call triggers compilation — benchmark subsequent calls
5. **Quantizing tiny models**: Overhead of quantization may exceed savings for small models

---

### Further Resources

- [torchao GitHub](https://github.com/pytorch/ao) — source code and documentation
- [torchao tutorials](https://pytorch.org/ao/stable/) — official torchao documentation
- [PyTorch Quantization Docs](https://pytorch.org/docs/stable/quantization.html) — quantization overview
- [Module 07 — Training Pipelines](../07_training/) — training fundamentals
- [Module 08 — torch.compile](../08_torch_compile/) — compilation deep dive
- [Module 29 — Mixed Precision](../29_mixed_precision/) — FP16, BF16, FP8 precision

---

<div align="center">

[← Previous Module (Debugging)](../30_debugging/) | [🏠 Home](../README.md) | Next Module →

**Notebook**: [`31_torchao.ipynb`](../notebooks/31_torchao.ipynb)

</div>
