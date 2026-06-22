<div align="center">

[← Previous Module](../28_benchmarking/) | [🏠 Home](../README.md) | [Next Module (Debugging) →](../30_debugging/)

</div>

---

> **Module 29** of the PyTorch Complete Learning Guide
> **Prerequisites:** [Module 07 — Training Pipelines](../07_training/), [Module 08 — torch.compile](../08_torch_compile/), [Module 20 — Backends Tuning](../20_backends_tuning/)
> **Time to complete:** ~3 hours

### 📁 Files in This Module

| File | Description |
|------|-------------|
| [`README.md`](README.md) | This guide — numerical formats, AMP, GradScaler, BF16, FP8, FSDP2 mixed precision |
| [`precision_formats.py`](precision_formats.py) | Dtype exploration, range/precision, memory comparison, conversion errors |
| [`mixed_precision_training.py`](mixed_precision_training.py) | AMP training loops, GradScaler, BF16 vs FP16 comparison, torch.compile integration |

---

# Mixed Precision Deep Dive — FP32, FP16, BF16, and FP8

## Table of Contents

1. [Numerical Formats Overview](#1-numerical-formats-overview)
2. [Why Mixed Precision?](#2-why-mixed-precision)
3. [AMP: Automatic Mixed Precision](#3-amp-automatic-mixed-precision)
4. [GradScaler](#4-gradscaler)
5. [BF16 vs FP16](#5-bf16-vs-fp16)
6. [FP8 Training](#6-fp8-training)
7. [Loss Scaling Deep Dive](#7-loss-scaling-deep-dive)
8. [Mixed Precision with torch.compile](#8-mixed-precision-with-torchcompile)
9. [Mixed Precision with FSDP2](#9-mixed-precision-with-fsdp2)
10. [Numerical Stability Checklist](#10-numerical-stability-checklist)
11. [Precision-Performance Tradeoffs](#11-precision-performance-tradeoffs)
12. [Upstream Updates (June 2026)](#12-upstream-updates-june-2026)

---

## 1. Numerical Formats Overview

Every floating-point number is represented as: `(-1)^sign × 2^exponent × (1 + mantissa)`.
The tradeoff is fundamental: **more exponent bits = larger representable range**, **more mantissa bits = more decimal precision**.

### IEEE 754 and PyTorch Formats

| Format | Bits | Exponent | Mantissa | Range | Precision | PyTorch dtype |
|--------|------|----------|----------|-------|-----------|---------------|
| FP32 | 32 | 8 | 23 | ±3.4e38 | High | `torch.float32` |
| TF32 | 19 | 8 | 10 | ±3.4e38 | Medium | (internal) |
| BF16 | 16 | 8 | 7 | ±3.4e38 | Low | `torch.bfloat16` |
| FP16 | 16 | 5 | 10 | ±65504 | Medium-low | `torch.float16` |
| FP8 E4M3 | 8 | 4 | 3 | ±448 | Very low | `torch.float8_e4m3fn` |
| FP8 E5M2 | 8 | 5 | 2 | ±57344 | Lowest | `torch.float8_e5m2` |

### Key Observations

**FP32** — The default. 8 exponent bits give a huge dynamic range (±3.4×10^38), and 23 mantissa bits give ~7 decimal digits of precision. This is more than enough for any training scenario but uses 4 bytes per parameter.

**TF32** — NVIDIA's "Tensor Float 32". Same range as FP32 (8 exponent bits) but only 10 mantissa bits. Not a user-facing dtype — it's an internal hardware format used by tensor cores when `torch.backends.cuda.matmul.allow_tf32 = True`. Gives near-FP32 accuracy with FP16-like throughput for matmuls.

**BF16 (Brain Float 16)** — Google's format. Same 8-bit exponent as FP32 (same range!) but only 7 mantissa bits (~2 decimal digits of precision). The key insight: for neural network training, range matters more than precision. Developed at Google Brain for TPU training.

**FP16 (Half)** — IEEE half-precision. Only 5 exponent bits means range is limited to ±65504. Values larger than this overflow to infinity. However, 10 mantissa bits give more precision than BF16. The limited range is the reason GradScaler exists.

**FP8 E4M3** — 4 exponent bits, 3 mantissa bits. Range up to ±448. Designed for the forward pass where more precision helps (activations stay in a moderate range after normalization).

**FP8 E5M2** — 5 exponent bits, 2 mantissa bits. Range up to ±57344. Designed for the backward pass where gradients can vary wildly in magnitude (larger range handles this better).

### Memory Implications

```
Parameters:  100M model
FP32:        400 MB (4 bytes × 100M)
FP16/BF16:   200 MB (2 bytes × 100M)  — 2× savings
FP8:         100 MB (1 byte × 100M)   — 4× savings
```

---

## 2. Why Mixed Precision?

Mixed precision means using **lower-precision formats for computation** while keeping **FP32 master copies** for numerical stability. The benefits:

### Memory Savings

| Component | FP32 | Mixed (BF16 compute) | Savings |
|-----------|------|----------------------|---------|
| Model parameters (compute copy) | 4B/param | 2B/param | 2× |
| Activations | 4B/element | 2B/element | 2× |
| Gradients | 4B/param | 2B/param | 2× |
| Optimizer states (Adam) | 8B/param | 8B/param | 1× (kept in FP32) |
| Master weights | — | 4B/param | Overhead |

For a 1B parameter model with Adam:
- **FP32 only**: 4 + 4 + 8 = 16 GB (params + grads + optimizer)
- **Mixed precision**: 2 + 2 + 8 + 4 = 16 GB (BF16 params + BF16 grads + FP32 optimizer + FP32 master)

The memory win comes from **activations** (which scale with batch size and sequence length) and from not needing FP32 gradients during backward:

```
Activation memory for a Transformer layer (seq_len=2048, hidden=4096, batch=8):
FP32: ~1.6 GB per layer
BF16: ~0.8 GB per layer
```

### Throughput Gains

NVIDIA Tensor Cores operate on lower-precision types:
- **FP16/BF16**: 2-3× throughput vs FP32 on A100/H100 (matmul and convolution)
- **FP8**: 2× throughput vs BF16 on H100 (matmul only)
- **TF32**: ~2× throughput vs FP32 (enabled by default on Ampere+)

The key: these speedups apply to **tensor core operations** (matmul, conv). Elementwise ops, reductions, and memory-bound ops see less benefit.

### Accuracy Impact

With proper techniques (loss scaling for FP16, FP32 accumulation), mixed precision training converges to the same accuracy as FP32 for virtually all workloads. The neural network optimization landscape is robust to reduced precision because:

1. Gradient noise from mini-batching already exceeds quantization noise
2. Normalization layers keep activations in representable ranges
3. FP32 master weights accumulate small updates that would be lost in FP16

---

## 3. AMP: Automatic Mixed Precision

The modern PyTorch AMP API uses `torch.amp.autocast` to automatically cast operations to the appropriate precision.

### Basic Usage

```python
import torch
import torch.nn as nn

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for data, target in dataloader:
    data, target = data.cuda(), target.cuda()
    optimizer.zero_grad()

    # autocast region: eligible ops run in float16
    with torch.amp.autocast('cuda', dtype=torch.float16):
        output = model(data)
        loss = criterion(output, target)

    # backward and step happen outside autocast
    loss.backward()
    optimizer.step()
```

### What autocast Does

autocast maintains **two lists** of operations:

**Cast to FP16/BF16 (compute-intensive, benefit from tensor cores):**
- `torch.mm`, `torch.matmul`, `torch.bmm`
- `torch.nn.functional.linear`
- `torch.nn.functional.conv1d/2d/3d`
- `torch.baddbmm`

**Keep in FP32 (numerically sensitive):**
- `torch.nn.functional.softmax`
- `torch.nn.functional.cross_entropy`, all loss functions
- `torch.nn.functional.layer_norm`, `batch_norm`, `group_norm`
- `torch.sum`, `torch.mean` (reductions)
- `torch.exp`, `torch.log`, `torch.pow`

**Rules for mixed inputs:**
- If any input is FP32 and the op is NOT in the cast-down list, it stays FP32
- If inputs are mixed (FP16 + FP32), they get promoted to the wider type
- autocast only affects CUDA ops (CPU autocast exists but has limited support)

### autocast Nesting and Disabling

```python
# Nested autocast — inner region can override dtype
with torch.amp.autocast('cuda', dtype=torch.float16):
    # FP16 region
    y = model.encoder(x)

    with torch.amp.autocast('cuda', enabled=False):
        # Force FP32 for this subcomputation
        y_float = y.float()
        sensitive_result = custom_numerics(y_float)

    z = model.decoder(sensitive_result.half())
```

### CPU autocast

```python
# Limited CPU autocast (mainly for BF16 on Intel CPUs with AMX)
with torch.amp.autocast('cpu', dtype=torch.bfloat16):
    output = model(data)
```

---

## 4. GradScaler

### The Problem: Gradient Underflow in FP16

FP16 has a minimum positive subnormal of ~5.96×10^-8. Gradients in deep networks routinely have magnitudes of 10^-5 to 10^-8 — right at the edge of FP16 representability. Small gradients underflow to zero, and the model stops learning.

**BF16 does NOT have this problem** because it shares FP32's exponent range. GradScaler is only needed for FP16 training.

### How GradScaler Works

```
Forward:   compute loss normally
Scale:     loss_scaled = loss × scale_factor (e.g., 65536)
Backward:  gradients are also scaled by scale_factor (chain rule)
           → small gradients become representable in FP16
Unscale:   divide gradients by scale_factor before optimizer step
Check:     if any gradient is inf/nan, skip optimizer step
Update:    adjust scale_factor dynamically
```

### Complete Training Loop with GradScaler

```python
import torch
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_

model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = GradScaler('cuda')

for epoch in range(num_epochs):
    for data, target in dataloader:
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()

        with autocast('cuda', dtype=torch.float16):
            output = model(data)
            loss = criterion(output, target)

        # Scale loss and call backward
        scaler.scale(loss).backward()

        # Unscale gradients for clipping
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Step (skips if inf/nan detected)
        scaler.step(optimizer)
        scaler.update()
```

### GradScaler Internals

```python
scaler = GradScaler(
    device='cuda',
    init_scale=2**16,        # Initial scale factor (65536)
    growth_factor=2.0,       # Multiply scale by this after growth_interval
    backoff_factor=0.5,      # Multiply scale by this on inf/nan
    growth_interval=2000,    # Steps between scale increases
)
```

The dynamic scaling algorithm:
1. Start with `init_scale` (default 65536)
2. If `growth_interval` consecutive steps have no inf/nan → multiply scale by `growth_factor`
3. If any step produces inf/nan → multiply scale by `backoff_factor`, skip that step
4. This finds the largest scale that doesn't overflow

---

## 5. BF16 vs FP16

### Why BF16 Is Preferred for LLMs

| Property | FP16 | BF16 |
|----------|------|------|
| Max value | 65504 | 3.4×10^38 |
| Min positive normal | 6.1×10^-5 | 1.2×10^-38 |
| Precision (decimal digits) | ~3.3 | ~2.1 |
| GradScaler needed | Yes | No |
| Overflow risk | High | None (same as FP32) |
| Hardware support | All GPUs with tensor cores | Ampere+ (A100, H100, RTX 3090+) |

### The Overflow Problem

```python
import torch

# FP16 overflow — values > 65504 become inf
x = torch.tensor(70000.0, dtype=torch.float16)
print(x)  # tensor(inf, dtype=torch.float16)

# BF16 handles it — same exponent range as FP32
x = torch.tensor(70000.0, dtype=torch.bfloat16)
print(x)  # tensor(70000., dtype=torch.bfloat16)
```

In LLM training, logits before softmax can easily exceed 65504 (especially early in training with random initialization). BF16 handles this naturally; FP16 would produce inf and NaN gradients.

### When to Use Each

**Use BF16 when:**
- Training LLMs or large models
- You have Ampere+ hardware (A100, H100, RTX 30/40 series)
- You want simplicity (no GradScaler)
- You don't need high precision for inference

**Use FP16 when:**
- Deploying on older GPUs (V100, T4) that lack BF16 tensor cores
- Running inference where overflow isn't a concern (inputs are bounded)
- Accuracy is critical and you can handle GradScaler complexity

### BF16 Training Loop (No Scaler)

```python
model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for data, target in dataloader:
    data, target = data.cuda(), target.cuda()
    optimizer.zero_grad()

    # BF16 autocast — no scaler needed!
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        output = model(data)
        loss = criterion(output, target)

    loss.backward()
    optimizer.step()
```

---

## 6. FP8 Training

FP8 is the cutting edge of low-precision training, offering 2× throughput over BF16 on H100 GPUs.

### Two Complementary Formats

**E4M3 (4 exponent, 3 mantissa):**
- Range: ±448
- Used for the **forward pass** (activations are bounded after normalization)
- More mantissa bits → better precision for weight × activation products

**E5M2 (5 exponent, 2 mantissa):**
- Range: ±57344
- Used for the **backward pass** (gradients span many orders of magnitude)
- More exponent bits → handles gradient dynamic range

### FP8 Tensors in PyTorch

```python
import torch

# Create FP8 tensors (requires explicit casting)
x_fp32 = torch.randn(4, 4)
x_e4m3 = x_fp32.to(torch.float8_e4m3fn)
x_e5m2 = x_fp32.to(torch.float8_e5m2)

print(x_e4m3.dtype)  # torch.float8_e4m3fn
print(x_e5m2.dtype)  # torch.float8_e5m2
```

### Scaled FP8 Matmul

Because FP8 has very limited range, **scaling** is required to keep values representable. PyTorch provides `torch._scaled_mm` for this:

```python
# FP8 scaled matrix multiplication
# a: [M, K] in float8_e4m3fn
# b: [K, N] in float8_e4m3fn (transposed)
# scale_a, scale_b: scalar tensors

a_fp32 = torch.randn(64, 128, device='cuda')
b_fp32 = torch.randn(256, 128, device='cuda')  # will be transposed

# Compute scales: scale = max_representable / absmax(tensor)
scale_a = torch.tensor(448.0 / a_fp32.abs().max(), device='cuda')
scale_b = torch.tensor(448.0 / b_fp32.abs().max(), device='cuda')

# Quantize to FP8
a_fp8 = (a_fp32 * scale_a).to(torch.float8_e4m3fn)
b_fp8 = (b_fp32 * scale_b).to(torch.float8_e4m3fn)

# Scaled matmul: result = (a_fp8 @ b_fp8.T) / (scale_a * scale_b)
result = torch._scaled_mm(
    a_fp8, b_fp8.t(),
    scale_a=scale_a.reciprocal(),
    scale_b=scale_b.reciprocal(),
    out_dtype=torch.bfloat16
)
```

### Per-Tensor vs Block Scaling

**Per-tensor scaling** (shown above):
- One scale factor for the entire tensor
- Simple but lossy if values have high dynamic range within a tensor
- Used in early FP8 implementations

**Block scaling (MX format):**
- One scale factor per block (e.g., 32 or 128 elements)
- Finer-grained: handles intra-tensor dynamic range better
- Hardware support on H100+ with newer CUDA versions

```python
# Conceptual block scaling
block_size = 128
scales = []
for i in range(0, tensor.numel(), block_size):
    block = tensor[i:i+block_size]
    scale = 448.0 / block.abs().max()
    scales.append(scale)
```

### When to Use FP8

- **Hardware**: H100 or newer (Ada Lovelace for inference only)
- **Model size**: Benefits increase with larger matmuls (≥1024 dimensions)
- **Use case**: LLM pretraining at scale, where 2× throughput over BF16 justifies complexity
- **Maturity**: Still evolving — API may change between PyTorch versions

---

## 7. Loss Scaling Deep Dive

### Dynamic vs Static Scaling

**Dynamic scaling** (default GradScaler behavior):
- Automatically finds the right scale
- Adapts to different training phases (early training may need different scale than fine-tuning)
- Costs: occasional wasted steps when scale is too high

**Static scaling** (manual):
```python
scaler = GradScaler('cuda', init_scale=1024, growth_interval=float('inf'))
# Scale stays at 1024 forever — no growth, but still backs off on inf/nan
```

When to use static: when you know the gradient magnitude distribution won't change (e.g., fine-tuning a pre-trained model with frozen layers).

### Scale Factor Dynamics

```
Training starts: scale = 65536
Step 1-2000: no overflow → scale grows to 131072
Step 2001: overflow detected → scale drops to 65536, step skipped
Step 2002-4001: no overflow → scale grows to 131072
...eventually finds stable maximum scale
```

### Handling inf/nan

When GradScaler detects overflow:
1. **Skip the optimizer step** — corrupted gradients would harm the model
2. **Reduce scale** — multiply by `backoff_factor` (default 0.5)
3. **Zero the gradients** — they're invalid
4. **Continue training** — next step uses the reduced scale

```python
# Monitoring scale factor during training
for step, (data, target) in enumerate(dataloader):
    # ... training step ...
    if step % 100 == 0:
        print(f"Step {step}: scale = {scaler.get_scale():.0f}")
```

### Common Patterns

```python
# Pattern: gradient clipping with GradScaler
scaler.scale(loss).backward()
scaler.unscale_(optimizer)  # MUST unscale before clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()

# Pattern: multiple losses
with autocast('cuda', dtype=torch.float16):
    loss1 = criterion1(output1, target1)
    loss2 = criterion2(output2, target2)
    loss = loss1 + 0.5 * loss2

scaler.scale(loss).backward()  # Single backward for combined loss

# Pattern: gradient accumulation
for i, (data, target) in enumerate(dataloader):
    with autocast('cuda', dtype=torch.float16):
        output = model(data)
        loss = criterion(output, target) / accumulation_steps

    scaler.scale(loss).backward()

    if (i + 1) % accumulation_steps == 0:
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

---

## 8. Mixed Precision with torch.compile

`torch.amp.autocast` composes cleanly with `torch.compile`. The compiler traces through autocast regions and can further optimize precision handling.

### Basic Composition

```python
model = MyModel().cuda()
compiled_model = torch.compile(model)

with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    output = compiled_model(data)  # compile traces through autocast
```

### What torch.compile Does with Precision

1. **Fuses cast operations** — instead of casting per-op, fuses multiple casts into one kernel
2. **Eliminates redundant casts** — if an op chain stays in one dtype, no cast needed
3. **Optimizes accumulation** — ensures FP32 accumulators where needed (e.g., large reductions)
4. **Pattern-matches precision** — recognizes patterns like "cast → matmul → cast back" and uses tensor core instructions directly

### torch.compile + GradScaler

```python
model = MyModel().cuda()
compiled_model = torch.compile(model, mode='max-autotune')
scaler = GradScaler('cuda')

for data, target in dataloader:
    optimizer.zero_grad()

    with autocast('cuda', dtype=torch.float16):
        output = compiled_model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

### set_float32_matmul_precision

This interacts with torch.compile by controlling TF32 usage:

```python
# 'highest' — pure FP32 matmul (slowest, most precise)
# 'high'    — TF32 for internal compute (default on Ampere+)
# 'medium'  — reduced precision (BF16 accumulation for large matmuls)
torch.set_float32_matmul_precision('high')

compiled = torch.compile(model)
# Now matmuls inside compiled regions use TF32 for internal compute
```

---

## 9. Mixed Precision with FSDP2

FSDP2 (`fully_shard`) has first-class mixed precision support through `MixedPrecisionPolicy`.

### MixedPrecisionPolicy

```python
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

# Define precision policy
mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,    # Parameters stored/computed in BF16
    reduce_dtype=torch.float32,    # All-reduce in FP32 for stability
)

# Apply to model
for layer in model.layers:
    fully_shard(layer, mp_policy=mp_policy)
fully_shard(model, mp_policy=mp_policy)
```

### What Each Setting Controls

| Setting | Effect | Recommendation |
|---------|--------|----------------|
| `param_dtype` | Cast parameters to this dtype for forward/backward | `torch.bfloat16` |
| `reduce_dtype` | Dtype for gradient all-reduce communication | `torch.float32` for stability |

### Why FP32 Reduce?

When averaging gradients across GPUs, small gradients can lose significant bits in FP16/BF16 addition. FP32 reduce ensures that:
1. Small gradient contributions from each worker aren't lost
2. The final averaged gradient is as accurate as possible
3. Master weight updates (which accumulate many small deltas) stay precise

### Full FSDP2 Mixed Precision Example

```python
import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

def train_fsdp_mixed_precision():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    model = LargeModel().to(device)

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )

    # Shard with mixed precision
    for block in model.transformer_blocks:
        fully_shard(block, mp_policy=mp_policy)
    fully_shard(model, mp_policy=mp_policy)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for data, target in dataloader:
        optimizer.zero_grad()
        # No autocast needed — FSDP handles casting via mp_policy
        output = model(data.to(device))
        loss = criterion(output, target.to(device))
        loss.backward()
        optimizer.step()
```

---

## 10. Numerical Stability Checklist

Common issues and solutions when training with mixed precision:

### Gradient Underflow (FP16 only)

**Symptom:** Loss stops decreasing, gradients are all zero.
**Diagnosis:** Check `(grad == 0).float().mean()` — if >50%, you have underflow.
**Solutions:**
1. Use GradScaler (standard fix)
2. Switch to BF16 (eliminates the problem)
3. Increase learning rate (larger gradients)

### Loss Explosion / NaN

**Symptom:** Loss suddenly becomes inf or NaN.
**Diagnosis:** Check for FP16 overflow in logits or intermediate values.
**Solutions:**
1. Switch to BF16 (larger range)
2. Add gradient clipping: `clip_grad_norm_(model.parameters(), 1.0)`
3. Reduce learning rate
4. Check for numerical instability in custom layers

### NaN in Softmax

**Symptom:** NaN output from attention or classification head.
**Root cause:** Large logit values overflow in `exp()` during softmax.
**Solutions:**
1. Keep softmax in FP32 (autocast does this automatically)
2. Use numerically stable softmax: `softmax(x - x.max())`
3. If using custom attention, ensure FP32 for the softmax step

### Accumulation Errors

**Symptom:** Reduced accuracy compared to FP32 baseline, especially for large models.
**Root cause:** Summing many FP16/BF16 values loses precision (catastrophic cancellation).
**Solutions:**
1. Use FP32 accumulators for reductions (PyTorch does this for matmul by default)
2. Keep layer norm, batch norm in FP32 (autocast handles this)
3. For custom kernels: accumulate in FP32, cast result to BF16

### Optimizer State Precision

**Symptom:** Training diverges after many steps.
**Root cause:** Adam's running averages (m, v) lose precision in FP16.
**Solution:** Always keep optimizer states in FP32. This is the default — never manually cast optimizer states to FP16.

### Debugging Checklist

```python
def check_precision_health(model, loss, step):
    """Call periodically during training to catch issues early."""
    # Check for NaN/inf in loss
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"Step {step}: Loss is {loss.item()}")
        return False

    # Check gradient statistics
    total_norm = 0.0
    num_zero = 0
    num_params = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.float().norm().item() ** 2
            num_zero += (p.grad == 0).sum().item()
            num_params += p.grad.numel()

    total_norm = total_norm ** 0.5
    zero_frac = num_zero / max(num_params, 1)

    if zero_frac > 0.5:
        print(f"Step {step}: {zero_frac:.1%} of gradients are zero (underflow?)")
    if total_norm > 100:
        print(f"Step {step}: Gradient norm = {total_norm:.1f} (explosion?)")

    return True
```

---

## 11. Precision-Performance Tradeoffs

### Expected Speedups (A100/H100)

| Precision | Matmul Throughput | Memory | Use Case |
|-----------|-------------------|--------|----------|
| FP32 | 1× (baseline) | 4B/param | Debugging, validation |
| TF32 | ~2× | 4B/param | Default (transparent) |
| FP16 + scaler | 2-3× | 2B/param | Older GPUs (V100, T4) |
| BF16 | 2-3× | 2B/param | LLM training (standard) |
| FP8 (H100) | 4-6× vs FP32 | 1B/param | Large-scale LLM training |

### When Each Format Is Appropriate

**FP32 only:**
- Debugging numerical issues
- Tiny models where speed doesn't matter
- Reference implementations for validation

**BF16 (most common for training):**
- LLM pretraining
- Fine-tuning large models
- Any Ampere+ GPU workload
- Default recommendation for new projects

**FP16 + GradScaler:**
- V100 / T4 deployments (no BF16 support)
- Inference on all GPUs (no overflow risk with bounded inputs)
- ONNX export (better ecosystem support)

**FP8:**
- H100 clusters running LLM pretraining
- When 2× over BF16 throughput justifies the engineering effort
- Models with matmul-heavy architectures (Transformers)

### Real-World Performance Numbers

Approximate speedups for a Transformer forward pass (batch=32, seq=2048, d=4096):

```
A100 GPU:
  FP32:             1.0× (baseline)
  TF32 (default):   1.8×
  BF16 autocast:    2.5×
  BF16 + compile:   3.2×

H100 GPU:
  BF16:             1.0× (new baseline)
  FP8:              1.6-2.0×
  FP8 + compile:    2.2-2.5×
```

### Memory Savings in Practice

For a 7B parameter model (LLaMA-like):

```
                     Parameters    Activations*   Optimizer    Total
FP32:                28 GB        ~40 GB          56 GB        ~124 GB
BF16 (mixed):        14 GB        ~20 GB          56 GB**      ~90 GB
FP8 (experimental):   7 GB        ~10 GB          56 GB**      ~73 GB

* Activations for batch=4, seq=4096 (approximate)
** Optimizer always in FP32 for stability
```

---

## 12. Upstream Updates (June 2026)

Recent PyTorch commits relevant to mixed precision and performance:

### SymmMem all_gather_offset (#187642)

Adds `all_gather_offset` to SymmMem for parameter-contiguous all-gather operations. Enables overlapping communication with computation in FSDP-style training by gathering only the offset portion of symmetrically allocated memory. Relevant for mixed-precision distributed training where parameter shards may be in different dtypes.

### all_to_all_nd for Ulysses-style attention (#178230)

Introduces N-dimensional all-to-all collective supporting Ulysses-style sequence parallel attention. This enables efficient attention computation across devices where KV pairs are distributed, working with BF16 attention tensors for memory efficiency.

### MPS FlexAttention KV batch broadcasting (#187722)

Extends FlexAttention on Apple Silicon (MPS backend) with KV batch broadcasting support. Allows K/V tensors with batch_size=1 to broadcast across query batches — critical for inference with KV cache in mixed-precision (float16 on MPS).

### Dynamo virtual iterator simplification (#187103)

Simplifies virtual iterator handling in TorchDynamo, reducing graph breaks in training loops that use custom iterators. Fewer graph breaks = more operations within a single compiled region = better opportunity for precision-related fusion optimizations.

### Native DSL RMSNorm fix for misaligned pointers (#186235)

Fixes a Native DSL RMSNorm implementation that could produce incorrect results with misaligned memory pointers. RMSNorm operates in FP32 for stability during mixed-precision training — a memory alignment bug here could silently corrupt the normalization, leading to training divergence.

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| BF16 | Default for training on Ampere+ GPUs — same range as FP32, no scaler needed |
| FP16 + GradScaler | Required for older GPUs — GradScaler prevents gradient underflow |
| FP8 | Cutting edge — 2× over BF16 on H100, requires careful scaling |
| autocast | Automatically handles per-op precision — just wrap forward pass |
| GradScaler | Only needed for FP16 — dynamically scales loss to prevent underflow |
| FSDP2 MixedPrecisionPolicy | Compute in BF16, reduce in FP32 for distributed stability |
| torch.compile | Fuses casts, eliminates redundant precision changes |

---

### Further Resources

- [PyTorch AMP documentation](https://pytorch.org/docs/stable/amp.html) — official autocast and GradScaler reference
- [NVIDIA Mixed Precision Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) — hardware perspective
- [Module 07 — Training Pipelines](../07_training/) — AMP in complete training loops
- [Module 08 — torch.compile](../08_torch_compile/) — compilation with mixed precision
- [Module 20 — Backends Tuning](../20_backends_tuning/) — TF32 and matmul precision settings
- [Module 10 — Distributed Training](../10_distributed/) — FSDP2 and mixed precision at scale
- [Module 28 — Benchmarking](../28_benchmarking/) — measuring precision-performance tradeoffs

---

<div align="center">

[← Previous Module](../28_benchmarking/) | [🏠 Home](../README.md) | [Next Module (Debugging) →](../30_debugging/)

**Notebook**: [`29_mixed_precision.ipynb`](../notebooks/29_mixed_precision.ipynb)

</div>
