# Module 30: Debugging PyTorch Models

<div align="center">

[← Previous Module (Mixed Precision)](../29_mixed_precision/) | [🏠 Home](../README.md) | [Next Module (torchao) →](../31_torchao/)

</div>

---

**Prerequisites**: [Module 07 — Training Pipelines](../07_training/), [Module 08 — torch.compile](../08_torch_compile/)
**Time**: ~2 hours
**Files**: `debugging_toolkit.py`, `compile_debugging.py`

---

## Table of Contents

1. [The Debugging Mindset](#1-the-debugging-mindset)
2. [Anomaly Detection](#2-anomaly-detection)
3. [NaN/Inf Detection](#3-naninf-detection)
4. [Gradient Flow Checking](#4-gradient-flow-checking)
5. [Shape Debugging](#5-shape-debugging)
6. [Device Mismatch](#6-device-mismatch)
7. [TORCH_SHOW_CPP_STACKTRACES](#7-torch_show_cpp_stacktraces)
8. [Debugging torch.compile](#8-debugging-torchcompile)
9. [Common Error Messages and Fixes](#9-common-error-messages-and-fixes)
10. [Memory Debugging](#10-memory-debugging)
11. [Performance Debugging](#11-performance-debugging)
12. [Reproducibility for Bug Reports](#12-reproducibility-for-bug-reports)
13. [Upstream Updates (June 20–22, 2026)](#13-upstream-updates-june-2022-2026)

---

## 1. The Debugging Mindset

Debugging PyTorch models requires a systematic approach. Random changes and trial-and-error waste hours. Follow this protocol:

```
Reproduce → Isolate → Identify → Fix → Verify
```

### The Protocol

**Step 1: Reproduce** — Create a minimal reproducer that triggers the bug *every time*. Strip away everything unnecessary: smaller batch size, fewer layers, synthetic data. A 20-line script that reproduces the bug is worth more than a 500-line training loop that "sometimes fails."

**Step 2: Isolate** — Narrow down where the problem occurs. Is it in the forward pass? Backward pass? Data loading? A specific layer? Use binary search: comment out half the model, check if the bug persists.

**Step 3: Identify** — Once isolated, understand *why* it happens. Read error messages carefully — PyTorch gives detailed tracebacks. Check tensor shapes, dtypes, devices, and values at the failure point.

**Step 4: Fix** — Apply the minimal fix. Don't rewrite working code around the bug.

**Step 5: Verify** — Run the original failing case AND related cases. Confirm the fix doesn't break other things.

### The Minimal Repro

Always start here. A good minimal repro:

```python
import torch
import torch.nn as nn

# Smallest model that triggers the bug
model = nn.Linear(10, 5)

# Simplest input that triggers the bug
x = torch.randn(2, 10)

# Exact sequence that fails
loss = model(x).sum()
loss.backward()
```

Strip data loading, logging, checkpointing, distributed — anything not needed to trigger the bug. If the bug disappears when you simplify, you've already learned something about its cause.

---

## 2. Anomaly Detection

PyTorch's autograd anomaly detection catches problems during the backward pass that are otherwise silent or produce cryptic errors later.

### Enabling Anomaly Detection

```python
# Context manager (preferred)
with torch.autograd.detect_anomaly():
    output = model(input)
    loss = criterion(output, target)
    loss.backward()

# Global setting
torch.autograd.set_detect_anomaly(True)
# ... training code ...
torch.autograd.set_detect_anomaly(False)
```

### What It Catches

| Problem | Without detect_anomaly | With detect_anomaly |
|---------|----------------------|---------------------|
| NaN in backward | Silent propagation | `RuntimeError` with traceback |
| In-place op on grad tensor | Cryptic error later | Immediate error at the op |
| Double backward without retain_graph | Confusing error | Clear traceback to the first backward |

### Example: Catching NaN in Backward

```python
import torch
import torch.nn as nn

class BuggyModel(nn.Module):
    def forward(self, x):
        # log(0) produces -inf, gradient becomes NaN
        return torch.log(x)

model = BuggyModel()
x = torch.zeros(5, requires_grad=True)  # log(0) = -inf

with torch.autograd.detect_anomaly():
    out = model(x)
    out.sum().backward()  # Raises RuntimeError with full traceback
```

### Performance Cost

Anomaly detection adds **significant overhead** (2-5x slower) because it:
- Records the full forward-pass traceback for every operation
- Validates every gradient in the backward pass

**Rule**: Enable only during debugging. Never in production training.

```python
# Good: enable only when investigating a bug
debug_mode = os.environ.get("DEBUG", "0") == "1"
torch.autograd.set_detect_anomaly(debug_mode)
```

---

## 3. NaN/Inf Detection

NaN (Not a Number) and Inf values are the most common silent killers in training. They propagate through computations and corrupt all downstream values.

### Manual Checks

```python
def check_tensor(t, name="tensor"):
    """Check a tensor for NaN/Inf values."""
    if torch.isnan(t).any():
        print(f"WARNING: NaN detected in {name}")
        print(f"  Shape: {t.shape}, NaN count: {torch.isnan(t).sum().item()}")
        return False
    if torch.isinf(t).any():
        print(f"WARNING: Inf detected in {name}")
        print(f"  Shape: {t.shape}, Inf count: {torch.isinf(t).sum().item()}")
        return False
    return True
```

### Hook-Based Automatic Detection

Register hooks to catch NaN/Inf as they appear, without modifying model code:

```python
def nan_hook(module, input, output):
    """Forward hook that detects NaN/Inf in module outputs."""
    if isinstance(output, torch.Tensor):
        if torch.isnan(output).any() or torch.isinf(output).any():
            raise RuntimeError(
                f"NaN/Inf detected in output of {module.__class__.__name__}\n"
                f"  Output shape: {output.shape}\n"
                f"  NaN count: {torch.isnan(output).sum().item()}\n"
                f"  Inf count: {torch.isinf(output).sum().item()}"
            )

# Register on all modules
for name, module in model.named_modules():
    module.register_forward_hook(nan_hook)
```

### Common Causes

| Cause | Example | Fix |
|-------|---------|-----|
| Learning rate too high | Gradients explode → weights overflow | Reduce LR, use gradient clipping |
| log(0) | `torch.log(probabilities)` where some are 0 | `torch.log(x + 1e-8)` or `torch.clamp(x, min=1e-8)` |
| Division by zero | `x / norm` where norm is 0 | `x / (norm + 1e-8)` |
| Softmax overflow | Very large logits → exp overflow | Use `log_softmax` instead of `log(softmax(x))` |
| sqrt(0) gradient | `torch.sqrt(x)` at x=0 has infinite gradient | `torch.sqrt(x + 1e-8)` |
| Unstable loss | Cross-entropy with raw probabilities | Use `F.cross_entropy` (numerically stable) |

### Gradient NaN Detection

```python
def grad_nan_hook(module, grad_input, grad_output):
    """Backward hook that detects NaN in gradients."""
    for i, grad in enumerate(grad_output):
        if grad is not None and torch.isnan(grad).any():
            raise RuntimeError(
                f"NaN gradient in {module.__class__.__name__} "
                f"grad_output[{i}], shape={grad.shape}"
            )

for module in model.modules():
    module.register_full_backward_hook(grad_nan_hook)
```

---

## 4. Gradient Flow Checking

Training may fail silently when gradients vanish or explode. The model "trains" but loss never decreases, or it diverges suddenly.

### Check That Gradients Exist

```python
# After loss.backward()
for name, param in model.named_parameters():
    if param.requires_grad:
        if param.grad is None:
            print(f"NO GRADIENT: {name}")
        elif param.grad.norm() == 0:
            print(f"ZERO GRADIENT: {name}")
```

### Gradient Flow Checker

```python
def check_gradient_flow(named_parameters):
    """Print gradient statistics for all parameters."""
    print(f"{'Layer':<40} {'Grad Norm':<12} {'Grad Mean':<12} {'Grad Max':<12}")
    print("-" * 76)
    for name, param in named_parameters:
        if param.requires_grad and param.grad is not None:
            grad = param.grad
            print(f"{name:<40} {grad.norm():<12.6f} "
                  f"{grad.mean():<12.2e} {grad.abs().max():<12.2e}")
```

### Detecting Vanishing Gradients

Symptoms: loss plateaus, early layers have near-zero gradients, later layers have normal gradients.

```python
def detect_vanishing_gradients(model, threshold=1e-7):
    """Detect layers with vanishing gradients."""
    vanishing = []
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.norm() < threshold:
            vanishing.append((name, param.grad.norm().item()))
    if vanishing:
        print("WARNING: Vanishing gradients detected:")
        for name, norm in vanishing:
            print(f"  {name}: grad_norm = {norm:.2e}")
    return vanishing
```

### Detecting Exploding Gradients

Symptoms: loss becomes NaN/Inf suddenly, gradient norms grow exponentially each step.

```python
def detect_exploding_gradients(model, threshold=100.0):
    """Detect layers with exploding gradients."""
    exploding = []
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.norm() > threshold:
            exploding.append((name, param.grad.norm().item()))
    if exploding:
        print("WARNING: Exploding gradients detected:")
        for name, norm in exploding:
            print(f"  {name}: grad_norm = {norm:.2e}")
    return exploding
```

### Fix: Gradient Clipping

```python
# Clip by norm (most common)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Clip by value
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

---

## 5. Shape Debugging

Shape mismatches are the most common PyTorch error. They produce clear error messages, but finding *which* operation caused the mismatch in a large model can be tricky.

### Strategy 1: Print Shapes at Each Step

```python
class DebugModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        print(f"Input: {x.shape}")
        for i, layer in enumerate(self.layers):
            x = layer(x)
            print(f"After layer {i} ({layer.__class__.__name__}): {x.shape}")
        return x
```

### Strategy 2: Shape-Logging Hook

```python
def shape_hook(name):
    """Create a hook that logs input/output shapes for a module."""
    def hook(module, input, output):
        in_shapes = [x.shape if isinstance(x, torch.Tensor) else type(x) for x in input]
        out_shape = output.shape if isinstance(output, torch.Tensor) else type(output)
        print(f"{name}: input={in_shapes} → output={out_shape}")
    return hook

# Register on all modules
for name, module in model.named_modules():
    if not list(module.children()):  # leaf modules only
        module.register_forward_hook(shape_hook(name))
```

### Strategy 3: Model Surgery (Isolate the Layer)

When a model is too large to debug as a whole, run each layer individually:

```python
x = torch.randn(batch_size, channels, height, width)
for name, module in model.named_children():
    try:
        x = module(x)
        print(f"✓ {name}: output shape = {x.shape}")
    except Exception as e:
        print(f"✗ {name}: FAILED — {e}")
        print(f"  Input shape was: {x.shape}")
        break
```

### Strategy 4: torchinfo

```python
from torchinfo import summary
summary(model, input_size=(1, 3, 224, 224))
```

This prints a table showing each layer's input/output shape, parameter count, and multiply-accumulate operations.

### Common Shape Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `mat1 and mat2 shapes cannot be multiplied` | Linear layer input size wrong | Check `in_features` matches input dim |
| `Expected 4D input (got 2D)` | Conv2d needs (B, C, H, W) | `x.unsqueeze(0).unsqueeze(0)` or fix data |
| `size mismatch, m1: [32 x 512], m2: [256 x 10]` | Flatten size doesn't match Linear input | Calculate correct flatten size |

---

## 6. Device Mismatch

The error `Expected all tensors to be on the same device` means you're mixing CPU and CUDA tensors in an operation.

### Systematic Fix

```python
# Step 1: Check where tensors live
def print_devices(model, inputs):
    """Print device of all model parameters and inputs."""
    print("Model parameters:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.device}")
    print("\nInputs:")
    if isinstance(inputs, torch.Tensor):
        print(f"  input: {inputs.device}")
    elif isinstance(inputs, (list, tuple)):
        for i, inp in enumerate(inputs):
            if isinstance(inp, torch.Tensor):
                print(f"  input[{i}]: {inp.device}")
```

### Device Checker Hook

```python
def device_check_hook(expected_device):
    """Hook that verifies all inputs/outputs are on the expected device."""
    def hook(module, input, output):
        for i, inp in enumerate(input):
            if isinstance(inp, torch.Tensor) and inp.device != expected_device:
                raise RuntimeError(
                    f"{module.__class__.__name__}: input[{i}] is on "
                    f"{inp.device}, expected {expected_device}"
                )
    return hook
```

### Common Causes

1. **Forgot to move input to GPU**: `x = x.to(device)` before passing to model
2. **Created tensor inside forward()**: Use `torch.zeros(..., device=x.device)` not `torch.zeros(...)`
3. **Loss target on wrong device**: `target = target.to(device)`
4. **Buffer not registered**: Use `self.register_buffer('name', tensor)` not `self.tensor = tensor`

### The Fix Pattern

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for batch in dataloader:
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    output = model(inputs)
    loss = criterion(output, targets)
```

---

## 7. TORCH_SHOW_CPP_STACKTRACES

When PyTorch crashes at the C++ level, Python tracebacks only show you the Python call that triggered the error. The actual cause is buried in C++ code.

### Enabling C++ Stacktraces

```bash
TORCH_SHOW_CPP_STACKTRACES=1 python script.py
```

### What It Shows

Without the environment variable:
```
RuntimeError: CUDA error: device-side assert triggered
```

With `TORCH_SHOW_CPP_STACKTRACES=1`:
```
RuntimeError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call...

C++ Stacktrace:
  at aten::index_select(self, dim, index)
  at torch::autograd::...
  ...
```

### When to Use

- Segfaults or crashes (not Python exceptions)
- CUDA runtime errors
- Internal PyTorch assertion failures
- Errors from custom C++ extensions

### Other Useful Environment Variables

```bash
# Synchronous CUDA errors (pinpoints the exact kernel)
CUDA_LAUNCH_BLOCKING=1 python script.py

# Both together for maximum debug info
TORCH_SHOW_CPP_STACKTRACES=1 CUDA_LAUNCH_BLOCKING=1 python script.py

# Disable CUDA caching allocator (for memory debugging)
PYTORCH_NO_CUDA_MEMORY_CACHING=1 python script.py
```

---

## 8. Debugging torch.compile

`torch.compile` introduces a new class of errors: graph breaks, recompilations, and backend failures. These are different from standard PyTorch bugs.

### Graph Breaks: Detection

A graph break means Dynamo couldn't compile a section of your code, falling back to eager mode. This hurts performance.

```python
# Method 1: TORCH_LOGS environment variable
# TORCH_LOGS="graph_breaks" python script.py

# Method 2: explain() API
import torch._dynamo as dynamo

def my_function(x):
    x = x * 2
    print(x)  # This causes a graph break!
    return x + 1

explanation = dynamo.explain(my_function)(torch.randn(10))
print(explanation)
# Shows: graph_break_count, break_reasons, out_guards
```

### Graph Breaks: Common Causes and Fixes

| Cause | Example | Fix |
|-------|---------|-----|
| `print()` in compiled code | `print(x.shape)` | Remove or guard with `if not torch.compiler.is_compiling()` |
| Data-dependent control flow | `if x.sum() > 0:` | Use `torch.where` or `torch.cond` |
| Unsupported Python builtin | `sorted(list)` | Rewrite with torch ops |
| Non-tensor data structures | Building a list in a loop | Use tensor operations |
| Calling uncompiled functions | External library calls | Wrap or inline |

### Recompilation Detection

Recompilation happens when Dynamo's guards are triggered (input shapes change, etc.):

```bash
TORCH_LOGS="recompiles" python script.py
```

```python
# Programmatic detection
import torch._dynamo as dynamo

# Count compilations
compile_count = 0
def counting_compiler(gm, example_inputs):
    global compile_count
    compile_count += 1
    return gm

compiled_fn = torch.compile(my_fn, backend=counting_compiler)
```

### The Minifier

When `torch.compile` produces wrong results or crashes, the minifier creates a minimal reproduction:

```python
import torch._dynamo.config
# Generate minimal repro after dynamo error
torch._dynamo.config.repro_after = "dynamo"
# Or after AOTAutograd
torch._dynamo.config.repro_after = "aot"
```

### Verbose Mode

```bash
# Full compilation logs
TORCH_LOGS="dynamo" python script.py

# Inductor-generated code
TORCH_LOGS="output_code" python script.py

# Everything (very verbose)
TORCH_LOGS="+dynamo,+inductor" python script.py
```

### Debugging Wrong Results

```python
# Compare compiled vs eager outputs
model_eager = MyModel()
model_compiled = torch.compile(MyModel())

# Load same weights
model_compiled.load_state_dict(model_eager.state_dict())

x = torch.randn(2, 10)
out_eager = model_eager(x)
out_compiled = model_compiled(x)

print(f"Max difference: {(out_eager - out_compiled).abs().max()}")
assert torch.allclose(out_eager, out_compiled, atol=1e-5)
```

---

## 9. Common Error Messages and Fixes

### Error Table

| # | Error | Cause | Fix |
|---|-------|-------|-----|
| 1 | `RuntimeError: CUDA out of memory` | GPU memory exhausted | Reduce batch size, use gradient checkpointing, use mixed precision, call `torch.cuda.empty_cache()` |
| 2 | `RuntimeError: Expected all tensors to be on the same device` | Mixing CPU and CUDA tensors | Move all tensors to same device with `.to(device)` |
| 3 | `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation` | In-place op on a tensor needed for backward | Replace `x.add_(1)` with `x = x + 1`, avoid in-place ops on leaf tensors |
| 4 | `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | Calling `.backward()` on a non-grad tensor | Ensure inputs have `requires_grad=True`, check model parameters |
| 5 | `torch._dynamo.exc.Unsupported` | Graph break in torch.compile | See Section 8 — remove unsupported ops or use `torch.compiler.is_compiling()` guard |
| 6 | `CUDA error: device-side assert triggered` | Index out of bounds in CUDA kernel | Run with `CUDA_LAUNCH_BLOCKING=1`, check label indices < num_classes |
| 7 | `RuntimeError: Trying to backward through the graph a second time` | Calling `.backward()` twice without `retain_graph=True` | Add `retain_graph=True` or restructure to avoid double backward |
| 8 | `RuntimeError: expected scalar type Float but found Half` | Dtype mismatch between FP32 and FP16 | Use `autocast` or explicit `.float()` / `.half()` conversion |
| 9 | `RuntimeError: mat1 and mat2 shapes cannot be multiplied` | Linear layer shape mismatch | Check `in_features` matches the flattened input dimension |
| 10 | `ValueError: optimizer got an empty parameter list` | No parameters passed to optimizer | Check `model.parameters()` is not empty, ensure modules are registered as attributes |
| 11 | `RuntimeError: Input type and weight type should be the same` | Mixed dtypes (e.g., double input, float weights) | Use `x = x.float()` or `model.double()` |
| 12 | `RuntimeError: expected stride to be a single integer or a list of integers` | Wrong argument type to operation | Check function signature — likely passing tensor where int expected |

### Detailed Examples

#### CUDA Out of Memory

```python
# Diagnose
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# Fix 1: Reduce batch size
# Fix 2: Gradient checkpointing
from torch.utils.checkpoint import checkpoint
# Fix 3: Mixed precision
with torch.autocast('cuda'):
    output = model(input)
# Fix 4: Clear cache (doesn't free PyTorch tensors, just cached blocks)
torch.cuda.empty_cache()
```

#### In-Place Operation Error

```python
# BAD: in-place modification of a tensor needed for backward
x = torch.randn(5, requires_grad=True)
y = x ** 2
x.mul_(2)  # In-place modification!
y.sum().backward()  # ERROR

# GOOD: create a new tensor
x = torch.randn(5, requires_grad=True)
y = x ** 2
x_new = x * 2  # New tensor, x unchanged
y.sum().backward()  # Works
```

#### Device-Side Assert (Index Out of Bounds)

```python
# Common cause: label index >= num_classes
num_classes = 10
labels = torch.tensor([0, 5, 10])  # 10 is out of bounds!
output = torch.randn(3, num_classes)
loss = F.cross_entropy(output, labels)  # CUDA assert!

# Fix: clamp or validate labels
assert labels.max() < num_classes, f"Label {labels.max()} >= {num_classes}"
```

---

## 10. Memory Debugging

### Detecting Memory Leaks

A memory leak in PyTorch usually means tensors are being held alive unintentionally.

```python
import gc

def check_memory_growth(model, dataloader, num_steps=10):
    """Check if memory grows over training steps."""
    memory_log = []
    for i, (x, y) in enumerate(dataloader):
        if i >= num_steps:
            break

        output = model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()

        # Record memory AFTER backward
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated()
        else:
            import psutil
            mem = psutil.Process().memory_info().rss

        memory_log.append(mem)

        # Critical: zero gradients
        model.zero_grad(set_to_none=True)

    # Check for growth
    if memory_log[-1] > memory_log[0] * 1.1:
        print(f"WARNING: Memory grew from {memory_log[0]/1e6:.1f}MB "
              f"to {memory_log[-1]/1e6:.1f}MB over {num_steps} steps")
    return memory_log
```

### Common Memory Leak Causes

1. **Storing loss history without `.item()`**:
   ```python
   # BAD: holds entire computation graph!
   losses.append(loss)

   # GOOD: detach the scalar value
   losses.append(loss.item())
   ```

2. **Not zeroing gradients**:
   ```python
   # Gradients accumulate by default
   optimizer.zero_grad()  # or model.zero_grad(set_to_none=True)
   ```

3. **Holding references in hooks**:
   ```python
   # BAD: closure holds reference to output
   outputs = []
   def hook(m, i, o):
       outputs.append(o)  # Keeps tensor alive!

   # GOOD: detach or only store what you need
   def hook(m, i, o):
       outputs.append(o.detach().cpu())
   ```

### Memory Snapshot (CUDA)

```python
# Record memory history
torch.cuda.memory._record_memory_history()

# ... run your code ...

# Save snapshot
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
torch.cuda.memory._record_memory_history(enabled=None)

# Analyze with: https://pytorch.org/memory_viz
```

---

## 11. Performance Debugging

### torch.profiler

```python
from torch.profiler import profile, ProfilerActivity, schedule

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, (x, y) in enumerate(dataloader):
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        prof.step()
```

### CPU-Bound vs GPU-Bound

```python
# Quick test: does CUDA synchronization slow things down?
import time

torch.cuda.synchronize()
start = time.time()

for _ in range(100):
    output = model(x)

torch.cuda.synchronize()
elapsed = time.time() - start

# If adding synchronize() doesn't change timing much → CPU-bound
# If it significantly increases time → GPU is already the bottleneck
```

### Data Loading Bottleneck

```python
# If GPU utilization is low, data loading may be the bottleneck
import time

# Time data loading
load_times = []
for i, batch in enumerate(dataloader):
    if i >= 10:
        break
    start = time.time()
    x, y = batch
    x = x.to(device)
    load_times.append(time.time() - start)

# Time model
model_times = []
for i, batch in enumerate(dataloader):
    if i >= 10:
        break
    x, y = batch[0].to(device), batch[1].to(device)
    start = time.time()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    torch.cuda.synchronize()
    model_times.append(time.time() - start)

print(f"Avg data load: {sum(load_times)/len(load_times)*1000:.1f}ms")
print(f"Avg model step: {sum(model_times)/len(model_times)*1000:.1f}ms")
```

**Fix data loading bottlenecks**: increase `num_workers`, enable `pin_memory=True`, use `persistent_workers=True`, pre-process data.

---

## 12. Reproducibility for Bug Reports

When filing a bug report (or debugging your own code), reproducibility is essential.

### Setting All Seeds

```python
import torch
import numpy as np
import random

def set_all_seeds(seed=42):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Deterministic algorithms (may be slower)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### Capturing Environment Info

```python
def print_environment():
    """Print all relevant environment information for bug reports."""
    import sys
    import platform
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"OS: {platform.platform()}")
    print(f"\nFull config:\n{torch.__config__.show()}")
```

### Minimal Repro Template

```python
"""
Minimal reproducer for [describe the bug].

Environment:
- PyTorch: [version]
- Python: [version]
- OS: [os]
- GPU: [gpu or CPU]

Steps to reproduce:
1. Run this script

Expected: [what should happen]
Actual: [what actually happens]
"""
import torch

torch.manual_seed(42)

# Minimal code that triggers the bug
model = ...
x = ...
output = model(x)  # Bug occurs here
```

---

## 13. Upstream Updates (June 20–22, 2026)

Recent PyTorch commits relevant to debugging and general usage:

| PR | Title | Impact |
|----|-------|--------|
| [#187768](https://github.com/pytorch/pytorch/pull/187768) | MPS FlexAttention `lse` return | FlexAttention on MPS now correctly returns log-sum-exp alongside attention output |
| [#187758](https://github.com/pytorch/pytorch/pull/187758) | `Sequential.__getitem__` type overloads | Better type checking when indexing `nn.Sequential` — clearer errors for invalid indexing |
| [#187776](https://github.com/pytorch/pytorch/pull/187776) | SymmMem copy optimization | Optimized symmetric memory copy for distributed training — reduced latency |
| [#187702](https://github.com/pytorch/pytorch/pull/187702) | vmap batching rule for `repeat_interleave` | `torch.vmap` now supports `repeat_interleave` — no more manual unbatching workaround |
| [#184653](https://github.com/pytorch/pytorch/pull/184653) | Dynamo globals fix for unregistered modules | Fixed graph break when accessing global modules not registered as submodules — helps torch.compile debugging |
| [#187778](https://github.com/pytorch/pytorch/pull/187778) | `all_to_all_nd` narrow-row throughput fix | Improved throughput for narrow-row all-to-all patterns common in MoE training |

### Impact on Debugging

- **#184653** is directly relevant: if you had graph breaks from accessing global module objects, this is now fixed. Upgrade PyTorch to resolve.
- **#187758** improves error messages when mis-indexing Sequential — less confusing debugging.
- **#187768** fixes a subtle bug where MPS FlexAttention returned wrong `lse` values — this would show up as incorrect loss values on Apple Silicon.

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PyTorch Debugging Quick Reference                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  NaN/Inf Detection:                                                      │
│    torch.autograd.set_detect_anomaly(True)                               │
│    torch.isnan(t).any()  /  torch.isinf(t).any()                        │
│                                                                          │
│  Gradient Debugging:                                                     │
│    param.grad is None → not connected to loss                            │
│    param.grad.norm() ≈ 0 → vanishing gradients                          │
│    param.grad.norm() → ∞ → exploding gradients                          │
│                                                                          │
│  torch.compile Debugging:                                                │
│    TORCH_LOGS="graph_breaks" python script.py                            │
│    TORCH_LOGS="recompiles" python script.py                              │
│    torch._dynamo.explain(fn)(inputs)                                     │
│                                                                          │
│  C++ Errors:                                                             │
│    TORCH_SHOW_CPP_STACKTRACES=1 python script.py                        │
│    CUDA_LAUNCH_BLOCKING=1 python script.py                               │
│                                                                          │
│  Memory:                                                                 │
│    torch.cuda.memory_allocated()                                         │
│    torch.cuda.memory_summary()                                           │
│    torch.cuda.memory._record_memory_history()                            │
│                                                                          │
│  Reproducibility:                                                        │
│    torch.manual_seed(42)                                                 │
│    torch.use_deterministic_algorithms(True)                              │
│    torch.__config__.show()                                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Takeaways

| Principle | Implementation |
|-----------|---------------|
| Always create a minimal repro | Strip to smallest code that reproduces the bug |
| Use anomaly detection | `torch.autograd.set_detect_anomaly(True)` — but only when debugging |
| Check NaN/Inf early | Register forward hooks on all modules |
| Monitor gradient norms | Log per-layer gradient norms each step |
| Print shapes systematically | Hooks > manual prints > torchinfo |
| Fix device mismatches at data boundary | `.to(device)` right after data loading |
| Use environment variables for C++ issues | `TORCH_SHOW_CPP_STACKTRACES=1`, `CUDA_LAUNCH_BLOCKING=1` |
| Use explain() for compile issues | `torch._dynamo.explain(fn)(inputs)` |
| Store scalars not tensors | `loss.item()` not `loss` |
| Set all seeds for repro | `torch.manual_seed`, `np.random.seed`, `random.seed` |

---

### Further Resources

- [PyTorch Debugging FAQ](https://pytorch.org/docs/stable/notes/faq.html) — official troubleshooting
- [torch.compile troubleshooting](https://pytorch.org/docs/stable/torch.compiler_troubleshooting.html) — Dynamo debugging guide
- [Module 07 — Training Pipelines](../07_training/) — gradient clipping and AMP
- [Module 08 — torch.compile](../08_torch_compile/) — compilation fundamentals
- [Module 26 — Memory Profiling](../26_memory_profiling/) — detailed memory analysis
- [Module 29 — Mixed Precision](../29_mixed_precision/) — dtype-related debugging

---

<div align="center">

[← Previous Module (Mixed Precision)](../29_mixed_precision/) | [🏠 Home](../README.md) | [Next Module (torchao) →](../31_torchao/)

**Notebook**: [`30_debugging.ipynb`](../notebooks/30_debugging.ipynb)

</div>
