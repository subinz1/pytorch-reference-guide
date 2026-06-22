"""
Module 30: Debugging PyTorch Models — Debugging Toolkit
========================================================

Runnable on CPU. Demonstrates:
- Anomaly detection (trigger and catch NaN)
- NaN/Inf checker function and hook
- Gradient flow checker
- Shape debugging helper
- Device checker utility
- Common error reproduction and fix
- Memory leak detection pattern
- Reproducibility setup

Run: python debugging_toolkit.py
"""

import gc
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. Anomaly Detection Demo
# ============================================================================


def demo_anomaly_detection():
    """Demonstrate autograd anomaly detection catching NaN in backward."""
    print("=" * 70)
    print("1. ANOMALY DETECTION")
    print("=" * 70)

    class NaNProducingModel(nn.Module):
        def forward(self, x):
            return torch.log(x)

    model = NaNProducingModel()
    x = torch.tensor([1.0, 0.5, 0.0, -1.0], requires_grad=True)

    print(f"Input: {x}")
    print(f"log(input): {torch.log(x.detach())}")
    print("Note: log(0) = -inf, log(-1) = nan")
    print()

    # Without anomaly detection: NaN propagates silently
    print("Without detect_anomaly:")
    try:
        out = model(x.clamp(min=1e-8))
        out.sum().backward()
        print(f"  Backward completed (NaN avoided via clamp)")
    except RuntimeError as e:
        print(f"  Error: {e}")

    # With anomaly detection: catches the problem
    print("\nWith detect_anomaly (triggering NaN):")
    x_bad = torch.tensor([0.0, 1.0], requires_grad=True)
    try:
        with torch.autograd.detect_anomaly():
            out = torch.log(x_bad)
            out.sum().backward()
        print("  Backward completed (no anomaly)")
    except RuntimeError as e:
        print(f"  Caught: {str(e)[:100]}...")

    print()


# ============================================================================
# 2. NaN/Inf Checker
# ============================================================================


def check_tensor(t: torch.Tensor, name: str = "tensor") -> bool:
    """Check a tensor for NaN/Inf values. Returns True if clean."""
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    if has_nan:
        nan_count = torch.isnan(t).sum().item()
        print(f"  WARNING: NaN in {name} — count={nan_count}, shape={t.shape}")
    if has_inf:
        inf_count = torch.isinf(t).sum().item()
        print(f"  WARNING: Inf in {name} — count={inf_count}, shape={t.shape}")
    return not (has_nan or has_inf)


class NaNDetectorHook:
    """Forward hook that detects NaN/Inf in module outputs."""

    def __init__(self):
        self.issues = []

    def __call__(self, module, input, output):
        if isinstance(output, torch.Tensor):
            if torch.isnan(output).any() or torch.isinf(output).any():
                issue = {
                    "module": module.__class__.__name__,
                    "nan_count": torch.isnan(output).sum().item(),
                    "inf_count": torch.isinf(output).sum().item(),
                    "shape": tuple(output.shape),
                }
                self.issues.append(issue)


def demo_nan_detection():
    """Demonstrate NaN/Inf detection utilities."""
    print("=" * 70)
    print("2. NaN/Inf DETECTION")
    print("=" * 70)

    # Manual checking
    clean = torch.randn(3, 4)
    dirty = torch.tensor([1.0, float("nan"), float("inf"), -float("inf")])

    print("Manual check (clean tensor):")
    check_tensor(clean, "clean")
    print("  OK — no issues")

    print("\nManual check (dirty tensor):")
    check_tensor(dirty, "dirty")

    # Hook-based detection
    print("\nHook-based detection on a model:")

    class LeakyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)

        def forward(self, x):
            x = self.linear(x)
            x = torch.log(x)  # Will produce NaN for negative inputs
            return x

    model = LeakyModel()
    detector = NaNDetectorHook()

    for name, module in model.named_modules():
        module.register_forward_hook(detector)

    x = torch.randn(2, 10)
    with torch.no_grad():
        _ = model(x)

    if detector.issues:
        print(f"  Detected {len(detector.issues)} modules with NaN/Inf:")
        for issue in detector.issues:
            print(f"    {issue['module']}: NaN={issue['nan_count']}, "
                  f"Inf={issue['inf_count']}, shape={issue['shape']}")
    else:
        print("  No NaN/Inf detected")

    print()


# ============================================================================
# 3. Gradient Flow Checker
# ============================================================================


def check_gradient_flow(named_parameters):
    """Print gradient statistics for all parameters."""
    print(f"  {'Layer':<35} {'Norm':<10} {'Mean':<12} {'Max':<12} {'Status'}")
    print(f"  {'-'*35} {'-'*10} {'-'*12} {'-'*12} {'-'*10}")

    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        if param.grad is None:
            print(f"  {name:<35} {'—':<10} {'—':<12} {'—':<12} NO GRAD")
            continue

        grad = param.grad
        norm = grad.norm().item()
        mean = grad.mean().item()
        max_val = grad.abs().max().item()

        if norm < 1e-7:
            status = "VANISHING"
        elif norm > 100:
            status = "EXPLODING"
        else:
            status = "OK"

        print(f"  {name:<35} {norm:<10.6f} {mean:<12.2e} {max_val:<12.2e} {status}")


def demo_gradient_flow():
    """Demonstrate gradient flow checking."""
    print("=" * 70)
    print("3. GRADIENT FLOW CHECKING")
    print("=" * 70)

    # Create a deep model that may have vanishing gradients
    class DeepModel(nn.Module):
        def __init__(self, depth=6):
            super().__init__()
            layers = []
            for _ in range(depth):
                layers.extend([nn.Linear(32, 32), nn.Sigmoid()])
            layers.append(nn.Linear(32, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    model = DeepModel(depth=6)
    x = torch.randn(4, 32)
    target = torch.randn(4, 1)

    output = model(x)
    loss = F.mse_loss(output, target)
    loss.backward()

    print("Gradient flow in a 6-layer Sigmoid network:")
    check_gradient_flow(model.named_parameters())
    print(f"\n  Note: Sigmoid causes vanishing gradients in deep networks.")
    print(f"  Fix: Use ReLU, residual connections, or proper initialization.")
    print()


# ============================================================================
# 4. Shape Debugging Helper
# ============================================================================


class ShapeLogger:
    """Hook-based shape logger for debugging dimension mismatches."""

    def __init__(self):
        self.log = []
        self._handles = []

    def _make_hook(self, name):
        def hook(module, input, output):
            in_shapes = []
            for x in input:
                if isinstance(x, torch.Tensor):
                    in_shapes.append(tuple(x.shape))
                else:
                    in_shapes.append(type(x).__name__)
            out_shape = tuple(output.shape) if isinstance(output, torch.Tensor) else type(output).__name__
            self.log.append({"name": name, "input": in_shapes, "output": out_shape})
        return hook

    def attach(self, model):
        """Attach shape logging hooks to all leaf modules."""
        for name, module in model.named_modules():
            if not list(module.children()):
                handle = module.register_forward_hook(self._make_hook(name))
                self._handles.append(handle)

    def remove(self):
        """Remove all hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def print_log(self):
        """Print the shape log."""
        print(f"  {'Module':<30} {'Input Shapes':<25} {'Output Shape'}")
        print(f"  {'-'*30} {'-'*25} {'-'*15}")
        for entry in self.log:
            in_str = str(entry["input"])
            print(f"  {entry['name']:<30} {in_str:<25} {entry['output']}")
        self.log.clear()


def demo_shape_debugging():
    """Demonstrate shape debugging tools."""
    print("=" * 70)
    print("4. SHAPE DEBUGGING")
    print("=" * 70)

    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

    logger = ShapeLogger()
    logger.attach(model)

    x = torch.randn(8, 784)
    print(f"Input shape: {x.shape}")
    print()

    _ = model(x)
    logger.print_log()
    logger.remove()

    # Show a shape mismatch error
    print("\nDeliberate shape mismatch:")
    bad_input = torch.randn(8, 100)  # Wrong input size
    try:
        _ = model(bad_input)
    except RuntimeError as e:
        print(f"  Error: {e}")
        print(f"  Fix: Input dim is 100 but Linear expects 784")

    print()


# ============================================================================
# 5. Device Checker Utility
# ============================================================================


def check_devices(model, *inputs, label=""):
    """Print devices of model parameters and inputs."""
    devices = set()
    print(f"  Device check{f' ({label})' if label else ''}:")

    for name, param in model.named_parameters():
        devices.add(str(param.device))
    print(f"    Model params: {devices}")

    for i, inp in enumerate(inputs):
        if isinstance(inp, torch.Tensor):
            print(f"    Input[{i}]: {inp.device}")
            devices.add(str(inp.device))

    if len(devices) > 1:
        print(f"    WARNING: Multiple devices detected: {devices}")
        return False
    print(f"    OK — all on {devices.pop()}")
    return True


def demo_device_checking():
    """Demonstrate device mismatch detection."""
    print("=" * 70)
    print("5. DEVICE MISMATCH DETECTION")
    print("=" * 70)

    model = nn.Linear(10, 5)
    x = torch.randn(2, 10)

    # Everything on CPU — OK
    check_devices(model, x, label="all CPU")
    print()

    # Simulate what would happen with mixed devices
    print("  Common device mismatch patterns:")
    print("    1. Model on CUDA, input on CPU → .to(device) the input")
    print("    2. Created tensor in forward() without device= → use input.device")
    print("    3. Target not moved to GPU → target = target.to(device)")
    print("    4. Unregistered buffer → use self.register_buffer()")
    print()


# ============================================================================
# 6. Common Error Reproduction and Fix
# ============================================================================


def demo_common_errors():
    """Reproduce and fix common PyTorch errors."""
    print("=" * 70)
    print("6. COMMON ERRORS — REPRODUCE AND FIX")
    print("=" * 70)

    # Error 1: In-place operation on grad-requiring tensor
    print("\n  Error: In-place op on tensor needed for gradient")
    x = torch.randn(5, requires_grad=True)
    y = x * 2
    try:
        x.mul_(3)  # In-place modification
        y.sum().backward()
    except RuntimeError as e:
        print(f"    Caught: {str(e)[:80]}...")
    print("    Fix: Use x_new = x * 3 instead of x.mul_(3)")

    # Error 2: Backward through graph second time
    print("\n  Error: Backward through graph a second time")
    x = torch.randn(3, requires_grad=True)
    y = x.sum()
    y.backward()
    try:
        y.backward()
    except RuntimeError as e:
        print(f"    Caught: {str(e)[:80]}...")
    print("    Fix: Use retain_graph=True or restructure computation")

    # Error 3: No grad function
    print("\n  Error: Tensor does not require grad")
    x = torch.randn(3)  # requires_grad=False by default
    try:
        x.backward()
    except RuntimeError as e:
        print(f"    Caught: {str(e)[:80]}...")
    print("    Fix: Set requires_grad=True on input tensors")

    # Error 4: Empty parameter list
    print("\n  Error: Empty parameter list for optimizer")

    class BrokenModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Bug: not registered as attribute
            self._hidden = nn.Linear(10, 5)

        def forward(self, x):
            return self._hidden(x)

    broken = BrokenModel()
    param_count = sum(1 for _ in broken.parameters())
    # In this case _hidden IS found (since nn.Module scans attributes),
    # but if stored in a plain list it wouldn't be:
    print(f"    Parameters found: {param_count}")
    print("    Tip: Use nn.ModuleList not Python list for sub-modules")
    print()


# ============================================================================
# 7. Memory Leak Detection Pattern
# ============================================================================


def demo_memory_leak_detection():
    """Demonstrate memory leak detection on CPU."""
    print("=" * 70)
    print("7. MEMORY LEAK DETECTION")
    print("=" * 70)

    import tracemalloc
    tracemalloc.start()

    model = nn.Linear(100, 50)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # BAD pattern: storing loss tensors (holds computation graph)
    bad_losses = []
    for i in range(50):
        x = torch.randn(32, 100)
        y = torch.randn(32, 50)
        out = model(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        bad_losses.append(loss)  # BUG: holds entire graph

    current, peak = tracemalloc.get_traced_memory()
    print(f"  BAD pattern (storing loss tensors):")
    print(f"    Current memory: {current / 1024:.1f} KB")
    print(f"    Peak memory:    {peak / 1024:.1f} KB")

    del bad_losses
    gc.collect()
    tracemalloc.stop()
    tracemalloc.start()

    # GOOD pattern: storing only scalar values
    good_losses = []
    for i in range(50):
        x = torch.randn(32, 100)
        y = torch.randn(32, 50)
        out = model(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        good_losses.append(loss.item())  # GOOD: only stores float

    current, peak = tracemalloc.get_traced_memory()
    print(f"\n  GOOD pattern (storing loss.item()):")
    print(f"    Current memory: {current / 1024:.1f} KB")
    print(f"    Peak memory:    {peak / 1024:.1f} KB")

    tracemalloc.stop()
    print(f"\n  Rule: Always use loss.item() when logging loss values")
    print()


# ============================================================================
# 8. Reproducibility Setup
# ============================================================================


def set_reproducibility(seed: int = 42):
    """Set all random seeds and enable deterministic mode."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def get_environment_info() -> str:
    """Collect environment info for bug reports."""
    import platform
    lines = [
        f"Python: {sys.version.split()[0]}",
        f"PyTorch: {torch.__version__}",
        f"NumPy: {np.__version__}",
        f"OS: {platform.platform()}",
        f"CUDA available: {torch.cuda.is_available()}",
    ]
    if torch.cuda.is_available():
        lines.append(f"CUDA version: {torch.version.cuda}")
        lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
    return "\n".join(lines)


def demo_reproducibility():
    """Demonstrate reproducibility setup."""
    print("=" * 70)
    print("8. REPRODUCIBILITY")
    print("=" * 70)

    # Show that seeds produce identical results
    set_reproducibility(42)
    a1 = torch.randn(3)
    model = nn.Linear(10, 5)
    w1 = model.weight.clone()

    set_reproducibility(42)
    a2 = torch.randn(3)
    model2 = nn.Linear(10, 5)
    w2 = model2.weight.clone()

    print(f"  Same seed produces identical tensors: {torch.equal(a1, a2)}")
    print(f"  Same seed produces identical weights: {torch.equal(w1, w2)}")

    print(f"\n  Environment info:")
    for line in get_environment_info().split("\n"):
        print(f"    {line}")

    print()


# ============================================================================
# Main
# ============================================================================


def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║          Module 30: Debugging Toolkit — Full Demo                   ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    demo_anomaly_detection()
    demo_nan_detection()
    demo_gradient_flow()
    demo_shape_debugging()
    demo_device_checking()
    demo_common_errors()
    demo_memory_leak_detection()
    demo_reproducibility()

    print("=" * 70)
    print("ALL DEMOS COMPLETE")
    print("=" * 70)
    print()
    print("Key takeaways:")
    print("  1. Use detect_anomaly() to find NaN sources (debug only)")
    print("  2. Register NaN hooks on all modules for automatic detection")
    print("  3. Check gradient norms to detect vanishing/exploding gradients")
    print("  4. Use ShapeLogger hooks to trace dimension flow")
    print("  5. Always use loss.item() when logging — never store loss tensors")
    print("  6. Set all seeds + deterministic mode for reproducible bugs")
    print()


if __name__ == "__main__":
    main()
