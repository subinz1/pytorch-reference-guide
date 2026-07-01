"""
Module 35: Custom Dispatch — Registering Ops in the Dispatcher
==============================================================

Runnable on CPU. Demonstrates registering custom ops with torch.library,
autograd integration, torch.compile compatibility, and TorchDispatchMode.

Usage:
    python custom_dispatch.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode


def section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}\n")


# ============================================================================
# 1. Register a Custom Op with torch.library (Library API)
# ============================================================================

def demo_library_api():
    """Register a custom op using the classic Library API."""
    section("1. Custom Op with torch.library (Library API)")

    from torch.library import Library, impl

    # Define the op in a custom namespace
    lib = Library("demo35", "DEF")
    lib.define("scaled_add(Tensor x, Tensor y, float scale) -> Tensor")

    # CPU implementation
    @impl(lib, "scaled_add", "CPU")
    def scaled_add_cpu(x, y, scale):
        return x + y * scale

    # Meta implementation (shape inference for torch.compile/export)
    @impl(lib, "scaled_add", "Meta")
    def scaled_add_meta(x, y, scale):
        return torch.empty_like(x)

    # Use the op
    x = torch.randn(4, 4)
    y = torch.randn(4, 4)
    result = torch.ops.demo35.scaled_add(x, y, scale=2.0)

    print(f"torch.ops.demo35.scaled_add(x, y, scale=2.0)")
    print(f"  x shape: {x.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  result shape: {result.shape}")
    print(f"  Correct: {torch.allclose(result, x + y * 2.0)}")
    print()

    # Verify meta works
    x_meta = torch.randn(4, 4, device='meta')
    y_meta = torch.randn(4, 4, device='meta')
    result_meta = torch.ops.demo35.scaled_add(x_meta, y_meta, scale=2.0)
    print(f"Meta dispatch (shape inference):")
    print(f"  Result shape: {result_meta.shape}, device: {result_meta.device}")
    print()

    # Show the dispatch table
    print("Dispatch table for demo35::scaled_add:")
    try:
        table = torch._C._dispatch_dump("demo35::scaled_add")
        for line in table.strip().split('\n')[:15]:
            print(f"  {line}")
    except Exception as e:
        print(f"  (dump unavailable: {e})")
    print()

    return lib


# ============================================================================
# 2. Register Autograd for the Custom Op
# ============================================================================

def demo_autograd_registration(lib):
    """Register an autograd formula for the custom op."""
    section("2. Autograd Registration")

    from torch.library import Library, impl

    # Register autograd via torch.autograd.Function
    class ScaledAddAutograd(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, y, scale):
            ctx.scale = scale
            return torch.ops.demo35.scaled_add(x, y, scale)

        @staticmethod
        def backward(ctx, grad_output):
            # d/dx (x + y * scale) = 1
            # d/dy (x + y * scale) = scale
            grad_x = grad_output
            grad_y = grad_output * ctx.scale
            return grad_x, grad_y, None  # None for scale (not a tensor)

    def scaled_add_autograd(x, y, scale):
        return ScaledAddAutograd.apply(x, y, scale)

    # Register for AutogradCPU
    autograd_lib = Library("demo35", "IMPL")
    autograd_lib.impl("scaled_add", scaled_add_autograd, "AutogradCPU")

    # Test autograd
    x = torch.randn(4, 4, requires_grad=True)
    y = torch.randn(4, 4, requires_grad=True)

    result = torch.ops.demo35.scaled_add(x, y, scale=3.0)
    loss = result.sum()
    loss.backward()

    print("Autograd test:")
    print(f"  x.grad (should be all 1s): {x.grad[0, :4].tolist()}")
    print(f"  y.grad (should be all 3s): {y.grad[0, :4].tolist()}")
    print(f"  Correct x.grad: {torch.allclose(x.grad, torch.ones_like(x))}")
    print(f"  Correct y.grad: {torch.allclose(y.grad, torch.full_like(y, 3.0))}")
    print()

    return autograd_lib


# ============================================================================
# 3. @custom_op — The Modern API
# ============================================================================

def demo_custom_op_api():
    """Register a custom op using the modern @custom_op decorator."""
    section("3. @custom_op — The Modern API")

    @torch.library.custom_op("demo35_modern::fused_mul_add", mutates_args=())
    def fused_mul_add(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return x * y + z

    # Register fake (Meta) implementation
    @fused_mul_add.register_fake
    def fused_mul_add_fake(x, y, z):
        return torch.empty_like(x)

    # Register autograd
    def fused_mul_add_setup_context(ctx, inputs, output):
        x, y, z = inputs
        ctx.save_for_backward(x, y)

    def fused_mul_add_backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_output * y
        grad_y = grad_output * x
        grad_z = grad_output
        return grad_x, grad_y, grad_z

    fused_mul_add.register_autograd(
        fused_mul_add_backward,
        setup_context=fused_mul_add_setup_context,
    )

    # Test basic usage
    x = torch.randn(3, 3)
    y = torch.randn(3, 3)
    z = torch.randn(3, 3)
    result = torch.ops.demo35_modern.fused_mul_add(x, y, z)
    print(f"fused_mul_add(x, y, z) = x * y + z")
    print(f"  Result shape: {result.shape}")
    print(f"  Correct: {torch.allclose(result, x * y + z)}")
    print()

    # Test autograd
    x_g = torch.randn(3, 3, requires_grad=True)
    y_g = torch.randn(3, 3, requires_grad=True)
    z_g = torch.randn(3, 3, requires_grad=True)
    out = torch.ops.demo35_modern.fused_mul_add(x_g, y_g, z_g)
    out.sum().backward()
    print(f"Autograd test:")
    print(f"  x.grad matches y: {torch.allclose(x_g.grad, y_g.data)}")
    print(f"  y.grad matches x: {torch.allclose(y_g.grad, x_g.data)}")
    print(f"  z.grad is all 1s: {torch.allclose(z_g.grad, torch.ones_like(z_g))}")
    print()

    # Test meta dispatch
    x_m = torch.randn(5, 5, device='meta')
    y_m = torch.randn(5, 5, device='meta')
    z_m = torch.randn(5, 5, device='meta')
    out_m = torch.ops.demo35_modern.fused_mul_add(x_m, y_m, z_m)
    print(f"Meta dispatch: shape={out_m.shape}, device={out_m.device}")
    print()

    return fused_mul_add


# ============================================================================
# 4. torch.compile Compatibility
# ============================================================================

def demo_compile_compatibility():
    """Show custom ops working with torch.compile."""
    section("4. torch.compile Compatibility")

    @torch.library.custom_op("demo35_compile::gelu_approx", mutates_args=())
    def gelu_approx(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)

    @gelu_approx.register_fake
    def gelu_approx_fake(x):
        return torch.empty_like(x)

    def gelu_setup_context(ctx, inputs, output):
        x, = inputs
        ctx.save_for_backward(x)

    def gelu_backward(ctx, grad_output):
        x, = ctx.saved_tensors
        sig = torch.sigmoid(1.702 * x)
        grad = sig + 1.702 * x * sig * (1 - sig)
        return (grad_output * grad,)

    gelu_approx.register_autograd(gelu_backward, setup_context=gelu_setup_context)

    # Use in a model
    class SmallModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(16, 32)
            self.linear2 = nn.Linear(32, 16)

        def forward(self, x):
            x = self.linear1(x)
            x = torch.ops.demo35_compile.gelu_approx(x)
            x = self.linear2(x)
            return x

    model = SmallModel()
    x = torch.randn(4, 16)

    # Eager mode
    out_eager = model(x)
    print(f"Eager mode output shape: {out_eager.shape}")

    # Compiled mode
    compiled_model = torch.compile(model, backend="eager")
    out_compiled = compiled_model(x)
    print(f"Compiled mode output shape: {out_compiled.shape}")
    print(f"Outputs match: {torch.allclose(out_eager, out_compiled, atol=1e-6)}")
    print()

    # Verify gradients work through compile
    x_g = torch.randn(4, 16, requires_grad=True)
    loss = compiled_model(x_g).sum()
    loss.backward()
    print(f"Gradients through compiled model:")
    print(f"  x.grad shape: {x_g.grad.shape}")
    print(f"  x.grad has values: {x_g.grad.abs().mean().item():.6f} (mean abs)")
    print()


# ============================================================================
# 5. TorchDispatchMode — Intercepting All Dispatch
# ============================================================================

def demo_torch_dispatch_mode():
    """Use TorchDispatchMode to intercept dispatch-level operations."""
    section("5. TorchDispatchMode — Intercepting Dispatch")

    from torch.utils._python_dispatch import TorchDispatchMode

    class DispatchLogger(TorchDispatchMode):
        def __init__(self):
            self.log = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            self.log.append({
                'op': str(func),
                'arg_shapes': [a.shape for a in args if isinstance(a, torch.Tensor)],
            })
            return func(*args, **kwargs)

    # Trace a computation at the dispatch level
    logger = DispatchLogger()
    x = torch.randn(4, 4)

    print("Tracing: y = F.relu(x @ x.T + x)")
    with logger:
        y = F.relu(x @ x.T + x)

    print(f"  Ops dispatched ({len(logger.log)} total):")
    for i, entry in enumerate(logger.log, 1):
        shapes = [str(s) for s in entry['arg_shapes']]
        print(f"    {i:2d}. {entry['op']:<40s} shapes: {shapes}")
    print()

    # Count op types
    print("  Note: TorchDispatchMode sees ops AFTER autograd")
    print("  (it intercepts at a lower level than TorchFunctionMode)")
    print()


# ============================================================================
# 6. Dispatch Trace During Forward Pass
# ============================================================================

def demo_forward_pass_trace():
    """Trace all dispatch keys hit during a neural network forward pass."""
    section("6. Dispatch Trace During a Forward Pass")

    from torch.utils._python_dispatch import TorchDispatchMode

    class OpCounter(TorchDispatchMode):
        def __init__(self):
            self.op_counts = {}

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            name = func.name()
            self.op_counts[name] = self.op_counts.get(name, 0) + 1
            return func(*args, **kwargs)

    # Build a small model
    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.LayerNorm(64),
        nn.Linear(64, 10),
    )
    x = torch.randn(8, 32)

    counter = OpCounter()
    print("Model: Linear(32,64) → ReLU → Linear(64,64) → LayerNorm → Linear(64,10)")
    print(f"Input: batch=8, features=32\n")

    with counter:
        out = model(x)

    print(f"Forward pass dispatched {sum(counter.op_counts.values())} ops:")
    print(f"  {'Op':<45s} {'Count':>5s}")
    print(f"  {'-'*45} {'-'*5}")
    for op, count in sorted(counter.op_counts.items(), key=lambda x: -x[1]):
        print(f"  {op:<45s} {count:>5d}")
    print()
    print(f"  Output shape: {out.shape}")
    print()


# ============================================================================
# 7. Custom Op That Mutates Input
# ============================================================================

def demo_mutating_op():
    """Register a custom op that mutates its input tensor."""
    section("7. Custom Op with Mutation (mutates_args)")

    @torch.library.custom_op("demo35_mut::inplace_scale", mutates_args=("x",))
    def inplace_scale(x: torch.Tensor, factor: float) -> None:
        x.mul_(factor)

    @inplace_scale.register_fake
    def inplace_scale_fake(x, factor):
        pass  # No-op for shape inference (mutation happens in place)

    # Test in-place operation
    x = torch.ones(3, 3)
    print(f"Before: x[0,0] = {x[0, 0].item()}")
    torch.ops.demo35_mut.inplace_scale(x, factor=5.0)
    print(f"After inplace_scale(x, 5.0): x[0,0] = {x[0, 0].item()}")
    print(f"  Correct: {x[0, 0].item() == 5.0}")
    print()

    print("Note: mutates_args=('x',) tells the dispatcher that 'x' is modified.")
    print("This is critical for torch.compile to handle mutations correctly.")
    print()


# ============================================================================
# 8. Comparing TorchFunctionMode vs TorchDispatchMode
# ============================================================================

def demo_mode_comparison():
    """Compare the two interception points in the dispatcher."""
    section("8. TorchFunctionMode vs TorchDispatchMode")

    from torch.utils._python_dispatch import TorchDispatchMode

    class FunctionLogger(TorchFunctionMode):
        def __init__(self):
            self.ops = []

        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            self.ops.append(f"[Function] {func.__name__}" if hasattr(func, '__name__') else f"[Function] {func}")
            return func(*args, **kwargs)

    class DispatchLogger(TorchDispatchMode):
        def __init__(self):
            self.ops = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            self.ops.append(f"[Dispatch] {func.name()}")
            return func(*args, **kwargs)

    x = torch.randn(4, 4, requires_grad=True)

    # TorchFunctionMode — sees Python-level calls
    func_logger = FunctionLogger()
    with func_logger:
        y = torch.nn.functional.gelu(x)

    print("TorchFunctionMode (Python level, BEFORE dispatch):")
    for op in func_logger.ops[:10]:
        print(f"  {op}")
    print()

    # TorchDispatchMode — sees dispatch-level calls
    dispatch_logger = DispatchLogger()
    x2 = torch.randn(4, 4)
    with dispatch_logger:
        y2 = torch.nn.functional.gelu(x2)

    print("TorchDispatchMode (C++ level, AFTER autograd):")
    for op in dispatch_logger.ops[:10]:
        print(f"  {op}")
    print()

    print("Key differences:")
    print("  TorchFunctionMode: intercepts at Python boundary (highest level)")
    print("  TorchDispatchMode: intercepts at dispatch layer (below autograd)")
    print("  TorchFunctionMode sees: torch.add, F.gelu, etc.")
    print("  TorchDispatchMode sees: aten::gelu, aten::mul, etc.")
    print()


# ============================================================================
# 9. Building a Profiling Mode
# ============================================================================

def demo_profiling_mode():
    """Build a simple op profiler using TorchDispatchMode."""
    section("9. Building a Profiling Mode")

    import time
    from torch.utils._python_dispatch import TorchDispatchMode

    class SimpleProfiler(TorchDispatchMode):
        def __init__(self):
            self.timings = {}

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            name = func.name()
            start = time.perf_counter_ns()
            result = func(*args, **kwargs)
            elapsed_us = (time.perf_counter_ns() - start) / 1000
            if name not in self.timings:
                self.timings[name] = {'count': 0, 'total_us': 0.0}
            self.timings[name]['count'] += 1
            self.timings[name]['total_us'] += elapsed_us
            return result

    # Profile a model
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.GELU(),
        nn.Linear(256, 128),
        nn.LayerNorm(128),
    )
    x = torch.randn(32, 128)

    profiler = SimpleProfiler()
    with profiler:
        for _ in range(10):
            out = model(x)

    # Report
    print(f"Op profiling (10 forward passes, batch=32, input=128):")
    print(f"  {'Op':<40s} {'Calls':>6s} {'Total (us)':>12s} {'Avg (us)':>10s}")
    print(f"  {'-'*40} {'-'*6} {'-'*12} {'-'*10}")

    sorted_ops = sorted(profiler.timings.items(), key=lambda x: -x[1]['total_us'])
    for name, stats in sorted_ops[:15]:
        avg = stats['total_us'] / stats['count']
        print(f"  {name:<40s} {stats['count']:>6d} {stats['total_us']:>12.1f} {avg:>10.1f}")
    print()


# ============================================================================
# 10. Custom Op with Multiple Dispatch Keys
# ============================================================================

def demo_multi_backend():
    """Show an op with implementations for multiple backends."""
    section("10. Multi-Backend Custom Op")

    from torch.library import Library, impl

    lib = Library("demo35_multi", "DEF")
    lib.define("normalize(Tensor x) -> Tensor")

    @impl(lib, "normalize", "CPU")
    def normalize_cpu(x):
        mean = x.mean()
        std = x.std()
        return (x - mean) / (std + 1e-8)

    @impl(lib, "normalize", "Meta")
    def normalize_meta(x):
        return torch.empty_like(x)

    # If CUDA were available, we'd register a CUDA kernel too
    if torch.cuda.is_available():
        @impl(lib, "normalize", "CUDA")
        def normalize_cuda(x):
            mean = x.mean()
            std = x.std()
            return (x - mean) / (std + 1e-8)

    # Test CPU
    x_cpu = torch.randn(100)
    result_cpu = torch.ops.demo35_multi.normalize(x_cpu)
    print(f"CPU dispatch:")
    print(f"  Input mean: {x_cpu.mean():.4f}, std: {x_cpu.std():.4f}")
    print(f"  Output mean: {result_cpu.mean():.4f}, std: {result_cpu.std():.4f}")
    print(f"  (Should be ~0 mean, ~1 std)")
    print()

    # Test Meta
    x_meta = torch.randn(100, device='meta')
    result_meta = torch.ops.demo35_multi.normalize(x_meta)
    print(f"Meta dispatch:")
    print(f"  Input: shape={x_meta.shape}, device={x_meta.device}")
    print(f"  Output: shape={result_meta.shape}, device={result_meta.device}")
    print()

    # Show dispatch table
    print("Dispatch table:")
    try:
        table = torch._C._dispatch_dump("demo35_multi::normalize")
        for line in table.strip().split('\n')[:10]:
            print(f"  {line}")
    except Exception as e:
        print(f"  (dump unavailable: {e})")
    print()

    return lib


# ============================================================================
# 11. Exercise: Fused Add-ReLU Op
# ============================================================================

def demo_fused_add_relu():
    """Exercise: implement a fused add + relu op with full dispatch support."""
    section("11. Exercise: Fused Add-ReLU Custom Op")

    print("Implementing fused_add_relu: relu(x + y)")
    print("With CPU, Meta, and Autograd support.\n")

    @torch.library.custom_op("demo35_exercise::fused_add_relu", mutates_args=())
    def fused_add_relu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + y)

    @fused_add_relu.register_fake
    def fused_add_relu_fake(x, y):
        return torch.empty_like(x)

    def fused_add_relu_setup_context(ctx, inputs, output):
        x, y = inputs
        ctx.save_for_backward(output)  # Save result to check > 0

    def fused_add_relu_backward(ctx, grad_output):
        output, = ctx.saved_tensors
        grad = grad_output * (output > 0).float()
        return grad, grad  # Same gradient for both x and y

    fused_add_relu.register_autograd(
        fused_add_relu_backward,
        setup_context=fused_add_relu_setup_context,
    )

    # Test correctness
    x = torch.randn(4, 4)
    y = torch.randn(4, 4)
    result = torch.ops.demo35_exercise.fused_add_relu(x, y)
    expected = torch.relu(x + y)
    print(f"  Correctness: {torch.allclose(result, expected)}")

    # Test autograd
    x_g = torch.randn(4, 4, requires_grad=True)
    y_g = torch.randn(4, 4, requires_grad=True)
    out = torch.ops.demo35_exercise.fused_add_relu(x_g, y_g)
    out.sum().backward()
    print(f"  Autograd works: x.grad shape={x_g.grad.shape}")

    # Verify gradient correctness
    mask = (x_g.data + y_g.data > 0).float()
    print(f"  Gradient correct: {torch.allclose(x_g.grad, mask)}")
    print()

    # Test with torch.compile
    @torch.compile(backend="eager")
    def compiled_fn(a, b):
        return torch.ops.demo35_exercise.fused_add_relu(a, b)

    result_compiled = compiled_fn(x, y)
    print(f"  torch.compile compatible: {torch.allclose(result_compiled, expected)}")
    print()

    # Show it in a model
    class ModelWithFusedOp(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(16, 16)
            self.linear2 = nn.Linear(16, 16)
            self.output = nn.Linear(16, 4)

        def forward(self, x):
            h1 = self.linear1(x)
            h2 = self.linear2(x)
            fused = torch.ops.demo35_exercise.fused_add_relu(h1, h2)
            return self.output(fused)

    model = ModelWithFusedOp()
    inp = torch.randn(8, 16)
    out = model(inp)
    loss = out.sum()
    loss.backward()
    print(f"  In model: output shape={out.shape}")
    print(f"  Gradients flow: {model.linear1.weight.grad is not None}")
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print(" Module 35: Custom Dispatch — Registering Ops in the Dispatcher")
    print(" " + "=" * 68)
    print(f" PyTorch version: {torch.__version__}")
    print(f" CUDA available: {torch.cuda.is_available()}")
    print("=" * 70)

    lib = demo_library_api()
    autograd_lib = demo_autograd_registration(lib)
    demo_custom_op_api()
    demo_compile_compatibility()
    demo_torch_dispatch_mode()
    demo_forward_pass_trace()
    demo_mutating_op()
    demo_mode_comparison()
    demo_profiling_mode()
    multi_lib = demo_multi_backend()
    demo_fused_add_relu()

    section("Summary")
    print("Custom dispatch registration patterns:")
    print("  1. Library API: Library('ns', 'DEF') + @impl for each key")
    print("  2. @custom_op: Decorator with register_fake + register_autograd")
    print("  3. TorchDispatchMode: Intercept all ops at dispatch level")
    print("  4. TorchFunctionMode: Intercept at Python level (higher)")
    print()
    print("For new ops, prefer @custom_op — less boilerplate, better integration.")
    print("For debugging/profiling, use TorchDispatchMode.")
    print()


if __name__ == "__main__":
    main()
