"""
Module 35: Dispatch Keys — Exploring PyTorch's Dispatcher
=========================================================

Runnable on CPU. Demonstrates how to inspect dispatch keys, priority chains,
and dispatch tables for PyTorch operations.

Usage:
    python dispatch_keys.py
"""

import torch
import torch.nn as nn
from contextlib import contextmanager


def section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}\n")


# ============================================================================
# 1. Dispatch Keys on Different Tensors
# ============================================================================

def explore_dispatch_keys():
    """Show how dispatch keys vary based on tensor properties."""
    section("1. Dispatch Keys on Different Tensors")

    # Basic CPU tensor
    x_cpu = torch.randn(3, 3)
    print(f"CPU tensor (no grad):")
    print(f"  Keys: {torch._C._dispatch_keys(x_cpu)}")
    print()

    # CPU tensor with requires_grad
    x_grad = torch.randn(3, 3, requires_grad=True)
    print(f"CPU tensor (requires_grad=True):")
    print(f"  Keys: {torch._C._dispatch_keys(x_grad)}")
    print()

    # Show that requires_grad adds AutogradCPU
    x_no_grad = torch.randn(3, 3)
    print(f"Before requires_grad_(): {torch._C._dispatch_keys(x_no_grad)}")
    x_no_grad.requires_grad_(True)
    print(f"After requires_grad_():  {torch._C._dispatch_keys(x_no_grad)}")
    print()

    # Meta tensor
    x_meta = torch.randn(3, 3, device='meta')
    print(f"Meta tensor:")
    print(f"  Keys: {torch._C._dispatch_keys(x_meta)}")
    print()

    # Meta tensor with grad
    x_meta_grad = torch.randn(3, 3, device='meta', requires_grad=True)
    print(f"Meta tensor (requires_grad=True):")
    print(f"  Keys: {torch._C._dispatch_keys(x_meta_grad)}")
    print()

    # CUDA tensor (if available)
    if torch.cuda.is_available():
        x_cuda = torch.randn(3, 3, device='cuda')
        print(f"CUDA tensor (no grad):")
        print(f"  Keys: {torch._C._dispatch_keys(x_cuda)}")
        print()

        x_cuda_grad = torch.randn(3, 3, device='cuda', requires_grad=True)
        print(f"CUDA tensor (requires_grad=True):")
        print(f"  Keys: {torch._C._dispatch_keys(x_cuda_grad)}")
        print()
    else:
        print("CUDA not available — skipping CUDA tensor examples")
        print()

    # Integer tensor (no autograd support)
    x_int = torch.randint(0, 10, (3, 3))
    print(f"Integer tensor (int64):")
    print(f"  Keys: {torch._C._dispatch_keys(x_int)}")
    print("  (No Autograd key — integer types don't support gradients)")
    print()


# ============================================================================
# 2. Dispatch Keys Inside Special Contexts
# ============================================================================

def explore_context_keys():
    """Show how context managers add dispatch keys."""
    section("2. Dispatch Keys Inside Special Contexts")

    x = torch.randn(3, 3)
    print(f"Normal context: {torch._C._dispatch_keys(x)}")

    # autocast changes how ops are dispatched but doesn't change tensor keys
    # The Autocast key is added via thread-local state
    print(f"\nNote: Autocast and other context-based keys are thread-local")
    print(f"They're added to the dispatch key set at call time, not stored on tensors")
    print()

    # Demonstrate with torch.inference_mode
    print(f"Outside inference_mode:")
    print(f"  x.requires_grad_(True)")
    x.requires_grad_(True)
    print(f"  Keys: {torch._C._dispatch_keys(x)}")

    with torch.inference_mode():
        y = torch.randn(3, 3)
        print(f"\nInside inference_mode:")
        print(f"  New tensor keys: {torch._C._dispatch_keys(y)}")
        print(f"  (InferenceMode tensors have no Autograd keys)")
    print()


# ============================================================================
# 3. The Full Dispatch Key Set
# ============================================================================

def explore_full_keyset():
    """List all available dispatch keys."""
    section("3. Full Dispatch Key Set")

    print("All dispatch keys in PyTorch:")
    print("-" * 50)

    # Get the full keyset
    full_keyset = torch._C._dispatch_keyset_full()
    print(f"Full keyset: {full_keyset}")
    print()

    # Categorize keys
    backend_keys = [
        "CPU", "CUDA", "MPS", "XPU", "Meta", "SparseCPU", "SparseCUDA",
        "QuantizedCPU", "QuantizedCUDA",
    ]
    autograd_keys = [
        "AutogradCPU", "AutogradCUDA", "AutogradMPS", "AutogradXPU",
        "AutogradMeta", "AutogradOther",
    ]
    feature_keys = [
        "Autocast", "FuncTorchBatched", "FuncTorchVmapMode",
        "Functionalize", "ADInplaceOrView", "BackendSelect",
    ]
    composite_keys = [
        "CompositeImplicitAutograd", "CompositeImplicitAutogradNestedTensor",
        "CompositeExplicitAutograd", "CompositeExplicitAutogradNonFunctional",
    ]

    print("Backend keys (actual compute):")
    for k in backend_keys:
        print(f"  - {k}")
    print()

    print("Autograd keys (record for backward):")
    for k in autograd_keys:
        print(f"  - {k}")
    print()

    print("Feature keys (transforms/modes):")
    for k in feature_keys:
        print(f"  - {k}")
    print()

    print("Composite keys (decomposition strategies):")
    for k in composite_keys:
        print(f"  - {k}")
    print()


# ============================================================================
# 4. Dumping Dispatch Tables
# ============================================================================

def explore_dispatch_tables():
    """Dump dispatch tables for common operations."""
    section("4. Dispatch Tables for Common Ops")

    ops_to_inspect = [
        "aten::add.Tensor",
        "aten::mm",
        "aten::relu",
    ]

    for op_name in ops_to_inspect:
        print(f"--- {op_name} ---")
        try:
            table = torch._C._dispatch_dump(op_name)
            # Print first 30 lines to keep output manageable
            lines = table.strip().split('\n')
            for line in lines[:30]:
                print(f"  {line}")
            if len(lines) > 30:
                print(f"  ... ({len(lines) - 30} more lines)")
        except Exception as e:
            print(f"  Error: {e}")
        print()


# ============================================================================
# 5. Priority Chain for a Specific Op
# ============================================================================

def explore_priority_chain():
    """Demonstrate the priority chain for dispatch."""
    section("5. Priority Chain Demonstration")

    print("When torch.add(x, y) is called with requires_grad=True CPU tensors:")
    print()
    print("  Key set on tensors: {CPU, AutogradCPU}")
    print()
    print("  Priority walk (high to low):")
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │ 1. AutogradCPU  → HAS KERNEL → Execute             │")
    print("  │    (saves for backward, creates grad_fn)            │")
    print("  │    Then redispatches with AutogradCPU excluded      │")
    print("  │                                                     │")
    print("  │ 2. CPU          → HAS KERNEL → Execute             │")
    print("  │    (performs actual addition on CPU)                 │")
    print("  │    Returns result                                   │")
    print("  └─────────────────────────────────────────────────────┘")
    print()

    # Verify by checking kernel registrations
    op = "aten::add.Tensor"
    for key in ["AutogradCPU", "CPU", "CompositeImplicitAutograd"]:
        has_kernel = torch._C._dispatch_has_kernel_for_dispatch_key(op, key)
        print(f"  {op} has kernel for {key}: {has_kernel}")
    print()

    # Compare with an op that uses CompositeImplicit
    print("For torch.addcmul (CompositeImplicitAutograd):")
    op2 = "aten::addcmul"
    for key in ["CPU", "CUDA", "CompositeImplicitAutograd"]:
        try:
            has_kernel = torch._C._dispatch_has_kernel_for_dispatch_key(op2, key)
            print(f"  {op2} has kernel for {key}: {has_kernel}")
        except Exception:
            print(f"  {op2} — key {key}: (check not available)")
    print()


# ============================================================================
# 6. Fallthrough Demonstration
# ============================================================================

def explore_fallthrough():
    """Show how ops fall through to CompositeImplicit when no backend kernel."""
    section("6. Fallthrough to CompositeImplicitAutograd")

    print("Some ops don't have backend-specific kernels.")
    print("They fall through to CompositeImplicitAutograd, which decomposes them.")
    print()

    # Find ops that are CompositeImplicit only
    composite_ops = [
        "aten::mish",
        "aten::selu",
        "aten::celu",
    ]

    for op_name in composite_ops:
        print(f"  {op_name}:")
        for key in ["CPU", "CUDA", "CompositeImplicitAutograd"]:
            try:
                has_kernel = torch._C._dispatch_has_kernel_for_dispatch_key(op_name, key)
                status = "REGISTERED" if has_kernel else "fallthrough"
                print(f"    {key:35s} → {status}")
            except Exception:
                print(f"    {key:35s} → (unavailable)")
        print()

    print("When a CompositeImplicit op is called, it decomposes into primitives:")
    print("  torch.selu(x) → x * alpha * (where condition selects path)")
    print("  The primitives (mul, where) DO have CPU/CUDA kernels")
    print()


# ============================================================================
# 7. Comparing Dispatch Behavior
# ============================================================================

def compare_dispatch_behavior():
    """Compare how different tensor types affect dispatch."""
    section("7. Comparing Dispatch Behavior Across Tensor Types")

    scenarios = [
        ("float32, no grad", torch.randn(4, 4)),
        ("float32, grad", torch.randn(4, 4, requires_grad=True)),
        ("float64, no grad", torch.randn(4, 4, dtype=torch.float64)),
        ("int32", torch.randint(0, 10, (4, 4), dtype=torch.int32)),
        ("bool", torch.ones(4, 4, dtype=torch.bool)),
        ("meta device", torch.randn(4, 4, device='meta')),
    ]

    print(f"{'Scenario':<25s} {'Dispatch Keys'}")
    print("-" * 70)
    for desc, tensor in scenarios:
        keys = str(torch._C._dispatch_keys(tensor))
        print(f"{desc:<25s} {keys}")
    print()

    # Show key set union for binary ops
    print("\nKey set union for binary operations:")
    print("-" * 70)
    x = torch.randn(4, 4, requires_grad=True)
    y = torch.randn(4, 4)
    print(f"x keys (requires_grad=True): {torch._C._dispatch_keys(x)}")
    print(f"y keys (no grad):            {torch._C._dispatch_keys(y)}")
    print(f"Union used by dispatcher:    Both Autograd AND backend keys present")
    print(f"Result: Autograd kernel runs first, then redispatches to CPU")
    print()


# ============================================================================
# 8. Meta Tensor Dispatch (Shape Inference)
# ============================================================================

def explore_meta_dispatch():
    """Show how Meta tensors use dispatch for shape inference."""
    section("8. Meta Tensor Dispatch (Shape-Only Computation)")

    print("Meta tensors have no data — only shape, dtype, and device metadata.")
    print("The Meta dispatch key routes to kernels that compute output shapes.")
    print()

    # Create meta tensors
    a = torch.randn(3, 4, device='meta')
    b = torch.randn(4, 5, device='meta')

    print(f"a: shape={a.shape}, device={a.device}, storage_size=0")
    print(f"b: shape={b.shape}, device={b.device}, storage_size=0")
    print()

    # Operations compute shapes without data
    c = torch.mm(a, b)
    print(f"torch.mm(a, b): shape={c.shape}, device={c.device}")
    print(f"  → Meta kernel computed output shape [3, 5] from inputs [3,4] @ [4,5]")
    print()

    d = torch.cat([a, torch.randn(2, 4, device='meta')], dim=0)
    print(f"torch.cat([a, ...], dim=0): shape={d.shape}")
    print()

    e = a.unsqueeze(0).expand(8, 3, 4)
    print(f"a.unsqueeze(0).expand(8,3,4): shape={e.shape}")
    print()

    # This is exactly what torch.compile uses for tracing
    print("This is how torch.compile traces shapes through your model")
    print("without executing actual computation — it uses Meta dispatch!")
    print()


# ============================================================================
# 9. BackendSelect Key
# ============================================================================

def explore_backend_select():
    """Show how BackendSelect routes factory ops."""
    section("9. BackendSelect — Routing Factory Operations")

    print("Factory ops like torch.randn() have NO input tensors.")
    print("The dispatcher can't infer the backend from inputs.")
    print("BackendSelect uses the `device` argument to route correctly.")
    print()

    # BackendSelect has a kernel for factory ops
    factory_ops = [
        "aten::empty.memory_format",
        "aten::randn",
    ]

    for op_name in factory_ops:
        try:
            has_bs = torch._C._dispatch_has_kernel_for_dispatch_key(
                op_name, "BackendSelect"
            )
            print(f"  {op_name:40s} BackendSelect: {has_bs}")
        except Exception:
            print(f"  {op_name:40s} (check unavailable)")

    # Non-factory ops don't need BackendSelect
    non_factory = ["aten::add.Tensor", "aten::mm"]
    print()
    for op_name in non_factory:
        try:
            has_bs = torch._C._dispatch_has_kernel_for_dispatch_key(
                op_name, "BackendSelect"
            )
            print(f"  {op_name:40s} BackendSelect: {has_bs}")
        except Exception:
            print(f"  {op_name:40s} (check unavailable)")
    print()
    print("  Non-factory ops infer backend from input tensor keys directly.")
    print()


# ============================================================================
# 10. Dispatch Key Ordering Verification
# ============================================================================

def verify_key_ordering():
    """Verify the priority ordering of dispatch keys."""
    section("10. Dispatch Key Priority Ordering")

    print("Dispatch keys have a fixed priority ordering.")
    print("Higher priority keys get first chance to handle an op.")
    print()

    # We can check ordering by looking at the enum values
    keys_in_priority_order = [
        ("PythonTLSSnapshot", "Highest — thread-local state"),
        ("PythonDispatcher", "torch.compile tracing"),
        ("Functionalize", "Convert mutations to functional"),
        ("Autocast", "Mixed precision casting"),
        ("AutogradCPU", "Record op for backward (CPU)"),
        ("AutogradCUDA", "Record op for backward (CUDA)"),
        ("ADInplaceOrView", "Track views/in-place for autograd"),
        ("BackendSelect", "Route factory ops"),
        ("CPU", "Actual CPU computation"),
        ("CUDA", "Actual CUDA computation"),
        ("Meta", "Shape-only computation"),
        ("CompositeImplicitAutograd", "Decompose into primitives"),
        ("CompositeExplicitAutograd", "Decompose with custom grad"),
    ]

    print(f"  {'Key':<35s} {'Purpose'}")
    print(f"  {'-'*35} {'-'*40}")
    for key, purpose in keys_in_priority_order:
        print(f"  {key:<35s} {purpose}")
    print()
    print("  When multiple keys are present, the highest-priority key with")
    print("  a registered kernel executes first, then may redispatch.")
    print()


# ============================================================================
# 11. Tracing Dispatch with torch.overrides
# ============================================================================

def trace_dispatch():
    """Use torch.overrides to see what ops are called."""
    section("11. Tracing Operations Through Dispatch")

    from torch.overrides import TorchFunctionMode

    class DispatchTracer(TorchFunctionMode):
        def __init__(self):
            self.ops_seen = []

        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            self.ops_seen.append(func.__name__ if hasattr(func, '__name__') else str(func))
            return func(*args, **kwargs)

    # Trace a simple computation
    print("Tracing: y = (x * 2 + 1).relu()")
    tracer = DispatchTracer()
    x = torch.randn(4, 4)
    with tracer:
        y = (x * 2 + 1).relu()

    print(f"  Operations dispatched:")
    for i, op in enumerate(tracer.ops_seen, 1):
        print(f"    {i}. {op}")
    print()

    # Trace a more complex computation
    print("Tracing: z = F.layer_norm(x, [4])")
    tracer2 = DispatchTracer()
    with tracer2:
        z = torch.nn.functional.layer_norm(x, [4])

    print(f"  Operations dispatched:")
    for i, op in enumerate(tracer2.ops_seen, 1):
        print(f"    {i}. {op}")
    print(f"  → layer_norm decomposes into {len(tracer2.ops_seen)} primitive ops")
    print()


# ============================================================================
# 12. Dispatch and Autograd Interaction
# ============================================================================

def explore_autograd_dispatch():
    """Show how autograd interacts with dispatch."""
    section("12. Autograd and Dispatch Interaction")

    x = torch.randn(3, 3, requires_grad=True)
    y = torch.randn(3, 3, requires_grad=True)

    print("Computing z = x @ y + x with requires_grad=True inputs:")
    print(f"  x keys: {torch._C._dispatch_keys(x)}")
    print(f"  y keys: {torch._C._dispatch_keys(y)}")
    print()

    z = x @ y + x

    print(f"  Result z: shape={z.shape}, requires_grad={z.requires_grad}")
    print(f"  z.grad_fn: {z.grad_fn}")
    print(f"  z.grad_fn chain: {z.grad_fn} → {z.grad_fn.next_functions}")
    print()

    print("  What happened in the dispatcher:")
    print("  1. torch.mm(x, y):")
    print("     - AutogradCPU kernel: save x, y; create MmBackward0")
    print("     - Redispatch → CPU kernel: compute x @ y")
    print("  2. torch.add(mm_result, x):")
    print("     - AutogradCPU kernel: create AddBackward0")
    print("     - Redispatch → CPU kernel: compute addition")
    print()

    # Show that no_grad skips autograd dispatch
    print("With torch.no_grad():")
    with torch.no_grad():
        w = x @ y + x
        print(f"  w.requires_grad: {w.requires_grad}")
        print(f"  w.grad_fn: {w.grad_fn}")
        print(f"  → AutogradCPU key is excluded from dispatch!")
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print(" Module 35: Exploring PyTorch Dispatch Keys")
    print(" " + "=" * 68)
    print(f" PyTorch version: {torch.__version__}")
    print(f" CUDA available: {torch.cuda.is_available()}")
    print("=" * 70)

    explore_dispatch_keys()
    explore_context_keys()
    explore_full_keyset()
    explore_dispatch_tables()
    explore_priority_chain()
    explore_fallthrough()
    compare_dispatch_behavior()
    explore_meta_dispatch()
    explore_backend_select()
    verify_key_ordering()
    trace_dispatch()
    explore_autograd_dispatch()

    section("Summary")
    print("Key takeaways:")
    print("  1. Every tensor carries a dispatch key set (bitset)")
    print("  2. Keys come from: device, requires_grad, thread-local state")
    print("  3. The dispatcher walks keys high→low, first registered kernel wins")
    print("  4. Higher keys redispatch to lower keys after their work")
    print("  5. CompositeImplicit provides fallback decompositions")
    print("  6. Meta dispatch enables shape inference without compute")
    print("  7. BackendSelect routes factory ops by device argument")
    print("  8. torch.no_grad() works by excluding Autograd keys from dispatch")
    print()


if __name__ == "__main__":
    main()
