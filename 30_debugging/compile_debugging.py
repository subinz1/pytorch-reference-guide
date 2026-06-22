"""
Module 30: Debugging torch.compile
====================================

Runnable on CPU. Demonstrates:
- Graph break detection with explain()
- Finding and fixing graph breaks
- CompileCounter to verify compilation
- EagerAndRecordGraphs backend for inspection
- Verbose mode and TORCH_LOGS
- Recompilation detection
- Common torch.compile errors and fixes

Run: python compile_debugging.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo as dynamo


# ============================================================================
# 1. Graph Break Detection with explain()
# ============================================================================


def demo_explain():
    """Use torch._dynamo.explain() to find graph breaks."""
    print("=" * 70)
    print("1. GRAPH BREAK DETECTION WITH explain()")
    print("=" * 70)

    # Function with a graph break
    def fn_with_break(x):
        x = x * 2
        print("debug:", x.shape)  # Graph break!
        x = x + 1
        return x

    # Function without a graph break
    def fn_clean(x):
        x = x * 2
        x = x + 1
        return x

    print("\n  Function WITH graph break (print statement):")
    dynamo.reset()
    explanation = dynamo.explain(fn_with_break)(torch.randn(10))
    print(f"    Graph break count: {explanation.graph_break_count}")
    print(f"    Graph count: {explanation.graph_count}")
    if explanation.break_reasons:
        for reason in explanation.break_reasons:
            print(f"    Reason: {reason.reason}")

    print("\n  Function WITHOUT graph break:")
    dynamo.reset()
    explanation = dynamo.explain(fn_clean)(torch.randn(10))
    print(f"    Graph break count: {explanation.graph_break_count}")
    print(f"    Graph count: {explanation.graph_count}")

    print()


# ============================================================================
# 2. Finding and Fixing Graph Breaks
# ============================================================================


def demo_fixing_graph_breaks():
    """Common graph break causes and their fixes."""
    print("=" * 70)
    print("2. FINDING AND FIXING GRAPH BREAKS")
    print("=" * 70)

    # Example 1: print() causes graph break
    print("\n  Example 1: print() → remove or guard")

    def bad_fn(x):
        x = x * 2
        print(x.shape)  # Break
        return x + 1

    def fixed_fn(x):
        x = x * 2
        # Guarded: won't execute during compilation
        if not torch.compiler.is_compiling():
            print(x.shape)
        return x + 1

    dynamo.reset()
    exp_bad = dynamo.explain(bad_fn)(torch.randn(5))
    dynamo.reset()
    exp_fixed = dynamo.explain(fixed_fn)(torch.randn(5))

    print(f"    Bad:   {exp_bad.graph_break_count} break(s)")
    print(f"    Fixed: {exp_fixed.graph_break_count} break(s)")

    # Example 2: data-dependent control flow
    print("\n  Example 2: Data-dependent if → torch.where")

    def bad_control(x):
        if x.sum() > 0:  # Data-dependent branch
            return x * 2
        return x * 3

    def fixed_control(x):
        return torch.where(x.sum() > 0, x * 2, x * 3)

    dynamo.reset()
    exp_bad = dynamo.explain(bad_control)(torch.randn(5))
    dynamo.reset()
    exp_fixed = dynamo.explain(fixed_control)(torch.randn(5))

    print(f"    Bad:   {exp_bad.graph_break_count} break(s)")
    print(f"    Fixed: {exp_fixed.graph_break_count} break(s)")

    # Example 3: Python builtin on tensor
    print("\n  Example 3: sorted() on list → use torch.sort")

    def bad_sort(x):
        values = x.tolist()  # Break: converting to Python
        return torch.tensor(sorted(values))

    def fixed_sort(x):
        return torch.sort(x).values

    dynamo.reset()
    exp_bad = dynamo.explain(bad_sort)(torch.randn(5))
    dynamo.reset()
    exp_fixed = dynamo.explain(fixed_sort)(torch.randn(5))

    print(f"    Bad:   {exp_bad.graph_break_count} break(s)")
    print(f"    Fixed: {exp_fixed.graph_break_count} break(s)")

    print()


# ============================================================================
# 3. CompileCounter — Verify Compilation Happens
# ============================================================================


class CompileCounter:
    """Backend that counts how many times compilation is triggered."""

    def __init__(self):
        self.frame_count = 0
        self.op_count = 0
        self.graphs = []

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        self.frame_count += 1
        self.op_count += len([n for n in gm.graph.nodes if n.op == "call_function"])
        self.graphs.append(gm)
        return gm.forward


def demo_compile_counter():
    """Use CompileCounter to verify compilation behavior."""
    print("=" * 70)
    print("3. COMPILE COUNTER")
    print("=" * 70)

    counter = CompileCounter()
    dynamo.reset()

    @torch.compile(backend=counter)
    def my_fn(x, y):
        return x * y + torch.sin(x)

    # First call triggers compilation
    x = torch.randn(10)
    y = torch.randn(10)
    _ = my_fn(x, y)

    print(f"  After first call:")
    print(f"    Compilations: {counter.frame_count}")
    print(f"    Ops captured: {counter.op_count}")

    # Same shapes: no recompilation
    _ = my_fn(torch.randn(10), torch.randn(10))
    print(f"\n  After second call (same shapes):")
    print(f"    Compilations: {counter.frame_count}")

    # Different shapes: may trigger recompilation
    _ = my_fn(torch.randn(20), torch.randn(20))
    print(f"\n  After third call (different shapes):")
    print(f"    Compilations: {counter.frame_count}")

    print()


# ============================================================================
# 4. EagerAndRecordGraphs — Inspect Captured Graphs
# ============================================================================


def demo_record_graphs():
    """Inspect the FX graphs captured by Dynamo."""
    print("=" * 70)
    print("4. RECORD AND INSPECT GRAPHS")
    print("=" * 70)

    counter = CompileCounter()
    dynamo.reset()

    class SmallModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)

        def forward(self, x):
            x = self.linear(x)
            x = F.relu(x)
            return x.sum()

    model = SmallModel()
    compiled_model = torch.compile(model, backend=counter)
    _ = compiled_model(torch.randn(2, 10))

    print(f"  Captured {len(counter.graphs)} graph(s)")
    if counter.graphs:
        print(f"\n  Graph 0 structure:")
        for node in counter.graphs[0].graph.nodes:
            if node.op != "placeholder" and node.op != "output":
                print(f"    {node.op}: {node.target} → {node.name}")

    print()


# ============================================================================
# 5. TORCH_LOGS Explanation
# ============================================================================


def demo_torch_logs():
    """Explain TORCH_LOGS environment variable options."""
    print("=" * 70)
    print("5. TORCH_LOGS ENVIRONMENT VARIABLE")
    print("=" * 70)

    logs_table = {
        "graph_breaks": "Show where and why graph breaks occur",
        "recompiles": "Show when recompilation is triggered and why",
        "dynamo": "Full Dynamo tracing logs",
        "inductor": "Inductor compilation details",
        "output_code": "Show generated Triton/C++ code",
        "guards": "Show guard expressions for each compiled frame",
        "aot": "AOTAutograd decomposition logs",
        "schedule": "Inductor scheduling decisions",
    }

    print("\n  Usage: TORCH_LOGS=\"<option>\" python script.py")
    print(f"\n  {'Option':<20} {'Description'}")
    print(f"  {'-'*20} {'-'*50}")
    for option, desc in logs_table.items():
        print(f"  {option:<20} {desc}")

    print("\n  Combine: TORCH_LOGS=\"graph_breaks,recompiles\"")
    print("  Verbose: TORCH_LOGS=\"+dynamo\" (+ prefix = DEBUG level)")

    # Programmatic equivalent
    print("\n  Programmatic equivalent:")
    print("    import logging")
    print("    torch._logging.set_logs(graph_breaks=True)")

    print()


# ============================================================================
# 6. Recompilation Detection
# ============================================================================


def demo_recompilation():
    """Detect and understand recompilation triggers."""
    print("=" * 70)
    print("6. RECOMPILATION DETECTION")
    print("=" * 70)

    counter = CompileCounter()
    dynamo.reset()

    @torch.compile(backend=counter)
    def dynamic_fn(x):
        return x * 2 + 1

    # Multiple different shapes trigger recompilation
    shapes = [(5,), (10,), (15,), (20,), (25,)]
    print("\n  Calling with different shapes:")
    for shape in shapes:
        _ = dynamic_fn(torch.randn(shape))
        print(f"    Shape {str(shape):<8} → total compilations: {counter.frame_count}")

    print(f"\n  {counter.frame_count} compilations for {len(shapes)} unique shapes")
    print("  Fix: Use dynamic shapes with torch.compile(dynamic=True)")

    # With dynamic=True
    counter2 = CompileCounter()
    dynamo.reset()

    @torch.compile(backend=counter2, dynamic=True)
    def dynamic_fn2(x):
        return x * 2 + 1

    print("\n  With dynamic=True:")
    for shape in shapes:
        _ = dynamic_fn2(torch.randn(shape))
        print(f"    Shape {str(shape):<8} → total compilations: {counter2.frame_count}")

    print(f"\n  Only {counter2.frame_count} compilation(s) with dynamic=True!")
    print()


# ============================================================================
# 7. Common torch.compile Errors and Fixes
# ============================================================================


def demo_common_compile_errors():
    """Demonstrate common torch.compile issues and solutions."""
    print("=" * 70)
    print("7. COMMON torch.compile ERRORS AND FIXES")
    print("=" * 70)

    # Error 1: Graph break from data-dependent shape
    print("\n  Issue 1: Non-tensor return values from compiled function")
    dynamo.reset()

    def returns_int(x):
        return x.sum(), x.shape[0]  # int return is fine since PT 2.1+

    counter = CompileCounter()
    compiled = torch.compile(returns_int, backend=counter)
    result = compiled(torch.randn(5, 3))
    print(f"    Compiled successfully: {counter.frame_count} frame(s)")
    print(f"    Result: tensor={result[0].item():.4f}, int={result[1]}")

    # Error 2: Mutation of inputs
    print("\n  Issue 2: Compiling functions that mutate inputs")
    dynamo.reset()

    def mutating_fn(x):
        x.add_(1)  # In-place mutation
        return x * 2

    counter2 = CompileCounter()
    try:
        compiled2 = torch.compile(mutating_fn, backend=counter2)
        x = torch.randn(5)
        result = compiled2(x)
        print(f"    Input mutation compiled OK (Dynamo handles copy-on-write)")
        print(f"    Compilations: {counter2.frame_count}")
    except Exception as e:
        print(f"    Error: {e}")

    # Error 3: Calling .item() or .numpy()
    print("\n  Issue 3: Calling .item() in compiled code")
    dynamo.reset()

    def uses_item(x):
        val = x.sum().item()  # Graph break: escapes to Python
        return x * val

    exp = dynamo.explain(uses_item)(torch.randn(5))
    print(f"    Graph breaks: {exp.graph_break_count}")
    print("    Fix: Keep computation in tensor-land, avoid .item()")

    # Error 4: Dynamic shapes with assertions
    print("\n  Issue 4: Assertions on shapes")
    dynamo.reset()

    def with_assert(x):
        assert x.shape[0] > 0, "empty batch"
        return x * 2

    counter3 = CompileCounter()
    compiled3 = torch.compile(with_assert, backend=counter3)
    _ = compiled3(torch.randn(5))
    print(f"    assert compiled OK (becomes guard): {counter3.frame_count} frame(s)")

    print()


# ============================================================================
# 8. Debugging Workflow Summary
# ============================================================================


def print_debugging_workflow():
    """Print the recommended torch.compile debugging workflow."""
    print("=" * 70)
    print("TORCH.COMPILE DEBUGGING WORKFLOW")
    print("=" * 70)
    print("""
  Step 1: Verify it works in eager mode
    model(input)  # No torch.compile

  Step 2: Compile and check for errors
    compiled = torch.compile(model)
    compiled(input)

  Step 3: If errors, check graph breaks
    TORCH_LOGS="graph_breaks" python script.py
    # or
    torch._dynamo.explain(model)(input)

  Step 4: If wrong results, compare against eager
    eager_out = model(input)
    compiled_out = compiled(input)
    assert torch.allclose(eager_out, compiled_out, atol=1e-5)

  Step 5: If slow, check recompilations
    TORCH_LOGS="recompiles" python script.py

  Step 6: If crash in backend, use minifier
    torch._dynamo.config.repro_after = "dynamo"
    # Generates minified_repro.py

  Environment variables for debugging:
    TORCH_LOGS="graph_breaks"        # Find graph breaks
    TORCH_LOGS="recompiles"          # Find recompilation
    TORCH_LOGS="output_code"         # See generated code
    TORCH_LOGS="+dynamo"             # Verbose Dynamo logs
    TORCH_COMPILE_DEBUG=1            # Full debug output directory
""")


# ============================================================================
# Main
# ============================================================================


def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║        Module 30: torch.compile Debugging — Full Demo              ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    demo_explain()
    demo_fixing_graph_breaks()
    demo_compile_counter()
    demo_record_graphs()
    demo_torch_logs()
    demo_recompilation()
    demo_common_compile_errors()
    print_debugging_workflow()

    print("=" * 70)
    print("ALL DEMOS COMPLETE")
    print("=" * 70)
    print()
    print("Key takeaways:")
    print("  1. Use explain() to find graph breaks without TORCH_LOGS")
    print("  2. Guard debug prints with torch.compiler.is_compiling()")
    print("  3. Replace data-dependent if/else with torch.where/torch.cond")
    print("  4. Use dynamic=True to avoid shape-triggered recompilation")
    print("  5. CompileCounter is the simplest way to verify compilation")
    print("  6. Always compare compiled vs eager output for correctness")
    print()


if __name__ == "__main__":
    main()
