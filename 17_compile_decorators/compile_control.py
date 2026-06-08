"""
torch.compile Decorators & Control APIs
=========================================
Fine-grained control over what gets compiled and how.
All examples run on CPU.
"""

import torch
import torch.nn as nn

print("=" * 65)
print("1. COMPILER STANCES — Global Behavior Control")
print("=" * 65)

@torch.compile(backend="eager")
def simple_fn(x):
    return x.sin() + x.cos()

x = torch.randn(10)

torch.compiler.set_stance("default")
out1 = simple_fn(x)
print(f"Default stance output: {out1[:3].tolist()}")

torch.compiler.set_stance("force_eager")
out2 = simple_fn(x)
print(f"Force eager output:   {out2[:3].tolist()}")
print(f"Outputs match: {torch.allclose(out1, out2)}")

torch.compiler.set_stance("default")

with torch.compiler.set_stance("force_eager"):
    out3 = simple_fn(x)
    print(f"Inside force_eager context: computed eagerly")

print(f"\nAll stances: default, force_eager, eager_on_recompile, "
      f"fail_on_recompile, eager_then_compile")

torch._dynamo.reset()

print("\n" + "=" * 65)
print("2. torch.compiler.disable — Skip Compilation")
print("=" * 65)

@torch.compiler.disable
def untraceable_preprocessing(x):
    """Uses .item() and Python lists — doesn't compile well."""
    results = []
    for i in range(min(x.shape[0], 5)):
        val = x[i].item()
        results.append(val * 2 if val > 0 else val * -1)
    return torch.tensor(results)

@torch.compile(backend="eager")
def pipeline(x):
    processed = untraceable_preprocessing(x)  # Runs eagerly (disabled)
    return processed.relu().sum()              # Compiled

x = torch.randn(10)
result = pipeline(x)
print(f"Pipeline with disabled preprocessing: {result.item():.4f}")

torch._dynamo.reset()

print("\n" + "=" * 65)
print("3. allow_in_graph — Opaque Graph Nodes")
print("=" * 65)

@torch.compiler.allow_in_graph
def custom_op(x, scale):
    """Treated as a single node — Dynamo won't trace into it."""
    return x * scale + torch.sin(x)

@torch.compile(backend="eager", fullgraph=True)
def model_with_custom_op(x):
    y = x + 1
    z = custom_op(y, 2.0)
    return z.mean()

result = model_with_custom_op(torch.randn(5))
print(f"Model with allow_in_graph op: {result.item():.4f}")

torch._dynamo.reset()

print("\n" + "=" * 65)
print("4. mark_dynamic / mark_static — Shape Control")
print("=" * 65)

@torch.compile(backend="eager")
def sum_fn(x):
    return x.sum(dim=-1)

x1 = torch.randn(8, 64)
torch._dynamo.mark_dynamic(x1, 0)
out1 = sum_fn(x1)
print(f"Batch 8:   {out1.shape}")

x2 = torch.randn(16, 64)
torch._dynamo.mark_dynamic(x2, 0)
out2 = sum_fn(x2)
print(f"Batch 16:  {out2.shape}")

x3 = torch.randn(32, 64)
torch._dynamo.mark_dynamic(x3, 0)
out3 = sum_fn(x3)
print(f"Batch 32:  {out3.shape}")
print("All ran without recompilation (dim 0 marked dynamic)")

torch._dynamo.reset()

print("\n" + "=" * 65)
print("5. graph_break — Explicit Breaks")
print("=" * 65)

cnt = torch._dynamo.testing.CompileCounter()

@torch.compile(backend=cnt)
def fn_with_break(x):
    y = x + 1
    torch._dynamo.graph_break()
    z = y * 2
    return z

result = fn_with_break(torch.randn(5))
print(f"Explicit break — compilations: {cnt.frame_count}")
print(f"Result: {result[:3].tolist()}")

torch._dynamo.reset()

print("\n" + "=" * 65)
print("6. fullgraph=True — Strict Mode")
print("=" * 65)

@torch.compile(backend="eager", fullgraph=True)
def strict_ok(x):
    return x.sin() + x.cos()

print(f"Strict mode (no breaks): {strict_ok(torch.randn(5))[:3].tolist()}")
print("fullgraph=True errors on any graph break — use for CI/strictness")

torch._dynamo.reset()

print("\n" + "=" * 65)
print("7. torch._dynamo.explain — Compilation Report")
print("=" * 65)

def complex_fn(x):
    y = x + 1
    z = y.relu()
    return z.sum()

explanation = torch._dynamo.explain(complex_fn)(torch.randn(10))
print(f"Explanation:\n{explanation}")

torch._dynamo.reset()

print("\n" + "=" * 65)
print("8. assume_constant_result")
print("=" * 65)

call_count = 0

@torch._dynamo.assume_constant_result
def get_scale_factor():
    global call_count
    call_count += 1
    return 2.5

@torch.compile(backend="eager")
def scaled_fn(x):
    scale = get_scale_factor()
    return x * scale

result = scaled_fn(torch.randn(5))
result2 = scaled_fn(torch.randn(5))
print(f"Scaled output: {result[:3].tolist()}")
print(f"get_scale_factor called {call_count} time(s) (constant-folded)")

torch._dynamo.reset()

print("\n" + "=" * 65)
print("9. CompileCounter — Tracking Compilations")
print("=" * 65)

cnt = torch._dynamo.testing.CompileCounter()

@torch.compile(backend=cnt)
def tracked_fn(x):
    return x.sin() + x.cos() + x.relu()

tracked_fn(torch.randn(10))
print(f"After call 1: {cnt.frame_count} compilation(s), {cnt.op_count} op(s)")

tracked_fn(torch.randn(10))
print(f"After call 2: {cnt.frame_count} compilation(s) (cache hit)")

tracked_fn(torch.randn(20))
print(f"After call 3 (different shape): {cnt.frame_count} compilation(s)")

torch._dynamo.reset()

print("\n" + "=" * 65)
print("10. EagerAndRecordGraphs — Inspect Captured Graphs")
print("=" * 65)

backend = torch._dynamo.testing.EagerAndRecordGraphs()

@torch.compile(backend=backend)
def inspectable_fn(x, y):
    z = x + y
    w = z.relu()
    return w.sum()

inspectable_fn(torch.randn(5), torch.randn(5))

print(f"Captured graphs: {len(backend.graphs)}")
if backend.graphs:
    print(f"\nFX graph:\n{backend.graphs[0].graph}")

torch._dynamo.reset()

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)

print("""
Problem                          | Solution
---------------------------------|--------------------------------
Code has untraceable Python      | @torch.compiler.disable
Custom op should be opaque       | @torch.compiler.allow_in_graph
Third-party lib won't trace      | substitute_in_graph()
Too many recompilations          | mark_dynamic() on varying dims
Need guaranteed single graph     | fullgraph=True
Config value is constant         | @assume_constant_result
Want compilation report          | torch._dynamo.explain()
Disable all compilation          | set_stance("force_eager")
Count compilations in tests      | CompileCounter backend
Inspect captured graphs          | EagerAndRecordGraphs backend
""")

print("Done!")
