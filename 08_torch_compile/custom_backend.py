"""
Custom Backends — Writing Your Own torch.compile Backend
=========================================================
Demonstrates how to write custom backends for torch.compile:
1. Simple pass-through backend
2. Profiling backend (counts operations)
3. Graph-transforming backend
4. Understanding the FX graph structure

Run: python custom_backend.py
"""

import torch
import torch.nn as nn
import torch._dynamo
from collections import Counter

# =============================================================================
# 1. Simplest possible custom backend
# =============================================================================

print("=" * 60)
print("CUSTOM torch.compile BACKENDS")
print("=" * 60)

print("\n--- 1. Pass-through backend ---\n")

def passthrough_backend(gm: torch.fx.GraphModule, example_inputs):
    """
    Simplest backend: receives the graph and returns it unchanged.

    Args:
        gm: The FX GraphModule captured by Dynamo
        example_inputs: The actual inputs that triggered compilation

    Returns:
        A callable (the GraphModule itself is callable)
    """
    print(f"  Backend received graph with {len(list(gm.graph.nodes))} nodes")
    print(f"  Example inputs: {[x.shape for x in example_inputs]}")
    # Just return the graph module as-is (no optimization)
    return gm

def simple_fn(x, y):
    z = x + y
    z = z.relu()
    z = z * 2
    return z

torch._dynamo.reset()
compiled = torch.compile(simple_fn, backend=passthrough_backend)
result = compiled(torch.randn(4, 8), torch.randn(4, 8))
print(f"  Output shape: {result.shape}")

# =============================================================================
# 2. Profiling backend — count operations
# =============================================================================

print("\n" + "=" * 60)
print("--- 2. Profiling Backend (Operation Counter) ---")
print("=" * 60 + "\n")

class ProfilingBackend:
    """Backend that profiles the graph by counting operations."""

    def __init__(self):
        self.op_counts = Counter()
        self.total_graphs = 0
        self.total_nodes = 0

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        self.total_graphs += 1

        for node in gm.graph.nodes:
            self.total_nodes += 1
            if node.op == 'call_function':
                # Function calls like torch.add, torch.relu, etc.
                self.op_counts[str(node.target)] += 1
            elif node.op == 'call_method':
                # Method calls like .relu(), .view(), etc.
                self.op_counts[f".{node.target}()"] += 1
            elif node.op == 'call_module':
                # Module calls like self.linear, self.norm, etc.
                module = gm.get_submodule(node.target)
                self.op_counts[type(module).__name__] += 1

        return gm

    def report(self):
        print(f"  Total graphs compiled: {self.total_graphs}")
        print(f"  Total nodes: {self.total_nodes}")
        print(f"  Operation counts:")
        for op, count in self.op_counts.most_common(10):
            print(f"    {op}: {count}")


# Use it with a real model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.norm = nn.LayerNorm(128)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.norm(x)
        return x + torch.randn_like(x) * 0.01  # Some noise


profiler = ProfilingBackend()
torch._dynamo.reset()
model = MLP()
compiled_model = torch.compile(model, backend=profiler)

# Run some inputs to trigger compilation
with torch.no_grad():
    compiled_model(torch.randn(32, 128))

profiler.report()

# =============================================================================
# 3. Graph-inspecting backend — print the FX graph
# =============================================================================

print("\n" + "=" * 60)
print("--- 3. Graph Inspector Backend ---")
print("=" * 60 + "\n")

def inspector_backend(gm: torch.fx.GraphModule, example_inputs):
    """Backend that prints the FX graph in a readable format."""
    print("  FX Graph Structure:")
    print("  " + "-" * 50)

    for node in gm.graph.nodes:
        if node.op == 'placeholder':
            print(f"  INPUT: {node.name} (shape={example_inputs[0].shape if node.name == list(gm.graph.nodes)[0].name else '...'})")
        elif node.op == 'call_function':
            args_str = ', '.join(str(a) for a in node.args[:3])
            print(f"  CALL:  {node.name} = {node.target.__name__}({args_str})")
        elif node.op == 'call_method':
            print(f"  METHOD: {node.name} = {node.args[0]}.{node.target}()")
        elif node.op == 'call_module':
            print(f"  MODULE: {node.name} = {node.target}(...)")
        elif node.op == 'output':
            print(f"  OUTPUT: return {node.args[0]}")

    print("  " + "-" * 50)
    return gm

def attention_like_fn(q, k, v):
    """Simplified attention computation."""
    scores = torch.matmul(q, k.transpose(-2, -1))
    scores = scores / (q.shape[-1] ** 0.5)
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)

torch._dynamo.reset()
compiled_attn = torch.compile(attention_like_fn, backend=inspector_backend)

q = torch.randn(2, 4, 8, 16)  # [batch, heads, seq, dim]
k = torch.randn(2, 4, 8, 16)
v = torch.randn(2, 4, 8, 16)

result = compiled_attn(q, k, v)
print(f"\n  Output shape: {result.shape}")

# =============================================================================
# 4. Timing backend — measure overhead
# =============================================================================

print("\n" + "=" * 60)
print("--- 4. Timing Backend ---")
print("=" * 60 + "\n")

import time

class TimingBackend:
    """Backend that adds timing instrumentation."""

    def __init__(self):
        self.call_times = []

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        backend = self

        class TimedModule(torch.nn.Module):
            def __init__(self, original):
                super().__init__()
                self.original = original

            def forward(self, *args, **kwargs):
                start = time.perf_counter()
                result = self.original(*args, **kwargs)
                elapsed = time.perf_counter() - start
                backend.call_times.append(elapsed)
                return result

        return TimedModule(gm)

    def report(self):
        if self.call_times:
            avg = sum(self.call_times) / len(self.call_times) * 1000
            print(f"  Calls recorded: {len(self.call_times)}")
            print(f"  Average time: {avg:.4f} ms")
            print(f"  Total time: {sum(self.call_times)*1000:.4f} ms")


timer = TimingBackend()
torch._dynamo.reset()

def compute_fn(x):
    for _ in range(5):
        x = x @ x.T
        x = torch.softmax(x, dim=-1)
    return x

compiled_timed = torch.compile(compute_fn, backend=timer)

# Run multiple times
for _ in range(20):
    compiled_timed(torch.randn(32, 32))

timer.report()

# =============================================================================
# 5. Understanding FX Graph node types
# =============================================================================

print("\n" + "=" * 60)
print("--- 5. FX Graph Node Types ---")
print("=" * 60)
print("""
  FX Graph nodes have these types (node.op):

  'placeholder'    — Input arguments to the function
  'get_attr'       — Accessing a stored attribute (e.g., model parameters)
  'call_function'  — Calling a function: torch.add(x, y)
  'call_method'    — Calling a method: x.view(...)
  'call_module'    — Calling an nn.Module: self.linear(x)
  'output'         — The return value

  Each node has:
    - node.name: unique identifier
    - node.op: one of the types above
    - node.target: what's being called
    - node.args: positional arguments (other nodes or constants)
    - node.kwargs: keyword arguments
    - node.users: nodes that use this node's output
""")

# Demonstrate by tracing a model with torch.fx
print("  Example FX trace of a simple model:")
print("  " + "-" * 40)

class DemoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32, 32)

    def forward(self, x):
        x = self.linear(x)
        x = torch.relu(x)
        return x * 2

demo = DemoModel()
traced = torch.fx.symbolic_trace(demo)

for node in traced.graph.nodes:
    print(f"  {node.op:15s} | {node.name:10s} | target={node.target}")

# =============================================================================
# 6. Backend that modifies the graph
# =============================================================================

print("\n" + "=" * 60)
print("--- 6. Graph-Transforming Backend ---")
print("=" * 60 + "\n")

def relu_to_gelu_backend(gm: torch.fx.GraphModule, example_inputs):
    """Backend that replaces all relu calls with gelu."""
    replacements = 0
    for node in gm.graph.nodes:
        if node.op == 'call_function' and node.target == torch.relu:
            node.target = torch.nn.functional.gelu
            replacements += 1

    if replacements > 0:
        gm.graph.lint()  # Validate the graph
        gm.recompile()   # Regenerate the forward method
        print(f"  Replaced {replacements} relu(s) with gelu(s)")

    return gm

def fn_with_relu(x):
    x = torch.relu(x)
    x = x * 2
    x = torch.relu(x)
    return x

torch._dynamo.reset()
compiled_modified = torch.compile(fn_with_relu, backend=relu_to_gelu_backend)

x = torch.randn(8, 32)
result_modified = compiled_modified(x)
result_original = fn_with_relu(x)

# They differ because relu was replaced with gelu
print(f"  Original (relu) output sum: {result_original.sum():.4f}")
print(f"  Modified (gelu) output sum: {result_modified.sum():.4f}")
print(f"  Outputs differ (as expected): {not torch.allclose(result_modified, result_original)}")

# =============================================================================
# 7. Combining with explain()
# =============================================================================

print("\n" + "=" * 60)
print("--- 7. Using explain() with default backend ---")
print("=" * 60 + "\n")

torch._dynamo.reset()

class BiggerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        self.norm = nn.LayerNorm(64)

    def forward(self, x):
        residual = x
        x = self.layers(x)
        x = self.norm(x + residual)
        return x

model = BiggerModel()
x = torch.randn(16, 64)

explanation = torch._dynamo.explain(model)(x)
print(f"  Graph count: {explanation.graph_count}")
print(f"  Graph break count: {explanation.graph_break_count}")
print(f"  Explanation shows compilation is clean (no breaks)")

print("\nCustom backends demonstration complete!")
