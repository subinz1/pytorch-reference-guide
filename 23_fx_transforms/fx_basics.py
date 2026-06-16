"""
torch.fx Basics — Symbolic Tracing, Graph Inspection, and ShapeProp

Demonstrates:
  - Symbolic tracing of a model
  - Printing the graph (tabular and generated code)
  - Iterating nodes: op, name, target, args
  - Counting operations by type
  - Shape propagation with ShapeProp
  - Graph validation with graph.lint()

Run: python fx_basics.py
"""

import torch
import torch.nn as nn
import torch.fx
from collections import Counter


# ─── Model Definitions ───────────────────────────────────────────────

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.bn = nn.BatchNorm1d(20)
        self.linear2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x


class MultiPathModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        residual = x
        x = torch.relu(self.conv2(x))
        x = x + residual
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


# ─── 1. Symbolic Tracing ─────────────────────────────────────────────

def demo_symbolic_tracing():
    print("=" * 70)
    print("1. SYMBOLIC TRACING")
    print("=" * 70)

    model = SimpleModel()
    traced = torch.fx.symbolic_trace(model)

    print(f"\nType of traced: {type(traced)}")
    print(f"Is nn.Module:   {isinstance(traced, nn.Module)}")
    print(f"Has .graph:     {hasattr(traced, 'graph')}")
    print(f"Has .code:      {hasattr(traced, 'code')}")

    print("\n--- Generated Code ---")
    print(traced.code)

    x = torch.randn(4, 10)
    out_orig = model(x)
    out_traced = traced(x)
    print(f"Outputs match: {torch.allclose(out_orig, out_traced)}")
    print(f"Output shape:  {out_traced.shape}")


# ─── 2. Print Tabular ────────────────────────────────────────────────

def demo_print_tabular():
    print("\n" + "=" * 70)
    print("2. GRAPH — TABULAR VIEW")
    print("=" * 70)

    model = SimpleModel()
    traced = torch.fx.symbolic_trace(model)

    print("\n--- graph.print_tabular() ---")
    traced.graph.print_tabular()

    print("\n--- str(graph) ---")
    print(traced.graph)


# ─── 3. Iterate Nodes ────────────────────────────────────────────────

def demo_iterate_nodes():
    print("\n" + "=" * 70)
    print("3. ITERATING NODES")
    print("=" * 70)

    model = SimpleModel()
    traced = torch.fx.symbolic_trace(model)

    print(f"\n{'op':15s} {'name':15s} {'target':30s} {'#args':6s} {'#users':6s}")
    print("-" * 75)
    for node in traced.graph.nodes:
        target_str = str(node.target)
        if len(target_str) > 28:
            target_str = target_str[:28] + ".."
        print(
            f"{node.op:15s} {node.name:15s} {target_str:30s} "
            f"{len(node.args):<6d} {len(node.users):<6d}"
        )

    print("\n--- Detailed args/kwargs ---")
    for node in traced.graph.nodes:
        print(f"\n  {node.name} (op={node.op})")
        print(f"    target = {node.target}")
        print(f"    args   = {node.args}")
        print(f"    kwargs = {node.kwargs}")
        user_names = [u.name for u in node.users]
        print(f"    users  = {user_names}")


# ─── 4. Count Operations ─────────────────────────────────────────────

def demo_count_operations():
    print("\n" + "=" * 70)
    print("4. COUNTING OPERATIONS")
    print("=" * 70)

    model = MultiPathModel()
    traced = torch.fx.symbolic_trace(model)

    print("\n--- Full graph ---")
    traced.graph.print_tabular()

    op_type_counts = Counter()
    detailed_counts = Counter()

    for node in traced.graph.nodes:
        op_type_counts[node.op] += 1

        if node.op == "call_function":
            name = getattr(node.target, "__name__", str(node.target))
            detailed_counts[f"fn:{name}"] += 1
        elif node.op == "call_module":
            mod = traced.get_submodule(node.target)
            detailed_counts[f"mod:{type(mod).__name__}"] += 1
        elif node.op == "call_method":
            detailed_counts[f"method:{node.target}"] += 1

    print("\n--- Operation type counts ---")
    for op, count in op_type_counts.most_common():
        print(f"  {op:20s}: {count}")

    print("\n--- Detailed operation counts ---")
    for op, count in detailed_counts.most_common():
        print(f"  {op:25s}: {count}")

    total_compute = sum(
        1 for n in traced.graph.nodes
        if n.op in ("call_function", "call_module", "call_method")
    )
    print(f"\n  Total compute ops: {total_compute}")


# ─── 5. Users and Dependencies ───────────────────────────────────────

def demo_users_and_deps():
    print("\n" + "=" * 70)
    print("5. USERS AND DEPENDENCIES (DATA FLOW)")
    print("=" * 70)

    model = MultiPathModel()
    traced = torch.fx.symbolic_trace(model)

    print("\n--- Data flow edges ---")
    for node in traced.graph.nodes:
        for user in node.users:
            print(f"  {node.name:15s} --> {user.name}")

    print("\n--- Nodes with multiple users (branches) ---")
    for node in traced.graph.nodes:
        if len(node.users) > 1:
            user_names = [u.name for u in node.users]
            print(f"  {node.name}: used by {user_names}")

    print("\n--- Nodes with multiple inputs ---")
    for node in traced.graph.nodes:
        input_nodes = [a for a in node.args if isinstance(a, torch.fx.Node)]
        if len(input_nodes) > 1:
            input_names = [n.name for n in input_nodes]
            print(f"  {node.name}: inputs from {input_names}")


# ─── 6. ShapeProp ────────────────────────────────────────────────────

def demo_shape_prop():
    print("\n" + "=" * 70)
    print("6. SHAPEPROP — PROPAGATING SHAPES")
    print("=" * 70)

    model = SimpleModel()
    traced = torch.fx.symbolic_trace(model)

    sample = torch.randn(8, 10)
    from torch.fx.passes.shape_prop import ShapeProp
    ShapeProp(traced).propagate(sample)

    print(f"\n{'node':15s} {'shape':25s} {'dtype':15s}")
    print("-" * 60)
    for node in traced.graph.nodes:
        meta = node.meta.get("tensor_meta")
        if meta is not None:
            if isinstance(meta, tuple):
                for i, m in enumerate(meta):
                    print(f"{node.name}[{i}]{'':10s} {str(m.shape):25s} {str(m.dtype):15s}")
            else:
                print(f"{node.name:15s} {str(meta.shape):25s} {str(meta.dtype):15s}")
        else:
            print(f"{node.name:15s} {'(no tensor meta)':25s}")

    print("\n--- ShapeProp on MultiPathModel ---")
    model2 = MultiPathModel()
    traced2 = torch.fx.symbolic_trace(model2)
    sample2 = torch.randn(2, 3, 32, 32)
    ShapeProp(traced2).propagate(sample2)

    for node in traced2.graph.nodes:
        meta = node.meta.get("tensor_meta")
        if meta is not None and not isinstance(meta, tuple):
            print(f"  {node.name:20s} {str(meta.shape):25s} {str(meta.dtype)}")


# ─── 7. graph.lint() ─────────────────────────────────────────────────

def demo_graph_lint():
    print("\n" + "=" * 70)
    print("7. GRAPH.LINT() — VALIDATION")
    print("=" * 70)

    model = SimpleModel()
    traced = torch.fx.symbolic_trace(model)

    print("\n--- Linting a valid graph ---")
    try:
        traced.graph.lint()
        print("  graph.lint() passed (no errors)")
    except Exception as e:
        print(f"  Lint error: {e}")

    print("\n--- Attempting to create an invalid graph ---")
    traced2 = torch.fx.symbolic_trace(SimpleModel())
    graph = traced2.graph
    nodes = list(graph.nodes)
    relu_node = None
    for n in nodes:
        if n.op == "call_function" and n.target == torch.relu:
            relu_node = n
            break

    if relu_node:
        print(f"  Found relu node: {relu_node.name}")
        print(f"  Users before removal attempt: {[u.name for u in relu_node.users]}")
        print("  Trying to erase node with users...")
        try:
            graph.erase_node(relu_node)
            print("  Erased (unexpected)")
        except RuntimeError as e:
            print(f"  RuntimeError: {e}")
            print("  (Expected — must remove users first)")


# ─── 8. Concrete Args ────────────────────────────────────────────────

def demo_concrete_args():
    print("\n" + "=" * 70)
    print("8. CONCRETE_ARGS — FIXING INPUTS")
    print("=" * 70)

    class ConditionalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)

        def forward(self, x, use_relu=True):
            x = self.linear(x)
            if use_relu:
                x = torch.relu(x)
            return x

    model = ConditionalModel()

    print("\n--- Tracing with use_relu=True ---")
    traced_relu = torch.fx.symbolic_trace(model, concrete_args={"use_relu": True})
    print(traced_relu.code)

    print("--- Tracing with use_relu=False ---")
    traced_no_relu = torch.fx.symbolic_trace(model, concrete_args={"use_relu": False})
    print(traced_no_relu.code)

    x = torch.randn(4, 10)
    print(f"With relu - has relu op: {'relu' in traced_relu.code}")
    print(f"Without relu - has relu op: {'relu' in traced_no_relu.code}")


# ─── 9. Comparing symbolic_trace vs torch.compile ────────────────────

def demo_trace_vs_compile():
    print("\n" + "=" * 70)
    print("9. SYMBOLIC_TRACE vs TORCH.COMPILE")
    print("=" * 70)

    print("""
  symbolic_trace:
    + Simple API, pure Python graph
    + Easy to inspect and transform
    - No data-dependent control flow
    - No dynamic shapes
    - Must trace the full forward

  torch.compile (Dynamo):
    + Handles control flow (graph breaks)
    + Dynamic shapes
    + Automatic recompilation
    - More complex internals
    - Graph is at ATen op level (more granular)

  Use symbolic_trace for:
    - Simple models without control flow
    - Quick graph inspection
    - Teaching / prototyping passes

  Use torch.compile for:
    - Production compilation
    - Models with control flow
    - Maximum performance
""")

    model = SimpleModel()

    print("--- symbolic_trace graph ---")
    traced = torch.fx.symbolic_trace(model)
    for node in traced.graph.nodes:
        print(f"  {node.op:15s} {node.name}")

    print("\n--- torch.compile backend receives ---")
    def debug_backend(gm, example_inputs):
        print("  (Graph from Dynamo):")
        for node in gm.graph.nodes:
            print(f"    {node.op:15s} {node.name}")
        return gm

    compiled = torch.compile(model, backend=debug_backend)
    compiled(torch.randn(4, 10))


# ─── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo_symbolic_tracing()
    demo_print_tabular()
    demo_iterate_nodes()
    demo_count_operations()
    demo_users_and_deps()
    demo_shape_prop()
    demo_graph_lint()
    demo_concrete_args()
    demo_trace_vs_compile()

    print("\n" + "=" * 70)
    print("All fx_basics demos complete!")
    print("=" * 70)
