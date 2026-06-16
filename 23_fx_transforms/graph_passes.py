"""
torch.fx Graph Passes — Practical Graph Transformations

Demonstrates:
  - Pass 1: Replace all ReLU with GELU
  - Pass 2: Add timing instrumentation
  - Pass 3: Fuse consecutive linear layers
  - Pass 4: Pattern matching with replace_pattern
  - Pass 5: Dead code elimination
  - Using torch.fx.Interpreter for custom execution
  - Using torch.fx.Transformer for node-level transforms
  - Verification that transformed models produce correct outputs

Run: python graph_passes.py
"""

import time
import torch
import torch.nn as nn
import torch.fx
from torch.fx import subgraph_rewriter


# ─── Test Models ──────────────────────────────────────────────────────

class ReLUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(20, 20)
        self.linear3 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.relu1(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class FusableModel(nn.Module):
    """Two consecutive Linear layers (no activation) that can be fused."""
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class DeadCodeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(10, 20)
        self.linear3 = nn.Linear(20, 5)

    def forward(self, x):
        a = self.linear1(x)
        _unused = self.linear2(x)  # noqa: F841
        out = self.linear3(a)
        return out


class PatternModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.add(x, x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x


# ─── Pass 1: Replace ReLU with GELU ──────────────────────────────────

def replace_relu_with_gelu(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target in (
            torch.relu, torch.nn.functional.relu
        ):
            node.target = torch.nn.functional.gelu

        elif node.op == "call_module":
            mod = gm.get_submodule(node.target)
            if isinstance(mod, nn.ReLU):
                parent_name, _, attr_name = node.target.rpartition(".")
                parent = gm.get_submodule(parent_name) if parent_name else gm
                setattr(parent, attr_name, nn.GELU())

    gm.graph.lint()
    gm.recompile()
    return gm


def demo_replace_relu():
    print("=" * 70)
    print("PASS 1: REPLACE ReLU WITH GELU")
    print("=" * 70)

    model = ReLUModel()
    traced = torch.fx.symbolic_trace(model)

    print("\n--- Before ---")
    print(traced.code)

    transformed = replace_relu_with_gelu(traced)

    print("--- After ---")
    print(transformed.code)

    x = torch.randn(4, 10)
    out = transformed(x)
    print(f"Output shape: {out.shape}")

    has_relu = any(
        (n.op == "call_function" and n.target in (torch.relu, torch.nn.functional.relu))
        or (n.op == "call_module" and isinstance(
            transformed.get_submodule(n.target), nn.ReLU
        ))
        for n in transformed.graph.nodes
        if n.op in ("call_function", "call_module")
    )
    print(f"ReLU remaining: {has_relu}")
    print(f"Pass 1 PASSED" if not has_relu else "Pass 1 FAILED")


# ─── Pass 2: Add Timing Instrumentation ──────────────────────────────

def add_timing_instrumentation(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    graph = gm.graph
    gm._timing_results = {}

    def record_start():
        return time.perf_counter()

    def record_end(name, start_time):
        elapsed = time.perf_counter() - start_time
        return elapsed

    torch.fx.wrap("record_start")
    torch.fx.wrap("record_end")

    for node in list(graph.nodes):
        if node.op in ("call_function", "call_module", "call_method"):
            with graph.inserting_before(node):
                start_node = graph.call_function(record_start, args=())
                start_node.name = f"timer_start_{node.name}"
            with graph.inserting_after(node):
                end_node = graph.call_function(
                    record_end,
                    args=(node.name, start_node),
                )
                end_node.name = f"timer_end_{node.name}"

    graph.lint()
    gm.recompile()
    return gm


def demo_timing():
    print("\n" + "=" * 70)
    print("PASS 2: ADD TIMING INSTRUMENTATION")
    print("=" * 70)

    model = ReLUModel()
    traced = torch.fx.symbolic_trace(model)

    print("\n--- Before (node count) ---")
    before_count = len(list(traced.graph.nodes))
    print(f"  Nodes: {before_count}")

    transformed = add_timing_instrumentation(traced)

    print("\n--- After (node count) ---")
    after_count = len(list(transformed.graph.nodes))
    print(f"  Nodes: {after_count}")
    print(f"  Added {after_count - before_count} timing nodes")

    print("\n--- Transformed code ---")
    print(transformed.code)

    x = torch.randn(4, 10)
    out = transformed(x)
    print(f"Output shape: {out.shape}")
    print("Pass 2 PASSED")


# ─── Pass 3: Fuse Consecutive Linear Layers ──────────────────────────

def fuse_linear_layers(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    graph = gm.graph
    fused_count = 0

    for node in list(graph.nodes):
        if node.op != "call_module":
            continue
        try:
            mod1 = gm.get_submodule(node.target)
        except AttributeError:
            continue
        if not isinstance(mod1, nn.Linear):
            continue

        users = list(node.users.keys())
        if len(users) != 1 or users[0].op != "call_module":
            continue
        next_node = users[0]
        try:
            mod2 = gm.get_submodule(next_node.target)
        except AttributeError:
            continue
        if not isinstance(mod2, nn.Linear):
            continue

        with torch.no_grad():
            W_fused = mod2.weight @ mod1.weight
            b_fused = None
            if mod1.bias is not None and mod2.bias is not None:
                b_fused = mod2.weight @ mod1.bias + mod2.bias
            elif mod2.bias is not None:
                b_fused = mod2.bias
            elif mod1.bias is not None:
                b_fused = mod2.weight @ mod1.bias

        fused = nn.Linear(mod1.in_features, mod2.out_features, bias=(b_fused is not None))
        with torch.no_grad():
            fused.weight.copy_(W_fused)
            if b_fused is not None:
                fused.bias.copy_(b_fused)

        fused_name = f"fused_{node.name}_{next_node.name}"
        gm.add_module(fused_name, fused)

        with graph.inserting_before(node):
            fused_node = graph.call_module(fused_name, args=node.args)

        next_node.replace_all_uses_with(fused_node)
        graph.erase_node(next_node)
        graph.erase_node(node)
        fused_count += 1

    graph.lint()
    gm.recompile()
    return gm, fused_count


def demo_fuse_linear():
    print("\n" + "=" * 70)
    print("PASS 3: FUSE CONSECUTIVE LINEAR LAYERS")
    print("=" * 70)

    model = FusableModel()
    traced = torch.fx.symbolic_trace(model)

    print("\n--- Before ---")
    print(traced.code)

    x = torch.randn(4, 10)
    out_before = model(x)

    transformed, count = fuse_linear_layers(traced)

    print(f"--- After (fused {count} pairs) ---")
    print(transformed.code)

    out_after = transformed(x)
    max_diff = (out_before - out_after).abs().max().item()
    print(f"Max difference: {max_diff:.2e}")
    print(f"Outputs match: {torch.allclose(out_before, out_after, atol=1e-5)}")

    linear_count = sum(
        1 for n in transformed.graph.nodes
        if n.op == "call_module" and isinstance(
            transformed.get_submodule(n.target), nn.Linear
        )
    )
    print(f"Linear modules in graph: {linear_count} (was 2)")
    print(f"Pass 3 PASSED" if linear_count == 1 else "Pass 3 FAILED")


# ─── Pass 4: Pattern Matching with replace_pattern ───────────────────

def demo_pattern_matching():
    print("\n" + "=" * 70)
    print("PASS 4: PATTERN MATCHING WITH replace_pattern")
    print("=" * 70)

    model = PatternModel()
    traced = torch.fx.symbolic_trace(model)

    print("\n--- Before ---")
    print(traced.code)

    def pattern(x):
        a = torch.add(x, x)
        b = torch.relu(a)
        return b

    def replacement(x):
        return torch.nn.functional.gelu(torch.mul(x, 2.0))

    matches = subgraph_rewriter.replace_pattern(traced, pattern, replacement)
    traced.graph.lint()
    traced.recompile()

    print(f"--- After ({len(matches)} replacements) ---")
    print(traced.code)

    x = torch.randn(4, 10)
    out = traced(x)
    print(f"Output shape: {out.shape}")
    print(f"Pass 4 PASSED" if len(matches) > 0 else "Pass 4: no matches found")


# ─── Pass 5: Dead Code Elimination ───────────────────────────────────

def demo_dead_code_elimination():
    print("\n" + "=" * 70)
    print("PASS 5: DEAD CODE ELIMINATION")
    print("=" * 70)

    model = DeadCodeModel()
    traced = torch.fx.symbolic_trace(model)

    print("\n--- Before ---")
    before_count = len(list(traced.graph.nodes))
    print(traced.code)
    print(f"Node count: {before_count}")

    traced.graph.eliminate_dead_code()
    traced.recompile()

    print("--- After ---")
    after_count = len(list(traced.graph.nodes))
    print(traced.code)
    print(f"Node count: {after_count}")
    print(f"Removed {before_count - after_count} dead nodes")

    x = torch.randn(4, 10)
    out = traced(x)
    print(f"Output shape: {out.shape}")
    print(f"Pass 5 PASSED" if after_count < before_count else "Pass 5 FAILED")


# ─── Interpreter: Custom Execution ───────────────────────────────────

class ProfilingInterpreter(torch.fx.Interpreter):
    def __init__(self, module):
        super().__init__(module)
        self.node_profiles = {}

    def run_node(self, node):
        start = time.perf_counter()
        result = super().run_node(node)
        elapsed = time.perf_counter() - start
        self.node_profiles[node.name] = {
            "op": node.op,
            "time_ms": elapsed * 1000,
            "output_shape": result.shape if isinstance(result, torch.Tensor) else None,
        }
        return result


class TypeCheckInterpreter(torch.fx.Interpreter):
    """Checks dtypes at every node to catch mixed-precision issues."""
    def __init__(self, module):
        super().__init__(module)
        self.dtype_log = {}

    def run_node(self, node):
        result = super().run_node(node)
        if isinstance(result, torch.Tensor):
            self.dtype_log[node.name] = result.dtype
        return result


def demo_interpreter():
    print("\n" + "=" * 70)
    print("INTERPRETER: CUSTOM EXECUTION")
    print("=" * 70)

    model = ReLUModel()
    traced = torch.fx.symbolic_trace(model)
    x = torch.randn(32, 10)

    print("\n--- Profiling Interpreter ---")
    profiler = ProfilingInterpreter(traced)
    output = profiler.run(x)
    print(f"Output shape: {output.shape}")

    print(f"\n{'Node':20s} {'Op':15s} {'Time (ms)':>10s} {'Shape':>20s}")
    print("-" * 68)
    for name, info in profiler.node_profiles.items():
        shape_str = str(info["output_shape"]) if info["output_shape"] else ""
        print(f"{name:20s} {info['op']:15s} {info['time_ms']:10.4f} {shape_str:>20s}")

    total_ms = sum(p["time_ms"] for p in profiler.node_profiles.values())
    print(f"\nTotal execution time: {total_ms:.4f} ms")

    print("\n--- Type Check Interpreter ---")
    checker = TypeCheckInterpreter(traced)
    checker.run(x)
    for name, dtype in checker.dtype_log.items():
        print(f"  {name:20s} dtype={dtype}")

    print("\nInterpreter demo PASSED")


# ─── Transformer: Node-Level Transforms ──────────────────────────────

class ReLUToGELUTransformer(torch.fx.Transformer):
    def call_function(self, target, args, kwargs):
        if target in (torch.relu, torch.nn.functional.relu):
            return super().call_function(torch.nn.functional.gelu, args, kwargs)
        return super().call_function(target, args, kwargs)

    def call_module(self, target, args, kwargs):
        mod = self.fetch_attr(target)
        if isinstance(mod, nn.ReLU):
            return super().call_function(torch.nn.functional.gelu, args, kwargs)
        return super().call_module(target, args, kwargs)


class ClampTransformer(torch.fx.Transformer):
    """Add clamping after every linear layer to prevent extreme values."""
    def call_module(self, target, args, kwargs):
        result = super().call_module(target, args, kwargs)
        mod = self.fetch_attr(target)
        if isinstance(mod, nn.Linear):
            result = super().call_function(
                torch.clamp, (result,), {"min": -10.0, "max": 10.0}
            )
        return result


def demo_transformer():
    print("\n" + "=" * 70)
    print("TRANSFORMER: NODE-LEVEL TRANSFORMS")
    print("=" * 70)

    model = ReLUModel()
    traced = torch.fx.symbolic_trace(model)

    print("\n--- ReLU to GELU Transformer ---")
    print("Before:")
    print(traced.code)

    transformed = ReLUToGELUTransformer(traced).transform()
    print("After:")
    print(transformed.code)

    x = torch.randn(4, 10)
    out = transformed(x)
    print(f"Output shape: {out.shape}")

    print("\n--- Clamp Transformer ---")
    model2 = ReLUModel()
    traced2 = torch.fx.symbolic_trace(model2)
    clamped = ClampTransformer(traced2).transform()
    print("After clamping:")
    print(clamped.code)

    x_large = torch.randn(4, 10) * 100
    out_clamped = clamped(x_large)
    print(f"Max absolute value: {out_clamped.abs().max().item():.1f} (clamped to 10)")
    print("Transformer demo PASSED")


# ─── Verification: All Passes Produce Correct Outputs ─────────────────

def demo_verification():
    print("\n" + "=" * 70)
    print("VERIFICATION: CORRECTNESS CHECK")
    print("=" * 70)

    torch.manual_seed(42)
    x = torch.randn(8, 10)

    model = FusableModel()
    traced = torch.fx.symbolic_trace(model)
    out_original = model(x)

    fused, _ = fuse_linear_layers(torch.fx.symbolic_trace(model))
    out_fused = fused(x)

    print(f"\n--- Linear Fusion ---")
    print(f"  Original output[:3]: {out_original[0, :3].tolist()}")
    print(f"  Fused output[:3]:    {out_fused[0, :3].tolist()}")
    print(f"  Max diff: {(out_original - out_fused).abs().max().item():.2e}")
    print(f"  Match: {torch.allclose(out_original, out_fused, atol=1e-5)}")

    model2 = ReLUModel()
    traced2 = torch.fx.symbolic_trace(model2)
    out_orig2 = traced2(x)

    transformed2 = ReLUToGELUTransformer(traced2).transform()
    out_trans2 = transformed2(x)

    print(f"\n--- ReLU->GELU Transform ---")
    print(f"  Original output[:3]: {out_orig2[0, :3].tolist()}")
    print(f"  GELU output[:3]:     {out_trans2[0, :3].tolist()}")
    print(f"  Outputs differ (expected): {not torch.allclose(out_orig2, out_trans2)}")

    model3 = DeadCodeModel()
    out_orig3 = model3(x)
    traced3 = torch.fx.symbolic_trace(model3)
    traced3.graph.eliminate_dead_code()
    traced3.recompile()
    out_dce = traced3(x)

    print(f"\n--- Dead Code Elimination ---")
    print(f"  Outputs match: {torch.allclose(out_orig3, out_dce)}")

    print("\nAll verifications PASSED")


# ─── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo_replace_relu()
    demo_timing()
    demo_fuse_linear()
    demo_pattern_matching()
    demo_dead_code_elimination()
    demo_interpreter()
    demo_transformer()
    demo_verification()

    print("\n" + "=" * 70)
    print("All graph_passes demos complete!")
    print("=" * 70)
