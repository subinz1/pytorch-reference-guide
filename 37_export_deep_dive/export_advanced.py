"""
Module 37: torch.export Deep Dive — Advanced Export Techniques
==============================================================

Runnable on CPU. Covers:
- ExportedProgram inspection
- Graph signature (input/output specs)
- Dynamic shapes with Dim API
- Dim.AUTO for automatic inference
- Runtime constraints with torch._check
- draft_export for debugging
- Pre-dispatch vs post-dispatch IR
- run_decompositions()
- Retraceability (re-exporting)
- Custom ops with register_fake
- Strict vs non-strict export
- Save and load round-trip

Usage:
    python export_advanced.py
"""

import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.export import Dim, export


# ============================================================
# 1. ExportedProgram Inspection
# ============================================================

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.register_buffer("scale", torch.tensor(2.0))

    def forward(self, x):
        return self.linear(x) * self.scale


def demo_exported_program_inspection():
    print("=" * 70)
    print("1. ExportedProgram Inspection")
    print("=" * 70)

    model = SimpleModel()
    example_input = torch.randn(3, 10)
    ep = export(model, (example_input,))

    print(f"\nType: {type(ep).__name__}")
    print(f"\nGraph module type: {type(ep.graph_module).__name__}")
    print(f"State dict keys: {list(ep.state_dict.keys())}")
    print(f"Range constraints: {ep.range_constraints}")
    print(f"Constants: {ep.constants}")

    print("\n--- FX Graph ---")
    ep.graph_module.graph.print_tabular()

    result_original = model(example_input)
    result_exported = ep.module()(example_input)
    print(f"\nResults match: {torch.allclose(result_original, result_exported)}")
    print()


# ============================================================
# 2. Graph Signature — Input/Output Specs
# ============================================================

class ModelWithBuffer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 4)
        self.register_buffer("running_mean", torch.zeros(4))

    def forward(self, x):
        out = self.linear(x)
        self.running_mean = self.running_mean * 0.9 + out.mean(dim=0) * 0.1
        return out


def demo_graph_signature():
    print("=" * 70)
    print("2. Graph Signature — Input/Output Specs")
    print("=" * 70)

    model = ModelWithBuffer()
    ep = export(model, (torch.randn(2, 8),))

    sig = ep.graph_signature

    print("\nInput Specs:")
    for spec in sig.input_specs:
        target = spec.target if spec.target else "N/A"
        print(f"  {spec.kind}: {spec.arg.name} -> {target}")

    print("\nOutput Specs:")
    for spec in sig.output_specs:
        target = spec.target if spec.target else "N/A"
        print(f"  {spec.kind}: {spec.arg.name} -> {target}")

    print()


# ============================================================
# 3. Dynamic Shapes with Dim
# ============================================================

class DynamicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 8)

    def forward(self, x):
        return self.linear(x)


def demo_dynamic_shapes():
    print("=" * 70)
    print("3. Dynamic Shapes with Dim API")
    print("=" * 70)

    model = DynamicModel()

    # Single dynamic dim
    batch = Dim("batch", min=1, max=128)
    ep = export(
        model,
        (torch.randn(4, 16),),
        dynamic_shapes={"x": {0: batch}},
    )
    print(f"\nSingle dim - Range constraints: {ep.range_constraints}")

    result_1 = ep.module()(torch.randn(1, 16))
    result_32 = ep.module()(torch.randn(32, 16))
    print(f"batch=1 output shape:  {result_1.shape}")
    print(f"batch=32 output shape: {result_32.shape}")

    # Multiple dynamic dims
    class TwoInputModel(nn.Module):
        def forward(self, x, y):
            return x + y

    batch2 = Dim("batch", min=1, max=64)
    model2 = TwoInputModel()
    ep2 = export(
        model2,
        (torch.randn(4, 10), torch.randn(4, 10)),
        dynamic_shapes={"x": {0: batch2}, "y": {0: batch2}},
    )
    print(f"\nShared dim - constraints: {ep2.range_constraints}")

    result_shared = ep2.module()(torch.randn(8, 10), torch.randn(8, 10))
    print(f"Shared batch=8 output shape: {result_shared.shape}")

    print()


# ============================================================
# 4. Dim.AUTO for Automatic Inference
# ============================================================

def demo_dim_auto():
    print("=" * 70)
    print("4. Dim.AUTO for Automatic Inference")
    print("=" * 70)

    model = DynamicModel()
    ep = export(
        model,
        (torch.randn(4, 16),),
        dynamic_shapes={"x": {0: Dim.AUTO}},
    )

    print(f"\nDim.AUTO constraints: {ep.range_constraints}")

    for batch_size in [1, 8, 16]:
        result = ep.module()(torch.randn(batch_size, 16))
        print(f"batch={batch_size}: output shape = {result.shape}")

    print()


# ============================================================
# 5. Runtime Constraints with torch._check
# ============================================================

class ConstrainedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        torch._check(x.shape[0] >= 1)
        torch._check(x.shape[0] <= 64)
        return self.linear(x)


def demo_constraints():
    print("=" * 70)
    print("5. Runtime Constraints with torch._check")
    print("=" * 70)

    model = ConstrainedModel()
    batch = Dim("batch", min=1, max=64)
    ep = export(
        model,
        (torch.randn(4, 10),),
        dynamic_shapes={"x": {0: batch}},
    )

    print(f"\nRange constraints: {ep.range_constraints}")

    result = ep.module()(torch.randn(32, 10))
    print(f"batch=32 output shape: {result.shape}")

    print()


# ============================================================
# 6. draft_export for Debugging
# ============================================================

class ProblematicModel(nn.Module):
    """Model with data-dependent control flow that export can't handle."""
    def __init__(self):
        super().__init__()
        self.linear_pos = nn.Linear(10, 5)
        self.linear_neg = nn.Linear(10, 5)

    def forward(self, x):
        if x.sum() > 0:
            return self.linear_pos(x)
        else:
            return self.linear_neg(x)


def demo_draft_export():
    print("=" * 70)
    print("6. draft_export for Debugging")
    print("=" * 70)

    model = ProblematicModel()
    args = (torch.randn(3, 10),)

    print("\nTrying strict export (expected to fail)...")
    try:
        ep = export(model, args)
        print("  Unexpectedly succeeded!")
    except Exception as e:
        error_msg = str(e).split("\n")[0]
        print(f"  Failed: {error_msg}")

    print("\nTrying draft_export...")
    try:
        from torch.export import draft_export
        ep, report = draft_export(model, args)
        print(f"  draft_export returned program: {ep is not None}")
        if report:
            print(f"  Report: {report}")
    except Exception as e:
        error_msg = str(e).split("\n")[0]
        print(f"  draft_export also encountered issue: {error_msg}")

    print()


# ============================================================
# 7. Pre-Dispatch vs Post-Dispatch IR
# ============================================================

class IRComparisonModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.norm = nn.LayerNorm(10)

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        x = torch.relu(x)
        return x


def demo_pre_post_dispatch():
    print("=" * 70)
    print("7. Pre-Dispatch vs Post-Dispatch IR")
    print("=" * 70)

    model = IRComparisonModel()
    args = (torch.randn(4, 10),)

    ep_post = export(model, args)
    print("\n--- Post-Dispatch (default) ---")
    post_ops = set()
    for node in ep_post.graph_module.graph.nodes:
        if node.op == "call_function":
            post_ops.add(str(node.target).split(".")[-1])
    print(f"Ops: {sorted(post_ops)}")

    ep_pre = export(model, args, pre_dispatch=True)
    print("\n--- Pre-Dispatch ---")
    pre_ops = set()
    for node in ep_pre.graph_module.graph.nodes:
        if node.op == "call_function":
            pre_ops.add(str(node.target).split(".")[-1])
    print(f"Ops: {sorted(pre_ops)}")

    print(f"\nPost-dispatch has {len(post_ops)} unique ops")
    print(f"Pre-dispatch has {len(pre_ops)} unique ops")
    print("(Post-dispatch typically has more ops due to decomposition)")

    print()


# ============================================================
# 8. run_decompositions(): Pre to Post
# ============================================================

def demo_run_decompositions():
    print("=" * 70)
    print("8. run_decompositions() — Converting Pre to Post")
    print("=" * 70)

    model = IRComparisonModel()
    args = (torch.randn(4, 10),)

    ep_pre = export(model, args, pre_dispatch=True)
    pre_nodes = sum(1 for n in ep_pre.graph_module.graph.nodes if n.op == "call_function")
    print(f"\nPre-dispatch: {pre_nodes} call_function nodes")

    ep_post = ep_pre.run_decompositions()
    post_nodes = sum(1 for n in ep_post.graph_module.graph.nodes if n.op == "call_function")
    print(f"After run_decompositions(): {post_nodes} call_function nodes")

    result_pre = ep_pre.module()(*args)
    result_post = ep_post.module()(*args)
    print(f"Results match: {torch.allclose(result_pre, result_post, atol=1e-6)}")

    print()


# ============================================================
# 9. Retraceability — Re-exporting an Exported Program
# ============================================================

def demo_retraceability():
    print("=" * 70)
    print("9. Retraceability — Re-exporting")
    print("=" * 70)

    model = DynamicModel()
    args = (torch.randn(4, 16),)

    batch_wide = Dim("batch", min=1, max=256)
    ep1 = export(model, args, dynamic_shapes={"x": {0: batch_wide}})
    print(f"\nFirst export constraints: {ep1.range_constraints}")

    batch_tight = Dim("batch", min=1, max=32)
    ep2 = export(ep1.module(), args, dynamic_shapes={"x": {0: batch_tight}})
    print(f"Re-exported constraints: {ep2.range_constraints}")

    result1 = ep1.module()(torch.randn(16, 16))
    result2 = ep2.module()(torch.randn(16, 16))
    print(f"Both produce valid outputs: shape={result1.shape}, {result2.shape}")

    print()


# ============================================================
# 10. Custom Op in Export — register_fake
# ============================================================

@torch.library.custom_op("export_demo::scale_and_shift", mutates_args=())
def scale_and_shift(x: torch.Tensor, scale: float, shift: float) -> torch.Tensor:
    return x * scale + shift


@scale_and_shift.register_fake
def scale_and_shift_fake(x, scale, shift):
    return torch.empty_like(x)


class CustomOpModel(nn.Module):
    def forward(self, x):
        return torch.ops.export_demo.scale_and_shift(x, 2.0, 1.0)


def demo_custom_op_export():
    print("=" * 70)
    print("10. Custom Op in Export")
    print("=" * 70)

    model = CustomOpModel()
    args = (torch.randn(3, 5),)

    ep = export(model, args)
    print(f"\nExport with custom op succeeded!")

    has_custom_op = any(
        "scale_and_shift" in str(n.target)
        for n in ep.graph_module.graph.nodes
        if n.op == "call_function"
    )
    print(f"Custom op in graph: {has_custom_op}")

    result_orig = model(*args)
    result_export = ep.module()(*args)
    print(f"Results match: {torch.allclose(result_orig, result_export)}")

    print()


# ============================================================
# 11. Strict vs Non-Strict Export
# ============================================================

class StrictTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.threshold = 0.5

    def forward(self, x):
        out = self.linear(x)
        out = torch.clamp(out, min=-self.threshold, max=self.threshold)
        return out


def demo_strict_vs_nonstrict():
    print("=" * 70)
    print("11. Strict vs Non-Strict Export")
    print("=" * 70)

    model = StrictTestModel()
    args = (torch.randn(3, 10),)

    print("\nStrict export (default):")
    try:
        ep_strict = export(model, args, strict=True)
        print(f"  Succeeded — {sum(1 for n in ep_strict.graph_module.graph.nodes)} nodes")
    except Exception as e:
        print(f"  Failed: {str(e)[:100]}")

    print("\nNon-strict export:")
    try:
        ep_nonstrict = export(model, args, strict=False)
        print(f"  Succeeded — {sum(1 for n in ep_nonstrict.graph_module.graph.nodes)} nodes")
    except Exception as e:
        print(f"  Failed: {str(e)[:100]}")

    ep_s = export(model, args, strict=True)
    ep_ns = export(model, args, strict=False)
    r1 = ep_s.module()(*args)
    r2 = ep_ns.module()(*args)
    print(f"\nBoth produce same result: {torch.allclose(r1, r2)}")

    print()


# ============================================================
# 12. Save and Load Round-Trip
# ============================================================

def demo_save_load():
    print("=" * 70)
    print("12. Save and Load Round-Trip")
    print("=" * 70)

    model = SimpleModel()
    args = (torch.randn(3, 10),)
    ep = export(model, args)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.pt2"

        torch.export.save(ep, str(path))
        file_size = path.stat().st_size
        print(f"\nSaved to {path.name} ({file_size:,} bytes)")

        ep_loaded = torch.export.load(str(path))
        print(f"Loaded ExportedProgram: {type(ep_loaded).__name__}")

        result_orig = ep.module()(*args)
        result_loaded = ep_loaded.module()(*args)
        print(f"Results match after round-trip: {torch.allclose(result_orig, result_loaded)}")

        print(f"\nLoaded state dict keys: {list(ep_loaded.state_dict.keys())}")

        sig = ep_loaded.graph_signature
        n_params = sum(1 for s in sig.input_specs if "PARAMETER" in str(s.kind))
        n_buffers = sum(1 for s in sig.input_specs if "BUFFER" in str(s.kind))
        n_user = sum(1 for s in sig.input_specs if "USER_INPUT" in str(s.kind))
        print(f"Inputs: {n_params} params, {n_buffers} buffers, {n_user} user inputs")

    print()


# ============================================================
# Main
# ============================================================

def main():
    print("\n" + "=" * 70)
    print(" Module 37: torch.export Deep Dive — Advanced Export")
    print("=" * 70 + "\n")

    demo_exported_program_inspection()
    demo_graph_signature()
    demo_dynamic_shapes()
    demo_dim_auto()
    demo_constraints()
    demo_draft_export()
    demo_pre_post_dispatch()
    demo_run_decompositions()
    demo_retraceability()
    demo_custom_op_export()
    demo_strict_vs_nonstrict()
    demo_save_load()

    print("=" * 70)
    print(" All demos completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
