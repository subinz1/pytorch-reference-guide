"""
Module 38 — AOTAutograd Explained
=================================

Demonstrates AOTAutograd: tracing both forward and backward at compile time,
inspecting the resulting FX graphs, and understanding the min-cut partitioner.

Runnable on CPU — no GPU required.

Usage:
    python aot_autograd_explained.py
"""

import torch
import torch.nn as nn
from torch._functorch.aot_autograd import aot_function


# ============================================================
# Part 1: Standard Autograd — The grad_fn Chain
# ============================================================

def show_standard_autograd():
    """Show how eager autograd builds the backward graph at runtime."""
    print("=" * 70)
    print("PART 1: Standard Autograd — grad_fn Chain")
    print("=" * 70)

    x = torch.randn(3, 4, requires_grad=True)
    weight = torch.randn(4, 2, requires_grad=True)
    bias = torch.randn(2, requires_grad=True)

    # Forward pass builds grad_fn chain
    mm = x @ weight
    added = mm + bias
    activated = torch.relu(added)
    loss = activated.sum()

    # Walk the grad_fn chain
    print("\ngrad_fn chain (built at runtime):")
    fn = loss.grad_fn
    depth = 0
    while fn is not None:
        indent = "  " * depth
        print(f"{indent}→ {fn.__class__.__name__}")
        if hasattr(fn, "next_functions"):
            nexts = fn.next_functions
            if nexts:
                fn = nexts[0][0]
            else:
                fn = None
        else:
            fn = None
        depth += 1

    loss.backward()
    print(f"\nx.grad shape: {x.grad.shape}")
    print(f"weight.grad shape: {weight.grad.shape}")
    print(f"bias.grad shape: {bias.grad.shape}")
    print()


# ============================================================
# Part 2: aot_function — Inspect Forward and Backward Graphs
# ============================================================

def show_aot_function_graphs():
    """Use aot_function to compile a function and inspect both graphs."""
    print("=" * 70)
    print("PART 2: aot_function — Forward and Backward Graphs")
    print("=" * 70)

    graphs = {"forward": None, "backward": None}

    def fw_compiler(gm, example_inputs):
        print("\n--- FORWARD GRAPH ---")
        gm.graph.print_tabular()
        graphs["forward"] = gm

        node_count = len([n for n in gm.graph.nodes if n.op == "call_function"])
        print(f"Forward op count: {node_count}")
        return gm

    def bw_compiler(gm, example_inputs):
        print("\n--- BACKWARD GRAPH ---")
        gm.graph.print_tabular()
        graphs["backward"] = gm

        node_count = len([n for n in gm.graph.nodes if n.op == "call_function"])
        print(f"Backward op count: {node_count}")
        return gm

    def simple_fn(x, weight, bias):
        return torch.relu(x @ weight + bias)

    compiled = aot_function(
        simple_fn,
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
    )

    x = torch.randn(4, 8, requires_grad=True)
    w = torch.randn(8, 3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)

    print("Calling compiled function (triggers tracing)...")
    out = compiled(x, w, b)
    print(f"\nOutput shape: {out.shape}")

    print("\nCalling backward (triggers backward compilation)...")
    out.sum().backward()
    print(f"x.grad shape: {x.grad.shape}")
    print(f"w.grad shape: {w.grad.shape}")
    print()


# ============================================================
# Part 3: Viewing the Joint Graph (Before Partitioning)
# ============================================================

def show_joint_graph():
    """Show the joint forward+backward graph before partitioning."""
    print("=" * 70)
    print("PART 3: The Joint Graph (Before Partitioning)")
    print("=" * 70)

    from torch._functorch.aot_autograd import aot_function
    from functools import partial

    joint_graph_holder = {}

    def joint_compiler(gm, example_inputs):
        joint_graph_holder["graph"] = gm
        return gm

    def partition_fn(joint_module, joint_inputs, *, num_fwd_outputs):
        """Custom partitioner that shows the joint graph before splitting."""
        print("\n--- JOINT GRAPH (forward + backward combined) ---")
        print(f"Number of forward outputs: {num_fwd_outputs}")

        all_nodes = list(joint_module.graph.nodes)
        call_nodes = [n for n in all_nodes if n.op == "call_function"]
        print(f"Total call_function nodes: {len(call_nodes)}")

        for node in call_nodes:
            target_name = str(node.target).split(".")[-1] if hasattr(node.target, "__name__") else str(node.target)
            print(f"  {node.name}: {target_name}")

        from torch._functorch.partitioners import default_partition
        return default_partition(joint_module, joint_inputs, num_fwd_outputs=num_fwd_outputs)

    def model_fn(x, weight):
        h = x @ weight
        h = torch.relu(h)
        return h.sum()

    compiled = aot_function(
        model_fn,
        fw_compiler=lambda gm, _: gm,
        bw_compiler=lambda gm, _: gm,
        partition_fn=partition_fn,
    )

    x = torch.randn(4, 8, requires_grad=True)
    w = torch.randn(8, 4, requires_grad=True)

    out = compiled(x, w)
    out.backward()
    print(f"\nGradients computed successfully.")
    print(f"x.grad norm: {x.grad.norm().item():.6f}")
    print(f"w.grad norm: {w.grad.norm().item():.6f}")
    print()


# ============================================================
# Part 4: What Gets Saved for Backward
# ============================================================

def show_saved_tensors():
    """Demonstrate what AOTAutograd saves from forward for backward."""
    print("=" * 70)
    print("PART 4: Saved Tensors — What Forward Passes to Backward")
    print("=" * 70)

    saved_info = {}

    def fw_compiler(gm, example_inputs):
        output_node = [n for n in gm.graph.nodes if n.op == "output"][0]
        output_args = output_node.args[0]
        n_outputs = len(output_args) if isinstance(output_args, (list, tuple)) else 1
        saved_info["fw_outputs"] = n_outputs
        print(f"\nForward graph outputs: {n_outputs}")
        return gm

    def bw_compiler(gm, example_inputs):
        placeholder_nodes = [n for n in gm.graph.nodes if n.op == "placeholder"]
        saved_info["bw_inputs"] = len(placeholder_nodes)
        print(f"Backward graph inputs: {len(placeholder_nodes)}")
        print("  (includes grad_outputs + saved tensors from forward)")
        return gm

    # Simple model
    def simple_model(x, w1, w2):
        h = torch.relu(x @ w1)
        return (h @ w2).sum()

    compiled = aot_function(simple_model, fw_compiler=fw_compiler, bw_compiler=bw_compiler)

    x = torch.randn(4, 8, requires_grad=True)
    w1 = torch.randn(8, 6, requires_grad=True)
    w2 = torch.randn(6, 3, requires_grad=True)

    print("Simple model: relu(x @ w1) @ w2")
    out = compiled(x, w1, w2)
    out.backward()

    # Deeper model
    def deep_model(x, w1, w2, w3):
        h = torch.relu(x @ w1)
        h = torch.sigmoid(h @ w2)
        return (h @ w3).sum()

    saved_info.clear()
    compiled_deep = aot_function(deep_model, fw_compiler=fw_compiler, bw_compiler=bw_compiler)

    w3 = torch.randn(3, 2, requires_grad=True)
    x2 = torch.randn(4, 8, requires_grad=True)

    print("\nDeeper model: sigmoid(relu(x @ w1) @ w2) @ w3")
    out = compiled_deep(x2, w1, w2, w3)
    out.backward()
    print()


# ============================================================
# Part 5: Eager vs AOT Backward — Correctness Check
# ============================================================

def compare_eager_vs_aot():
    """Verify that AOT backward produces the same gradients as eager."""
    print("=" * 70)
    print("PART 5: Eager vs AOT Backward — Same Results")
    print("=" * 70)

    torch.manual_seed(42)

    def model_fn(x, w1, b1, w2, b2):
        h = torch.relu(x @ w1 + b1)
        return (h @ w2 + b2).sum()

    # Shared initial tensors
    x_data = torch.randn(8, 16)
    w1_data = torch.randn(16, 12)
    b1_data = torch.randn(12)
    w2_data = torch.randn(12, 4)
    b2_data = torch.randn(4)

    # Eager
    x_e = x_data.clone().requires_grad_(True)
    w1_e = w1_data.clone().requires_grad_(True)
    b1_e = b1_data.clone().requires_grad_(True)
    w2_e = w2_data.clone().requires_grad_(True)
    b2_e = b2_data.clone().requires_grad_(True)

    loss_e = model_fn(x_e, w1_e, b1_e, w2_e, b2_e)
    loss_e.backward()

    # AOT
    x_a = x_data.clone().requires_grad_(True)
    w1_a = w1_data.clone().requires_grad_(True)
    b1_a = b1_data.clone().requires_grad_(True)
    w2_a = w2_data.clone().requires_grad_(True)
    b2_a = b2_data.clone().requires_grad_(True)

    compiled = aot_function(model_fn, fw_compiler=lambda gm, _: gm, bw_compiler=lambda gm, _: gm)
    loss_a = compiled(x_a, w1_a, b1_a, w2_a, b2_a)
    loss_a.backward()

    print(f"\nLoss match:    {torch.allclose(loss_e, loss_a)}")
    print(f"x grad match:  {torch.allclose(x_e.grad, x_a.grad)}")
    print(f"w1 grad match: {torch.allclose(w1_e.grad, w1_a.grad)}")
    print(f"b1 grad match: {torch.allclose(b1_e.grad, b1_a.grad)}")
    print(f"w2 grad match: {torch.allclose(w2_e.grad, w2_a.grad)}")
    print(f"b2 grad match: {torch.allclose(b2_e.grad, b2_a.grad)}")

    max_diff = max(
        (x_e.grad - x_a.grad).abs().max().item(),
        (w1_e.grad - w1_a.grad).abs().max().item(),
        (b1_e.grad - b1_a.grad).abs().max().item(),
        (w2_e.grad - w2_a.grad).abs().max().item(),
        (b2_e.grad - b2_a.grad).abs().max().item(),
    )
    print(f"Max gradient difference: {max_diff:.2e}")
    print()


# ============================================================
# Part 6: Min-Cut Effect — Saved vs Recomputed
# ============================================================

def show_min_cut_effect():
    """Demonstrate the min-cut partitioner's save vs recompute decisions."""
    print("=" * 70)
    print("PART 6: Min-Cut — Which Ops Are Saved vs Recomputed")
    print("=" * 70)

    def analyze_graph(gm, label):
        """Count and categorize operations in a graph."""
        call_nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
        output_node = [n for n in gm.graph.nodes if n.op == "output"][0]
        placeholder_nodes = [n for n in gm.graph.nodes if n.op == "placeholder"]

        output_args = output_node.args[0]
        n_outputs = len(output_args) if isinstance(output_args, (list, tuple)) else 1

        print(f"\n  [{label}]")
        print(f"    Inputs (placeholders): {len(placeholder_nodes)}")
        print(f"    Operations: {len(call_nodes)}")
        print(f"    Outputs: {n_outputs}")

        return {"ops": len(call_nodes), "inputs": len(placeholder_nodes), "outputs": n_outputs}

    def model_with_many_ops(x, weight):
        h = x @ weight
        h = h + 1.0
        h = torch.relu(h)
        h = h * 2.0
        h = torch.sigmoid(h)
        h = h - 0.5
        return h.sum()

    fw_info = {}
    bw_info = {}

    def fw_compiler(gm, example_inputs):
        fw_info.update(analyze_graph(gm, "Forward"))
        return gm

    def bw_compiler(gm, example_inputs):
        bw_info.update(analyze_graph(gm, "Backward"))
        return gm

    compiled = aot_function(model_with_many_ops, fw_compiler=fw_compiler, bw_compiler=bw_compiler)

    x = torch.randn(4, 8, requires_grad=True)
    w = torch.randn(8, 4, requires_grad=True)

    print("\nModel: mm → add → relu → mul → sigmoid → sub → sum")
    out = compiled(x, w)
    out.backward()

    if fw_info and bw_info:
        n_model_outputs = 1
        saved_count = fw_info["outputs"] - n_model_outputs
        print(f"\n  Summary:")
        print(f"    Forward saves {saved_count} tensor(s) for backward")
        print(f"    (Remaining activations are recomputed in backward)")
    print()


# ============================================================
# Part 7: TORCH_LOGS Hints for Debugging
# ============================================================

def show_debug_hints():
    """Show how to enable logging for AOTAutograd debugging."""
    print("=" * 70)
    print("PART 7: Debugging AOTAutograd with TORCH_LOGS")
    print("=" * 70)

    print("""
To debug AOTAutograd, use these environment variables:

  # See forward/backward graphs produced by AOTAutograd
  TORCH_LOGS="aot" python script.py

  # See generated Inductor code for both graphs
  TORCH_LOGS="output_code" python script.py

  # See graph breaks (where compilation stops)
  TORCH_LOGS="graph_breaks" python script.py

  # See dynamic shape decisions
  TORCH_LOGS="dynamic" python script.py

  # Everything at once
  TORCH_LOGS="aot,output_code,graph_breaks,dynamic" python script.py

Programmatic logging:
  import logging
  torch._logging.set_logs(aot=logging.DEBUG)

Partitioner debugging:
  torch._functorch.config.debug_partitioner = True
""")


# ============================================================
# Part 8: Memory Comparison — Saved Tensor Counts
# ============================================================

def memory_comparison():
    """Compare saved tensor counts for different model architectures."""
    print("=" * 70)
    print("PART 8: Memory — Saved Tensor Counts Across Architectures")
    print("=" * 70)

    def count_saved_tensors(fn, *args):
        """Return the number of tensors saved from forward for backward."""
        info = {"fw_outputs": 0, "model_outputs": 0}

        def fw_compiler(gm, example_inputs):
            output_node = [n for n in gm.graph.nodes if n.op == "output"][0]
            output_args = output_node.args[0]
            info["fw_outputs"] = len(output_args) if isinstance(output_args, (list, tuple)) else 1
            return gm

        compiled = aot_function(fn, fw_compiler=fw_compiler, bw_compiler=lambda gm, _: gm)
        out = compiled(*args)

        if isinstance(out, torch.Tensor) and out.dim() == 0:
            info["model_outputs"] = 1
        elif isinstance(out, torch.Tensor):
            info["model_outputs"] = 1
        else:
            info["model_outputs"] = len(out) if isinstance(out, (list, tuple)) else 1

        out.sum().backward() if isinstance(out, torch.Tensor) else out[0].sum().backward()
        return info["fw_outputs"] - info["model_outputs"]

    # Model 1: Simple linear
    def linear_model(x, w, b):
        return (x @ w + b).sum()

    x = torch.randn(4, 8, requires_grad=True)
    w1 = torch.randn(8, 4, requires_grad=True)
    b1 = torch.randn(4, requires_grad=True)
    saved1 = count_saved_tensors(linear_model, x.clone().requires_grad_(True), w1, b1)

    # Model 2: With ReLU
    def relu_model(x, w, b):
        return torch.relu(x @ w + b).sum()

    saved2 = count_saved_tensors(relu_model, x.clone().requires_grad_(True), w1, b1)

    # Model 3: Two layers
    def two_layer(x, w1, b1, w2, b2):
        h = torch.relu(x @ w1 + b1)
        return (h @ w2 + b2).sum()

    w2 = torch.randn(4, 3, requires_grad=True)
    b2 = torch.randn(3, requires_grad=True)
    saved3 = count_saved_tensors(
        two_layer, x.clone().requires_grad_(True), w1, b1, w2, b2
    )

    # Model 4: Three layers with different activations
    def three_layer(x, w1, w2, w3):
        h = torch.relu(x @ w1)
        h = torch.sigmoid(h @ w2)
        return (h @ w3).sum()

    w3 = torch.randn(3, 2, requires_grad=True)
    saved4 = count_saved_tensors(
        three_layer, x.clone().requires_grad_(True), w1,
        torch.randn(4, 3, requires_grad=True), w3
    )

    print(f"\n  {'Architecture':<40} {'Saved Tensors':>14}")
    print(f"  {'─' * 40} {'─' * 14}")
    print(f"  {'Linear (x @ w + b)':<40} {saved1:>14}")
    print(f"  {'ReLU(linear)':<40} {saved2:>14}")
    print(f"  {'2-layer MLP (relu)':<40} {saved3:>14}")
    print(f"  {'3-layer (relu + sigmoid)':<40} {saved4:>14}")

    print("""
  Note: The min-cut partitioner decides which activations to save vs
  recompute. Cheap ops (relu, add) are typically recomputed; expensive
  ops (matmul) are typically saved. More layers = more saved tensors,
  but the count grows sub-linearly thanks to rematerialization.
""")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print()
    print("Module 38: AOTAutograd Explained")
    print("================================")
    print("Tracing forward AND backward at compile time.\n")

    show_standard_autograd()
    show_aot_function_graphs()
    show_joint_graph()
    show_saved_tensors()
    compare_eager_vs_aot()
    show_min_cut_effect()
    show_debug_hints()
    memory_comparison()

    print("=" * 70)
    print("Done! AOTAutograd compiles both forward and backward into")
    print("optimized FX graphs that Inductor can fuse and accelerate.")
    print("=" * 70)
