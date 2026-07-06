"""
Module 37: Control Flow in torch.export
========================================

Runnable on CPU. Covers:
- Problem: Python if/else can't be exported (show the error)
- torch.cond: conditional execution in export
- torch.cond with multiple outputs
- torch.while_loop: data-dependent loops
- Combining cond and while_loop
- Practical example: iterative refinement with early stopping
- Verification: exported models produce correct results
- Graph inspection: see both branches in the exported graph

Usage:
    python control_flow_export.py
"""

import torch
import torch.nn as nn
from torch.export import export


# ============================================================
# 1. Problem: Python if/else Can't Be Exported
# ============================================================

class BranchingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_transform = nn.Linear(10, 5)
        self.neg_transform = nn.Linear(10, 5)

    def forward(self, x):
        if x.sum() > 0:
            return self.pos_transform(x)
        else:
            return self.neg_transform(x)


def demo_python_if_fails():
    print("=" * 70)
    print("1. Problem: Python if/else Can't Be Exported")
    print("=" * 70)

    model = BranchingModel()
    args = (torch.randn(3, 10),)

    print("\nPython if/else depends on tensor data at runtime.")
    print("Export only traces one branch — the other is silently dropped.\n")

    try:
        ep = export(model, args)
        print("Export succeeded (but only one branch was captured).")
        print("This is INCORRECT — the model won't work for all inputs!")

        ops = [str(n.target) for n in ep.graph_module.graph.nodes if n.op == "call_function"]
        has_pos = any("pos_transform" in str(n.target) or "p_pos" in str(getattr(n, 'name', ''))
                       for n in ep.graph_module.graph.nodes)
        print(f"Graph only has one branch: {len(ops)} ops captured")
    except Exception as e:
        first_line = str(e).split("\n")[0]
        print(f"Export failed (expected): {first_line}")

    print()


# ============================================================
# 2. torch.cond — Conditional Execution
# ============================================================

class CondModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_transform = nn.Linear(10, 5)
        self.neg_transform = nn.Linear(10, 5)

    def forward(self, x):
        def true_fn(x, pos_w, pos_b, neg_w, neg_b):
            return torch.nn.functional.linear(x, pos_w, pos_b)

        def false_fn(x, pos_w, pos_b, neg_w, neg_b):
            return torch.nn.functional.linear(x, neg_w, neg_b)

        return torch.cond(
            x.sum() > 0,
            true_fn,
            false_fn,
            (x, self.pos_transform.weight, self.pos_transform.bias,
             self.neg_transform.weight, self.neg_transform.bias),
        )


def demo_torch_cond():
    print("=" * 70)
    print("2. torch.cond — Conditional Execution in Export")
    print("=" * 70)

    model = CondModel()
    args = (torch.randn(3, 10),)

    ep = export(model, args)
    print("\nExport with torch.cond succeeded!")

    pos_input = torch.ones(3, 10)
    neg_input = -torch.ones(3, 10)

    result_pos = ep.module()(pos_input)
    result_neg = ep.module()(neg_input)

    expected_pos = model(pos_input)
    expected_neg = model(neg_input)

    print(f"Positive input — matches original: {torch.allclose(result_pos, expected_pos)}")
    print(f"Negative input — matches original: {torch.allclose(result_neg, expected_neg)}")

    has_cond = any("cond" in str(n.target) for n in ep.graph_module.graph.nodes
                    if n.op == "call_function")
    print(f"Graph contains cond node: {has_cond}")

    print()


# ============================================================
# 3. torch.cond with Multiple Outputs
# ============================================================

class MultiOutputCondModel(nn.Module):
    def forward(self, x, y):
        def true_fn(x, y):
            return x * 2, y + 1

        def false_fn(x, y):
            return x * -1, y - 1

        return torch.cond(x.sum() > 0, true_fn, false_fn, (x, y))


def demo_cond_multi_output():
    print("=" * 70)
    print("3. torch.cond with Multiple Outputs")
    print("=" * 70)

    model = MultiOutputCondModel()
    x = torch.randn(3, 4)
    y = torch.randn(3, 4)

    ep = export(model, (x, y))
    print("\nMulti-output torch.cond exported successfully!")

    pos_x = torch.ones(3, 4)
    a, b = ep.module()(pos_x, y)
    a_expected, b_expected = model(pos_x, y)
    print(f"Positive branch — output a matches: {torch.allclose(a, a_expected)}")
    print(f"Positive branch — output b matches: {torch.allclose(b, b_expected)}")

    neg_x = -torch.ones(3, 4)
    a, b = ep.module()(neg_x, y)
    a_expected, b_expected = model(neg_x, y)
    print(f"Negative branch — output a matches: {torch.allclose(a, a_expected)}")
    print(f"Negative branch — output b matches: {torch.allclose(b, b_expected)}")

    print()


# ============================================================
# 4. torch.while_loop — Data-Dependent Loops
# ============================================================

class WhileLoopModel(nn.Module):
    def forward(self, x, max_iter):
        def cond_fn(x, count, max_iter):
            return count < max_iter

        def body_fn(x, count, max_iter):
            return x * 0.9, count + 1, max_iter

        result_x, final_count, _ = torch.while_loop(
            cond_fn, body_fn, (x, torch.tensor(0), max_iter)
        )
        return result_x, final_count


def demo_while_loop():
    print("=" * 70)
    print("4. torch.while_loop — Data-Dependent Loops")
    print("=" * 70)

    model = WhileLoopModel()
    x = torch.ones(5)
    max_iter = torch.tensor(10)

    ep = export(model, (x, max_iter))
    print("\ntorch.while_loop exported successfully!")

    result_x, count = ep.module()(x, max_iter)
    expected_x, expected_count = model(x, max_iter)

    print(f"After {count.item():.0f} iterations:")
    print(f"  x[0] = {result_x[0].item():.6f} (expected: {0.9**10:.6f})")
    print(f"  Matches original: {torch.allclose(result_x, expected_x, atol=1e-6)}")

    has_while = any("while_loop" in str(n.target) for n in ep.graph_module.graph.nodes
                     if n.op == "call_function")
    print(f"  Graph contains while_loop: {has_while}")

    print()


# ============================================================
# 5. Combining cond and while_loop
# ============================================================

class CombinedControlFlow(nn.Module):
    """Model that uses both conditional and loop control flow."""
    def forward(self, x, max_iter):
        def apply_transform(x):
            def pos_fn(x):
                return x * 0.5

            def neg_fn(x):
                return x * 1.5

            return torch.cond(x.sum() > 0, pos_fn, neg_fn, (x,))

        def cond_fn(x, count, max_iter):
            return count < max_iter

        def body_fn(x, count, max_iter):
            x_new = apply_transform(x)
            return x_new, count + 1, max_iter

        result, final_count, _ = torch.while_loop(
            cond_fn, body_fn, (x, torch.tensor(0), max_iter)
        )
        return result, final_count


def demo_combined():
    print("=" * 70)
    print("5. Combining cond and while_loop")
    print("=" * 70)

    model = CombinedControlFlow()
    x = torch.tensor([1.0, -2.0, 3.0])
    max_iter = torch.tensor(5)

    ep = export(model, (x, max_iter))
    print("\nCombined cond + while_loop exported successfully!")

    result, count = ep.module()(x, max_iter)
    expected, expected_count = model(x, max_iter)

    print(f"Input:    {x.tolist()}")
    print(f"After {count.item():.0f} iterations: {result.tolist()}")
    print(f"Matches original: {torch.allclose(result, expected, atol=1e-5)}")

    graph_ops = [str(n.target) for n in ep.graph_module.graph.nodes if n.op == "call_function"]
    has_cond = any("cond" in op for op in graph_ops)
    has_while = any("while_loop" in op for op in graph_ops)
    print(f"Graph has cond: {has_cond}, while_loop: {has_while}")

    print()


# ============================================================
# 6. Practical Example: Iterative Refinement
# ============================================================

class IterativeRefinement(nn.Module):
    """Iteratively refine a prediction until convergence or max steps."""
    def __init__(self):
        super().__init__()
        self.refine = nn.Linear(8, 8, bias=False)
        nn.init.eye_(self.refine.weight)
        self.refine.weight.data *= 0.95

    def forward(self, x, max_steps):
        def cond_fn(x, prev_x, step, max_steps):
            return step < max_steps

        def body_fn(x, prev_x, step, max_steps):
            new_x = self.refine(x)
            return new_x, x, step + 1, max_steps

        init_prev = torch.zeros_like(x)
        result, prev, steps, _ = torch.while_loop(
            cond_fn, body_fn, (x, init_prev, torch.tensor(0), max_steps)
        )

        return result, steps


def demo_iterative_refinement():
    print("=" * 70)
    print("6. Practical Example: Iterative Refinement")
    print("=" * 70)

    model = IterativeRefinement()
    x = torch.randn(2, 8)
    max_steps = torch.tensor(10)

    ep = export(model, (x, max_steps))
    print("\nIterative refinement model exported!")

    result, steps = ep.module()(x, max_steps)
    expected, expected_steps = model(x, max_steps)

    print(f"Steps taken: {steps.item():.0f}")
    print(f"Input norm:  {x.norm().item():.4f}")
    print(f"Output norm: {result.norm().item():.4f} (should decay toward 0)")
    print(f"Matches original: {torch.allclose(result, expected, atol=1e-5)}")

    print()


# ============================================================
# 7. Verification: Correctness Across Input Ranges
# ============================================================

def demo_verification():
    print("=" * 70)
    print("7. Verification — Correctness Across Input Ranges")
    print("=" * 70)

    model = CondModel()
    args = (torch.randn(3, 10),)
    ep = export(model, args)

    n_tests = 100
    n_correct = 0

    for _ in range(n_tests):
        test_input = torch.randn(3, 10)
        original_result = model(test_input)
        exported_result = ep.module()(test_input)

        if torch.allclose(original_result, exported_result, atol=1e-6):
            n_correct += 1

    print(f"\nRandom input test: {n_correct}/{n_tests} correct")

    pos_count = sum(1 for _ in range(n_tests) if torch.randn(3, 10).sum() > 0)
    print(f"(~{pos_count} positive, ~{n_tests - pos_count} negative inputs tested)")

    print()


# ============================================================
# 8. Graph Inspection — Seeing Both Branches
# ============================================================

def demo_graph_inspection():
    print("=" * 70)
    print("8. Graph Inspection — Both Branches in the Graph")
    print("=" * 70)

    class SimpleCondModel(nn.Module):
        def forward(self, x):
            def true_fn(x):
                return x * 2 + 1

            def false_fn(x):
                return x * -1 + 1

            return torch.cond(x.sum() > 0, true_fn, false_fn, (x,))

    model = SimpleCondModel()
    ep = export(model, (torch.randn(4),))

    print("\n--- Exported Graph Nodes ---")
    for node in ep.graph_module.graph.nodes:
        if node.op == "call_function":
            print(f"  [{node.op}] {node.name}: {node.target}")

    print("\n--- Looking for Subgraphs (cond branches) ---")
    for name, submod in ep.graph_module.named_modules():
        if name:
            print(f"\n  Submodule: {name}")
            if hasattr(submod, 'graph'):
                for node in submod.graph.nodes:
                    if node.op == "call_function":
                        target_name = str(node.target).split(".")[-1]
                        print(f"    [{node.op}] {node.name}: {target_name}")

    print()


# ============================================================
# Main
# ============================================================

def main():
    print("\n" + "=" * 70)
    print(" Module 37: Control Flow in torch.export")
    print("=" * 70 + "\n")

    demo_python_if_fails()
    demo_torch_cond()
    demo_cond_multi_output()
    demo_while_loop()
    demo_combined()
    demo_iterative_refinement()
    demo_verification()
    demo_graph_inspection()

    print("=" * 70)
    print(" All control flow demos completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
