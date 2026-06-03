"""
torch.export Basics
====================

Demonstrates the fundamentals of torch.export:
- Exporting a simple model
- Running the exported model
- Examining the ExportedProgram
- Handling control flow with torch.cond

Run:
    python export_basics.py
"""

import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self, in_features: int = 10, hidden: int = 32, out_features: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModelWithBranch(nn.Module):
    """A model using torch.cond for data-dependent control flow."""

    def __init__(self):
        super().__init__()
        self.linear_pos = nn.Linear(10, 5)
        self.linear_neg = nn.Linear(10, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cond(
            x.sum() > 0,
            self._positive_branch,
            self._negative_branch,
            (x,),
        )

    def _positive_branch(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_pos(x)

    def _negative_branch(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_neg(x) * 2


class ModelWithShapeBranch(nn.Module):
    """Control flow based on shapes (not values) works without torch.cond."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.linear_large = nn.Linear(10, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] > 4:
            return self.linear_large(x[:4])
        return self.linear(x)


def demo_basic_export():
    print("=" * 60)
    print("  1. Basic Export")
    print("=" * 60)

    model = SimpleModel()
    example_input = (torch.randn(3, 10),)

    exported = torch.export.export(model, example_input)

    print(f"  Type: {type(exported).__name__}")
    print(f"  Module type: {type(exported.module()).__name__}")

    # Run with example input
    result = exported.module()(*example_input)
    print(f"  Input shape:  {example_input[0].shape}")
    print(f"  Output shape: {result.shape}")

    # Verify same output as eager
    eager_result = model(*example_input)
    max_diff = (result - eager_result).abs().max().item()
    print(f"  Max difference from eager: {max_diff:.2e}")
    print()


def demo_exported_program_contents():
    print("=" * 60)
    print("  2. ExportedProgram Contents")
    print("=" * 60)

    model = SimpleModel()
    example_input = (torch.randn(3, 10),)
    exported = torch.export.export(model, example_input)

    # State dict
    print("  State dict keys:")
    for key in exported.state_dict:
        shape = exported.state_dict[key].shape
        print(f"    {key}: {list(shape)}")

    # Graph signature
    sig = exported.graph_signature
    print(f"\n  Input specs ({len(sig.input_specs)}):")
    for spec in sig.input_specs:
        print(f"    {spec.kind.name}: {spec.arg}")

    print(f"\n  Output specs ({len(sig.output_specs)}):")
    for spec in sig.output_specs:
        print(f"    {spec.kind.name}: {spec.arg}")

    # Generated code
    print("\n  Generated code:")
    code_lines = exported.graph_module.code.strip().split("\n")
    for line in code_lines:
        print(f"    {line}")
    print()


def demo_control_flow():
    print("=" * 60)
    print("  3. Control Flow with torch.cond")
    print("=" * 60)

    model = ModelWithBranch()

    positive_input = (torch.ones(2, 10),)
    negative_input = (torch.ones(2, 10) * -1,)

    exported = torch.export.export(model, positive_input)

    result_pos = exported.module()(torch.ones(2, 10))
    result_neg = exported.module()(torch.ones(2, 10) * -1)

    print("  Model with torch.cond exported successfully.")
    print(f"  Positive input sum > 0: output mean = {result_pos.mean():.4f}")
    print(f"  Negative input sum > 0: output mean = {result_neg.mean():.4f}")
    print("  (Different branches produce different outputs.)")
    print()


def demo_shape_branch():
    print("=" * 60)
    print("  4. Shape-based Control Flow (no torch.cond needed)")
    print("=" * 60)

    model = ModelWithShapeBranch()

    small_input = (torch.randn(3, 10),)
    exported = torch.export.export(model, small_input)

    result = exported.module()(torch.randn(3, 10))
    print(f"  Exported with shape {small_input[0].shape}")
    print(f"  Output shape: {result.shape}")
    print("  Shape-based branches are resolved at export time.")
    print()


def demo_multiple_inputs():
    print("=" * 60)
    print("  5. Multiple Inputs")
    print("=" * 60)

    class MultiInputModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj_a = nn.Linear(10, 5)
            self.proj_b = nn.Linear(20, 5)

        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return self.proj_a(a) + self.proj_b(b)

    model = MultiInputModel()
    example = (torch.randn(3, 10), torch.randn(3, 20))

    exported = torch.export.export(model, example)

    result = exported.module()(torch.randn(3, 10), torch.randn(3, 20))
    print(f"  Input shapes: {example[0].shape}, {example[1].shape}")
    print(f"  Output shape: {result.shape}")
    print()


def demo_kwargs_export():
    print("=" * 60)
    print("  6. Export with Keyword Arguments")
    print("=" * 60)

    class KwargsModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)

        def forward(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
            return self.linear(x) * scale

    model = KwargsModel()
    args = (torch.randn(3, 10),)
    kwargs = {"scale": 2.0}

    exported = torch.export.export(model, args, kwargs=kwargs)

    result = exported.module()(torch.randn(3, 10), scale=2.0)
    print(f"  Exported with scale=2.0")
    print(f"  Output shape: {result.shape}")
    print()


def main():
    print("\ntorch.export Basics Demo")
    print("=" * 60)
    print()

    demo_basic_export()
    demo_exported_program_contents()
    demo_control_flow()
    demo_shape_branch()
    demo_multiple_inputs()
    demo_kwargs_export()

    print("All export demos completed successfully!\n")


if __name__ == "__main__":
    main()
