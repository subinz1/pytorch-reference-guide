"""
Dynamic Shapes with torch.export
==================================

Demonstrates the Dim API for exporting models with variable input sizes:
- Single dynamic dimension (batch size)
- Multiple dynamic dimensions (batch + sequence length)
- Shared dimensions across inputs
- Dim.AUTO for automatic inference
- Verifying dynamic behavior

Run:
    python dynamic_shapes.py
"""

import torch
import torch.nn as nn
from torch.export import Dim, export


class ClassificationModel(nn.Module):
    def __init__(self, input_dim: int = 64, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SequenceModel(nn.Module):
    def __init__(self, d_model: int = 32, nhead: int = 4):
        super().__init__()
        self.embed = nn.Linear(d_model, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=64, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        h = self.encoder(h)
        return self.head(h[:, -1, :])


class DualInputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_a = nn.Linear(16, 8)
        self.proj_b = nn.Linear(32, 8)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.proj_a(a) + self.proj_b(b)


def demo_single_dynamic_dim():
    print("=" * 60)
    print("  1. Single Dynamic Dimension (batch size)")
    print("=" * 60)

    model = ClassificationModel(input_dim=64, num_classes=10)
    model.eval()

    batch = Dim("batch", min=1, max=256)

    exported = export(
        model,
        (torch.randn(4, 64),),
        dynamic_shapes={"x": {0: batch}},
    )

    test_sizes = [1, 8, 32, 128, 256]
    print(f"  Exported with batch dim dynamic (min=1, max=256)")
    print(f"  Testing with batch sizes: {test_sizes}")

    for bs in test_sizes:
        result = exported.module()(torch.randn(bs, 64))
        print(f"    batch={bs:>3d} → output shape: {list(result.shape)}")

    print()


def demo_multiple_dynamic_dims():
    print("=" * 60)
    print("  2. Multiple Dynamic Dimensions (batch + seq_len)")
    print("=" * 60)

    model = SequenceModel(d_model=32, nhead=4)
    model.eval()

    batch = Dim("batch", min=1, max=64)
    seq_len = Dim("seq_len", min=4, max=512)

    exported = export(
        model,
        (torch.randn(4, 16, 32),),
        dynamic_shapes={"x": {0: batch, 1: seq_len}},
    )

    test_configs = [(1, 4), (8, 32), (16, 128), (32, 256)]
    print(f"  Exported with batch and seq_len dynamic")
    print(f"  d_model=32 is static")

    for bs, sl in test_configs:
        result = exported.module()(torch.randn(bs, sl, 32))
        print(f"    batch={bs:>2d}, seq_len={sl:>3d} → output: {list(result.shape)}")

    print()


def demo_shared_dimensions():
    print("=" * 60)
    print("  3. Shared Dimensions Across Inputs")
    print("=" * 60)

    model = DualInputModel()
    model.eval()

    batch = Dim("batch", min=1, max=128)

    exported = export(
        model,
        (torch.randn(4, 16), torch.randn(4, 32)),
        dynamic_shapes={
            "a": {0: batch},
            "b": {0: batch},
        },
    )

    test_sizes = [1, 8, 64]
    print("  Both inputs share the same 'batch' Dim.")
    print("  This enforces a.shape[0] == b.shape[0]")

    for bs in test_sizes:
        result = exported.module()(torch.randn(bs, 16), torch.randn(bs, 32))
        print(f"    batch={bs:>2d} → output: {list(result.shape)}")

    # Demonstrate constraint violation
    print("\n  Mismatched batch sizes will raise an error:")
    try:
        exported.module()(torch.randn(3, 16), torch.randn(5, 32))
        print("    (no error — constraint not enforced at this level)")
    except Exception as e:
        error_msg = str(e).split("\n")[0][:80]
        print(f"    Error: {error_msg}")

    print()


def demo_auto_dim():
    print("=" * 60)
    print("  4. Automatic Dynamic Shapes (Dim.AUTO)")
    print("=" * 60)

    model = ClassificationModel(input_dim=64, num_classes=10)
    model.eval()

    exported = export(
        model,
        (torch.randn(4, 64),),
        dynamic_shapes={"x": {0: Dim.AUTO}},
    )

    print("  Using Dim.AUTO lets PyTorch infer constraints.")
    print("  Testing with various batch sizes:")

    for bs in [1, 4, 16, 64]:
        result = exported.module()(torch.randn(bs, 64))
        print(f"    batch={bs:>2d} → output: {list(result.shape)}")

    print()


def demo_static_vs_dynamic():
    print("=" * 60)
    print("  5. Static vs Dynamic: Side-by-Side")
    print("=" * 60)

    model = ClassificationModel(input_dim=64, num_classes=10)
    model.eval()

    # Static export
    exported_static = export(model, (torch.randn(4, 64),))

    # Dynamic export
    batch = Dim("batch", min=1, max=128)
    exported_dynamic = export(
        model,
        (torch.randn(4, 64),),
        dynamic_shapes={"x": {0: batch}},
    )

    print("  Static export (batch=4 only):")
    result = exported_static.module()(torch.randn(4, 64))
    print(f"    batch=4 → output: {list(result.shape)}")

    print("\n  Trying batch=8 with static export:")
    try:
        exported_static.module()(torch.randn(8, 64))
        print("    Succeeded (may work in some cases)")
    except Exception as e:
        error_msg = str(e).split("\n")[0][:70]
        print(f"    Error: {error_msg}")

    print("\n  Dynamic export (batch=1..128):")
    for bs in [1, 4, 8, 128]:
        result = exported_dynamic.module()(torch.randn(bs, 64))
        print(f"    batch={bs:>3d} → output: {list(result.shape)}")

    print()


def demo_range_constraints():
    print("=" * 60)
    print("  6. Range Constraints")
    print("=" * 60)

    model = ClassificationModel(input_dim=64, num_classes=10)
    model.eval()

    batch = Dim("batch", min=2, max=32)

    exported = export(
        model,
        (torch.randn(4, 64),),
        dynamic_shapes={"x": {0: batch}},
    )

    print("  Exported with batch in [2, 32]")

    # Within range
    print("\n  Within range:")
    for bs in [2, 16, 32]:
        result = exported.module()(torch.randn(bs, 64))
        print(f"    batch={bs:>2d} → OK, output: {list(result.shape)}")

    # Out of range
    print("\n  Outside range (batch=1, below min=2):")
    try:
        exported.module()(torch.randn(1, 64))
        print("    Succeeded (range may not be enforced at runtime)")
    except Exception as e:
        error_msg = str(e).split("\n")[0][:70]
        print(f"    Error: {error_msg}")

    print("\n  Outside range (batch=64, above max=32):")
    try:
        exported.module()(torch.randn(64, 64))
        print("    Succeeded (range may not be enforced at runtime)")
    except Exception as e:
        error_msg = str(e).split("\n")[0][:70]
        print(f"    Error: {error_msg}")

    print()


def main():
    print("\nDynamic Shapes with torch.export")
    print("=" * 60)
    print()

    demo_single_dynamic_dim()
    demo_multiple_dynamic_dims()
    demo_shared_dimensions()
    demo_auto_dim()
    demo_static_vs_dynamic()
    demo_range_constraints()

    print("All dynamic shapes demos completed!\n")


if __name__ == "__main__":
    main()
