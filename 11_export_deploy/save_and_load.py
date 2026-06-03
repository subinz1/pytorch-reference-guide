"""
Saving and Loading Exported Models (PT2 Archives)
===================================================

Demonstrates:
- Exporting and saving to PT2 archive format
- Loading without the original model code
- Saving with dynamic shapes
- Verifying loaded model correctness
- Inspecting saved archive contents

Run:
    python save_and_load.py
"""

import os
import tempfile

import torch
import torch.nn as nn
from torch.export import Dim, export


class ImageClassifier(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int = 1000, d_model: int = 64, num_classes: int = 5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=128, batch_first=True
        )
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        h = self.encoder(h)
        return self.head(h[:, 0, :])


def demo_basic_save_load():
    print("=" * 60)
    print("  1. Basic Save and Load")
    print("=" * 60)

    model = ImageClassifier(num_classes=10).eval()
    example_input = (torch.randn(1, 3, 32, 32),)

    exported = export(model, example_input)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "image_classifier.pt2")

        # Save
        torch.export.save(exported, save_path)
        file_size = os.path.getsize(save_path)
        print(f"  Saved to: {save_path}")
        print(f"  File size: {file_size / 1024:.1f} KB")

        # Load (no need for model definition!)
        loaded = torch.export.load(save_path)

        # Verify
        test_input = torch.randn(1, 3, 32, 32)
        original_output = exported.module()(test_input)
        loaded_output = loaded.module()(test_input)

        max_diff = (original_output - loaded_output).abs().max().item()
        print(f"  Max difference (original vs loaded): {max_diff:.2e}")
        print(f"  Outputs match: {max_diff < 1e-6}")

    print()


def demo_save_with_dynamic_shapes():
    print("=" * 60)
    print("  2. Save with Dynamic Shapes")
    print("=" * 60)

    model = ImageClassifier(num_classes=10).eval()

    batch = Dim("batch", min=1, max=64)
    exported = export(
        model,
        (torch.randn(4, 3, 32, 32),),
        dynamic_shapes={"x": {0: batch}},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "classifier_dynamic.pt2")

        torch.export.save(exported, save_path)
        loaded = torch.export.load(save_path)

        print("  Saved with dynamic batch dimension [1, 64]")
        print("  Testing loaded model with various batch sizes:")

        for bs in [1, 8, 32, 64]:
            result = loaded.module()(torch.randn(bs, 3, 32, 32))
            print(f"    batch={bs:>2d} → output: {list(result.shape)}")

    print()


def demo_save_sequence_model():
    print("=" * 60)
    print("  3. Save Sequence Model with Multiple Dynamic Dims")
    print("=" * 60)

    model = TextEncoder(vocab_size=1000, d_model=64, num_classes=5).eval()

    batch = Dim("batch", min=1, max=32)
    seq_len = Dim("seq_len", min=4, max=256)

    exported = export(
        model,
        (torch.randint(0, 1000, (2, 16)),),
        dynamic_shapes={"x": {0: batch, 1: seq_len}},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "text_encoder.pt2")

        torch.export.save(exported, save_path)
        file_size = os.path.getsize(save_path)
        print(f"  Saved text encoder: {file_size / 1024:.1f} KB")

        loaded = torch.export.load(save_path)

        test_configs = [(1, 4), (4, 32), (16, 128), (32, 256)]
        print("  Testing with various (batch, seq_len) combinations:")
        for bs, sl in test_configs:
            result = loaded.module()(torch.randint(0, 1000, (bs, sl)))
            print(f"    ({bs:>2d}, {sl:>3d}) → output: {list(result.shape)}")

    print()


def demo_inspect_archive():
    print("=" * 60)
    print("  4. Inspecting a Saved Archive")
    print("=" * 60)

    model = ImageClassifier(num_classes=10).eval()
    exported = export(model, (torch.randn(1, 3, 32, 32),))

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "model.pt2")
        torch.export.save(exported, save_path)

        loaded = torch.export.load(save_path)

        # Inspect contents
        print("  State dict keys:")
        for key in loaded.state_dict:
            tensor = loaded.state_dict[key]
            print(f"    {key}: shape={list(tensor.shape)}, dtype={tensor.dtype}")

        print(f"\n  Graph signature:")
        sig = loaded.graph_signature
        n_params = sum(1 for s in sig.input_specs if s.kind.name == "PARAMETER")
        n_buffers = sum(1 for s in sig.input_specs if s.kind.name == "BUFFER")
        n_inputs = sum(1 for s in sig.input_specs if s.kind.name == "USER_INPUT")
        print(f"    Parameters: {n_params}")
        print(f"    Buffers: {n_buffers}")
        print(f"    User inputs: {n_inputs}")
        print(f"    Outputs: {len(sig.output_specs)}")

        # Count ops in the graph
        ops = set()
        for node in loaded.graph_module.graph.nodes:
            if node.op == "call_function":
                ops.add(str(node.target).split(".")[-1])
        print(f"\n  Operations in graph ({len(ops)} unique):")
        for op in sorted(ops):
            print(f"    {op}")

    print()


def demo_multiple_models():
    print("=" * 60)
    print("  5. Saving Multiple Models")
    print("=" * 60)

    models = {
        "image_classifier": (
            ImageClassifier(num_classes=10).eval(),
            (torch.randn(1, 3, 32, 32),),
        ),
        "text_encoder": (
            TextEncoder(vocab_size=1000).eval(),
            (torch.randint(0, 1000, (1, 16)),),
        ),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        saved_paths = {}

        print("  Exporting and saving models:")
        for name, (model, example) in models.items():
            exported = export(model, example)
            path = os.path.join(tmpdir, f"{name}.pt2")
            torch.export.save(exported, path)
            size = os.path.getsize(path)
            saved_paths[name] = path
            param_count = sum(p.numel() for p in model.parameters())
            print(f"    {name}: {size / 1024:.1f} KB, {param_count:,} parameters")

        print("\n  Loading and verifying:")
        for name, path in saved_paths.items():
            loaded = torch.export.load(path)
            model, example = models[name]
            result = loaded.module()(*example)
            original = model(*example)
            diff = (result - original).abs().max().item()
            print(f"    {name}: outputs match = {diff < 1e-6} (diff={diff:.2e})")

    print()


def demo_workflow_summary():
    print("=" * 60)
    print("  6. Complete Workflow Summary")
    print("=" * 60)

    print("""
  The typical export → save → load workflow:

  TRAINING SIDE:
    1. model = MyModel()
    2. # ... train the model ...
    3. model.eval()
    4. exported = torch.export.export(model, example_input,
                                      dynamic_shapes=...)
    5. torch.export.save(exported, "model.pt2")

  DEPLOYMENT SIDE:
    1. loaded = torch.export.load("model.pt2")
    2. result = loaded.module()(real_input)

  KEY POINTS:
  - No model source code needed on deployment side
  - Dynamic shapes are preserved in the archive
  - Model weights are included in the archive
  - The archive is self-contained and portable
  - Compatible across PyTorch versions (same major version)
""")


def main():
    print("\nSaving and Loading Exported Models")
    print("=" * 60)
    print()

    demo_basic_save_load()
    demo_save_with_dynamic_shapes()
    demo_save_sequence_model()
    demo_inspect_archive()
    demo_multiple_models()
    demo_workflow_summary()

    print("All save/load demos completed!\n")


if __name__ == "__main__":
    main()
