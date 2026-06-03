"""
torch.export Basics — Exporting Models for Deployment
=======================================================
Covers: basic export, dynamic shapes, saving/loading, inspection.
"""

import torch
import torch.nn as nn

print("=" * 60)
print("1. BASIC EXPORT")
print("=" * 60)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = SimpleModel()
model.eval()

example_input = (torch.randn(1, 100),)

# Export the model
exported = torch.export.export(model, example_input)
print(f"Exported program type: {type(exported).__name__}")

# Run the exported model
output = exported.module()(torch.randn(1, 100))
print(f"Output shape: {output.shape}")

# Verify correctness
with torch.no_grad():
    eager_out = model(example_input[0])
    export_out = exported.module()(example_input[0])
    print(f"Outputs match: {torch.allclose(eager_out, export_out)}")

print("\n" + "=" * 60)
print("2. DYNAMIC SHAPES")
print("=" * 60)

from torch.export import Dim

# Define dynamic dimensions
batch = Dim("batch", min=1, max=256)

exported_dynamic = torch.export.export(
    model,
    (torch.randn(1, 100),),
    dynamic_shapes={"x": {0: batch}}  # Batch dim is dynamic
)

# Works with any batch size
for bs in [1, 4, 16, 64]:
    out = exported_dynamic.module()(torch.randn(bs, 100))
    print(f"  Batch size {bs:3d} -> output: {out.shape}")

print("\n" + "=" * 60)
print("3. INSPECTING THE GRAPH")
print("=" * 60)

print("Graph nodes:")
for node in exported.graph_module.graph.nodes:
    print(f"  {node.op:20s} | {node.name}")

print("\n" + "=" * 60)
print("4. SAVE & LOAD")
print("=" * 60)

import tempfile, os

with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, "model.pt2")

    # Save
    torch.export.save(exported_dynamic, path)
    print(f"Saved exported model to {path}")
    print(f"File size: {os.path.getsize(path) / 1024:.1f} KB")

    # Load
    loaded = torch.export.load(path)
    out = loaded.module()(torch.randn(8, 100))
    print(f"Loaded model output: {out.shape}")

print("\n" + "=" * 60)
print("5. EXPORT WITH CONTROL FLOW")
print("=" * 60)

class ConditionalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x, flag: bool):
        if flag:
            return self.fc(x).relu()
        else:
            return self.fc(x).sigmoid()

cond_model = ConditionalModel()
cond_model.eval()

# Export traces the specific path taken
exported_true = torch.export.export(
    cond_model,
    (torch.randn(1, 10), True)
)
print(f"Exported with flag=True, output: {exported_true.module()(torch.randn(1, 10), True).shape}")

print("\nDone!")
