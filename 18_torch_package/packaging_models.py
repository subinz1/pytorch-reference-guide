"""
torch.package — Self-Contained Model Packaging
================================================
Bundle models + Python source code into hermetic archives.
All examples run on CPU.
"""

import torch
import torch.nn as nn
import tempfile
import os

print("=" * 65)
print("1. BASIC MODEL PACKAGING")
print("=" * 65)

# Define a model (normally in your own module)
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.act(self.bn(self.fc1(x)))
        return self.fc2(x)

model = SimpleClassifier()
model.eval()
print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

# Package the model
from torch.package import PackageExporter, PackageImporter

tmpdir = tempfile.mkdtemp()
pkg_path = os.path.join(tmpdir, "model.pt")

with PackageExporter(pkg_path) as exporter:
    # Intern: bundle this module's source code inside the package
    exporter.intern("**")

    # Extern: these must be installed on the target machine
    exporter.extern("torch.**")
    exporter.extern("numpy.**")

    # Save the model
    exporter.save_pickle("model", "model.pkl", model)

print(f"Package saved to: {pkg_path}")
print(f"Package size: {os.path.getsize(pkg_path) / 1024:.1f} KB")

print("\n" + "=" * 65)
print("2. LOADING A PACKAGE")
print("=" * 65)

# Load the package (could be on a completely different machine)
importer = PackageImporter(pkg_path)
loaded_model = importer.load_pickle("model", "model.pkl")

# Verify it works
x = torch.randn(4, 784)
with torch.no_grad():
    out_original = model(x)
    out_loaded = loaded_model(x)

print(f"Original output shape: {out_original.shape}")
print(f"Loaded output shape:   {out_loaded.shape}")
print(f"Outputs match: {torch.allclose(out_original, out_loaded)}")

print("\n" + "=" * 65)
print("3. SAVING MULTIPLE OBJECTS")
print("=" * 65)

pkg_path2 = os.path.join(tmpdir, "full_package.pt")

config = {
    "input_dim": 784,
    "hidden_dim": 256,
    "num_classes": 10,
    "version": "1.0.0",
    "training_epochs": 50,
    "best_accuracy": 0.95,
}

with PackageExporter(pkg_path2) as exporter:
    exporter.intern("**")
    exporter.extern("torch.**")
    exporter.extern("numpy.**")

    # Save model
    exporter.save_pickle("model", "model.pkl", model)

    # Save config
    exporter.save_pickle("config", "config.pkl", config)

    # Save raw text
    exporter.save_text("metadata", "info.txt",
                       "SimpleClassifier v1.0\nTrained on MNIST\n50 epochs")

print(f"Full package saved: {os.path.getsize(pkg_path2) / 1024:.1f} KB")

# Load everything
importer2 = PackageImporter(pkg_path2)
loaded_model = importer2.load_pickle("model", "model.pkl")
loaded_config = importer2.load_pickle("config", "config.pkl")
loaded_info = importer2.load_text("metadata", "info.txt")

print(f"Loaded config: {loaded_config}")
print(f"Loaded info:\n{loaded_info}")

print("\n" + "=" * 65)
print("4. INSPECTING PACKAGE CONTENTS")
print("=" * 65)

importer3 = PackageImporter(pkg_path2)
file_structure = importer3.file_structure()
print(f"Package file structure:\n{file_structure}")

# You can also use zipfile since .pt is a zip
import zipfile
with zipfile.ZipFile(pkg_path2, 'r') as zf:
    print(f"\nZip contents ({len(zf.namelist())} files):")
    for name in sorted(zf.namelist())[:15]:
        info = zf.getinfo(name)
        print(f"  {name:50s} {info.compress_size:>8d} bytes")
    if len(zf.namelist()) > 15:
        print(f"  ... and {len(zf.namelist()) - 15} more files")

print("\n" + "=" * 65)
print("5. MODULE ACTIONS: intern, extern, mock, deny")
print("=" * 65)

print("""
┌──────────────┬────────────────────────────────────────────────────┐
│ Action       │ What It Does                                       │
├──────────────┼────────────────────────────────────────────────────┤
│ intern("x")  │ Bundle x's source code INSIDE the package          │
│ extern("x")  │ x must be installed on the target machine          │
│ mock("x")    │ Replace x with a stub (MockedObject)               │
│ deny("x")    │ Error if x is encountered                          │
└──────────────┴────────────────────────────────────────────────────┘

Patterns use glob syntax:
  "my_module"       — exact match
  "my_module.*"     — my_module and direct submodules
  "my_module.**"    — my_module and ALL nested submodules
""")

print("=" * 65)
print("6. MOCK EXAMPLE — Stub Out Unused Dependencies")
print("=" * 65)

# Imagine your model code imports matplotlib for visualization
# but you don't need it at inference time

class ModelWithVizDep(nn.Module):
    """A model whose source code imports matplotlib (mocked out)."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

    def plot_weights(self):
        """This method uses matplotlib, but we don't need it for inference."""
        # import matplotlib.pyplot as plt  # Would fail without mock
        pass

pkg_mock_path = os.path.join(tmpdir, "mocked_model.pt")

with PackageExporter(pkg_mock_path) as exporter:
    exporter.intern("**")
    exporter.extern("torch.**")
    exporter.extern("numpy.**")
    exporter.mock("matplotlib.**")  # Stub out matplotlib

    model_viz = ModelWithVizDep()
    exporter.save_pickle("model", "model.pkl", model_viz)

print(f"Package with mocked matplotlib: {os.path.getsize(pkg_mock_path) / 1024:.1f} KB")

# Load works even without matplotlib installed!
imp_mock = PackageImporter(pkg_mock_path)
loaded = imp_mock.load_pickle("model", "model.pkl")
print(f"Loaded model output: {loaded(torch.randn(2, 10)).shape}")

print("\n" + "=" * 65)
print("7. PACKAGING A MORE COMPLEX MODEL")
print("=" * 65)

class Encoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim), nn.ReLU())
            for _ in range(3)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x  # Residual
        return x

class Decoder(nn.Module):
    def __init__(self, dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(dim, output_dim)

    def forward(self, x):
        return self.fc(x)

class AutoEncoder(nn.Module):
    def __init__(self, dim=128, output_dim=10):
        super().__init__()
        self.encoder = Encoder(dim)
        self.decoder = Decoder(dim, output_dim)

    def forward(self, x):
        return self.decoder(self.encoder(x))

ae_model = AutoEncoder()
ae_model.eval()
params = sum(p.numel() for p in ae_model.parameters())
print(f"AutoEncoder: {params:,} parameters")

pkg_ae_path = os.path.join(tmpdir, "autoencoder.pt")

with PackageExporter(pkg_ae_path) as exporter:
    exporter.intern("**")
    exporter.extern("torch.**")
    exporter.extern("numpy.**")
    exporter.save_pickle("model", "model.pkl", ae_model)

imp_ae = PackageImporter(pkg_ae_path)
loaded_ae = imp_ae.load_pickle("model", "model.pkl")

x_test = torch.randn(8, 128)
with torch.no_grad():
    out1 = ae_model(x_test)
    out2 = loaded_ae(x_test)

print(f"Original: {out1.shape}, Loaded: {out2.shape}")
print(f"Outputs match: {torch.allclose(out1, out2)}")

print("\n" + "=" * 65)
print("8. COMPARISON: torch.save vs torch.package vs torch.export")
print("=" * 65)

# torch.save — just weights
save_path = os.path.join(tmpdir, "model_save.pt")
torch.save(ae_model.state_dict(), save_path)

# torch.package — weights + source code
pkg_size = os.path.getsize(pkg_ae_path)

# torch.export — full graph
try:
    exported = torch.export.export(ae_model, (torch.randn(1, 128),))
    export_path = os.path.join(tmpdir, "model_export.pt2")
    torch.export.save(exported, export_path)
    export_size = os.path.getsize(export_path)
except Exception as e:
    export_size = 0
    print(f"  (torch.export note: {e})")

save_size = os.path.getsize(save_path)

print(f"""
File sizes:
  torch.save:    {save_size / 1024:>8.1f} KB  (weights only)
  torch.package: {pkg_size / 1024:>8.1f} KB  (weights + source code)
  torch.export:  {export_size / 1024:>8.1f} KB  (full computation graph)

When to use:
  torch.save     → Training checkpoints, quick save/load
  torch.package  → Sharing models with all Python dependencies
  torch.export   → Production deployment (C++, mobile, ONNX)
""")

# Cleanup
import shutil
shutil.rmtree(tmpdir)

print("Done!")
