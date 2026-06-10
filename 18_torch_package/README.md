<div align="center">

[← Previous Module](../17_compile_decorators/) | [🏠 Home](../README.md) | [Next Module →](../19_torch_function_dispatch/)

</div>

---

> **Module 18** of the PyTorch Complete Learning Guide
> **Prerequisites:** [Module 04 — Neural Networks](../04_neural_networks/)
> **Time to complete:** ~1 hour

### 📁 Files in This Module

| File | Description |
|------|-------------|
| `README.md` | This guide |
| `packaging_models.py` | torch.package — bundling models and Python source code into hermetic archives |

---

# Module 18: torch.package — Self-Contained Model Packaging

*Day 4 of the incremental learning series*

---

## The Problem: "It Works on My Machine"

You train a model. You save `model.pt`. You send it to a colleague. It fails because:
- They don't have the same version of your custom modules
- An import path changed between your environments
- A dependency you forgot about isn't installed on their machine

**torch.package** solves this by bundling the model **and all its Python dependencies** into a single `.pt` archive.

---

## Table of Contents

1. [What is torch.package?](#1-what-is-torchpackage)
2. [PackageExporter — Creating Packages](#2-packageexporter)
3. [PackageImporter — Loading Packages](#3-packageimporter)
4. [Module Actions: intern, extern, mock, deny](#4-module-actions)
5. [Packaging Models with Weights](#5-packaging-models-with-weights)
6. [Inspecting Package Contents](#6-inspecting-package-contents)
7. [Re-Packaging and Dependency Analysis](#7-re-packaging)
8. [torch.package vs torch.save vs torch.export](#8-comparison)
9. [Practical Workflow](#9-practical-workflow)
10. [What's New Upstream (June 8-9, 2026)](#10-upstream-updates)

---

## 1. What is torch.package?

`torch.package` creates a **hermetic zip archive** containing:
- Your model's **Python source code** (the actual `.py` files)
- The model's **pickled state** (weights, buffers, config)
- A **manifest** of external dependencies

When someone loads the package, it uses its **own import system** — code is loaded from inside the archive, not from the local Python installation. This means:
- The exact code you packaged runs, regardless of what's installed locally
- Only explicitly listed external dependencies are loaded from the system
- No "accidental" dependencies can sneak in

```
┌──────────────────────────────────┐
│        my_model.pt (zip)         │
├──────────────────────────────────┤
│  .data/                          │
│    model.pkl        (pickled)    │
│    weights.pt       (tensors)    │
│  my_module/                      │
│    model.py         (source)     │
│    layers.py        (source)     │
│    config.py        (source)     │
│  extern_modules     (manifest)   │
│    torch                         │
│    numpy                         │
└──────────────────────────────────┘
```

---

## 2. PackageExporter — Creating Packages

```python
from torch.package import PackageExporter

# Create a package
with PackageExporter("my_model.pt") as exporter:
    # INTERN: Include this module's source inside the package
    exporter.intern("my_module.**")

    # EXTERN: This module is expected to exist on the target machine
    exporter.extern("torch.**")
    exporter.extern("numpy.**")

    # MOCK: Replace this module with a stub (for unused optional deps)
    exporter.mock("matplotlib.**")

    # Save the model object
    exporter.save_pickle("model", "model.pkl", my_model)
```

### The Four Module Actions

| Action | What It Does | When to Use |
|--------|-------------|-------------|
| `intern(pattern)` | Bundle the module's **source code** into the package | Your own code, custom modules |
| `extern(pattern)` | Expect the module to be installed on the target machine | PyTorch, NumPy, standard libs |
| `mock(pattern)` | Replace with a stub that returns `MockedObject` | Unused optional dependencies |
| `deny(pattern)` | Error if this module is encountered | Known-bad dependencies |

**Patterns** use glob syntax: `"my_module.**"` matches `my_module` and all submodules.

---

## 3. PackageImporter — Loading Packages

```python
from torch.package import PackageImporter

# Load a package
importer = PackageImporter("my_model.pt")

# Load the pickled model
model = importer.load_pickle("model", "model.pkl")

# The model runs using code FROM the package, not your local installation
output = model(torch.randn(1, 3, 224, 224))

# Import a module from the package (hermetic import)
my_config = importer.import_module("my_module.config")
```

**Key property:** The loaded code runs in an isolated namespace. If `my_module/model.py` inside the package says `import my_module.layers`, it loads `layers.py` **from the package**, not from your filesystem.

---

## 4. Module Actions in Detail

### intern — Bundle Source Code

```python
# Include specific modules
exporter.intern("my_project.models.**")
exporter.intern("my_project.utils.**")

# Include everything in your project
exporter.intern("my_project.**")
```

**What happens:** The `.py` source files are copied into the zip archive. When loaded, Python reads them from the archive.

### extern — External Dependencies

```python
# Standard externals
exporter.extern("torch.**")
exporter.extern("torchvision.**")
exporter.extern("numpy.**")

# Stdlib is automatically handled, but you can be explicit:
exporter.extern("os")
exporter.extern("json")
```

**What happens:** A list of external modules is saved in `extern_modules`. When loading, these are imported from the system Python.

### mock — Stub Out Dependencies

```python
# Mock out modules that aren't needed at inference time
exporter.mock("wandb.**")          # Logging library
exporter.mock("matplotlib.**")     # Plotting
exporter.mock("tensorboard.**")    # TensorBoard
```

**What happens:** A lightweight `_mock` module replaces the real one. Any attribute access on a mocked module returns `MockedObject`.

### deny — Prevent Inclusion

```python
# Error if these are encountered
exporter.deny("secret_module.**")
exporter.deny("credentials.**")
```

---

## 5. Packaging Models with Weights

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# Train your model...
model = MyModel(784, 256, 10)

# Package it with all dependencies
with PackageExporter("my_model_package.pt") as exporter:
    exporter.intern("__main__")    # Include the current module
    exporter.extern("torch.**")
    exporter.extern("numpy.**")

    # Save model
    exporter.save_pickle("model", "model.pkl", model)

    # You can also save arbitrary data
    exporter.save_pickle("config", "config.pkl", {
        "input_dim": 784,
        "hidden_dim": 256,
        "output_dim": 10,
        "version": "1.0",
    })

    # Save raw text/binary files
    exporter.save_text("info", "README.txt", "My model v1.0")

# Load on another machine
importer = PackageImporter("my_model_package.pt")
model = importer.load_pickle("model", "model.pkl")
config = importer.load_pickle("config", "config.pkl")
readme = importer.load_text("info", "README.txt")
```

---

## 6. Inspecting Package Contents

```python
importer = PackageImporter("my_model_package.pt")

# View the file structure
file_structure = importer.file_structure()
print(file_structure)
# Prints a tree of all files in the archive

# List all extern modules
print(file_structure.has_file("extern_modules"))
```

You can also inspect with standard zip tools since `.pt` files are zip archives:

```bash
unzip -l my_model_package.pt
python -m zipfile -l my_model_package.pt
```

---

## 7. Re-Packaging

You can load a package and re-export it (e.g., to add/remove dependencies):

```python
importer = PackageImporter("model_v1.pt")

with PackageExporter("model_v2.pt", importer=(importer,)) as exporter:
    exporter.intern("my_module.**")
    exporter.extern("torch.**")

    # Load the old model and save it in the new package
    model = importer.load_pickle("model", "model.pkl")
    exporter.save_pickle("model", "model.pkl", model)
```

---

## 8. torch.package vs torch.save vs torch.export

| Feature | `torch.save` | `torch.package` | `torch.export` |
|---------|-------------|-----------------|----------------|
| Saves weights | Yes | Yes | Yes |
| Saves code | No | **Yes (source)** | Yes (graph) |
| Hermetic loading | No | **Yes** | Yes |
| Python control flow | N/A | **Full** | Limited |
| Works cross-version | Fragile | **Robust** | Robust |
| Deployment target | Python | **Python** | C++, mobile, ONNX |
| File format | pickle | **zip (with source)** | PT2 archive |
| Speed | Fast | Medium | Compile required |

**When to use each:**
- **`torch.save`**: Quick checkpoints during training
- **`torch.package`**: Ship Python models with all dependencies, research sharing
- **`torch.export`**: Production deployment, C++ inference, mobile

---

## 9. Practical Workflow

### Research → Deployment Pipeline

```python
# 1. Researcher trains model (research/train.py)
model = train_my_model()

# 2. Researcher packages model with all custom code
with PackageExporter("model_v1.pt") as pe:
    pe.intern("my_research_code.**")
    pe.extern("torch.**")
    pe.extern("torchvision.**")
    pe.mock("wandb.**")   # Don't need logging in production
    pe.mock("matplotlib.**")
    pe.save_pickle("model", "model.pkl", model)

# 3. Engineer loads on a different machine (no my_research_code installed!)
importer = PackageImporter("model_v1.pt")
model = importer.load_pickle("model", "model.pkl")
output = model(input_data)  # Just works!
```

### Tips

1. **Always extern `torch`** — it must match the installed version
2. **Mock unused dependencies** — logging, visualization, experiment tracking
3. **Test the package** — load it in a clean environment to verify
4. **Version your packages** — include version info as saved text/pickle
5. **Inspect before shipping** — use `file_structure()` to verify contents

---

## 10. Upstream Updates (June 8-9, 2026)

Recent PyTorch main commits (since last update):

- **Inductor CUTLASS GELU fusion** — Folding decomposed GELU back into native CUTLASS functor for better performance (`#185824`)
- **Inductor TP pattern fusion** — Fusing slice-cat tensor parallel collective patterns (`#184911`)
- **Dynamo Python 3.15 support** — Build dynamo with Python 3.15, including updated `IMPORT_NAME` generation (`#186402`)
- **Dynamo operator support** — Added `divmod`, `remainder`, `true_divide`, `floor_divide` operators (`#185652-#185655`)
- **Deterministic topk** — `torch.topk` now respects `torch.use_deterministic_algorithms()` (`#186653`)
- **XPU oneDNN LSTM** — Intel GPU LSTM inference via oneDNN primitives (`#185531`)
- **Stable ABI generator** — New `torch/csrc/stable/generator.h` for stable C API

---

## Further Reading

- Source: `torch/package/package_exporter.py`, `torch/package/package_importer.py`
- PyTorch docs: [torch.package](https://pytorch.org/docs/stable/package.html)
- Tutorial: [Loading a TorchScript Model in C++](https://pytorch.org/tutorials/advanced/cpp_export.html) (comparison)

---

<div align="center">

[← Previous Module](../17_compile_decorators/) | [🏠 Home](../README.md) | [Next Module →](../19_torch_function_dispatch/)

**No dedicated notebook** — see [Practical Workflow](#9-practical-workflow) above

</div>
