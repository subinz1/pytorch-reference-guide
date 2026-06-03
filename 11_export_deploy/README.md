# Module 11: Export and Deployment

## Table of Contents
1. [Why Export?](#why-export)
2. [The PyTorch Deployment Landscape](#the-pytorch-deployment-landscape)
3. [torch.export](#torchexport)
4. [Static vs Dynamic Shapes](#static-vs-dynamic-shapes)
5. [draft_export](#draft_export)
6. [Saving and Loading](#saving-and-loading)
7. [Graph Inspection](#graph-inspection)
8. [AOTInductor](#aotinductor)
9. [NativeRT](#nativert)
10. [ONNX Export](#onnx-export)
11. [TorchServe](#torchserve)
12. [ExecuTorch](#executorch)
13. [Quantization for Deployment](#quantization-for-deployment)
14. [Practical Workflow](#practical-workflow)

---

## Why Export?

During research, you run PyTorch in **eager mode**: Python executes operations
one at a time, giving you full flexibility (print statements, breakpoints,
dynamic control flow). But production deployment has different requirements:

- **No Python dependency**: Servers, mobile devices, and embedded systems may
  not have (or want) a Python runtime.
- **Performance**: Compiler optimizations (operator fusion, memory planning,
  kernel selection) require seeing the full computation graph ahead of time.
- **Predictability**: Production systems need deterministic latency and memory
  usage. Eager mode's dynamism makes this hard to guarantee.
- **Portability**: The same model may run on server GPUs, edge TPUs, mobile
  phones, or web browsers.

Export bridges the gap: it captures your Python model as a self-contained
computation graph that can be optimized and deployed without Python.

```
Research (Python, eager)  →  Export (capture graph)  →  Deploy (optimized, no Python)
```

---

## The PyTorch Deployment Landscape

PyTorch provides multiple paths from research to production:

```
                          ┌──────────────┐
                          │  Your Model  │
                          │  (nn.Module) │
                          └──────┬───────┘
                                 │
                          ┌──────┴───────┐
                          │ torch.export │
                          │ (capture     │
                          │  full graph) │
                          └──────┬───────┘
                                 │
              ┌──────────┬───────┼────────┬──────────┐
              ▼          ▼       ▼        ▼          ▼
        ┌──────────┐ ┌───────┐ ┌────┐ ┌───────┐ ┌─────────┐
        │AOTInductor│ │NativeRT│ │ONNX│ │Torch- │ │ExecuTorch│
        │(.so lib) │ │(C++   │ │    │ │Serve  │ │(mobile/ │
        │          │ │engine)│ │    │ │       │ │edge)    │
        └──────────┘ └───────┘ └────┘ └───────┘ └─────────┘
           Server      Server   Any    Server    Mobile/
           (C++/Py)    (C++)   runtime  (Py)     Embedded
```

| Path | Output | Target | Key Advantage |
|------|--------|--------|---------------|
| **AOTInductor** | .so shared library | Server (C++ or Python) | Maximum performance, no Python needed |
| **NativeRT** | Serialized model | Server (C++) | C++ inference engine, easy deployment |
| **ONNX** | .onnx file | Any ONNX Runtime | Cross-framework interop |
| **TorchServe** | Model archive | Server (Python) | Full serving stack (batching, scaling) |
| **ExecuTorch** | .pte file | Mobile/edge devices | Small footprint, on-device inference |

---

## torch.export

`torch.export` is the primary tool for capturing a PyTorch model as a complete,
self-contained graph. It traces through your model and produces an
`ExportedProgram` containing the full computation graph with no Python
dependency.

### Basic Export

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))

model = MyModel()
example_input = (torch.randn(3, 10),)

# Export the model
exported = torch.export.export(model, example_input)

# The exported program can be called like the original
result = exported.module()(torch.randn(3, 10))
```

### What ExportedProgram Contains

An `ExportedProgram` captures:

1. **Graph Module**: An `fx.GraphModule` containing the computation graph as
   a series of operations (ATen operators).
2. **Graph Signature**: Maps graph inputs/outputs to parameters, buffers,
   and user inputs.
3. **State Dict**: The model's parameters and buffers.
4. **Range Constraints**: Valid ranges for dynamic dimensions.
5. **Module Call graph**: Preserves the module hierarchy for debugging.

```python
exported = torch.export.export(model, example_input)

# Access the graph
print(exported.graph_module.graph)

# Access parameters
print(exported.state_dict.keys())

# The graph shows ATen-level operations
for node in exported.graph_module.graph.nodes:
    print(f"  {node.op}: {node.target}")
```

### What Gets Captured

`torch.export` traces through your model's `forward` method and captures:

- All tensor operations as ATen ops
- Control flow (with restrictions — see below)
- Shape computations

It does NOT capture:
- Print statements or side effects
- Operations on non-tensor values that aren't related to shapes
- Dynamic Python control flow based on tensor values (use `torch.cond` instead)

### Handling Control Flow

Standard if/else based on tensor values won't work:

```python
# This will fail with torch.export:
def forward(self, x):
    if x.sum() > 0:  # Can't branch on tensor value!
        return x * 2
    return x * 3

# Use torch.cond instead:
def forward(self, x):
    return torch.cond(
        x.sum() > 0,
        lambda x: x * 2,
        lambda x: x * 3,
        (x,),
    )
```

Control flow based on tensor **shapes** (not values) is fine because shapes
are known at export time (or constrained for dynamic shapes):

```python
# This is fine:
def forward(self, x):
    if x.shape[0] > 5:  # Shape is known at export time
        return x[:5]
    return x
```

---

## Static vs Dynamic Shapes

By default, `torch.export` traces with **static shapes**: the exported model
only accepts inputs with the exact same shapes as the example inputs. For
production, you usually want **dynamic shapes** so the model handles variable
batch sizes, sequence lengths, etc.

### The Dim API

Use `torch.export.Dim` to declare which dimensions are dynamic:

```python
from torch.export import Dim, export

# Declare a dynamic dimension named "batch"
batch = Dim("batch", min=1, max=128)

# Export with dynamic first dimension
exported = export(
    model,
    (torch.randn(4, 10),),  # Example with batch=4
    dynamic_shapes={"x": {0: batch}},  # dim 0 is dynamic
)

# Now works with any batch size from 1 to 128:
exported.module()(torch.randn(1, 10))    # batch=1
exported.module()(torch.randn(64, 10))   # batch=64
exported.module()(torch.randn(128, 10))  # batch=128
```

### Multiple Dynamic Dimensions

```python
batch = Dim("batch", min=1, max=256)
seq_len = Dim("seq_len", min=1, max=2048)

exported = export(
    model,
    (torch.randn(4, 128, 512),),
    dynamic_shapes={"x": {0: batch, 1: seq_len}},
    # dim 0 = batch (dynamic), dim 1 = seq_len (dynamic), dim 2 = 512 (static)
)
```

### Constraints Between Dimensions

When multiple inputs share a dimension (e.g., same batch size), use the same
`Dim` object:

```python
batch = Dim("batch", min=1, max=128)

def forward(self, x, y):
    return x + y  # x and y must have same batch size

exported = export(
    model,
    (torch.randn(4, 10), torch.randn(4, 20)),
    dynamic_shapes={
        "x": {0: batch},
        "y": {0: batch},  # Same Dim → enforces same batch size
    },
)
```

### Automatic Dynamic Shapes

For convenience, you can let PyTorch infer dynamic shapes:

```python
from torch.export import export

exported = export(
    model,
    (torch.randn(4, 10),),
    dynamic_shapes={"x": {0: Dim.AUTO}},
)
```

---

## draft_export

When `torch.export` fails (due to unsupported Python constructs, dynamic
control flow, etc.), `draft_export` helps you debug by producing a best-
effort export with detailed error information.

```python
from torch.export import draft_export

# If regular export fails:
try:
    exported = torch.export.export(model, example_input)
except Exception as e:
    print(f"Export failed: {e}")

# Use draft_export to get a partial graph + diagnostics
ep, report = draft_export(model, example_input)

# The report shows what went wrong and how to fix it
print(report)
```

`draft_export` returns:
1. An `ExportedProgram` (possibly with graph breaks or approximations)
2. A report detailing what couldn't be captured and suggesting fixes

Common fixes suggested by draft_export:
- Replace `if tensor_val > 0` with `torch.cond`
- Replace `for i in range(tensor.shape[0])` with bounded loop
- Mark data-dependent shapes with `torch.export.Dim`

---

## Saving and Loading

### PT2 Archive Format

The recommended format for saving exported programs is the PT2 Archive:

```python
import torch

# Export
model = MyModel()
exported = torch.export.export(model, (torch.randn(1, 10),))

# Save as PT2 archive
torch.export.save(exported, "model.pt2")

# Load (no need for original model code!)
loaded = torch.export.load("model.pt2")
result = loaded.module()(torch.randn(1, 10))
```

The PT2 archive contains:
- The computation graph (serialized FX graph)
- Model weights (state dict)
- Metadata (dynamic shape constraints, etc.)

This is a self-contained format. You do not need the original Python model
definition to load and run the model.

### Saving for Different Backends

```python
# Save for later AOTInductor compilation
torch.export.save(exported, "model.pt2")

# Save for ONNX (different path)
torch.onnx.export(model, example_input, "model.onnx", dynamo=True)
```

### Versioning and Compatibility

PT2 archives are versioned. PyTorch maintains backward compatibility: a model
saved with an older PyTorch version can be loaded with a newer version (within
the same major version).

---

## Graph Inspection

After exporting, you can inspect the computation graph to understand what
was captured and verify correctness.

### Viewing the Graph

```python
exported = torch.export.export(model, example_input)

# Print the graph (human-readable)
print(exported.graph_module.graph)

# Print generated code (more readable)
print(exported.graph_module.code)
```

### Walking the Graph Nodes

Each node in the graph represents an operation:

```python
for node in exported.graph_module.graph.nodes:
    print(f"Op: {node.op:15s} | Target: {node.target} | Args: {node.args}")
```

Node types:
- `placeholder`: Input to the graph (parameters, buffers, user inputs)
- `call_function`: A function call (ATen operator)
- `output`: The graph's return value
- `get_attr`: Access a stored attribute

### Listing All Operations

```python
# Get the set of all ATen ops used in the graph
ops = set()
for node in exported.graph_module.graph.nodes:
    if node.op == "call_function":
        ops.add(str(node.target))

print(f"Operations used ({len(ops)}):")
for op in sorted(ops):
    print(f"  {op}")
```

### Understanding the Graph Signature

```python
sig = exported.graph_signature

# What are the inputs?
print("User inputs:", sig.input_specs)

# What are the outputs?
print("Outputs:", sig.output_specs)

# Parameters and buffers
print("Parameters:", [s for s in sig.input_specs if s.kind.name == "PARAMETER"])
```

---

## AOTInductor

AOTInductor (Ahead-Of-Time Inductor) compiles an exported model into a native
shared library (`.so` on Linux, `.dylib` on macOS). The result runs without
Python, with maximum performance.

### Compilation Flow

```
ExportedProgram → AOTInductor → .so shared library → C++ or Python inference
```

### Python API

```python
import torch

model = MyModel().eval()
example_input = (torch.randn(1, 3, 224, 224),)

# Export
exported = torch.export.export(model, example_input)

# Compile to .so
compiled_path = torch._inductor.aot_compile(
    exported.module(),
    example_input,
    options={"aot_inductor.output_path": "model.so"},
)

# Load and run in Python (for testing)
compiled_model = torch._inductor.aot_load(compiled_path)
result = compiled_model(torch.randn(1, 3, 224, 224))
```

### Package API (Recommended)

The package API bundles the model, weights, and metadata:

```python
# Compile and package
torch._inductor.aoti_compile_and_package(
    exported,
    package_path="model_package.pt2",
)

# Load the package
loaded = torch._inductor.aoti_load_package("model_package.pt2")
result = loaded(torch.randn(1, 3, 224, 224))
```

### C++ Deployment

The primary use case for AOTInductor is deploying without Python:

```cpp
#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>

int main() {
    // Load the compiled model
    auto runner = torch::inductor::AOTIModelContainerRunner("model.so");

    // Create input tensor
    auto input = torch::randn({1, 3, 224, 224});
    std::vector<torch::Tensor> inputs = {input};

    // Run inference
    auto outputs = runner.run(inputs);
    auto result = outputs[0];

    return 0;
}
```

### When to Use AOTInductor

- Maximum inference performance (compiled, optimized kernels)
- Deployment without Python
- Server-side GPU inference
- When you can compile ahead of time (shapes known in advance)

---

## NativeRT

NativeRT is a C++ inference engine for running PyTorch models. While
AOTInductor compiles to native code, NativeRT interprets the exported graph
using PyTorch's C++ runtime.

### NativeRT vs AOTInductor

| Feature | AOTInductor | NativeRT |
|---------|-------------|----------|
| Compilation | Ahead of time (slow) | No compilation |
| Inference speed | Fastest (native code) | Fast (C++ runtime) |
| Startup time | Fast (pre-compiled) | Fast (load + interpret) |
| Flexibility | Fixed graph | More flexible |
| Python needed | No | No |

### When to Use NativeRT

- When AOTInductor compilation is too slow or complex
- When you need C++ inference without ahead-of-time compilation
- Quick prototyping of C++ deployment

### Usage Pattern

```python
# Export and save
exported = torch.export.export(model, example_input)
torch.export.save(exported, "model.pt2")

# In C++ with NativeRT:
# auto runner = torch::nativert::ModelRunner("model.pt2");
# auto output = runner.run(inputs);
```

---

## ONNX Export

ONNX (Open Neural Network Exchange) is an open format for representing ML
models. PyTorch can export to ONNX for running on ONNX Runtime, which supports
multiple hardware backends.

### Modern ONNX Export (Dynamo-based)

```python
import torch

model = MyModel().eval()
example_input = (torch.randn(1, 3, 224, 224),)

# Export to ONNX using the new dynamo-based exporter
onnx_program = torch.onnx.export(
    model,
    example_input,
    dynamo=True,
)

# Save to file
onnx_program.save("model.onnx")
```

### Running with ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model.onnx")

# Get input name
input_name = session.get_inputs()[0].name

# Run inference
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
result = session.run(None, {input_name: input_data})
```

### Dynamic Shapes with ONNX

```python
from torch.export import Dim

batch = Dim("batch", min=1, max=128)

onnx_program = torch.onnx.export(
    model,
    (torch.randn(4, 3, 224, 224),),
    dynamo=True,
    dynamic_shapes={"x": {0: batch}},
)
onnx_program.save("model_dynamic.onnx")
```

### When to Use ONNX

- Cross-framework deployment (model trained in PyTorch, deployed with
  TensorFlow Serving, etc.)
- Hardware with ONNX Runtime support but not PyTorch C++ runtime
- When inference hardware vendor provides an ONNX Runtime backend

---

## TorchServe

TorchServe is PyTorch's model serving framework. It handles the operational
concerns of serving models in production: batching, scaling, monitoring, and
A/B testing.

### Overview

```
Client → TorchServe ┬→ Worker 1 → Model
   (HTTP/gRPC)       ├→ Worker 2 → Model
                     └→ Worker N → Model
```

### Key Features

- **Dynamic batching**: Aggregates individual requests into batches for
  efficient GPU utilization
- **Multi-model serving**: Serve multiple models from one instance
- **Model versioning**: A/B testing, canary deployments
- **Monitoring**: Prometheus metrics, logging
- **Auto-scaling**: Scale workers based on load

### Basic Workflow

```bash
# 1. Archive the model
torch-model-archiver \
    --model-name my_model \
    --version 1.0 \
    --serialized-file model.pt \
    --handler image_classifier \
    --export-path model_store

# 2. Start TorchServe
torchserve --start \
    --model-store model_store \
    --models my_model=my_model.mar

# 3. Send requests
curl http://localhost:8080/predictions/my_model -T input.jpg
```

---

## ExecuTorch

ExecuTorch is PyTorch's solution for on-device inference on mobile phones,
wearables, and embedded systems. It produces small, efficient models that
run without a full PyTorch runtime.

### Key Features

- **Small footprint**: Runtime is ~100s of KB (vs PyTorch's ~100s of MB)
- **Delegate system**: Hardware-specific optimizations (Core ML, XNNPACK,
  Qualcomm QNN, etc.)
- **No Python**: Runs on C++ runtime

### Workflow

```python
import torch
from executorch.exir import to_edge_transform_and_lower

model = MyModel().eval()
example_input = (torch.randn(1, 3, 224, 224),)

# Export
exported = torch.export.export(model, example_input)

# Lower to edge
edge_program = to_edge_transform_and_lower(exported)

# Save
edge_program.save("model.pte")
```

### Target Platforms

| Platform | Delegate |
|----------|----------|
| iOS | Core ML, Metal |
| Android | XNNPACK, Qualcomm QNN |
| Microcontrollers | Custom delegates |
| Web | WebAssembly |

---

## Quantization for Deployment

Quantization reduces model size and increases inference speed by using lower
precision arithmetic (e.g., INT8 instead of FP32).

### PT2E Quantization Flow

The modern quantization flow is built on torch.export:

```python
import torch
from torch.ao.quantization.quantize_pt2e import (
    prepare_pt2e,
    convert_pt2e,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)

model = MyModel().eval()
example_input = (torch.randn(1, 3, 224, 224),)

# Step 1: Export
exported = torch.export.export(model, example_input)

# Step 2: Prepare for quantization (inserts observers)
quantizer = XNNPACKQuantizer().set_global(
    get_symmetric_quantization_config()
)
prepared = prepare_pt2e(exported, quantizer)

# Step 3: Calibrate with representative data
with torch.no_grad():
    for data in calibration_dataloader:
        prepared(data)

# Step 4: Convert to quantized model
quantized = convert_pt2e(prepared)

# Step 5: Deploy (e.g., with AOTInductor or ExecuTorch)
```

### Quantization Types

| Type | Precision | Speed | Accuracy | Use Case |
|------|-----------|-------|----------|----------|
| FP32 | 32-bit | Baseline | Best | Training |
| FP16 | 16-bit | ~2× | Negligible loss | GPU inference |
| BF16 | 16-bit | ~2× | Negligible loss | GPU inference |
| INT8 | 8-bit | ~2-4× | Small loss | Server inference |
| INT4 | 4-bit | ~4-8× | Moderate loss | Edge/mobile, LLMs |

### Dynamic vs Static Quantization

- **Static quantization**: Calibrate ranges with representative data. Best
  accuracy. Requires calibration dataset.
- **Dynamic quantization**: Compute ranges at runtime. Simpler setup. Slightly
  worse performance on some workloads.

---

## Practical Workflow

### The End-to-End Journey

```
1. RESEARCH & TRAINING
   ├── Train in eager mode (nn.Module, autograd)
   ├── Validate accuracy
   └── Save checkpoint

2. EXPORT
   ├── torch.export.export(model, example_inputs)
   ├── Add dynamic shapes for variable inputs
   ├── Fix export issues (torch.cond for control flow, etc.)
   ├── Use draft_export to debug failures
   └── Save: torch.export.save(exported, "model.pt2")

3. OPTIMIZE
   ├── Quantize (PT2E flow) for smaller/faster model
   ├── Profile to identify bottlenecks
   └── Choose deployment target

4. DEPLOY
   ├── Server GPU → AOTInductor (.so) or NativeRT
   ├── Server CPU → ONNX Runtime or AOTInductor
   ├── Model serving → TorchServe
   ├── Mobile/Edge → ExecuTorch (.pte)
   └── Cross-framework → ONNX
```

### Common Patterns

**Pattern 1: Quick Server Deployment**
```python
model = load_trained_model()
exported = torch.export.export(model, example_input)
torch.export.save(exported, "model.pt2")
# Load in production with torch.export.load("model.pt2")
```

**Pattern 2: Maximum Performance Server**
```python
model = load_trained_model()
exported = torch.export.export(model, example_input)
torch._inductor.aoti_compile_and_package(exported, "model_pkg.pt2")
# Deploy .so or package without Python
```

**Pattern 3: Mobile Deployment**
```python
model = load_trained_model()
exported = torch.export.export(model, example_input)
# Quantize → Lower to edge → Save .pte
```

**Pattern 4: Cross-Platform**
```python
model = load_trained_model()
onnx_program = torch.onnx.export(model, example_input, dynamo=True)
onnx_program.save("model.onnx")
# Deploy with ONNX Runtime on any platform
```

### Debugging Export Failures

1. **Start with draft_export**: Get a partial graph and diagnostic report.
2. **Check control flow**: Replace data-dependent branches with `torch.cond`.
3. **Check dynamic shapes**: Use `Dim` for variable dimensions.
4. **Simplify**: Export a smaller part of the model first to isolate issues.
5. **Check operators**: Some custom ops may need registration for export.

### Performance Comparison (Rough Guidelines)

| Deployment Path | Relative Latency | Setup Effort |
|-----------------|-------------------|-------------|
| Eager (Python) | 1.0× (baseline) | None |
| torch.compile | 0.5-0.8× | One line |
| AOTInductor | 0.3-0.6× | Moderate |
| Quantized (INT8) | 0.2-0.4× | Significant |
| ExecuTorch (mobile) | Varies by hardware | Significant |

---

## Files in This Module

| File | Description | Run Command |
|------|-------------|-------------|
| `export_basics.py` | Basic export, run exported model | `python export_basics.py` |
| `dynamic_shapes.py` | Dim API, multiple dynamic dims | `python dynamic_shapes.py` |
| `export_inspection.py` | Inspect the graph, list ops | `python export_inspection.py` |
| `save_and_load.py` | Save/load PT2 archives | `python save_and_load.py` |
