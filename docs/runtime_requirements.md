# Runtime Requirements

Every runnable example belongs to one of three execution classes. The class
should be clear before an example allocates tensors or starts a process group.

| Class | Examples | Requirement | Behavior when unavailable |
|---|---|---|---|
| CPU | Tensor, autograd, module, optimizer, data-loading, and testing material | Python 3.10+ and PyTorch 2.14+ | Run normally |
| Accelerator optional | `torch.compile`, export, attention, AMP, quantization, profiling | A matching CUDA or ROCm build can unlock additional paths | Run the CPU path when it exists; state which accelerated path was skipped |
| Accelerator required | CUDA graphs, Triton kernels, GPU memory profiling, multi-GPU inference, CUDA extensions | A compatible accelerator build and the module-specific tools | Print a skip reason and exit successfully before doing GPU-only work |

## Hardware checks

Use an explicit check near the entry point of an accelerator-required example:

```python
import sys
import torch

if not torch.cuda.is_available():
    print("SKIP: this example requires an available CUDA or ROCm device")
    sys.exit(0)
```

For a multi-device example, verify the device count before initializing the
process group:

```python
if torch.cuda.device_count() < 2:
    print("SKIP: this example requires at least two accelerator devices")
    sys.exit(0)
```

This makes a missing GPU an intentional skip rather than a failing CPU smoke
test. It does not make a GPU-only result equivalent to CPU coverage.

## Module groups

| Area | Typical runtime |
|---|---|
| Foundations through neural-network basics | CPU |
| Training, `torch.compile`, attention, export, and quantization | CPU with optional acceleration |
| Distributed training and DDP patterns | CPU for concepts; accelerator hardware for performance and multi-GPU paths |
| CUDA graphs, Triton, memory profiling, and multi-GPU inference | Accelerator required |
| Custom C++ extensions | Compiler toolchain; CUDA toolkit only for CUDA extension paths |
| Operational guides for CRCR and targeted tests | CPU |

## Validation contract

The [CPU smoke manifest](../tools/cpu_smoke_manifest.json) lists examples that
must remain CPU-safe. Do not add an accelerator-required example to that
manifest. For a new hardware-specific example, document the required runtime
in its module README and follow the safe-skip pattern above.

See [compatibility and installation](compatibility.md) for selecting the
PyTorch build and [validation](validation.md) for the available checks.
