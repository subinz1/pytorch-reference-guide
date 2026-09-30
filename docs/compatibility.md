# Compatibility and Installation

The maintained baseline for this guide is **Python 3.10+** and **PyTorch
2.14+**. The examples are written against current PyTorch APIs; an older
environment can run only the material supported by its installed version.

`requirements.txt` is a convenience list for the guide's Python dependencies.
It is not a CUDA, ROCm, or system-toolchain installer. Install the PyTorch
build that matches the machine first, then install the remaining dependencies.

## Environment matrix

| Guide area | Supported starting point | Additional requirements |
|---|---|---|
| Foundations through training | Python 3.10+, PyTorch 2.14+, CPU | None |
| `torch.compile`, export, and profiling | Python 3.10+, PyTorch 2.14+, CPU | Results and available backends vary by operating system and accelerator |
| CUDA graphs, memory profiling, mixed precision | Matching CUDA PyTorch build | NVIDIA GPU and a PyTorch build with CUDA support |
| Distributed and multi-GPU examples | Matching accelerator PyTorch build | One process per device and an appropriate distributed backend |
| Triton and custom CUDA extensions | Matching CUDA PyTorch build | NVIDIA GPU, supported compiler toolchain, and module-specific dependencies |
| Quantization and deployment integrations | Python 3.10+, PyTorch 2.14+ | Optional packages named by the individual module |

## Installation paths

### CPU learning path

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use this path for the foundations, tensor, autograd, neural-network, optimizer,
data-loading, and most testing material.

### CUDA or ROCm learning path

1. Install a PyTorch build selected for the machine from the
   [official PyTorch installer](https://pytorch.org/get-started/locally/).
2. Install the remaining guide dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Confirm that the runtime can see the expected accelerator before running an
   accelerator-specific example:

   ```bash
   python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
   ```

The package selector is the source of truth for current CUDA and ROCm wheel
combinations. Do not mix a system CUDA installation with an arbitrary PyTorch
wheel solely to make an example start.

## Version-aware reading

- Start from the CPU material when setting up a new environment.
- Read each module's prerequisites before copying an accelerator-specific
  command.
- Treat performance numbers as machine-specific; use the examples to compare
  changes on the same environment.
- When an API is unavailable, upgrade to the maintained baseline rather than
  silently replacing the example with a different API.

See [runtime requirements](runtime_requirements.md) for the CPU, accelerator,
and multi-device execution contract used by the examples.
