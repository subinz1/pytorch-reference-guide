<p align="center">
  <img src="https://pytorch.org/assets/images/pytorch-logo.png" width="120" alt="PyTorch Logo">
</p>

<h1 align="center">PyTorch: The Complete Learning Guide</h1>
<h3 align="center">From Absolute Beginner to Advanced Practitioner</h3>

<p align="center">
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.13%2B-EE4C2C?logo=pytorch" alt="PyTorch"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python"></a>
  <a href="#course-structure"><img src="https://img.shields.io/badge/Modules-14-blue" alt="Modules"></a>
  <a href="#course-structure"><img src="https://img.shields.io/badge/Code_Examples-55%2B-green" alt="Examples"></a>
  <a href="#course-structure"><img src="https://img.shields.io/badge/Lines-30%2C000%2B-orange" alt="Lines"></a>
</p>

<p align="center">
  A structured, self-contained PyTorch course organized into <strong>14 modules</strong>.<br>
  Each module contains detailed explanations, theory, formulas, and fully runnable Python examples.
</p>

---

## About This Guide

This repository takes someone with **basic Python knowledge** and makes them a **proficient PyTorch developer**. Every module contains:

- **`README.md`** — In-depth explanations with theory, math formulas, diagrams, and inline code
- **Python scripts** — Self-contained, runnable examples (`python filename.py`) with detailed output

**Updated for PyTorch 2.13+ (June 2026)** — includes modern features like `torch.compile`, FlexAttention, FSDP2, and more.

---

## Course Structure

### Part I: Foundations

> *Start here. Master the building blocks everything else depends on.*

| # | Module | Description | Files |
|---|--------|-------------|-------|
| 01 | [**Foundations & Math**](01_foundations/) | Installation, PyTorch philosophy, linear algebra, calculus, probability, optimization theory | 1 README + 1 script |
| 02 | [**Tensors**](02_tensors/) | Creation, dtypes, operations, indexing, broadcasting, views, strides, memory layout | 1 README + 5 scripts |
| 03 | [**Autograd**](03_autograd/) | Computation graphs, backward pass, gradient control, custom functions, Jacobians, Hessians | 1 README + 4 scripts |

### Part II: Building Blocks

> *Learn every component you need to build neural networks.*

| # | Module | Description | Files |
|---|--------|-------------|-------|
| 04 | [**Neural Networks**](04_neural_networks/) | `nn.Module` lifecycle, all layer types (linear, conv, norm, attention), losses, hooks, save/load | 1 README + 5 scripts |
| 05 | [**Optimizers**](05_optimizers/) | SGD, Adam, AdamW, Muon, Adafactor, all LR schedulers, gradient clipping | 1 README + 3 scripts |
| 06 | [**Data Loading**](06_data_loading/) | `Dataset`, `DataLoader`, custom collate, samplers, MixUp, CutMix, augmentation | 1 README + 4 scripts |

### Part III: Training & Performance

> *Put it all together. Train models fast.*

| # | Module | Description | Files |
|---|--------|-------------|-------|
| 07 | [**Training Pipelines**](07_training/) | Complete training loops, mixed precision (AMP), gradient accumulation/checkpointing, transfer learning, EMA | 1 README + 5 scripts |
| 08 | [**torch.compile**](08_torch_compile/) | Dynamo, AOTAutograd, Inductor — modes, graph breaks, dynamic shapes, custom backends, debugging | 1 README + 5 scripts |
| 09 | [**Attention Mechanisms**](09_attention/) | SDPA, Flash Attention, multi-head attention, FlexAttention, Transformer blocks, RoPE, KV cache | 1 README + 5 scripts |

### Part IV: Scale & Deploy

> *Go from single-GPU to production.*

| # | Module | Description | Files |
|---|--------|-------------|-------|
| 10 | [**Distributed Training**](10_distributed/) | DDP, FSDP2 (`fully_shard`), DeviceMesh, DTensor, Tensor Parallel, Pipeline Parallel, DCP | 1 README + 5 scripts |
| 11 | [**Export & Deployment**](11_export_deploy/) | `torch.export`, dynamic shapes, AOTInductor, NativeRT, ONNX, PT2 Archive | 1 README + 4 scripts |

### Part V: Mastery

> *Build real architectures. Use advanced features. Write proper tests.*

| # | Module | Description | Files |
|---|--------|-------------|-------|
| 12 | [**Model Architectures**](12_model_architectures/) | ResNet, Transformer, GPT (with generation), Vision Transformer (ViT), VAE — complete implementations | 1 README + 5 scripts |
| 13 | [**Advanced Features**](13_advanced/) | Functorch (`vmap`, `grad`), custom operators, quantization, profiling, sparse tensors, debugging | 1 README + 6 scripts |
| 14 | [**Testing & Benchmarking**](14_testing/) | PyTorch `TestCase`, reproducibility, `torch.utils.benchmark` | 1 README + 3 scripts |

---

## Recommended Learning Path

```
 Week 1-2                    Week 3-4                     Week 5-6
┌──────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│  01 Foundations   │   │  04 Neural Networks  │   │  07 Training         │
│  02 Tensors      │──▶│  05 Optimizers       │──▶│  08 torch.compile    │
│  03 Autograd     │   │  06 Data Loading     │   │  09 Attention        │
└──────────────────┘   └──────────────────────┘   └──────────────────────┘
                                                            │
                            Week 9-10                Week 7-8│
                       ┌──────────────────────┐   ┌─────────▼────────────┐
                       │  12 Architectures    │   │  10 Distributed      │
                       │  13 Advanced         │◀──│  11 Export & Deploy  │
                       │  14 Testing          │   └──────────────────────┘
                       └──────────────────────┘
```

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/subinz1/pytorch-reference-guide.git
cd pytorch-reference-guide

# Install PyTorch
pip install torch torchvision torchaudio

# Start with Module 01
cd 01_foundations
python math_with_pytorch.py

# Progress through modules
cd ../02_tensors
python creation_and_properties.py
python operations.py
python broadcasting.py
```

> Most examples run on **CPU only** — no GPU required.
> Examples that need GPU or multi-GPU are clearly marked.

---

## What's Covered

### Core PyTorch
Tensors, Autograd, `nn.Module`, Optimizers, DataLoaders, Training Loops

### Modern PyTorch (2.x+)
| Feature | Module |
|---------|--------|
| `torch.compile` (Dynamo + Inductor) | [08](08_torch_compile/) |
| FlexAttention | [09](09_attention/) |
| FSDP2 (`fully_shard`) | [10](10_distributed/) |
| DTensor & DeviceMesh | [10](10_distributed/) |
| Pipeline Parallelism (ZeroBubble, DualPipeV) | [10](10_distributed/) |
| `torch.export` & PT2 Archive | [11](11_export_deploy/) |
| NativeRT C++ inference engine | [11](11_export_deploy/) |
| Compiled Autograd | [03](03_autograd/), [08](08_torch_compile/) |
| RMSNorm, SiLU, Flash Attention | [04](04_neural_networks/), [09](09_attention/) |
| Functorch (`vmap`, `grad`, `jacrev`) | [13](13_advanced/) |

### Complete Architecture Implementations
| Architecture | Description | Module |
|-------------|-------------|--------|
| **ResNet** | BasicBlock + Bottleneck, ResNet-18/34/50/101 | [12](12_model_architectures/) |
| **Transformer** | Full encoder-decoder with cross-attention | [12](12_model_architectures/) |
| **GPT** | Decoder-only with temperature, top-k, top-p generation | [12](12_model_architectures/) |
| **ViT** | Vision Transformer with patch embedding | [12](12_model_architectures/) |
| **VAE** | Variational Autoencoder with reparameterization trick | [12](12_model_architectures/) |

---

## Repository Stats

```
14 modules | 14 detailed READMEs | 55+ Python scripts | 30,000+ lines of content
```

| Module | README Lines | Python Files | Key Topics |
|--------|:-----------:|:------------:|------------|
| 01 Foundations | 640 | 1 | Math, installation, philosophy |
| 02 Tensors | 930 | 5 | Creation, ops, indexing, broadcasting |
| 03 Autograd | 833 | 4 | Gradients, custom functions, Jacobians |
| 04 Neural Networks | 1,003 | 5 | All layers, losses, hooks, save/load |
| 05 Optimizers | 469 | 3 | SGD, Adam, schedulers |
| 06 Data Loading | 559 | 4 | Dataset, DataLoader, augmentation |
| 07 Training | 832 | 5 | AMP, transfer learning, EMA |
| 08 torch.compile | 526 | 5 | Dynamo, Inductor, graph breaks |
| 09 Attention | 495 | 5 | SDPA, FlexAttention, RoPE |
| 10 Distributed | 1,322 | 5 | DDP, FSDP2, TP, PP, DCP |
| 11 Export & Deploy | 836 | 4 | torch.export, AOTInductor, NativeRT |
| 12 Architectures | 646 | 5 | ResNet, GPT, ViT, VAE |
| 13 Advanced | 624 | 6 | functorch, profiling, custom ops |
| 14 Testing | 363 | 3 | TestCase, reproducibility, benchmarks |

---

## Requirements

```
Python >= 3.10
PyTorch >= 2.0
```

```bash
pip install torch torchvision torchaudio
```

---

## Contributing

This is a personal learning reference. If you find errors or have suggestions, feel free to open an issue.

## License

This guide is a personal compilation for educational purposes. PyTorch itself is BSD-licensed.

---

<p align="center">
  <i>Built with PyTorch v2.13+ — Updated June 2026</i>
</p>
