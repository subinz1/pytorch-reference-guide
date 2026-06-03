# PyTorch: The Complete Learning Guide
## From Absolute Beginner to Advanced Practitioner

A structured, self-contained PyTorch course organized into 14 modules. Each module contains:
- **README.md** — Detailed explanations, theory, formulas, and inline code examples
- **Python scripts** — Fully runnable examples you can execute with `python filename.py`

**Target:** Take someone with basic Python knowledge and make them a proficient PyTorch developer.

**PyTorch Version:** 2.13+ (June 2026)

---

## Course Structure

### Part I: Foundations (Modules 01-03)

| Module | Topic | What You'll Learn |
|--------|-------|-------------------|
| **[01_foundations](01_foundations/)** | Math & Setup | Installation, PyTorch philosophy, linear algebra, calculus, probability — all with PyTorch code |
| **[02_tensors](02_tensors/)** | Tensors | Creation, operations, indexing, broadcasting, views, strides, memory layout |
| **[03_autograd](03_autograd/)** | Automatic Differentiation | Computation graphs, gradients, custom functions, Jacobians, Hessians |

### Part II: Building Blocks (Modules 04-06)

| Module | Topic | What You'll Learn |
|--------|-------|-------------------|
| **[04_neural_networks](04_neural_networks/)** | nn.Module | All layer types (linear, conv, norm, attention), losses, hooks, save/load |
| **[05_optimizers](05_optimizers/)** | Optimization | SGD, Adam, AdamW, learning rate schedulers, gradient clipping |
| **[06_data_loading](06_data_loading/)** | Data Pipeline | Dataset, DataLoader, augmentation, samplers, custom collate |

### Part III: Training (Modules 07-09)

| Module | Topic | What You'll Learn |
|--------|-------|-------------------|
| **[07_training](07_training/)** | Training Pipelines | Complete training loops, AMP, gradient accumulation, transfer learning, EMA |
| **[08_torch_compile](08_torch_compile/)** | Compilation | torch.compile, Dynamo, Inductor, dynamic shapes, graph breaks, debugging |
| **[09_attention](09_attention/)** | Attention Mechanisms | SDPA, multi-head attention, FlexAttention, Transformer blocks |

### Part IV: Scale & Deploy (Modules 10-11)

| Module | Topic | What You'll Learn |
|--------|-------|-------------------|
| **[10_distributed](10_distributed/)** | Distributed Training | DDP, FSDP2, DeviceMesh, DTensor, Tensor/Pipeline Parallelism |
| **[11_export_deploy](11_export_deploy/)** | Deployment | torch.export, AOTInductor, NativeRT, ONNX, dynamic shapes |

### Part V: Mastery (Modules 12-14)

| Module | Topic | What You'll Learn |
|--------|-------|-------------------|
| **[12_model_architectures](12_model_architectures/)** | Architectures | ResNet, Transformer, GPT, ViT, VAE — complete implementations |
| **[13_advanced](13_advanced/)** | Advanced Features | Functorch, custom ops, quantization, profiling, sparse tensors, debugging |
| **[14_testing](14_testing/)** | Testing & Benchmarking | PyTorch TestCase, reproducibility, benchmarking |

---

## Recommended Learning Path

### Week 1-2: Core Fundamentals
```
01_foundations → 02_tensors → 03_autograd
```
Understand tensors, operations, and how gradients work.

### Week 3-4: Building Models
```
04_neural_networks → 05_optimizers → 06_data_loading
```
Learn all the building blocks: layers, losses, optimizers, data pipelines.

### Week 5-6: Training & Performance
```
07_training → 08_torch_compile → 09_attention
```
Master training loops, compilation for speed, and attention mechanisms.

### Week 7-8: Scale & Production
```
10_distributed → 11_export_deploy
```
Scale to multiple GPUs and deploy models.

### Week 9-10: Expertise
```
12_model_architectures → 13_advanced → 14_testing
```
Build real architectures, use advanced features, write proper tests.

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
cat README.md            # Read the theory
python math_with_pytorch.py  # Run the examples

# Progress through modules sequentially
cd ../02_tensors
cat README.md
python creation_and_properties.py
# ... and so on
```

Most examples run on **CPU only** — no GPU required. Examples that need GPU/multi-GPU are clearly marked.

---

## What's Covered (Key Features)

### Core PyTorch
- Tensors, Autograd, nn.Module, Optimizers, DataLoaders

### Modern PyTorch (2.x+)
- `torch.compile` (Dynamo + Inductor)
- FlexAttention API
- FSDP2 (`fully_shard`)
- DTensor and DeviceMesh
- Pipeline Parallelism (ZeroBubble, DualPipeV)
- `torch.export` and PT2 Archive
- NativeRT inference engine
- Compiled Autograd
- RMSNorm, SiLU, Flash Attention
- Float8/BFloat16 dtypes

### Architecture Implementations
- ResNet (BasicBlock + Bottleneck)
- Transformer (Encoder-Decoder)
- GPT (Decoder-only with generation)
- Vision Transformer (ViT)
- VAE (with reparameterization trick)

---

## Requirements

```
Python >= 3.10
PyTorch >= 2.0 (most examples)
```

```bash
pip install torch torchvision torchaudio
```

## License

Personal reference compilation. PyTorch is BSD-licensed.
