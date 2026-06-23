<p align="center">
  <img src="https://pytorch.org/assets/images/pytorch-logo.png" width="120" alt="PyTorch Logo">
</p>

<h1 align="center">PyTorch: The Complete Learning Guide</h1>
<h3 align="center">From Absolute Beginner to Advanced Practitioner</h3>

<p align="center">
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.14%2B-EE4C2C?logo=pytorch" alt="PyTorch"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python"></a>
  <a href="#course-structure"><img src="https://img.shields.io/badge/Modules-31-blue" alt="Modules"></a>
  <a href="#course-structure"><img src="https://img.shields.io/badge/Code_Examples-90%2B-green" alt="Examples"></a>
  <a href="#interactive-notebooks"><img src="https://img.shields.io/badge/Notebooks-31-blueviolet?logo=jupyter" alt="Notebooks"></a>
  <a href="#course-structure"><img src="https://img.shields.io/badge/Lines-55%2C000%2B-orange" alt="Lines"></a>
</p>

<p align="center">
  A structured, self-contained PyTorch course organized into <strong>31 modules</strong> and <strong>31 interactive notebooks</strong>.<br>
  Each module contains detailed explanations, theory, formulas, runnable Python scripts, and a Jupyter playbook.
</p>

---

## About This Guide

This repository takes someone with **basic Python knowledge** and makes them a **proficient PyTorch developer**. Three ways to learn:

- **`README.md`** — In-depth explanations with theory, math formulas, diagrams, and inline code
- **Python scripts** — Self-contained, runnable examples (`python filename.py`) with detailed output
- **Jupyter notebooks** — Interactive playbooks with markdown + code cells, exercises, and visualizations

**Updated for PyTorch 2.14+ (June 2026)** — includes modern features like `torch.compile`, FlexAttention, FSDP2, and more.

> **New modules added daily** — see the [Bonus: Practical Deep Dives](#bonus-practical-deep-dives) section for the latest additions.

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

### Bonus: Practical Deep Dives

> *The hidden toolkit most tutorials never teach. Updated incrementally.*

| # | Module | Description | Files |
|---|--------|-------------|-------|
| 15 | [**Practical Utilities**](15_practical_utilities/) | Weight parametrization, pruning, spectral/weight norm, sequence packing, nested tensors, Conv-BN fusion | 1 README + 4 scripts |
| 16 | [**Activation Checkpointing**](16_activation_checkpointing/) | Basic checkpointing, checkpoint_sequential, Selective Activation Checkpointing (SAC), policies | 1 README + 1 script |
| 17 | [**Compile Decorators & Control**](17_compile_decorators/) | Stances, disable, allow_in_graph, substitute_in_graph, mark_dynamic, explain, TORCH_LOGS, upstream updates | 1 README + 1 script |
| 18 | [**torch.package**](18_torch_package/) | PackageExporter, PackageImporter, intern/extern/mock/deny, hermetic model archives, comparison with torch.save/export | 1 README + 1 script |
| 19 | [**Tensor Subclassing & Dispatch**](19_torch_function_dispatch/) | `__torch_function__`, `__torch_dispatch__`, TorchFunctionMode, TorchDispatchMode, custom tensor types | 1 README + 1 script |
| 20 | [**Backends Tuning**](20_backends_tuning/) | `torch.backends.cudnn`, TF32, `set_float32_matmul_precision`, OpenMP, opt_einsum, performance checklist | 1 README + 1 script |
| 21 | [**CUDA Graphs**](21_cuda_graphs/) | `torch.cuda.CUDAGraph`, capture/replay, static inputs, `reduce-overhead` mode, `make_graphed_callables` | 1 README + 1 script |
| 22 | [**LLM Training Recipes**](22_llm_recipes/) | RoPE, KV Cache, GQA, Sliding Window, RMSNorm, SwiGLU, weight tying, bf16, gradient accumulation, mini-LLM | 1 README + 3 scripts |
| 23 | [**torch.fx Graph Transforms**](23_fx_transforms/) | Symbolic tracing, FX Graph IR, graph passes, pattern matching, Interpreter, Transformer | 1 README + 2 scripts |
| 24 | [**torch.masked (MaskedTensor)**](24_masked_tensor/) | MaskedTensor, masked reductions, masked softmax, mask propagation, padded sequences | 1 README + 1 script |
| 25 | [**Custom Triton Kernels**](25_triton_kernels/) | Triton programming model, fused kernels, torch.library integration, autotuning, TorchInductor | 1 README + 2 scripts |
| 26 | [**Memory Profiling & Optimization**](26_memory_profiling/) | GPU memory anatomy, profiling tools, snapshots, optimization techniques, memory-efficient training | 1 README + 2 scripts |
| 27 | [**Multi-GPU Inference Patterns**](27_multi_gpu_inference/) | Tensor Parallel, Pipeline Parallel, device_map, KV cache, quantization, continuous batching, AOTInductor | 1 README + 2 scripts |
| 28 | [**torch.utils.benchmark Deep Dive**](28_benchmarking/) | Timer, blocked_autorange, Compare, Fuzzer, Callgrind, torch.compile benchmarking, shape sweeps | 1 README + 2 scripts |
| 29 | [**Mixed Precision Deep Dive**](29_mixed_precision/) | FP32, FP16, BF16, FP8, AMP autocast, GradScaler, FSDP2 mixed precision, torch.compile | 1 README + 2 scripts |
| 30 | [**Debugging PyTorch Models**](30_debugging/) | Anomaly detection, NaN/Inf checks, gradient flow, shape debugging, torch.compile debugging, memory leaks | 1 README + 2 scripts |
| 31 | [**torchao — Architecture Optimization**](31_torchao/) | Quantization (INT8/INT4/FP8), sparsity (2:4), `quantize_()`, torch.compile integration, PT2E flow | 1 README + 2 scripts |

### Interactive Notebooks

> *Open in Jupyter or Google Colab. Run cells, experiment, learn by doing.*

The [`notebooks/`](notebooks/) folder contains **31 interactive playbooks** — one per module:

| # | Notebook | Topic |
|---|----------|-------|
| 01 | [`01_tensors_masterclass.ipynb`](notebooks/01_tensors_masterclass.ipynb) | Tensor creation, operations, broadcasting, views |
| 02 | [`02_autograd_from_scratch.ipynb`](notebooks/02_autograd_from_scratch.ipynb) | Gradients, computation graphs, custom functions |
| 03 | [`03_neural_networks_playbook.ipynb`](notebooks/03_neural_networks_playbook.ipynb) | nn.Module, layers, losses, save/load |
| 04 | [`04_training_complete_guide.ipynb`](notebooks/04_training_complete_guide.ipynb) | Training loops, AMP, checkpointing, plotting |
| 05 | [`05_optimizers_and_schedulers.ipynb`](notebooks/05_optimizers_and_schedulers.ipynb) | Optimizer comparison, LR schedulers visualized |
| 06 | [`06_data_loading_pipeline.ipynb`](notebooks/06_data_loading_pipeline.ipynb) | Dataset, DataLoader, collate, augmentation |
| 07 | [`07_attention_and_transformers.ipynb`](notebooks/07_attention_and_transformers.ipynb) | SDPA, MHA, FlexAttention, Transformer blocks |
| 08 | [`08_torch_compile_masterclass.ipynb`](notebooks/08_torch_compile_masterclass.ipynb) | Compile modes, graph breaks, dynamic shapes |
| 09 | [`09_model_architectures.ipynb`](notebooks/09_model_architectures.ipynb) | Build ResNet, GPT, ViT from scratch |
| 10 | [`10_distributed_overview.ipynb`](notebooks/10_distributed_overview.ipynb) | DDP, FSDP2, DeviceMesh, parallelism strategies |
| 11 | [`11_export_and_deploy.ipynb`](notebooks/11_export_and_deploy.ipynb) | torch.export, dynamic shapes, deployment paths |
| 12 | [`12_advanced_features.ipynb`](notebooks/12_advanced_features.ipynb) | Functorch, custom ops, sparse, FFT, profiling |
| 13 | [`13_testing_and_reproducibility.ipynb`](notebooks/13_testing_and_reproducibility.ipynb) | Seeds, deterministic mode, benchmarking |
| 14 | [`14_practical_utilities.ipynb`](notebooks/14_practical_utilities.ipynb) | Parametrize, pruning, packing, fusion |
| 15 | [`15_selective_checkpointing.ipynb`](notebooks/15_selective_checkpointing.ipynb) | Activation checkpointing, SAC, memory tradeoffs |
| 16 | [`16_activation_checkpointing.ipynb`](notebooks/16_activation_checkpointing.ipynb) | Basic/selective checkpointing, policies, benchmarks |
| 17 | [`17_compile_decorators.ipynb`](notebooks/17_compile_decorators.ipynb) | Stances, disable, allow_in_graph, mark_dynamic, explain |
| 18 | [`18_torch_package.ipynb`](notebooks/18_torch_package.ipynb) | PackageExporter/Importer, intern/extern/mock, model shipping |
| 19 | [`19_torch_function_dispatch.ipynb`](notebooks/19_torch_function_dispatch.ipynb) | `__torch_function__`, `__torch_dispatch__`, modes, custom tensors |
| 20 | [`20_backends_tuning.ipynb`](notebooks/20_backends_tuning.ipynb) | cuDNN, TF32, OpenMP, opt_einsum, performance checklist |
| 21 | [`21_cuda_graphs.ipynb`](notebooks/21_cuda_graphs.ipynb) | CUDA Graph capture/replay, static inputs, benchmarking |
| 22 | [`22_llm_recipes.ipynb`](notebooks/22_llm_recipes.ipynb) | RoPE, KV Cache, GQA, SwiGLU, RMSNorm, mini-LLM |
| 23 | [`23_fx_transforms.ipynb`](notebooks/23_fx_transforms.ipynb) | Symbolic tracing, FX Graph IR, passes, pattern matching |
| 24 | [`24_masked_tensor.ipynb`](notebooks/24_masked_tensor.ipynb) | torch.masked, MaskedTensor, masked reductions, masked softmax |
| 25 | [`25_triton_kernels.ipynb`](notebooks/25_triton_kernels.ipynb) | Custom Triton kernels, fusion, torch.library, autotuning |
| 26 | [`26_memory_profiling.ipynb`](notebooks/26_memory_profiling.ipynb) | GPU memory profiling, optimization techniques, memory estimation |
| 27 | [`27_multi_gpu_inference.ipynb`](notebooks/27_multi_gpu_inference.ipynb) | Multi-GPU inference, TP, PP, quantization, continuous batching |
| 28 | [`28_benchmarking.ipynb`](notebooks/28_benchmarking.ipynb) | Timer, blocked_autorange, Compare, Fuzzer, torch.compile benchmarking |
| 29 | [`29_mixed_precision.ipynb`](notebooks/29_mixed_precision.ipynb) | FP32, FP16, BF16, FP8, AMP, GradScaler, FSDP2 mixed precision |
| 30 | [`30_debugging.ipynb`](notebooks/30_debugging.ipynb) | Anomaly detection, NaN checks, gradient flow, compile debugging |
| 31 | [`31_torchao.ipynb`](notebooks/31_torchao.ipynb) | Quantization, INT8/INT4, sparsity, torch.compile integration |

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
                       └──────────┬───────────┘
                                  │
                           Ongoing Deep Dives
                       ┌──────────▼───────────┐
                       │  15 Utilities         │
                       │  16 Checkpointing     │
                       │  17 Compile Control   │
                       │  18 torch.package     │
                       │  19 Tensor Dispatch   │
                       │  ... more coming      │
                       └──────────────────────┘
```

---

## Quick Start

### Option A: Read & Run Scripts
```bash
git clone https://github.com/subinz1/pytorch-reference-guide.git
cd pytorch-reference-guide
pip install torch torchvision torchaudio

# Start with Module 01
cd 01_foundations
python math_with_pytorch.py

# Progress through modules
cd ../02_tensors
python creation_and_properties.py
```

### Option B: Interactive Notebooks (Recommended for Beginners)
```bash
pip install jupyter
cd notebooks
jupyter notebook 01_tensors_masterclass.ipynb
```

### Option C: Google Colab
Upload any notebook from the `notebooks/` folder to [Google Colab](https://colab.research.google.com/) and run it with a free GPU.

> Most examples run on **CPU only** — no GPU required.
> Examples that need GPU or multi-GPU are clearly marked.

---

## What's Covered

### Core PyTorch
Tensors, Autograd, `nn.Module`, Optimizers, DataLoaders, Training Loops

### Modern PyTorch (2.x+)
| Feature | Module |
|---------|--------|
| `torch.compile` (Dynamo + Inductor) | [08](08_torch_compile/), [17](17_compile_decorators/) |
| Compile decorators & stances | [17](17_compile_decorators/) |
| FlexAttention | [09](09_attention/) |
| FSDP2 (`fully_shard`) | [10](10_distributed/) |
| DTensor & DeviceMesh | [10](10_distributed/) |
| Pipeline Parallelism (ZeroBubble, DualPipeV) | [10](10_distributed/) |
| `torch.export` & PT2 Archive | [11](11_export_deploy/) |
| NativeRT C++ inference engine | [11](11_export_deploy/) |
| `torch.package` (hermetic model archives) | [18](18_torch_package/) |
| Compiled Autograd | [03](03_autograd/), [08](08_torch_compile/) |
| Selective Activation Checkpointing (SAC) | [16](16_activation_checkpointing/) |
| RMSNorm, SiLU, Flash Attention | [04](04_neural_networks/), [09](09_attention/) |
| Functorch (`vmap`, `grad`, `jacrev`) | [13](13_advanced/) |
| `__torch_function__` & `__torch_dispatch__` | [19](19_torch_function_dispatch/) |
| Weight parametrization & pruning | [15](15_practical_utilities/) |
| Nested tensors & sequence packing | [15](15_practical_utilities/) |

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
31 modules | 31 READMEs | 90+ Python scripts | 31 Jupyter notebooks | 60,000+ lines of content
```

| Module | README | Scripts | Notebook | Key Topics |
|--------|:------:|:-------:|:--------:|------------|
| 01 Foundations | 640 | 1 | 01 | Math, installation, philosophy |
| 02 Tensors | 930 | 5 | 01 | Creation, ops, indexing, broadcasting |
| 03 Autograd | 833 | 4 | 02 | Gradients, custom functions, Jacobians |
| 04 Neural Networks | 1,003 | 5 | 03 | All layers, losses, hooks, save/load |
| 05 Optimizers | 469 | 3 | 05 | SGD, Adam, schedulers |
| 06 Data Loading | 559 | 4 | 06 | Dataset, DataLoader, augmentation |
| 07 Training | 832 | 5 | 04 | AMP, transfer learning, EMA |
| 08 torch.compile | 526 | 5 | 08 | Dynamo, Inductor, graph breaks |
| 09 Attention | 495 | 5 | 07 | SDPA, FlexAttention, RoPE |
| 10 Distributed | 1,322 | 5 | 10 | DDP, FSDP2, TP, PP, DCP |
| 11 Export & Deploy | 836 | 4 | 11 | torch.export, AOTInductor, NativeRT |
| 12 Architectures | 646 | 5 | 09 | ResNet, GPT, ViT, VAE |
| 13 Advanced | 624 | 6 | 12 | functorch, profiling, custom ops |
| 14 Testing | 363 | 3 | 13 | TestCase, reproducibility, benchmarks |
| 15 Practical Utilities | 432 | 4 | 14 | Parametrize, pruning, nested tensors |
| 16 Activation Checkpointing | 260 | 1 | 15, 16 | SAC, memory/compute tradeoffs |
| 17 Compile Decorators | 224 | 1 | 17 | Stances, disable, mark_dynamic |
| 18 torch.package | 331 | 1 | 18 | PackageExporter/Importer |
| 19 Tensor Dispatch | 387 | 1 | 19 | `__torch_function__`, `__torch_dispatch__` |
| 20 Backends Tuning | 320 | 1 | 20 | cuDNN, TF32, OpenMP, opt_einsum |
| 21 CUDA Graphs | 350+ | 1 | 21 | Graph capture, static inputs, reduce-overhead |
| 22 LLM Training Recipes | 400+ | 3 | 22 | RoPE, KV Cache, GQA, SwiGLU, mini-LLM |
| 23 FX Graph Transforms | 400+ | 2 | 23 | Symbolic tracing, graph IR, passes, patterns |
| 24 MaskedTensor | 300+ | 1 | 24 | torch.masked, masked reductions, softmax, propagation |
| 25 Triton Kernels | 400+ | 2 | 25 | Triton programming, fusion, torch.library, autotuning |
| 26 Memory Profiling | 400+ | 2 | 26 | GPU memory anatomy, profiling, optimization, estimation |
| 27 Multi-GPU Inference | 400+ | 2 | 27 | TP, PP, device_map, quantization, continuous batching |
| 28 Benchmarking | 400+ | 2 | 28 | Timer, Compare, Fuzzer, Callgrind, torch.compile |
| 29 Mixed Precision | 450+ | 2 | 29 | FP32, FP16, BF16, FP8, AMP, GradScaler, FSDP2 |
| 30 Debugging | 400+ | 2 | 30 | Anomaly detection, NaN, gradients, compile debugging |
| 31 torchao | 400+ | 2 | 31 | Quantization, INT8/INT4/FP8, sparsity, torch.compile |

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

## Daily Updates Roadmap

This guide grows incrementally. Upcoming topics:

| Day | Topic | Status |
|-----|-------|--------|
| Day 1 | Practical Utilities (parametrize, pruning, fusion) | Done |
| Day 2 | Selective Activation Checkpointing (SAC) | Done |
| Day 3 | `torch.compile` Decorators Deep Dive | **Done** |
| Day 4 | `torch.package` — Model Packaging | **Done** |
| Day 5 | `__torch_function__` & Tensor Subclassing | **Done** |
| Day 6 | `torch.backends` Performance Tuning | **Done** |
| Day 7 | CUDA Graphs | **Done** |
| Day 8 | LLM Training Recipes (RoPE, KV Cache) | **Done** |
| Day 9 | `torch.fx` Graph Transforms | **Done** |
| Day 10 | `torch.masked` (MaskedTensor) | **Done** |
| Day 11 | Custom Triton Kernels | **Done** |
| Day 12 | Memory Profiling & Optimization | **Done** |
| Day 13 | Multi-GPU Inference Patterns | **Done** |
| Day 14 | `torch.utils.benchmark` Deep Dive | **Done** |
| Day 15 | Mixed Precision Deep Dive (FP8) | **Done** |
| Day 16 | Debugging PyTorch Models | **Done** |
| Day 17 | torchao — Architecture Optimization | **Done** |

## Contributing

This is a personal learning reference. If you find errors or have suggestions, feel free to open an issue.

## License

This guide is a personal compilation for educational purposes. PyTorch itself is BSD-licensed.

---

<p align="center">
  <i>Built with PyTorch v2.14+ — Updated June 2026</i>
</p>
