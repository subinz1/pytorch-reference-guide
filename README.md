# PyTorch: The Definitive Reference Guide

A comprehensive, single-source PyTorch reference — from mathematical foundations to production deployment. Updated for PyTorch **v2.13+** (June 2026).

## Repository Structure

```
├── README.md                              # This file
├── PYTORCH_MASTER_REFERENCE_2026.md       # Complete reference (3,400+ lines)
├── codes/                                 # Runnable example code
│   ├── 01_tensors/                        # Tensor creation, operations, broadcasting
│   ├── 02_autograd/                       # Automatic differentiation, custom functions
│   ├── 03_neural_networks/                # nn.Module, layers, loss functions
│   ├── 04_training/                       # Training loops, AMP, checkpointing
│   ├── 05_torch_compile/                  # torch.compile, Dynamo, Inductor
│   ├── 06_distributed/                    # DDP, FSDP2, DeviceMesh, DTensor
│   ├── 07_attention/                      # SDPA, FlexAttention, Transformers
│   ├── 08_data_loading/                   # Dataset, DataLoader, augmentation
│   ├── 09_export_deploy/                  # torch.export, AOTInductor
│   └── 10_advanced/                       # functorch, sparse, quantization, profiling
└── docs/                                  # Additional documentation
    └── QUICK_REFERENCE.md                 # Cheat sheet / quick lookup
```

## What's Covered

### Reference Guide
- **Part 0**: Mathematical Foundations (linear algebra, calculus, probability, optimization)
- **Parts I-III**: Introduction, Architecture & Internals, Tensor Operations
- **Part IV**: Autograd + Compiled Autograd
- **Part V**: Neural Networks + FlexAttention, RMSNorm, modern activations
- **Part VI**: Optimization (Adam, AdamW, Muon, Adafactor, schedulers)
- **Parts VII-VIII**: Data Loading, Training Pipelines (AMP, gradient checkpointing)
- **Part IX**: torch.compile (Dynamo, AOTAutograd, Inductor, dynamic shapes)
- **Part X**: Distributed (FSDP2, DTensor, Tensor/Pipeline Parallelism, DeviceMesh)
- **Part XI**: Export & Deployment (torch.export, NativeRT, AOTInductor)
- **Parts XII-XVI**: Hardware, Advanced Features, Model Architectures, Testing, Build System
- **Appendices**: Math derivations, full nn reference, linalg, distributions, debugging

### Example Code
Every `codes/` subfolder contains self-contained, runnable Python scripts with inline explanations and expected output.

## What's New (vs March 2026 references)
- FlexAttention (`torch.nn.attention.flex_attention`)
- FSDP2 (`fully_shard`) — composable sharded training
- NativeRT — C++ inference engine for exported models
- Pipeline Parallelism (1F1B, GPipe, ZeroBubble, DualPipeV)
- DeviceMesh, DTensor (stable at `torch.distributed.tensor`)
- Compiled Autograd and Compiled Optimizers
- Muon, Adafactor optimizers
- Context Parallel for long sequences
- PT2 Archive format, draft_export
- Compiler stances, `torch.accelerator` API
- Float8/Float4 dtypes, CuTeDSL codegen

## Requirements

```bash
pip install torch torchvision torchaudio
```

Most examples run on CPU. GPU examples note requirements in comments.

## License

This is a personal reference compilation. PyTorch is BSD-licensed.
