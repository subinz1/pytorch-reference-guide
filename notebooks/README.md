# 📓 Interactive Jupyter Notebooks

These are interactive playbooks — open in Jupyter or Google Colab, run cells, experiment, learn by doing.

## How to Use

### Local Jupyter
```bash
pip install jupyter torch torchvision torchaudio
cd notebooks
jupyter notebook
```

### Google Colab
Upload any `.ipynb` file to [Google Colab](https://colab.research.google.com/) — free GPU included.

## Notebook Index

| # | Notebook | Module | Topics |
|---|----------|--------|--------|
| 01 | `01_tensors_masterclass.ipynb` | [Tensors](../02_tensors/) | Creation, ops, broadcasting, views, exercises |
| 02 | `02_autograd_from_scratch.ipynb` | [Autograd](../03_autograd/) | Gradients, graphs, custom functions, gradient descent exercise |
| 03 | `03_neural_networks_playbook.ipynb` | [Neural Networks](../04_neural_networks/) | nn.Module, layers, losses, CNN building |
| 04 | `04_training_complete_guide.ipynb` | [Training](../07_training/) | Full loop, AMP, grad accumulation, transfer learning |
| 05 | `05_optimizers_and_schedulers.ipynb` | [Optimizers](../05_optimizers/) | Loss surface viz, optimizer trajectories, LR plots |
| 06 | `06_data_loading_pipeline.ipynb` | [Data Loading](../06_data_loading/) | Custom datasets, collate, MixUp, splits |
| 07 | `07_attention_and_transformers.ipynb` | [Attention](../09_attention/) | SDPA, MHA, FlexAttention, Transformer blocks |
| 08 | `08_torch_compile_masterclass.ipynb` | [torch.compile](../08_torch_compile/) | Compile stages, graph breaks, dynamic shapes, benchmarks |
| 09 | `09_model_architectures.ipynb` | [Architectures](../12_model_architectures/) | ResNet, GPT, ViT from scratch |
| 10 | `10_distributed_overview.ipynb` | [Distributed](../10_distributed/) | DDP, FSDP2, DeviceMesh, parallelism strategies |
| 11 | `11_export_and_deploy.ipynb` | [Export](../11_export_deploy/) | torch.export, dynamic shapes, deployment paths |
| 12 | `12_advanced_features.ipynb` | [Advanced](../13_advanced/) | functorch, custom ops, sparse, FFT, profiling |
| 13 | `13_testing_and_reproducibility.ipynb` | [Testing](../14_testing/) | Seeds, deterministic mode, benchmarking |
| 14 | `14_practical_utilities.ipynb` | [Utilities](../15_practical_utilities/) | Parametrize, pruning, packing, fusion |
| 15 | `15_selective_checkpointing.ipynb` | [Checkpointing](../16_activation_checkpointing/) | Activation checkpointing, SAC, memory tradeoffs |
| 16 | `16_activation_checkpointing.ipynb` | [Checkpointing](../16_activation_checkpointing/) | Basic/selective checkpointing, SAC, memory tradeoffs |
| 17 | `17_compile_decorators.ipynb` | [Compile Control](../17_compile_decorators/) | Stances, disable, allow_in_graph, mark_dynamic, explain |
| 18 | `18_torch_package.ipynb` | [torch.package](../18_torch_package/) | PackageExporter/Importer, intern/extern/mock, model shipping |
| 19 | `19_torch_function_dispatch.ipynb` | [Dispatch](../19_torch_function_dispatch/) | __torch_function__, __torch_dispatch__, modes, custom tensors |

## Requirements

All notebooks run on **CPU only** — no GPU required. GPU-specific examples are clearly marked.

```bash
pip install torch torchvision torchaudio jupyter
```
