# Guided Learning Path — Beginner → Advanced

A recommended order through this repository. Each stage lists **core modules**
(do these) and **optional deep dives** (when you need them). Pair every module
README with its script(s) and notebook when available.

## Stage 0 — Setup (½ day)

- Install PyTorch (`requirements.txt`), confirm `import torch` and a tiny tensor op.
- Skim the root [README](../README.md) course map so you know where things live.

## Stage 1 — Foundations (1–2 weeks)

| Order | Module | Goal |
|------:|--------|------|
| 1 | [01 Foundations](../01_foundations/) | Math + PyTorch mental model |
| 2 | [02 Tensors](../02_tensors/) | Shapes, broadcasting, memory |
| 3 | [03 Autograd](../03_autograd/) | Graphs, `backward`, grad control |

**Checkpoint:** Write `y = (Wx + b).relu().sum(); y.backward()` and explain every grad.

## Stage 2 — Building Blocks (1–2 weeks)

| Order | Module | Goal |
|------:|--------|------|
| 4 | [04 Neural Networks](../04_neural_networks/) | `nn.Module`, layers, losses |
| 5 | [05 Optimizers](../05_optimizers/) | AdamW, schedulers, clipping |
| 6 | [06 Data Loading](../06_data_loading/) | Dataset/DataLoader patterns |

**Checkpoint:** Train an MLP on a toy dataset with validation logging.

## Stage 3 — Training & Performance (2 weeks)

| Order | Module | Goal |
|------:|--------|------|
| 7 | [07 Training](../07_training/) | Full loops, AMP, EMA |
| 8 | [08 torch.compile](../08_torch_compile/) | Dynamo/Inductor basics |
| 9 | [09 Attention](../09_attention/) | SDPA, MHA, modern attention |
| — | [29 Mixed Precision](../29_mixed_precision/) | Optional AMP/FP8 depth |
| — | [16 Checkpointing](../16_activation_checkpointing/) | Optional memory tradeoffs |

**Checkpoint:** Compile a small model; explain one graph break you hit.

## Stage 4 — Scale & Deploy (2 weeks)

| Order | Module | Goal |
|------:|--------|------|
| 10 | [10 Distributed](../10_distributed/) | DDP → FSDP overview |
| 11 | [11 Export & Deploy](../11_export_deploy/) | `torch.export`, deploy paths |
| — | [47 DDP Patterns](../47_ddp_patterns/) | Pitfalls + `ddp_pitfalls.md` |
| — | [43 Production Serving](../43_production_serving/) | Optional serving patterns |

**Checkpoint:** Launch a 2-process DDP smoke test with `torchrun`.

## Stage 5 — Architectures & Projects (2–3 weeks)

| Order | Module | Goal |
|------:|--------|------|
| 12 | [12 Architectures](../12_model_architectures/) | ResNet/GPT/ViT/VAE |
| 39 | [Text Classifier](../39_text_classifier/) | End-to-end NLP project |
| 40 | [Image Classifier](../40_image_classifier/) | End-to-end vision project |
| 41 | [Diffusion](../41_diffusion_model/) | Generative project track |

Pick **one** project module and finish it fully before starting another.

## Stage 6 — Advanced Tooling (ongoing)

Study when your work requires them — not all at once:

| Need | Modules |
|------|---------|
| Custom gradients | [48 Custom Autograd](../48_custom_autograd/), [38 Compiled Autograd](../38_compiled_autograd/) |
| Memory at LLM scale | [49 Advanced Checkpointing](../49_gradient_checkpointing_advanced/), [26 Memory](../26_memory_profiling/) |
| Sparse data / graphs | [50 Sparse](../50_torch_sparse/) |
| Profiling | [45 Profiler](../45_torch_profiler/) + `chrome_trace_guide.md` |
| Quantization | [46 Quantization](../46_quantization_recipes/) + `qat_walkthrough.md` |
| Kernels / internals | [25 Triton](../25_triton_kernels/), [35 Dispatcher](../35_dispatcher/), [36 C++ Extensions](../36_cpp_extensions/) |
| LLM recipes | [22 LLM Recipes](../22_llm_recipes/), [34 Fine-Tuning](../34_llm_finetuning/) |

## Stage 7 — CI / Contributor Track (optional)

For PyTorch contributor / backend CI workflows:

- [CRCR Downstream CI](../40_crcr_downstream_ci/)
- [Targeted Tests](../41_targeted_tests/)

## Weekly Rhythm (suggested)

1. **Read** the module README (theory + when-to-use).
2. **Run** every script; change one hyperparameter / shape and re-run.
3. **Work** the notebook cells without peeking at outputs first.
4. **Write** 5–10 lines of notes: what surprised you, what you’d reuse.

## Done When…

You can train, compile, profile, and (optionally) distribute a model you understand
end-to-end — and you know which deep-dive module to open when something breaks.
