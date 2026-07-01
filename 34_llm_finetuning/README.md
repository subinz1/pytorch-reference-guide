# Module 34: End-to-End — Fine-Tuning an LLM

<div align="center">

[← Previous Module (Model Interpretability)](../33_interpretability/) | [🏠 Home](../README.md) | Next Module →

**Capstone Project**: Tying everything together

</div>

---

> **Prerequisites**: [Module 04 (Neural Networks)](../04_neural_networks/), [Module 07 (Training)](../07_training/), [Module 08 (torch.compile)](../08_torch_compile/), [Module 09 (Attention)](../09_attention/), [Module 16 (Activation Checkpointing)](../16_activation_checkpointing/), [Module 22 (LLM Recipes)](../22_llm_recipes/), [Module 29 (Mixed Precision)](../29_mixed_precision/)
>
> **Time**: ~4 hours
>
> **Files**: `lora_adapter.py`, `finetuning_pipeline.py`, `evaluation_and_export.py`

---

## Table of Contents

1. [Why Fine-Tune?](#1-why-fine-tune)
2. [LoRA (Low-Rank Adaptation)](#2-lora-low-rank-adaptation)
3. [QLoRA](#3-qlora)
4. [Applying LoRA to a Transformer](#4-applying-lora-to-a-transformer)
5. [Data Preparation](#5-data-preparation)
6. [Training Loop with All Best Practices](#6-training-loop-with-all-best-practices)
7. [Scaling with FSDP2](#7-scaling-with-fsdp2)
8. [Evaluation](#8-evaluation)
9. [Merging and Exporting](#9-merging-and-exporting)
10. [Complete Workflow Summary](#10-complete-workflow-summary)
11. [Hyperparameter Guide](#11-hyperparameter-guide)
12. [Upstream Updates (June 29-30, 2026)](#12-upstream-updates-june-29-30-2026)

---

## 1. Why Fine-Tune?

Pretrained large language models (LLMs) are general-purpose: they learn broad linguistic patterns from trillions of tokens of web text. However, they rarely perform optimally out of the box for specific downstream tasks — medical Q&A, code generation for a proprietary API, legal document summarization, etc.

**Fine-tuning** adapts a pretrained model to your task with far less data and compute than training from scratch.

### Full Fine-Tuning vs Parameter-Efficient Fine-Tuning

| Aspect | Full Fine-Tuning | Parameter-Efficient (PEFT) |
|--------|-----------------|---------------------------|
| Parameters updated | All (billions) | Small subset (millions) |
| Memory | Very high (full optimizer state) | Low (only adapter state) |
| Training speed | Slow | Fast |
| Risk of forgetting | Higher | Lower |
| Multiple tasks | One model per task | One base + multiple adapters |
| GPU requirement | Multi-GPU for 7B+ | Single GPU for 7B (QLoRA) |

Full fine-tuning updates every parameter in the model. For a 7B parameter model with AdamW, that means storing 7B weights + 7B gradients + 14B optimizer states (momentum + variance) = **~42B float parameters** in memory.

PEFT methods freeze the pretrained weights and only train a small number of additional parameters. The dominant PEFT method today is **LoRA**.

---

## 2. LoRA (Low-Rank Adaptation)

### The Core Idea

Instead of updating a weight matrix `W ∈ R^(d×k)` directly, LoRA learns a low-rank update:

```
W' = W + B @ A
```

where:
- `B ∈ R^(d×r)` — down-projection
- `A ∈ R^(r×k)` — up-projection
- `r << min(d, k)` — the rank (typically 8 or 16)

The original weight `W` is frozen. Only `A` and `B` are trained.

### Parameter Savings

For a weight matrix of shape `(d, k)`:
- Full fine-tuning: `d × k` trainable parameters
- LoRA: `r × (d + k)` trainable parameters

For a typical attention projection with `d = k = 4096` and `r = 16`:
- Full: 16,777,216 parameters
- LoRA: 131,072 parameters → **128× reduction**

### Initialization

- `A` is initialized from `N(0, 1/r)` so the initial magnitude is controlled
- `B` is initialized to zeros so that `B @ A = 0` at the start — the model begins as the pretrained model

### Scaling Factor

A scaling factor `alpha / r` is applied to the LoRA output:

```
output = W @ x + (B @ A @ x) * (alpha / r)
```

Typical `alpha` equals `r` (so scaling = 1) or `2 * r`.

### Implementation

```python
class LoRALinear(nn.Module):
    def __init__(self, base_linear, rank=8, alpha=16):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        d_out, d_in = base_linear.weight.shape
        self.lora_A = nn.Parameter(torch.randn(rank, d_in) / rank)
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        self.scaling = alpha / rank

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out
```

See `lora_adapter.py` for the complete implementation.

---

## 3. QLoRA

QLoRA combines LoRA with **weight quantization** to reduce memory even further:

1. **Quantize** the base model weights to INT4 or NF4 (4-bit NormalFloat)
2. **Add LoRA adapters** in full precision (BF16)
3. **Train** only the adapters

### Memory Comparison (7B Model)

| Method | Base Weights | Adapters | Optimizer | Total |
|--------|-------------|----------|-----------|-------|
| Full FT (FP32) | 28 GB | — | 56 GB | ~84 GB |
| Full FT (BF16) | 14 GB | — | 28 GB | ~42 GB |
| LoRA (BF16) | 14 GB | ~50 MB | ~100 MB | ~14.2 GB |
| QLoRA (NF4) | 3.5 GB | ~50 MB | ~100 MB | ~3.7 GB |

QLoRA makes fine-tuning a 7B model possible on a single 8GB consumer GPU.

### Pattern

```python
# Pseudocode for QLoRA
model = load_pretrained("llama-7b")
model = quantize_to_4bit(model)       # Base weights → NF4
model = apply_lora(model, rank=16)     # Adapters in BF16
train(model)                           # Only adapter gradients computed
```

The key insight: quantization errors in the base weights are compensated by the LoRA adapters during training. The adapters learn to correct for quantization noise.

---

## 4. Applying LoRA to a Transformer

### Which Layers to Adapt

Not all layers benefit equally from LoRA. Common targets:

| Layer | Benefit | Typically Adapted |
|-------|---------|-------------------|
| Q projection | High | Yes |
| K projection | Medium | Yes |
| V projection | High | Yes |
| O projection | Medium | Sometimes |
| FFN up-projection | Medium | Yes |
| FFN down-projection | Medium | Yes |
| Embeddings | Low | No |
| LayerNorm/RMSNorm | Low | No |

### Replacing Linear Layers

```python
def apply_lora_to_model(model, rank=8, alpha=16, target_modules=None):
    """Replace target nn.Linear layers with LoRALinear."""
    if target_modules is None:
        target_modules = {"q_proj", "k_proj", "v_proj", "ffn"}

    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear) and child_name in target_modules:
                lora_layer = LoRALinear(child, rank=rank, alpha=alpha)
                setattr(module, child_name, lora_layer)
```

### Merging Adapters Back

After training, fold the LoRA weights back into the base weights for inference with zero overhead:

```python
def merge_lora(model):
    for module in model.modules():
        if isinstance(module, LoRALinear):
            # W_merged = W + B @ A * scaling
            module.base.weight.data += (
                module.lora_B @ module.lora_A * module.scaling
            )
```

After merging, the model is identical to a regular model — no extra inference cost.

---

## 5. Data Preparation

### Instruction Tuning Format

The standard format for instruction fine-tuning:

```json
{
    "instruction": "Summarize the following text in one sentence.",
    "input": "PyTorch is an open-source machine learning framework...",
    "output": "PyTorch is an open-source ML framework for deep learning research and production."
}
```

### Tokenization

```python
def format_example(example):
    prompt = f"### Instruction:\n{example['instruction']}\n"
    if example.get("input"):
        prompt += f"### Input:\n{example['input']}\n"
    prompt += f"### Response:\n{example['output']}"
    return prompt

def tokenize_and_pad(text, tokenizer, max_length=512):
    tokens = tokenizer.encode(text)
    tokens = tokens[:max_length]
    padding = max_length - len(tokens)
    input_ids = tokens + [tokenizer.pad_id] * padding
    labels = tokens + [-100] * padding  # -100 = ignore in loss
    return input_ids, labels
```

Setting `labels=-100` for padding tokens tells `CrossEntropyLoss` to ignore those positions (via its `ignore_index` parameter).

### Train/Validation Split

```python
from torch.utils.data import random_split

dataset = InstructionDataset(data)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
```

---

## 6. Training Loop with All Best Practices

This is where the capstone brings together techniques from across the guide:

### Mixed Precision (Module 29)

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler("cuda")
with autocast("cuda", dtype=torch.bfloat16):
    loss = model(input_ids, labels=labels)
```

BF16 is preferred for LLMs because it has the same exponent range as FP32 (no overflow risk), and modern GPUs (A100, H100) have native BF16 tensor cores.

### Gradient Accumulation (Module 07)

Simulate larger batch sizes without more memory:

```python
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(**batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Activation Checkpointing (Module 16)

Trade compute for memory — recompute activations during backward instead of storing them:

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedTransformerBlock(nn.Module):
    def forward(self, x):
        return checkpoint(self._forward_impl, x, use_reentrant=False)
```

### torch.compile (Module 08)

Compile the model for kernel fusion and optimization:

```python
model = torch.compile(model)
```

For LoRA fine-tuning, `torch.compile` fuses the base linear + LoRA computation, giving 10-30% speedup.

### Gradient Clipping

Prevent exploding gradients, especially important for LLMs:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Learning Rate Schedule

Cosine warmup is standard for LLM fine-tuning:

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=2e-4, total_steps=total_steps,
    pct_start=0.03, anneal_strategy="cos"
)
```

### Checkpointing

Save only the LoRA parameters (much smaller):

```python
def save_lora_checkpoint(model, path):
    lora_state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            lora_state[name] = param.data
    torch.save(lora_state, path)
```

See `finetuning_pipeline.py` for the complete training loop.

---

## 7. Scaling with FSDP2

For models too large for a single GPU, use FSDP2 (`fully_shard`) from Module 10:

```python
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
)

# Shard each transformer block
for block in model.blocks:
    fully_shard(block, mp_policy=mp_policy)
fully_shard(model, mp_policy=mp_policy)
```

### Distributed Checkpointing

```python
from torch.distributed.checkpoint import save, load
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict, get_optimizer_state_dict
)

# Save
model_state = get_model_state_dict(model)
optim_state = get_optimizer_state_dict(model, optimizer)
save({"model": model_state, "optim": optim_state}, checkpoint_dir)

# Load
load({"model": model_state, "optim": optim_state}, checkpoint_dir)
```

### FSDP2 + LoRA

When combining FSDP2 with LoRA, only the LoRA parameters participate in gradient all-reduce. The frozen base weights are still sharded across GPUs for memory efficiency, but they don't accumulate gradients:

```python
# Apply LoRA first, then FSDP
model = create_model()
apply_lora_to_model(model, rank=16)

# FSDP shards everything, but only LoRA params have requires_grad=True
for block in model.blocks:
    fully_shard(block, mp_policy=mp_policy)
fully_shard(model, mp_policy=mp_policy)
```

---

## 8. Evaluation

### Perplexity

Perplexity measures how well the model predicts the next token. Lower is better:

```
PPL = exp(average cross-entropy loss)
```

```python
@torch.no_grad()
def compute_perplexity(model, dataloader):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in dataloader:
        logits = model(batch["input_ids"])
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch["labels"][:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += (shift_labels != -100).sum().item()
    return math.exp(total_loss / total_tokens)
```

### Generation with KV Cache (Module 22)

```python
@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens=100, temperature=0.8,
             top_k=50, top_p=0.9):
    model.eval()
    generated = list(prompt_ids)
    kv_cache = None

    for _ in range(max_new_tokens):
        input_ids = torch.tensor([generated[-1:]]) if kv_cache else torch.tensor([generated])
        logits, kv_cache = model(input_ids, kv_cache=kv_cache)
        next_logits = logits[0, -1, :] / temperature

        # Top-k filtering
        if top_k > 0:
            topk_vals, _ = torch.topk(next_logits, top_k)
            next_logits[next_logits < topk_vals[-1]] = float("-inf")

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cumulative_probs > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            next_logits[sorted_indices[remove]] = float("-inf")

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_token)

    return generated
```

### Sampling Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| Greedy (`temperature=0`) | Always pick highest probability | Factual Q&A |
| Temperature | Scale logits before softmax | Control randomness |
| Top-k | Keep only k highest-probability tokens | Moderate diversity |
| Top-p (nucleus) | Keep smallest set with cumulative prob >= p | Dynamic vocabulary |

---

## 9. Merging and Exporting

### Step 1: Merge LoRA Weights

```python
def merge_and_unload(model):
    """Merge LoRA weights into base model and remove adapters."""
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.base.weight.data += (
                module.lora_B @ module.lora_A * module.scaling
            )
            # Replace LoRALinear with the merged base Linear
            parent = get_parent_module(model, name)
            setattr(parent, name.split(".")[-1], module.base)
    return model
```

### Step 2: Export with torch.export

```python
merged_model = merge_and_unload(model)
merged_model.eval()

example_input = torch.randint(0, vocab_size, (1, 128))
exported = torch.export.export(merged_model, (example_input,))
torch.export.save(exported, "finetuned_model.pt2")
```

### Step 3: Size Comparison

```python
# Base model (BF16): ~14 GB for 7B
# LoRA checkpoint: ~50 MB (rank=16, all attention + FFN)
# Merged model (BF16): ~14 GB (same as base, but specialized)
```

The LoRA checkpoint is 280× smaller than the full model — you can store hundreds of task-specific adapters alongside one base model.

See `evaluation_and_export.py` for the complete workflow.

---

## 10. Complete Workflow Summary

```
┌─────────────────┐
│ Pretrained Model │
│   (frozen W)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Add LoRA       │  B ∈ R^(d×r), A ∈ R^(r×k)
│   Adapters       │  Only A, B are trainable
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Prepare Data     │  Instruction format
│ (tokenize, pad)  │  Labels with -100 masking
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│             Training Loop                │
│  ┌─────────┐ ┌──────────┐ ┌───────────┐ │
│  │  BF16   │ │ Grad     │ │ Activation│ │
│  │ autocast│ │ accum    │ │ ckpt      │ │
│  └─────────┘ └──────────┘ └───────────┘ │
│  ┌─────────┐ ┌──────────┐ ┌───────────┐ │
│  │ compile │ │ grad     │ │ cosine LR │ │
│  │         │ │ clip     │ │ warmup    │ │
│  └─────────┘ └──────────┘ └───────────┘ │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│    Evaluate      │  Perplexity
│    Generate      │  Temperature, top-k, top-p
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Merge LoRA      │  W' = W + B @ A * scaling
│  (zero overhead) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Export         │  torch.export → .pt2
│    Deploy         │  AOTInductor / NativeRT
└─────────────────┘
```

---

## 11. Hyperparameter Guide

Recommended settings by model size:

| Parameter | 1B | 7B | 13B | 70B |
|-----------|-----|-----|------|------|
| LoRA rank (r) | 8 | 16 | 16 | 32 |
| LoRA alpha | 16 | 32 | 32 | 64 |
| LoRA target layers | QKV + FFN | QKV + FFN | QKV + FFN | QKV + FFN |
| Learning rate | 3e-4 | 2e-4 | 1e-4 | 5e-5 |
| Batch size (effective) | 32 | 64 | 128 | 128 |
| Grad accumulation steps | 4 | 8 | 16 | 16 |
| Max sequence length | 512 | 1024 | 2048 | 2048 |
| Warmup ratio | 0.03 | 0.03 | 0.03 | 0.03 |
| Epochs | 3 | 3 | 2 | 1 |
| Weight decay | 0.01 | 0.01 | 0.01 | 0.01 |
| Gradient clip | 1.0 | 1.0 | 1.0 | 1.0 |
| Precision | BF16 | BF16 | BF16 | BF16 |
| Method | LoRA | LoRA/QLoRA | QLoRA | QLoRA |
| GPUs needed | 1 | 1 (QLoRA) / 2 (LoRA) | 2-4 | 4-8 |
| Memory per GPU | ~6 GB | ~8 GB (QLoRA) | ~20 GB | ~40 GB |

### Tips

- **Start with rank 8**, increase if the model underfits
- **Lower learning rates** for larger models — they're more sensitive
- **Cosine schedule** with 3% warmup works well across scales
- **Gradient clipping at 1.0** is nearly universal for LLMs
- **BF16** over FP16 — no loss scaling needed, same exponent range as FP32
- **Evaluate every 100-500 steps** on validation set for early stopping

---

## 12. Upstream Updates (June 29-30, 2026)

Recent PyTorch changes relevant to LLM fine-tuning:

### FlexAttention Blocksparse Fix (#188484)

Fixed a bug where blocksparse attention masks could produce incorrect results with certain block sizes. If you use FlexAttention with custom masks for fine-tuning, update to the latest nightly.

### cuDNN Heuristic Fast Path (#187212)

New fast path for cuDNN convolution heuristics that reduces kernel selection overhead. While primarily a CNN optimization, this also benefits models that combine attention with convolutional layers (e.g., vision-language models).

### CUPTI Monitor Tests (#186812)

Added comprehensive tests for CUPTI-based monitoring, improving reliability of profiling during training. Use `torch.profiler` with the CUPTI backend for accurate kernel-level profiling during fine-tuning.

### Dynamo Literal Types (#188486)

Improved handling of literal types in TorchDynamo. This fixes graph breaks that could occur when using constant values in model definitions — relevant when `torch.compile` encounters LoRA scaling factors or rank constants.

### CUBLASLt Tunable GEMM Headers

Updated headers for CUBLASLt tunable GEMMs, enabling better autotuning of matrix multiplication kernels. The LoRA forward pass `x @ A^T @ B^T` benefits from tuned GEMM kernels, especially for the non-standard shapes that LoRA introduces (tall-skinny matrices with rank << hidden_dim).

---

## Files in This Module

| File | Description | Lines |
|------|-------------|-------|
| `README.md` | This guide — complete theory and workflow | 450+ |
| `lora_adapter.py` | LoRA implementation, apply/merge, QLoRA concept | 250+ |
| `finetuning_pipeline.py` | Mini-LLM + LoRA + full training loop | 300+ |
| `evaluation_and_export.py` | Perplexity, generation, merge, export | 200+ |

---

## Key Takeaways

This capstone module demonstrates how the entire PyTorch ecosystem comes together for a real-world task:

1. **nn.Module** (Module 04) — the foundation for model definition and LoRA layers
2. **Training loops** (Module 07) — gradient accumulation, checkpointing, scheduling
3. **torch.compile** (Module 08) — automatic kernel fusion for faster training
4. **Attention** (Module 09) — FlexAttention, SDPA for efficient self-attention
5. **Activation checkpointing** (Module 16) — trade compute for memory in long sequences
6. **LLM building blocks** (Module 22) — RoPE, KV cache, RMSNorm, SwiGLU
7. **Mixed precision** (Module 29) — BF16 for 2× memory reduction and faster matmuls

Fine-tuning is not just about LoRA — it's about combining all these techniques into a cohesive, efficient pipeline.

---

### Further Resources

- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021) — original LoRA paper
- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023) — QLoRA paper
- [PyTorch FSDP2 Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) — official distributed training guide
- [Module 10 — Distributed Training](../10_distributed/) — DDP, FSDP2, tensor parallelism
- [Module 11 — Export & Deployment](../11_export_deploy/) — torch.export, AOTInductor

---

<div align="center">

[← Previous Module (Model Interpretability)](../33_interpretability/) | [🏠 Home](../README.md) | [Next Module (PyTorch Internals: The Dispatcher) →](../35_dispatcher/)

**Notebook**: [`34_llm_finetuning.ipynb`](../notebooks/34_llm_finetuning.ipynb)

</div>
