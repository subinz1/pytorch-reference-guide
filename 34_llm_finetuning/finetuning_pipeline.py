"""
Module 34: Fine-Tuning Pipeline
================================

Complete LLM fine-tuning pipeline with LoRA:
- Mini-LLM (RoPE, RMSNorm, SwiGLU from Module 22 patterns)
- LoRA applied to Q, K, V, and FFN layers
- Synthetic instruction-tuning dataset
- Training loop with BF16, gradient accumulation, gradient clipping,
  cosine warmup LR scheduling, and checkpoint saving

Runnable on CPU — no GPU required.

Usage:
    python finetuning_pipeline.py
"""

import math
import os
import tempfile
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# 1. Model Components (Module 22 patterns)
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    B, n_heads, T, head_dim = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(B, n_heads, T, head_dim // 2, 2))
    freqs = freqs[:T].unsqueeze(0).unsqueeze(0)
    x_rotated = torch.view_as_real(x_complex * freqs).reshape(B, n_heads, T, head_dim)
    return x_rotated.to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.ffn_up = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_gate = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn_down(F.silu(self.ffn_gate(x)) * self.ffn_up(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, rope_freqs: torch.Tensor):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        self.register_buffer("rope_freqs", rope_freqs, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, self.rope_freqs)
        k = apply_rope(k, self.rope_freqs)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.o_proj(attn)
        x = x + self.ffn(self.norm2(x))
        return x


class MiniLLM(nn.Module):
    """Small LLM following LLaMA patterns: RoPE, RMSNorm, SwiGLU, weight tying."""

    def __init__(self, vocab_size: int = 256, d_model: int = 192,
                 n_heads: int = 6, n_layers: int = 4, d_ff: int = 512,
                 max_seq_len: int = 256):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        rope_freqs = precompute_rope_freqs(d_model // n_heads, max_seq_len)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, rope_freqs)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight

    def forward(self, input_ids: torch.Tensor,
                labels: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.tok_emb(input_ids)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss


# =============================================================================
# 2. LoRA Implementation (from lora_adapter.py)
# =============================================================================

class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base_linear
        self.rank = rank
        self.scaling = alpha / rank
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        d_out, d_in = base_linear.weight.shape
        self.lora_A = nn.Parameter(torch.randn(rank, d_in) / rank)
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


def apply_lora(model: nn.Module, rank: int = 8, alpha: float = 16.0,
               target_modules: set[str] | None = None) -> nn.Module:
    if target_modules is None:
        target_modules = {"q_proj", "k_proj", "v_proj", "ffn_up", "ffn_down"}
    for _, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in target_modules:
                setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha))
    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# =============================================================================
# 3. Synthetic Instruction Dataset
# =============================================================================

INSTRUCTION_DATA = [
    {"instruction": "Add the numbers", "input": "3 + 5", "output": "8"},
    {"instruction": "Multiply the numbers", "input": "4 * 7", "output": "28"},
    {"instruction": "Subtract", "input": "10 - 3", "output": "7"},
    {"instruction": "What is the capital of France", "input": "", "output": "Paris"},
    {"instruction": "Reverse the word", "input": "hello", "output": "olleh"},
    {"instruction": "Count the letters", "input": "pytorch", "output": "7"},
    {"instruction": "Is this even", "input": "4", "output": "yes"},
    {"instruction": "Is this even", "input": "7", "output": "no"},
    {"instruction": "Upper case", "input": "hello", "output": "HELLO"},
    {"instruction": "Lower case", "input": "WORLD", "output": "world"},
    {"instruction": "First letter", "input": "tensor", "output": "t"},
    {"instruction": "Last letter", "input": "gradient", "output": "t"},
    {"instruction": "Add the numbers", "input": "1 + 1", "output": "2"},
    {"instruction": "Multiply the numbers", "input": "3 * 3", "output": "9"},
    {"instruction": "Double it", "input": "5", "output": "10"},
    {"instruction": "Half of", "input": "8", "output": "4"},
    {"instruction": "Repeat twice", "input": "ab", "output": "abab"},
    {"instruction": "Length of", "input": "torch", "output": "5"},
    {"instruction": "Sort letters", "input": "cba", "output": "abc"},
    {"instruction": "Reverse the word", "input": "world", "output": "dlrow"},
]


class CharTokenizer:
    """Simple character-level tokenizer for demonstration."""

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2

    def encode(self, text: str) -> list[int]:
        tokens = [self.bos_id]
        for ch in text:
            token_id = ord(ch) % (self.vocab_size - 3) + 3
            tokens.append(token_id)
        tokens.append(self.eos_id)
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        chars = []
        for t in token_ids:
            if t <= 2:
                continue
            chars.append(chr((t - 3) % 128 + 32))
        return "".join(chars)


def format_instruction(example: dict) -> str:
    text = f"### Instruction:\n{example['instruction']}\n"
    if example.get("input"):
        text += f"### Input:\n{example['input']}\n"
    text += f"### Response:\n{example['output']}"
    return text


class InstructionDataset(Dataset):
    def __init__(self, data: list[dict], tokenizer: CharTokenizer, max_length: int = 128):
        self.examples = []
        for example in data:
            text = format_instruction(example)
            tokens = tokenizer.encode(text)
            tokens = tokens[:max_length]
            padding_len = max_length - len(tokens)
            input_ids = tokens + [tokenizer.pad_id] * padding_len
            labels = tokens + [-100] * padding_len
            self.examples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.examples[idx]


# =============================================================================
# 4. Learning Rate Schedule: Cosine with Warmup
# =============================================================================

def cosine_warmup_lr(step: int, total_steps: int, max_lr: float,
                     min_lr: float = 1e-6, warmup_steps: int = 10) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# =============================================================================
# 5. Training Loop
# =============================================================================

def save_lora_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                         step: int, loss: float, path: str) -> None:
    lora_state = {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    checkpoint = {
        "step": step,
        "loss": loss,
        "lora_state_dict": lora_state,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    max_steps: int = 100,
    max_lr: float = 3e-4,
    warmup_steps: int = 10,
    grad_accum_steps: int = 2,
    max_grad_norm: float = 1.0,
    checkpoint_dir: str | None = None,
    checkpoint_every: int = 50,
    use_amp: bool = False,
    device: str = "cpu",
) -> list[float]:
    """Training loop with all best practices from the guide."""

    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=max_lr,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )

    amp_dtype = torch.bfloat16 if torch.cuda.is_available() or hasattr(torch.cpu, "is_available") else torch.float32
    if not use_amp:
        amp_dtype = torch.float32

    losses = []
    step = 0
    optimizer.zero_grad()

    print(f"\n{'='*60}")
    print(f" Training — {max_steps} steps, grad_accum={grad_accum_steps}")
    print(f" LR: cosine warmup to {max_lr}, warmup_steps={warmup_steps}")
    print(f" AMP: {'BF16' if use_amp else 'disabled'}")
    print(f"{'='*60}\n")

    start_time = time.time()
    data_iter = iter(train_loader)

    while step < max_steps:
        accum_loss = 0.0

        for micro_step in range(grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                _, loss = model(input_ids, labels=labels)
                loss = loss / grad_accum_steps

            loss.backward()
            accum_loss += loss.item()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=max_grad_norm,
        )

        # LR schedule
        lr = cosine_warmup_lr(step, max_steps, max_lr, warmup_steps=warmup_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()
        optimizer.zero_grad()

        losses.append(accum_loss)
        step += 1

        if step % 10 == 0 or step == 1:
            elapsed = time.time() - start_time
            print(f"  Step {step:>4d}/{max_steps} | Loss: {accum_loss:.4f} | "
                  f"LR: {lr:.2e} | Grad norm: {grad_norm:.2f} | "
                  f"Time: {elapsed:.1f}s")

        # Checkpoint
        if checkpoint_dir and step % checkpoint_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
            save_lora_checkpoint(model, optimizer, step, accum_loss, ckpt_path)
            print(f"  -> Saved checkpoint: {ckpt_path}")

    elapsed = time.time() - start_time
    print(f"\nTraining complete: {step} steps in {elapsed:.1f}s")
    print(f"Final loss: {losses[-1]:.4f}")

    # Final checkpoint
    if checkpoint_dir:
        final_path = os.path.join(checkpoint_dir, "checkpoint_final.pt")
        save_lora_checkpoint(model, optimizer, step, losses[-1], final_path)
        print(f"Saved final checkpoint: {final_path}")

    return losses


# =============================================================================
# 6. Main: End-to-End Pipeline
# =============================================================================

def main():
    print("=" * 60)
    print(" Module 34: Fine-Tuning Pipeline")
    print("=" * 60)

    torch.manual_seed(42)
    device = "cpu"

    # -- Create model --
    print("\n--- Creating Mini-LLM ---")
    model = MiniLLM(
        vocab_size=256, d_model=192, n_heads=6,
        n_layers=4, d_ff=512, max_seq_len=128,
    )

    trainable_before, total_before = count_parameters(model)
    print(f"  Parameters: {total_before:,} (all trainable)")

    # -- Apply LoRA --
    print("\n--- Applying LoRA (rank=8) ---")
    target = {"q_proj", "k_proj", "v_proj", "ffn_up", "ffn_down"}
    apply_lora(model, rank=8, alpha=16, target_modules=target)

    trainable_after, total_after = count_parameters(model)
    pct = 100.0 * trainable_after / total_after
    print(f"  Total parameters:     {total_after:,}")
    print(f"  Trainable (LoRA):     {trainable_after:,} ({pct:.2f}%)")
    print(f"  Frozen (base model):  {total_after - trainable_after:,}")

    # -- Prepare data --
    print("\n--- Preparing Data ---")
    tokenizer = CharTokenizer(vocab_size=256)

    # Repeat data to get more examples
    expanded_data = INSTRUCTION_DATA * 5
    train_data = expanded_data[:80]
    val_data = expanded_data[80:]

    train_dataset = InstructionDataset(train_data, tokenizer, max_length=128)
    val_dataset = InstructionDataset(val_data, tokenizer, max_length=128)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    print(f"  Train examples: {len(train_dataset)}")
    print(f"  Val examples:   {len(val_dataset)}")

    # Show a formatted example
    sample = format_instruction(INSTRUCTION_DATA[0])
    print(f"\n  Sample instruction:\n    {sample[:80]}...")

    # -- Train --
    with tempfile.TemporaryDirectory() as checkpoint_dir:
        losses = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            max_steps=60,
            max_lr=3e-4,
            warmup_steps=5,
            grad_accum_steps=2,
            max_grad_norm=1.0,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=30,
            use_amp=False,
            device=device,
        )

        # -- Show loss progression --
        print("\n--- Loss Progression ---")
        checkpoints = [0, len(losses) // 4, len(losses) // 2, 3 * len(losses) // 4, len(losses) - 1]
        for i in checkpoints:
            print(f"  Step {i+1:>3d}: {losses[i]:.4f}")

        if losses[-1] < losses[0]:
            print(f"\n  Loss decreased: {losses[0]:.4f} -> {losses[-1]:.4f} "
                  f"({(1 - losses[-1]/losses[0]) * 100:.1f}% reduction)")
        else:
            print(f"\n  Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

        # -- List checkpoints --
        ckpts = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
        print(f"\n  Checkpoints saved: {len(ckpts)}")
        for f in sorted(ckpts):
            size = os.path.getsize(os.path.join(checkpoint_dir, f))
            print(f"    {f}: {size / 1024:.1f} KB")

    print("\nDone!")


if __name__ == "__main__":
    main()
