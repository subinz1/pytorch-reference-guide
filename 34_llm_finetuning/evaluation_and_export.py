"""
Module 34: Evaluation and Export
=================================

Post-training evaluation and deployment pipeline:
- Compute perplexity on validation set
- Generate text with temperature, top-k, top-p sampling
- Merge LoRA adapters back into base model
- Verify merged model produces identical output
- Export merged model with torch.export
- Size comparison: base vs LoRA checkpoint vs merged

Runnable on CPU — no GPU required.

Usage:
    python evaluation_and_export.py
"""

import math
import os
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# 1. Model Definition (same as finetuning_pipeline.py)
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = RMSNorm(d_model)
        self.ffn_up = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.o_proj(attn)
        h = self.norm2(x)
        x = x + self.ffn_down(F.silu(self.ffn_up(h)))
        return x


class MiniLLM(nn.Module):
    def __init__(self, vocab_size: int = 256, d_model: int = 192,
                 n_heads: int = 6, n_layers: int = 4, d_ff: int = 512,
                 max_seq_len: int = 256):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))


# =============================================================================
# 2. LoRA (minimal reimplementation for self-contained demo)
# =============================================================================

class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base_linear
        self.rank = rank
        self.scaling = alpha / rank
        self.base.weight.requires_grad_(False)
        d_out, d_in = base_linear.weight.shape
        self.lora_A = nn.Parameter(torch.randn(rank, d_in) / rank)
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


def apply_lora(model: nn.Module, rank: int = 8, alpha: float = 16.0,
               targets: set[str] | None = None) -> nn.Module:
    if targets is None:
        targets = {"q_proj", "k_proj", "v_proj", "ffn_up", "ffn_down"}
    for _, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in targets:
                setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha))
    return model


def merge_lora(model: nn.Module) -> nn.Module:
    """Merge LoRA adapters into base weights and replace with plain Linear."""
    replacements: list[tuple[nn.Module, str, nn.Linear]] = []
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, LoRALinear):
                with torch.no_grad():
                    child.base.weight.data += child.lora_B @ child.lora_A * child.scaling
                replacements.append((module, child_name, child.base))
    for parent, child_name, merged_linear in replacements:
        merged_linear.weight.requires_grad_(True)
        setattr(parent, child_name, merged_linear)
    return model


# =============================================================================
# 3. Evaluation: Perplexity
# =============================================================================

@torch.no_grad()
def compute_perplexity(model: nn.Module, dataloader: DataLoader) -> float:
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
        num_tokens = (shift_labels != -100).sum().item()
        total_loss += loss.item()
        total_tokens += num_tokens
    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss)


# =============================================================================
# 4. Text Generation with Sampling
# =============================================================================

class CharTokenizer:
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2

    def encode(self, text: str) -> list[int]:
        tokens = [self.bos_id]
        for ch in text:
            tokens.append(ord(ch) % (self.vocab_size - 3) + 3)
        tokens.append(self.eos_id)
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        chars = []
        for t in token_ids:
            if t <= 2:
                continue
            chars.append(chr((t - 3) % 128 + 32))
        return "".join(chars)


@torch.no_grad()
def generate(model: nn.Module, prompt_ids: list[int], max_new_tokens: int = 50,
             temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9) -> list[int]:
    """Autoregressive generation with temperature, top-k, and top-p sampling."""
    model.eval()
    generated = list(prompt_ids)

    for _ in range(max_new_tokens):
        input_tensor = torch.tensor([generated], dtype=torch.long)
        logits = model(input_tensor)
        next_logits = logits[0, -1, :] / max(temperature, 1e-8)

        if top_k > 0:
            topk_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < topk_vals[-1]] = float("-inf")

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum_probs > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            next_logits[sorted_indices[remove]] = float("-inf")

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        if next_token == 2:
            break
        generated.append(next_token)

    return generated


# =============================================================================
# 5. Dummy Validation Dataset
# =============================================================================

class SimpleDataset(Dataset):
    def __init__(self, num_examples: int = 20, seq_len: int = 64, vocab_size: int = 256):
        self.examples = []
        for i in range(num_examples):
            torch.manual_seed(i + 1000)
            ids = torch.randint(3, vocab_size, (seq_len,))
            self.examples.append({"input_ids": ids, "labels": ids.clone()})

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.examples[idx]


# =============================================================================
# 6. Main: Evaluation and Export Pipeline
# =============================================================================

def main():
    print("=" * 60)
    print(" Module 34: Evaluation and Export")
    print("=" * 60)

    torch.manual_seed(42)
    vocab_size = 256
    tokenizer = CharTokenizer(vocab_size)

    # -- Create and "train" model (simulate by randomizing LoRA weights) --
    print("\n--- Setting Up Model with LoRA ---")
    model = MiniLLM(vocab_size=vocab_size, d_model=192, n_heads=6,
                    n_layers=4, d_ff=512, max_seq_len=128)
    apply_lora(model, rank=8, alpha=16)

    # Simulate training: give LoRA weights some non-zero values
    for name, param in model.named_parameters():
        if param.requires_grad and "lora_B" in name:
            nn.init.normal_(param, std=0.01)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total:,}")
    print(f"  LoRA params:  {trainable:,} ({100 * trainable / total:.2f}%)")

    # -- Compute perplexity --
    print("\n--- Computing Perplexity ---")
    val_dataset = SimpleDataset(num_examples=20, seq_len=64, vocab_size=vocab_size)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    ppl = compute_perplexity(model, val_loader)
    print(f"  Validation perplexity: {ppl:.2f}")
    print("  (High perplexity expected — model was not truly trained)")

    # -- Generate text --
    print("\n--- Text Generation ---")
    prompt = "### Instruction:\nSay hello\n### Response:\n"
    prompt_ids = tokenizer.encode(prompt)

    strategies = [
        ("Greedy (temp=0.1)", 0.1, 0, 1.0),
        ("Sampling (temp=0.8, top-k=50)", 0.8, 50, 1.0),
        ("Nucleus (temp=0.9, top-p=0.9)", 0.9, 0, 0.9),
        ("Creative (temp=1.2, top-k=100)", 1.2, 100, 1.0),
    ]

    for label, temp, k, p in strategies:
        output_ids = generate(model, prompt_ids, max_new_tokens=30,
                              temperature=temp, top_k=k, top_p=p)
        output_text = tokenizer.decode(output_ids)
        display = output_text[:60].replace("\n", " ")
        print(f"  {label}:")
        print(f"    -> {display}...")

    # -- Capture output before merge --
    print("\n--- Merging LoRA Adapters ---")
    test_input = torch.randint(0, vocab_size, (1, 32))
    with torch.no_grad():
        output_before = model(test_input)

    lora_count_before = sum(1 for m in model.modules() if isinstance(m, LoRALinear))
    print(f"  LoRA modules before merge: {lora_count_before}")

    # -- Merge --
    merge_lora(model)

    lora_count_after = sum(1 for m in model.modules() if isinstance(m, LoRALinear))
    print(f"  LoRA modules after merge:  {lora_count_after}")

    # -- Verify merge --
    with torch.no_grad():
        output_after = model(test_input)
    max_diff = (output_before - output_after).abs().max().item()
    print(f"  Max output difference: {max_diff:.2e}")
    assert max_diff < 1e-5, f"Merge changed output by {max_diff}"
    print("  Merge verified: outputs match!")

    # -- Perplexity after merge (should be identical) --
    ppl_merged = compute_perplexity(model, val_loader)
    print(f"  Perplexity after merge: {ppl_merged:.2f}")
    assert abs(ppl_merged - ppl) / ppl < 0.01, "Perplexity changed after merge!"

    # -- Export --
    print("\n--- Exporting Model ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save full model state dict
        model_path = os.path.join(tmpdir, "merged_model.pt")
        torch.save(model.state_dict(), model_path)
        model_size = os.path.getsize(model_path)
        print(f"  Merged model state dict: {model_size / 1024:.1f} KB")

        # Save LoRA-only checkpoint (simulate — just save trainable param count)
        lora_state = {}
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                lora_state[name] = param.data
        lora_path = os.path.join(tmpdir, "lora_adapter.pt")
        if lora_state:
            torch.save(lora_state, lora_path)
            lora_size = os.path.getsize(lora_path)
        else:
            lora_size = trainable * 4
            print("  (LoRA params already merged — estimating adapter size)")

        # Export with torch.export
        print("\n  Exporting with torch.export...")
        model.eval()
        example_input = torch.randint(0, vocab_size, (1, 32))
        try:
            exported = torch.export.export(model, (example_input,), strict=False)
            export_path = os.path.join(tmpdir, "model.pt2")
            torch.export.save(exported, export_path)
            export_size = os.path.getsize(export_path)
            print(f"  Exported model (.pt2): {export_size / 1024:.1f} KB")
        except Exception as e:
            export_size = model_size
            print(f"  torch.export skipped (expected on some configs): {e}")

        # -- Size comparison --
        print("\n--- Size Comparison ---")
        print(f"  {'Component':<30s} {'Size':>10s}")
        print(f"  {'-'*30} {'-'*10}")
        print(f"  {'Merged model (state dict)':<30s} {model_size / 1024:>8.1f} KB")
        print(f"  {'LoRA adapter (estimated)':<30s} {lora_size / 1024:>8.1f} KB")
        if model_size > 0 and lora_size > 0:
            ratio = model_size / lora_size
            print(f"  {'Ratio (full / LoRA)':<30s} {ratio:>8.1f}x")

        # Scaling projections
        print("\n--- Scaling Projections ---")
        scales = [
            ("1B model (BF16)", 2e9, 16),
            ("7B model (BF16)", 14e9, 16),
            ("13B model (BF16)", 26e9, 16),
            ("70B model (BF16)", 140e9, 32),
        ]
        print(f"  {'Model':<25s} {'Base Size':>12s} {'LoRA (est.)':>12s} {'Ratio':>8s}")
        print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*8}")
        for label, base_bytes, rank in scales:
            # Rough estimate: LoRA for attn+FFN ~ rank * 2 * d_model * n_layers * 5 * 2 bytes
            d_model_est = int((base_bytes / 2 / 40) ** 0.5)  # rough estimate
            n_layers_est = max(int(base_bytes / 2 / (d_model_est ** 2) / 10), 1)
            lora_est = rank * 2 * d_model_est * n_layers_est * 5 * 2
            print(f"  {label:<25s} {base_bytes / 1e9:>10.1f} GB "
                  f"{lora_est / 1e6:>10.1f} MB {base_bytes / max(lora_est, 1):>7.0f}x")

    print("\nDone!")


if __name__ == "__main__":
    main()
