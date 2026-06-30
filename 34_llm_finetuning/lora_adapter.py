"""
Module 34: LoRA (Low-Rank Adaptation) Implementation
=====================================================

Complete LoRA implementation for fine-tuning LLMs:
- LoRALinear: wraps nn.Linear with low-rank A, B matrices
- apply_lora_to_model: replace target layers with LoRA variants
- merge_lora: fold adapters back into base weights
- QLoRA concept: simulated INT8 quantized base + FP32 adapters

Runnable on CPU — no GPU required.

Usage:
    python lora_adapter.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. LoRA Linear Layer
# =============================================================================

class LoRALinear(nn.Module):
    """Linear layer with Low-Rank Adaptation.

    Wraps an existing nn.Linear and adds trainable low-rank matrices A and B.
    The original weight is frozen; only A and B are trained.

    Forward: y = W @ x + (B @ A @ x) * scaling
    where W is frozen, B ∈ R^(d_out × r), A ∈ R^(r × d_in)
    """

    def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        d_out, d_in = base_linear.weight.shape
        self.lora_A = nn.Parameter(torch.randn(rank, d_in) / rank)
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out

    def extra_repr(self) -> str:
        d_out, d_in = self.base.weight.shape
        return f"in={d_in}, out={d_out}, rank={self.rank}, alpha={self.alpha}"


# =============================================================================
# 2. QLoRA Linear Layer (Simulated INT8 Quantization)
# =============================================================================

class QLoRALinear(nn.Module):
    """LoRA with quantized base weights (simulated INT8).

    In real QLoRA, base weights are NF4 (4-bit NormalFloat).
    Here we simulate with INT8 quantization for demonstration.
    The adapters remain in full precision.
    """

    def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        weight = base_linear.weight.data
        self.weight_scale = weight.abs().max() / 127.0
        quantized = torch.clamp(
            torch.round(weight / self.weight_scale), -128, 127
        ).to(torch.int8)
        self.register_buffer("quantized_weight", quantized)
        self.register_buffer("weight_scale_buf", self.weight_scale.unsqueeze(0))

        self.bias = base_linear.bias
        if self.bias is not None:
            self.bias.requires_grad_(False)

        d_out, d_in = base_linear.weight.shape
        self.lora_A = nn.Parameter(torch.randn(rank, d_in) / rank)
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dequantized = self.quantized_weight.float() * self.weight_scale_buf
        base_out = F.linear(x, dequantized, self.bias)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out

    def extra_repr(self) -> str:
        d_out, _ = self.quantized_weight.shape
        d_in = self.lora_A.shape[1]
        return f"in={d_in}, out={d_out}, rank={self.rank}, quantized=INT8"


# =============================================================================
# 3. Apply LoRA to a Model
# =============================================================================

def apply_lora_to_model(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: set[str] | None = None,
    use_qlora: bool = False,
) -> nn.Module:
    """Replace target nn.Linear layers with LoRALinear (or QLoRALinear).

    Args:
        model: the model to modify (in-place)
        rank: LoRA rank
        alpha: LoRA scaling factor
        target_modules: set of child module names to replace
        use_qlora: if True, use QLoRALinear with INT8 base weights
    """
    if target_modules is None:
        target_modules = {"q_proj", "k_proj", "v_proj", "ffn_up", "ffn_down"}

    lora_cls = QLoRALinear if use_qlora else LoRALinear

    for name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in target_modules:
                lora_layer = lora_cls(child, rank=rank, alpha=alpha)
                setattr(module, child_name, lora_layer)

    return model


# =============================================================================
# 4. Merge LoRA Weights Back
# =============================================================================

def merge_lora(model: nn.Module) -> nn.Module:
    """Merge LoRA adapters into base weights for zero-overhead inference."""
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                module.base.weight.data += (
                    module.lora_B @ module.lora_A * module.scaling
                )

    replacements: list[tuple[nn.Module, str, nn.Linear]] = []
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, LoRALinear):
                replacements.append((module, child_name, child.base))

    for parent, child_name, merged_linear in replacements:
        setattr(parent, child_name, merged_linear)

    return model


# =============================================================================
# 5. Parameter Counting
# =============================================================================

def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count trainable vs frozen parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + frozen
    return {"trainable": trainable, "frozen": frozen, "total": total}


def print_parameter_summary(model: nn.Module, label: str = "Model") -> None:
    counts = count_parameters(model)
    pct = 100.0 * counts["trainable"] / max(counts["total"], 1)
    print(f"\n{'='*60}")
    print(f" {label} Parameter Summary")
    print(f"{'='*60}")
    print(f"  Trainable:  {counts['trainable']:>12,}")
    print(f"  Frozen:     {counts['frozen']:>12,}")
    print(f"  Total:      {counts['total']:>12,}")
    print(f"  Trainable:  {pct:.2f}%")
    print(f"{'='*60}")


# =============================================================================
# 6. Demo: Small Transformer with LoRA
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
        self.norm1 = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
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


class MiniTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, max_seq_len: int = 512):
        super().__init__()
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
        x = self.norm(x)
        return self.lm_head(x)


# =============================================================================
# 7. Demo Script
# =============================================================================

def main():
    print("=" * 60)
    print(" Module 34: LoRA Adapter Demo")
    print("=" * 60)

    torch.manual_seed(42)

    # -- Create a small transformer --
    vocab_size, d_model, n_heads, n_layers, d_ff = 256, 256, 4, 4, 512
    model = MiniTransformer(vocab_size, d_model, n_heads, n_layers, d_ff)

    print_parameter_summary(model, "Before LoRA (all trainable)")

    # -- Apply LoRA --
    target = {"q_proj", "k_proj", "v_proj", "ffn_up", "ffn_down"}
    apply_lora_to_model(model, rank=8, alpha=16, target_modules=target)

    print_parameter_summary(model, "After LoRA (only adapters trainable)")

    # -- Verify forward pass works --
    x = torch.randint(0, vocab_size, (2, 32))
    logits = model(x)
    print(f"\nForward pass: input {x.shape} -> logits {logits.shape}")

    # -- Show LoRA layers --
    print("\nLoRA layers:")
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            print(f"  {name}: {module}")

    # -- Save only LoRA parameters --
    lora_state = {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    print(f"\nLoRA checkpoint: {len(lora_state)} tensors")
    lora_size = sum(t.numel() * t.element_size() for t in lora_state.values())
    print(f"LoRA checkpoint size: {lora_size / 1024:.1f} KB")

    # -- Capture output before merge --
    with torch.no_grad():
        out_before = model(x)

    # -- Merge LoRA into base weights --
    model = merge_lora(model)
    print_parameter_summary(model, "After Merge (LoRA folded into base)")

    # -- Verify output is unchanged --
    with torch.no_grad():
        out_after = model(x)
    diff = (out_before - out_after).abs().max().item()
    print(f"\nMax output difference after merge: {diff:.2e}")
    assert diff < 1e-5, "Merge changed the output!"
    print("Merge verified: outputs match!")

    # -- Verify no LoRA modules remain --
    lora_count = sum(1 for m in model.modules() if isinstance(m, LoRALinear))
    print(f"LoRA modules remaining: {lora_count}")
    assert lora_count == 0

    # -- QLoRA demo --
    print("\n" + "=" * 60)
    print(" QLoRA Demo (Simulated INT8)")
    print("=" * 60)

    model_q = MiniTransformer(vocab_size, d_model, n_heads, n_layers, d_ff)

    with torch.no_grad():
        base_out = model_q(x)

    apply_lora_to_model(model_q, rank=8, alpha=16, target_modules=target, use_qlora=True)
    print_parameter_summary(model_q, "QLoRA Model")

    with torch.no_grad():
        qlora_out = model_q(x)

    quant_diff = (base_out - qlora_out).abs().mean().item()
    print(f"\nMean output difference (quantization noise): {quant_diff:.4f}")
    print("(LoRA adapters will learn to compensate for this during training)")

    # -- Memory comparison --
    print("\n" + "=" * 60)
    print(" Memory Comparison")
    print("=" * 60)
    base_params = sum(p.numel() for p in model.parameters())
    base_memory_fp32 = base_params * 4
    base_memory_bf16 = base_params * 2
    lora_memory = lora_size
    print(f"  Base model (FP32): {base_memory_fp32 / 1024:.1f} KB")
    print(f"  Base model (BF16): {base_memory_bf16 / 1024:.1f} KB")
    print(f"  LoRA checkpoint:   {lora_memory / 1024:.1f} KB")
    ratio = base_memory_bf16 / max(lora_memory, 1)
    print(f"  Size ratio (BF16 / LoRA): {ratio:.0f}x smaller")

    print("\nDone!")


if __name__ == "__main__":
    main()
