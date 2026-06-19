"""
Memory Profiling Tools — Estimation, Monitoring, and Analysis
=============================================================

Demonstrates:
  1. Meta-device memory estimation (no GPU needed)
  2. Parameter counting and memory breakdown by model size
  3. Optimizer state memory estimation (SGD vs Adam vs AdamW)
  4. Activation memory estimation formulas
  5. "Will it fit?" GPU calculator
  6. GPU-specific tools (if available): memory_allocated, memory_reserved,
     memory_summary, max_memory_allocated, memory_stats, empty_cache

Run:
    python memory_tools.py
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────
# 1. Parameter counting with meta device
# ─────────────────────────────────────────────────────────────────────

class TransformerConfig:
    def __init__(self, vocab_size=32000, hidden_dim=4096, num_layers=32,
                 num_heads=32, intermediate_dim=11008, max_seq_len=2048):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.max_seq_len = max_seq_len


class SimpleLLMBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        d = config.hidden_dim
        self.attn_qkv = nn.Linear(d, 3 * d, bias=False)
        self.attn_out = nn.Linear(d, d, bias=False)
        self.ffn_up = nn.Linear(d, config.intermediate_dim, bias=False)
        self.ffn_gate = nn.Linear(d, config.intermediate_dim, bias=False)
        self.ffn_down = nn.Linear(config.intermediate_dim, d, bias=False)
        self.norm1 = nn.RMSNorm(d)
        self.norm2 = nn.RMSNorm(d)

    def forward(self, x):
        h = self.norm1(x)
        qkv = self.attn_qkv(h)
        h = self.attn_out(qkv)
        x = x + h
        h = self.norm2(x)
        gate = torch.sigmoid(self.ffn_gate(h))
        x = x + self.ffn_down(gate * self.ffn_up(h))
        return x


class SimpleLLM(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.layers = nn.ModuleList([SimpleLLMBlock(config) for _ in range(config.num_layers)])
        self.norm = nn.RMSNorm(config.hidden_dim)
        self.head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))


def count_parameters_meta(config: TransformerConfig):
    """Build model on meta device to count parameters without allocating memory."""
    with torch.device("meta"):
        model = SimpleLLM(config)

    total_params = 0
    total_bytes_fp32 = 0
    total_bytes_bf16 = 0
    layer_breakdown = {}

    for name, p in model.named_parameters():
        numel = p.numel()
        total_params += numel
        total_bytes_fp32 += numel * 4
        total_bytes_bf16 += numel * 2
        top_level = name.split(".")[0]
        layer_breakdown[top_level] = layer_breakdown.get(top_level, 0) + numel

    return {
        "total_params": total_params,
        "total_gb_fp32": total_bytes_fp32 / 1e9,
        "total_gb_bf16": total_bytes_bf16 / 1e9,
        "breakdown": layer_breakdown,
    }


def demo_meta_device_estimation():
    print("=" * 70)
    print("META DEVICE PARAMETER ESTIMATION (no GPU needed)")
    print("=" * 70)

    configs = {
        "~1B (like TinyLlama)": TransformerConfig(hidden_dim=2048, num_layers=22, intermediate_dim=5632),
        "~7B (like Llama-2-7B)": TransformerConfig(hidden_dim=4096, num_layers=32, intermediate_dim=11008),
        "~13B (like Llama-2-13B)": TransformerConfig(hidden_dim=5120, num_layers=40, intermediate_dim=13824),
        "~70B (like Llama-2-70B)": TransformerConfig(hidden_dim=8192, num_layers=80, intermediate_dim=28672, num_heads=64),
    }

    for label, config in configs.items():
        info = count_parameters_meta(config)
        print(f"\n{label}:")
        print(f"  Parameters:  {info['total_params']:>15,}")
        print(f"  fp32 size:   {info['total_gb_fp32']:>12.2f} GB")
        print(f"  bf16 size:   {info['total_gb_bf16']:>12.2f} GB")
        print(f"  Breakdown:")
        for component, count in info["breakdown"].items():
            pct = 100.0 * count / info["total_params"]
            print(f"    {component:20s}: {count:>12,} ({pct:5.1f}%)")


# ─────────────────────────────────────────────────────────────────────
# 2. Optimizer state memory estimation
# ─────────────────────────────────────────────────────────────────────

@dataclass
class OptimizerMemoryEstimate:
    name: str
    state_bytes: int
    master_weights_bytes: int
    total_bytes: int

    @property
    def total_gb(self):
        return self.total_bytes / 1e9


def estimate_optimizer_memory(num_params: int, optimizer: str = "adam",
                               param_dtype_bytes: int = 2) -> OptimizerMemoryEstimate:
    """Estimate optimizer state memory for a given parameter count."""
    needs_master = param_dtype_bytes < 4

    if optimizer == "sgd":
        state_bytes = 0
        master_bytes = num_params * 4 if needs_master else 0
    elif optimizer == "sgd_momentum":
        state_bytes = num_params * 4  # momentum buffer (fp32)
        master_bytes = num_params * 4 if needs_master else 0
    elif optimizer in ("adam", "adamw"):
        state_bytes = num_params * 4 * 2  # m + v in fp32
        master_bytes = num_params * 4 if needs_master else 0
    elif optimizer == "adam_8bit":
        state_bytes = num_params * 1 * 2  # m + v in int8
        master_bytes = num_params * 4 if needs_master else 0
    elif optimizer == "adafactor":
        hidden = int(math.sqrt(num_params))
        state_bytes = (hidden + hidden) * 4  # row + col factors (rough)
        master_bytes = num_params * 4 if needs_master else 0
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    return OptimizerMemoryEstimate(
        name=optimizer,
        state_bytes=state_bytes,
        master_weights_bytes=master_bytes,
        total_bytes=state_bytes + master_bytes,
    )


def demo_optimizer_memory():
    print("\n" + "=" * 70)
    print("OPTIMIZER STATE MEMORY ESTIMATION")
    print("=" * 70)

    model_sizes = {"1B": 1_000_000_000, "7B": 7_000_000_000, "13B": 13_000_000_000}
    optimizers = ["sgd", "sgd_momentum", "adam", "adamw", "adam_8bit", "adafactor"]

    for model_label, num_params in model_sizes.items():
        print(f"\n{model_label} parameters (bf16 training):")
        print(f"  {'Optimizer':<15s} {'State':>10s} {'Master Wt':>10s} {'Total':>10s}")
        print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10}")
        for opt in optimizers:
            est = estimate_optimizer_memory(num_params, opt, param_dtype_bytes=2)
            print(f"  {est.name:<15s} {est.state_bytes/1e9:>9.1f}G {est.master_weights_bytes/1e9:>9.1f}G {est.total_gb:>9.1f}G")


# ─────────────────────────────────────────────────────────────────────
# 3. Activation memory estimation
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ActivationEstimate:
    per_layer_bytes: int
    attention_bytes: int
    total_bytes: int
    num_layers_stored: int

    @property
    def total_gb(self):
        return self.total_bytes / 1e9


def estimate_activation_memory(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    dtype_bytes: int = 2,
    flash_attention: bool = True,
    gradient_checkpointing: bool = False,
) -> ActivationEstimate:
    """Estimate activation memory for a Transformer model.

    This is a rough estimate. Real activation memory depends on the exact
    operations saved for backward by autograd.
    """
    per_layer = 2 * batch_size * seq_len * hidden_dim * dtype_bytes

    if flash_attention:
        attn = batch_size * num_heads * seq_len * 64 * dtype_bytes
    else:
        attn = batch_size * num_heads * seq_len * seq_len * dtype_bytes

    if gradient_checkpointing:
        layers_stored = max(1, int(math.sqrt(num_layers)))
    else:
        layers_stored = num_layers

    total = layers_stored * (per_layer + attn)
    return ActivationEstimate(per_layer, attn, total, layers_stored)


def demo_activation_memory():
    print("\n" + "=" * 70)
    print("ACTIVATION MEMORY ESTIMATION")
    print("=" * 70)

    configs = [
        ("7B, bs=1, seq=2048", dict(batch_size=1, seq_len=2048, hidden_dim=4096, num_layers=32, num_heads=32)),
        ("7B, bs=4, seq=2048", dict(batch_size=4, seq_len=2048, hidden_dim=4096, num_layers=32, num_heads=32)),
        ("7B, bs=1, seq=8192", dict(batch_size=1, seq_len=8192, hidden_dim=4096, num_layers=32, num_heads=32)),
        ("13B, bs=4, seq=2048", dict(batch_size=4, seq_len=2048, hidden_dim=5120, num_layers=40, num_heads=40)),
    ]

    for label, kwargs in configs:
        print(f"\n{label}:")
        for flash in [True, False]:
            for ckpt in [False, True]:
                est = estimate_activation_memory(**kwargs, flash_attention=flash, gradient_checkpointing=ckpt)
                tag = f"{'flash' if flash else 'standard':>8s} attn, {'checkpointed' if ckpt else 'full':>12s}"
                print(f"  {tag}: {est.total_gb:>8.2f} GB ({est.num_layers_stored} layers stored)")


# ─────────────────────────────────────────────────────────────────────
# 4. "Will it fit?" calculator
# ─────────────────────────────────────────────────────────────────────

GPU_SPECS = {
    "RTX-3090": 24, "RTX-4090": 24, "A10G": 24,
    "A100-40GB": 40, "A100-80GB": 80,
    "H100-80GB": 80, "H200-141GB": 141,
}


@dataclass
class MemoryBudget:
    parameters_gb: float
    gradients_gb: float
    optimizer_gb: float
    activations_gb: float
    cuda_context_gb: float = 0.5

    @property
    def total_gb(self):
        return self.parameters_gb + self.gradients_gb + self.optimizer_gb + self.activations_gb + self.cuda_context_gb


def estimate_full_budget(
    num_params: int,
    dtype_bytes: int = 2,
    optimizer: str = "adam",
    batch_size: int = 1,
    seq_len: int = 2048,
    hidden_dim: int = 4096,
    num_layers: int = 32,
    num_heads: int = 32,
    flash_attention: bool = True,
    gradient_checkpointing: bool = False,
) -> MemoryBudget:
    param_gb = num_params * dtype_bytes / 1e9
    grad_gb = param_gb
    opt_est = estimate_optimizer_memory(num_params, optimizer, dtype_bytes)
    act_est = estimate_activation_memory(
        batch_size, seq_len, hidden_dim, num_layers, num_heads,
        dtype_bytes, flash_attention, gradient_checkpointing,
    )
    return MemoryBudget(param_gb, grad_gb, opt_est.total_gb, act_est.total_gb)


def will_it_fit(budget: MemoryBudget, gpu: str = "A100-80GB") -> dict:
    available = GPU_SPECS.get(gpu, 80)
    usable = available * 0.90
    fits = budget.total_gb <= usable
    return {
        "gpu": gpu,
        "gpu_memory_gb": available,
        "usable_gb": usable,
        "estimated_gb": budget.total_gb,
        "fits": fits,
        "headroom_gb": max(0, usable - budget.total_gb),
        "gpus_needed": max(1, math.ceil(budget.total_gb / usable)),
    }


def demo_will_it_fit():
    print("\n" + "=" * 70)
    print("'WILL IT FIT?' GPU CALCULATOR")
    print("=" * 70)

    scenarios = [
        ("1B bf16 Adam bs=8", dict(num_params=1_000_000_000, dtype_bytes=2, optimizer="adam",
                                    batch_size=8, seq_len=2048, hidden_dim=2048, num_layers=22, num_heads=32)),
        ("7B bf16 Adam bs=4", dict(num_params=7_000_000_000, dtype_bytes=2, optimizer="adam",
                                    batch_size=4, seq_len=2048, hidden_dim=4096, num_layers=32, num_heads=32)),
        ("7B bf16 SGD bs=4", dict(num_params=7_000_000_000, dtype_bytes=2, optimizer="sgd_momentum",
                                   batch_size=4, seq_len=2048, hidden_dim=4096, num_layers=32, num_heads=32)),
        ("13B bf16 Adam bs=4", dict(num_params=13_000_000_000, dtype_bytes=2, optimizer="adam",
                                     batch_size=4, seq_len=2048, hidden_dim=5120, num_layers=40, num_heads=40)),
        ("13B bf16 Adam bs=4 + ckpt", dict(num_params=13_000_000_000, dtype_bytes=2, optimizer="adam",
                                            batch_size=4, seq_len=2048, hidden_dim=5120, num_layers=40, num_heads=40,
                                            gradient_checkpointing=True)),
    ]

    for label, kwargs in scenarios:
        budget = estimate_full_budget(**kwargs)
        print(f"\n{label}:")
        print(f"  Parameters:  {budget.parameters_gb:>8.2f} GB")
        print(f"  Gradients:   {budget.gradients_gb:>8.2f} GB")
        print(f"  Optimizer:   {budget.optimizer_gb:>8.2f} GB")
        print(f"  Activations: {budget.activations_gb:>8.2f} GB")
        print(f"  CUDA ctx:    {budget.cuda_context_gb:>8.2f} GB")
        print(f"  ─────────────────────────")
        print(f"  TOTAL:       {budget.total_gb:>8.2f} GB")

        for gpu in ["RTX-4090", "A100-40GB", "A100-80GB", "H100-80GB", "H200-141GB"]:
            result = will_it_fit(budget, gpu)
            status = "YES" if result["fits"] else f"NO (need {result['gpus_needed']} GPUs)"
            headroom = f"+{result['headroom_gb']:.1f}GB free" if result["fits"] else ""
            print(f"    {gpu:>15s} ({result['gpu_memory_gb']}GB): {status:>20s}  {headroom}")


# ─────────────────────────────────────────────────────────────────────
# 5. GPU-specific memory monitoring tools
# ─────────────────────────────────────────────────────────────────────

def demo_gpu_memory_tools():
    if not torch.cuda.is_available():
        print("\n" + "=" * 70)
        print("GPU MEMORY TOOLS — Skipped (no CUDA device available)")
        print("=" * 70)
        print("The following tools require a CUDA GPU:")
        print("  - torch.cuda.memory_allocated()")
        print("  - torch.cuda.memory_reserved()")
        print("  - torch.cuda.memory_summary()")
        print("  - torch.cuda.max_memory_allocated()")
        print("  - torch.cuda.memory_stats()")
        print("  - torch.cuda.empty_cache()")
        return

    print("\n" + "=" * 70)
    print("GPU MEMORY MONITORING TOOLS")
    print("=" * 70)
    device = torch.device("cuda")

    def mb(b): return b / 1024**2

    # --- memory_allocated / memory_reserved ---
    print("\n--- memory_allocated / memory_reserved ---")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    baseline_alloc = torch.cuda.memory_allocated()
    baseline_res = torch.cuda.memory_reserved()
    print(f"Baseline: allocated={mb(baseline_alloc):.1f} MB, reserved={mb(baseline_res):.1f} MB")

    x = torch.randn(5000, 5000, device=device)
    after_alloc = torch.cuda.memory_allocated()
    after_res = torch.cuda.memory_reserved()
    print(f"After 5000×5000 tensor: allocated={mb(after_alloc):.1f} MB, reserved={mb(after_res):.1f} MB")
    print(f"  Tensor size: {x.numel() * x.element_size() / 1024**2:.1f} MB")

    del x
    after_del_alloc = torch.cuda.memory_allocated()
    after_del_res = torch.cuda.memory_reserved()
    print(f"After del: allocated={mb(after_del_alloc):.1f} MB, reserved={mb(after_del_res):.1f} MB")
    print(f"  Gap (cached): {mb(after_del_res - after_del_alloc):.1f} MB")

    torch.cuda.empty_cache()
    after_cache_res = torch.cuda.memory_reserved()
    print(f"After empty_cache: reserved={mb(after_cache_res):.1f} MB")

    # --- memory_summary ---
    print("\n--- memory_summary (abbreviated) ---")
    x = torch.randn(2000, 2000, device=device)
    y = torch.randn(2000, 2000, device=device)
    z = x @ y
    print(torch.cuda.memory_summary(abbreviated=True))
    del x, y, z

    # --- peak tracking ---
    print("--- Peak memory tracking ---")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    a = torch.randn(3000, 3000, device=device)
    b = torch.randn(3000, 3000, device=device)
    c = a @ b
    peak_after_matmul = torch.cuda.max_memory_allocated()
    print(f"Peak after matmul: {mb(peak_after_matmul):.1f} MB")

    del a, b, c
    current = torch.cuda.memory_allocated()
    print(f"Current after del: {mb(current):.1f} MB")
    print(f"Peak still recorded: {mb(torch.cuda.max_memory_allocated()):.1f} MB")

    torch.cuda.reset_peak_memory_stats()
    print(f"After reset: peak={mb(torch.cuda.max_memory_allocated()):.1f} MB")

    # --- memory_stats ---
    print("\n--- memory_stats (key fields) ---")
    x = torch.randn(1000, 1000, device=device)
    stats = torch.cuda.memory_stats()
    key_fields = [
        "allocated_bytes.all.current",
        "allocated_bytes.all.peak",
        "reserved_bytes.all.current",
        "reserved_bytes.all.peak",
        "active.all.current",
        "active.all.peak",
        "num_alloc_retries",
        "num_ooms",
    ]
    for field in key_fields:
        val = stats.get(field, "N/A")
        if isinstance(val, int) and val > 1_000_000:
            print(f"  {field:<42s}: {mb(val):>10.1f} MB")
        else:
            print(f"  {field:<42s}: {val:>10}")
    del x

    torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║       Memory Profiling Tools — Estimation & Monitoring             ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    demo_meta_device_estimation()
    demo_optimizer_memory()
    demo_activation_memory()
    demo_will_it_fit()
    demo_gpu_memory_tools()

    print("\n" + "=" * 70)
    print("Done! All memory estimation tools demonstrated.")
    print("=" * 70)
