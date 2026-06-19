"""
Multi-GPU Inference Patterns — Estimation, Metrics, and Strategy Selection
==========================================================================

Demonstrates (all runnable on CPU):
  1. Model size estimation: will it fit on N GPUs?
  2. Simple device_map sharding with meta tensors
  3. KV cache size estimation across GPUs
  4. Inference latency metrics (TTFT, ITL)
  5. torch.compile for inference (CPU demo)
  6. Quantized model size comparison (FP32 / FP16 / INT8 / INT4)
  7. Decision tree: recommend strategy based on model size and GPU count

Run:
    python inference_patterns.py
"""

import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────
# 1. Model size estimation
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    name: str
    vocab_size: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    intermediate_dim: int
    max_seq_len: int

KNOWN_MODELS = {
    "7B": ModelConfig("Llama-7B", 32000, 4096, 32, 32, 32, 11008, 4096),
    "13B": ModelConfig("Llama-13B", 32000, 5120, 40, 40, 40, 13824, 4096),
    "34B": ModelConfig("CodeLlama-34B", 32000, 8192, 48, 64, 8, 22016, 4096),
    "70B": ModelConfig("Llama-70B", 32000, 8192, 80, 64, 8, 28672, 4096),
    "405B": ModelConfig("Llama-405B", 128000, 16384, 126, 128, 8, 53248, 8192),
}

GPU_MEMORY = {
    "RTX-3090": 24, "RTX-4090": 24, "A10": 24, "L4": 24,
    "A100-40GB": 40, "A100-80GB": 80, "H100-80GB": 80, "H200-141GB": 141,
}


def estimate_model_params(cfg: ModelConfig) -> int:
    """Estimate total parameter count from architecture config."""
    embed = cfg.vocab_size * cfg.hidden_dim
    head_dim = cfg.hidden_dim // cfg.num_heads

    per_layer = (
        cfg.hidden_dim * cfg.num_heads * head_dim       # Q projection
        + cfg.hidden_dim * cfg.num_kv_heads * head_dim  # K projection
        + cfg.hidden_dim * cfg.num_kv_heads * head_dim  # V projection
        + cfg.num_heads * head_dim * cfg.hidden_dim      # output projection
        + cfg.hidden_dim * cfg.intermediate_dim          # FFN up
        + cfg.hidden_dim * cfg.intermediate_dim          # FFN gate
        + cfg.intermediate_dim * cfg.hidden_dim          # FFN down
        + 2 * cfg.hidden_dim                             # RMSNorm (x2)
    )
    total_layers = per_layer * cfg.num_layers
    output_head = cfg.vocab_size * cfg.hidden_dim
    final_norm = cfg.hidden_dim

    return embed + total_layers + output_head + final_norm


def model_memory_gb(num_params: int, dtype_bits: int = 16) -> float:
    """Model weight memory in GB for a given precision."""
    return num_params * (dtype_bits / 8) / 1e9


def will_it_fit(model_params: int, dtype_bits: int, gpu: str,
                kv_cache_gb: float = 0.0) -> dict:
    """Check if model + KV cache fits on a single GPU."""
    available = GPU_MEMORY.get(gpu, 80)
    usable = available * 0.90
    weight_gb = model_memory_gb(model_params, dtype_bits)
    total_gb = weight_gb + kv_cache_gb

    fits = total_gb <= usable
    gpus_needed = 1 if fits else math.ceil(total_gb / usable)

    return {
        "gpu": gpu, "gpu_memory_gb": available, "usable_gb": usable,
        "weight_gb": weight_gb, "kv_cache_gb": kv_cache_gb,
        "total_gb": total_gb, "fits": fits, "gpus_needed": gpus_needed,
    }


def demo_model_estimation():
    print("=" * 70)
    print("1. MODEL SIZE ESTIMATION")
    print("=" * 70)

    for name, cfg in KNOWN_MODELS.items():
        params = estimate_model_params(cfg)
        fp16 = model_memory_gb(params, 16)
        int8 = model_memory_gb(params, 8)
        int4 = model_memory_gb(params, 4)
        print(f"\n{cfg.name} ({name}):")
        print(f"  Parameters:  {params / 1e9:.1f}B")
        print(f"  FP16:        {fp16:.1f} GB")
        print(f"  INT8:        {int8:.1f} GB")
        print(f"  INT4:        {int4:.1f} GB")

    print("\n--- Will a 70B model fit? ---")
    cfg_70b = KNOWN_MODELS["70B"]
    params_70b = estimate_model_params(cfg_70b)
    for gpu in ["RTX-4090", "A100-40GB", "A100-80GB", "H200-141GB"]:
        for bits in [16, 8, 4]:
            result = will_it_fit(params_70b, bits, gpu)
            status = "YES" if result["fits"] else f"NO (need {result['gpus_needed']} GPUs)"
            dtype_name = {16: "FP16", 8: "INT8", 4: "INT4"}[bits]
            print(f"  {gpu:15s} {dtype_name}: {result['weight_gb']:.0f} GB → {status}")


# ─────────────────────────────────────────────────────────────────────
# 2. Simple device_map sharding with meta tensors
# ─────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * dim, dim, bias=False),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class SimpleLLM(nn.Module):
    def __init__(self, vocab=32000, dim=512, heads=8, layers=8):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))


def compute_device_map(model: nn.Module, num_gpus: int) -> dict:
    """Compute a balanced device map for a model across N GPUs."""
    components = []
    for name, module in model.named_children():
        param_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
        components.append((name, param_bytes))

    total = sum(size for _, size in components)
    per_gpu = total / num_gpus

    device_map = {}
    current_gpu, current_load = 0, 0
    for name, size in components:
        device_map[name] = f"cuda:{current_gpu}"
        current_load += size
        if current_load >= per_gpu and current_gpu < num_gpus - 1:
            current_gpu += 1
            current_load = 0

    return device_map


def demo_device_map():
    print("\n" + "=" * 70)
    print("2. DEVICE MAP SHARDING (meta tensors)")
    print("=" * 70)

    with torch.device("meta"):
        model = SimpleLLM(vocab=32000, dim=512, heads=8, layers=8)

    total_params = sum(p.numel() for p in model.parameters())
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"\nModel: {total_params:,} parameters ({total_bytes / 1e6:.1f} MB)")

    for num_gpus in [2, 4]:
        dmap = compute_device_map(model, num_gpus)
        print(f"\nDevice map ({num_gpus} GPUs):")
        for component, device in dmap.items():
            module = getattr(model, component)
            size = sum(p.numel() * p.element_size() for p in module.parameters())
            print(f"  {component:10s} → {device}  ({size / 1e6:.1f} MB)")


# ─────────────────────────────────────────────────────────────────────
# 3. KV cache size estimation
# ─────────────────────────────────────────────────────────────────────

def estimate_kv_cache(cfg: ModelConfig, batch_size: int, seq_len: int,
                      dtype_bytes: int = 2) -> dict:
    """Estimate KV cache memory for a model config."""
    head_dim = cfg.hidden_dim // cfg.num_heads
    per_token = 2 * cfg.num_layers * cfg.num_kv_heads * head_dim * dtype_bytes
    per_sequence = per_token * seq_len
    total = per_sequence * batch_size

    return {
        "per_token_bytes": per_token,
        "per_sequence_mb": per_sequence / 1e6,
        "total_gb": total / 1e9,
        "per_token_kb": per_token / 1024,
    }


def estimate_kv_cache_tp(cfg: ModelConfig, batch_size: int, seq_len: int,
                         tp_degree: int, dtype_bytes: int = 2) -> dict:
    """Estimate KV cache per GPU with Tensor Parallelism."""
    full = estimate_kv_cache(cfg, batch_size, seq_len, dtype_bytes)
    return {
        "total_gb": full["total_gb"],
        "per_gpu_gb": full["total_gb"] / tp_degree,
        "tp_degree": tp_degree,
    }


def demo_kv_cache():
    print("\n" + "=" * 70)
    print("3. KV CACHE ESTIMATION")
    print("=" * 70)

    for name in ["7B", "70B", "405B"]:
        cfg = KNOWN_MODELS[name]
        print(f"\n{cfg.name}:")

        kv = estimate_kv_cache(cfg, batch_size=1, seq_len=4096)
        print(f"  Per-token KV: {kv['per_token_kb']:.1f} KB")
        print(f"  Single seq (4096 tokens): {kv['per_sequence_mb']:.1f} MB")

        for batch in [1, 8, 32]:
            kv = estimate_kv_cache(cfg, batch_size=batch, seq_len=4096)
            print(f"  Batch={batch:2d}, 4096 tokens: {kv['total_gb']:.2f} GB")

        print(f"\n  With Tensor Parallelism:")
        for tp in [2, 4, 8]:
            kv_tp = estimate_kv_cache_tp(cfg, batch_size=32, seq_len=4096, tp_degree=tp)
            print(f"    TP={tp}: {kv_tp['per_gpu_gb']:.2f} GB/GPU (total {kv_tp['total_gb']:.2f} GB)")


# ─────────────────────────────────────────────────────────────────────
# 4. Inference latency metrics
# ─────────────────────────────────────────────────────────────────────

def demo_latency_metrics():
    print("\n" + "=" * 70)
    print("4. INFERENCE LATENCY METRICS (TTFT, ITL)")
    print("=" * 70)

    print("""
    TTFT (Time to First Token):
      Time from receiving a request to generating the first output token.
      Dominated by the prefill phase -- processing the entire prompt.
      TTFT = prefill_time(prompt_length)

    ITL (Inter-Token Latency):
      Time between consecutive generated tokens during decode.
      Dominated by memory bandwidth (reading all weights for 1 token).
      ITL = decode_time_per_token

    Throughput:
      Total tokens generated per second across all concurrent requests.
      throughput = total_generated_tokens / wall_clock_time
    """)

    model = SimpleLLM(vocab=32000, dim=256, heads=4, layers=4)
    model.eval()

    prompt = torch.randint(0, 32000, (1, 64))
    with torch.no_grad():
        start = time.perf_counter()
        logits = model(prompt)
        prefill_time = time.perf_counter() - start
    print(f"  Simulated TTFT (CPU, 64 tokens): {prefill_time * 1000:.1f} ms")

    single = torch.randint(0, 32000, (1, 1))
    decode_times = []
    with torch.no_grad():
        for _ in range(20):
            start = time.perf_counter()
            _ = model(single)
            decode_times.append(time.perf_counter() - start)

    avg_itl = sum(decode_times) / len(decode_times)
    print(f"  Simulated ITL (CPU, 1 token): {avg_itl * 1000:.2f} ms")
    print(f"  Max tokens/sec (single stream): {1.0 / avg_itl:.0f}")


# ─────────────────────────────────────────────────────────────────────
# 5. torch.compile for inference (CPU demo)
# ─────────────────────────────────────────────────────────────────────

def demo_compile_inference():
    print("\n" + "=" * 70)
    print("5. torch.compile FOR INFERENCE (CPU demo)")
    print("=" * 70)

    model = SimpleLLM(vocab=32000, dim=256, heads=4, layers=4)
    model.eval()
    input_ids = torch.randint(0, 32000, (1, 64))

    with torch.no_grad():
        times_eager = []
        for _ in range(10):
            start = time.perf_counter()
            _ = model(input_ids)
            times_eager.append(time.perf_counter() - start)

    eager_avg = sum(times_eager[2:]) / len(times_eager[2:])
    print(f"\n  Eager mode (avg of 8 runs):    {eager_avg * 1000:.2f} ms")

    try:
        compiled = torch.compile(model, mode="default")
        with torch.no_grad():
            _ = compiled(input_ids)
            _ = compiled(input_ids)

            times_compiled = []
            for _ in range(10):
                start = time.perf_counter()
                _ = compiled(input_ids)
                times_compiled.append(time.perf_counter() - start)

        compiled_avg = sum(times_compiled[2:]) / len(times_compiled[2:])
        print(f"  Compiled mode (avg of 8 runs): {compiled_avg * 1000:.2f} ms")
        speedup = eager_avg / compiled_avg if compiled_avg > 0 else float("inf")
        print(f"  Speedup: {speedup:.2f}x")
    except Exception as e:
        print(f"  torch.compile not available: {e}")

    print("\n  Compile modes for inference:")
    print("    mode='default'          - Good balance of compile time and perf")
    print("    mode='reduce-overhead'  - CUDA Graphs, lowest latency (GPU only)")
    print("    mode='max-autotune'     - Longest compile, best steady-state perf")


# ─────────────────────────────────────────────────────────────────────
# 6. Quantized model size comparison
# ─────────────────────────────────────────────────────────────────────

def demo_quantization_comparison():
    print("\n" + "=" * 70)
    print("6. QUANTIZED MODEL SIZE COMPARISON")
    print("=" * 70)

    precisions = [
        ("FP32", 32), ("FP16/BF16", 16), ("INT8", 8), ("INT4", 4),
    ]

    for name in ["7B", "13B", "70B", "405B"]:
        cfg = KNOWN_MODELS[name]
        params = estimate_model_params(cfg)
        print(f"\n{cfg.name} ({params / 1e9:.1f}B parameters):")
        print(f"  {'Precision':<12s} {'Size':>8s}  {'A100-80GB':>12s}  {'RTX-4090':>10s}")
        print(f"  {'-'*12} {'-'*8}  {'-'*12}  {'-'*10}")

        for prec_name, bits in precisions:
            size_gb = model_memory_gb(params, bits)
            fits_a100 = "fits" if size_gb < 72 else f"need {math.ceil(size_gb / 72)}"
            fits_4090 = "fits" if size_gb < 21.6 else f"need {math.ceil(size_gb / 21.6)}"
            print(f"  {prec_name:<12s} {size_gb:>6.1f} GB  {fits_a100:>12s}  {fits_4090:>10s}")


# ─────────────────────────────────────────────────────────────────────
# 7. Decision tree: recommend strategy
# ─────────────────────────────────────────────────────────────────────

def recommend_strategy(model_params_b: float, num_gpus: int,
                       gpu_type: str = "A100-80GB",
                       priority: str = "latency") -> dict:
    """Recommend an inference strategy based on model size and resources.

    Args:
        model_params_b: Model parameters in billions
        num_gpus: Available GPUs
        gpu_type: GPU type (for memory estimation)
        priority: "latency" or "throughput"

    Returns:
        Dict with strategy recommendation and reasoning.
    """
    gpu_mem = GPU_MEMORY.get(gpu_type, 80)
    usable = gpu_mem * 0.85

    fp16_gb = model_params_b * 2
    int8_gb = model_params_b * 1
    int4_gb = model_params_b * 0.5

    fits_fp16 = fp16_gb <= usable
    fits_int8 = int8_gb <= usable
    fits_int4 = int4_gb <= usable

    result = {
        "model_params_b": model_params_b,
        "fp16_gb": fp16_gb, "int8_gb": int8_gb, "int4_gb": int4_gb,
        "num_gpus": num_gpus, "gpu_type": gpu_type, "priority": priority,
    }

    if fits_fp16 and num_gpus == 1:
        result["strategy"] = "Single GPU + torch.compile"
        result["quantization"] = "None (FP16)"
        result["compile_mode"] = "reduce-overhead" if priority == "latency" else "max-autotune"
        result["reasoning"] = "Model fits on 1 GPU in FP16. Use torch.compile for best performance."

    elif fits_int4 and num_gpus == 1:
        result["strategy"] = "Single GPU + INT4 quantization"
        result["quantization"] = "INT4 weight-only"
        result["compile_mode"] = "max-autotune"
        result["reasoning"] = "Model too large for FP16 on 1 GPU but fits with INT4."

    elif num_gpus <= 4:
        tp_degree = min(num_gpus, 4)
        per_gpu = fp16_gb / tp_degree
        quant = "None (FP16)" if per_gpu <= usable else "INT8"
        result["strategy"] = f"Tensor Parallel (TP={tp_degree})"
        result["quantization"] = quant
        result["compile_mode"] = "reduce-overhead" if priority == "latency" else "max-autotune"
        result["reasoning"] = f"TP={tp_degree} splits weights across GPUs. Best for {priority}."

    elif num_gpus <= 8:
        tp = 4
        pp = num_gpus // tp
        result["strategy"] = f"Hybrid TP={tp} + PP={pp}"
        result["quantization"] = "INT8" if fp16_gb / num_gpus > usable else "None (FP16)"
        result["compile_mode"] = "max-autotune"
        result["reasoning"] = "TP within node (4 GPUs), PP across nodes for throughput."

    else:
        tp = 8
        pp = num_gpus // tp
        result["strategy"] = f"Hybrid TP={tp} + PP={pp} + continuous batching"
        result["quantization"] = "INT8" if fp16_gb / num_gpus > usable else "FP16"
        result["compile_mode"] = "max-autotune"
        result["reasoning"] = "Large-scale deployment: TP within node, PP across, continuous batching."

    return result


def demo_decision_tree():
    print("\n" + "=" * 70)
    print("7. DECISION TREE — STRATEGY RECOMMENDATIONS")
    print("=" * 70)

    scenarios = [
        (7,   1, "A100-80GB", "latency"),
        (7,   1, "RTX-4090",  "throughput"),
        (70,  2, "A100-80GB", "latency"),
        (70,  4, "H100-80GB", "latency"),
        (70,  8, "H100-80GB", "throughput"),
        (405, 16, "H100-80GB", "throughput"),
        (70,  1, "RTX-4090",  "latency"),
    ]

    for params_b, gpus, gpu_type, priority in scenarios:
        rec = recommend_strategy(params_b, gpus, gpu_type, priority)
        print(f"\n  {params_b}B model, {gpus}x {gpu_type}, priority={priority}:")
        print(f"    Strategy:     {rec['strategy']}")
        print(f"    Quantization: {rec['quantization']}")
        print(f"    Compile:      mode='{rec['compile_mode']}'")
        print(f"    Reasoning:    {rec['reasoning']}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo_model_estimation()
    demo_device_map()
    demo_kv_cache()
    demo_latency_metrics()
    demo_compile_inference()
    demo_quantization_comparison()
    demo_decision_tree()
    print("\n" + "=" * 70)
    print("All inference pattern demos complete!")
    print("=" * 70)
