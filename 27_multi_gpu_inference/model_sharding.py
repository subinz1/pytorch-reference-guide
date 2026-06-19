"""
Model Sharding & Advanced Inference Patterns
=============================================

Demonstrates:
  1. Manual model sharding across devices (meta device for estimation)
  2. Tensor Parallel API patterns (code structure)
  3. Pipeline inference pattern
  4. Continuous batching concept implementation (simplified)
  5. AOTInductor export workflow explanation

Run:
    python model_sharding.py
"""

import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────
# 1. Manual model sharding across devices (meta device)
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


class ShardedLLM(nn.Module):
    """Model that can be sharded across multiple devices."""

    def __init__(self, vocab=32000, dim=512, heads=8, layers=8):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self._shard_boundary = len(self.layers) // 2

    def get_shard_info(self, num_devices: int) -> dict:
        """Report how this model would be sharded across N devices."""
        layers_per_device = len(self.layers) // num_devices
        shards = {}

        embed_size = sum(p.numel() * p.element_size() for p in self.embed.parameters())
        head_size = sum(p.numel() * p.element_size() for p in self.head.parameters())
        norm_size = sum(p.numel() * p.element_size() for p in self.norm.parameters())

        for dev_idx in range(num_devices):
            start = dev_idx * layers_per_device
            end = start + layers_per_device if dev_idx < num_devices - 1 else len(self.layers)
            layer_size = sum(
                p.numel() * p.element_size()
                for i in range(start, end)
                for p in self.layers[i].parameters()
            )
            shard_size = layer_size
            components = [f"layers[{start}:{end}]"]

            if dev_idx == 0:
                shard_size += embed_size
                components.insert(0, "embed")
            if dev_idx == num_devices - 1:
                shard_size += head_size + norm_size
                components.extend(["norm", "head"])

            shards[f"device:{dev_idx}"] = {
                "components": components,
                "size_mb": shard_size / 1e6,
                "num_layers": end - start,
            }
        return shards

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))


def demo_manual_sharding():
    print("=" * 70)
    print("1. MANUAL MODEL SHARDING (meta device)")
    print("=" * 70)

    with torch.device("meta"):
        model = ShardedLLM(vocab=32000, dim=1024, heads=16, layers=16)

    total = sum(p.numel() for p in model.parameters())
    total_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    print(f"\nModel: {total:,} parameters ({total_mb:.1f} MB in FP32)")

    for num_devs in [2, 4]:
        shards = model.get_shard_info(num_devs)
        print(f"\nSharding across {num_devs} devices:")
        for device, info in shards.items():
            print(f"  {device}: {info['size_mb']:.1f} MB "
                  f"({info['num_layers']} layers) — {', '.join(info['components'])}")

        sizes = [info["size_mb"] for info in shards.values()]
        imbalance = max(sizes) / min(sizes)
        print(f"  Memory imbalance ratio: {imbalance:.2f}x")


# ─────────────────────────────────────────────────────────────────────
# 2. Tensor Parallel API patterns
# ─────────────────────────────────────────────────────────────────────

def demo_tensor_parallel_patterns():
    print("\n" + "=" * 70)
    print("2. TENSOR PARALLEL API PATTERNS")
    print("=" * 70)

    print("""
    Tensor Parallel splits weight matrices across GPUs. The key PyTorch APIs:

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
        parallelize_module,
    )

    # 1. Create device mesh
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("tp",))

    # 2. Define parallelization plan
    #    ColwiseParallel: split output features across GPUs
    #    RowwiseParallel: split input features, all-reduce output
    plan = {
        "attn.qkv_proj":  ColwiseParallel(),   # each GPU gets subset of heads
        "attn.out_proj":  RowwiseParallel(),    # all-reduce after
        "ffn.up_proj":    ColwiseParallel(),    # each GPU gets subset of FFN
        "ffn.gate_proj":  ColwiseParallel(),    # parallel gating
        "ffn.down_proj":  RowwiseParallel(),    # all-reduce after
    }

    # 3. Apply to each layer
    for layer in model.layers:
        parallelize_module(layer, mesh["tp"], plan)
    """)

    print("  Communication per Transformer layer with TP:")
    print("    - 1 all-reduce after attention output projection")
    print("    - 1 all-reduce after FFN down projection")
    print("    - Total: 2 all-reduce ops per layer")

    dim = 4096
    seq_len = 2048
    dtype_bytes = 2
    comm_per_layer = 2 * seq_len * dim * dtype_bytes
    for num_layers in [32, 80]:
        total_comm = comm_per_layer * num_layers
        print(f"\n  {num_layers} layers, dim={dim}, seq={seq_len}, bf16:")
        print(f"    Per-layer comm: {comm_per_layer / 1e6:.1f} MB")
        print(f"    Total per forward: {total_comm / 1e6:.1f} MB")
        for bw_name, bw_gbps in [("PCIe 5.0", 64), ("NVLink H100", 900)]:
            time_ms = total_comm / (bw_gbps * 1e9) * 1000
            print(f"    Time on {bw_name}: {time_ms:.2f} ms")


# ─────────────────────────────────────────────────────────────────────
# 3. Pipeline inference pattern
# ─────────────────────────────────────────────────────────────────────

class PipelineStage:
    """Simulates one pipeline stage (runs on one device)."""

    def __init__(self, stage_id: int, num_layers: int, dim: int, heads: int):
        self.stage_id = stage_id
        self.layers = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def simulate_pipeline_bubble(pp_degree: int, num_microbatches: int) -> float:
    """Calculate pipeline bubble fraction."""
    bubble = (pp_degree - 1) / (pp_degree + num_microbatches - 1)
    return bubble


def demo_pipeline_inference():
    print("\n" + "=" * 70)
    print("3. PIPELINE INFERENCE PATTERN")
    print("=" * 70)

    print("\n  Pipeline parallel assigns different layers to different GPUs.")
    print("  Communication: point-to-point send/recv between stages (hidden states only).")

    print("\n  Pipeline bubble analysis:")
    print(f"  {'PP Degree':>10s}  {'Microbatches':>13s}  {'Bubble':>8s}  {'Utilization':>12s}")
    print(f"  {'-'*10}  {'-'*13}  {'-'*8}  {'-'*12}")

    for pp in [2, 4, 8]:
        for mb in [4, 8, 16, 32]:
            bubble = simulate_pipeline_bubble(pp, mb)
            util = 1 - bubble
            print(f"  {pp:>10d}  {mb:>13d}  {bubble:>7.1%}  {util:>11.1%}")

    print("\n  Cross-stage communication cost (hidden state transfer):")
    for dim, name in [(4096, "7B"), (8192, "70B"), (16384, "405B")]:
        for seq_len in [2048, 4096]:
            data_mb = seq_len * dim * 2 / 1e6
            print(f"    {name} (dim={dim}), seq={seq_len}: {data_mb:.1f} MB per boundary")


# ─────────────────────────────────────────────────────────────────────
# 4. Continuous batching concept
# ─────────────────────────────────────────────────────────────────────

@dataclass
class InferenceRequest:
    id: int
    prompt_len: int
    max_new_tokens: int
    tokens_generated: int = 0

    @property
    def is_finished(self) -> bool:
        return self.tokens_generated >= self.max_new_tokens


class ContinuousBatchScheduler:
    """Simplified continuous batching scheduler for demonstration."""

    def __init__(self, max_batch_size: int):
        self.max_batch_size = max_batch_size
        self.active: dict[int, InferenceRequest] = {}
        self.waiting: list[InferenceRequest] = []
        self.completed: list[InferenceRequest] = []
        self.step_count = 0

    def add_request(self, request: InferenceRequest):
        self.waiting.append(request)

    def step(self) -> dict:
        """Execute one decode step. Returns step statistics."""
        self.step_count += 1

        finished_ids = [rid for rid, req in self.active.items() if req.is_finished]
        for rid in finished_ids:
            self.completed.append(self.active.pop(rid))

        admitted = 0
        while len(self.active) < self.max_batch_size and self.waiting:
            req = self.waiting.pop(0)
            self.active[req.id] = req
            admitted += 1

        for req in self.active.values():
            if not req.is_finished:
                req.tokens_generated += 1

        return {
            "step": self.step_count,
            "active": len(self.active),
            "waiting": len(self.waiting),
            "completed": len(self.completed),
            "finished_this_step": len(finished_ids),
            "admitted_this_step": admitted,
        }


def demo_continuous_batching():
    print("\n" + "=" * 70)
    print("4. CONTINUOUS BATCHING SIMULATION")
    print("=" * 70)

    scheduler = ContinuousBatchScheduler(max_batch_size=4)

    requests = [
        InferenceRequest(0, prompt_len=100, max_new_tokens=10),
        InferenceRequest(1, prompt_len=50,  max_new_tokens=5),
        InferenceRequest(2, prompt_len=200, max_new_tokens=15),
        InferenceRequest(3, prompt_len=80,  max_new_tokens=8),
        InferenceRequest(4, prompt_len=120, max_new_tokens=12),
        InferenceRequest(5, prompt_len=60,  max_new_tokens=3),
        InferenceRequest(6, prompt_len=150, max_new_tokens=20),
        InferenceRequest(7, prompt_len=90,  max_new_tokens=7),
    ]

    for req in requests:
        scheduler.add_request(req)

    print(f"\n  {len(requests)} requests submitted, max batch size = {scheduler.max_batch_size}")
    print(f"\n  {'Step':>6s}  {'Active':>7s}  {'Waiting':>8s}  {'Done':>6s}  {'Finished':>9s}  {'Admitted':>9s}")
    print(f"  {'-'*6}  {'-'*7}  {'-'*8}  {'-'*6}  {'-'*9}  {'-'*9}")

    while scheduler.active or scheduler.waiting:
        stats = scheduler.step()
        print(f"  {stats['step']:>6d}  {stats['active']:>7d}  {stats['waiting']:>8d}  "
              f"{stats['completed']:>6d}  {stats['finished_this_step']:>9d}  "
              f"{stats['admitted_this_step']:>9d}")

    total_tokens = sum(r.max_new_tokens for r in requests)
    print(f"\n  Completed all {len(requests)} requests in {scheduler.step_count} steps")
    print(f"  Total tokens generated: {total_tokens}")
    print(f"  Avg GPU utilization: {total_tokens / (scheduler.step_count * scheduler.max_batch_size):.1%}")

    # Compare with static batching
    static_steps = sum(r.max_new_tokens for r in requests[:4]) + sum(r.max_new_tokens for r in requests[4:])
    static_max = max(r.max_new_tokens for r in requests[:4]) + max(r.max_new_tokens for r in requests[4:])
    print(f"\n  Static batching (2 batches of 4) would take {static_max} steps")
    print(f"  Continuous batching saved {static_max - scheduler.step_count} steps "
          f"({(static_max - scheduler.step_count) / static_max:.0%} improvement)")


# ─────────────────────────────────────────────────────────────────────
# 5. AOTInductor export workflow
# ─────────────────────────────────────────────────────────────────────

def demo_aotinductor_workflow():
    print("\n" + "=" * 70)
    print("5. AOTInductor EXPORT WORKFLOW")
    print("=" * 70)

    print("""
    AOTInductor compiles a PyTorch model into a shared library (.so) that can
    be loaded and run from C++ without Python.

    WORKFLOW:
    ┌──────────────┐     ┌───────────────┐     ┌──────────────┐
    │ PyTorch Model │ ──▶ │ torch.export  │ ──▶ │ AOT Compile  │
    │  (Python)     │     │ (ExportedProg)│     │   (.so file) │
    └──────────────┘     └───────────────┘     └──────────────┘
                                                      │
                                                      ▼
                                               ┌──────────────┐
                                               │ C++ Runtime  │
                                               │ (no Python)  │
                                               └──────────────┘
    """)

    print("  Step 1: Export the model")
    print("  ─────────────────────────")
    print("    import torch")
    print("    from torch._export import aot_compile")
    print()
    print("    model = MyModel().eval().cuda()")
    print("    example = torch.randint(0, vocab, (1, seq_len), device='cuda')")
    print("    so_path = aot_compile(model, args=(example,),")
    print("                          options={'max_autotune': True})")

    print()
    print("  Step 2: Load in C++")
    print("  ─────────────────────────")
    print("    #include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>")
    print()
    print("    auto runner = std::make_unique<AOTIModelContainerRunnerCuda>(so_path);")
    print("    auto outputs = runner->run({input_tensor});")

    print()
    print("  Step 3: Or use PT2 Archive (newer approach)")
    print("  ─────────────────────────")
    print("    from torch._inductor.package import package_aoti, load_package")
    print()
    print("    ep = torch.export.export(model, (example,))")
    print("    package_aoti('model.pt2', ep)")
    print("    runner = load_package('model.pt2')")
    print("    output = runner(input_tensor)")

    print()
    print("  Benefits:")
    print("    - No Python GIL: true parallel request handling")
    print("    - Instant startup: no compilation on first call")
    print("    - Smaller deployment: just .so + libtorch, no Python env")
    print("    - Reproducible: same binary across all servers")

    model = ShardedLLM(vocab=32000, dim=256, heads=4, layers=4)
    model.eval()
    example = torch.randint(0, 32000, (1, 64))

    print("\n  Demo: exporting a small model via torch.export...")
    try:
        ep = torch.export.export(model, (example,), strict=False)
        print(f"    Exported successfully!")
        print(f"    Graph nodes: {len(ep.graph.nodes)}")
        print(f"    Input specs: {len(ep.graph_signature.input_specs)}")
        print(f"    Output specs: {len(ep.graph_signature.output_specs)}")
    except Exception as e:
        print(f"    Export demo skipped: {e}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo_manual_sharding()
    demo_tensor_parallel_patterns()
    demo_pipeline_inference()
    demo_continuous_batching()
    demo_aotinductor_workflow()
    print("\n" + "=" * 70)
    print("All model sharding demos complete!")
    print("=" * 70)
