"""
Case Study 5: Multi-GPU Communication Overlap

Problem: Naive all-reduce blocks compute — GPUs sit idle while gradients
are being synchronized across devices.
Fix: Gradient bucketing and overlapping communication with backward compute.
Expected speedup: 1.3–1.8× at 4+ GPU scale.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import os
from typing import Optional


# ============================================================
# BEFORE: Naive gradient sync (sequential all-reduce)
# ============================================================

def naive_all_reduce(model: nn.Module):
    """Manually all-reduce every parameter gradient sequentially.

    This is what happens without DDP's bucketing — each parameter's
    gradient is synced one at a time, blocking compute.
    """
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= dist.get_world_size()


def train_step_naive(model, data, target, criterion, optimizer):
    """Training step with sequential all-reduce after full backward."""
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    naive_all_reduce(model)
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()


# ============================================================
# AFTER: DDP with bucketed, overlapped communication
# ============================================================

def setup_ddp(model, bucket_cap_mb=25.0):
    """Wrap model in DDP with tuned bucket size.

    DDP buckets gradients and overlaps all-reduce with backward:
    - While layer N's gradient is being all-reduced,
      layer N-1's backward is running on the GPU.
    """
    ddp_model = DDP(
        model,
        bucket_cap_mb=bucket_cap_mb,
        gradient_as_bucket_view=True,
        static_graph=True,
    )
    return ddp_model


def train_step_ddp(model, data, target, criterion, optimizer):
    """DDP training step — communication automatically overlapped."""
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()


# ============================================================
# DDP Configuration Tuning
# ============================================================

class DDPConfig:
    """Configuration options for DDP performance tuning."""

    @staticmethod
    def small_model_config():
        """For models < 100M params — smaller buckets fill faster."""
        return {
            "bucket_cap_mb": 10.0,
            "gradient_as_bucket_view": True,
            "static_graph": True,
        }

    @staticmethod
    def large_model_config():
        """For models > 1B params — larger buckets reduce comm overhead."""
        return {
            "bucket_cap_mb": 50.0,
            "gradient_as_bucket_view": True,
            "static_graph": True,
        }

    @staticmethod
    def find_optimal_bucket_size(model, data_sample, criterion, target_sample, device):
        """Sweep bucket sizes to find optimal for this model."""
        param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        # Heuristic: bucket_cap = total_param_size / num_layers
        n_layers = sum(1 for _ in model.parameters())
        suggested = max(5.0, min(50.0, param_size_mb / max(1, n_layers // 4)))
        return suggested


# ============================================================
# Communication Profiling Utilities
# ============================================================

class CommProfiler:
    """Profile communication vs computation time in distributed training."""

    def __init__(self):
        self.compute_times: list[float] = []
        self.comm_times: list[float] = []
        self.total_times: list[float] = []

    def record_step(self, compute_ms: float, total_ms: float):
        comm_ms = total_ms - compute_ms
        self.compute_times.append(compute_ms)
        self.comm_times.append(max(0, comm_ms))
        self.total_times.append(total_ms)

    @property
    def comm_compute_ratio(self) -> float:
        if not self.compute_times:
            return 0.0
        avg_comm = sum(self.comm_times) / len(self.comm_times)
        avg_compute = sum(self.compute_times) / len(self.compute_times)
        return avg_comm / avg_compute if avg_compute > 0 else 0.0

    @property
    def overlap_efficiency(self) -> float:
        """1.0 = perfect overlap, 0.0 = zero overlap (fully sequential)."""
        if not self.total_times:
            return 0.0
        avg_total = sum(self.total_times) / len(self.total_times)
        avg_compute = sum(self.compute_times) / len(self.compute_times)
        avg_comm = sum(self.comm_times) / len(self.comm_times)
        sequential_time = avg_compute + avg_comm
        if sequential_time == 0:
            return 1.0
        overlap = 1.0 - (avg_total - avg_compute) / avg_comm if avg_comm > 0 else 1.0
        return max(0.0, min(1.0, overlap))

    def summary(self) -> dict:
        n = len(self.total_times)
        if n == 0:
            return {}
        return {
            "steps": n,
            "avg_compute_ms": sum(self.compute_times) / n,
            "avg_comm_ms": sum(self.comm_times) / n,
            "avg_total_ms": sum(self.total_times) / n,
            "comm_compute_ratio": self.comm_compute_ratio,
            "overlap_efficiency": self.overlap_efficiency,
        }


# ============================================================
# Gradient Compression (optional advanced technique)
# ============================================================

class GradientCompressor:
    """Compress gradients before all-reduce to reduce communication volume."""

    def __init__(self, compression_ratio: float = 0.1):
        self.compression_ratio = compression_ratio
        self.residuals: dict[str, torch.Tensor] = {}

    def compress(self, name: str, grad: torch.Tensor) -> torch.Tensor:
        """Top-K sparsification: only send the largest gradients."""
        if name in self.residuals:
            grad = grad + self.residuals[name]

        numel = grad.numel()
        k = max(1, int(numel * self.compression_ratio))
        flat = grad.flatten()

        _, indices = flat.abs().topk(k)
        mask = torch.zeros_like(flat, dtype=torch.bool)
        mask[indices] = True

        compressed = flat * mask.float()
        self.residuals[name] = (flat - compressed).view_as(grad)

        return compressed.view_as(grad)


# ============================================================
# SIMULATION (works without actual multi-GPU)
# ============================================================

def simulate_scaling():
    """Simulate communication overhead at different GPU counts."""
    print("--- Simulated Scaling Analysis ---\n")

    # Model: 100M params, FP32 → 400MB gradient per step
    param_size_mb = 400.0
    compute_ms = 50.0  # Forward + backward compute time

    # Network bandwidth assumptions
    intra_node_bandwidth_gbps = 600  # NVLink
    inter_node_bandwidth_gbps = 100  # InfiniBand

    print(f"Model gradient size: {param_size_mb:.0f} MB")
    print(f"Compute time (fwd+bwd): {compute_ms:.0f} ms")
    print()

    configs = [
        (1, "1 GPU", 0),
        (2, "2 GPU (1 node)", intra_node_bandwidth_gbps),
        (4, "4 GPU (1 node)", intra_node_bandwidth_gbps),
        (8, "8 GPU (1 node)", intra_node_bandwidth_gbps),
        (16, "16 GPU (2 nodes)", inter_node_bandwidth_gbps),
        (32, "32 GPU (4 nodes)", inter_node_bandwidth_gbps),
    ]

    print(f"{'Config':<20} | {'Comm (ms)':>10} | {'Total (ms)':>10} | {'Speedup':>8} | {'Efficiency':>10}")
    print("-" * 75)

    base_throughput = None
    for n_gpus, label, bandwidth_gbps in configs:
        if n_gpus == 1:
            comm_ms = 0
            total_ms = compute_ms
        else:
            # Ring all-reduce: 2*(N-1)/N * data_size / bandwidth
            ring_factor = 2 * (n_gpus - 1) / n_gpus
            comm_ms = ring_factor * param_size_mb * 8 / bandwidth_gbps  # MB→Mb conversion

            # With DDP overlap, comm is partially hidden behind compute
            overlap_fraction = min(0.85, compute_ms / (compute_ms + comm_ms))
            effective_comm_ms = comm_ms * (1 - overlap_fraction)
            total_ms = compute_ms + effective_comm_ms

        throughput = n_gpus / total_ms
        if base_throughput is None:
            base_throughput = throughput
        speedup = throughput / base_throughput
        efficiency = speedup / n_gpus * 100

        print(f"{label:<20} | {comm_ms if n_gpus > 1 else 0:>8.1f}ms | {total_ms:>8.1f}ms | {speedup:>6.2f}× | {efficiency:>8.1f}%")

    print()


def main():
    print("--- Case 5: Multi-GPU Communication Overlap ---\n")

    simulate_scaling()

    print("--- DDP Best Practices ---\n")
    print("  1. Use gradient_as_bucket_view=True (avoids copy)")
    print("  2. Set static_graph=True if model graph doesn't change")
    print("  3. Tune bucket_cap_mb:")
    print("     - Small models (<100M): 10-15 MB")
    print("     - Large models (>1B):   40-50 MB")
    print("  4. Overlap comm/compute happens automatically with DDP")
    print("  5. For very large models, consider FSDP over DDP")

    print("\n--- Communication Patterns ---\n")
    print("  Naive (no DDP):  backward → all_reduce → optimizer step")
    print("  DDP (bucketed):  backward ←overlap→ all_reduce → optimizer step")
    print("  FSDP (sharded):  gather → fwd → scatter → bwd → reduce-scatter")

    print("\n--- Key Takeaways ---")
    print("  • DDP's bucketed all-reduce overlaps 60-85% of comm with compute")
    print("  • At 8+ GPUs, communication becomes >30% of step time without overlap")
    print("  • NVLink (600 Gbps) vs InfiniBand (100 Gbps) matters a lot at scale")
    print("  • Gradient compression can help on bandwidth-limited interconnects")


if __name__ == "__main__":
    main()
