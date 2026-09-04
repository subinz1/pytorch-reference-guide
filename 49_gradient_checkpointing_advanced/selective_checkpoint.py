"""
Module 49: Selective / Advanced Gradient Checkpointing
=======================================================
Compares vanilla checkpointing, reentrant vs non-reentrant modes,
a selective save/recompute policy, and a simple CPU offload pattern.

Run: python selective_checkpoint.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------
class MLPBlock(nn.Module):
    """Two-layer MLP used as a checkpointable unit."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


def peak_memory_mb() -> float:
    """Return peak CUDA memory in MiB, or -1.0 on CPU-only runs."""
    if not torch.cuda.is_available():
        return -1.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# PART 1: Vanilla checkpoint (non-reentrant)
# ---------------------------------------------------------------------------
def demo_vanilla_checkpoint(device: str = "cpu"):
    print("=" * 70)
    print("PART 1: Vanilla checkpoint (use_reentrant=False)")
    print("=" * 70)
    dim, layers, batch, seq = 256, 8, 4, 64
    blocks = nn.ModuleList([MLPBlock(dim, 4 * dim) for _ in range(layers)]).to(device)
    x = torch.randn(batch, seq, dim, device=device, requires_grad=True)

    def run(use_ckpt: bool):
        reset_peak()
        h = x
        for block in blocks:
            if use_ckpt:
                h = checkpoint(block, h, use_reentrant=False)
            else:
                h = block(h)
        loss = h.square().mean()
        loss.backward()
        return peak_memory_mb(), loss.item()

    mem_full, loss_full = run(False)
    # Clear grads between runs
    for p in blocks.parameters():
        p.grad = None
    x.grad = None
    mem_ckpt, loss_ckpt = run(True)

    print(f"loss full={loss_full:.6f}, ckpt={loss_ckpt:.6f}")
    if mem_full >= 0:
        print(f"peak mem full={mem_full:.1f} MiB, ckpt={mem_ckpt:.1f} MiB")
    else:
        print("CPU run — memory comparison skipped (enable CUDA to measure).")


# ---------------------------------------------------------------------------
# PART 2: Reentrant vs non-reentrant API surface
# ---------------------------------------------------------------------------
def demo_reentrant_flags():
    print("\n" + "=" * 70)
    print("PART 2: Reentrant vs non-reentrant")
    print("=" * 70)
    block = MLPBlock(64, 128)
    x = torch.randn(2, 16, 64, requires_grad=True)

    y1 = checkpoint(block, x, use_reentrant=False)
    y1.sum().backward()
    g1 = x.grad.clone()
    x.grad = None

    y2 = checkpoint(block, x, use_reentrant=True)
    y2.sum().backward()
    g2 = x.grad.clone()

    print(f"grad max|diff| reentrant vs non-reentrant: {(g1 - g2).abs().max().item():.2e}")
    print("Prefer use_reentrant=False unless you need legacy behavior.")


# ---------------------------------------------------------------------------
# PART 3: Selective policy — save matmuls, recompute pointwise
# ---------------------------------------------------------------------------
def demo_selective_policy():
    """
    Educational selective policy: decide per-op whether to save or recompute.
    Real SAC uses create_selective_checkpoint_contexts; here we show the idea
    with an explicit pack/unpack list for a tiny graph.
    """
    print("\n" + "=" * 70)
    print("PART 3: Selective save/recompute (conceptual)")
    print("=" * 70)

    SAVE_OPS = {"aten.mm.default", "aten.addmm.default", "aten.bmm.default"}

    def policy(op_name: str) -> str:
        # Save expensive matmuls; recompute cheap pointwise in backward
        return "save" if op_name in SAVE_OPS else "recompute"

    examples = [
        "aten.mm.default",
        "aten.gelu.default",
        "aten.add.Tensor",
        "aten.bmm.default",
    ]
    for name in examples:
        print(f"  {name:24s} -> {policy(name)}")

    # Tiny manual selective checkpoint: only wrap the matmul region
    w1 = torch.randn(32, 64, requires_grad=True)
    w2 = torch.randn(64, 32, requires_grad=True)
    x = torch.randn(8, 32, requires_grad=True)

    def expensive(h):
        # Checkpoint only the heavy matmul chain
        return (h @ w1) @ w2

    def full_forward(h):
        h = checkpoint(expensive, h, use_reentrant=False)
        return F.gelu(h)  # pointwise outside — activations kept or cheap

    y = full_forward(x)
    y.sum().backward()
    print(f"selective-style loss grad norms: x={x.grad.norm():.4f}, "
          f"w1={w1.grad.norm():.4f}, w2={w2.grad.norm():.4f}")


# ---------------------------------------------------------------------------
# PART 4: Activation offload sketch (CPU pinned)
# ---------------------------------------------------------------------------
class OffloadSaveContext:
    """
    Minimal educational stand-in for activation offload:
    move saved tensors to pinned CPU memory, restore on unpack.
    """

    def __init__(self):
        self.storage = []

    def pack(self, t: torch.Tensor):
        cpu = t.detach().to("cpu", non_blocking=True).pin_memory()
        self.storage.append((cpu, t.device))
        return len(self.storage) - 1

    def unpack(self, idx: int) -> torch.Tensor:
        cpu, device = self.storage[idx]
        return cpu.to(device, non_blocking=True)


def demo_offload_pack_unpack():
    print("\n" + "=" * 70)
    print("PART 4: Activation offload pack/unpack sketch")
    print("=" * 70)
    ctx = OffloadSaveContext()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    activation = torch.randn(64, 128, device=device)
    handle = ctx.pack(activation)
    restored = ctx.unpack(handle)
    print(f"device={device}, max|diff| after offload roundtrip: "
          f"{(activation.cpu() - restored.cpu()).abs().max().item():.2e}")
    print("Production: use torch.distributed / FSDP offload or custom autograd hooks.")


# ---------------------------------------------------------------------------
# PART 5: Stack of checkpointed blocks (realistic pattern)
# ---------------------------------------------------------------------------
def demo_stacked_blocks():
    print("\n" + "=" * 70)
    print("PART 5: Checkpoint every Transformer-style block")
    print("=" * 70)
    model = nn.Sequential(*[MLPBlock(128, 256) for _ in range(6)])
    x = torch.randn(2, 32, 128, requires_grad=True)

    h = x
    for layer in model:
        h = checkpoint(layer, h, use_reentrant=False)
    loss = h.mean()
    loss.backward()
    print(f"stacked checkpoint loss={loss.item():.6f}, grad_norm={x.grad.norm():.4f}")


if __name__ == "__main__":
    demo_vanilla_checkpoint("cpu")
    demo_reentrant_flags()
    demo_selective_policy()
    demo_offload_pack_unpack()
    demo_stacked_blocks()
    print("\nAll selective checkpoint demos finished.")
