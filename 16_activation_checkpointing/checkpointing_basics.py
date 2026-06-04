"""
Activation Checkpointing — Trading Memory for Compute
=======================================================
Demonstrates basic and selective activation checkpointing.
Runs on CPU to show the concepts (memory savings are most impactful on GPU).
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
import time

print("=" * 65)
print("1. THE MEMORY PROBLEM — Why Checkpointing Exists")
print("=" * 65)

class HeavyLayer(nn.Module):
    """A layer that produces large intermediate activations."""
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = torch.relu(self.fc1(x))  # 4x expansion — large activation
        x = self.fc2(x)
        return x + residual

# Stack many layers
class DeepModel(nn.Module):
    def __init__(self, dim=256, n_layers=12):
        super().__init__()
        self.layers = nn.ModuleList([HeavyLayer(dim) for _ in range(n_layers)])
        self.head = nn.Linear(dim, 10)

    def forward(self, x, use_checkpoint=False):
        for layer in self.layers:
            if use_checkpoint:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return self.head(x.mean(dim=1))

model = DeepModel(dim=256, n_layers=12)
params = sum(p.numel() for p in model.parameters())
print(f"Model: 12 layers, 256 dim, {params:,} parameters")

# Estimate activation memory
batch, seq, dim = 16, 64, 256
activation_per_layer = batch * seq * dim * 4 * 4  # 4x expansion, float32
total_activation = activation_per_layer * 12
print(f"\nActivation memory estimate (no checkpointing):")
print(f"  Per layer: {activation_per_layer / 1e6:.1f} MB")
print(f"  Total (12 layers): {total_activation / 1e6:.1f} MB")
print(f"\nWith checkpointing: ~{activation_per_layer * 2 / 1e6:.1f} MB "
      f"(only 2 layers worth)")

print("\n" + "=" * 65)
print("2. BASIC CHECKPOINTING")
print("=" * 65)

x = torch.randn(16, 64, 256, requires_grad=True)
target = torch.randint(0, 10, (16,))

# Without checkpointing
model_no_ckpt = DeepModel(256, 12)
model_no_ckpt.train()
output = model_no_ckpt(x, use_checkpoint=False)
loss = nn.functional.cross_entropy(output, target)
loss.backward()
print(f"Without checkpointing: loss = {loss.item():.4f}")

# With checkpointing
model_ckpt = DeepModel(256, 12)
model_ckpt.load_state_dict(model_no_ckpt.state_dict())
model_ckpt.train()

x2 = x.detach().clone().requires_grad_(True)
output_ckpt = model_ckpt(x2, use_checkpoint=True)
loss_ckpt = nn.functional.cross_entropy(output_ckpt, target)
loss_ckpt.backward()
print(f"With checkpointing:    loss = {loss_ckpt.item():.4f}")
print(f"Outputs match: {torch.allclose(output, output_ckpt, atol=1e-5)}")

# Gradients should match
grad_match = all(
    torch.allclose(p1.grad, p2.grad, atol=1e-5)
    for p1, p2 in zip(model_no_ckpt.parameters(), model_ckpt.parameters())
    if p1.grad is not None
)
print(f"Gradients match: {grad_match}")

print("\n" + "=" * 65)
print("3. CHECKPOINT_SEQUENTIAL — For nn.Sequential Models")
print("=" * 65)

seq_model = nn.Sequential(*[HeavyLayer(256) for _ in range(8)])

x = torch.randn(8, 32, 256, requires_grad=True)

# Split into 4 segments (each segment = 2 layers checkpointed together)
output = checkpoint_sequential(seq_model, segments=4, input=x, use_reentrant=False)
loss = output.sum()
loss.backward()

print(f"checkpoint_sequential with 4 segments:")
print(f"  Input: {x.shape}")
print(f"  Output: {output.shape}")
print(f"  8 layers split into 4 checkpointed segments")
print(f"  Memory: stores activations at 4 boundaries (not all 8)")

print("\n" + "=" * 65)
print("4. COMPUTE OVERHEAD MEASUREMENT")
print("=" * 65)

model = DeepModel(256, 12)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train_step(model, use_ckpt):
    x = torch.randn(16, 64, 256)
    target = torch.randint(0, 10, (16,))
    optimizer.zero_grad(set_to_none=True)
    output = model(x, use_checkpoint=use_ckpt)
    loss = nn.functional.cross_entropy(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()

# Warmup
for _ in range(3):
    train_step(model, False)
    train_step(model, True)

# Benchmark without checkpointing
N = 20
start = time.perf_counter()
for _ in range(N):
    train_step(model, False)
no_ckpt_time = (time.perf_counter() - start) / N * 1000

# Benchmark with checkpointing
start = time.perf_counter()
for _ in range(N):
    train_step(model, True)
ckpt_time = (time.perf_counter() - start) / N * 1000

overhead = (ckpt_time - no_ckpt_time) / no_ckpt_time * 100
print(f"Without checkpointing: {no_ckpt_time:.1f} ms/step")
print(f"With checkpointing:    {ckpt_time:.1f} ms/step")
print(f"Overhead:              {overhead:.1f}%")
print(f"(Typically 20-33% on GPU due to recomputation)")

print("\n" + "=" * 65)
print("5. SELECTIVE CHECKPOINTING (SAC)")
print("=" * 65)

from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

print("""
Selective Activation Checkpointing (SAC) lets you choose per-op
whether to save or recompute during backward:

Policies:
  MUST_SAVE         — Always save (expensive ops like matmul)
  MUST_RECOMPUTE    — Always recompute (cheap ops like relu)
  PREFER_SAVE       — Save unless torch.compile overrides
  PREFER_RECOMPUTE  — Recompute unless torch.compile overrides
  MUST_CPU_OFFLOAD  — Save to CPU, reload to GPU during backward
""")

# Method 1: List of ops to save (simplest)
ops_to_save = [
    torch.ops.aten.mm.default,
    torch.ops.aten.addmm.default,
    torch.ops.aten.bmm.default,
]

print("Ops we'll save (expensive linear algebra):")
for op in ops_to_save:
    print(f"  {op}")

# Create the context function
context_fn = create_selective_checkpoint_contexts(ops_to_save)

# Use with checkpoint
layer = HeavyLayer(256)
x = torch.randn(8, 32, 256, requires_grad=True)
output = checkpoint(layer, x, use_reentrant=False, context_fn=context_fn)
output.sum().backward()
print(f"\nSAC output shape: {output.shape}")
print("SAC completed successfully — matmuls saved, activations recomputed")

print("\n" + "=" * 65)
print("6. CUSTOM POLICY FUNCTION")
print("=" * 65)

# Method 2: Custom policy function for fine-grained control
def my_policy(ctx, op, *args, **kwargs):
    """Save matmuls and attention, recompute everything else."""
    # Save expensive operations
    expensive_ops = {
        torch.ops.aten.mm.default,
        torch.ops.aten.addmm.default,
        torch.ops.aten.bmm.default,
    }
    if op in expensive_ops:
        return CheckpointPolicy.MUST_SAVE

    # Recompute cheap operations (activations, norms, adds)
    return CheckpointPolicy.PREFER_RECOMPUTE

context_fn_custom = create_selective_checkpoint_contexts(my_policy)

x = torch.randn(8, 32, 256, requires_grad=True)
output = checkpoint(layer, x, use_reentrant=False, context_fn=context_fn_custom)
output.sum().backward()
print("Custom policy checkpoint completed")
print("  → Matmuls: SAVED (expensive, don't recompute)")
print("  → ReLU/LayerNorm/Add: RECOMPUTED (cheap)")

print("\n" + "=" * 65)
print("7. PRACTICAL TRAINING EXAMPLE")
print("=" * 65)

class CheckpointedTransformer(nn.Module):
    """A Transformer that uses activation checkpointing."""
    def __init__(self, d_model=256, n_layers=8, use_sac=False):
        super().__init__()
        self.layers = nn.ModuleList([HeavyLayer(d_model) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, 10)
        self.use_sac = use_sac

        if use_sac:
            self.context_fn = create_selective_checkpoint_contexts([
                torch.ops.aten.mm.default,
                torch.ops.aten.addmm.default,
            ])
        else:
            self.context_fn = None

    def forward(self, x):
        for layer in self.layers:
            if self.context_fn is not None:
                x = checkpoint(
                    layer, x,
                    use_reentrant=False,
                    context_fn=self.context_fn
                )
            else:
                x = checkpoint(layer, x, use_reentrant=False)
        return self.head(x.mean(dim=1))

# Train with basic checkpointing
model_basic = CheckpointedTransformer(d_model=256, n_layers=8, use_sac=False)
optimizer = torch.optim.AdamW(model_basic.parameters(), lr=1e-3)

print("Training with basic checkpointing:")
for step in range(5):
    x = torch.randn(16, 32, 256)
    target = torch.randint(0, 10, (16,))
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(model_basic(x), target)
    loss.backward()
    optimizer.step()
    print(f"  Step {step+1}: loss = {loss.item():.4f}")

# Train with SAC
model_sac = CheckpointedTransformer(d_model=256, n_layers=8, use_sac=True)
optimizer = torch.optim.AdamW(model_sac.parameters(), lr=1e-3)

print("\nTraining with selective checkpointing (SAC):")
for step in range(5):
    x = torch.randn(16, 32, 256)
    target = torch.randint(0, 10, (16,))
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(model_sac(x), target)
    loss.backward()
    optimizer.step()
    print(f"  Step {step+1}: loss = {loss.item():.4f}")

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print("""
Activation Checkpointing Cheat Sheet:

1. BASIC: checkpoint(layer, x, use_reentrant=False)
   → Saves ~60% activation memory, costs ~33% more compute

2. SEQUENTIAL: checkpoint_sequential(model, segments=N, input=x)
   → For nn.Sequential models, split into N checkpointed segments

3. SELECTIVE (SAC): checkpoint(layer, x, context_fn=...)
   → Save expensive ops (matmul), recompute cheap ones (relu, norm)
   → Best memory/compute tradeoff

4. Always use use_reentrant=False (modern, correct behavior)
5. Stacks with: AMP, torch.compile, DDP, FSDP2
""")

print("Done!")
