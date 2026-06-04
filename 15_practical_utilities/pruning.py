"""
Model Pruning — Making Neural Networks Smaller
================================================
Learn how to prune weights from a model to reduce size and computation.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

print("=" * 65)
print("1. BASIC UNSTRUCTURED PRUNING")
print("=" * 65)

# Create a simple linear layer
linear = nn.Linear(10, 5)
print(f"Original weight shape: {linear.weight.shape}")
print(f"Original weight:\n{linear.weight.data.round(decimals=3)}")

# Prune 40% of weights by L1 magnitude (smallest absolute values)
prune.l1_unstructured(linear, name="weight", amount=0.4)

print(f"\nAfter pruning 40% by L1 magnitude:")
print(f"Weight:\n{linear.weight.data.round(decimals=3)}")

# Count zeros
zeros = (linear.weight == 0).sum().item()
total = linear.weight.numel()
print(f"Sparsity: {zeros}/{total} = {100*zeros/total:.1f}%")

# PyTorch stores the original weight and a mask
print(f"\nInternal state:")
print(f"  weight_orig shape: {linear.weight_orig.shape}")
print(f"  weight_mask:\n{linear.weight_mask}")

print("\n" + "=" * 65)
print("2. DIFFERENT PRUNING METHODS")
print("=" * 65)

# --- Random unstructured ---
linear1 = nn.Linear(10, 5)
prune.random_unstructured(linear1, name="weight", amount=0.3)
sparsity1 = (linear1.weight == 0).sum().item() / linear1.weight.numel()
print(f"Random unstructured (30%): actual sparsity = {sparsity1:.1%}")

# --- L1 unstructured (prune by magnitude) ---
linear2 = nn.Linear(10, 5)
prune.l1_unstructured(linear2, name="weight", amount=0.3)
sparsity2 = (linear2.weight == 0).sum().item() / linear2.weight.numel()
print(f"L1 unstructured (30%):     actual sparsity = {sparsity2:.1%}")

# --- Structured pruning (entire neurons/channels) ---
linear3 = nn.Linear(10, 5)
# Prune 2 out of 5 output neurons (rows) with smallest L2 norm
prune.ln_structured(linear3, name="weight", amount=2, n=2, dim=0)
print(f"\nStructured pruning (2 neurons by L2 norm):")
row_norms = linear3.weight.data.norm(dim=1)
print(f"  Row norms: {row_norms.round(decimals=3)}")
print(f"  Zero rows: {(row_norms == 0).sum().item()}")

print("\n" + "=" * 65)
print("3. PRUNING A CONV LAYER (Structured — Channel Pruning)")
print("=" * 65)

conv = nn.Conv2d(16, 32, 3, padding=1)
print(f"Conv weight shape: {conv.weight.shape}")  # (32, 16, 3, 3)

# Prune 50% of output channels (structured along dim 0)
prune.ln_structured(conv, name="weight", amount=0.5, n=1, dim=0)

# Count pruned channels
channel_norms = conv.weight.data.reshape(32, -1).norm(dim=1)
pruned_channels = (channel_norms == 0).sum().item()
print(f"Pruned {pruned_channels}/32 output channels ({pruned_channels/32:.0%})")

print("\n" + "=" * 65)
print("4. ITERATIVE PRUNING (Multiple Rounds)")
print("=" * 65)

linear = nn.Linear(20, 10)

# Prune in three rounds: 20% → 20% → 20%
for round_num in range(1, 4):
    prune.l1_unstructured(linear, name="weight", amount=0.2)
    sparsity = (linear.weight == 0).sum().item() / linear.weight.numel()
    print(f"Round {round_num}: sparsity = {sparsity:.1%}")

# After 3 rounds of 20% pruning on remaining weights:
# Round 1: 20% of 100% = 20% sparse
# Round 2: 20% of remaining 80% = 36% sparse
# Round 3: 20% of remaining 64% = ~49% sparse

print("\n" + "=" * 65)
print("5. GLOBAL PRUNING (Across All Layers)")
print("=" * 65)

model = nn.Sequential(
    nn.Linear(100, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10),
)

# Collect all weight parameters
parameters_to_prune = [
    (model[0], "weight"),
    (model[2], "weight"),
    (model[4], "weight"),
]

# Before pruning
total_params = sum(m.weight.numel() for m, _ in parameters_to_prune)
print(f"Total weight parameters: {total_params}")

# Globally prune 50% (the smallest weights across ALL layers are pruned first)
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.5,
)

# Check per-layer sparsity
print("\nPer-layer sparsity after global pruning:")
for name, (module, param_name) in zip(["fc1", "fc2", "fc3"], parameters_to_prune):
    w = getattr(module, param_name)
    s = (w == 0).sum().item() / w.numel()
    print(f"  {name}: {s:.1%} sparse")

total_zeros = sum((getattr(m, p) == 0).sum().item() for m, p in parameters_to_prune)
print(f"  Overall: {total_zeros/total_params:.1%} sparse")

print("\n" + "=" * 65)
print("6. MAKING PRUNING PERMANENT")
print("=" * 65)

linear = nn.Linear(10, 5)
prune.l1_unstructured(linear, name="weight", amount=0.5)

print("Before remove():")
print(f"  Has weight_orig: {hasattr(linear, 'weight_orig')}")
print(f"  Has weight_mask: {hasattr(linear, 'weight_mask')}")

# Make permanent: fold the mask into the weight
prune.remove(linear, "weight")

print("\nAfter remove():")
print(f"  Has weight_orig: {hasattr(linear, 'weight_orig')}")
print(f"  Has weight_mask: {hasattr(linear, 'weight_mask')}")
print(f"  Sparsity preserved: {(linear.weight == 0).sum().item() / linear.weight.numel():.1%}")

print("\n" + "=" * 65)
print("7. CUSTOM PRUNING METHOD")
print("=" * 65)

class TopKPruning(prune.BasePruningMethod):
    """Keep only the top-k weights by magnitude in each row."""
    PRUNING_TYPE = "unstructured"

    def __init__(self, k):
        super().__init__()
        self.k = k

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        # For each row, keep only top-k by magnitude
        for i in range(t.shape[0]):
            row = t[i].abs()
            if self.k < row.numel():
                threshold = row.topk(self.k).values[-1]
                mask[i][row < threshold] = 0
        return mask

linear = nn.Linear(10, 5)
TopKPruning.apply(linear, name="weight", k=3)

print("Custom TopK pruning (keep 3 per row):")
for i in range(5):
    nonzero = (linear.weight[i] != 0).sum().item()
    print(f"  Row {i}: {nonzero} non-zero weights")

print("\n" + "=" * 65)
print("8. PRUNING + TRAINING WORKFLOW")
print("=" * 65)

# Typical workflow: train → prune → fine-tune → (repeat)
model = nn.Sequential(nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, 5))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Simulate training
x, y = torch.randn(32, 20), torch.randn(32, 5)
for _ in range(10):
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
print(f"Pre-pruning loss: {loss.item():.4f}")

# Prune
for module in [model[0], model[2]]:
    prune.l1_unstructured(module, name="weight", amount=0.3)

# Fine-tune (the pruned weights stay zero due to the mask)
for step in range(20):
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
print(f"Post-pruning fine-tuned loss: {loss.item():.4f}")

# Verify zeros are maintained during fine-tuning
s0 = (model[0].weight == 0).sum().item() / model[0].weight.numel()
s2 = (model[2].weight == 0).sum().item() / model[2].weight.numel()
print(f"Sparsity maintained: layer0={s0:.1%}, layer2={s2:.1%}")

# Make permanent
for module in [model[0], model[2]]:
    prune.remove(module, "weight")

print("\nDone!")
