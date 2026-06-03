"""
Gradient Techniques — Accumulation, Checkpointing, and Clipping
================================================================
Demonstrates three key gradient management techniques:
1. Gradient Accumulation — simulate larger batch sizes
2. Gradient Checkpointing — trade compute for memory
3. Gradient Clipping — prevent exploding gradients

Run: python gradient_techniques.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.checkpoint import checkpoint

# =============================================================================
# Shared setup
# =============================================================================

torch.manual_seed(42)

num_samples = 2000
input_dim = 128
num_classes = 5

X = torch.randn(num_samples, input_dim)
y = torch.randint(0, num_classes, (num_samples,))
dataset = TensorDataset(X, y)

class DeepMLP(nn.Module):
    """A deeper network to demonstrate gradient issues."""

    def __init__(self, input_dim=128, hidden_dim=256, num_layers=8, output_dim=5):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# =============================================================================
# 1. GRADIENT ACCUMULATION
# =============================================================================

print("=" * 60)
print("1. GRADIENT ACCUMULATION")
print("=" * 60)
print("\nSimulating batch_size=128 with actual batch_size=32")
print("(4 accumulation steps)\n")

model = DeepMLP()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Small batch dataloader
small_batch_loader = DataLoader(dataset, batch_size=32, shuffle=True)
accumulation_steps = 4  # Effective batch size = 32 * 4 = 128

model.train()
optimizer.zero_grad()

total_loss = 0.0
steps_done = 0

for i, (inputs, targets) in enumerate(small_batch_loader):
    # Forward pass
    output = model(inputs)
    loss = loss_fn(output, targets)

    # IMPORTANT: Scale loss by accumulation steps
    # This makes accumulated gradients equivalent to a single large-batch gradient
    scaled_loss = loss / accumulation_steps
    scaled_loss.backward()

    total_loss += loss.item()

    # Only step every accumulation_steps batches
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
        steps_done += 1

        if steps_done <= 3:
            avg_loss = total_loss / accumulation_steps
            print(f"  Optimizer step {steps_done}: avg loss = {avg_loss:.4f} "
                  f"(accumulated over {accumulation_steps} mini-batches)")
            total_loss = 0.0

    if steps_done >= 3:
        break

# Verify: compare gradients from accumulation vs large batch
print("\n  Verifying gradient equivalence:")

model_accum = DeepMLP()
model_large = DeepMLP()
model_large.load_state_dict(model_accum.state_dict())

# Accumulation approach: 4 batches of 32
loader_small = DataLoader(dataset, batch_size=32, shuffle=False)
model_accum.zero_grad()
for i, (inp, tgt) in enumerate(loader_small):
    if i >= 4:
        break
    loss = loss_fn(model_accum(inp), tgt) / 4
    loss.backward()

# Large batch approach: 1 batch of 128
loader_large = DataLoader(dataset, batch_size=128, shuffle=False)
model_large.zero_grad()
inp, tgt = next(iter(loader_large))
loss = loss_fn(model_large(inp), tgt)
loss.backward()

# Compare gradients
first_param_accum = list(model_accum.parameters())[0].grad
first_param_large = list(model_large.parameters())[0].grad
grad_diff = (first_param_accum - first_param_large).abs().max().item()
print(f"  Max gradient difference: {grad_diff:.8f}")
print(f"  Gradients match: {grad_diff < 1e-5}")

# =============================================================================
# 2. GRADIENT CHECKPOINTING
# =============================================================================

print("\n" + "=" * 60)
print("2. GRADIENT CHECKPOINTING")
print("=" * 60)
print("\nTrades ~33% more compute for ~60-80% memory savings on activations\n")

class CheckpointedModel(nn.Module):
    """Model that uses gradient checkpointing for memory efficiency."""

    def __init__(self, input_dim=128, hidden_dim=256, output_dim=5):
        super().__init__()
        # Define blocks that will be checkpointed
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.block3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, use_checkpoint=False):
        if use_checkpoint:
            # Activations are NOT saved; recomputed during backward
            x = checkpoint(self.block1, x, use_reentrant=False)
            x = checkpoint(self.block2, x, use_reentrant=False)
            x = checkpoint(self.block3, x, use_reentrant=False)
        else:
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
        return self.head(x)


# Compare memory usage (approximation via tensor counting)
model_cp = CheckpointedModel()
sample = torch.randn(64, 128)

# Without checkpointing — measure saved tensors
model_cp.zero_grad()
out = model_cp(sample, use_checkpoint=False)
loss = out.sum()
loss.backward()
print("  Without checkpointing:")
print(f"    Forward pass completes normally")
print(f"    All intermediate activations stored in memory")

# With checkpointing
model_cp.zero_grad()
out = model_cp(sample, use_checkpoint=True)
loss = out.sum()
loss.backward()
print("  With checkpointing:")
print(f"    Only block boundary activations stored")
print(f"    Intermediates recomputed during backward")

# Verify gradients are the same
model_a = CheckpointedModel()
model_b = CheckpointedModel()
model_b.load_state_dict(model_a.state_dict())

inp = torch.randn(32, 128)

model_a.zero_grad()
loss_a = model_a(inp, use_checkpoint=False).sum()
loss_a.backward()

model_b.zero_grad()
loss_b = model_b(inp, use_checkpoint=True).sum()
loss_b.backward()

max_diff = max(
    (pa.grad - pb.grad).abs().max().item()
    for pa, pb in zip(model_a.parameters(), model_b.parameters())
)
print(f"\n  Gradient difference (should be ~0): {max_diff:.2e}")
print(f"  Checkpointing produces correct gradients: {max_diff < 1e-5}")

# =============================================================================
# 3. GRADIENT CLIPPING
# =============================================================================

print("\n" + "=" * 60)
print("3. GRADIENT CLIPPING")
print("=" * 60)

# Create a scenario with potentially large gradients
class UnstableModel(nn.Module):
    """Model without activation functions — gradients can explode."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 5),
        )
        # Initialize with larger weights to induce gradient explosion
        for layer in self.layers:
            if hasattr(layer, 'weight'):
                nn.init.normal_(layer.weight, std=2.0)

    def forward(self, x):
        return self.layers(x)


model_unstable = UnstableModel()
optimizer_unstable = optim.SGD(model_unstable.parameters(), lr=0.01)

sample_input = torch.randn(32, 128)
sample_target = torch.randint(0, 5, (32,))

# Forward + backward to get large gradients
optimizer_unstable.zero_grad()
output = model_unstable(sample_input)
loss = loss_fn(output, sample_target)
loss.backward()

# Check gradient norm BEFORE clipping
total_norm_before = torch.nn.utils.clip_grad_norm_(
    model_unstable.parameters(), max_norm=float('inf')
)
print(f"\n  Gradient norm BEFORE clipping: {total_norm_before:.2f}")

# Now actually clip
optimizer_unstable.zero_grad()
output = model_unstable(sample_input)
loss = loss_fn(output, sample_target)
loss.backward()

# clip_grad_norm_: scales ALL gradients so total L2 norm <= max_norm
total_norm_after = torch.nn.utils.clip_grad_norm_(
    model_unstable.parameters(), max_norm=1.0
)
print(f"  Gradient norm AFTER clip_grad_norm_(max_norm=1.0): ", end="")

# Verify the norm is now <= 1.0
actual_norm = torch.sqrt(sum(
    p.grad.norm() ** 2 for p in model_unstable.parameters() if p.grad is not None
))
print(f"{actual_norm:.4f}")

# Demonstrate clip_grad_value_
print("\n  --- clip_grad_value_ ---")
optimizer_unstable.zero_grad()
output = model_unstable(sample_input)
loss = loss_fn(output, sample_target)
loss.backward()

# Check max gradient value before clipping
max_grad = max(
    p.grad.abs().max().item()
    for p in model_unstable.parameters() if p.grad is not None
)
print(f"  Max gradient value BEFORE: {max_grad:.4f}")

torch.nn.utils.clip_grad_value_(model_unstable.parameters(), clip_value=0.5)

max_grad_after = max(
    p.grad.abs().max().item()
    for p in model_unstable.parameters() if p.grad is not None
)
print(f"  Max gradient value AFTER clip_grad_value_(0.5): {max_grad_after:.4f}")

# =============================================================================
# 4. Combining all three techniques
# =============================================================================

print("\n" + "=" * 60)
print("4. COMBINING ALL TECHNIQUES")
print("=" * 60)
print("\nComplete training loop with accumulation + checkpointing + clipping:\n")

model_combined = CheckpointedModel()
optimizer_combined = optim.Adam(model_combined.parameters(), lr=1e-3)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

accumulation_steps = 4
max_grad_norm = 1.0
num_steps = 10

model_combined.train()
optimizer_combined.zero_grad()

step_count = 0
accum_loss = 0.0

for i, (inputs, targets) in enumerate(loader):
    # Forward with checkpointing (saves memory)
    output = model_combined(inputs, use_checkpoint=True)
    loss = loss_fn(output, targets)

    # Scale for accumulation
    (loss / accumulation_steps).backward()
    accum_loss += loss.item()

    if (i + 1) % accumulation_steps == 0:
        # Clip gradients (prevents explosions)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model_combined.parameters(), max_norm=max_grad_norm
        )

        # Step
        optimizer_combined.step()
        optimizer_combined.zero_grad()

        step_count += 1
        print(f"  Step {step_count}: loss={accum_loss/accumulation_steps:.4f}, "
              f"grad_norm={grad_norm:.4f}")
        accum_loss = 0.0

        if step_count >= num_steps:
            break

print("\nAll gradient techniques demonstrated successfully!")
