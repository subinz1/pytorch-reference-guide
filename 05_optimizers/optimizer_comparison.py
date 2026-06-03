"""
Module 05: Optimizer Comparison
================================
Compare convergence behavior of SGD, SGD+momentum, Adam, and AdamW
on the same optimization problem. Includes gradient clipping demonstration.

Run: python optimizer_comparison.py
"""

import torch
import torch.nn as nn
import torch.optim as optim

print("=" * 70)
print("PART 1: Convergence Comparison on Regression Task")
print("=" * 70)

# Create a reproducible regression problem
torch.manual_seed(42)
N = 500
X = torch.randn(N, 20)
# True function: linear combination with some noise
true_weights = torch.randn(20, 1)
y = X @ true_weights + torch.randn(N, 1) * 0.5


class RegressionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


criterion = nn.MSELoss()
num_epochs = 200


def train_and_record(model, optimizer, scheduler=None):
    """Train model and record loss history."""
    losses = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        losses.append(loss.item())
    return losses


# Test each optimizer
print(f"\nTraining for {num_epochs} epochs on regression task (N={N}, features=20):")
print(f"\n{'Optimizer':<25} {'LR':<8} {'Final Loss':<12} {'Best Loss':<12} {'Converged@':<12}")
print("-" * 69)

results = {}

configs = [
    ("SGD (lr=0.01)", lambda m: optim.SGD(m.parameters(), lr=0.01)),
    ("SGD+momentum (lr=0.01)", lambda m: optim.SGD(m.parameters(), lr=0.01, momentum=0.9)),
    ("SGD+Nesterov (lr=0.01)", lambda m: optim.SGD(m.parameters(), lr=0.01, momentum=0.9, nesterov=True)),
    ("Adam (lr=0.001)", lambda m: optim.Adam(m.parameters(), lr=0.001)),
    ("Adam (lr=0.01)", lambda m: optim.Adam(m.parameters(), lr=0.01)),
    ("AdamW (lr=0.001)", lambda m: optim.AdamW(m.parameters(), lr=0.001, weight_decay=0.01)),
    ("AdamW (lr=0.01)", lambda m: optim.AdamW(m.parameters(), lr=0.01, weight_decay=0.01)),
    ("RMSprop (lr=0.001)", lambda m: optim.RMSprop(m.parameters(), lr=0.001)),
    ("Adagrad (lr=0.01)", lambda m: optim.Adagrad(m.parameters(), lr=0.01)),
]

for name, opt_fn in configs:
    torch.manual_seed(42)
    model = RegressionNet()
    optimizer = opt_fn(model)
    losses = train_and_record(model, optimizer)

    # Find when it "converged" (first time loss < 0.5)
    converged_at = "never"
    for i, l in enumerate(losses):
        if l < 0.5:
            converged_at = str(i)
            break

    best_loss = min(losses)
    results[name] = losses
    lr = optimizer.param_groups[0]["lr"]
    print(f"{name:<25} {lr:<8.4f} {losses[-1]:<12.4f} {best_loss:<12.4f} {converged_at:<12}")

print("\n" + "=" * 70)
print("PART 2: Effect of Learning Rate")
print("=" * 70)

print(f"\nAdam with different learning rates:")
print(f"{'LR':<12} {'Loss@10':<12} {'Loss@50':<12} {'Loss@200':<12} {'Diverged?':<12}")
print("-" * 60)

for lr in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]:
    torch.manual_seed(42)
    model = RegressionNet()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = train_and_record(model, optimizer)

    diverged = "Yes" if losses[-1] > losses[0] or losses[-1] != losses[-1] else "No"
    loss_10 = losses[9] if len(losses) > 9 else float("nan")
    loss_50 = losses[49] if len(losses) > 49 else float("nan")
    print(f"{lr:<12.1e} {loss_10:<12.4f} {loss_50:<12.4f} {losses[-1]:<12.4f} {diverged:<12}")

print("\n" + "=" * 70)
print("PART 3: Effect of Momentum")
print("=" * 70)

print(f"\nSGD with different momentum values (lr=0.01):")
print(f"{'Momentum':<12} {'Loss@50':<12} {'Loss@100':<12} {'Loss@200':<12}")
print("-" * 48)

for momentum in [0.0, 0.5, 0.9, 0.95, 0.99]:
    torch.manual_seed(42)
    model = RegressionNet()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=momentum)
    losses = train_and_record(model, optimizer)
    print(f"{momentum:<12.2f} {losses[49]:<12.4f} {losses[99]:<12.4f} {losses[-1]:<12.4f}")

print("\n" + "=" * 70)
print("PART 4: Effect of Weight Decay")
print("=" * 70)

print(f"\nAdamW with different weight decay values (lr=0.001):")
print(f"{'Weight Decay':<14} {'Final Loss':<12} {'Weight Norm':<12}")
print("-" * 38)

for wd in [0.0, 0.001, 0.01, 0.05, 0.1, 0.5]:
    torch.manual_seed(42)
    model = RegressionNet()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=wd)
    losses = train_and_record(model, optimizer)
    weight_norm = sum(p.norm().item() ** 2 for p in model.parameters()) ** 0.5
    print(f"{wd:<14.3f} {losses[-1]:<12.4f} {weight_norm:<12.2f}")

print("\nNote: Higher weight decay -> smaller weights but potentially higher loss")
print("Sweet spot is usually 0.01-0.1")

print("\n" + "=" * 70)
print("PART 5: Gradient Clipping")
print("=" * 70)

print("""
Gradient clipping prevents exploding gradients by limiting gradient magnitude.
Two methods:
  1. clip_grad_norm_: Scale all gradients so total L2 norm <= max_norm
  2. clip_grad_value_: Clamp each element to [-clip_value, clip_value]
""")


class DeepNet(nn.Module):
    """A deeper network more prone to gradient issues."""

    def __init__(self):
        super().__init__()
        layers = []
        for i in range(10):
            in_dim = 20 if i == 0 else 64
            layers.extend([nn.Linear(in_dim, 64), nn.ReLU()])
        layers.append(nn.Linear(64, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Without clipping
print("\n--- Without Gradient Clipping ---")
torch.manual_seed(42)
model_noclip = DeepNet()
optimizer_noclip = optim.SGD(model_noclip.parameters(), lr=0.01, momentum=0.9)

losses_noclip = []
grad_norms_noclip = []
for epoch in range(100):
    optimizer_noclip.zero_grad()
    loss = criterion(model_noclip(X), y)
    loss.backward()
    total_norm = torch.nn.utils.clip_grad_norm_(model_noclip.parameters(), float("inf"))
    grad_norms_noclip.append(total_norm.item())
    optimizer_noclip.step()
    losses_noclip.append(loss.item())

print(f"Final loss: {losses_noclip[-1]:.4f}")
print(f"Max gradient norm: {max(grad_norms_noclip):.2f}")
print(f"Mean gradient norm: {sum(grad_norms_noclip)/len(grad_norms_noclip):.2f}")

# With clip_grad_norm_
print("\n--- With clip_grad_norm_(max_norm=1.0) ---")
torch.manual_seed(42)
model_clip = DeepNet()
optimizer_clip = optim.SGD(model_clip.parameters(), lr=0.01, momentum=0.9)

losses_clip = []
grad_norms_clip = []
clipped_count = 0
for epoch in range(100):
    optimizer_clip.zero_grad()
    loss = criterion(model_clip(X), y)
    loss.backward()
    # Clip gradients
    total_norm = torch.nn.utils.clip_grad_norm_(model_clip.parameters(), max_norm=1.0)
    if total_norm > 1.0:
        clipped_count += 1
    grad_norms_clip.append(min(total_norm.item(), 1.0))
    optimizer_clip.step()
    losses_clip.append(loss.item())

print(f"Final loss: {losses_clip[-1]:.4f}")
print(f"Max gradient norm (after clip): {max(grad_norms_clip):.2f}")
print(f"Times clipped: {clipped_count}/{100}")

# With clip_grad_value_
print("\n--- With clip_grad_value_(clip_value=0.5) ---")
torch.manual_seed(42)
model_clipv = DeepNet()
optimizer_clipv = optim.SGD(model_clipv.parameters(), lr=0.01, momentum=0.9)

losses_clipv = []
for epoch in range(100):
    optimizer_clipv.zero_grad()
    loss = criterion(model_clipv(X), y)
    loss.backward()
    torch.nn.utils.clip_grad_value_(model_clipv.parameters(), clip_value=0.5)
    optimizer_clipv.step()
    losses_clipv.append(loss.item())

print(f"Final loss: {losses_clipv[-1]:.4f}")

# Compare
print("\n--- Comparison ---")
print(f"{'Method':<30} {'Final Loss':<12}")
print("-" * 42)
print(f"{'No clipping':<30} {losses_noclip[-1]:<12.4f}")
print(f"{'clip_grad_norm_(1.0)':<30} {losses_clip[-1]:<12.4f}")
print(f"{'clip_grad_value_(0.5)':<30} {losses_clipv[-1]:<12.4f}")

print("\n" + "=" * 70)
print("PART 6: Learning Rate Warmup Effect")
print("=" * 70)

print("\nComparing training with and without warmup:")

# Without warmup
torch.manual_seed(42)
model_nowarm = RegressionNet()
optimizer_nowarm = optim.AdamW(model_nowarm.parameters(), lr=0.01, weight_decay=0.01)
losses_nowarm = train_and_record(model_nowarm, optimizer_nowarm)

# With warmup
torch.manual_seed(42)
model_warm = RegressionNet()
optimizer_warm = optim.AdamW(model_warm.parameters(), lr=0.01, weight_decay=0.01)
warmup = optim.lr_scheduler.LinearLR(optimizer_warm, start_factor=0.01, total_iters=20)
cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer_warm, T_max=180, eta_min=1e-5)
scheduler_warm = optim.lr_scheduler.SequentialLR(
    optimizer_warm, schedulers=[warmup, cosine], milestones=[20]
)
losses_warm = train_and_record(model_warm, optimizer_warm, scheduler_warm)

print(f"\n{'Epoch':<8} {'No Warmup':<12} {'With Warmup':<12}")
print("-" * 32)
for epoch in [0, 5, 10, 20, 50, 100, 150, 199]:
    print(f"{epoch:<8} {losses_nowarm[epoch]:<12.4f} {losses_warm[epoch]:<12.4f}")

print(f"\nFinal: No warmup={losses_nowarm[-1]:.4f}, With warmup={losses_warm[-1]:.4f}")

print("\n" + "=" * 70)
print("PART 7: Optimizer Memory Usage")
print("=" * 70)

print("""
Optimizers store additional state per parameter:
  - SGD: 0 or 1 buffer (momentum)
  - Adam/AdamW: 2 buffers (first + second moment)
  - Adagrad: 1 buffer (sum of squared gradients)
""")

model = RegressionNet()
param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"\nModel parameters memory: {param_memory / 1024:.1f} KB")

# SGD (no momentum)
opt = optim.SGD(model.parameters(), lr=0.01)
opt.zero_grad()
criterion(model(X), y).backward()
opt.step()
sgd_state_size = sum(
    sum(v.numel() * v.element_size() for v in state.values() if isinstance(v, torch.Tensor))
    for state in opt.state.values()
)
print(f"SGD state memory: {sgd_state_size / 1024:.1f} KB (no extra state)")

# SGD with momentum
opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
opt.zero_grad()
criterion(model(X), y).backward()
opt.step()
sgd_mom_state_size = sum(
    sum(v.numel() * v.element_size() for v in state.values() if isinstance(v, torch.Tensor))
    for state in opt.state.values()
)
print(f"SGD+momentum state memory: {sgd_mom_state_size / 1024:.1f} KB (1x params)")

# Adam
opt = optim.Adam(model.parameters(), lr=0.001)
opt.zero_grad()
criterion(model(X), y).backward()
opt.step()
adam_state_size = sum(
    sum(v.numel() * v.element_size() for v in state.values() if isinstance(v, torch.Tensor))
    for state in opt.state.values()
)
print(f"Adam state memory: {adam_state_size / 1024:.1f} KB (2x params)")

print(f"\nTotal training memory (model + optimizer):")
print(f"  SGD:          {(param_memory + sgd_state_size) / 1024:.1f} KB")
print(f"  SGD+momentum: {(param_memory + sgd_mom_state_size) / 1024:.1f} KB")
print(f"  Adam:         {(param_memory + adam_state_size) / 1024:.1f} KB")

print("\n" + "=" * 70)
print("PART 8: Complete Training Recipe")
print("=" * 70)

print("\nPutting it all together: a production-quality training loop")


def train_model(model, train_X, train_y, val_X, val_y,
                num_epochs=100, lr=0.001, weight_decay=0.01,
                max_grad_norm=1.0, patience=20):
    """Complete training loop with best practices."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Warmup + cosine schedule
    warmup_epochs = max(1, num_epochs // 10)
    warmup_sched = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs]
    )

    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        output = model(train_X)
        train_loss = criterion(output, train_y)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(val_X)
            val_loss = criterion(val_output, val_y)

        # Early stopping check
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}: train_loss={train_loss.item():.4f}, "
                  f"val_loss={val_loss.item():.4f}, "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)
    return best_val_loss


# Run the complete training
torch.manual_seed(42)
X_all = torch.randn(600, 20)
y_all = X_all @ true_weights + torch.randn(600, 1) * 0.5

train_X, val_X = X_all[:400], X_all[400:]
train_y, val_y = y_all[:400], y_all[400:]

model = RegressionNet()
print("\nTraining with complete recipe:")
best_loss = train_model(model, train_X, train_y, val_X, val_y, num_epochs=200)
print(f"\nBest validation loss: {best_loss:.4f}")

print("\n" + "=" * 70)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
print("=" * 70)
