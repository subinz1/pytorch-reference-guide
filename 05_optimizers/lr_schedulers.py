"""
Module 05: Learning Rate Schedulers
=====================================
Complete guide to all major LR schedulers in PyTorch, with
demonstrations of their behavior over training.

Run: python lr_schedulers.py
"""

import torch
import torch.nn as nn
import torch.optim as optim

print("=" * 70)
print("PART 1: StepLR — Fixed-step Decay")
print("=" * 70)

print("""
StepLR: Multiply LR by gamma every step_size epochs.
Pattern: constant -> drop -> constant -> drop -> ...
""")

model = nn.Linear(10, 1)
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

print(f"StepLR(step_size=10, gamma=0.5):")
lrs = []
for epoch in range(40):
    lrs.append(optimizer.param_groups[0]["lr"])
    scheduler.step()

# Print LR at key epochs
for epoch in [0, 9, 10, 19, 20, 29, 30, 39]:
    print(f"  Epoch {epoch:2d}: lr = {lrs[epoch]:.4f}")

print("\n" + "=" * 70)
print("PART 2: MultiStepLR — Decay at Specific Milestones")
print("=" * 70)

optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[30, 60, 80], gamma=0.1
)

print(f"MultiStepLR(milestones=[30,60,80], gamma=0.1):")
lrs = []
for epoch in range(100):
    lrs.append(optimizer.param_groups[0]["lr"])
    scheduler.step()

for epoch in [0, 29, 30, 59, 60, 79, 80, 99]:
    print(f"  Epoch {epoch:2d}: lr = {lrs[epoch]:.6f}")

print("\n" + "=" * 70)
print("PART 3: ExponentialLR — Continuous Decay")
print("=" * 70)

optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

print(f"ExponentialLR(gamma=0.95): lr_t = lr_0 * gamma^t")
lrs = []
for epoch in range(50):
    lrs.append(optimizer.param_groups[0]["lr"])
    scheduler.step()

for epoch in [0, 10, 20, 30, 40, 49]:
    print(f"  Epoch {epoch:2d}: lr = {lrs[epoch]:.6f}")

print("\n" + "=" * 70)
print("PART 4: CosineAnnealingLR — Smooth Cosine Decay")
print("=" * 70)

print("""
CosineAnnealingLR: Follows a half-cosine from lr_max to eta_min.
Formula: lr_t = eta_min + (lr_max - eta_min) * (1 + cos(pi * t / T_max)) / 2
""")

optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=50, eta_min=1e-5
)

print(f"CosineAnnealingLR(T_max=50, eta_min=1e-5):")
lrs = []
for epoch in range(50):
    lrs.append(optimizer.param_groups[0]["lr"])
    scheduler.step()

for epoch in [0, 10, 20, 25, 30, 40, 49]:
    print(f"  Epoch {epoch:2d}: lr = {lrs[epoch]:.6f}")

print("\n" + "=" * 70)
print("PART 5: OneCycleLR — Warmup + Cosine Decay (Per Batch)")
print("=" * 70)

print("""
OneCycleLR implements the 1-cycle policy (Smith, 2018):
1. Warmup: LR ramps up from initial to max_lr
2. Annealing: LR decays from max_lr to min_lr

IMPORTANT: Call scheduler.step() after each BATCH, not each epoch!
""")

model = nn.Linear(10, 1)
optimizer = optim.SGD(model.parameters(), lr=0.01)
total_steps = 100  # Total number of batches across all epochs
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.1,
    total_steps=total_steps,
    pct_start=0.3,  # 30% warmup
    anneal_strategy="cos",
    div_factor=10,  # initial_lr = max_lr / div_factor
    final_div_factor=100,  # min_lr = initial_lr / final_div_factor
)

print(f"OneCycleLR(max_lr=0.1, pct_start=0.3):")
lrs = []
for step in range(total_steps):
    lrs.append(optimizer.param_groups[0]["lr"])
    # In real training: do forward/backward/step here
    scheduler.step()

for step in [0, 10, 20, 30, 50, 70, 90, 99]:
    print(f"  Step {step:3d}: lr = {lrs[step]:.6f}")

print(f"\n  Initial LR: {lrs[0]:.6f}")
print(f"  Peak LR: {max(lrs):.6f} (at step ~30)")
print(f"  Final LR: {lrs[-1]:.8f}")

print("\n" + "=" * 70)
print("PART 6: ReduceLROnPlateau — Adaptive Based on Metric")
print("=" * 70)

print("""
ReduceLROnPlateau monitors a metric and reduces LR when it stops improving.
Unlike other schedulers, you pass the metric value to step().
""")

optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",  # Reduce when metric stops decreasing
    factor=0.5,  # Multiply LR by 0.5
    patience=5,  # Wait 5 epochs of no improvement
    threshold=1e-4,  # Minimum change to qualify as improvement
    min_lr=1e-6,
    verbose=False,
)

print(f"ReduceLROnPlateau(factor=0.5, patience=5):")
# Simulate training with plateauing loss
torch.manual_seed(42)
simulated_losses = [1.0]
for i in range(50):
    if i < 10:
        simulated_losses.append(simulated_losses[-1] * 0.95)  # Improving
    elif i < 25:
        simulated_losses.append(simulated_losses[-1] + torch.randn(1).item() * 0.01)  # Plateau
    else:
        simulated_losses.append(simulated_losses[-1] * 0.98)  # Improving again

lrs = []
for epoch, val_loss in enumerate(simulated_losses[:40]):
    lrs.append(optimizer.param_groups[0]["lr"])
    scheduler.step(val_loss)  # Pass the metric!

for epoch in [0, 5, 10, 15, 16, 20, 25, 30, 35, 39]:
    if epoch < len(lrs):
        print(f"  Epoch {epoch:2d}: lr = {lrs[epoch]:.6f}, val_loss = {simulated_losses[epoch]:.4f}")

print("\n" + "=" * 70)
print("PART 7: LinearLR — Linear Warmup/Decay")
print("=" * 70)

optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.1,  # Start at lr * 0.1 = 0.01
    end_factor=1.0,  # End at lr * 1.0 = 0.1
    total_iters=10,  # Over 10 epochs
)

print(f"LinearLR(start_factor=0.1, end_factor=1.0, total_iters=10):")
lrs = []
for epoch in range(15):
    lrs.append(optimizer.param_groups[0]["lr"])
    scheduler.step()

for epoch in range(15):
    print(f"  Epoch {epoch:2d}: lr = {lrs[epoch]:.4f}")

print("\n" + "=" * 70)
print("PART 8: SequentialLR — Chaining Schedulers")
print("=" * 70)

print("""
SequentialLR chains multiple schedulers. Common pattern:
  LinearLR (warmup) -> CosineAnnealingLR (decay)
""")

optimizer = optim.SGD(model.parameters(), lr=0.1)

# Phase 1: Linear warmup for 10 epochs
warmup = optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.01, end_factor=1.0, total_iters=10
)
# Phase 2: Cosine decay for remaining 90 epochs
cosine = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=90, eta_min=1e-5
)
# Chain them
scheduler = optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup, cosine], milestones=[10]
)

print(f"SequentialLR: LinearLR(10 epochs) -> CosineAnnealing(90 epochs):")
lrs = []
for epoch in range(100):
    lrs.append(optimizer.param_groups[0]["lr"])
    scheduler.step()

for epoch in [0, 5, 10, 20, 30, 50, 70, 90, 99]:
    print(f"  Epoch {epoch:2d}: lr = {lrs[epoch]:.6f}")

print("\n" + "=" * 70)
print("PART 9: CosineAnnealingWarmRestarts")
print("=" * 70)

print("""
Cosine annealing with periodic warm restarts:
  - T_0: number of epochs until first restart
  - T_mult: factor to increase period after each restart
    (T_mult=2: 10, 20, 40, 80, ...)
""")

optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-5
)

print(f"CosineAnnealingWarmRestarts(T_0=10, T_mult=2):")
lrs = []
for epoch in range(70):
    lrs.append(optimizer.param_groups[0]["lr"])
    scheduler.step()

# Show restarts
for epoch in [0, 5, 9, 10, 15, 20, 29, 30, 40, 50, 60, 69]:
    print(f"  Epoch {epoch:2d}: lr = {lrs[epoch]:.6f}"
          + (" <- RESTART" if lrs[epoch] > 0.09 and epoch > 0 else ""))

print("\n" + "=" * 70)
print("PART 10: Practical Training Loop with Scheduler")
print("=" * 70)


class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Full training setup
torch.manual_seed(42)
X = torch.randn(200, 10)
y = X[:, 0:1] * 2 + X[:, 1:2] * 3 + torch.randn(200, 1) * 0.1

model = SmallNet()
optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)

# Warmup + Cosine schedule
num_epochs = 50
warmup_epochs = 5
warmup_scheduler = optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
)
cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-5
)
scheduler = optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_epochs]
)

criterion = nn.MSELoss()

print(f"\nTraining with warmup ({warmup_epochs} epochs) + cosine decay:")
print(f"{'Epoch':<8}{'LR':<12}{'Train Loss':<12}")
print("-" * 32)

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % 10 == 0 or epoch == num_epochs - 1:
        print(f"{epoch:<8}{optimizer.param_groups[0]['lr']:<12.6f}{loss.item():<12.6f}")

print("\n" + "=" * 70)
print("PART 11: LambdaLR — Custom Schedule")
print("=" * 70)

optimizer = optim.SGD(model.parameters(), lr=0.1)

# Custom: linear decay to 0 over 100 epochs
scheduler = optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda epoch: 1 - epoch / 100
)

print("LambdaLR with custom linear decay:")
lrs = []
for epoch in range(100):
    lrs.append(optimizer.param_groups[0]["lr"])
    scheduler.step()

for epoch in [0, 25, 50, 75, 99]:
    print(f"  Epoch {epoch}: lr = {lrs[epoch]:.4f}")

# Different lambda per parameter group
print("\n--- Different schedules per group ---")
model2 = SmallNet()
optimizer2 = optim.SGD([
    {"params": model2.fc1.parameters(), "lr": 0.1},
    {"params": model2.fc2.parameters(), "lr": 0.05},
    {"params": model2.fc3.parameters(), "lr": 0.01},
])

# Each group gets its own lambda
scheduler2 = optim.lr_scheduler.LambdaLR(
    optimizer2,
    lr_lambda=[
        lambda e: 0.95 ** e,  # Exponential decay for group 0
        lambda e: 1.0,  # Constant for group 1
        lambda e: max(0.1, 1 - e / 50),  # Linear decay with floor for group 2
    ],
)

print("Multi-group LambdaLR:")
for epoch in [0, 10, 20, 30, 40]:
    lrs = [g["lr"] for g in optimizer2.param_groups]
    print(f"  Epoch {epoch}: {[f'{lr:.5f}' for lr in lrs]}")
    scheduler2.step()

print("\n" + "=" * 70)
print("SUMMARY: Scheduler Selection Guide")
print("=" * 70)

print("""
Scheduler              | Best For                   | Step Per
-----------------------|----------------------------|----------
StepLR                 | Simple baseline             | Epoch
MultiStepLR            | Known milestone schedule    | Epoch
ExponentialLR          | Smooth continuous decay     | Epoch
CosineAnnealingLR      | Most tasks (smooth)         | Epoch
OneCycleLR             | Super-convergence           | Batch
ReduceLROnPlateau      | When you have val metric    | Epoch (with metric)
LinearLR               | Warmup phase                | Epoch
SequentialLR           | Warmup + main schedule      | Epoch
CosineWarmRestarts     | Long training with resets   | Epoch
LambdaLR               | Custom schedules            | Epoch

Modern best practice:
  1. Use warmup (LinearLR for 5-10% of training)
  2. Followed by CosineAnnealing to near-zero
  3. Or just use OneCycleLR (all-in-one)
""")

print("=" * 70)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
print("=" * 70)
