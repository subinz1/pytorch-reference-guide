"""
Module 05: Optimizer Basics
============================
Complete guide to using PyTorch optimizers: creating, configuring,
parameter groups, state management, and the training loop.

Run: python optimizer_basics.py
"""

import torch
import torch.nn as nn
import torch.optim as optim

print("=" * 70)
print("PART 1: Basic Optimizer Usage")
print("=" * 70)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 32)
        self.layer2 = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)


model = SimpleModel()
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

# Create an optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
print(f"\nOptimizer: {optimizer.__class__.__name__}")
print(f"Number of parameter groups: {len(optimizer.param_groups)}")
print(f"LR: {optimizer.param_groups[0]['lr']}")
print(f"Momentum: {optimizer.param_groups[0]['momentum']}")

# The training loop
print("\n--- The Training Loop ---")
criterion = nn.MSELoss()

# Generate some fake data
torch.manual_seed(42)
X = torch.randn(100, 10)
y = torch.randn(100, 1)

losses = []
for step in range(50):
    # 1. Zero gradients (MUST do before backward)
    optimizer.zero_grad()

    # 2. Forward pass
    predictions = model(X)
    loss = criterion(predictions, y)

    # 3. Backward pass (compute gradients)
    loss.backward()

    # 4. Update parameters
    optimizer.step()

    losses.append(loss.item())
    if step % 10 == 0:
        print(f"  Step {step:3d}: loss = {loss.item():.4f}")

print(f"\nLoss decreased from {losses[0]:.4f} to {losses[-1]:.4f}")

print("\n" + "=" * 70)
print("PART 2: Different Optimizers")
print("=" * 70)

# Compare different optimizers on the same problem
optimizers_to_test = {
    "SGD": lambda p: optim.SGD(p, lr=0.01),
    "SGD+momentum": lambda p: optim.SGD(p, lr=0.01, momentum=0.9),
    "SGD+Nesterov": lambda p: optim.SGD(p, lr=0.01, momentum=0.9, nesterov=True),
    "Adam": lambda p: optim.Adam(p, lr=0.001),
    "AdamW": lambda p: optim.AdamW(p, lr=0.001, weight_decay=0.01),
    "RMSprop": lambda p: optim.RMSprop(p, lr=0.001),
    "Adagrad": lambda p: optim.Adagrad(p, lr=0.01),
}

print("\nTraining each optimizer for 100 steps:")
print(f"{'Optimizer':<20} {'Final Loss':<12} {'Best Loss':<12}")
print("-" * 44)

for name, opt_fn in optimizers_to_test.items():
    torch.manual_seed(42)
    model = SimpleModel()
    optimizer = opt_fn(model.parameters())

    best_loss = float("inf")
    final_loss = 0
    for step in range(100):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        best_loss = min(best_loss, loss.item())
        final_loss = loss.item()

    print(f"{name:<20} {final_loss:<12.4f} {best_loss:<12.4f}")

print("\n" + "=" * 70)
print("PART 3: Parameter Groups")
print("=" * 70)

print("""
Parameter groups let you apply different hyperparameters to different
parts of the model. Common use case: lower LR for pretrained backbone,
higher LR for randomly-initialized head.
""")


class TransferModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


model = TransferModel()

# Different LR for backbone and head
optimizer = optim.AdamW(
    [
        {"params": model.backbone.parameters(), "lr": 1e-4, "weight_decay": 0.01},
        {"params": model.head.parameters(), "lr": 1e-3, "weight_decay": 0.0},
    ],
    # Default values (applied to groups without explicit settings)
    lr=1e-3,
)

print(f"Number of parameter groups: {len(optimizer.param_groups)}")
for i, group in enumerate(optimizer.param_groups):
    num_params = sum(p.numel() for p in group["params"])
    print(f"  Group {i}: lr={group['lr']}, weight_decay={group['weight_decay']}, "
          f"params={num_params:,}")

# Modify LR for a specific group during training
print("\n--- Modifying LR during training ---")
print(f"Before: group 0 lr = {optimizer.param_groups[0]['lr']}")
optimizer.param_groups[0]["lr"] = 5e-5
print(f"After:  group 0 lr = {optimizer.param_groups[0]['lr']}")

print("\n" + "=" * 70)
print("PART 4: Optimizer State")
print("=" * 70)

model = SimpleModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Before any step: state is empty
print(f"\nOptimizer state before training: {len(optimizer.state)} entries")

# After one step: state is populated
loss = criterion(model(X), y)
loss.backward()
optimizer.step()
print(f"Optimizer state after 1 step: {len(optimizer.state)} entries")

# Inspect the state of one parameter
param = list(model.parameters())[0]
state = optimizer.state[param]
print(f"\nState for layer1.weight:")
print(f"  step: {state['step']}")
print(f"  exp_avg (m_t) shape: {state['exp_avg'].shape}")
print(f"  exp_avg_sq (v_t) shape: {state['exp_avg_sq'].shape}")
print(f"  exp_avg mean: {state['exp_avg'].mean().item():.6f}")
print(f"  exp_avg_sq mean: {state['exp_avg_sq'].mean().item():.6f}")

print("\n" + "=" * 70)
print("PART 5: State Dict — Save and Resume")
print("=" * 70)

import tempfile
import os

model = SimpleModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train for a few steps
for _ in range(10):
    optimizer.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()

print(f"After 10 steps — Loss: {loss.item():.4f}")

# Save state
save_dir = tempfile.mkdtemp()
model_path = os.path.join(save_dir, "model.pth")
opt_path = os.path.join(save_dir, "optimizer.pth")

torch.save(model.state_dict(), model_path)
torch.save(optimizer.state_dict(), opt_path)

# Inspect optimizer state_dict
opt_state = optimizer.state_dict()
print(f"\nOptimizer state_dict keys: {list(opt_state.keys())}")
print(f"  Number of param states: {len(opt_state['state'])}")
print(f"  Number of param groups: {len(opt_state['param_groups'])}")
print(f"  Group 0 settings: lr={opt_state['param_groups'][0]['lr']}, "
      f"betas={opt_state['param_groups'][0]['betas']}")

# Resume training
print("\n--- Resuming training ---")
model_resumed = SimpleModel()
optimizer_resumed = optim.Adam(model_resumed.parameters(), lr=0.001)

model_resumed.load_state_dict(torch.load(model_path, weights_only=True))
optimizer_resumed.load_state_dict(torch.load(opt_path, weights_only=True))

# Continue training
for _ in range(10):
    optimizer_resumed.zero_grad()
    loss = criterion(model_resumed(X), y)
    loss.backward()
    optimizer_resumed.step()

print(f"After 10 more steps — Loss: {loss.item():.4f}")

# Verify step count continued
param = list(model_resumed.parameters())[0]
print(f"Adam step counter: {optimizer_resumed.state[param]['step']}")

# Cleanup
import shutil
shutil.rmtree(save_dir)

print("\n" + "=" * 70)
print("PART 6: zero_grad() — set_to_none vs zero")
print("=" * 70)

print("""
optimizer.zero_grad() clears gradients before backward().
Two modes:
  - set_to_none=True (default since PyTorch 2.0): Sets .grad to None
    - More memory efficient (no zero tensor stored)
    - Slightly faster
  - set_to_none=False: Fills .grad with zeros
    - Needed for gradient accumulation patterns
    - Safer with some custom autograd functions
""")

model = SimpleModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Mode 1: set_to_none=True (default)
loss = criterion(model(X), y)
loss.backward()
print(f"After backward — grad is tensor: {model.layer1.weight.grad is not None}")
optimizer.zero_grad(set_to_none=True)
print(f"After zero_grad(set_to_none=True) — grad is None: {model.layer1.weight.grad is None}")

# Mode 2: set_to_none=False
loss = criterion(model(X), y)
loss.backward()
optimizer.zero_grad(set_to_none=False)
print(f"After zero_grad(set_to_none=False) — grad is zero: "
      f"{model.layer1.weight.grad.sum().item() == 0}")

print("\n" + "=" * 70)
print("PART 7: Gradient Accumulation")
print("=" * 70)

print("""
When batch size is limited by GPU memory, you can accumulate gradients
over multiple mini-batches to simulate a larger effective batch.
""")

model = SimpleModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Simulating batch_size=100 using 5 mini-batches of 20
accumulation_steps = 5
effective_batch_size = 100
mini_batch_size = effective_batch_size // accumulation_steps

# Split data into mini-batches
X_batches = X[:effective_batch_size].split(mini_batch_size)
y_batches = y[:effective_batch_size].split(mini_batch_size)

print(f"Effective batch size: {effective_batch_size}")
print(f"Mini-batch size: {mini_batch_size}")
print(f"Accumulation steps: {accumulation_steps}")

optimizer.zero_grad()
total_loss = 0
for i, (x_batch, y_batch) in enumerate(zip(X_batches, y_batches)):
    output = model(x_batch)
    # Divide loss by accumulation steps to get correct mean
    loss = criterion(output, y_batch) / accumulation_steps
    loss.backward()  # Gradients ACCUMULATE (not zeroed between mini-batches)
    total_loss += loss.item()

# Now do the optimizer step (with accumulated gradients)
optimizer.step()
optimizer.zero_grad()
print(f"\nAccumulated loss: {total_loss:.4f}")

# Compare with single large batch
model2 = SimpleModel()
model2.load_state_dict(model.state_dict())  # Restart from same point
optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
optimizer2.zero_grad()
output = model2(X[:effective_batch_size])
loss_single = criterion(output, y[:effective_batch_size])
print(f"Single batch loss: {loss_single.item():.4f}")
print("(Slight difference due to optimizer state, but gradients should be similar)")

print("\n" + "=" * 70)
print("PART 8: Weight Decay vs L2 Regularization")
print("=" * 70)

print("""
In SGD, weight_decay and L2 regularization are equivalent:
  grad_new = grad + weight_decay * param
  param = param - lr * grad_new
  
In Adam/AdamW, they are DIFFERENT:
  - Adam + weight_decay: applies decay to gradient (scaled by adaptive LR)
  - AdamW: applies decay directly to parameters (NOT scaled)
  
AdamW is the correct way to do weight decay with adaptive optimizers.
""")

# Demonstrate the difference
model_adam = SimpleModel()
model_adamw = SimpleModel()
# Same initial weights
model_adamw.load_state_dict(model_adam.state_dict())

opt_adam = optim.Adam(model_adam.parameters(), lr=0.001, weight_decay=0.01)
opt_adamw = optim.AdamW(model_adamw.parameters(), lr=0.001, weight_decay=0.01)

# Train both for 50 steps
for _ in range(50):
    opt_adam.zero_grad()
    loss = criterion(model_adam(X), y)
    loss.backward()
    opt_adam.step()

    opt_adamw.zero_grad()
    loss = criterion(model_adamw(X), y)
    loss.backward()
    opt_adamw.step()

# Compare weight magnitudes (AdamW should have smaller weights due to proper decay)
adam_weight_norm = sum(p.norm().item() for p in model_adam.parameters())
adamw_weight_norm = sum(p.norm().item() for p in model_adamw.parameters())
print(f"\nAfter 50 steps:")
print(f"  Adam (L2 reg) weight norm:       {adam_weight_norm:.4f}")
print(f"  AdamW (decoupled WD) weight norm: {adamw_weight_norm:.4f}")
print("  AdamW typically produces smaller weights (stronger regularization)")

print("\n" + "=" * 70)
print("PART 9: LBFGS Optimizer (Second-Order)")
print("=" * 70)

print("""
LBFGS requires a closure function that re-evaluates the loss.
It's much more expensive per step but can converge faster.
Best for: small problems, scientific computing, fine-tuning.
""")

torch.manual_seed(42)
model_lbfgs = SimpleModel()
optimizer_lbfgs = optim.LBFGS(model_lbfgs.parameters(), lr=0.1, max_iter=20,
                               history_size=10)

print("Training with LBFGS:")
for step in range(5):
    def closure():
        optimizer_lbfgs.zero_grad()
        output = model_lbfgs(X)
        loss = criterion(output, y)
        loss.backward()
        return loss

    loss = optimizer_lbfgs.step(closure)
    print(f"  Step {step}: loss = {loss.item():.6f}")

print("(Notice: LBFGS converges much faster but is more expensive per step)")

print("\n" + "=" * 70)
print("PART 10: Excluding Parameters from Optimization")
print("=" * 70)

model = TransferModel()

# Method 1: Only pass specific parameters
print("\n--- Method 1: Only pass certain parameters ---")
optimizer = optim.Adam(model.head.parameters(), lr=0.001)
print(f"Optimizing only head: {sum(p.numel() for p in model.head.parameters())} params")

# Method 2: Filter by requires_grad
print("\n--- Method 2: Freeze + filter ---")
for param in model.backbone.parameters():
    param.requires_grad = False

trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(trainable, lr=0.001)
print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Method 3: Different weight decay for bias vs weights
print("\n--- Method 3: No weight decay for biases/norms ---")
model = TransferModel()
decay_params = []
no_decay_params = []
for name, param in model.named_parameters():
    if "bias" in name or "norm" in name:
        no_decay_params.append(param)
    else:
        decay_params.append(param)

optimizer = optim.AdamW([
    {"params": decay_params, "weight_decay": 0.01},
    {"params": no_decay_params, "weight_decay": 0.0},
], lr=0.001)
print(f"Params with weight decay: {sum(p.numel() for p in decay_params):,}")
print(f"Params without weight decay: {sum(p.numel() for p in no_decay_params):,}")

print("\n" + "=" * 70)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
print("=" * 70)
