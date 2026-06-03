"""
Regularization Techniques — EMA, Label Smoothing, Weight Decay
===============================================================
Demonstrates:
1. EMA (Exponential Moving Average)
2. Label Smoothing
3. Weight Decay comparison (L2 reg vs decoupled)
4. Early Stopping

Run: python regularization.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy

# =============================================================================
# Setup
# =============================================================================

torch.manual_seed(42)

# Synthetic classification problem
num_samples = 1500
input_dim = 50
num_classes = 5

X = torch.randn(num_samples, input_dim)
W_true = torch.randn(input_dim, num_classes)
logits_true = X @ W_true
y = logits_true.argmax(dim=1)
# Add noise to make it challenging
noise_mask = torch.rand(num_samples) < 0.1
y[noise_mask] = torch.randint(0, num_classes, (noise_mask.sum(),))

train_X, val_X = X[:1200], X[1200:]
train_y, val_y = y[:1200], y[1200:]
train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(val_X, val_y), batch_size=300)


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, targets in loader:
            out = model(inputs)
            total_loss += loss_fn(out, targets).item() * inputs.size(0)
            correct += (out.argmax(1) == targets).sum().item()
            total += targets.size(0)
    return correct / total, total_loss / total


# =============================================================================
# 1. EXPONENTIAL MOVING AVERAGE (EMA)
# =============================================================================

print("=" * 60)
print("1. EXPONENTIAL MOVING AVERAGE (EMA)")
print("=" * 60)
print("\nEMA maintains a smoothed copy of model weights.")
print("Formula: ema_param = decay * ema_param + (1-decay) * param\n")


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        # Store shadow copies of all parameters
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        """Update shadow parameters with exponential moving average."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                # shadow = decay * shadow + (1 - decay) * current
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self, model):
        """Replace model params with EMA params (for evaluation)."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore original model params (after evaluation)."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


# Train with EMA
model_ema = SimpleNet()
optimizer_ema = optim.Adam(model_ema.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
ema = EMA(model_ema, decay=0.99)

model_ema.train()
for epoch in range(30):
    for inputs, targets in train_loader:
        optimizer_ema.zero_grad()
        loss = loss_fn(model_ema(inputs), targets)
        loss.backward()
        optimizer_ema.step()
        ema.update(model_ema)  # Update EMA after each step

# Compare: original model vs EMA model
acc_orig, _ = evaluate(model_ema, val_loader)

ema.apply_shadow(model_ema)  # Temporarily use EMA weights
acc_ema, _ = evaluate(model_ema, val_loader)
ema.restore(model_ema)  # Restore original weights

print(f"  Original model accuracy: {acc_orig:.3f}")
print(f"  EMA model accuracy:      {acc_ema:.3f}")
print(f"  EMA {'improves' if acc_ema > acc_orig else 'same as'} accuracy")

# Show how EMA smooths weights
print(f"\n  Weight comparison (first layer, first 5 values):")
orig_w = list(model_ema.parameters())[0].data[0, :5]
ema_w = ema.shadow[list(ema.shadow.keys())[0]][0, :5]
print(f"    Original: {[f'{v:.4f}' for v in orig_w.tolist()]}")
print(f"    EMA:      {[f'{v:.4f}' for v in ema_w.tolist()]}")

# =============================================================================
# 2. LABEL SMOOTHING
# =============================================================================

print("\n" + "=" * 60)
print("2. LABEL SMOOTHING")
print("=" * 60)
print("\nSoftens hard targets to prevent overconfidence.")
print("smooth_target = (1 - eps) * one_hot + eps / num_classes\n")

# Compare training with and without label smoothing
results = {}

for smoothing in [0.0, 0.1, 0.2]:
    model_ls = SimpleNet()
    optimizer_ls = optim.Adam(model_ls.parameters(), lr=1e-3)
    # PyTorch has built-in label smoothing support!
    loss_fn_smooth = nn.CrossEntropyLoss(label_smoothing=smoothing)

    model_ls.train()
    for epoch in range(30):
        for inputs, targets in train_loader:
            optimizer_ls.zero_grad()
            loss = loss_fn_smooth(model_ls(inputs), targets)
            loss.backward()
            optimizer_ls.step()

    train_acc, train_loss = evaluate(model_ls, train_loader)
    val_acc, val_loss = evaluate(model_ls, val_loader)
    results[smoothing] = (train_acc, val_acc, val_loss)

    print(f"  Smoothing={smoothing:.1f}: "
          f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}, val_loss={val_loss:.4f}")

# Show what label smoothing does to the target distribution
print("\n  What the targets look like (5 classes, true class=2):")
one_hot = torch.zeros(num_classes)
one_hot[2] = 1.0
print(f"    Hard target:         {one_hot.tolist()}")

smoothing = 0.1
smooth = one_hot * (1 - smoothing) + smoothing / num_classes
print(f"    Smoothed (eps=0.1):  {[f'{v:.3f}' for v in smooth.tolist()]}")

smoothing = 0.2
smooth = one_hot * (1 - smoothing) + smoothing / num_classes
print(f"    Smoothed (eps=0.2):  {[f'{v:.3f}' for v in smooth.tolist()]}")

# Show confidence calibration effect
print("\n  Effect on prediction confidence:")
model_hard = SimpleNet()
model_smooth = SimpleNet()
model_smooth.load_state_dict(model_hard.state_dict())

opt_h = optim.Adam(model_hard.parameters(), lr=1e-3)
opt_s = optim.Adam(model_smooth.parameters(), lr=1e-3)
loss_hard = nn.CrossEntropyLoss(label_smoothing=0.0)
loss_smooth = nn.CrossEntropyLoss(label_smoothing=0.1)

for epoch in range(30):
    for inputs, targets in train_loader:
        opt_h.zero_grad()
        loss_hard(model_hard(inputs), targets).backward()
        opt_h.step()

        opt_s.zero_grad()
        loss_smooth(model_smooth(inputs), targets).backward()
        opt_s.step()

model_hard.eval()
model_smooth.eval()
with torch.no_grad():
    probs_hard = torch.softmax(model_hard(val_X), dim=1)
    probs_smooth = torch.softmax(model_smooth(val_X), dim=1)

print(f"    Without smoothing — max prob (mean): {probs_hard.max(dim=1).values.mean():.3f}")
print(f"    With smoothing    — max prob (mean): {probs_smooth.max(dim=1).values.mean():.3f}")
print(f"    Label smoothing reduces overconfidence!")

# =============================================================================
# 3. WEIGHT DECAY COMPARISON
# =============================================================================

print("\n" + "=" * 60)
print("3. WEIGHT DECAY: L2 Regularization vs Decoupled")
print("=" * 60)
print("\nL2 reg (SGD): adds lambda*w to gradient → same as weight_decay in SGD")
print("Decoupled (AdamW): directly shrinks weights → better for Adam\n")

# Compare: Adam + L2 vs AdamW (decoupled weight decay)
configs = [
    ("Adam (no decay)", optim.Adam, {'lr': 1e-3, 'weight_decay': 0}),
    ("Adam (L2 decay=0.01)", optim.Adam, {'lr': 1e-3, 'weight_decay': 0.01}),
    ("AdamW (decoupled=0.01)", optim.AdamW, {'lr': 1e-3, 'weight_decay': 0.01}),
    ("AdamW (decoupled=0.1)", optim.AdamW, {'lr': 1e-3, 'weight_decay': 0.1}),
]

for name, opt_class, opt_kwargs in configs:
    model_wd = SimpleNet()
    optimizer_wd = opt_class(model_wd.parameters(), **opt_kwargs)

    model_wd.train()
    for epoch in range(30):
        for inputs, targets in train_loader:
            optimizer_wd.zero_grad()
            loss = loss_fn(model_wd(inputs), targets)
            loss.backward()
            optimizer_wd.step()

    train_acc, _ = evaluate(model_wd, train_loader)
    val_acc, _ = evaluate(model_wd, val_loader)

    # Measure weight magnitudes (weight decay shrinks weights)
    weight_norm = sum(p.norm().item() ** 2 for p in model_wd.parameters()) ** 0.5

    print(f"  {name:30s}: train={train_acc:.3f}, val={val_acc:.3f}, "
          f"|W|={weight_norm:.2f}")

print("\n  Note: AdamW (decoupled) applies decay independently of gradient,")
print("  which is generally better for adaptive optimizers like Adam.")

# =============================================================================
# 4. EARLY STOPPING
# =============================================================================

print("\n" + "=" * 60)
print("4. EARLY STOPPING")
print("=" * 60)
print("\nStop training when validation loss stops improving.\n")


class EarlyStopping:
    """Stop training when validation loss doesn't improve for `patience` epochs."""

    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.should_stop = False

    def __call__(self, val_loss, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# Train with early stopping
model_es = SimpleNet()
optimizer_es = optim.Adam(model_es.parameters(), lr=1e-3)
early_stopping = EarlyStopping(patience=5, min_delta=0.001)
best_model_state = None

max_epochs = 100
for epoch in range(max_epochs):
    model_es.train()
    for inputs, targets in train_loader:
        optimizer_es.zero_grad()
        loss = loss_fn(model_es(inputs), targets)
        loss.backward()
        optimizer_es.step()

    _, val_loss = evaluate(model_es, val_loader)

    # Save best model
    if val_loss <= early_stopping.best_loss:
        best_model_state = copy.deepcopy(model_es.state_dict())

    if early_stopping(val_loss, epoch):
        print(f"  Early stopping triggered at epoch {epoch + 1}!")
        print(f"  Best epoch was {early_stopping.best_epoch + 1} "
              f"with val_loss={early_stopping.best_loss:.4f}")
        break

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}: val_loss={val_loss:.4f}, "
              f"patience_counter={early_stopping.counter}")
else:
    print(f"  Trained full {max_epochs} epochs without early stopping")

# Load best model
if best_model_state is not None:
    model_es.load_state_dict(best_model_state)
    acc_best, _ = evaluate(model_es, val_loader)
    print(f"  Best model accuracy: {acc_best:.3f}")

# =============================================================================
# 5. COMBINING REGULARIZATION TECHNIQUES
# =============================================================================

print("\n" + "=" * 60)
print("5. COMBINING TECHNIQUES")
print("=" * 60)
print("\nUsing EMA + Label Smoothing + Weight Decay + Early Stopping\n")

model_combo = SimpleNet()
optimizer_combo = optim.AdamW(model_combo.parameters(), lr=1e-3, weight_decay=0.01)
loss_fn_combo = nn.CrossEntropyLoss(label_smoothing=0.1)
ema_combo = EMA(model_combo, decay=0.99)
early_stop_combo = EarlyStopping(patience=7)
best_state = None

for epoch in range(100):
    model_combo.train()
    for inputs, targets in train_loader:
        optimizer_combo.zero_grad()
        loss = loss_fn_combo(model_combo(inputs), targets)
        loss.backward()
        optimizer_combo.step()
        ema_combo.update(model_combo)

    # Evaluate with EMA weights
    ema_combo.apply_shadow(model_combo)
    _, val_loss = evaluate(model_combo, val_loader)
    ema_combo.restore(model_combo)

    if val_loss <= early_stop_combo.best_loss:
        best_state = {k: v.clone() for k, v in ema_combo.shadow.items()}

    if early_stop_combo(val_loss, epoch):
        print(f"  Stopped at epoch {epoch + 1} (best: {early_stop_combo.best_epoch + 1})")
        break

# Apply best EMA weights
if best_state is not None:
    for name, param in model_combo.named_parameters():
        if name in best_state:
            param.data.copy_(best_state[name])

acc_combo, _ = evaluate(model_combo, val_loader)
print(f"  Combined technique accuracy: {acc_combo:.3f}")

# Baseline without any regularization
model_base = SimpleNet()
opt_base = optim.Adam(model_base.parameters(), lr=1e-3)
for epoch in range(30):
    model_base.train()
    for inputs, targets in train_loader:
        opt_base.zero_grad()
        nn.CrossEntropyLoss()(model_base(inputs), targets).backward()
        opt_base.step()

acc_base, _ = evaluate(model_base, val_loader)
print(f"  Baseline (no regularization): {acc_base:.3f}")
print(f"  Improvement: {acc_combo - acc_base:+.3f}")

print("\nRegularization techniques demonstration complete!")
