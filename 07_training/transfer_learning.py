"""
Transfer Learning — Freeze/Unfreeze and Differential Learning Rates
====================================================================
Demonstrates transfer learning strategies:
1. Feature extraction (freeze backbone)
2. Fine-tuning with different learning rates
3. Progressive unfreezing

Uses a simple pretrained-like model (no actual pretrained weights needed).

Run: python transfer_learning.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict

# =============================================================================
# 1. Simulated "pretrained" backbone
# =============================================================================

class Backbone(nn.Module):
    """Simulates a pretrained feature extractor (like ResNet without final FC)."""

    def __init__(self, input_dim=64, hidden_dim=128, output_dim=64):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class TransferModel(nn.Module):
    """Model with pretrained backbone + new classification head."""

    def __init__(self, backbone, num_classes=3):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


# =============================================================================
# 2. Setup: "pretrain" the backbone on source task
# =============================================================================

torch.manual_seed(42)

print("=" * 60)
print("TRANSFER LEARNING DEMONSTRATION")
print("=" * 60)

# Pretrain backbone on a source task (10 classes)
print("\n--- Phase 0: Pretraining backbone on source task ---")
backbone = Backbone()
pretrain_head = nn.Linear(64, 10)
pretrain_model = nn.Sequential(OrderedDict([
    ('backbone', backbone),
    ('head', pretrain_head),
]))

# Source task data
source_X = torch.randn(2000, 64)
source_y = torch.randint(0, 10, (2000,))
source_loader = DataLoader(TensorDataset(source_X, source_y), batch_size=64, shuffle=True)

optimizer = optim.Adam(pretrain_model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    for inputs, targets in source_loader:
        optimizer.zero_grad()
        loss = loss_fn(pretrain_model(inputs), targets)
        loss.backward()
        optimizer.step()

print(f"  Pretrained for 10 epochs, final loss: {loss.item():.4f}")

# Target task data (smaller, different number of classes)
num_classes = 3
target_X = torch.randn(300, 64)
target_y = torch.randint(0, num_classes, (300,))
train_X, val_X = target_X[:240], target_X[240:]
train_y, val_y = target_y[:240], target_y[240:]
target_train = DataLoader(TensorDataset(train_X, train_y), batch_size=32, shuffle=True)
target_val = DataLoader(TensorDataset(val_X, val_y), batch_size=60)


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            preds = model(inputs).argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / total


# =============================================================================
# 3. Strategy 1: Feature Extraction (Freeze Backbone)
# =============================================================================

print("\n" + "=" * 60)
print("STRATEGY 1: FEATURE EXTRACTION (Freeze Backbone)")
print("=" * 60)

# Create transfer model with frozen backbone
model_frozen = TransferModel(Backbone(), num_classes=num_classes)
model_frozen.backbone.load_state_dict(backbone.state_dict())

# Freeze all backbone parameters
for param in model_frozen.backbone.parameters():
    param.requires_grad = False

# Count trainable vs total parameters
total_params = sum(p.numel() for p in model_frozen.parameters())
trainable_params = sum(p.numel() for p in model_frozen.parameters() if p.requires_grad)
print(f"\n  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

# Only optimize head parameters
optimizer_frozen = optim.Adam(
    filter(lambda p: p.requires_grad, model_frozen.parameters()),
    lr=1e-3
)

model_frozen.train()
for epoch in range(15):
    for inputs, targets in target_train:
        optimizer_frozen.zero_grad()
        loss = loss_fn(model_frozen(inputs), targets)
        loss.backward()
        optimizer_frozen.step()

acc = evaluate(model_frozen, target_val)
print(f"  After 15 epochs (head only): val accuracy = {acc:.3f}")

# Verify backbone didn't change
backbone_changed = False
for (name, p_orig), (_, p_new) in zip(
    backbone.named_parameters(), model_frozen.backbone.named_parameters()
):
    if not torch.equal(p_orig, p_new):
        backbone_changed = True
        break
print(f"  Backbone weights changed: {backbone_changed}")

# =============================================================================
# 4. Strategy 2: Differential Learning Rates
# =============================================================================

print("\n" + "=" * 60)
print("STRATEGY 2: DIFFERENTIAL LEARNING RATES")
print("=" * 60)
print("  Lower LR for backbone (preserve knowledge)")
print("  Higher LR for head (learn new task fast)\n")

model_diff_lr = TransferModel(Backbone(), num_classes=num_classes)
model_diff_lr.backbone.load_state_dict(backbone.state_dict())

# Unfreeze everything
for param in model_diff_lr.parameters():
    param.requires_grad = True

# Different LR per parameter group
optimizer_diff = optim.Adam([
    {'params': model_diff_lr.backbone.layer1.parameters(), 'lr': 1e-5},
    {'params': model_diff_lr.backbone.layer2.parameters(), 'lr': 1e-5},
    {'params': model_diff_lr.backbone.layer3.parameters(), 'lr': 5e-5},
    {'params': model_diff_lr.backbone.layer4.parameters(), 'lr': 1e-4},
    {'params': model_diff_lr.head.parameters(), 'lr': 1e-3},
])

print("  Learning rates:")
for i, group in enumerate(optimizer_diff.param_groups):
    num_params = sum(p.numel() for p in group['params'])
    print(f"    Group {i}: lr={group['lr']:.1e}, params={num_params:,}")

model_diff_lr.train()
for epoch in range(15):
    for inputs, targets in target_train:
        optimizer_diff.zero_grad()
        loss = loss_fn(model_diff_lr(inputs), targets)
        loss.backward()
        optimizer_diff.step()

acc = evaluate(model_diff_lr, target_val)
print(f"\n  After 15 epochs (diff LR): val accuracy = {acc:.3f}")

# =============================================================================
# 5. Strategy 3: Progressive Unfreezing
# =============================================================================

print("\n" + "=" * 60)
print("STRATEGY 3: PROGRESSIVE UNFREEZING")
print("=" * 60)
print("  Start frozen, gradually unfreeze from top to bottom\n")

model_prog = TransferModel(Backbone(), num_classes=num_classes)
model_prog.backbone.load_state_dict(backbone.state_dict())

# Start fully frozen
for param in model_prog.backbone.parameters():
    param.requires_grad = False

layer_groups = [
    ('head', model_prog.head),
    ('backbone.layer4', model_prog.backbone.layer4),
    ('backbone.layer3', model_prog.backbone.layer3),
    ('backbone.layer2', model_prog.backbone.layer2),
    ('backbone.layer1', model_prog.backbone.layer1),
]

epochs_per_phase = 5

for phase, (name, layer_group) in enumerate(layer_groups):
    # Unfreeze this layer group
    for param in layer_group.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model_prog.parameters() if p.requires_grad)
    print(f"  Phase {phase}: Unfreezing '{name}' | Trainable params: {trainable:,}")

    # Create optimizer with currently trainable params
    optimizer_prog = optim.Adam(
        filter(lambda p: p.requires_grad, model_prog.parameters()),
        lr=1e-3 * (0.5 ** phase),  # Decrease LR as we unfreeze more
    )

    model_prog.train()
    for epoch in range(epochs_per_phase):
        for inputs, targets in target_train:
            optimizer_prog.zero_grad()
            loss = loss_fn(model_prog(inputs), targets)
            loss.backward()
            optimizer_prog.step()

    acc = evaluate(model_prog, target_val)
    print(f"    After {epochs_per_phase} epochs: val acc = {acc:.3f}, "
          f"lr = {1e-3 * (0.5 ** phase):.1e}")

# =============================================================================
# 6. Strategy comparison
# =============================================================================

print("\n" + "=" * 60)
print("STRATEGY COMPARISON: Training from scratch vs transfer")
print("=" * 60)

# From scratch (no pretrained weights)
model_scratch = TransferModel(Backbone(), num_classes=num_classes)
optimizer_scratch = optim.Adam(model_scratch.parameters(), lr=1e-3)

model_scratch.train()
for epoch in range(15):
    for inputs, targets in target_train:
        optimizer_scratch.zero_grad()
        loss = loss_fn(model_scratch(inputs), targets)
        loss.backward()
        optimizer_scratch.step()

acc_scratch = evaluate(model_scratch, target_val)
acc_frozen = evaluate(model_frozen, target_val)
acc_diff = evaluate(model_diff_lr, target_val)
acc_prog = evaluate(model_prog, target_val)

print(f"\n  From scratch (15 epochs):     {acc_scratch:.3f}")
print(f"  Frozen backbone (15 epochs):  {acc_frozen:.3f}")
print(f"  Differential LR (15 epochs):  {acc_diff:.3f}")
print(f"  Progressive unfreeze (25 ep): {acc_prog:.3f}")

# =============================================================================
# 7. Useful utilities for transfer learning
# =============================================================================

print("\n" + "=" * 60)
print("UTILITIES FOR TRANSFER LEARNING")
print("=" * 60)

# Listing which parameters are frozen/unfrozen
print("\n  Parameter freeze status:")
for name, param in model_frozen.named_parameters():
    status = "TRAINABLE" if param.requires_grad else "FROZEN"
    if "layer1.0.weight" in name or "head.0.weight" in name:
        print(f"    {name}: {status} (shape={list(param.shape)})")

# Selectively freezing by name pattern
print("\n  Freeze by pattern (e.g., freeze all BatchNorm):")
bn_frozen = 0
for name, param in model_diff_lr.named_parameters():
    if 'bn' in name or 'BatchNorm' in name or '.1.' in name:
        # In our Sequential, index 1 is BatchNorm
        param.requires_grad = False
        bn_frozen += 1

print(f"    Froze {bn_frozen} BatchNorm parameters")

print("\nTransfer learning demonstration complete!")
