"""
Complete Training Loop — Best Practices Template
==================================================
Covers: training/eval loops, AMP, gradient clipping, LR scheduling, checkpointing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

print("=" * 60)
print("COMPLETE TRAINING LOOP EXAMPLE")
print("=" * 60)

# --- Config ---
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
USE_AMP = torch.cuda.is_available()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {DEVICE}")
print(f"AMP: {USE_AMP}")

# --- Synthetic Dataset ---
N_TRAIN, N_VAL = 2000, 500
INPUT_DIM, NUM_CLASSES = 100, 10

train_x = torch.randn(N_TRAIN, INPUT_DIM)
train_y = torch.randint(0, NUM_CLASSES, (N_TRAIN,))
val_x = torch.randn(N_VAL, INPUT_DIM)
val_y = torch.randint(0, NUM_CLASSES, (N_VAL,))

train_loader = DataLoader(
    TensorDataset(train_x, train_y),
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)
val_loader = DataLoader(
    TensorDataset(val_x, val_y),
    batch_size=BATCH_SIZE
)

# --- Model ---
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)

model = MLP(INPUT_DIM, 256, NUM_CLASSES).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# --- Optimizer & Scheduler ---
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LR,
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS
)
scaler = torch.amp.GradScaler('cuda') if USE_AMP else None

# --- Training Function ---
def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                output = model(data)
                loss = F.cross_entropy(output, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        scheduler.step()

        total_loss += loss.item() * data.size(0)
        correct += output.argmax(1).eq(target).sum().item()
        total += data.size(0)

    return total_loss / total, 100.0 * correct / total

# --- Evaluation Function ---
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.cross_entropy(output, target)
        total_loss += loss.item() * data.size(0)
        correct += output.argmax(1).eq(target).sum().item()
        total += data.size(0)

    return total_loss / total, 100.0 * correct / total

# --- Training Loop ---
print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7} | {'LR':>10}")
print("-" * 65)

best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, optimizer, scheduler, scaler, DEVICE, epoch
    )
    val_loss, val_acc = evaluate(model, val_loader, DEVICE)

    lr = optimizer.param_groups[0]['lr']
    print(f"{epoch:5d} | {train_loss:10.4f} | {train_acc:8.2f}% | {val_loss:8.4f} | {val_acc:6.2f}% | {lr:10.6f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc

print(f"\nBest validation accuracy: {best_val_acc:.2f}%")

# --- Gradient Accumulation Example ---
print("\n" + "=" * 60)
print("GRADIENT ACCUMULATION")
print("=" * 60)

ACCUMULATION_STEPS = 4
model2 = MLP(INPUT_DIM, 256, NUM_CLASSES).to(DEVICE)
optimizer2 = torch.optim.AdamW(model2.parameters(), lr=LR)

model2.train()
optimizer2.zero_grad(set_to_none=True)

for i, (data, target) in enumerate(train_loader):
    data, target = data.to(DEVICE), target.to(DEVICE)
    loss = F.cross_entropy(model2(data), target) / ACCUMULATION_STEPS
    loss.backward()

    if (i + 1) % ACCUMULATION_STEPS == 0:
        nn.utils.clip_grad_norm_(model2.parameters(), GRAD_CLIP)
        optimizer2.step()
        optimizer2.zero_grad(set_to_none=True)

    if i >= 7:
        break

print(f"Effective batch size: {BATCH_SIZE * ACCUMULATION_STEPS}")
print("Gradient accumulation complete")

print("\nDone!")
