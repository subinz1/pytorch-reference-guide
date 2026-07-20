"""
Module 40 — Image Classifier: Model & Training
================================================

SimpleCNN, ResNet-style model, transfer learning setup, and training loop
with AMP, learning rate scheduling, early stopping, and MixUp/CutMix.

Usage:
    python model_and_training.py

Requirements:
    pip install torch torchvision
"""

import math
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_pipeline import (
    NUM_CLASSES,
    CLASS_NAMES,
    build_dataloaders,
    mixup,
    cutmix,
    mixup_criterion,
)


# ---------------------------------------------------------------------------
# 1. SimpleCNN — Baseline Model
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """A lightweight CNN for small image classification tasks."""

    def __init__(self, in_channels: int = 3, num_classes: int = NUM_CLASSES, img_size: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# 2. ResNet-style Model (Mini ResNet)
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.shortcut(x), inplace=True)
        return out


class MiniResNet(nn.Module):
    """Simplified ResNet suitable for 32x32 images (like CIFAR)."""

    def __init__(self, num_blocks: list[int] = [2, 2, 2], num_classes: int = NUM_CLASSES):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

        self._init_weights()

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(planes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# 3. Transfer Learning Wrapper
# ---------------------------------------------------------------------------

class TransferModel(nn.Module):
    """Wraps a pretrained backbone and replaces the classifier head.

    This demonstrates the pattern; actual pretrained weights require torchvision
    and internet access, so we simulate with random initialization here.
    """

    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int = NUM_CLASSES, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad() if not any(p.requires_grad for p in self.backbone.parameters()) else nullcontext():
            features = self.backbone(x)
        return self.head(features)

    def unfreeze_backbone(self, lr_factor: float = 0.1):
        """Unfreeze backbone for fine-tuning with a lower learning rate."""
        for p in self.backbone.parameters():
            p.requires_grad = True
        return [
            {"params": self.backbone.parameters(), "lr": lr_factor},
            {"params": self.head.parameters()},
        ]


def create_transfer_model(num_classes: int = NUM_CLASSES) -> TransferModel:
    """Create a transfer learning model using MiniResNet as backbone."""
    backbone_full = MiniResNet(num_classes=num_classes)
    backbone = nn.Sequential(
        backbone_full.conv1, backbone_full.bn1, nn.ReLU(inplace=True),
        backbone_full.layer1, backbone_full.layer2, backbone_full.layer3,
        backbone_full.avgpool, nn.Flatten(),
    )
    return TransferModel(backbone, feature_dim=256, num_classes=num_classes)


# ---------------------------------------------------------------------------
# 4. Label Smoothing Cross-Entropy
# ---------------------------------------------------------------------------

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(pred, dim=-1)
        nll_loss = F.nll_loss(log_probs, target, reduction="none")
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


# ---------------------------------------------------------------------------
# 5. Cosine Annealing with Warmup
# ---------------------------------------------------------------------------

class CosineWarmupScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, min_lr: float = 1e-6):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            factor = self.last_epoch / max(1, self.warmup_epochs)
            return [base_lr * factor for base_lr in self.base_lrs]
        progress = (self.last_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return [self.min_lr + (base_lr - self.min_lr) * cosine for base_lr in self.base_lrs]


# ---------------------------------------------------------------------------
# 6. EMA (Exponential Moving Average)
# ---------------------------------------------------------------------------

class EMA:
    """Maintains exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {name: p.clone().detach() for name, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self, model: nn.Module):
        """Replace model params with EMA values. Returns backup for restore."""
        backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name])
        return backup

    def restore(self, model: nn.Module, backup: dict):
        for name, p in model.named_parameters():
            if name in backup:
                p.data.copy_(backup[name])


# ---------------------------------------------------------------------------
# 7. Training Loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    use_mixup: bool = False,
    mixup_alpha: float = 0.2,
    ema: EMA | None = None,
    grad_clip: float = 1.0,
) -> dict:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if use_mixup and torch.rand(1).item() > 0.5:
            if torch.rand(1).item() > 0.5:
                images, y_a, y_b, lam = mixup(images, labels, mixup_alpha)
            else:
                images, y_a, y_b, lam = cutmix(images, labels, 1.0)
            use_mixed_loss = True
        else:
            use_mixed_loss = False

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(images)
            if use_mixed_loss:
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            else:
                loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * images.size(0)
        if not use_mixed_loss:
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / max(1, total)
    return {"loss": avg_loss, "accuracy": accuracy}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {"loss": total_loss / total, "accuracy": correct / total}


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    use_amp: bool = False,
    use_mixup: bool = True,
    use_ema: bool = True,
    patience: int = 5,
    device: torch.device | None = None,
) -> dict:
    """Full training pipeline with scheduler, AMP, MixUp, EMA, early stopping."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=3, total_epochs=epochs)
    scaler = torch.amp.GradScaler() if use_amp and device.type == "cuda" else None
    ema = EMA(model, decay=0.999) if use_ema else None

    best_val_acc = 0.0
    best_state = None
    no_improve = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    print(f"\nTraining on {device} | AMP={use_amp} | MixUp={use_mixup} | EMA={use_ema}")
    print(f"{'Epoch':>5} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>9} {'Val Acc':>8} {'LR':>10}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            use_amp=use_amp, scaler=scaler, use_mixup=use_mixup, ema=ema,
        )

        if ema is not None:
            backup = ema.apply(model)
            val_metrics = evaluate(model, val_loader, criterion, device)
            ema.restore(model, backup)
        else:
            val_metrics = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["lr"].append(current_lr)

        elapsed = time.time() - t0
        print(f"{epoch:5d} {train_metrics['loss']:11.4f} {train_metrics['accuracy']:10.4f} "
              f"{val_metrics['loss']:9.4f} {val_metrics['accuracy']:8.4f} {current_lr:10.6f}")

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    return history


# ---------------------------------------------------------------------------
# 8. Model Summary
# ---------------------------------------------------------------------------

def model_summary(model: nn.Module, input_size: tuple = (1, 3, 32, 32)) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Frozen parameters:    {total_params - trainable:,}")
    x = torch.randn(*input_size)
    with torch.no_grad():
        out = model(x)
    print(f"  Input shape:          {list(input_size)}")
    print(f"  Output shape:         {list(out.shape)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("MODULE 40 — IMAGE CLASSIFIER: MODEL & TRAINING")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Model summaries ---
    print("\n1. SimpleCNN")
    print("-" * 40)
    cnn = SimpleCNN()
    model_summary(cnn)

    print("\n2. MiniResNet")
    print("-" * 40)
    resnet = MiniResNet()
    model_summary(resnet)

    print("\n3. Transfer Learning Model")
    print("-" * 40)
    transfer = create_transfer_model()
    model_summary(transfer)

    # --- Train SimpleCNN ---
    print("\n4. Training SimpleCNN")
    print("-" * 40)
    torch.manual_seed(42)
    train_loader, val_loader, test_loader = build_dataloaders(
        num_train=2000, num_val=300, num_test=300, batch_size=64,
    )
    model = SimpleCNN()
    history = train(
        model, train_loader, val_loader,
        epochs=10, lr=3e-3, use_amp=False, use_mixup=True, use_ema=True, patience=8,
        device=device,
    )

    # --- Test set evaluation ---
    print("\n5. Test Set Evaluation")
    print("-" * 40)
    criterion = LabelSmoothingCrossEntropy()
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"  Test Loss:     {test_metrics['loss']:.4f}")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")

    # --- torch.compile ---
    print("\n6. torch.compile")
    print("-" * 40)
    try:
        compiled_model = torch.compile(model, mode="reduce-overhead")
        dummy = torch.randn(1, 3, 32, 32, device=device)
        with torch.no_grad():
            _ = compiled_model(dummy)
        print("  torch.compile succeeded")
    except Exception as e:
        print(f"  torch.compile skipped: {e}")

    # --- Save model ---
    print("\n7. Save/Load Model")
    print("-" * 40)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "num_classes": NUM_CLASSES,
        "history": history,
    }
    path = "/tmp/image_classifier_checkpoint.pt"
    torch.save(checkpoint, path)
    print(f"  Saved checkpoint to {path}")
    loaded = torch.load(path, weights_only=True)
    print(f"  Loaded keys: {list(loaded.keys())}")

    print("\n" + "=" * 70)
    print("MODEL & TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
