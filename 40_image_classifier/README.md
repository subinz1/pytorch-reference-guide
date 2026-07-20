# Module 40: Building an Image Classifier — End-to-End Computer Vision Project

Build a complete image classification system from scratch: data pipeline with augmentation, CNN and ResNet models, transfer learning, training with mixed precision, evaluation metrics, test-time augmentation, and Grad-CAM visualization.

**No pretrained weights required** — every component is implemented from scratch so you understand the full pipeline.

| Input | Output |
|-------|--------|
| `32x32 RGB image of a circle` | `circle (0.95)` |
| `32x32 RGB image of a star` | `star (0.91)` |
| `32x32 RGB image of a triangle` | `triangle (0.88)` |

---

## Table of Contents

1. [Overview](#overview)
2. [Data Pipeline](#data-pipeline)
3. [Data Augmentation](#data-augmentation)
4. [MixUp and CutMix](#mixup-and-cutmix)
5. [CNN Architecture](#cnn-architecture)
6. [ResNet Architecture](#resnet-architecture)
7. [Transfer Learning](#transfer-learning)
8. [Training Pipeline](#training-pipeline)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Test-Time Augmentation](#test-time-augmentation)
11. [Grad-CAM Visualization](#grad-cam-visualization)
12. [Confidence Calibration](#confidence-calibration)
13. [Inference Pipeline](#inference-pipeline)
14. [Key Takeaways](#key-takeaways)

---

## Overview

Image classification is the canonical computer vision task: given an image, assign it to one of N categories. This module builds the full pipeline:

```
Raw Images → Augmentation → CNN/ResNet → Training (AMP + MixUp) → Evaluation → TTA → Grad-CAM
```

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        IMAGE CLASSIFICATION PIPELINE                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────┐    ┌──────────────┐    ┌────────────┐    ┌───────────┐  │
│  │   Data     │    │ Augmentation │    │   Model    │    │  Training │  │
│  │ Pipeline   │───▶│  Transform   │───▶│  (CNN /   │───▶│   Loop    │  │
│  │ (Synthetic)│    │  MixUp/Cut   │    │  ResNet)  │    │ AMP + EMA │  │
│  └───────────┘    └──────────────┘    └────────────┘    └─────┬─────┘  │
│                                                               │         │
│  ┌───────────┐    ┌──────────────┐    ┌────────────┐    ┌─────▼─────┐  │
│  │ Inference  │    │   Grad-CAM   │    │    TTA     │    │ Evaluation│  │
│  │ Pipeline   │◀───│ Visualization│◀───│  (5-aug)  │◀───│  Metrics  │  │
│  └───────────┘    └──────────────┘    └────────────┘    └───────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Files

| File | Lines | Description |
|------|-------|-------------|
| `data_pipeline.py` | 270+ | Synthetic dataset, augmentation, MixUp, CutMix |
| `model_and_training.py` | 310+ | SimpleCNN, MiniResNet, transfer learning, training loop |
| `evaluation.py` | 270+ | Metrics, TTA, Grad-CAM, confidence analysis, inference |

---

## Data Pipeline

### Synthetic Shape Dataset

We generate synthetic images with 10 geometric shape classes. This lets us train and demonstrate the full pipeline without downloading large datasets.

```python
CLASS_NAMES = [
    "circle", "square", "triangle", "cross", "diamond",
    "star", "ring", "arrow", "pentagon", "hexagon",
]
```

Each image is a 3-channel (RGB) tensor with:
- A random background intensity (0.0–0.3)
- A shape drawn with a foreground intensity (0.5–1.0)
- Gaussian noise for realism

```python
class SyntheticShapeDataset(Dataset):
    def __init__(self, num_samples=5000, img_size=32, channels=3, transform=None, seed=42):
        ...

    def __getitem__(self, idx):
        # 1. Create blank image with random background
        img = torch.full((self.channels, self.img_size, self.img_size), bg)
        # 2. Draw the shape (circle, square, etc.)
        _DRAW_FNS[label](img, cx, cy, r, fg)
        # 3. Add noise
        img = (img + torch.randn_like(img) * 0.05).clamp(0, 1)
        # 4. Apply transforms
        if self.transform:
            img = self.transform(img)
        return img, label
```

### Data Splits

Standard three-way split with separate seeds for reproducibility:

| Split | Samples | Purpose | Augmentation |
|-------|---------|---------|-------------|
| Train | 4,000 | Weight updates | Full augmentation |
| Val | 500 | Hyperparameter tuning, early stopping | Normalize only |
| Test | 500 | Final evaluation | Normalize only |

```python
train_loader, val_loader, test_loader = build_dataloaders(
    num_train=4000, num_val=500, num_test=500,
    img_size=32, batch_size=64,
)
```

---

## Data Augmentation

Data augmentation artificially increases training set diversity by applying random transformations. This reduces overfitting and improves generalization.

### Why Augmentation Works

A model that sees many variations of the same shape learns invariances:
- **Flip invariance** → recognizes shapes regardless of orientation
- **Color invariance** → focuses on shape, not brightness
- **Occlusion robustness** → handles partially visible objects

### Implemented Transforms

| Transform | Parameters | Effect |
|-----------|-----------|--------|
| `RandomHorizontalFlip` | `p=0.5` | Mirror left-right |
| `RandomVerticalFlip` | `p=0.3` | Mirror top-bottom |
| `RandomRotation90` | — | Rotate 0°/90°/180°/270° |
| `ColorJitter` | `brightness=0.2, contrast=0.2` | Vary brightness and contrast |
| `RandomErasing` | `p=0.3, scale=(0.02, 0.15)` | Cutout-style occlusion |
| `Normalize` | `mean=0.5, std=0.5` | Center to [-1, 1] range |

### Augmentation Pipeline

```python
train_transform = Compose([
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.3),
    RandomRotation90(),
    ColorJitter(brightness=0.2, contrast=0.2),
    RandomErasing(p=0.3),
    Normalize(),
])

val_transform = Compose([
    Normalize(),  # No augmentation for validation/test
])
```

### Pure PyTorch Transforms

All transforms are implemented using pure PyTorch tensor operations — no torchvision dependency:

```python
class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return img.flip(-1)  # Flip width dimension
        return img


class ColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, img):
        b_factor = 1.0 + (random.random() * 2 - 1) * self.brightness
        c_factor = 1.0 + (random.random() * 2 - 1) * self.contrast
        mean = img.mean()
        img = (img - mean) * c_factor + mean  # Contrast around mean
        img = img * b_factor                   # Brightness scaling
        return img.clamp(0, 1)
```

### Random Erasing (Cutout)

Random erasing occludes a rectangular region, forcing the model to use all spatial regions rather than relying on a single discriminative patch:

```python
class RandomErasing:
    def __init__(self, p=0.3, scale=(0.02, 0.15)):
        self.p = p
        self.scale = scale

    def __call__(self, img):
        if random.random() > self.p:
            return img
        C, H, W = img.shape
        area = H * W
        erase_area = random.uniform(*self.scale) * area
        eh = int(math.sqrt(erase_area * aspect))
        ew = int(math.sqrt(erase_area / aspect))
        img[:, y0:y0+eh, x0:x0+ew] = torch.rand(C, eh, ew)
        return img
```

---

## MixUp and CutMix

MixUp and CutMix are regularization techniques that blend training samples, creating soft labels that improve generalization.

### MixUp

MixUp creates virtual training examples by linearly interpolating between two samples:

```
x_mixed = λ · x_a + (1 - λ) · x_b
loss = λ · L(pred, y_a) + (1 - λ) · L(pred, y_b)
```

Where λ ~ Beta(α, α), typically α = 0.2.

```python
def mixup(images, labels, alpha=0.2):
    lam = Beta(alpha, alpha).sample().item()
    perm = torch.randperm(images.size(0))
    mixed = lam * images + (1 - lam) * images[perm]
    return mixed, labels, labels[perm], lam
```

### CutMix

CutMix pastes a rectangular patch from one image onto another, combining spatial regions:

```
x_mixed[:, :, cy:cy+ch, cx:cx+cw] = x_b[:, :, cy:cy+ch, cx:cx+cw]
λ_actual = 1 - (ch · cw) / (H · W)
```

```python
def cutmix(images, labels, alpha=1.0):
    lam = Beta(alpha, alpha).sample().item()
    cut_ratio = sqrt(1 - lam)
    ch, cw = int(H * cut_ratio), int(W * cut_ratio)
    mixed = images.clone()
    mixed[:, :, cy:cy+ch, cx:cx+cw] = images[perm, :, cy:cy+ch, cx:cx+cw]
    return mixed, labels, labels[perm], actual_lam
```

### MixUp vs CutMix

| Aspect | MixUp | CutMix |
|--------|-------|--------|
| Blending | Global pixel-wise | Local rectangular patch |
| Label mixing | Based on λ | Based on patch area ratio |
| Effect | Smoother decision boundaries | Better localization |
| Best α | 0.2 | 1.0 |
| Use case | General regularization | When spatial features matter |

### Training with Mixed Labels

```python
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

---

## CNN Architecture

### SimpleCNN

A lightweight 3-layer CNN with batch normalization and adaptive pooling:

```
Input (3, 32, 32)
  │
  ├─ Conv2d(3→32, 3x3) + BN + ReLU + MaxPool(2)    → (32, 16, 16)
  ├─ Conv2d(32→64, 3x3) + BN + ReLU + MaxPool(2)   → (64, 8, 8)
  ├─ Conv2d(64→128, 3x3) + BN + ReLU + AdaptPool(4) → (128, 4, 4)
  │
  ├─ Flatten                                          → (2048,)
  ├─ Dropout(0.3) + Linear(2048→256) + ReLU
  ├─ Dropout(0.2) + Linear(256→10)
  │
  └─ Output: 10 class logits
```

```python
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, img_size=32):
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

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)
```

### Design Decisions

| Choice | Reason |
|--------|--------|
| BatchNorm after Conv | Normalizes activations, enables higher learning rates |
| ReLU (inplace) | Saves memory, standard non-linearity |
| AdaptiveAvgPool(4) | Works with any input resolution |
| Dropout before Linear | Regularization in classifier head |
| 3x3 kernels only | Modern best practice (VGG insight) |

### Parameter Count

```
SimpleCNN:  ~570K parameters
  Conv layers:  ~110K
  Classifier:   ~460K (dominated by the 2048→256 linear)
```

---

## ResNet Architecture

### Residual Connections

The key insight: instead of learning H(x), learn the residual F(x) = H(x) - x:

```
output = F(x) + x
```

This solves the degradation problem — deeper networks can always learn the identity by setting F(x) = 0.

### BasicBlock

```
x ──┬── Conv(3x3) → BN → ReLU → Conv(3x3) → BN ──┐
    │                                                │
    └──────────── Shortcut (identity or 1x1) ───────┘
                         │
                      ReLU(sum)
```

```python
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # Shortcut: identity when dimensions match, 1x1 conv otherwise
        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.shortcut(x), inplace=True)  # Residual add
        return out
```

### MiniResNet

Adapted for 32x32 images (CIFAR-style, no initial downsampling):

```
Input (3, 32, 32)
  ├─ Conv(3→64, 3x3) + BN + ReLU          → (64, 32, 32)
  ├─ Layer1: 2 × BasicBlock(64→64)         → (64, 32, 32)
  ├─ Layer2: 2 × BasicBlock(64→128, ↓2)    → (128, 16, 16)
  ├─ Layer3: 2 × BasicBlock(128→256, ↓2)   → (256, 8, 8)
  ├─ AdaptiveAvgPool(1)                     → (256, 1, 1)
  └─ Linear(256→10)                         → (10,)
```

### Weight Initialization

Kaiming initialization for convolutional layers ensures variance is preserved through ReLU networks:

```python
def _init_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
```

---

## Transfer Learning

Transfer learning reuses features learned on a large dataset (e.g., ImageNet) and fine-tunes for a new task.

### The Pattern

```
┌──────────────────────────────────────────────┐
│ Pretrained Backbone (frozen)                  │
│  Conv layers → learned features               │
│  [Parameters: requires_grad = False]          │
├──────────────────────────────────────────────┤
│ New Classification Head (trainable)           │
│  Dropout → Linear → ReLU → Linear → Output   │
│  [Parameters: requires_grad = True]           │
└──────────────────────────────────────────────┘
```

### Two-Phase Training

**Phase 1: Train head only** (backbone frozen)
```python
model = TransferModel(backbone, feature_dim=256, num_classes=10, freeze_backbone=True)
optimizer = AdamW(model.head.parameters(), lr=1e-3)  # Only head params
```

**Phase 2: Fine-tune everything** (backbone unfrozen with lower LR)
```python
param_groups = model.unfreeze_backbone(lr_factor=0.1)
optimizer = AdamW([
    {"params": model.backbone.parameters(), "lr": 1e-4},  # Lower LR
    {"params": model.head.parameters(), "lr": 1e-3},      # Normal LR
])
```

### Implementation

```python
class TransferModel(nn.Module):
    def __init__(self, backbone, feature_dim, num_classes=10, freeze_backbone=True):
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

    def unfreeze_backbone(self, lr_factor=0.1):
        for p in self.backbone.parameters():
            p.requires_grad = True
        return [
            {"params": self.backbone.parameters(), "lr": lr_factor},
            {"params": self.head.parameters()},
        ]
```

---

## Training Pipeline

### Components

| Component | Implementation | Purpose |
|-----------|---------------|---------|
| Optimizer | AdamW | Weight decay decoupled from gradient |
| Loss | Label Smoothing CE | Prevents overconfident predictions |
| Scheduler | Cosine + Warmup | Smooth LR decay with warmup |
| AMP | `torch.autocast` | Mixed precision for speed |
| EMA | Exponential moving average | Smoother, more stable weights |
| MixUp/CutMix | Random per batch | Regularization |
| Early Stopping | Patience-based | Prevents overfitting |
| Gradient Clipping | Max norm = 1.0 | Training stability |

### Label Smoothing

Instead of hard targets `[0, 0, 1, 0, ...]`, use soft targets `[ε/K, ε/K, 1-ε+ε/K, ε/K, ...]`:

```
L = (1 - ε) · NLL(pred, target) + ε · mean(-log_probs)
```

With ε = 0.1, the model is penalized less for being "not 100% sure," which improves calibration.

```python
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        log_probs = F.log_softmax(pred, dim=-1)
        nll_loss = F.nll_loss(log_probs, target, reduction="none")
        smooth_loss = -log_probs.mean(dim=-1)
        return ((1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss).mean()
```

### Cosine Warmup Scheduler

```
LR
 │  ╱──╲
 │ ╱    ╲
 │╱      ╲
 │ warmup  ╲ cosine decay
 │           ╲
 └────────────╲──────▶ epoch
   0    3         20
```

```python
class CosineWarmupScheduler(LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        ...

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            factor = self.last_epoch / max(1, self.warmup_epochs)
            return [base_lr * factor for base_lr in self.base_lrs]
        progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
        cosine = 0.5 * (1 + cos(pi * progress))
        return [self.min_lr + (base_lr - self.min_lr) * cosine for base_lr in self.base_lrs]
```

### EMA (Exponential Moving Average)

Maintains a shadow copy of parameters that updates slowly:

```
shadow = decay · shadow + (1 - decay) · current_params
```

With decay = 0.999, the EMA model is a smoothed version of the training model — often generalizes better.

```python
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {name: p.clone() for name, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1 - self.decay)
```

### Training Loop with AMP

```python
for images, labels in loader:
    images, labels = images.to(device), labels.to(device)

    # Optional MixUp / CutMix
    if use_mixup:
        images, y_a, y_b, lam = mixup(images, labels, alpha=0.2)

    # Forward with AMP
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        logits = model(images)
        loss = mixup_criterion(criterion, logits, y_a, y_b, lam)

    # Backward with GradScaler
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    # EMA update
    ema.update(model)
```

### Early Stopping

```python
if val_acc > best_val_acc:
    best_val_acc = val_acc
    best_state = model.state_dict()
    no_improve = 0
else:
    no_improve += 1
    if no_improve >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

model.load_state_dict(best_state)  # Restore best model
```

---

## Evaluation Metrics

### Beyond Accuracy

Accuracy alone can be misleading, especially with class imbalance. We compute:

| Metric | Formula | What It Tells You |
|--------|---------|-------------------|
| Precision | TP / (TP + FP) | Of predicted positives, how many are correct |
| Recall | TP / (TP + FN) | Of actual positives, how many were found |
| F1 Score | 2 · P · R / (P + R) | Harmonic mean of precision and recall |
| Macro F1 | avg(F1_per_class) | Equal weight to each class |
| Weighted F1 | weighted avg(F1_per_class) | Weight by class support |
| Top-k Acc | correct in top-k / total | Useful when k>1 classes are reasonable |

### Per-Class Report

```python
metrics = ClassificationMetrics(num_classes=10, class_names=CLASS_NAMES)
for images, labels in test_loader:
    logits = model(images)
    metrics.update(logits, labels)

metrics.print_report()
```

Output:
```
Class        Precision   Recall       F1   Support
--------------------------------------------------
circle          0.9200   0.9200   0.9200        50
square          0.8800   0.8800   0.8800        50
triangle        0.8600   0.8600   0.8600        50
...
--------------------------------------------------
Accuracy                                    0.8900
Macro F1                                    0.8900
Weighted F1                                 0.8900
Top-3 Acc                                   0.9700
```

### Confusion Matrix

```python
metrics.print_confusion_matrix()
```

The confusion matrix reveals which classes are most often confused with each other — a circle might be confused with a ring, for example.

---

## Test-Time Augmentation

### How TTA Works

At test time, apply multiple augmentations to the same image and average the predictions:

```
           ┌─── original ──── pred_1 ───┐
           ├─── h-flip ────── pred_2 ───┤
 image ────┼─── v-flip ────── pred_3 ───┼── average → final prediction
           ├─── rot90 ─────── pred_4 ───┤
           └─── rot180 ────── pred_5 ───┘
```

### Implementation

```python
class TTAAugmentation:
    def __init__(self, num_augmentations=5):
        self.augmentations = [
            lambda x: x,                           # Original
            lambda x: x.flip(-1),                   # Horizontal flip
            lambda x: x.flip(-2),                   # Vertical flip
            lambda x: torch.rot90(x, 1, [-2, -1]),  # 90° rotation
            lambda x: torch.rot90(x, 2, [-2, -1]),  # 180° rotation
        ]

    @torch.no_grad()
    def predict(self, model, images):
        all_probs = []
        for aug_fn in self.augmentations[:self.num_augmentations]:
            probs = F.softmax(model(aug_fn(images)), dim=-1)
            all_probs.append(probs)
        return torch.stack(all_probs).mean(dim=0)
```

### When to Use TTA

| Scenario | Use TTA? |
|----------|----------|
| Competition / final submission | Yes — free accuracy boost |
| Real-time inference | No — multiplies latency by N |
| Medical imaging / safety-critical | Yes — reliability matters |
| Development / prototyping | No — slower iteration |

Typical improvement: **+0.5–2% accuracy** at the cost of N× inference time.

---

## Grad-CAM Visualization

### What Is Grad-CAM?

Gradient-weighted Class Activation Mapping produces a heatmap showing which spatial regions of the input image most influenced the model's prediction.

### How It Works

1. **Forward pass**: record activations at the target conv layer
2. **Backward pass**: record gradients flowing into that layer
3. **Weight**: global-average-pool the gradients → per-channel importance weights
4. **Combine**: weighted sum of activation maps → heatmap
5. **ReLU**: keep only positive contributions (features that increase the score)

```
Activations A (C, H, W)     Gradients G (C, H, W)
        │                           │
        │                    GAP over (H,W)
        │                           │
        │                    Weights α (C,)
        │                           │
        └─── Σ(αc · Ac) ──── ReLU ─── Upsample ─── Heatmap
```

### Implementation

```python
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=-1)

        # Backward for target class
        one_hot = torch.zeros_like(output)
        one_hot[range(len(target_class)), target_class] = 1.0
        output.backward(gradient=one_hot)

        # Compute CAM
        weights = self.gradients.mean(dim=(-2, -1), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear")

        # Normalize to [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.squeeze(1)
```

### Using Grad-CAM

```python
grad_cam = GradCAM(model, target_layer=model.features[-3])
heatmaps = grad_cam.generate(images)
# heatmaps shape: (B, H, W), values in [0, 1]
# High values = important regions
```

---

## Confidence Calibration

### Expected Calibration Error (ECE)

A well-calibrated model's confidence should match its accuracy:
- If the model says "90% confident" for a set of predictions, 90% should be correct.

```
ECE = Σ (|Bm| / N) · |accuracy(Bm) - confidence(Bm)|
```

Where Bm are bins of predictions grouped by confidence level.

```python
def confidence_analysis(model, loader, device):
    # Compute per-prediction confidence
    probs = F.softmax(logits, dim=-1)
    max_probs, preds = probs.max(dim=-1)

    # Bin predictions by confidence
    for bin in confidence_bins:
        bin_acc = accuracy of predictions in this bin
        bin_conf = mean confidence in this bin
        ece += |bin_acc - bin_conf| * bin_size / total

    return {
        "mean_correct_confidence": ...,
        "mean_incorrect_confidence": ...,
        "ece": ece,
    }
```

### Interpreting ECE

| ECE | Calibration Quality |
|-----|-------------------|
| < 0.02 | Excellent |
| 0.02–0.05 | Good |
| 0.05–0.10 | Fair |
| > 0.10 | Poor — consider temperature scaling |

---

## Inference Pipeline

### Production-Ready Wrapper

```python
class ImageClassifier:
    def __init__(self, model, class_names, device, use_tta=False):
        self.model = model.to(device).eval()
        self.class_names = class_names
        self.normalize = Normalize()
        self.tta = TTAAugmentation(5) if use_tta else None

    @torch.no_grad()
    def predict(self, images, top_k=3):
        images = self.normalize(images).to(self.device)
        if self.tta:
            probs = self.tta.predict(self.model, images)
        else:
            probs = F.softmax(self.model(images), dim=-1)

        results = []
        for i in range(images.size(0)):
            top_probs, top_indices = probs[i].topk(top_k)
            results.append({
                "top_class": self.class_names[top_indices[0]],
                "confidence": top_probs[0].item(),
                "predictions": [...],
            })
        return results
```

### Usage

```python
classifier = ImageClassifier(model, CLASS_NAMES, device, use_tta=False)
results = classifier.predict(image_tensor)
print(results[0]["top_class"])     # "circle"
print(results[0]["confidence"])    # 0.95
```

### Saving and Loading

```python
# Save
checkpoint = {
    "model_state_dict": model.state_dict(),
    "class_names": CLASS_NAMES,
    "num_classes": NUM_CLASSES,
    "history": history,
}
torch.save(checkpoint, "classifier.pt")

# Load
checkpoint = torch.load("classifier.pt", weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
```

### torch.compile for Inference

```python
compiled_model = torch.compile(model, mode="reduce-overhead")
# First call triggers compilation, subsequent calls are faster
with torch.no_grad():
    output = compiled_model(images)
```

---

## Running the Scripts

```bash
cd 40_image_classifier

# Step 1: Data pipeline and augmentation demo
python data_pipeline.py

# Step 2: Model training (trains SimpleCNN with full pipeline)
python model_and_training.py

# Step 3: Evaluation, TTA, Grad-CAM, confidence analysis
python evaluation.py
```

---

## Key Takeaways

1. **Data augmentation is essential** — random flips, color jitter, and random erasing significantly reduce overfitting on small datasets
2. **MixUp and CutMix create virtual samples** — by blending images and labels, they smooth decision boundaries and improve generalization
3. **Residual connections enable depth** — the skip connection in BasicBlock lets gradients flow through deep networks without degradation
4. **Transfer learning saves compute** — freeze a pretrained backbone, train only the head, then optionally fine-tune with differential learning rates
5. **Label smoothing improves calibration** — soft targets prevent overconfident predictions and reduce the gap between confidence and accuracy
6. **EMA stabilizes training** — maintaining a moving average of parameters often yields a model that generalizes better than any single checkpoint
7. **TTA boosts accuracy for free** — averaging predictions over augmented views of the same image typically adds +0.5–2% accuracy
8. **Grad-CAM reveals what the model sees** — heatmaps show which spatial regions drive predictions, which is crucial for debugging and trust
9. **ECE measures calibration quality** — a model's confidence should match its accuracy; ECE quantifies the gap

---

### Further Resources

- [Module 04 — Neural Networks](../04_neural_networks/) — `nn.Module`, layers, losses
- [Module 06 — Data Loading](../06_data_loading/) — Dataset, DataLoader, custom collate
- [Module 07 — Training Pipelines](../07_training/) — full training loops, mixed precision
- [Module 12 — Model Architectures](../12_model_architectures/) — ResNet, ViT complete implementations
- [Module 29 — Mixed Precision](../29_mixed_precision/) — AMP, GradScaler deep dive
- [Module 33 — Interpretability](../33_interpretability/) — Grad-CAM, saliency maps, hooks
- [Module 39 — Text Classifier](../39_text_classifier/) — End-to-end NLP classification project
- [PyTorch Vision Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

---

<div align="center">

[← Previous Module (Text Classifier)](../39_text_classifier/) | [🏠 Home](../README.md) | Next Module → (coming soon)

**Notebook**: [`40_image_classifier.ipynb`](../notebooks/40_image_classifier.ipynb)

</div>
