"""
Module 06: Data Augmentation Patterns
=======================================
Demonstrates data augmentation techniques for images and other data types,
including MixUp, CutMix, and custom augmentations — all without
requiring torchvision (pure PyTorch implementations).

Run: python augmentation.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

print("=" * 70)
print("PART 1: Image Augmentation Basics (Pure PyTorch)")
print("=" * 70)

print("""
Data augmentation increases training data diversity by applying
random transformations. This reduces overfitting and improves
generalization. Here we implement common augmentations in pure PyTorch.
""")


class RandomHorizontalFlip:
    """Randomly flip image horizontally with probability p."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() < self.p:
            return img.flip(-1)  # Flip width dimension
        return img


class RandomVerticalFlip:
    """Randomly flip image vertically with probability p."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() < self.p:
            return img.flip(-2)  # Flip height dimension
        return img


class RandomCrop:
    """Randomly crop a patch of given size from the image."""

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img):
        _, h, w = img.shape
        th, tw = self.size
        if h == th and w == tw:
            return img
        top = torch.randint(0, h - th + 1, (1,)).item()
        left = torch.randint(0, w - tw + 1, (1,)).item()
        return img[:, top : top + th, left : left + tw]


class RandomNoise:
    """Add random Gaussian noise."""

    def __init__(self, std=0.05):
        self.std = std

    def __call__(self, img):
        noise = torch.randn_like(img) * self.std
        return (img + noise).clamp(0, 1)


class ColorJitter:
    """Randomly adjust brightness and contrast."""

    def __init__(self, brightness=0.2, contrast=0.2):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, img):
        # Brightness: add random value
        if self.brightness > 0:
            factor = 1.0 + (torch.rand(1).item() * 2 - 1) * self.brightness
            img = img * factor

        # Contrast: adjust toward/away from mean
        if self.contrast > 0:
            factor = 1.0 + (torch.rand(1).item() * 2 - 1) * self.contrast
            mean = img.mean()
            img = (img - mean) * factor + mean

        return img.clamp(0, 1)


class RandomRotation90:
    """Randomly rotate image by 0, 90, 180, or 270 degrees."""

    def __call__(self, img):
        k = torch.randint(0, 4, (1,)).item()
        return torch.rot90(img, k, dims=(-2, -1))


class RandomErasing:
    """Randomly erase a rectangular region (Cutout-like)."""

    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        self.p = p
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img):
        if torch.rand(1).item() > self.p:
            return img

        _, h, w = img.shape
        area = h * w
        target_area = torch.empty(1).uniform_(self.scale[0], self.scale[1]).item() * area
        aspect_ratio = torch.empty(1).uniform_(self.ratio[0], self.ratio[1]).item()

        eh = int(round((target_area * aspect_ratio) ** 0.5))
        ew = int(round((target_area / aspect_ratio) ** 0.5))

        if eh < h and ew < w:
            top = torch.randint(0, h - eh, (1,)).item()
            left = torch.randint(0, w - ew, (1,)).item()
            img = img.clone()
            img[:, top : top + eh, left : left + ew] = torch.rand(img.shape[0], eh, ew)

        return img


class Compose:
    """Chain multiple transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class Normalize:
    """Normalize with mean and std."""

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, img):
        return (img - self.mean) / self.std


# Demonstrate augmentations
torch.manual_seed(42)
sample_image = torch.rand(3, 32, 32)  # Random RGB image

print("\nOriginal image stats:")
print(f"  Shape: {sample_image.shape}")
print(f"  Mean per channel: {sample_image.mean(dim=(1,2)).tolist()}")

# Apply augmentations
train_transform = Compose([
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.2),
    ColorJitter(brightness=0.3, contrast=0.3),
    RandomNoise(std=0.03),
    RandomErasing(p=0.3),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

print("\nApplying train augmentation 5 times to same image:")
for i in range(5):
    augmented = train_transform(sample_image.clone())
    print(f"  Attempt {i}: mean={augmented.mean():.4f}, std={augmented.std():.4f}, "
          f"min={augmented.min():.4f}, max={augmented.max():.4f}")

print("\n" + "=" * 70)
print("PART 2: MixUp Augmentation")
print("=" * 70)

print("""
MixUp (Zhang et al., 2018):
  Creates virtual training examples by linearly interpolating between pairs:
    mixed_input = lambda * x_i + (1 - lambda) * x_j
    mixed_target = lambda * y_i + (1 - lambda) * y_j
  
  lambda ~ Beta(alpha, alpha), where alpha controls mixing strength
  - alpha=0: no mixing
  - alpha=0.2-0.4: typical for image classification
  - alpha=1.0: strong mixing (uniform distribution)
""")


def mixup_data(x, y, alpha=0.2):
    """Apply MixUp augmentation to a batch.
    
    Args:
        x: Batch of inputs (batch_size, ...)
        y: Batch of labels (batch_size,) as class indices
        alpha: Beta distribution parameter
    
    Returns:
        mixed_x: Mixed inputs
        y_a, y_b: Original labels for the two mixed samples
        lam: Mixing coefficient
    """
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0

    batch_size = x.size(0)
    # Random permutation for mixing partners
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for MixUp: weighted combination of two losses."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# Demonstrate MixUp
torch.manual_seed(42)
batch_size = 8
num_classes = 5
x = torch.randn(batch_size, 3, 32, 32)
y = torch.randint(0, num_classes, (batch_size,))

print(f"\nOriginal labels: {y.tolist()}")
mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.4)
print(f"Lambda: {lam:.4f}")
print(f"Labels A: {y_a.tolist()}")
print(f"Labels B: {y_b.tolist()}")
print(f"Mixed input range: [{mixed_x.min():.3f}, {mixed_x.max():.3f}]")

# Loss computation with MixUp
model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, num_classes))
criterion = nn.CrossEntropyLoss()
pred = model(mixed_x)
loss = mixup_criterion(criterion, pred, y_a, y_b, lam)
print(f"MixUp loss: {loss.item():.4f}")

print("\n" + "=" * 70)
print("PART 3: CutMix Augmentation")
print("=" * 70)

print("""
CutMix (Yun et al., 2019):
  Cuts a rectangular patch from one image and pastes onto another:
    mixed_input = mask * x_i + (1 - mask) * x_j
    mixed_target = lambda * y_i + (1 - lambda) * y_j
  
  Where lambda = 1 - (patch_area / total_area)
  
  Benefits over MixUp:
  - Preserves local structure (no blurring)
  - Forces model to attend to less discriminative parts
  - Achieves better results on ImageNet
""")


def rand_bbox(size, lam):
    """Generate random bounding box for CutMix.
    
    Returns (x1, y1, x2, y2) for the cut region.
    """
    _, _, H, W = size
    cut_ratio = (1.0 - lam) ** 0.5
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    # Center of the cut
    cy = torch.randint(0, H, (1,)).item()
    cx = torch.randint(0, W, (1,)).item()

    # Bounding box
    y1 = max(0, cy - cut_h // 2)
    y2 = min(H, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(W, cx + cut_w // 2)

    return x1, y1, x2, y2


def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix augmentation to a batch.
    
    Args:
        x: Batch of images (batch_size, C, H, W)
        y: Batch of labels (batch_size,)
        alpha: Beta distribution parameter
    
    Returns:
        mixed_x: CutMixed images
        y_a, y_b: Original labels
        lam: Actual mixing ratio (after bbox adjustment)
    """
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    batch_size = x.size(0)
    index = torch.randperm(batch_size)

    x1, y1, x2, y2 = rand_bbox(x.size(), lam)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    # Adjust lambda based on actual bbox area
    _, _, H, W = x.size()
    lam = 1 - (x2 - x1) * (y2 - y1) / (H * W)

    return mixed_x, y, y[index], lam


# Demonstrate CutMix
torch.manual_seed(42)
x = torch.randn(8, 3, 32, 32)
y = torch.randint(0, 5, (8,))

mixed_x, y_a, y_b, lam = cutmix_data(x, y, alpha=1.0)
print(f"\nCutMix results:")
print(f"  Lambda (area ratio): {lam:.4f}")
print(f"  Labels A: {y_a.tolist()}")
print(f"  Labels B: {y_b.tolist()}")

# Verify the cut region
diff = (x[0] - mixed_x[0]).abs().sum(dim=0)
modified_pixels = (diff > 0).float().mean().item()
print(f"  Fraction of image replaced: {modified_pixels:.3f} (should be ~{1-lam:.3f})")

print("\n" + "=" * 70)
print("PART 4: Training Loop with MixUp/CutMix")
print("=" * 70)


class ImageClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


class AugmentedImageDataset(Dataset):
    """Synthetic dataset with train/val transforms."""

    def __init__(self, num_samples, num_classes=10, transform=None):
        self.images = torch.rand(num_samples, 3, 32, 32)
        self.labels = torch.randint(0, num_classes, (num_samples,))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# Create datasets
torch.manual_seed(42)
train_transform = Compose([
    RandomHorizontalFlip(0.5),
    ColorJitter(0.2, 0.2),
    Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
val_transform = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

train_dataset = AugmentedImageDataset(400, num_classes=10, transform=train_transform)
val_dataset = AugmentedImageDataset(100, num_classes=10, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Train with MixUp
model = ImageClassifier(num_classes=10)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print(f"\nTraining with MixUp (alpha=0.2):")
print(f"{'Epoch':<8}{'Train Loss':<12}{'Val Loss':<12}{'Val Acc':<10}")
print("-" * 42)

for epoch in range(10):
    model.train()
    train_loss = 0
    train_batches = 0
    for batch_x, batch_y in train_loader:
        # Apply MixUp to batch
        if torch.rand(1).item() < 0.5:  # 50% chance of MixUp
            mixed_x, y_a, y_b, lam = mixup_data(batch_x, batch_y, alpha=0.2)
            output = model(mixed_x)
            loss = mixup_criterion(criterion, output, y_a, y_b, lam)
        else:
            output = model(batch_x)
            loss = criterion(output, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_batches += 1

    # Validation (no augmentation)
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            output = model(batch_x)
            val_loss += criterion(output, batch_y).item()
            val_correct += (output.argmax(1) == batch_y).sum().item()
            val_total += batch_x.size(0)

    if epoch % 2 == 0:
        print(f"{epoch:<8}{train_loss/train_batches:<12.4f}"
              f"{val_loss/len(val_loader):<12.4f}"
              f"{val_correct/val_total:<10.3f}")

print("\n" + "=" * 70)
print("PART 5: Sequence Augmentation")
print("=" * 70)

print("""
Augmentation is not just for images! Common sequence augmentations:
  - Random deletion: Remove random tokens
  - Random insertion: Insert random tokens
  - Random swap: Swap adjacent tokens
  - Synonym replacement: Replace with similar tokens
  - Back-translation: Translate to another language and back
""")


class SequenceAugmentation:
    """Augmentation strategies for token sequences."""

    def __init__(self, vocab_size, p_delete=0.1, p_swap=0.1, p_insert=0.05):
        self.vocab_size = vocab_size
        self.p_delete = p_delete
        self.p_swap = p_swap
        self.p_insert = p_insert

    def random_delete(self, tokens):
        """Randomly delete tokens with probability p."""
        mask = torch.rand(len(tokens)) > self.p_delete
        if mask.sum() == 0:
            # Keep at least one token
            mask[0] = True
        return tokens[mask]

    def random_swap(self, tokens):
        """Randomly swap adjacent token pairs."""
        tokens = tokens.clone()
        for i in range(len(tokens) - 1):
            if torch.rand(1).item() < self.p_swap:
                tokens[i], tokens[i + 1] = tokens[i + 1].item(), tokens[i].item()
        return tokens

    def random_insert(self, tokens):
        """Randomly insert random tokens."""
        result = []
        for token in tokens:
            result.append(token.item())
            if torch.rand(1).item() < self.p_insert:
                result.append(torch.randint(1, self.vocab_size, (1,)).item())
        return torch.tensor(result)

    def __call__(self, tokens):
        """Apply random augmentations."""
        if torch.rand(1).item() < 0.33:
            tokens = self.random_delete(tokens)
        elif torch.rand(1).item() < 0.5:
            tokens = self.random_swap(tokens)
        else:
            tokens = self.random_insert(tokens)
        return tokens


# Demonstrate
seq_aug = SequenceAugmentation(vocab_size=100, p_delete=0.2, p_swap=0.2, p_insert=0.1)
original = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80])

print(f"\nOriginal sequence: {original.tolist()}")
print("Augmented versions:")
for i in range(5):
    torch.manual_seed(i)
    augmented = seq_aug(original.clone())
    print(f"  {i}: {augmented.tolist()}")

print("\n" + "=" * 70)
print("PART 6: Tabular Data Augmentation")
print("=" * 70)

print("""
Augmentation for tabular/numerical data:
  - Gaussian noise injection
  - Feature masking (dropout-like)
  - SMOTE-inspired interpolation
  - Feature permutation
""")


class TabularAugmentation:
    """Augmentation for tabular data."""

    def __init__(self, noise_std=0.1, mask_prob=0.1, mixup_alpha=0.2):
        self.noise_std = noise_std
        self.mask_prob = mask_prob
        self.mixup_alpha = mixup_alpha

    def add_noise(self, x):
        """Add Gaussian noise."""
        return x + torch.randn_like(x) * self.noise_std

    def feature_mask(self, x):
        """Randomly zero out features."""
        mask = torch.rand_like(x) > self.mask_prob
        return x * mask

    def __call__(self, x):
        """Apply random tabular augmentation."""
        r = torch.rand(1).item()
        if r < 0.5:
            return self.add_noise(x)
        else:
            return self.feature_mask(x)


# Demonstrate
tab_aug = TabularAugmentation(noise_std=0.1, mask_prob=0.15)
original = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

print(f"\nOriginal: {original.tolist()}")
print("Augmented versions:")
for i in range(5):
    torch.manual_seed(i * 10)
    augmented = tab_aug(original.clone())
    print(f"  {i}: {[f'{v:.3f}' for v in augmented.tolist()]}")

print("\n" + "=" * 70)
print("PART 7: Augmentation in Dataset vs DataLoader")
print("=" * 70)

print("""
Two approaches for applying augmentation:

1. In Dataset.__getitem__() — per-sample transform
   Pros: Simple, different augmentation each access
   Cons: Cannot do batch-level augmentations (MixUp, CutMix)

2. In training loop — batch-level transform
   Pros: Can mix between samples (MixUp, CutMix)
   Cons: Slightly more complex code

Best practice: Use BOTH!
  - Per-sample transforms (flip, crop, jitter) in Dataset
  - Batch-level transforms (MixUp, CutMix) in training loop
""")


class FullyAugmentedDataset(Dataset):
    """Dataset with per-sample augmentation."""

    def __init__(self, images, labels, training=True):
        self.images = images
        self.labels = labels
        self.training = training

        if training:
            self.transform = Compose([
                RandomHorizontalFlip(0.5),
                RandomCrop(28),
                ColorJitter(0.3, 0.3),
                RandomNoise(0.02),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# Training with both per-sample and batch-level augmentation
train_imgs = torch.rand(200, 3, 32, 32)
train_labels = torch.randint(0, 10, (200,))

dataset = FullyAugmentedDataset(train_imgs, train_labels, training=True)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

print("\nFull augmentation pipeline:")
batch_x, batch_y = next(iter(loader))
print(f"  After per-sample augmentation: {batch_x.shape}")

# Apply batch-level augmentation
mixed_x, y_a, y_b, lam = mixup_data(batch_x, batch_y, alpha=0.2)
print(f"  After MixUp: {mixed_x.shape}, lambda={lam:.3f}")

print("\n" + "=" * 70)
print("PART 8: Test-Time Augmentation (TTA)")
print("=" * 70)

print("""
Test-Time Augmentation: Apply multiple augmentations at inference time
and average predictions for better accuracy.

Common TTA transforms:
  - Original + horizontal flip (2x)
  - Original + 4 rotations (5x)
  - Multi-scale (resize to different sizes)
""")


def predict_with_tta(model, image, num_augmentations=5):
    """Apply TTA and average predictions."""
    model.eval()
    predictions = []

    with torch.no_grad():
        # Original
        pred = model(image.unsqueeze(0))
        predictions.append(F.softmax(pred, dim=1))

        # Horizontal flip
        flipped = image.flip(-1)
        pred = model(flipped.unsqueeze(0))
        predictions.append(F.softmax(pred, dim=1))

        # Random crops (if image is large enough)
        for _ in range(num_augmentations - 2):
            # Add small noise as a form of augmentation
            noisy = image + torch.randn_like(image) * 0.01
            pred = model(noisy.unsqueeze(0))
            predictions.append(F.softmax(pred, dim=1))

    # Average all predictions
    avg_pred = torch.stack(predictions).mean(dim=0)
    return avg_pred


# Demonstrate TTA
model = ImageClassifier(num_classes=10)
model.eval()
test_image = torch.rand(3, 32, 32)

# Single prediction
with torch.no_grad():
    single_pred = F.softmax(model(test_image.unsqueeze(0)), dim=1)

# TTA prediction
tta_pred = predict_with_tta(model, test_image, num_augmentations=8)

print(f"\nSingle prediction - top class: {single_pred.argmax().item()}, "
      f"confidence: {single_pred.max().item():.4f}")
print(f"TTA prediction - top class: {tta_pred.argmax().item()}, "
      f"confidence: {tta_pred.max().item():.4f}")
print(f"TTA smooths predictions and often improves accuracy")

print("\n" + "=" * 70)
print("SUMMARY: Augmentation Selection Guide")
print("=" * 70)

print("""
Data Type    | Augmentations                    | Where
-------------|----------------------------------|------------------
Images       | Flip, Crop, Jitter, Rotate       | Dataset.__getitem__
Images       | MixUp, CutMix, CutOut            | Training loop (batch)
Text         | Delete, Swap, Insert, Synonym    | Dataset.__getitem__
Tabular      | Noise, Mask, Interpolation       | Dataset.__getitem__
Audio        | Speed, Pitch, Noise, TimeStretch | Dataset.__getitem__
Time Series  | Jitter, Scale, Window warp       | Dataset.__getitem__

Best practices:
  1. Start simple (flip + crop for images)
  2. Add complexity if overfitting persists
  3. Use MixUp/CutMix for state-of-the-art results
  4. Never augment validation/test data (except TTA)
  5. Match augmentation intensity to dataset size:
     - Small dataset: aggressive augmentation
     - Large dataset: light augmentation
  6. Always use Normalize as the LAST transform
""")

print("=" * 70)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
print("=" * 70)
