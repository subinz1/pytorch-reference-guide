"""
Module 40 — Image Classifier: Data Pipeline
============================================

Synthetic image dataset, augmentation transforms, MixUp, and CutMix.

Usage:
    python data_pipeline.py

Requirements:
    pip install torch torchvision
"""

import math
import random
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


# ---------------------------------------------------------------------------
# 1. Synthetic Image Dataset
# ---------------------------------------------------------------------------

CLASS_NAMES = [
    "circle", "square", "triangle", "cross", "diamond",
    "star", "ring", "arrow", "pentagon", "hexagon",
]
NUM_CLASSES = len(CLASS_NAMES)


def _draw_circle(img: torch.Tensor, cx: int, cy: int, r: int, val: float) -> None:
    H, W = img.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    mask = ((xx - cx).float().pow(2) + (yy - cy).float().pow(2)).sqrt() <= r
    img[:, mask] = val


def _draw_square(img: torch.Tensor, cx: int, cy: int, r: int, val: float) -> None:
    H, W = img.shape[-2:]
    y0, y1 = max(0, cy - r), min(H, cy + r)
    x0, x1 = max(0, cx - r), min(W, cx + r)
    img[:, y0:y1, x0:x1] = val


def _draw_triangle(img: torch.Tensor, cx: int, cy: int, r: int, val: float) -> None:
    H, W = img.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    top_y = cy - r
    base_y = cy + r
    in_height = (yy >= top_y) & (yy <= base_y)
    height = (yy - top_y).float().clamp(min=0)
    half_width = height * (r / max(1, base_y - top_y))
    in_width = (xx - cx).float().abs() <= half_width
    mask = in_height & in_width
    img[:, mask] = val


def _draw_cross(img: torch.Tensor, cx: int, cy: int, r: int, val: float) -> None:
    t = max(2, r // 3)
    H, W = img.shape[-2:]
    y0, y1 = max(0, cy - r), min(H, cy + r)
    x0, x1 = max(0, cx - t), min(W, cx + t)
    img[:, y0:y1, x0:x1] = val
    y0, y1 = max(0, cy - t), min(H, cy + t)
    x0, x1 = max(0, cx - r), min(W, cx + r)
    img[:, y0:y1, x0:x1] = val


def _draw_diamond(img: torch.Tensor, cx: int, cy: int, r: int, val: float) -> None:
    H, W = img.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    mask = ((xx - cx).float().abs() + (yy - cy).float().abs()) <= r
    img[:, mask] = val


def _draw_star(img: torch.Tensor, cx: int, cy: int, r: int, val: float) -> None:
    _draw_cross(img, cx, cy, r, val)
    _draw_diamond(img, cx, cy, r, val)


def _draw_ring(img: torch.Tensor, cx: int, cy: int, r: int, val: float) -> None:
    H, W = img.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    dist = ((xx - cx).float().pow(2) + (yy - cy).float().pow(2)).sqrt()
    inner = max(1, r - max(2, r // 3))
    mask = (dist >= inner) & (dist <= r)
    img[:, mask] = val


def _draw_arrow(img: torch.Tensor, cx: int, cy: int, r: int, val: float) -> None:
    t = max(2, r // 4)
    H, W = img.shape[-2:]
    img[:, max(0, cy - t):min(H, cy + t), max(0, cx - r):min(W, cx + r)] = val
    _draw_triangle(img, cx + r - r // 3, cy, r // 2, val)


def _draw_pentagon(img: torch.Tensor, cx: int, cy: int, r: int, val: float) -> None:
    _draw_circle(img, cx, cy, r, val)
    _draw_circle(img, cx, cy, max(1, r - max(3, r // 3)), 0.0)
    _draw_diamond(img, cx, cy, r, val)


def _draw_hexagon(img: torch.Tensor, cx: int, cy: int, r: int, val: float) -> None:
    H, W = img.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    dx = (xx - cx).float().abs()
    dy = (yy - cy).float().abs()
    mask = (dx <= r) & (dy <= r * 0.866) & (dx + dy * 0.577 <= r)
    img[:, mask] = val


_DRAW_FNS = [
    _draw_circle, _draw_square, _draw_triangle, _draw_cross, _draw_diamond,
    _draw_star, _draw_ring, _draw_arrow, _draw_pentagon, _draw_hexagon,
]


class SyntheticShapeDataset(Dataset):
    """Generates synthetic 3-channel images with geometric shapes."""

    def __init__(
        self,
        num_samples: int = 5000,
        img_size: int = 32,
        channels: int = 3,
        transform=None,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.img_size = img_size
        self.channels = channels
        self.transform = transform

        rng = random.Random(seed)
        self.labels = [rng.randint(0, NUM_CLASSES - 1) for _ in range(num_samples)]
        self.params = []
        for _ in range(num_samples):
            margin = img_size // 4
            cx = rng.randint(margin, img_size - margin)
            cy = rng.randint(margin, img_size - margin)
            r = rng.randint(img_size // 6, img_size // 3)
            bg = rng.random() * 0.3
            fg = 0.5 + rng.random() * 0.5
            self.params.append((cx, cy, r, bg, fg))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        label = self.labels[idx]
        cx, cy, r, bg, fg = self.params[idx]
        img = torch.full((self.channels, self.img_size, self.img_size), bg)
        _DRAW_FNS[label](img, cx, cy, r, fg)
        noise = torch.randn_like(img) * 0.05
        img = (img + noise).clamp(0, 1)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# 2. Data Augmentation Transforms (Pure PyTorch)
# ---------------------------------------------------------------------------

class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return img.flip(-1)
        return img


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return img.flip(-2)
        return img


class RandomRotation90:
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        k = random.randint(0, 3)
        return torch.rot90(img, k, [-2, -1])


class ColorJitter:
    def __init__(self, brightness: float = 0.2, contrast: float = 0.2):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        b_factor = 1.0 + (random.random() * 2 - 1) * self.brightness
        c_factor = 1.0 + (random.random() * 2 - 1) * self.contrast
        mean = img.mean()
        img = (img - mean) * c_factor + mean
        img = img * b_factor
        return img.clamp(0, 1)


class RandomErasing:
    """Randomly erases a rectangular patch (cutout-style regularization)."""

    def __init__(self, p: float = 0.3, scale: tuple[float, float] = (0.02, 0.15)):
        self.p = p
        self.scale = scale

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img
        C, H, W = img.shape
        area = H * W
        erase_area = random.uniform(*self.scale) * area
        aspect = random.uniform(0.5, 2.0)
        eh = int(math.sqrt(erase_area * aspect))
        ew = int(math.sqrt(erase_area / aspect))
        eh, ew = min(eh, H), min(ew, W)
        y0 = random.randint(0, H - eh)
        x0 = random.randint(0, W - ew)
        img[:, y0:y0 + eh, x0:x0 + ew] = torch.rand(C, eh, ew)
        return img


class Normalize:
    def __init__(self, mean: tuple = (0.5, 0.5, 0.5), std: tuple = (0.5, 0.5, 0.5)):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return (img - self.mean) / self.std


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            img = t(img)
        return img


# ---------------------------------------------------------------------------
# 3. MixUp and CutMix
# ---------------------------------------------------------------------------

def mixup(images: torch.Tensor, labels: torch.Tensor, alpha: float = 0.2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """MixUp: blend two images and their labels with a Beta-sampled ratio."""
    lam = torch.distributions.Beta(alpha, alpha).sample().item() if alpha > 0 else 1.0
    batch_size = images.size(0)
    perm = torch.randperm(batch_size)
    mixed_images = lam * images + (1 - lam) * images[perm]
    return mixed_images, labels, labels[perm], lam


def cutmix(images: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """CutMix: paste a random patch from one image onto another."""
    lam = torch.distributions.Beta(alpha, alpha).sample().item() if alpha > 0 else 1.0
    B, C, H, W = images.shape
    perm = torch.randperm(B)

    cut_ratio = math.sqrt(1 - lam)
    ch, cw = int(H * cut_ratio), int(W * cut_ratio)
    cy = random.randint(0, H - ch) if ch < H else 0
    cx = random.randint(0, W - cw) if cw < W else 0

    mixed = images.clone()
    mixed[:, :, cy:cy + ch, cx:cx + cw] = images[perm, :, cy:cy + ch, cx:cx + cw]
    actual_lam = 1 - (ch * cw) / (H * W)
    return mixed, labels, labels[perm], actual_lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for MixUp/CutMix mixed labels."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ---------------------------------------------------------------------------
# 4. Build Complete Data Pipeline
# ---------------------------------------------------------------------------

def build_dataloaders(
    num_train: int = 4000,
    num_val: int = 500,
    num_test: int = 500,
    img_size: int = 32,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders with appropriate augmentations."""

    train_transform = Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.3),
        RandomRotation90(),
        ColorJitter(brightness=0.2, contrast=0.2),
        RandomErasing(p=0.3),
        Normalize(),
    ])

    val_transform = Compose([
        Normalize(),
    ])

    train_ds = SyntheticShapeDataset(num_train, img_size, transform=train_transform, seed=42)
    val_ds = SyntheticShapeDataset(num_val, img_size, transform=val_transform, seed=123)
    test_ds = SyntheticShapeDataset(num_test, img_size, transform=val_transform, seed=456)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("MODULE 40 — IMAGE CLASSIFIER: DATA PIPELINE")
    print("=" * 70)

    # --- Dataset demo ---
    print("\n1. Synthetic Shape Dataset")
    print("-" * 40)
    ds = SyntheticShapeDataset(num_samples=100, img_size=32)
    img, label = ds[0]
    print(f"   Image shape:   {img.shape}")
    print(f"   Label:         {label} ({CLASS_NAMES[label]})")
    print(f"   Pixel range:   [{img.min():.3f}, {img.max():.3f}]")
    print(f"   Dataset size:  {len(ds)}")

    # --- Augmentation demo ---
    print("\n2. Data Augmentation")
    print("-" * 40)
    transforms = Compose([
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ColorJitter(),
        RandomErasing(p=0.5),
        Normalize(),
    ])
    aug_img = transforms(img.clone())
    print(f"   Original range:    [{img.min():.3f}, {img.max():.3f}]")
    print(f"   Augmented range:   [{aug_img.min():.3f}, {aug_img.max():.3f}]")
    print(f"   Shape preserved:   {aug_img.shape == img.shape}")

    # --- MixUp demo ---
    print("\n3. MixUp")
    print("-" * 40)
    batch_imgs = torch.stack([ds[i][0] for i in range(8)])
    batch_labels = torch.tensor([ds[i][1] for i in range(8)])
    mixed, y_a, y_b, lam = mixup(batch_imgs, batch_labels, alpha=0.4)
    print(f"   Batch shape:   {batch_imgs.shape}")
    print(f"   Mixed shape:   {mixed.shape}")
    print(f"   Lambda:        {lam:.3f}")
    print(f"   Labels A:      {y_a.tolist()}")
    print(f"   Labels B:      {y_b.tolist()}")

    # --- CutMix demo ---
    print("\n4. CutMix")
    print("-" * 40)
    cut_mixed, y_a, y_b, lam = cutmix(batch_imgs, batch_labels, alpha=1.0)
    print(f"   CutMix shape:  {cut_mixed.shape}")
    print(f"   Lambda:        {lam:.3f}")

    # --- DataLoader demo ---
    print("\n5. Complete Data Pipeline")
    print("-" * 40)
    train_loader, val_loader, test_loader = build_dataloaders(
        num_train=2000, num_val=300, num_test=300, batch_size=32,
    )
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches:   {len(val_loader)}")
    print(f"   Test batches:  {len(test_loader)}")

    batch = next(iter(train_loader))
    print(f"   Batch images:  {batch[0].shape}")
    print(f"   Batch labels:  {batch[1].shape}")
    print(f"   Label dist:    {torch.bincount(batch[1], minlength=NUM_CLASSES).tolist()}")

    print("\n" + "=" * 70)
    print("DATA PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
