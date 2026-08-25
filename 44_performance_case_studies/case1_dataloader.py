"""
Case Study 1: Memory-Bound DataLoader

Problem: CPU→GPU tensor copies block the training loop, causing GPU idle time.
Fix: Use pin_memory=True + non_blocking transfers with prefetching.
Expected speedup: 2–3× on data-heavy workloads.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
from typing import Iterator


class SyntheticImageDataset(Dataset):
    """Simulates a dataset returning large tensors (like images)."""

    def __init__(self, size: int = 1000, image_shape: tuple = (3, 224, 224)):
        self.size = size
        self.image_shape = image_shape

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image = torch.randn(*self.image_shape)
        label = torch.randint(0, 10, (1,)).item()
        return image, label


# ============================================================
# BEFORE: Naive DataLoader (no pinned memory)
# ============================================================

def create_naive_loader(dataset, batch_size=32, num_workers=2):
    """Standard DataLoader without memory pinning."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        shuffle=True,
    )


def train_step_naive(model, batch, criterion, optimizer, device):
    """Blocking .to(device) call — GPU waits for transfer to complete."""
    images, labels = batch
    images = images.to(device)
    labels = labels.to(device)

    output = model(images)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()


# ============================================================
# AFTER: Optimized DataLoader (pinned memory + non-blocking)
# ============================================================

def create_optimized_loader(dataset, batch_size=32, num_workers=2):
    """DataLoader with pinned memory for async transfers."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        persistent_workers=True,
        prefetch_factor=3,
    )


def train_step_optimized(model, batch, criterion, optimizer, device):
    """Non-blocking transfer — GPU can overlap copy with previous compute."""
    images, labels = batch
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    output = model(images)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()


class CUDAPrefetcher:
    """Prefetches next batch to GPU while current batch is being processed."""

    def __init__(self, loader: DataLoader, device: torch.device):
        self.loader = loader
        self.device = device
        self._stream = torch.cuda.Stream() if device.type == "cuda" else None

    def __iter__(self) -> Iterator:
        loader_iter = iter(self.loader)
        first_batch = next(loader_iter, None)
        if first_batch is None:
            return

        if self._stream:
            with torch.cuda.stream(self._stream):
                next_images = first_batch[0].to(self.device, non_blocking=True)
                next_labels = first_batch[1].to(self.device, non_blocking=True)
        else:
            next_images = first_batch[0].to(self.device)
            next_labels = first_batch[1].to(self.device)

        for batch in loader_iter:
            current_images = next_images
            current_labels = next_labels

            if self._stream:
                torch.cuda.current_stream().wait_stream(self._stream)
                with torch.cuda.stream(self._stream):
                    next_images = batch[0].to(self.device, non_blocking=True)
                    next_labels = batch[1].to(self.device, non_blocking=True)
            else:
                next_images = batch[0].to(self.device)
                next_labels = batch[1].to(self.device)

            yield current_images, current_labels

        yield next_images, next_labels


# ============================================================
# BENCHMARK
# ============================================================

def benchmark_epoch(loader, model, criterion, optimizer, device, use_prefetcher=False):
    """Time one full epoch."""
    model.train()

    if use_prefetcher and device.type == "cuda":
        data_source = CUDAPrefetcher(loader, device)
    else:
        data_source = loader

    start = time.perf_counter()
    n_batches = 0

    for batch in data_source:
        if use_prefetcher:
            images, labels = batch
        else:
            images, labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        n_batches += 1

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    return elapsed, n_batches


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = SyntheticImageDataset(size=200, image_shape=(3, 64, 64))

    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 10),
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    print("\n--- Case 1: DataLoader Optimization ---\n")

    # BEFORE
    naive_loader = create_naive_loader(dataset, batch_size=32, num_workers=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Warmup
    benchmark_epoch(naive_loader, model, criterion, optimizer, device)

    elapsed_naive, n = benchmark_epoch(naive_loader, model, criterion, optimizer, device)
    print(f"Naive:     {elapsed_naive:.3f}s ({n} batches, {n/elapsed_naive:.1f} batches/sec)")

    # AFTER
    optimized_loader = create_optimized_loader(dataset, batch_size=32, num_workers=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Warmup
    benchmark_epoch(optimized_loader, model, criterion, optimizer, device, use_prefetcher=True)

    elapsed_opt, n = benchmark_epoch(optimized_loader, model, criterion, optimizer, device, use_prefetcher=True)
    print(f"Optimized: {elapsed_opt:.3f}s ({n} batches, {n/elapsed_opt:.1f} batches/sec)")

    speedup = elapsed_naive / elapsed_opt if elapsed_opt > 0 else 1.0
    print(f"\nSpeedup: {speedup:.2f}×")

    print("\n--- Key Differences ---")
    print("  Naive:     pin_memory=False, blocking .to(device)")
    print("  Optimized: pin_memory=True, non_blocking=True, prefetch_factor=3")
    if device.type == "cuda":
        print("             + CUDAPrefetcher overlaps next batch transfer with compute")


if __name__ == "__main__":
    main()
