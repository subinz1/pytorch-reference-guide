"""
Module 06: Dataset Basics
==========================
Introduction to PyTorch's Dataset class, built-in datasets,
TensorDataset, and basic DataLoader usage.

Run: python dataset_basics.py
"""

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

print("=" * 70)
print("PART 1: The Dataset Protocol")
print("=" * 70)

print("""
A PyTorch Dataset must implement:
  __len__()         -> int       (total number of samples)
  __getitem__(idx)  -> sample    (one sample at given index)

The DataLoader uses these methods to create batches.
""")


# Simplest possible custom dataset
class SimpleDataset(Dataset):
    """A dataset of (x, y) pairs where y = 2*x + noise."""

    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        self.x = torch.randn(num_samples, 1)
        self.y = 2 * self.x + torch.randn(num_samples, 1) * 0.1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


dataset = SimpleDataset(1000)
print(f"\nSimpleDataset:")
print(f"  Length: {len(dataset)}")
print(f"  Sample 0: x={dataset[0][0].item():.4f}, y={dataset[0][1].item():.4f}")
print(f"  Sample 5: x={dataset[5][0].item():.4f}, y={dataset[5][1].item():.4f}")

# Supports slicing via DataLoader, not directly (unless you implement it)
print(f"  Type of dataset[0]: {type(dataset[0])}")

print("\n" + "=" * 70)
print("PART 2: TensorDataset — Wrapping Tensors")
print("=" * 70)

print("""
TensorDataset wraps existing tensors. Each tensor's first dimension
must have the same length. Indexing returns a tuple of tensors.
""")

# Create data
X = torch.randn(500, 20)  # 500 samples, 20 features
y = torch.randint(0, 5, (500,))  # 5 classes
weights = torch.ones(500)  # Sample weights

# Single pair
dataset_simple = TensorDataset(X, y)
print(f"TensorDataset with (X, y):")
print(f"  Length: {len(dataset_simple)}")
sample = dataset_simple[0]
print(f"  dataset[0]: tuple of {len(sample)} tensors")
print(f"  X shape: {sample[0].shape}, y value: {sample[1].item()}")

# Multiple tensors
dataset_multi = TensorDataset(X, y, weights)
sample = dataset_multi[0]
print(f"\nTensorDataset with (X, y, weights):")
print(f"  dataset[0]: tuple of {len(sample)} tensors")

print("\n" + "=" * 70)
print("PART 3: Basic DataLoader Usage")
print("=" * 70)

dataset = TensorDataset(X, y)

# Default DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=False)
print(f"\nDataLoader(batch_size=32, shuffle=False):")
print(f"  Number of batches: {len(loader)}")

# Iterate through first few batches
for i, (batch_x, batch_y) in enumerate(loader):
    if i < 3:
        print(f"  Batch {i}: X shape={batch_x.shape}, y shape={batch_y.shape}")
    if i == 0:
        print(f"           First labels: {batch_y[:5].tolist()}")

# Shuffled DataLoader
loader_shuffled = DataLoader(dataset, batch_size=32, shuffle=True)
print(f"\nDataLoader(batch_size=32, shuffle=True):")
first_batch_labels = []
for epoch in range(3):
    for batch_x, batch_y in loader_shuffled:
        first_batch_labels.append(batch_y[:5].tolist())
        break
print(f"  First 5 labels in batch 0, epoch 0: {first_batch_labels[0]}")
print(f"  First 5 labels in batch 0, epoch 1: {first_batch_labels[1]}")
print(f"  First 5 labels in batch 0, epoch 2: {first_batch_labels[2]}")
print("  (Different each epoch due to shuffling)")

# drop_last
loader_drop = DataLoader(dataset, batch_size=32, drop_last=True)
loader_keep = DataLoader(dataset, batch_size=32, drop_last=False)
print(f"\ndrop_last effect (500 samples, batch_size=32):")
print(f"  drop_last=False: {len(loader_keep)} batches (last batch has {500 % 32} samples)")
print(f"  drop_last=True:  {len(loader_drop)} batches (last small batch dropped)")

print("\n" + "=" * 70)
print("PART 4: Dataset Splitting")
print("=" * 70)

full_dataset = TensorDataset(
    torch.randn(1000, 10),
    torch.randint(0, 2, (1000,))
)

# Method 1: random_split
print("\n--- Method 1: random_split ---")
train_set, val_set, test_set = random_split(
    full_dataset,
    [0.7, 0.15, 0.15],
    generator=torch.Generator().manual_seed(42)
)
print(f"Total: {len(full_dataset)}")
print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
print(f"Sum:   {len(train_set) + len(val_set) + len(test_set)}")

# Create DataLoaders for each split
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

print(f"\nTrain batches: {len(train_loader)}")
print(f"Val batches:   {len(val_loader)}")
print(f"Test batches:  {len(test_loader)}")

# Method 2: Manual splitting with Subset
print("\n--- Method 2: Subset with manual indices ---")
from torch.utils.data import Subset

indices = torch.randperm(len(full_dataset)).tolist()
train_idx = indices[:700]
val_idx = indices[700:850]
test_idx = indices[850:]

train_subset = Subset(full_dataset, train_idx)
val_subset = Subset(full_dataset, val_idx)
test_subset = Subset(full_dataset, test_idx)
print(f"Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_subset)}")

print("\n" + "=" * 70)
print("PART 5: ConcatDataset — Combining Datasets")
print("=" * 70)

from torch.utils.data import ConcatDataset

dataset_a = TensorDataset(torch.randn(100, 5), torch.zeros(100, dtype=torch.long))
dataset_b = TensorDataset(torch.randn(200, 5), torch.ones(200, dtype=torch.long))
dataset_c = TensorDataset(torch.randn(50, 5), torch.full((50,), 2, dtype=torch.long))

combined = ConcatDataset([dataset_a, dataset_b, dataset_c])
print(f"\nCombined dataset:")
print(f"  dataset_a: {len(dataset_a)} samples")
print(f"  dataset_b: {len(dataset_b)} samples")
print(f"  dataset_c: {len(dataset_c)} samples")
print(f"  combined:  {len(combined)} samples")

# Access across boundaries
print(f"\n  combined[0] from dataset_a: label={combined[0][1].item()}")
print(f"  combined[100] from dataset_b: label={combined[100][1].item()}")
print(f"  combined[300] from dataset_c: label={combined[300][1].item()}")

print("\n" + "=" * 70)
print("PART 6: Training Loop with DataLoader")
print("=" * 70)

import torch.nn as nn
import torch.optim as optim

# Create a classification dataset
torch.manual_seed(42)
num_samples = 500
num_features = 20
num_classes = 3

X = torch.randn(num_samples, num_features)
# Create linearly separable classes
W_true = torch.randn(num_features, num_classes)
y = (X @ W_true).argmax(dim=1)

dataset = TensorDataset(X, y)
train_set, val_set = random_split(dataset, [0.8, 0.2])
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        return self.net(x)


model = Classifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("\nTraining loop with DataLoader:")
for epoch in range(10):
    # Training
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch_x.size(0)
        correct += (output.argmax(1) == batch_y).sum().item()
        total += batch_x.size(0)

    train_loss /= total
    train_acc = correct / total

    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            output = model(batch_x)
            loss = criterion(output, batch_y)
            val_loss += loss.item() * batch_x.size(0)
            val_correct += (output.argmax(1) == batch_y).sum().item()
            val_total += batch_x.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total

    if epoch % 2 == 0:
        print(f"  Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}")

print("\n" + "=" * 70)
print("PART 7: DataLoader as an Iterator")
print("=" * 70)

loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Method 1: for loop (most common)
print("\n--- Method 1: For loop ---")
for i, (x, y) in enumerate(loader):
    if i == 0:
        print(f"  Batch 0: x.shape={x.shape}")
    if i >= 2:
        break

# Method 2: Manual iterator
print("\n--- Method 2: Manual iterator ---")
iterator = iter(loader)
batch1 = next(iterator)
batch2 = next(iterator)
print(f"  Manual batch 1: {batch1[0].shape}")
print(f"  Manual batch 2: {batch2[0].shape}")

# Method 3: Get a single batch (useful for testing/debugging)
print("\n--- Method 3: Single batch for debugging ---")
single_batch = next(iter(loader))
print(f"  Quick single batch: x={single_batch[0].shape}, y={single_batch[1].shape}")

print("\n" + "=" * 70)
print("PART 8: DataLoader with Multiple Workers")
print("=" * 70)

import time


class SlowDataset(Dataset):
    """Simulates slow data loading (I/O bound)."""

    def __init__(self, size=200):
        self.size = size
        self.data = torch.randn(size, 100)
        self.labels = torch.randint(0, 10, (size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Simulate some processing time
        _ = self.data[idx] * 2 + 1
        return self.data[idx], self.labels[idx]


slow_dataset = SlowDataset(200)

# Benchmark with different num_workers
print("\nBenchmarking num_workers (200 samples, batch_size=32):")
for num_workers in [0, 2]:
    loader = DataLoader(slow_dataset, batch_size=32, num_workers=num_workers)
    start = time.time()
    for batch_x, batch_y in loader:
        pass  # Just iterate
    elapsed = time.time() - start
    print(f"  num_workers={num_workers}: {elapsed:.3f}s")

print("\n(With real I/O-heavy datasets, multi-worker speedup is much more dramatic)")

print("\n" + "=" * 70)
print("PART 9: Reproducibility")
print("=" * 70)

print("\nEnsuring reproducible data loading order:")


def get_first_batch(seed):
    """Get first batch with given seed."""
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, generator=g)
    batch_x, batch_y = next(iter(loader))
    return batch_y.tolist()


batch_seed42_a = get_first_batch(42)
batch_seed42_b = get_first_batch(42)
batch_seed99 = get_first_batch(99)

print(f"  Seed 42 (run 1): {batch_seed42_a}")
print(f"  Seed 42 (run 2): {batch_seed42_b}")
print(f"  Seed 99:         {batch_seed99}")
print(f"  Same seed = same order: {batch_seed42_a == batch_seed42_b}")
print(f"  Different seed = different order: {batch_seed42_a != batch_seed99}")

print("\n" + "=" * 70)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
print("=" * 70)
