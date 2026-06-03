"""
Module 06: Advanced DataLoader Features
=========================================
Covers custom collate functions, samplers (weighted, distributed),
pin_memory, persistent workers, and performance optimization.

Run: python dataloader_advanced.py
"""

import torch
from torch.utils.data import (
    Dataset, DataLoader, TensorDataset,
    RandomSampler, SequentialSampler, WeightedRandomSampler,
    BatchSampler, SubsetRandomSampler,
)
import time

print("=" * 70)
print("PART 1: Custom Collate Functions")
print("=" * 70)

print("""
The collate function converts a list of individual samples into a batch.
Default collator stacks tensors, but custom collation handles:
  - Variable-length sequences (padding)
  - Mixed data types
  - Filtering out corrupt samples
""")


# Dataset with variable-length sequences
class VariableLengthDataset(Dataset):
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        # Random length sequences (5 to 30 tokens)
        self.sequences = [
            torch.randint(1, 100, (torch.randint(5, 31, (1,)).item(),))
            for _ in range(num_samples)
        ]
        self.labels = torch.randint(0, 3, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


dataset = VariableLengthDataset(100)
print(f"\nVariable-length dataset:")
print(f"  Sample 0 length: {len(dataset[0][0])}")
print(f"  Sample 1 length: {len(dataset[1][0])}")
print(f"  Sample 2 length: {len(dataset[2][0])}")


# Custom collate: pad to max length in batch
def pad_collate(batch):
    """Pad sequences to the longest in the batch."""
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences])
    max_len = lengths.max().item()

    # Pad sequences
    padded = torch.zeros(len(sequences), max_len, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded[i, : len(seq)] = seq

    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = torch.zeros(len(sequences), max_len, dtype=torch.bool)
    for i, length in enumerate(lengths):
        attention_mask[i, :length] = True

    labels = torch.stack(labels)
    return padded, attention_mask, lengths, labels


loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=pad_collate)
padded, mask, lengths, labels = next(iter(loader))
print(f"\nWith pad_collate (batch_size=8):")
print(f"  Padded sequences: {padded.shape}")
print(f"  Attention mask: {mask.shape}")
print(f"  Lengths: {lengths.tolist()}")
print(f"  Labels: {labels.tolist()}")
print(f"  Max length in this batch: {lengths.max().item()}")


# Custom collate: filtering None values
print("\n--- Collate with filtering ---")


class NoisyDataset(Dataset):
    """Dataset where some samples are 'corrupt' (return None)."""

    def __init__(self, size=100):
        self.size = size
        self.data = torch.randn(size, 10)
        self.labels = torch.randint(0, 5, (size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 10% chance of "corrupt" sample
        if idx % 10 == 7:  # Deterministic "corruption"
            return None
        return self.data[idx], self.labels[idx]


def filter_none_collate(batch):
    """Filter out None samples and collate the rest."""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    features, labels = zip(*batch)
    return torch.stack(features), torch.stack(labels)


noisy_dataset = NoisyDataset(50)
loader = DataLoader(noisy_dataset, batch_size=8, collate_fn=filter_none_collate)
batch_sizes = []
for batch in loader:
    if batch is not None:
        batch_sizes.append(batch[0].shape[0])

print(f"Batch sizes (some < 8 due to filtering): {batch_sizes}")


# Custom collate: dictionary batching
print("\n--- Collate for dictionary samples ---")


class DictDataset(Dataset):
    def __init__(self, size=50):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "input": torch.randn(10),
            "target": torch.randint(0, 5, (1,)).item(),
            "weight": 1.0 if idx % 3 != 0 else 2.0,
            "id": f"sample_{idx}",
        }


def dict_collate(batch):
    """Collate a list of dicts into a dict of batched tensors."""
    return {
        "input": torch.stack([item["input"] for item in batch]),
        "target": torch.tensor([item["target"] for item in batch]),
        "weight": torch.tensor([item["weight"] for item in batch]),
        "id": [item["id"] for item in batch],
    }


dict_dataset = DictDataset(50)
loader = DataLoader(dict_dataset, batch_size=4, collate_fn=dict_collate)
batch = next(iter(loader))
print(f"Dict batch:")
print(f"  input: {batch['input'].shape}")
print(f"  target: {batch['target'].tolist()}")
print(f"  weight: {batch['weight'].tolist()}")
print(f"  id: {batch['id']}")

print("\n" + "=" * 70)
print("PART 2: Samplers")
print("=" * 70)

# Create an imbalanced dataset
torch.manual_seed(42)
# 1000 class-0, 100 class-1, 50 class-2
data = torch.randn(1150, 20)
labels = torch.cat([
    torch.zeros(1000, dtype=torch.long),  # Class 0: majority
    torch.ones(100, dtype=torch.long),  # Class 1: minority
    torch.full((50,), 2, dtype=torch.long),  # Class 2: rare
])
imbalanced_dataset = TensorDataset(data, labels)
print(f"\nImbalanced dataset: {len(imbalanced_dataset)} samples")
print(f"  Class 0: 1000 (87%)")
print(f"  Class 1: 100  (8.7%)")
print(f"  Class 2: 50   (4.3%)")

# --- WeightedRandomSampler ---
print("\n--- WeightedRandomSampler (for class imbalance) ---")

# Calculate sample weights: inverse class frequency
class_counts = torch.tensor([1000.0, 100.0, 50.0])
class_weights = 1.0 / class_counts
sample_weights = class_weights[labels]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(imbalanced_dataset),  # Sample same number as dataset
    replacement=True,  # Must be True for oversampling
)

loader = DataLoader(imbalanced_dataset, batch_size=64, sampler=sampler)

# Check class distribution in one epoch
all_labels = []
for _, batch_labels in loader:
    all_labels.append(batch_labels)
all_labels = torch.cat(all_labels)
print(f"  After weighted sampling (one epoch):")
for c in range(3):
    count = (all_labels == c).sum().item()
    print(f"    Class {c}: {count} samples ({count/len(all_labels)*100:.1f}%)")
print("  (Classes are now roughly balanced!)")

# --- SubsetRandomSampler (for train/val split) ---
print("\n--- SubsetRandomSampler (train/val split without copying data) ---")

indices = torch.randperm(len(imbalanced_dataset)).tolist()
train_indices = indices[:900]
val_indices = indices[900:]

train_loader = DataLoader(
    imbalanced_dataset, batch_size=32,
    sampler=SubsetRandomSampler(train_indices)
)
val_loader = DataLoader(
    imbalanced_dataset, batch_size=64,
    sampler=SubsetRandomSampler(val_indices)
)
print(f"  Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# --- BatchSampler (custom batch construction) ---
print("\n--- BatchSampler (group samples into custom batches) ---")

# Sort by "length" (simulating similar-length batching for sequences)
fake_lengths = torch.randint(5, 50, (200,))
sorted_indices = fake_lengths.argsort().tolist()

# Create batches of sorted samples (bucket batching)
batch_sampler = BatchSampler(
    sampler=sorted_indices,  # Pre-sorted indices
    batch_size=16,
    drop_last=False,
)

simple_dataset = TensorDataset(torch.randn(200, 10), fake_lengths)
loader = DataLoader(simple_dataset, batch_sampler=batch_sampler)

print(f"  Bucket batching by length:")
for i, (features, lengths) in enumerate(loader):
    if i < 3:
        print(f"    Batch {i}: lengths range [{lengths.min().item()}, {lengths.max().item()}]")
    if i >= 5:
        break
print("  (Samples within each batch have similar 'lengths')")

# --- RandomSampler with replacement (bootstrap) ---
print("\n--- RandomSampler with replacement (bootstrap) ---")
small_dataset = TensorDataset(torch.randn(50, 5), torch.arange(50))
bootstrap_sampler = RandomSampler(
    small_dataset, replacement=True, num_samples=200
)
loader = DataLoader(small_dataset, batch_size=50, sampler=bootstrap_sampler)
all_indices = []
for _, indices in loader:
    all_indices.append(indices)
all_indices = torch.cat(all_indices)
unique_seen = len(all_indices.unique())
print(f"  Dataset size: 50, Sampled: 200 (with replacement)")
print(f"  Unique samples seen: {unique_seen}/50")
print(f"  Some samples seen multiple times (bootstrap)")

print("\n" + "=" * 70)
print("PART 3: pin_memory and Data Transfer")
print("=" * 70)

print("""
pin_memory=True allocates batch tensors in page-locked (pinned) memory.
This makes CPU -> GPU transfers faster because:
  - Pinned memory can be transferred via DMA (direct memory access)
  - Regular memory must first be copied to pinned memory before transfer
  - With pinned memory, the intermediate copy is eliminated

Usage:
  loader = DataLoader(..., pin_memory=True)
  for batch in loader:
      batch = batch.to(device, non_blocking=True)  # Non-blocking transfer

When to use:
  - ALWAYS when training on GPU
  - NOT needed for CPU-only training

Note: pin_memory uses more RAM (page-locked memory can't be swapped out)
""")

# Demonstrate pin_memory behavior (CPU only, but shows the concept)
dataset = TensorDataset(torch.randn(100, 50), torch.randint(0, 5, (100,)))

loader_pinned = DataLoader(dataset, batch_size=32, pin_memory=True)
loader_normal = DataLoader(dataset, batch_size=32, pin_memory=False)

batch_pinned = next(iter(loader_pinned))
batch_normal = next(iter(loader_normal))

print(f"\n  pin_memory=True:  is_pinned={batch_pinned[0].is_pinned()}")
print(f"  pin_memory=False: is_pinned={batch_normal[0].is_pinned()}")

print("\n" + "=" * 70)
print("PART 4: persistent_workers and prefetch_factor")
print("=" * 70)

print("""
persistent_workers=True:
  - Keeps worker processes alive between epochs
  - Avoids the cost of starting workers at each epoch
  - Workers maintain their state (cached data, opened files)
  - Significant speedup for small datasets with many epochs

prefetch_factor (default=2):
  - Number of batches each worker pre-loads ahead of time
  - Higher values hide I/O latency but use more memory
  - Lower values reduce memory but may cause GPU stalls
""")


class TimedDataset(Dataset):
    """Dataset that tracks access patterns."""

    def __init__(self, size=500):
        self.size = size
        self.data = torch.randn(size, 100)
        self.labels = torch.randint(0, 10, (size,))
        self.access_count = 0

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        self.access_count += 1
        return self.data[idx], self.labels[idx]


# Demonstrate epoch overhead
dataset = TimedDataset(200)

# Non-persistent workers: workers restart each epoch
print("\n--- Non-persistent workers (num_workers=2) ---")
loader = DataLoader(dataset, batch_size=32, num_workers=2, persistent_workers=False)
start = time.time()
for epoch in range(3):
    for batch in loader:
        pass
non_persistent_time = time.time() - start
print(f"  3 epochs: {non_persistent_time:.3f}s")

# Persistent workers: workers stay alive
print("\n--- Persistent workers (num_workers=2) ---")
loader = DataLoader(dataset, batch_size=32, num_workers=2, persistent_workers=True)
start = time.time()
for epoch in range(3):
    for batch in loader:
        pass
persistent_time = time.time() - start
print(f"  3 epochs: {persistent_time:.3f}s")
print(f"  (Persistent workers avoid worker restart overhead)")

print("\n" + "=" * 70)
print("PART 5: Worker Initialization")
print("=" * 70)

print("""
worker_init_fn: Called in each worker process after creation.
Common uses:
  - Set different random seeds per worker
  - Open database connections
  - Initialize worker-specific resources
""")


class SeededDataset(Dataset):
    """Dataset that uses randomness in __getitem__."""

    def __init__(self, size=100):
        self.size = size
        self.base_data = torch.randn(size, 10)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Add random noise (randomness depends on worker seed)
        noise = torch.randn(10) * 0.01
        return self.base_data[idx] + noise, idx


def seed_worker(worker_id):
    """Set unique seed for each worker."""
    worker_seed = torch.initial_seed() % 2 ** 32
    import random
    random.seed(worker_seed)
    # If using numpy: np.random.seed(worker_seed)


# Reproducible data loading
g = torch.Generator()
g.manual_seed(42)

loader = DataLoader(
    SeededDataset(100),
    batch_size=16,
    num_workers=2,
    worker_init_fn=seed_worker,
    generator=g,
)

# Verify reproducibility
batches_run1 = [batch[1].tolist() for batch in DataLoader(
    SeededDataset(100), batch_size=16, shuffle=True,
    generator=torch.Generator().manual_seed(42)
)]
batches_run2 = [batch[1].tolist() for batch in DataLoader(
    SeededDataset(100), batch_size=16, shuffle=True,
    generator=torch.Generator().manual_seed(42)
)]
print(f"\n  Reproducible ordering: {batches_run1[0] == batches_run2[0]}")

print("\n" + "=" * 70)
print("PART 6: Performance Benchmarking")
print("=" * 70)


class ProcessingDataset(Dataset):
    """Dataset with adjustable processing cost."""

    def __init__(self, size=500, processing_steps=100):
        self.size = size
        self.data = torch.randn(size, 64)
        self.labels = torch.randint(0, 10, (size,))
        self.processing_steps = processing_steps

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = self.data[idx]
        # Simulate some processing
        for _ in range(self.processing_steps):
            x = x * 1.001
        return x, self.labels[idx]


dataset = ProcessingDataset(300, processing_steps=50)

print("\nBenchmarking DataLoader configurations:")
print(f"{'Configuration':<40} {'Time (s)':<10} {'Samples/sec':<12}")
print("-" * 62)

configs = [
    ("num_workers=0", {"batch_size": 32, "num_workers": 0}),
    ("num_workers=2", {"batch_size": 32, "num_workers": 2}),
    ("num_workers=4", {"batch_size": 32, "num_workers": 4}),
    ("nw=2, persistent", {"batch_size": 32, "num_workers": 2, "persistent_workers": True}),
    ("batch_size=64, nw=2", {"batch_size": 64, "num_workers": 2}),
    ("batch_size=128, nw=2", {"batch_size": 128, "num_workers": 2}),
]

for name, kwargs in configs:
    loader = DataLoader(dataset, **kwargs)
    # Warm up
    for batch in loader:
        break

    start = time.time()
    total_samples = 0
    for epoch in range(2):
        for batch_x, batch_y in loader:
            total_samples += batch_x.shape[0]
    elapsed = time.time() - start
    samples_per_sec = total_samples / elapsed
    print(f"{name:<40} {elapsed:<10.3f} {samples_per_sec:<12.0f}")

print("\n" + "=" * 70)
print("PART 7: DataLoader with GPU Training Pattern")
print("=" * 70)

print("""
Full training pattern with DataLoader (CPU version shown, GPU would add .to(device)):

1. Create Dataset
2. Split into train/val
3. Create DataLoaders with appropriate settings
4. Training loop with proper eval
""")

import torch.nn as nn
import torch.optim as optim

# Setup
torch.manual_seed(42)
X = torch.randn(1000, 20)
y = (X[:, :5].sum(dim=1) > 0).long()  # Binary classification

full_dataset = TensorDataset(X, y)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# DataLoaders optimized for training
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=2,
    pin_memory=False,  # Set True for GPU
    drop_last=True,
    persistent_workers=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=128,  # Larger batch for eval (no gradients stored)
    shuffle=False,
    num_workers=2,
    pin_memory=False,  # Set True for GPU
    persistent_workers=True,
)

# Model
model = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 2),
)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print(f"\nTraining with DataLoader:")
print(f"  Train samples: {len(train_dataset)}, batches: {len(train_loader)}")
print(f"  Val samples: {len(val_dataset)}, batches: {len(val_loader)}")
print(f"\n{'Epoch':<8}{'Train Loss':<12}{'Train Acc':<12}{'Val Loss':<12}{'Val Acc':<12}")
print("-" * 56)

for epoch in range(10):
    # Training phase
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch_x.size(0)
        train_correct += (output.argmax(1) == batch_y).sum().item()
        train_total += batch_x.size(0)

    # Validation phase
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

    if epoch % 2 == 0:
        print(f"{epoch:<8}"
              f"{train_loss/train_total:<12.4f}"
              f"{train_correct/train_total:<12.3f}"
              f"{val_loss/val_total:<12.4f}"
              f"{val_correct/val_total:<12.3f}")

print("\n" + "=" * 70)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
print("=" * 70)
