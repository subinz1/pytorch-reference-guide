"""
Data Loading — Dataset, DataLoader, and Augmentation
======================================================
Covers: custom Dataset, DataLoader best practices, collate functions.
"""

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

print("=" * 60)
print("1. BASIC TENSORDATASET + DATALOADER")
print("=" * 60)

X = torch.randn(1000, 10)
y = torch.randint(0, 3, (1000,))

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

batch_x, batch_y = next(iter(loader))
print(f"Dataset size: {len(dataset)}")
print(f"Batch X shape: {batch_x.shape}")
print(f"Batch y shape: {batch_y.shape}")

print("\n" + "=" * 60)
print("2. CUSTOM DATASET")
print("=" * 60)

class SineDataset(Dataset):
    """Generates noisy sine wave data."""

    def __init__(self, n_samples=1000, noise=0.1):
        self.x = torch.linspace(0, 4 * 3.14159, n_samples).unsqueeze(1)
        self.y = torch.sin(self.x) + noise * torch.randn_like(self.x)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

sine_ds = SineDataset(500)
print(f"SineDataset length: {len(sine_ds)}")
print(f"Sample: x={sine_ds[0][0].item():.3f}, y={sine_ds[0][1].item():.3f}")

print("\n" + "=" * 60)
print("3. TRAIN/VAL SPLIT")
print("=" * 60)

full_dataset = SineDataset(1000)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches:   {len(val_loader)}")

print("\n" + "=" * 60)
print("4. DATALOADER BEST PRACTICES")
print("=" * 60)

# Full-featured DataLoader for training
best_loader = DataLoader(
    train_ds,
    batch_size=64,
    shuffle=True,              # Random order each epoch
    num_workers=0,             # Set to 4+ for real workloads
    pin_memory=False,          # Set True with GPU
    drop_last=True,            # Drop incomplete last batch
    # persistent_workers=True, # Keep workers alive (with num_workers>0)
    # prefetch_factor=2,       # Prefetch batches (with num_workers>0)
)

for i, (bx, by) in enumerate(best_loader):
    if i == 0:
        print(f"First batch: x={bx.shape}, y={by.shape}")
    if i >= 2:
        break

print(f"Total batches (drop_last=True): {len(best_loader)}")

print("\n" + "=" * 60)
print("5. CUSTOM COLLATE FUNCTION")
print("=" * 60)

class VariableLengthDataset(Dataset):
    def __init__(self, n=100):
        self.data = [torch.randn(torch.randint(5, 20, (1,)).item()) for _ in range(n)]
        self.labels = torch.randint(0, 3, (n,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def pad_collate(batch):
    """Pad variable-length sequences to same length."""
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences])
    padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    return padded, torch.stack(list(labels)), lengths

var_ds = VariableLengthDataset(100)
var_loader = DataLoader(var_ds, batch_size=8, collate_fn=pad_collate)

padded, labels, lengths = next(iter(var_loader))
print(f"Padded batch shape: {padded.shape}")
print(f"Labels:  {labels}")
print(f"Lengths: {lengths}")

print("\n" + "=" * 60)
print("6. ITERATING WITH EPOCHS")
print("=" * 60)

loader = DataLoader(SineDataset(200), batch_size=32, shuffle=True)

for epoch in range(3):
    total_samples = 0
    for bx, by in loader:
        total_samples += bx.size(0)
    print(f"Epoch {epoch+1}: processed {total_samples} samples")

print("\nDone!")
