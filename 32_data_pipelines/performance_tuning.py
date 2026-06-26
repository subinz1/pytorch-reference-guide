"""
DataLoader Performance Tuning — Benchmarks, Bucketing, and Bottleneck Detection
=================================================================================
Practical techniques for making data loading fast.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
import time
import math

print("=" * 65)
print("1. BENCHMARK: num_workers EFFECT")
print("=" * 65)

class SyntheticDataset(Dataset):
    def __init__(self, size=2000, dim=256):
        self.data = torch.randn(size, dim)
        self.labels = torch.randint(0, 10, (size,))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        time.sleep(0.0001)
        return self.data[idx], self.labels[idx]

dataset = SyntheticDataset(1000)

for nw in [0, 1, 2, 4]:
    loader = DataLoader(dataset, batch_size=64, num_workers=nw)
    start = time.perf_counter()
    for batch_x, batch_y in loader:
        pass
    elapsed = (time.perf_counter() - start) * 1000
    print(f"  num_workers={nw}: {elapsed:.0f} ms")

print("\n" + "=" * 65)
print("2. PIN_MEMORY AND PREFETCH_FACTOR")
print("=" * 65)

print("""
pin_memory=True:
  Allocates tensors in page-locked (pinned) memory.
  Enables faster CPU→GPU transfer via DMA.
  Always use for GPU training. No effect on CPU-only.

prefetch_factor=N (default=2):
  Each worker prefetches N batches ahead.
  Increase for slow I/O (network storage, complex transforms).
  Decrease for memory-constrained setups.

Best practice:
  DataLoader(dataset, batch_size=64, num_workers=4,
             pin_memory=True, persistent_workers=True,
             prefetch_factor=2)
""")

print("=" * 65)
print("3. DETECTING DATA LOADING BOTTLENECK")
print("=" * 65)

model = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))
loader = DataLoader(SyntheticDataset(500, 256), batch_size=64, num_workers=0)

data_time_total = 0
compute_time_total = 0
end = time.perf_counter()

for batch_x, batch_y in loader:
    data_time = time.perf_counter() - end
    data_time_total += data_time

    start = time.perf_counter()
    with torch.no_grad():
        _ = model(batch_x)
    compute_time = time.perf_counter() - start
    compute_time_total += compute_time

    end = time.perf_counter()

total = data_time_total + compute_time_total
print(f"Data loading:  {data_time_total*1000:.0f} ms ({data_time_total/total*100:.0f}%)")
print(f"Compute:       {compute_time_total*1000:.0f} ms ({compute_time_total/total*100:.0f}%)")
if data_time_total > compute_time_total:
    print("→ DATA-BOUND: increase num_workers, use pin_memory, faster storage")
else:
    print("→ COMPUTE-BOUND: data loading is not the bottleneck")

print("\n" + "=" * 65)
print("4. BUCKETED BATCHING (Variable-Length Sequences)")
print("=" * 65)

class VariableLengthDataset(Dataset):
    def __init__(self, n=500):
        self.lengths = torch.randint(5, 100, (n,))
        self.data = [torch.randn(l.item(), 32) for l in self.lengths]
        self.labels = torch.randint(0, 5, (n,))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class BucketSampler(Sampler):
    """Groups sequences by similar length to minimize padding waste."""
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.lengths = [len(d) for d, _ in dataset]
        self.sorted_indices = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])

    def __iter__(self):
        batches = []
        for i in range(0, len(self.sorted_indices), self.batch_size):
            batch = self.sorted_indices[i:i + self.batch_size]
            batches.append(batch)
        import random
        random.shuffle(batches)
        for batch in batches:
            yield from batch

    def __len__(self):
        return len(self.sorted_indices)

def pad_collate(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    padded = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    return padded, torch.stack(list(labels)), lengths

var_dataset = VariableLengthDataset(200)

# Without bucketing
loader_naive = DataLoader(var_dataset, batch_size=16, shuffle=True, collate_fn=pad_collate)
total_padding_naive = 0
total_elements_naive = 0
for padded, labels, lengths in loader_naive:
    total_elements_naive += padded.numel()
    real_elements = sum(l.item() * padded.shape[2] for l in lengths)
    total_padding_naive += padded.numel() - real_elements

# With bucketing
loader_bucket = DataLoader(var_dataset, batch_size=16,
                           sampler=BucketSampler(var_dataset, 16),
                           collate_fn=pad_collate)
total_padding_bucket = 0
total_elements_bucket = 0
for padded, labels, lengths in loader_bucket:
    total_elements_bucket += padded.numel()
    real_elements = sum(l.item() * padded.shape[2] for l in lengths)
    total_padding_bucket += padded.numel() - real_elements

waste_naive = total_padding_naive / total_elements_naive * 100
waste_bucket = total_padding_bucket / total_elements_bucket * 100
print(f"Random batching:   {waste_naive:.1f}% padding waste")
print(f"Bucketed batching: {waste_bucket:.1f}% padding waste")
print(f"Saved: {waste_naive - waste_bucket:.1f}% less wasted computation")

print("\n" + "=" * 65)
print("5. DYNAMIC BATCHING BY TOKEN COUNT")
print("=" * 65)

def dynamic_batch_by_tokens(dataset, max_tokens=1024):
    """Create batches with roughly equal total tokens instead of equal sample count."""
    indices = sorted(range(len(dataset)), key=lambda i: len(dataset[i][0]))
    batches = []
    current_batch = []
    current_tokens = 0
    max_len = 0

    for idx in indices:
        seq_len = len(dataset[idx][0])
        new_max = max(max_len, seq_len)
        new_tokens = new_max * (len(current_batch) + 1)

        if new_tokens > max_tokens and current_batch:
            batches.append(current_batch)
            current_batch = [idx]
            current_tokens = seq_len
            max_len = seq_len
        else:
            current_batch.append(idx)
            max_len = new_max
            current_tokens = new_tokens

    if current_batch:
        batches.append(current_batch)

    return batches

batches = dynamic_batch_by_tokens(var_dataset, max_tokens=2048)
batch_sizes = [len(b) for b in batches]
print(f"Dynamic batching: {len(batches)} batches")
print(f"Batch sizes: min={min(batch_sizes)}, max={max(batch_sizes)}, "
      f"mean={sum(batch_sizes)/len(batch_sizes):.1f}")

print("\n" + "=" * 65)
print("6. CURRICULUM SAMPLER")
print("=" * 65)

class CurriculumSampler(Sampler):
    """Start with easy examples, gradually include harder ones."""
    def __init__(self, difficulty_scores, epoch, total_epochs):
        self.difficulty_scores = difficulty_scores
        self.progress = min(epoch / total_epochs, 1.0)
        sorted_indices = sorted(range(len(difficulty_scores)),
                                key=lambda i: difficulty_scores[i])
        cutoff = int(len(sorted_indices) * (0.3 + 0.7 * self.progress))
        self.indices = sorted_indices[:cutoff]

    def __iter__(self):
        import random
        perm = list(self.indices)
        random.shuffle(perm)
        return iter(perm)

    def __len__(self):
        return len(self.indices)

difficulties = torch.rand(200).tolist()
for epoch in [0, 5, 10, 20]:
    sampler = CurriculumSampler(difficulties, epoch, total_epochs=20)
    print(f"  Epoch {epoch:2d}: training on {len(sampler):3d}/{len(difficulties)} "
          f"samples ({len(sampler)/len(difficulties)*100:.0f}%)")

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)

print("""
Performance Tuning Checklist:
  1. Set num_workers=4-8 (benchmark for your setup)
  2. pin_memory=True for GPU training
  3. persistent_workers=True to avoid restart overhead
  4. prefetch_factor=2+ for slow I/O
  5. Bucketed batching for variable-length sequences
  6. Dynamic batching by token count for LLMs
  7. Profile: is data loading > 10% of step time? Fix it.
""")
print("Done!")
