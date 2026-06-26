"""
Module 32 — Streaming Datasets & Memory-Mapped Data
====================================================

Demonstrates:
  1. IterableDataset with proper worker splitting
  2. Memory-mapped dataset (creates temp binary file, mmaps it)
  3. LLM-style chunked dataset (pack sequences into fixed-length chunks)
  4. Multi-worker seeding with worker_init_fn
  5. Comparison: map-style vs iterable dataset loading speed
  6. DistributedSampler pattern (API demonstration without dist init)

Run: python streaming_datasets.py
"""

import math
import os
import random
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    DistributedSampler,
    IterableDataset,
)

# ---------------------------------------------------------------------------
# 1. IterableDataset with Worker Splitting
# ---------------------------------------------------------------------------
print("=" * 70)
print("1. IterableDataset with Worker Splitting")
print("=" * 70)


class LineStreamDataset(IterableDataset):
    """Reads lines from text files, splitting work across DataLoader workers."""

    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            files = self.file_paths
        else:
            per_worker = int(math.ceil(len(self.file_paths) / worker_info.num_workers))
            start = worker_info.id * per_worker
            end = min(start + per_worker, len(self.file_paths))
            files = self.file_paths[start:end]

        for path in files:
            with open(path) as f:
                for line in f:
                    values = [float(x) for x in line.strip().split(",")]
                    yield torch.tensor(values)


tmpdir = tempfile.mkdtemp()
num_files = 4
lines_per_file = 50
features = 8

for i in range(num_files):
    path = os.path.join(tmpdir, f"data_{i}.csv")
    with open(path, "w") as f:
        for _ in range(lines_per_file):
            vals = ",".join(str(round(random.gauss(0, 1), 4)) for _ in range(features))
            f.write(vals + "\n")

file_paths = [os.path.join(tmpdir, f"data_{i}.csv") for i in range(num_files)]
stream_ds = LineStreamDataset(file_paths)

loader = DataLoader(stream_ds, batch_size=16, num_workers=2)
total = 0
for batch in loader:
    total += batch.size(0)
print(f"  Loaded {total} samples from {num_files} files (2 workers)")
print(f"  Batch shape: {batch.shape}")

loader_single = DataLoader(stream_ds, batch_size=16, num_workers=0)
total_single = 0
for batch in loader_single:
    total_single += batch.size(0)
print(f"  Single-worker loaded {total_single} samples")
print()


# ---------------------------------------------------------------------------
# 2. Memory-Mapped Dataset
# ---------------------------------------------------------------------------
print("=" * 70)
print("2. Memory-Mapped Dataset")
print("=" * 70)


class MemoryMappedDataset(Dataset):
    """Map-style dataset backed by a memory-mapped binary file."""

    def __init__(self, path, feature_dim, dtype=np.float32):
        self.data = np.memmap(path, dtype=dtype, mode="r")
        self.feature_dim = feature_dim
        self.n_samples = len(self.data) // feature_dim

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.feature_dim
        end = start + self.feature_dim
        row = self.data[start:end].copy()
        x = torch.from_numpy(row[:-1])
        y = torch.tensor(row[-1])
        return x, y


num_samples = 10000
feature_dim = 16
mmap_path = os.path.join(tmpdir, "mmap_data.bin")

arr = np.random.randn(num_samples * feature_dim).astype(np.float32)
arr.tofile(mmap_path)

mmap_ds = MemoryMappedDataset(mmap_path, feature_dim)
print(f"  Dataset size: {len(mmap_ds)} samples")
print(f"  File size: {os.path.getsize(mmap_path) / 1024:.1f} KB")

x, y = mmap_ds[0]
print(f"  Sample: x.shape={x.shape}, y.shape={y.shape}")

mmap_loader = DataLoader(mmap_ds, batch_size=64, num_workers=2, shuffle=True)
batch_x, batch_y = next(iter(mmap_loader))
print(f"  Batch: x.shape={batch_x.shape}, y.shape={batch_y.shape}")
print(f"  Startup cost: ~0ms (memory-mapped, no loading)")
print()


# ---------------------------------------------------------------------------
# 3. LLM-Style Chunked Dataset
# ---------------------------------------------------------------------------
print("=" * 70)
print("3. LLM-Style Chunked Dataset")
print("=" * 70)


class ChunkedTokenDataset(Dataset):
    """
    Packs pre-tokenized data into fixed-length chunks for causal LM training.
    Input (x) is chunk[:-1], target (y) is chunk[1:] (next-token prediction).
    """

    def __init__(self, token_file, chunk_size=128, dtype=np.uint16):
        self.data = np.memmap(token_file, dtype=dtype, mode="r")
        self.chunk_size = chunk_size
        self.n_chunks = len(self.data) // chunk_size

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = idx * self.chunk_size
        chunk = self.data[start : start + self.chunk_size].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


vocab_size = 32000
total_tokens = 100_000
token_path = os.path.join(tmpdir, "tokens.bin")

fake_tokens = np.random.randint(0, vocab_size, size=total_tokens, dtype=np.uint16)
fake_tokens.tofile(token_path)

chunk_size = 128
chunk_ds = ChunkedTokenDataset(token_path, chunk_size=chunk_size)
print(f"  Total tokens: {total_tokens:,}")
print(f"  Chunk size: {chunk_size}")
print(f"  Number of chunks: {len(chunk_ds)}")
print(f"  Wasted tokens: {total_tokens - len(chunk_ds) * chunk_size}")

x, y = chunk_ds[0]
print(f"  x.shape={x.shape} (input), y.shape={y.shape} (target)")
print(f"  x[:5] = {x[:5].tolist()}")
print(f"  y[:5] = {y[:5].tolist()} (shifted by 1)")

chunk_loader = DataLoader(chunk_ds, batch_size=8, shuffle=True, num_workers=2)
bx, by = next(iter(chunk_loader))
print(f"  Batch: x.shape={bx.shape}, y.shape={by.shape}")
print()


# ---------------------------------------------------------------------------
# 4. Multi-Worker Seeding
# ---------------------------------------------------------------------------
print("=" * 70)
print("4. Multi-Worker Seeding with worker_init_fn")
print("=" * 70)


class AugmentedDataset(Dataset):
    def __init__(self, size=200):
        self.size = size
        self.data = torch.randn(size, 4)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = self.data[idx]
        noise = torch.randn_like(x) * 0.1
        return x + noise


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


aug_ds = AugmentedDataset(200)

g = torch.Generator()
g.manual_seed(42)
loader_seeded = DataLoader(
    aug_ds,
    batch_size=32,
    num_workers=2,
    worker_init_fn=seed_worker,
    generator=g,
)

batches_a = [b.clone() for b in loader_seeded]

g.manual_seed(42)
loader_seeded2 = DataLoader(
    aug_ds,
    batch_size=32,
    num_workers=2,
    worker_init_fn=seed_worker,
    generator=g,
)
batches_b = [b.clone() for b in loader_seeded2]

match_count = sum(
    1 for a, b in zip(batches_a, batches_b) if torch.allclose(a, b, atol=1e-6)
)
print(f"  Reproducible batches: {match_count}/{len(batches_a)}")
print(f"  (With proper seeding, runs are deterministic)")
print()


# ---------------------------------------------------------------------------
# 5. Map-Style vs Iterable Dataset Speed Comparison
# ---------------------------------------------------------------------------
print("=" * 70)
print("5. Map-Style vs Iterable Dataset Loading Speed")
print("=" * 70)


class MapDataset(Dataset):
    def __init__(self, file_paths, feature_dim):
        rows = []
        for path in file_paths:
            with open(path) as f:
                for line in f:
                    rows.append([float(x) for x in line.strip().split(",")])
        self.data = torch.tensor(rows)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


map_ds = MapDataset(file_paths, features)
iterable_ds = LineStreamDataset(file_paths)

for name, ds in [("Map-style", map_ds), ("Iterable", iterable_ds)]:
    loader = DataLoader(ds, batch_size=32, num_workers=0)
    t0 = time.perf_counter()
    for _ in range(5):
        for batch in loader:
            pass
    elapsed = time.perf_counter() - t0
    print(f"  {name:12s}: {elapsed*1000:.1f}ms (5 epochs)")

print()
print("  Map-style: faster for small data (all in RAM, random access)")
print("  Iterable:  better for large/streaming data (lazy, no random access)")
print()


# ---------------------------------------------------------------------------
# 6. DistributedSampler Pattern (API Demo)
# ---------------------------------------------------------------------------
print("=" * 70)
print("6. DistributedSampler Pattern (API Demo)")
print("=" * 70)

simple_ds = MapDataset(file_paths, features)

world_size = 4
for rank in range(world_size):
    sampler = DistributedSampler(
        simple_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42,
    )
    sampler.set_epoch(0)
    indices = list(sampler)
    print(f"  Rank {rank}: {len(indices)} samples, first 5 indices: {indices[:5]}")

overlap_01 = set(list(DistributedSampler(simple_ds, 4, 0, shuffle=False))) & set(
    list(DistributedSampler(simple_ds, 4, 1, shuffle=False))
)
print(f"  Overlap between rank 0 and rank 1: {len(overlap_01)} samples (should be 0)")

print()
print("  NOTE: DistributedSampler works without torch.distributed.init_process_group")
print("  by explicitly passing num_replicas and rank.")
print()


# ---------------------------------------------------------------------------
# Iterable + Distributed Pattern
# ---------------------------------------------------------------------------
print("=" * 70)
print("Bonus: IterableDataset with Rank + Worker Splitting")
print("=" * 70)


class DistributedIterableDataset(IterableDataset):
    """IterableDataset that splits across both ranks and workers."""

    def __init__(self, file_paths, rank=0, world_size=1):
        self.file_paths = file_paths
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        rank_files = self.file_paths[self.rank :: self.world_size]

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            rank_files = rank_files[worker_info.id :: worker_info.num_workers]

        for path in rank_files:
            with open(path) as f:
                for line in f:
                    values = [float(x) for x in line.strip().split(",")]
                    yield torch.tensor(values)


for rank in range(2):
    ds = DistributedIterableDataset(file_paths, rank=rank, world_size=2)
    loader = DataLoader(ds, batch_size=16, num_workers=0)
    count = sum(b.size(0) for b in loader)
    print(f"  Rank {rank}: {count} samples")

print(f"  Total across ranks: {sum(sum(b.size(0) for b in DataLoader(DistributedIterableDataset(file_paths, r, 2), batch_size=16)) for r in range(2))}")
print()


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
import shutil

shutil.rmtree(tmpdir)
print("Temp files cleaned up.")
print("Done! See performance_tuning.py for DataLoader benchmarking.")
