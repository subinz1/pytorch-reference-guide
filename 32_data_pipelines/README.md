# Module 32: Efficient Data Pipelines

<div align="center">

[← Previous Module (torchao)](../31_torchao/) | [🏠 Home](../README.md) | Next Module →

</div>

---

> **Prerequisites**: [Module 06 — Data Loading](../06_data_loading/), [Module 07 — Training Pipelines](../07_training/)
> **Time**: ~2 hours
> **Files**: `streaming_datasets.py`, `performance_tuning.py`

---

## Table of Contents

1. [Beyond Basic DataLoader](#1-beyond-basic-dataloader)
2. [IterableDataset](#2-iterabledataset)
3. [Memory-Mapped Files](#3-memory-mapped-files)
4. [Efficient Tokenization for LLMs](#4-efficient-tokenization-for-llms)
5. [Multi-Worker DataLoader](#5-multi-worker-dataloader)
6. [Prefetching](#6-prefetching)
7. [pin_memory and Non-Blocking Transfer](#7-pin_memory-and-non-blocking-transfer)
8. [Custom Samplers](#8-custom-samplers)
9. [Distributed Data Loading](#9-distributed-data-loading)
10. [DataLoader Performance Profiling](#10-dataloader-performance-profiling)
11. [Collate Optimization](#11-collate-optimization)
12. [Data Pipeline Patterns](#12-data-pipeline-patterns)
13. [Upstream Updates (June 23–25, 2026)](#13-upstream-updates-june-2326-2026)

---

## 1. Beyond Basic DataLoader

Module 06 covered the fundamentals: `Dataset`, `DataLoader`, custom collate functions, and basic samplers. That's enough for most research prototypes — datasets that fit in RAM, single-GPU training, moderate throughput requirements.

Production-scale training is different. Consider:

- **TB-scale datasets** that can't fit in RAM (or even on a single disk)
- **Streaming data** from databases, object stores, or log pipelines
- **Multi-GPU training** where each rank must see a disjoint slice of data
- **GPU utilization** — if data loading can't keep up, your expensive GPUs sit idle

This module covers the patterns and tools for building data pipelines that scale.

### The Data Loading Bottleneck

In a typical training loop:

```
┌─────────────┐     ┌──────────────┐     ┌──────────┐
│ Load Batch   │────▶│ Forward/Back │────▶│ Optimize │
│ (CPU/Disk)   │     │ (GPU)        │     │ (GPU)    │
└─────────────┘     └──────────────┘     └──────────┘
     ▲                                        │
     └────────────────────────────────────────┘
```

If loading takes longer than compute, the GPU blocks waiting for data. The goal is to ensure the next batch is always ready before the GPU finishes the current step.

### Key Metrics

| Metric | Target | Problem if missed |
|--------|--------|-------------------|
| GPU utilization | >90% | Data starvation |
| Data loading time | < compute time | GPU idle cycles |
| Memory usage | Stable over time | OOM from leaks |
| Worker utilization | Balanced | Stragglers slow everything |

---

## 2. IterableDataset

A standard `Dataset` (map-style) requires `__getitem__` and `__len__`. This assumes random access and known size — assumptions that break for streaming data.

`IterableDataset` replaces these with a single `__iter__` method:

```python
from torch.utils.data import IterableDataset, DataLoader

class LogStreamDataset(IterableDataset):
    def __init__(self, log_files):
        self.log_files = log_files

    def __iter__(self):
        for path in self.log_files:
            with open(path) as f:
                for line in f:
                    yield self.parse(line)

    def parse(self, line):
        # Convert raw log line to tensor
        return torch.tensor([float(x) for x in line.strip().split(',')])
```

### When to Use IterableDataset

| Use case | Map-style | Iterable |
|----------|-----------|----------|
| Data fits in RAM | ✓ | |
| Random access needed | ✓ | |
| Streaming/infinite data | | ✓ |
| Database queries | | ✓ |
| Very large file collections | | ✓ |
| Need `len()` for progress bars | ✓ | |

### Worker Splitting

With `num_workers > 0`, each worker gets a **full copy** of the `IterableDataset` object. Without explicit splitting, every worker yields the same data — duplicated batches:

```python
class ShardedStreamDataset(IterableDataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process loading
            files = self.file_list
        else:
            # Split files across workers
            per_worker = len(self.file_list) // worker_info.num_workers
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = start + per_worker if worker_id < worker_info.num_workers - 1 else len(self.file_list)
            files = self.file_list[start:end]

        for path in files:
            with open(path) as f:
                for line in f:
                    yield self.process(line)
```

### Combining with DistributedSampler

In distributed training, you need to split across both ranks **and** workers:

```python
def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # First: split by rank
    rank_files = self.file_list[rank::world_size]

    # Then: split by worker within this rank
    if worker_info is not None:
        rank_files = rank_files[worker_info.id::worker_info.num_workers]

    for path in rank_files:
        yield from self.read_file(path)
```

---

## 3. Memory-Mapped Files

For datasets too large for RAM but stored on disk, memory mapping lets the OS manage paging:

```python
import numpy as np
import torch

# Create a memory-mapped file (data stays on disk)
data = np.memmap('data.bin', dtype=np.int32, mode='r', shape=(num_tokens,))

# Access works like a normal array — OS pages data in/out
batch = torch.from_numpy(data[start:end].copy())
```

### How It Works

Memory mapping maps a file into virtual address space without loading it into RAM. The OS loads pages on demand and evicts them under memory pressure:

```
Virtual Address Space          Physical RAM           Disk
┌──────────────┐              ┌──────────┐       ┌──────────┐
│ Page 0       │──mapped──────│ Page 0   │◀──────│ Page 0   │
│ Page 1       │──page fault──│          │       │ Page 1   │
│ Page 2       │──mapped──────│ Page 2   │◀──────│ Page 2   │
│ ...          │              │          │       │ ...      │
│ Page N       │              └──────────┘       │ Page N   │
└──────────────┘                                 └──────────┘
```

### Benefits

- **Near-zero startup time** — no loading delay, just mmap the file
- **Automatic memory management** — OS handles page eviction
- **Efficient for sequential access** — OS prefetches sequentially
- **Shared across processes** — multiple workers share the same pages

### torch.from_file

PyTorch has a built-in memory-mapped tensor:

```python
# Create a storage backed by a file
storage = torch.FloatStorage.from_file('weights.bin', shared=False, size=num_elements)
tensor = torch.tensor(storage).reshape(shape)
```

### Best Practices

1. **Pre-process into binary format** — tokenize text, encode images, save as contiguous binary
2. **Use fixed-size records** — enables O(1) random access by index
3. **Align to page boundaries** — typically 4KB on Linux
4. **Copy before modifying** — `data[i:j].copy()` avoids modifying the mmap

---

## 4. Efficient Tokenization for LLMs

Tokenizing on-the-fly during training wastes compute. The pattern for LLM training:

### Step 1: Pre-tokenize Offline

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8b")

tokens = []
for doc in documents:
    tokens.extend(tokenizer.encode(doc))
    tokens.append(tokenizer.eos_token_id)

# Save as binary
arr = np.array(tokens, dtype=np.uint16)
arr.tofile('train_tokens.bin')
```

### Step 2: Pack into Fixed-Length Chunks

Rather than padding each document to `max_seq_len` (wasteful), concatenate all tokens and slice into chunks:

```python
class ChunkedTokenDataset(torch.utils.data.Dataset):
    def __init__(self, token_file, chunk_size=2048):
        self.data = np.memmap(token_file, dtype=np.uint16, mode='r')
        self.chunk_size = chunk_size
        self.n_chunks = len(self.data) // chunk_size

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = idx * self.chunk_size
        chunk = self.data[start:start + self.chunk_size].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y
```

This packing approach wastes zero tokens — every training sample is exactly `chunk_size` tokens with no padding.

### Document Boundaries

The above approach ignores document boundaries (attention spans across documents). For models sensitive to this, insert boundary markers or use attention masks:

```python
# Track document boundaries within each chunk
boundaries = []
pos = 0
for doc_len in doc_lengths:
    pos += doc_len + 1  # +1 for EOS
    if pos >= chunk_size:
        break
    boundaries.append(pos)
```

---

## 5. Multi-Worker DataLoader

`num_workers > 0` spawns separate processes that load data in parallel:

```
Main Process                 Worker Processes
┌───────────────┐     ┌────────────────────┐
│ Training Loop │     │ Worker 0: load,    │
│               │◀────│   transform, send  │
│ GPU compute   │     ├────────────────────┤
│               │◀────│ Worker 1: load,    │
│ Optimizer     │     │   transform, send  │
│               │     ├────────────────────┤
│               │◀────│ Worker 2: load,    │
│               │     │   transform, send  │
└───────────────┘     └────────────────────┘
       IPC via shared memory / pipes
```

### How Workers Operate

1. Each worker is a separate process (forked or spawned)
2. Workers prefetch `prefetch_factor` batches each
3. Data transfers via shared memory (tensors) or pipes (other Python objects)
4. The main process consumes batches round-robin from workers

### Common Pitfalls

**RNG Seeding**: By default, each worker inherits the same random seed. This means augmentations are correlated across workers:

```python
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

loader = DataLoader(
    dataset,
    num_workers=4,
    worker_init_fn=seed_worker,
    generator=torch.Generator().manual_seed(42),
)
```

**File Handle Leaks**: Opening files in `__init__` and forking creates shared file descriptors. Open files in `__iter__` or `__getitem__` instead, or use `worker_init_fn` to open per-worker handles.

**Memory Growth**: Workers that accumulate state (caches, buffers) can grow memory over time. Use `persistent_workers=True` to avoid restarting workers each epoch — but monitor memory:

```python
loader = DataLoader(
    dataset,
    num_workers=4,
    persistent_workers=True,  # Workers survive across epochs
)
```

### Choosing num_workers

Rule of thumb: start with `num_workers = num_cpu_cores` and benchmark. Too few workers starve the GPU; too many cause contention on I/O and CPU cache thrashing.

---

## 6. Prefetching

Each worker prefetches `prefetch_factor` batches ahead of consumption:

```python
loader = DataLoader(
    dataset,
    num_workers=4,
    prefetch_factor=2,  # Default: each worker prefetches 2 batches
)
```

### How It Works

```
Time ──────────────────────────────────────────▶

Worker 0: [Load B0] [Load B4] [Load B8]  ...
Worker 1: [Load B1] [Load B5] [Load B9]  ...
Worker 2: [Load B2] [Load B6] [Load B10] ...
Worker 3: [Load B3] [Load B7] [Load B11] ...

Queue:    B0 B1 B2 B3 | B4 B5 B6 B7 | ...
                       ▲
           Main process consumes from here
```

With `prefetch_factor=2`, each of the 4 workers has 2 batches in flight, so the queue holds up to 8 ready batches.

### When to Increase prefetch_factor

- **Slow I/O** (network storage, spinning disks) — increase to 4-8
- **Variable loading time** — higher prefetch smooths out spikes
- **Memory pressure** — lower prefetch reduces memory usage

### GPU Prefetching

CPU-side prefetching is only half the story. To overlap CPU→GPU transfer with GPU compute:

```python
loader = DataLoader(dataset, pin_memory=True, num_workers=4)

for batch in loader:
    # Non-blocking transfer overlaps with GPU compute
    x = batch.to(device, non_blocking=True)
    output = model(x)
```

---

## 7. pin_memory and Non-Blocking Transfer

### What Is Pinned Memory?

Normal (pageable) memory can be swapped to disk by the OS. GPU transfers from pageable memory require an extra copy through a staging buffer:

```
Pageable Memory ──copy──▶ Pinned Buffer ──DMA──▶ GPU Memory
                (CPU)                     (PCIe)
```

Pinned (page-locked) memory skips the staging copy:

```
Pinned Memory ──────────DMA──────────▶ GPU Memory
              (direct PCIe transfer)
```

### Using pin_memory in DataLoader

```python
loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=4,
    pin_memory=True,     # Allocate batches in pinned memory
)

for data, target in loader:
    # Non-blocking: starts transfer, returns immediately
    data = data.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    # GPU compute can overlap with the transfer
    output = model(data)
    loss = criterion(output, target)
```

### When pin_memory Helps

It helps **always** for GPU training. The overhead of pinning is negligible compared to the transfer speedup. The combination of `pin_memory=True` + `non_blocking=True` enables overlap of data transfer and computation.

### Caveats

- Pinned memory is a limited resource — don't pin large tensors unnecessarily
- `non_blocking=True` requires a CUDA stream synchronization before using the tensor on CPU again
- Custom collate functions that return non-tensor objects won't benefit from pin_memory

---

## 8. Custom Samplers

Samplers control the order indices are fed to the DataLoader.

### WeightedRandomSampler (Class Imbalance)

```python
from torch.utils.data import WeightedRandomSampler

# Class counts: [9000, 500, 500] — heavily imbalanced
class_weights = [1.0/9000, 1.0/500, 1.0/500]
sample_weights = [class_weights[label] for label in all_labels]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(all_labels),
    replacement=True,
)

loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

### Curriculum Learning Sampler

Train on easy examples first, progressively introduce harder ones:

```python
class CurriculumSampler(torch.utils.data.Sampler):
    def __init__(self, difficulties, epoch=0, total_epochs=10):
        self.difficulties = difficulties
        self.epoch = epoch
        self.total_epochs = total_epochs

    def __iter__(self):
        # Fraction of data available increases with epoch
        fraction = min(1.0, 0.3 + 0.7 * self.epoch / self.total_epochs)
        threshold = sorted(self.difficulties)[int(len(self.difficulties) * fraction) - 1]

        indices = [i for i, d in enumerate(self.difficulties) if d <= threshold]
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        fraction = min(1.0, 0.3 + 0.7 * self.epoch / self.total_epochs)
        return int(len(self.difficulties) * fraction)

    def set_epoch(self, epoch):
        self.epoch = epoch
```

### Hard Example Mining Sampler

Over-sample examples with high loss:

```python
class HardExampleSampler(torch.utils.data.Sampler):
    def __init__(self, dataset_size, initial_weights=None):
        self.weights = initial_weights or torch.ones(dataset_size)

    def update_weights(self, indices, losses):
        for idx, loss in zip(indices, losses):
            self.weights[idx] = loss.item()

    def __iter__(self):
        probs = self.weights / self.weights.sum()
        indices = torch.multinomial(probs, len(self.weights), replacement=True)
        return iter(indices.tolist())

    def __len__(self):
        return len(self.weights)
```

---

## 9. Distributed Data Loading

### DistributedSampler

Ensures each GPU processes a disjoint subset of the data:

```python
from torch.utils.data import DistributedSampler

sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    drop_last=True,
)

loader = DataLoader(dataset, batch_size=32, sampler=sampler)

for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # CRITICAL: different shuffle each epoch
    for batch in loader:
        train_step(batch)
```

### Why set_epoch Matters

Without `set_epoch()`, every epoch uses the same shuffle permutation. Each rank always sees the same subset of data — effectively training on 1/world_size of the dataset:

```python
# Without set_epoch: same permutation every epoch
# Rank 0 always sees indices [0, 3, 6, 9, ...]
# Rank 1 always sees indices [1, 4, 7, 10, ...]

# With set_epoch: different permutation each epoch
# Epoch 0 Rank 0: [5, 2, 8, 1, ...]
# Epoch 1 Rank 0: [3, 9, 0, 7, ...]
```

### Sharded Data Files

For very large datasets, pre-shard data files so each rank reads different files:

```python
class ShardedDataset(torch.utils.data.Dataset):
    def __init__(self, shard_dir, rank, world_size):
        all_shards = sorted(Path(shard_dir).glob('shard_*.bin'))
        self.shards = all_shards[rank::world_size]
        self.data = np.concatenate([
            np.memmap(s, dtype=np.int32, mode='r') for s in self.shards
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx])
```

This avoids the DistributedSampler overhead entirely — no cross-rank coordination needed.

---

## 10. DataLoader Performance Profiling

### Detecting the Bottleneck

If GPU utilization is below 90%, data loading is likely the bottleneck. Measure it:

```python
import time

data_times = []
compute_times = []

for batch in loader:
    t0 = time.perf_counter()
    x, y = batch[0].to(device), batch[1].to(device)
    t1 = time.perf_counter()

    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    torch.cuda.synchronize()
    t2 = time.perf_counter()

    data_times.append(t1 - t0)
    compute_times.append(t2 - t1)

avg_data = sum(data_times) / len(data_times)
avg_compute = sum(compute_times) / len(compute_times)
print(f"Data: {avg_data*1000:.1f}ms  Compute: {avg_compute*1000:.1f}ms")
print(f"Data fraction: {avg_data/(avg_data+avg_compute)*100:.1f}%")
```

### Reading the Results

| Data % | Diagnosis | Fix |
|--------|-----------|-----|
| < 10% | Compute-bound (good) | Focus on model optimization |
| 10-30% | Mild bottleneck | More workers, prefetching |
| 30-50% | Significant bottleneck | Pre-process data, faster storage |
| > 50% | Severe bottleneck | Restructure pipeline entirely |

### Solutions by Severity

1. **Quick wins**: Increase `num_workers`, set `pin_memory=True`, increase `prefetch_factor`
2. **Medium effort**: Pre-process expensive transforms offline, cache decoded images
3. **Significant effort**: Move data to SSD/NVMe, use memory-mapped files
4. **Architecture change**: Pre-shard data, use streaming datasets, move to WebDataset format

---

## 11. Collate Optimization

### Variable-Length Sequences

The default collate pads to max length in the batch — wasteful if lengths vary widely:

```python
# Naive: pad everything to max_len (512)
# Batch of sequences: [3, 7, 12, 490] tokens
# Padded: [512, 512, 512, 512] — 93% padding!
```

### Bucketed Batching

Group sequences by length to minimize padding:

```python
class BucketBatchSampler(torch.utils.data.Sampler):
    def __init__(self, lengths, batch_size, bucket_boundaries=None):
        self.lengths = lengths
        self.batch_size = batch_size

        if bucket_boundaries is None:
            bucket_boundaries = [32, 64, 128, 256, 512]

        # Assign each sample to a bucket
        buckets = {b: [] for b in bucket_boundaries}
        for idx, length in enumerate(lengths):
            for boundary in bucket_boundaries:
                if length <= boundary:
                    buckets[boundary].append(idx)
                    break

        # Create batches within each bucket
        self.batches = []
        for boundary, indices in buckets.items():
            random.shuffle(indices)
            for i in range(0, len(indices), batch_size):
                self.batches.append(indices[i:i + batch_size])
        random.shuffle(self.batches)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)
```

### Dynamic Batching by Token Count

Instead of a fixed number of sequences per batch, fix the total number of tokens:

```python
class TokenBatchSampler(torch.utils.data.Sampler):
    def __init__(self, lengths, max_tokens=4096):
        sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
        self.batches = []
        current_batch = []
        current_max_len = 0

        for idx in sorted_indices:
            new_max = max(current_max_len, lengths[idx])
            if new_max * (len(current_batch) + 1) > max_tokens and current_batch:
                self.batches.append(current_batch)
                current_batch = [idx]
                current_max_len = lengths[idx]
            else:
                current_batch.append(idx)
                current_max_len = new_max

        if current_batch:
            self.batches.append(current_batch)

    def __iter__(self):
        random.shuffle(self.batches)
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)
```

### Custom Collate for Packed Sequences

```python
def packed_collate(batch):
    sequences, labels = zip(*batch)
    lengths = [len(s) for s in sequences]
    padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    mask = torch.arange(padded.size(1)).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
    return padded, torch.stack(labels), mask
```

---

## 12. Data Pipeline Patterns

### Pattern 1: Offline Pre-processing

Process data once, save results, load during training:

```
Raw Data ──[preprocess.py]──▶ Processed Files ──[DataLoader]──▶ Training
  (images, text)                (tensors, .bin)
```

**When to use**: Expensive transforms (tokenization, image decoding, feature extraction) that produce the same result every time.

### Pattern 2: On-the-Fly Augmentation

Apply random transforms during loading:

```python
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

**When to use**: Stochastic augmentations that should differ each epoch.

### Pattern 3: Cached Transforms

Cache the expensive part, apply cheap augmentations on-the-fly:

```python
class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, cache_dir):
        self.base = base_dataset
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def __getitem__(self, idx):
        cache_path = self.cache_dir / f"{idx}.pt"
        if cache_path.exists():
            return torch.load(cache_path, weights_only=True)
        item = self.base[idx]
        torch.save(item, cache_path)
        return item

    def __len__(self):
        return len(self.base)
```

### Pattern 4: Multi-Stage Pipeline

```
┌────────┐    ┌────────┐    ┌──────────┐    ┌─────────┐
│  Load  │───▶│ Decode │───▶│ Augment  │───▶│ Collate │
│ (I/O)  │    │ (CPU)  │    │ (CPU/GPU)│    │ (CPU)   │
└────────┘    └────────┘    └──────────┘    └─────────┘
  Workers       Workers       Workers        Main proc
```

Each stage can be parallelized independently. Workers handle load + decode + augment; the main process collates and sends to GPU.

### Summary of Patterns

| Pattern | I/O cost | CPU cost | Flexibility | Memory |
|---------|----------|----------|-------------|--------|
| Offline pre-process | Low | Low | Low | High (disk) |
| On-the-fly | High | High | High | Low |
| Cached transforms | Low (after warmup) | Low | Medium | High (disk) |
| Multi-stage | Low | Distributed | High | Medium |

---

## 13. Upstream Updates (June 23-25, 2026)

Recent PyTorch commits relevant to data pipelines and general training infrastructure:

### CUDAGraph Multiple Pools in Single Graph ([#187929](https://github.com/pytorch/pytorch/pull/187929))

Support for using multiple memory pools within a single CUDA graph capture. Previously, all allocations within a graph had to come from a single pool. This enables more flexible memory management when composing graphs from subgraphs that use different pools.

### nonstrict_trace Added to torch.compiler ([#187737](https://github.com/pytorch/pytorch/pull/187737))

A new `nonstrict_trace` API in `torch.compiler` that provides non-strict tracing semantics. This relaxes some of the constraints of strict tracing, making it easier to trace models with data-dependent control flow while still producing exportable graphs.

### SymmMem Barrier for NCCL Backend ([#188051](https://github.com/pytorch/pytorch/pull/188051))

Symmetric memory barrier support for the NCCL communication backend. This provides fine-grained synchronization primitives for distributed training, reducing the overhead of full-barrier synchronization when only local ordering guarantees are needed.

### CUTLASS SiLU Epilogue Fusion ([#186197](https://github.com/pytorch/pytorch/pull/186197))

Fuses SiLU activation into CUTLASS GEMM epilogues, eliminating a separate kernel launch for the activation function. Particularly beneficial for LLM architectures that use SwiGLU (which contains SiLU) in their feed-forward layers.

### FlexGEMM Captured Tensor Epilogue Args ([#187254](https://github.com/pytorch/pytorch/pull/187254))

FlexGEMM now supports captured tensor arguments in epilogue computations. This allows more complex epilogue patterns (like residual additions or bias terms) to be fused directly into the GEMM kernel without additional kernel launches.

### Profiler NodeTimerObserver for Per-Node Timing ([#186802](https://github.com/pytorch/pytorch/pull/186802))

A new `NodeTimerObserver` for the PyTorch profiler that provides per-node timing information in the execution graph. This enables fine-grained performance analysis of individual operations, making it easier to identify bottlenecks in data processing and model execution.

### ROCm Origami Enabled ([#186644](https://github.com/pytorch/pytorch/pull/186644))

ROCm Origami optimization enabled for AMD GPUs. This provides optimized kernel implementations for common operations on ROCm, improving training throughput for AMD GPU users.

---

## Putting It All Together

A production-grade data pipeline combines several techniques:

```python
# 1. Pre-tokenize data offline → binary files
# 2. Memory-map the binary files
# 3. Use IterableDataset with worker splitting
# 4. pin_memory + non_blocking transfer
# 5. Profile and tune num_workers + prefetch_factor

loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=8,
    pin_memory=True,
    prefetch_factor=4,
    persistent_workers=True,
    worker_init_fn=seed_worker,
    generator=torch.Generator().manual_seed(42),
)
```

### Checklist

- [ ] Data loading time < compute time?
- [ ] GPU utilization > 90%?
- [ ] Memory usage stable over training?
- [ ] Workers properly seeded?
- [ ] Distributed: set_epoch() called each epoch?
- [ ] Variable-length data: using bucketed batching?
- [ ] Large data: using memory-mapped files or streaming?
- [ ] Expensive transforms: pre-processed offline?

---

### Further Resources

- [PyTorch DataLoader docs](https://pytorch.org/docs/stable/data.html) — official DataLoader reference
- [Module 06 — Data Loading](../06_data_loading/) — DataLoader fundamentals
- [Module 07 — Training Pipelines](../07_training/) — training loop patterns
- [Module 10 — Distributed Training](../10_distributed/) — multi-GPU training
- [Module 22 — LLM Recipes](../22_llm_recipes/) — LLM-specific training patterns

---

<div align="center">

[← Previous Module (torchao)](../31_torchao/) | [🏠 Home](../README.md) | [Next Module (Model Interpretability with Hooks) →](../33_interpretability/)

**Notebook**: [`32_data_pipelines.ipynb`](../notebooks/32_data_pipelines.ipynb)

</div>
