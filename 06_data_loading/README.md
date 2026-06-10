<div align="center">

[← Previous Module](../05_optimizers/) | [🏠 Home](../README.md) | [Next Module →](../07_training/)

</div>

---

> **Module 06** of the PyTorch Complete Learning Guide
> **Prerequisites:** Module 02
> **Time to complete:** ~2 hours

### 📁 Files in This Module

| File | Description |
|------|-------------|
| `README.md` | This guide — theory, explanations, and inline examples |
| `dataset_basics.py` | Dataset basics |
| `custom_datasets.py` | Custom datasets |
| `dataloader_advanced.py` | Advanced DataLoader features |
| `augmentation.py` | Data augmentation patterns |

---

# Module 06: Data Loading in PyTorch

## Table of Contents
1. [Dataset Class](#dataset-class)
2. [IterableDataset](#iterabledataset)
3. [Built-in Datasets](#built-in-datasets)
4. [DataLoader](#dataloader)
5. [Worker Processes](#worker-processes)
6. [Custom Collate Functions](#custom-collate-functions)
7. [Samplers](#samplers)
8. [Data Augmentation](#data-augmentation)
9. [Memory-Mapped Datasets](#memory-mapped-datasets)
10. [Best Practices](#best-practices)

---

## Dataset Class

The `torch.utils.data.Dataset` is the base class for all map-style datasets. You implement
two methods:

- `__len__()`: Returns the total number of samples
- `__getitem__(index)`: Returns one sample given an index

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
```

Key properties:
- Supports random access by index
- Length is known ahead of time
- DataLoader can shuffle efficiently
- Can be split into train/val subsets

### When data doesn't fit in memory:
```python
class LargeFileDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load only one file at a time
        data = load_file(self.file_paths[idx])
        return process(data)
```

---

## IterableDataset

For streaming data or when random access is not possible/efficient:

```python
from torch.utils.data import IterableDataset

class StreamDataset(IterableDataset):
    def __init__(self, url):
        self.url = url
    
    def __iter__(self):
        # Yield samples one at a time
        for line in open_stream(self.url):
            yield process(line)
```

Use cases:
- Reading from network streams
- Very large files that can only be read sequentially
- Data generated on-the-fly
- Databases with cursor-based access

Key differences from map-style Dataset:
- No `__len__()` — size may be unknown
- No `__getitem__()` — no random access
- DataLoader cannot shuffle (must shuffle upstream)
- Multi-worker requires careful work splitting

### Multi-worker IterableDataset:
```python
class ShardedDataset(IterableDataset):
    def __init__(self, file_list):
        self.file_list = file_list
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process loading
            files = self.file_list
        else:
            # Split files among workers
            per_worker = len(self.file_list) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker
            files = self.file_list[start:end]
        
        for f in files:
            for sample in read_file(f):
                yield sample
```

---

## Built-in Datasets

### TensorDataset
Wraps tensors — each sample is a tuple indexed along the first dimension:

```python
from torch.utils.data import TensorDataset

X = torch.randn(1000, 20)
y = torch.randint(0, 5, (1000,))
dataset = TensorDataset(X, y)
# dataset[0] returns (X[0], y[0])
```

### ConcatDataset
Concatenates multiple datasets end-to-end:

```python
from torch.utils.data import ConcatDataset

combined = ConcatDataset([dataset_train, dataset_extra])
# len(combined) == len(dataset_train) + len(dataset_extra)
```

### Subset
Selects a subset of a dataset by indices:

```python
from torch.utils.data import Subset

# Manual train/val split
indices = torch.randperm(len(dataset))
train_set = Subset(dataset, indices[:800])
val_set = Subset(dataset, indices[800:])
```

### random_split
Convenience function for splitting:

```python
from torch.utils.data import random_split

train_set, val_set, test_set = random_split(
    dataset, [0.7, 0.15, 0.15],  # Fractions
    generator=torch.Generator().manual_seed(42)
)
```

---

## DataLoader

The DataLoader is the workhorse that turns a Dataset into an iterable of batches:

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,          # Samples per batch
    shuffle=True,           # Randomize order each epoch
    num_workers=4,          # Parallel data loading processes
    pin_memory=True,        # Speed up CPU->GPU transfer
    drop_last=False,        # Drop incomplete final batch?
    persistent_workers=True,# Keep workers alive between epochs
    prefetch_factor=2,      # Batches to prefetch per worker
)

for batch_X, batch_y in loader:
    # batch_X: (32, ...), batch_y: (32, ...)
    output = model(batch_X)
    loss = criterion(output, batch_y)
    ...
```

### Key Parameters:

**batch_size**: Number of samples per batch. Larger = faster training but more memory.
Typical values: 16, 32, 64, 128, 256.

**shuffle**: Randomize sample order each epoch. Always True for training,
False for validation/test (for reproducible evaluation).

**num_workers**: Number of subprocess workers for data loading. Set to 0 for
debugging (main process only). Typical: 2-8 depending on CPU cores and data complexity.

**pin_memory**: Pre-allocates batch tensors in pinned (page-locked) memory.
Makes CPU->GPU transfers faster. Always True when training on GPU.

**drop_last**: If True, drops the last batch if it's smaller than batch_size.
Important for BatchNorm (needs consistent batch size) and distributed training.

**persistent_workers**: Keeps worker processes alive across epochs instead of
respawning. Saves the cost of worker initialization. Use with num_workers > 0.

**prefetch_factor**: Number of batches each worker pre-loads. Higher values use more
memory but can hide I/O latency better. Default is 2.

---

## Worker Processes

### How Workers Work

When `num_workers > 0`, DataLoader spawns separate processes (not threads!) that:
1. Receive batch indices from the main process
2. Call `dataset.__getitem__()` for each index
3. Apply the collate function to form a batch
4. Send the batch back to the main process via shared memory

### Why num_workers > 0 is Faster

- Data loading (disk I/O, decompression, augmentation) happens in parallel
- While GPU trains on batch N, workers prepare batch N+1, N+2, ...
- Python's GIL doesn't affect separate processes

### Common Pitfalls

1. **Too many workers**: More workers != always faster. Each worker duplicates the
   dataset object in memory. Start with num_workers=4 and tune.

2. **Fork vs Spawn**: On Linux, workers are forked (fast, shares memory). On macOS/Windows,
   workers are spawned (slow startup, separate memory). Set:
   ```python
   torch.multiprocessing.set_start_method('spawn')  # If needed
   ```

3. **Random state in workers**: Each worker gets the same random seed by default.
   Use `worker_init_fn` to set different seeds:
   ```python
   def worker_init_fn(worker_id):
       seed = torch.initial_seed() % 2**32
       numpy.random.seed(seed + worker_id)
   
   loader = DataLoader(..., worker_init_fn=worker_init_fn)
   ```

4. **File handles**: If your dataset opens files, each worker opens its own copy.
   With many workers, you can run out of file descriptors.

5. **Shared memory limits**: Workers send data via shared memory. Very large batches
   can exhaust `/dev/shm`. Increase shared memory or reduce batch size.

---

## Custom Collate Functions

The collate function converts a list of samples into a batch. The default collator
stacks tensors, but custom collation handles variable-length sequences:

```python
def custom_collate(batch):
    """Pad variable-length sequences to same length."""
    sequences, labels = zip(*batch)
    # Pad sequences to max length in this batch
    lengths = [len(s) for s in sequences]
    max_len = max(lengths)
    padded = torch.zeros(len(sequences), max_len)
    for i, (seq, length) in enumerate(zip(sequences, lengths)):
        padded[i, :length] = seq
    labels = torch.tensor(labels)
    lengths = torch.tensor(lengths)
    return padded, labels, lengths

loader = DataLoader(dataset, batch_size=32, collate_fn=custom_collate)
```

Common custom collate patterns:
- Padding variable-length sequences
- Creating attention masks
- Handling nested data structures (dicts, lists)
- Filtering out None values (corrupt samples)

---

## Samplers

Samplers control the order in which indices are provided to the Dataset.

### SequentialSampler
Iterates indices 0, 1, 2, ..., N-1. Used when shuffle=False.

### RandomSampler
Random permutation of indices. Used when shuffle=True.

```python
from torch.utils.data import RandomSampler

# With replacement (for bootstrap):
sampler = RandomSampler(dataset, replacement=True, num_samples=10000)
```

### WeightedRandomSampler
For handling class imbalance — oversamples rare classes:

```python
from torch.utils.data import WeightedRandomSampler

# Assign weight to each sample (higher weight = sampled more often)
class_counts = [1000, 100, 50]  # Imbalanced classes
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
sample_weights = class_weights[labels]  # Weight per sample

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(dataset),
    replacement=True
)
loader = DataLoader(dataset, batch_size=32, sampler=sampler)
# Note: cannot use shuffle=True with a sampler
```

### SubsetRandomSampler
Random sampling from a fixed set of indices (for train/val splits):

```python
from torch.utils.data import SubsetRandomSampler

indices = list(range(len(dataset)))
train_indices = indices[:800]
val_indices = indices[800:]

train_loader = DataLoader(dataset, batch_size=32,
                          sampler=SubsetRandomSampler(train_indices))
val_loader = DataLoader(dataset, batch_size=32,
                        sampler=SubsetRandomSampler(val_indices))
```

### BatchSampler
Groups indices into batches (useful for custom batching logic):

```python
from torch.utils.data import BatchSampler, SequentialSampler

# Create batches of similar-length sequences (for efficient padding)
sampler = BatchSampler(
    SequentialSampler(dataset),
    batch_size=32,
    drop_last=False
)
```

### DistributedSampler
For multi-GPU training — ensures each GPU sees different data:

```python
from torch.utils.data import DistributedSampler

sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,  # Number of GPUs
    rank=rank,                # This GPU's index
    shuffle=True
)
loader = DataLoader(dataset, batch_size=32, sampler=sampler)

# IMPORTANT: Must call set_epoch each epoch for proper shuffling
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)
    for batch in loader:
        ...
```

---

## Data Augmentation

Data augmentation applies random transformations during training to increase
data diversity and reduce overfitting.

### torchvision.transforms.v2

The modern transforms API for images:

```python
from torchvision.transforms import v2

train_transform = v2.Compose([
    v2.RandomResizedCrop(224),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    v2.RandomRotation(15),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = v2.Compose([
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### MixUp

Linearly interpolates between pairs of samples:
```
mixed_input = lambda * input_i + (1-lambda) * input_j
mixed_target = lambda * target_i + (1-lambda) * target_j
```

### CutMix

Cuts a patch from one image and pastes it onto another:
```
mixed_input = mask * input_i + (1-mask) * input_j
mixed_target = lambda * target_i + (1-lambda) * target_j
```

Where lambda is the area ratio.

### Custom Augmentations

```python
class CustomAugmentation:
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image):
        if torch.rand(1) < self.p:
            return apply_augmentation(image)
        return image
```

---

## Memory-Mapped Datasets

For datasets too large to fit in RAM, memory-mapping lets the OS page data in/out:

```python
import numpy as np

class MemmapDataset(Dataset):
    def __init__(self, data_path, shape, dtype='float32'):
        # Memory-map the file (doesn't load into RAM)
        self.data = np.memmap(data_path, dtype=dtype, mode='r', shape=shape)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        # Only this sample is loaded into RAM
        return torch.from_numpy(self.data[idx].copy())
```

Benefits:
- Works with datasets much larger than RAM
- OS handles caching transparently
- Sequential access patterns are very efficient
- Multiple workers can share the same memory map

---

## Best Practices

### Performance Checklist

1. **pin_memory=True** when using GPU training
2. **persistent_workers=True** with num_workers > 0 to avoid respawn cost
3. **prefetch_factor=2** (default) is usually good; increase if I/O-bound
4. **num_workers**: Start with 4, increase until CPU is saturated or you run out of RAM
5. **drop_last=True** for training with BatchNorm

### Data Loading Bottleneck Detection

If your GPU utilization is low, data loading might be the bottleneck:
```python
import time

for batch in loader:
    start = time.time()
    output = model(batch)
    loss.backward()
    optimizer.step()
    gpu_time = time.time() - start
    # If data loading time >> gpu_time, you're data-bound
```

### Memory Efficiency

- Use `uint8` for images until the augmentation step
- Convert to float32 only in the transform/collate
- Use memory-mapped files for very large datasets
- Consider chunked reading for CSV/parquet files

### Reproducibility

```python
# Seed everything for reproducible data loading
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    import numpy as np
    np.random.seed(worker_seed)
    import random
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    worker_init_fn=seed_worker,
    generator=g,
)
```

### Common Patterns

**Training with validation:**
```python
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, 
                          num_workers=4, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=128, shuffle=False,
                        num_workers=4, pin_memory=True)
```

**Infinite data loader (for step-based training):**
```python
def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch

for step, batch in enumerate(infinite_loader(train_loader)):
    if step >= max_steps:
        break
    train_step(batch)
```

**Progress bar with tqdm:**
```python
from tqdm import tqdm

for batch in tqdm(loader, desc="Training"):
    ...
```

---

<div align="center">

[← Previous Module](../05_optimizers/) | [🏠 Home](../README.md) | [Next Module →](../07_training/)

**[📓 Open Notebook](../notebooks/06_data_loading_pipeline.ipynb)** — Interactive version of this module

</div>
