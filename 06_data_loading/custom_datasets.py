"""
Module 06: Custom Datasets
============================
Creating custom Dataset classes for various data types:
tabular data, text sequences, image-like data, and streaming data.

Run: python custom_datasets.py
"""

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
import os
import tempfile

print("=" * 70)
print("PART 1: Custom Dataset for Tabular Data")
print("=" * 70)


class TabularDataset(Dataset):
    """Dataset for tabular (CSV-like) data with feature engineering."""

    def __init__(self, features, labels, normalize=True):
        """
        Args:
            features: Tensor of shape (num_samples, num_features)
            labels: Tensor of shape (num_samples,)
            normalize: Whether to z-score normalize features
        """
        self.labels = labels
        if normalize:
            self.mean = features.mean(dim=0)
            self.std = features.std(dim=0)
            self.std[self.std == 0] = 1.0  # Avoid division by zero
            self.features = (features - self.mean) / self.std
        else:
            self.features = features
            self.mean = torch.zeros(features.shape[1])
            self.std = torch.ones(features.shape[1])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def get_normalization_params(self):
        """Return params for normalizing new data at inference."""
        return self.mean, self.std


# Usage
torch.manual_seed(42)
raw_features = torch.randn(1000, 15) * 10 + 5  # Non-zero mean, large std
raw_labels = torch.randint(0, 3, (1000,))

dataset = TabularDataset(raw_features, raw_labels, normalize=True)
print(f"\nTabularDataset:")
print(f"  Size: {len(dataset)}")
print(f"  Raw feature stats — mean: {raw_features.mean():.2f}, std: {raw_features.std():.2f}")
x, y = dataset[0]
print(f"  Normalized sample — mean: {dataset.features.mean():.4f}, "
      f"std: {dataset.features.std():.4f}")
print(f"  Sample shape: features={x.shape}, label={y.item()}")

print("\n" + "=" * 70)
print("PART 2: Custom Dataset for Text/Sequences")
print("=" * 70)


class TextClassificationDataset(Dataset):
    """Dataset that tokenizes text and returns padded sequences."""

    def __init__(self, texts, labels, vocab=None, max_length=50):
        """
        Args:
            texts: List of strings
            labels: List of int labels
            vocab: Dict mapping word -> index (built from data if None)
            max_length: Maximum sequence length (truncate/pad to this)
        """
        self.labels = labels
        self.max_length = max_length

        # Build vocabulary if not provided
        if vocab is None:
            self.vocab = {"<pad>": 0, "<unk>": 1}
            for text in texts:
                for word in text.lower().split():
                    if word not in self.vocab:
                        self.vocab[word] = len(self.vocab)
        else:
            self.vocab = vocab

        # Tokenize all texts
        self.encoded = []
        for text in texts:
            tokens = [self.vocab.get(w, self.vocab["<unk>"])
                      for w in text.lower().split()]
            self.encoded.append(tokens)

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        tokens = self.encoded[idx]
        label = self.labels[idx]

        # Truncate
        tokens = tokens[: self.max_length]
        # Record actual length before padding
        length = len(tokens)
        # Pad
        tokens = tokens + [0] * (self.max_length - length)

        return (
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )


# Example usage
texts = [
    "this movie was absolutely fantastic and wonderful",
    "terrible film waste of time do not watch",
    "pretty good acting but the plot was weak",
    "one of the best movies i have ever seen",
    "boring and predictable would not recommend",
    "excellent cinematography and a gripping story",
    "awful dialogue and terrible special effects",
    "a masterpiece of modern cinema truly brilliant",
]
labels = [1, 0, 1, 1, 0, 1, 0, 1]  # 1=positive, 0=negative

dataset = TextClassificationDataset(texts, labels, max_length=12)
print(f"\nTextClassificationDataset:")
print(f"  Size: {len(dataset)}")
print(f"  Vocab size: {len(dataset.vocab)}")
print(f"  Max length: {dataset.max_length}")

tokens, length, label = dataset[0]
print(f"\n  Sample 0: '{texts[0]}'")
print(f"  Tokens: {tokens.tolist()}")
print(f"  Length: {length.item()}")
print(f"  Label: {label.item()}")

tokens, length, label = dataset[1]
print(f"\n  Sample 1: '{texts[1]}'")
print(f"  Tokens: {tokens.tolist()}")
print(f"  Length: {length.item()}")

# DataLoader for text
loader = DataLoader(dataset, batch_size=4, shuffle=True)
batch_tokens, batch_lengths, batch_labels = next(iter(loader))
print(f"\n  Batch: tokens={batch_tokens.shape}, lengths={batch_lengths.tolist()}, "
      f"labels={batch_labels.tolist()}")

print("\n" + "=" * 70)
print("PART 3: Custom Dataset for Image-like Data")
print("=" * 70)


class SyntheticImageDataset(Dataset):
    """Dataset that generates synthetic image-like tensors."""

    def __init__(self, num_samples=100, image_size=(3, 32, 32), num_classes=10,
                 transform=None):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        self.transform = transform

        # Pre-generate all data (in real use, you'd load from disk)
        self.images = torch.randn(num_samples, *image_size)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# Simple transforms (without torchvision dependency)
class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        if torch.rand(1).item() < self.p:
            return image.flip(-1)  # Flip width dimension
        return image


class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, image):
        return (image - self.mean) / self.std


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image


# Create dataset with transforms
train_transform = Compose([
    RandomHorizontalFlip(p=0.5),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

train_dataset = SyntheticImageDataset(
    num_samples=200, image_size=(3, 32, 32), num_classes=10,
    transform=train_transform
)
val_dataset = SyntheticImageDataset(
    num_samples=50, image_size=(3, 32, 32), num_classes=10,
    transform=Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
)

print(f"\nSyntheticImageDataset:")
print(f"  Train: {len(train_dataset)} images")
print(f"  Val:   {len(val_dataset)} images")

image, label = train_dataset[0]
print(f"  Image shape: {image.shape}")
print(f"  Image range: [{image.min():.2f}, {image.max():.2f}]")
print(f"  Label: {label.item()}")

loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
batch_images, batch_labels = next(iter(loader))
print(f"  Batch: images={batch_images.shape}, labels={batch_labels.shape}")

print("\n" + "=" * 70)
print("PART 4: IterableDataset for Streaming Data")
print("=" * 70)


class LineByLineDataset(IterableDataset):
    """Reads data line-by-line from a file (streaming)."""

    def __init__(self, filepath, transform=None):
        self.filepath = filepath
        self.transform = transform

    def __iter__(self):
        with open(self.filepath, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 2:
                    continue
                features = torch.tensor([float(x) for x in parts[:-1]])
                label = torch.tensor(int(parts[-1]))

                if self.transform:
                    features = self.transform(features)

                yield features, label


# Create a temporary file with CSV-like data
tmp_dir = tempfile.mkdtemp()
data_file = os.path.join(tmp_dir, "data.csv")

torch.manual_seed(42)
with open(data_file, "w") as f:
    for i in range(100):
        features = torch.randn(5)
        label = torch.randint(0, 3, (1,)).item()
        line = ",".join([f"{x:.4f}" for x in features.tolist()] + [str(label)])
        f.write(line + "\n")

dataset = LineByLineDataset(data_file)
loader = DataLoader(dataset, batch_size=16)

print(f"\nIterableDataset (streaming from file):")
print(f"  File: {data_file}")
batch_count = 0
for features, labels in loader:
    if batch_count == 0:
        print(f"  First batch: features={features.shape}, labels={labels.shape}")
    batch_count += 1
print(f"  Total batches: {batch_count}")


# Multi-worker IterableDataset
class ShardedIterableDataset(IterableDataset):
    """IterableDataset that properly splits work among workers."""

    def __init__(self, data_shards):
        self.data_shards = data_shards  # List of data chunks

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single-process: use all shards
            shards = self.data_shards
        else:
            # Multi-process: split shards among workers
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            # Each worker gets a subset of shards
            shards = self.data_shards[worker_id::num_workers]

        for shard in shards:
            for item in shard:
                yield item


# Create sharded data
shards = []
for shard_id in range(8):
    shard_data = [(torch.randn(10), torch.tensor(shard_id % 3)) for _ in range(25)]
    shards.append(shard_data)

sharded_dataset = ShardedIterableDataset(shards)

# Test with 2 workers
loader = DataLoader(sharded_dataset, batch_size=16, num_workers=2)
total_samples = sum(batch[0].shape[0] for batch in loader)
print(f"\n  Sharded IterableDataset with 2 workers:")
print(f"  Total shards: 8, Total samples: {total_samples}")
print(f"  (Each worker processes 4 shards)")

print("\n" + "=" * 70)
print("PART 5: Dataset with On-Disk Storage")
print("=" * 70)


class OnDiskDataset(Dataset):
    """Dataset that loads individual samples from disk.
    
    Simulates loading images/files one at a time.
    """

    def __init__(self, root_dir, num_samples=50):
        self.root_dir = root_dir
        self.num_samples = num_samples
        # Create sample files
        os.makedirs(root_dir, exist_ok=True)
        for i in range(num_samples):
            data = torch.randn(3, 16, 16)  # Small "image"
            label = i % 5
            torch.save({"data": data, "label": label},
                       os.path.join(root_dir, f"sample_{i:04d}.pt"))
        self.file_list = sorted(
            [f for f in os.listdir(root_dir) if f.endswith(".pt")]
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = os.path.join(self.root_dir, self.file_list[idx])
        sample = torch.load(filepath, weights_only=True)
        return sample["data"], torch.tensor(sample["label"])


disk_dir = os.path.join(tmp_dir, "disk_dataset")
disk_dataset = OnDiskDataset(disk_dir, num_samples=50)

print(f"\nOnDiskDataset:")
print(f"  Directory: {disk_dir}")
print(f"  Samples: {len(disk_dataset)}")

image, label = disk_dataset[0]
print(f"  Sample 0: image={image.shape}, label={label.item()}")

loader = DataLoader(disk_dataset, batch_size=8, shuffle=True, num_workers=0)
batch_imgs, batch_labels = next(iter(loader))
print(f"  Batch: images={batch_imgs.shape}, labels={batch_labels.tolist()}")

print("\n" + "=" * 70)
print("PART 6: Dataset with Caching")
print("=" * 70)


class CachedDataset(Dataset):
    """Dataset that caches processed samples in memory."""

    def __init__(self, raw_data, expensive_transform=None):
        self.raw_data = raw_data
        self.transform = expensive_transform
        self.cache = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        item = self.raw_data[idx]
        if self.transform:
            item = self.transform(item)

        self.cache[idx] = item
        return item

    def clear_cache(self):
        self.cache.clear()

    @property
    def cache_hit_rate(self):
        return len(self.cache) / len(self.raw_data)


def expensive_processing(x):
    """Simulates expensive data processing."""
    return x ** 2 + torch.sin(x) * 3


raw_data = [torch.randn(20) for _ in range(100)]
cached_dataset = CachedDataset(raw_data, expensive_transform=expensive_processing)

print(f"\nCachedDataset:")
print(f"  Cache hit rate (before): {cached_dataset.cache_hit_rate:.1%}")

# First pass: populates cache
for i in range(len(cached_dataset)):
    _ = cached_dataset[i]
print(f"  Cache hit rate (after full pass): {cached_dataset.cache_hit_rate:.1%}")

# Timing comparison
import time
start = time.time()
for i in range(len(cached_dataset)):
    _ = cached_dataset[i]  # All cache hits
elapsed_cached = time.time() - start

cached_dataset.clear_cache()
start = time.time()
for i in range(len(cached_dataset)):
    _ = cached_dataset[i]  # All cache misses
elapsed_uncached = time.time() - start
print(f"  Cached access: {elapsed_cached*1000:.2f}ms")
print(f"  Uncached access: {elapsed_uncached*1000:.2f}ms")

print("\n" + "=" * 70)
print("PART 7: Dataset with Data Augmentation at __getitem__ Time")
print("=" * 70)


class AugmentedDataset(Dataset):
    """Dataset that applies different augmentations during train vs eval."""

    def __init__(self, data, labels, training=True):
        self.data = data
        self.labels = labels
        self.training = training

    def __len__(self):
        return len(self.data)

    def set_training(self, mode):
        self.training = mode

    def __getitem__(self, idx):
        x = self.data[idx].clone()
        y = self.labels[idx]

        if self.training:
            # Random noise augmentation
            x = x + torch.randn_like(x) * 0.1
            # Random scaling
            scale = 0.8 + torch.rand(1).item() * 0.4  # [0.8, 1.2]
            x = x * scale
            # Random dropout of features
            mask = torch.rand_like(x) > 0.1
            x = x * mask

        return x, y


data = torch.randn(200, 50)
labels = torch.randint(0, 5, (200,))

aug_dataset = AugmentedDataset(data, labels, training=True)

# Same sample looks different each time during training
print(f"\nAugmentedDataset — same sample accessed 3 times during training:")
for i in range(3):
    x, y = aug_dataset[0]
    print(f"  Access {i}: mean={x.mean():.4f}, std={x.std():.4f}")

aug_dataset.set_training(False)
print(f"\nSame sample during eval (no augmentation):")
for i in range(3):
    x, y = aug_dataset[0]
    print(f"  Access {i}: mean={x.mean():.4f}, std={x.std():.4f}")
print("  (Identical each time)")

print("\n" + "=" * 70)
print("PART 8: Dataset with Multiple Return Values")
print("=" * 70)


class RichDataset(Dataset):
    """Dataset that returns dictionaries instead of tuples."""

    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        self.features = torch.randn(num_samples, 10)
        self.labels = torch.randint(0, 5, (num_samples,))
        self.metadata = [f"sample_{i}" for i in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "label": self.labels[idx],
            "index": idx,
            "id": self.metadata[idx],
        }


rich_dataset = RichDataset(100)
print(f"\nRichDataset (returns dicts):")
sample = rich_dataset[0]
print(f"  Keys: {list(sample.keys())}")
print(f"  features shape: {sample['features'].shape}")
print(f"  label: {sample['label'].item()}")
print(f"  id: {sample['id']}")

# DataLoader handles dicts automatically!
loader = DataLoader(rich_dataset, batch_size=8, shuffle=True)
batch = next(iter(loader))
print(f"\n  Batched dict:")
print(f"    features: {batch['features'].shape}")
print(f"    labels: {batch['label'].shape}")
print(f"    indices: {batch['index'].tolist()}")
print(f"    ids: {batch['id']}")

# Cleanup
import shutil
shutil.rmtree(tmp_dir)

print("\n" + "=" * 70)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
print("=" * 70)
