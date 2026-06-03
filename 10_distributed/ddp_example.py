"""
DistributedDataParallel (DDP) Complete Training Example
=======================================================

A full DDP training script with synthetic data. Demonstrates:
- Process group initialization
- Model creation and DDP wrapping
- DistributedSampler usage
- Training loop with proper synchronization
- Checkpoint saving (rank 0 only)
- Metric aggregation across ranks

Run with:
    torchrun --nproc_per_node=2 ddp_example.py

For CPU-only (Gloo backend), the script auto-detects no CUDA.
"""

import os
import tempfile

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class SyntheticClassificationDataset(Dataset):
    """A simple synthetic dataset: random features with deterministic labels."""

    def __init__(self, num_samples: int = 1000, input_dim: int = 64, num_classes: int = 10):
        self.num_samples = num_samples
        generator = torch.Generator().manual_seed(42)
        self.features = torch.randn(num_samples, input_dim, generator=generator)
        self.labels = torch.randint(0, num_classes, (num_samples,), generator=generator)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def setup():
    """Initialize the distributed process group."""
    if torch.cuda.is_available():
        backend = "nccl"
    else:
        backend = "gloo"
    dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return rank, local_rank, world_size, device


def cleanup():
    dist.destroy_process_group()


def reduce_metric(value: float, device: torch.device) -> float:
    """Average a scalar metric across all ranks."""
    tensor = torch.tensor([value], device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item() / dist.get_world_size()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    rank: int,
) -> tuple[float, float]:
    """Train for one epoch, return (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def save_checkpoint(
    model: DDP,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_dir: str,
):
    """Save checkpoint from rank 0 only."""
    path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )
    print(f"  [Rank 0] Checkpoint saved: {path}")


def main():
    rank, local_rank, world_size, device = setup()

    if rank == 0:
        print(f"DDP Training: {world_size} processes, device={device}")
        print(f"Backend: {dist.get_backend()}")
        print()

    # Create model
    model = SimpleClassifier(input_dim=64, hidden_dim=128, num_classes=10).to(device)

    if torch.cuda.is_available():
        model = DDP(model, device_ids=[local_rank])
    else:
        model = DDP(model)

    # Create dataset and distributed sampler
    dataset = SyntheticClassificationDataset(num_samples=1000, input_dim=64, num_classes=10)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    checkpoint_dir = tempfile.mkdtemp() if rank == 0 else ""

    num_epochs = 5
    for epoch in range(num_epochs):
        # IMPORTANT: set_epoch ensures different shuffling each epoch
        sampler.set_epoch(epoch)

        epoch_loss, epoch_acc = train_one_epoch(
            model, dataloader, optimizer, loss_fn, device, epoch, rank
        )

        # Average metrics across all ranks
        avg_loss = reduce_metric(epoch_loss, device)
        avg_acc = reduce_metric(epoch_acc, device)

        if rank == 0:
            print(
                f"  Epoch {epoch + 1}/{num_epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Accuracy: {avg_acc:.2%}"
            )

        # Save checkpoint from rank 0
        if rank == 0 and (epoch + 1) % 2 == 0:
            save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_dir)

    # Final summary
    if rank == 0:
        print(f"\n  Training complete! Final loss: {avg_loss:.4f}, accuracy: {avg_acc:.2%}")
        print(f"  Checkpoints saved to: {checkpoint_dir}")
        print()

    cleanup()


if __name__ == "__main__":
    main()
