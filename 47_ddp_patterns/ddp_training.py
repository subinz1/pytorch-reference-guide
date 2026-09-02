"""
DDP Training Patterns
=====================

Complete DistributedDataParallel training loop demonstrating:
- Process group initialization
- DDP model wrapping
- Gradient synchronization and accumulation
- Mixed precision training with DDP
- Checkpointing across ranks

Launch: torchrun --nproc_per_node=NUM_GPUS ddp_training.py
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


def setup(backend="nccl"):
    """Initialize the distributed process group."""
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup():
    dist.destroy_process_group()


def create_model():
    """ResNet-like block for training demonstrations."""
    return nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )


def create_dataloader(rank, world_size, num_samples=2048, batch_size=64):
    """Build a DataLoader with DistributedSampler for even data splits."""
    x = torch.randn(num_samples, 784)
    y = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(x, y)

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler), sampler


def basic_ddp_training(local_rank, num_epochs=3):
    """Standard DDP training loop with per-epoch sampler shuffle."""
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    model = create_model().cuda(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    loader, sampler = create_dataloader(rank, world_size)

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        ddp_model.train()
        epoch_loss = 0.0

        for x, y in loader:
            x, y = x.cuda(local_rank), y.cuda(local_rank)

            optimizer.zero_grad()
            output = ddp_model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if rank == 0:
            avg_loss = epoch_loss / len(loader)
            print(f"Epoch {epoch + 1}/{num_epochs}  loss={avg_loss:.4f}")

    return ddp_model


def gradient_accumulation_training(local_rank, accumulation_steps=4, num_epochs=2):
    """
    Accumulate gradients over micro-batches to simulate larger batch sizes.
    Use model.no_sync() to skip all-reduce on intermediate steps.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    model = create_model().cuda(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    loader, sampler = create_dataloader(rank, world_size, batch_size=16)

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        optimizer.zero_grad()

        for step, (x, y) in enumerate(loader):
            x, y = x.cuda(local_rank), y.cuda(local_rank)

            is_accumulating = (step + 1) % accumulation_steps != 0
            context = ddp_model.no_sync if is_accumulating else lambda: torch.enable_grad()

            with context():
                output = ddp_model(x)
                loss = loss_fn(output, y) / accumulation_steps
                loss.backward()

            if not is_accumulating:
                optimizer.step()
                optimizer.zero_grad()

        if rank == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} (grad accum) done")


def mixed_precision_ddp(local_rank, num_epochs=2):
    """DDP with automatic mixed precision (AMP) for FP16 training."""
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    model = create_model().cuda(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")
    loader, sampler = create_dataloader(rank, world_size)

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)

        for x, y in loader:
            x, y = x.cuda(local_rank), y.cuda(local_rank)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                output = ddp_model(x)
                loss = loss_fn(output, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if rank == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} (AMP) done")


def save_checkpoint(ddp_model, optimizer, epoch, path="checkpoint.pt"):
    """Save checkpoint from rank 0 only to avoid file corruption."""
    if dist.get_rank() == 0:
        torch.save({
            "epoch": epoch,
            "model_state_dict": ddp_model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, path)
        print(f"Checkpoint saved: {path}")
    dist.barrier()


def load_checkpoint(model, optimizer, path="checkpoint.pt"):
    """Load checkpoint on all ranks (map to correct device)."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    map_location = f"cuda:{local_rank}"

    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"]
    return 0


def main():
    local_rank = setup()

    if dist.get_rank() == 0:
        print("=== Basic DDP Training ===")
    basic_ddp_training(local_rank)

    if dist.get_rank() == 0:
        print("\n=== Gradient Accumulation ===")
    gradient_accumulation_training(local_rank)

    if dist.get_rank() == 0:
        print("\n=== Mixed Precision DDP ===")
    mixed_precision_ddp(local_rank)

    cleanup()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("DDP requires CUDA. Run with: torchrun --nproc_per_node=N ddp_training.py")
        print("Showing code structure only (no execution without GPUs).")
    elif "RANK" not in os.environ:
        print("Launch with: torchrun --nproc_per_node=NUM_GPUS ddp_training.py")
    else:
        main()
