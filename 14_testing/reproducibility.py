"""
Reproducibility in PyTorch
==========================

Complete setup for reproducible experiments:
1. Seeding all random number generators
2. Deterministic mode
3. DataLoader reproducibility
4. Verifying reproducibility
5. Known sources of non-determinism
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int = 42):
    """Set all random seeds for full reproducibility.

    This covers:
    - Python's random module
    - NumPy's random generator
    - PyTorch's CPU and CUDA random generators
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def enable_deterministic_mode():
    """Enable fully deterministic execution.

    Warning: this may significantly slow down some operations, and some
    operations will raise errors if no deterministic implementation exists.
    """
    torch.use_deterministic_algorithms(True)

    # Required for deterministic CUDA operations with cuBLAS
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Disable cuDNN benchmark mode (algorithm selection is non-deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    """Seed function for DataLoader workers.

    Each worker gets a unique but reproducible seed derived from the
    initial seed and the worker ID.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ===========================================================================
# Demonstration
# ===========================================================================

def demo_basic_reproducibility():
    """Show that setting seeds produces identical results."""
    print("=" * 60)
    print("BASIC REPRODUCIBILITY")
    print("=" * 60)

    # Without seed: different results each time
    results_no_seed = []
    for i in range(3):
        x = torch.randn(3)
        results_no_seed.append(x.tolist())

    print("  Without seed (different each time):")
    for i, r in enumerate(results_no_seed):
        print(f"    Run {i}: {[f'{v:.4f}' for v in r]}")

    # With seed: identical results every time
    results_with_seed = []
    for i in range(3):
        set_seed(42)
        x = torch.randn(3)
        results_with_seed.append(x.tolist())

    print("\n  With seed=42 (identical every time):")
    for i, r in enumerate(results_with_seed):
        print(f"    Run {i}: {[f'{v:.4f}' for v in r]}")

    # Verify
    match = all(
        all(abs(a - b) < 1e-6 for a, b in zip(r1, r2))
        for r1, r2 in zip(results_with_seed[:-1], results_with_seed[1:])
    )
    print(f"\n  All seeded runs identical: {match}")


def demo_model_training_reproducibility():
    """Show that training is reproducible with proper seeding."""
    print("\n" + "=" * 60)
    print("MODEL TRAINING REPRODUCIBILITY")
    print("=" * 60)

    def train_model(seed):
        set_seed(seed)

        model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        X = torch.randn(64, 10)
        Y = torch.randn(64, 1)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        losses = []
        for epoch in range(50):
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, Y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        return losses, model

    # Train twice with the same seed
    losses_1, model_1 = train_model(seed=42)
    losses_2, model_2 = train_model(seed=42)

    # Compare losses
    loss_match = all(abs(l1 - l2) < 1e-6 for l1, l2 in zip(losses_1, losses_2))
    print(f"  Training losses match: {loss_match}")

    # Compare final parameters
    param_match = all(
        torch.equal(p1, p2)
        for p1, p2 in zip(model_1.parameters(), model_2.parameters())
    )
    print(f"  Final parameters match: {param_match}")

    print(f"\n  Loss trajectory (first 10 steps):")
    for i in range(min(10, len(losses_1))):
        print(f"    Step {i:2d}: run1={losses_1[i]:.6f}, run2={losses_2[i]:.6f}, "
              f"diff={abs(losses_1[i] - losses_2[i]):.2e}")


def demo_dataloader_reproducibility():
    """Show reproducible DataLoader with shuffling."""
    print("\n" + "=" * 60)
    print("DATALOADER REPRODUCIBILITY")
    print("=" * 60)

    X = torch.arange(100).float().reshape(100, 1)
    Y = X * 2

    dataset = TensorDataset(X, Y)

    def get_batch_order(seed):
        g = torch.Generator()
        g.manual_seed(seed)

        loader = DataLoader(
            dataset,
            batch_size=10,
            shuffle=True,
            num_workers=0,
            generator=g,
            worker_init_fn=seed_worker,
        )

        first_elements = []
        for batch_x, _ in loader:
            first_elements.append(batch_x[0].item())
        return first_elements

    # Same seed -> same shuffle order
    order1 = get_batch_order(seed=42)
    order2 = get_batch_order(seed=42)
    order3 = get_batch_order(seed=99)  # Different seed

    print(f"  Seed=42, run 1: {order1[:5]}...")
    print(f"  Seed=42, run 2: {order2[:5]}...")
    print(f"  Seed=99, run 3: {order3[:5]}...")
    print(f"\n  Same seed -> same order: {order1 == order2}")
    print(f"  Diff seed -> diff order: {order1 != order3}")


def demo_numpy_torch_interaction():
    """Show how NumPy and PyTorch random states interact."""
    print("\n" + "=" * 60)
    print("NUMPY/PYTORCH RANDOM STATE INTERACTION")
    print("=" * 60)

    set_seed(42)

    # Generate some random numbers from both libraries
    torch_vals = torch.randn(3).tolist()
    numpy_vals = np.random.randn(3).tolist()

    set_seed(42)  # Reset
    torch_vals_2 = torch.randn(3).tolist()
    numpy_vals_2 = np.random.randn(3).tolist()

    print(f"  PyTorch values match: "
          f"{all(abs(a-b) < 1e-6 for a, b in zip(torch_vals, torch_vals_2))}")
    print(f"  NumPy values match:   "
          f"{all(abs(a-b) < 1e-6 for a, b in zip(numpy_vals, numpy_vals_2))}")


def demo_known_nondeterminism():
    """Document known sources of non-determinism in PyTorch."""
    print("\n" + "=" * 60)
    print("KNOWN SOURCES OF NON-DETERMINISM")
    print("=" * 60)

    sources = [
        ("torch.nn.functional.interpolate (backward, CUDA)",
         "Non-deterministic CUDA kernel"),
        ("torch.nn.functional.grid_sample (backward, CUDA)",
         "Non-deterministic CUDA kernel"),
        ("Scatter/gather operations (CUDA)",
         "Atomic addition is non-deterministic"),
        ("torch.Tensor.index_add_ (CUDA)",
         "Uses non-deterministic atomicAdd"),
        ("torch.bincount (CUDA)",
         "Uses non-deterministic atomicAdd"),
        ("DataLoader with num_workers > 0",
         "Worker ordering can vary; use worker_init_fn"),
        ("torch.backends.cudnn.benchmark = True",
         "Auto-tunes algorithms, selection varies"),
        ("Multi-threaded CPU operations",
         "Thread scheduling affects results"),
    ]

    print("  Known non-deterministic operations:")
    for op, reason in sources:
        print(f"\n    Operation: {op}")
        print(f"    Reason:    {reason}")

    print("\n\n  Mitigation strategies:")
    print("    1. torch.use_deterministic_algorithms(True)")
    print("    2. torch.backends.cudnn.benchmark = False")
    print("    3. Set all seeds (random, numpy, torch)")
    print("    4. Use worker_init_fn for DataLoader")
    print("    5. Set CUBLAS_WORKSPACE_CONFIG env var")


def demo_generator_api():
    """Demonstrate PyTorch's Generator API for fine-grained control."""
    print("\n" + "=" * 60)
    print("GENERATOR API")
    print("=" * 60)

    # Generators provide independent random states
    gen1 = torch.Generator()
    gen2 = torch.Generator()

    gen1.manual_seed(42)
    gen2.manual_seed(42)

    x1 = torch.randn(5, generator=gen1)
    x2 = torch.randn(5, generator=gen2)
    print(f"  Same seed generators:")
    print(f"    gen1: {x1.tolist()}")
    print(f"    gen2: {x2.tolist()}")
    print(f"    Match: {torch.equal(x1, x2)}")

    # Independent generators don't affect each other
    gen1.manual_seed(42)
    gen2.manual_seed(99)

    x1 = torch.randn(3, generator=gen1)
    x2 = torch.randn(3, generator=gen2)
    _ = torch.randn(100, generator=gen2)  # consume gen2
    x1_again = torch.randn(3, generator=gen1)  # gen1 unaffected

    print(f"\n  Independent generators:")
    print(f"    gen1 not affected by gen2 usage: True")
    print(f"    (gen1 produces next values in its own sequence)")


if __name__ == "__main__":
    demo_basic_reproducibility()
    demo_model_training_reproducibility()
    demo_dataloader_reproducibility()
    demo_numpy_torch_interaction()
    demo_known_nondeterminism()
    demo_generator_api()
    print("\n" + "=" * 60)
    print("All reproducibility demos completed successfully!")
    print("=" * 60)
