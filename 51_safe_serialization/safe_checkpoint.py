"""Safe, explicit state-dict checkpointing patterns."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import torch
from torch import nn


class TinyClassifier(nn.Module):
    """A small model used to demonstrate portable state-dict loading."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def sha256(path: Path) -> str:
    """Return a content digest for a trusted checkpoint artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
) -> None:
    """Save a versioned, tensor-oriented training checkpoint."""
    checkpoint = {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "model_config": {"input_features": 4, "num_classes": 3},
    }
    torch.save(checkpoint, path)


def load_training_checkpoint(
    path: Path,
    expected_digest: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Verify and load a checkpoint with an explicit schema contract."""
    if sha256(path) != expected_digest:
        raise ValueError("Checkpoint digest does not match the trusted value")

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format version")

    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["step"])


def inspect_unsafe_globals(path: Path) -> list[str]:
    """Report static unsafe-global findings when the runtime supports it."""
    inspect = getattr(torch.serialization, "get_unsafe_globals_in_checkpoint", None)
    return [] if inspect is None else list(inspect(path))


def main() -> None:
    torch.manual_seed(0)
    model = TinyClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    inputs = torch.randn(8, 4)
    targets = torch.randint(0, 3, (8,))
    loss = nn.CrossEntropyLoss()(model(inputs), targets)
    loss.backward()
    optimizer.step()

    with tempfile.TemporaryDirectory() as temporary_directory:
        checkpoint_path = Path(temporary_directory) / "classifier.pt"
        save_training_checkpoint(checkpoint_path, model, optimizer, step=1)
        digest = sha256(checkpoint_path)

        restored_model = TinyClassifier()
        restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
        restored_step = load_training_checkpoint(
            checkpoint_path,
            digest,
            restored_model,
            restored_optimizer,
        )

        torch.testing.assert_close(model(inputs), restored_model(inputs))
        print(f"Loaded verified checkpoint at step {restored_step}")
        print(f"Unsafe globals: {inspect_unsafe_globals(checkpoint_path)}")


if __name__ == "__main__":
    main()
