"""Exercise a model state-dictionary save/load contract.

Run: python test_serialization_contract.py
"""

from __future__ import annotations

import io
import unittest

import torch


class TinyClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class TestSerializationContract(unittest.TestCase):
    def test_state_dict_round_trip_preserves_output(self) -> None:
        torch.manual_seed(0)
        original = TinyClassifier().eval()
        sample = torch.randn(3, 4)
        expected = original(sample)

        buffer = io.BytesIO()
        torch.save(original.state_dict(), buffer)
        buffer.seek(0)

        restored = TinyClassifier().eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        torch.testing.assert_close(restored(sample), expected)


if __name__ == "__main__":
    unittest.main()
