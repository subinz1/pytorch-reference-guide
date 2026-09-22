"""Test small module invariants with deterministic inputs.

Run: python test_module_invariants.py
"""

from __future__ import annotations

import unittest

import torch


class CenteredScale(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - x.mean(dim=-1, keepdim=True)) * self.scale


class TestCenteredScale(unittest.TestCase):
    def test_output_is_centered_before_scaling(self) -> None:
        module = CenteredScale(3)
        output = module(torch.tensor([[1.0, 3.0, 5.0]]))
        torch.testing.assert_close(output.mean(dim=-1), torch.zeros(1))

    def test_preserves_input_shape(self) -> None:
        module = CenteredScale(3)
        self.assertEqual(module(torch.randn(2, 4, 3)).shape, torch.Size((2, 4, 3)))


if __name__ == "__main__":
    unittest.main()
