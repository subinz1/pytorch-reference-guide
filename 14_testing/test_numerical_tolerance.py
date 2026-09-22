"""Choose explicit numerical assertions for PyTorch tests.

Run: python test_numerical_tolerance.py
"""

from __future__ import annotations

import unittest

import torch


class TestNumericalTolerance(unittest.TestCase):
    def test_reduction_matches_higher_precision_reference(self) -> None:
        values = torch.linspace(-1.0, 1.0, 101, dtype=torch.float32)
        result = values.square().sum()
        reference = values.to(torch.float64).square().sum().to(torch.float32)
        torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-6)

    def test_non_finite_values_are_checked_deliberately(self) -> None:
        values = torch.tensor([1.0, float("nan"), float("inf")])
        self.assertTrue(torch.isnan(values[1]))
        self.assertTrue(torch.isinf(values[2]))


if __name__ == "__main__":
    unittest.main()
