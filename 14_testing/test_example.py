"""
Example Test File Using PyTorch's TestCase
==========================================

Demonstrates how to write tests using PyTorch's testing framework:
- TestCase with tensor-aware assertEqual
- Parametrized tests
- Testing models, operations, gradients, and dtypes
- setUp/tearDown for test fixtures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_utils import run_tests, TestCase, parametrize


# ---------------------------------------------------------------------------
# A simple model to test
# ---------------------------------------------------------------------------

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestTensorOperations(TestCase):
    """Test basic tensor operations."""

    def test_addition(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        result = a + b
        expected = torch.tensor([5.0, 7.0, 9.0])
        self.assertEqual(result, expected)

    def test_matmul_shapes(self):
        A = torch.randn(3, 4)
        B = torch.randn(4, 5)
        C = A @ B
        self.assertEqual(C.shape, torch.Size([3, 5]))

    def test_broadcasting(self):
        x = torch.randn(3, 4)
        bias = torch.randn(4)
        result = x + bias
        self.assertEqual(result.shape, x.shape)

    def test_reduction_dims(self):
        x = torch.randn(2, 3, 4)
        self.assertEqual(x.sum(dim=0).shape, torch.Size([3, 4]))
        self.assertEqual(x.sum(dim=1).shape, torch.Size([2, 4]))
        self.assertEqual(x.sum(dim=2).shape, torch.Size([2, 3]))
        self.assertEqual(x.sum().shape, torch.Size([]))

    @parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preservation(self, dtype):
        x = torch.randn(5, dtype=dtype)
        y = x * 2
        self.assertEqual(y.dtype, dtype)

    @parametrize("size", [(2, 3), (4, 5), (1, 10)])
    def test_zeros_sum(self, size):
        x = torch.zeros(size)
        self.assertEqual(x.sum().item(), 0.0)

    def test_error_on_shape_mismatch(self):
        a = torch.randn(3)
        b = torch.randn(4)
        with self.assertRaises(RuntimeError):
            _ = a + b


class TestSoftmax(TestCase):
    """Test softmax properties."""

    def test_sums_to_one(self):
        x = torch.randn(5, 10)
        probs = F.softmax(x, dim=-1)
        ones = torch.ones(5)
        self.assertEqual(probs.sum(dim=-1), ones, atol=1e-6, rtol=0)

    def test_non_negative(self):
        x = torch.randn(5, 10)
        probs = F.softmax(x, dim=-1)
        self.assertTrue((probs >= 0).all())

    def test_argmax_preserved(self):
        """softmax shouldn't change which element is largest."""
        x = torch.randn(100)
        probs = F.softmax(x, dim=0)
        self.assertEqual(x.argmax(), probs.argmax())

    @parametrize("dim", [0, 1, -1])
    def test_softmax_dim(self, dim):
        x = torch.randn(4, 5)
        probs = F.softmax(x, dim=dim)
        sums = probs.sum(dim=dim)
        expected_shape = [4, 5]
        expected_shape.pop(dim if dim >= 0 else len(expected_shape) + dim)
        self.assertEqual(sums.shape, torch.Size(expected_shape))
        self.assertEqual(sums, torch.ones_like(sums), atol=1e-6, rtol=0)


class TestSimpleClassifier(TestCase):
    """Test our SimpleClassifier model."""

    def setUp(self):
        """Create model and test data before each test."""
        torch.manual_seed(42)
        self.model = SimpleClassifier(input_dim=10, hidden_dim=32, num_classes=5)
        self.model.eval()
        self.x = torch.randn(4, 10)

    def test_output_shape(self):
        with torch.no_grad():
            out = self.model(self.x)
        self.assertEqual(out.shape, torch.Size([4, 5]))

    @parametrize("batch_size", [1, 4, 16, 64])
    def test_variable_batch_size(self, batch_size):
        x = torch.randn(batch_size, 10)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(out.shape, torch.Size([batch_size, 5]))

    def test_eval_deterministic(self):
        """Same input should produce same output in eval mode."""
        with torch.no_grad():
            out1 = self.model(self.x)
            out2 = self.model(self.x)
        self.assertEqual(out1, out2)

    def test_gradients_flow(self):
        """All parameters should receive gradients."""
        self.model.train()
        out = self.model(self.x)
        loss = out.sum()
        loss.backward()
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, f"No gradient for {name}")
            self.assertFalse(
                torch.all(param.grad == 0),
                f"Zero gradient for {name}",
            )

    def test_parameter_count(self):
        total = sum(p.numel() for p in self.model.parameters())
        # fc1: 10*32 + 32 = 352, fc2: 32*5 + 5 = 165, total = 517
        self.assertEqual(total, 517)

    def test_no_nan_output(self):
        with torch.no_grad():
            out = self.model(self.x)
        self.assertFalse(torch.any(torch.isnan(out)))
        self.assertFalse(torch.any(torch.isinf(out)))

    def test_can_overfit_single_batch(self):
        """A basic sanity check: model should be able to memorize a small batch."""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        x = torch.randn(4, 10)
        y = torch.tensor([0, 1, 2, 3])

        for _ in range(300):
            optimizer.zero_grad()
            pred = self.model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            predictions = self.model(x).argmax(dim=1)
        self.assertEqual(predictions, y)


class TestGradientChecking(TestCase):
    """Test gradient correctness for custom operations."""

    def test_linear_grad(self):
        """Verify gradients of a linear operation using finite differences."""
        W = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        x = torch.randn(4, dtype=torch.float64, requires_grad=True)
        b = torch.randn(3, dtype=torch.float64, requires_grad=True)

        def func(W, x, b):
            return (W @ x + b).sum()

        self.assertTrue(
            torch.autograd.gradcheck(func, (W, x, b), eps=1e-6, atol=1e-4)
        )

    def test_relu_grad(self):
        """ReLU gradient: 1 where x > 0, 0 where x < 0."""
        x = torch.tensor([2.0, -1.0, 0.5, -3.0], requires_grad=True)
        y = F.relu(x)
        y.sum().backward()

        expected_grad = torch.tensor([1.0, 0.0, 1.0, 0.0])
        self.assertEqual(x.grad, expected_grad)


class TestAssertClose(TestCase):
    """Demonstrate torch.testing.assert_close usage."""

    def test_approximate_equality(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = a + 1e-7
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)

    def test_relative_tolerance(self):
        """For large values, relative tolerance matters more."""
        a = torch.tensor([1000.0, 2000.0])
        b = torch.tensor([1000.001, 2000.002])
        torch.testing.assert_close(a, b, atol=0.01, rtol=1e-5)


if __name__ == "__main__":
    run_tests()
