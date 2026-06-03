"""
Custom Operators with torch.library
====================================

When PyTorch's built-in operators aren't sufficient, you can define custom ops
that integrate with autograd, torch.compile, and torch.export.

This file demonstrates:
1. Defining a custom op with torch.library
2. Registering CPU (and Meta) implementations
3. Adding autograd support via setup_context + backward
4. Using custom_op decorator (simpler API)
5. Using the custom op in a model
"""

import torch
import torch.nn as nn
from torch.library import Library, impl, custom_op


# ===========================================================================
# Method 1: Using Library and impl (traditional approach)
# ===========================================================================

# Create a library namespace for our custom ops
mylib = Library("myops", "DEF")

# Define the op signature (name and types)
mylib.define("softplus(Tensor x, float beta=1.0) -> Tensor")

# Register CPU implementation
@impl(mylib, "softplus", "CPU")
def softplus_cpu(x, beta=1.0):
    """Numerically stable softplus: log(1 + exp(beta * x)) / beta."""
    return torch.where(
        x * beta > 20,
        x,  # for large values, softplus(x) ~ x
        torch.log1p(torch.exp(beta * x)) / beta,
    )

# Register Meta implementation for shape inference (needed for torch.compile)
@impl(mylib, "softplus", "Meta")
def softplus_meta(x, beta=1.0):
    return torch.empty_like(x)


# ===========================================================================
# Method 2: Using @custom_op decorator (simpler, recommended for new code)
# ===========================================================================

@custom_op("myops::gated_linear", mutates_args=())
def gated_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Gated linear unit: x * sigmoid(Wx + b)."""
    gate = torch.sigmoid(x @ weight.T + bias)
    return x * gate

@gated_linear.register_fake
def gated_linear_fake(x, weight, bias):
    """Shape inference for torch.compile / torch.export."""
    return torch.empty_like(x)


# ===========================================================================
# Method 3: Custom op with autograd support
# ===========================================================================

@custom_op("myops::smooth_l1", mutates_args=())
def smooth_l1(input: torch.Tensor, target: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Smooth L1 loss (Huber loss) — custom implementation with autograd."""
    diff = input - target
    abs_diff = diff.abs()
    loss = torch.where(
        abs_diff < beta,
        0.5 * diff ** 2 / beta,
        abs_diff - 0.5 * beta,
    )
    return loss

@smooth_l1.register_fake
def smooth_l1_fake(input, target, beta=1.0):
    return torch.empty_like(input)

def smooth_l1_setup_context(ctx, inputs, output):
    input, target, beta = inputs
    ctx.save_for_backward(input, target)
    ctx.beta = beta

def smooth_l1_backward(ctx, grad_output):
    input, target = ctx.saved_tensors
    beta = ctx.beta
    diff = input - target
    abs_diff = diff.abs()
    grad_input = torch.where(
        abs_diff < beta,
        diff / beta,
        diff.sign(),
    ) * grad_output
    return grad_input, -grad_input, None  # None for beta (not a Tensor)

smooth_l1.register_autograd(smooth_l1_backward, setup_context=smooth_l1_setup_context)


# ===========================================================================
# Using Custom Ops in a Model
# ===========================================================================

class GatedNetwork(nn.Module):
    """A small network using our custom operators."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gate_weight = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.gate_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Use our custom softplus instead of F.relu
        h = torch.ops.myops.softplus(self.fc1(x), beta=2.0)

        # Use our custom gated linear unit
        h = torch.ops.myops.gated_linear(h, self.gate_weight, self.gate_bias)

        return self.fc2(h)


# ===========================================================================
# Test
# ===========================================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("CUSTOM OP: softplus")
    print("=" * 60)

    x = torch.randn(5)
    result = torch.ops.myops.softplus(x, beta=1.0)
    expected = torch.nn.functional.softplus(x, beta=1.0)
    print(f"  Input:    {x.tolist()}")
    print(f"  Custom:   {result.tolist()}")
    print(f"  PyTorch:  {expected.tolist()}")
    print(f"  Match:    {torch.allclose(result, expected, atol=1e-6)}")

    # Test numerical stability for large values
    x_large = torch.tensor([100.0, 200.0, -100.0])
    result_large = torch.ops.myops.softplus(x_large)
    print(f"\n  Large values: {x_large.tolist()}")
    print(f"  Result:       {result_large.tolist()}")
    print(f"  (No overflow for large positive values)")

    print("\n" + "=" * 60)
    print("CUSTOM OP: gated_linear")
    print("=" * 60)

    x = torch.randn(3, 4)
    w = torch.randn(4, 4)
    b = torch.zeros(4)
    result = torch.ops.myops.gated_linear(x, w, b)
    print(f"  Input shape:  {list(x.shape)}")
    print(f"  Output shape: {list(result.shape)}")
    print(f"  Output range: [{result.min().item():.3f}, {result.max().item():.3f}]")

    print("\n" + "=" * 60)
    print("CUSTOM OP: smooth_l1 (with autograd)")
    print("=" * 60)

    input_t = torch.randn(5, requires_grad=True)
    target_t = torch.randn(5)
    loss = torch.ops.myops.smooth_l1(input_t, target_t, beta=1.0)
    loss_sum = loss.sum()
    loss_sum.backward()

    print(f"  Input:    {input_t.detach().tolist()}")
    print(f"  Target:   {target_t.tolist()}")
    print(f"  Loss:     {loss.detach().tolist()}")
    print(f"  Gradient: {input_t.grad.tolist()}")

    # Verify against PyTorch's smooth_l1_loss
    expected_loss = torch.nn.functional.smooth_l1_loss(
        input_t.detach(), target_t, reduction="none", beta=1.0,
    )
    print(f"  PyTorch:  {expected_loss.tolist()}")
    print(f"  Match:    {torch.allclose(loss.detach(), expected_loss, atol=1e-6)}")

    print("\n" + "=" * 60)
    print("CUSTOM OPS IN A MODEL")
    print("=" * 60)

    model = GatedNetwork(input_dim=10, hidden_dim=32, output_dim=5)
    x = torch.randn(4, 10)
    output = model(x)
    print(f"  Input:  {list(x.shape)}")
    print(f"  Output: {list(output.shape)}")

    # Verify gradients flow through custom ops
    loss = output.sum()
    loss.backward()
    has_grads = all(p.grad is not None for p in model.parameters())
    print(f"  All gradients computed: {has_grads}")

    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params}")

    print("\n" + "=" * 60)
    print("All custom operator demos completed successfully!")
    print("=" * 60)
