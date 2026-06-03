"""
Module 03: Custom Autograd Functions
======================================
Writing custom forward/backward passes, using ctx to save state,
and verifying correctness with gradcheck.

Run: python custom_functions.py
"""

import torch
from torch.autograd import Function, gradcheck

print("=" * 70)
print("PART 1: BASIC CUSTOM FUNCTION — RELU")
print("=" * 70)

class MyReLU(Function):
    """Custom ReLU: f(x) = max(0, x)."""

    @staticmethod
    def forward(ctx, input):
        # Save input for backward pass
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output is the upstream gradient (d_loss/d_output)
        input, = ctx.saved_tensors
        # ReLU gradient: 1 where input > 0, 0 where input <= 0
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

# Test it
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
y = MyReLU.apply(x)
print(f"Input:  {x.tolist()}")
print(f"Output: {y.tolist()}")

loss = y.sum()
loss.backward()
print(f"Gradient: {x.grad.tolist()}")
print("(0 for negative inputs, 1 for positive — ReLU derivative)")

# Verify with gradcheck
x_check = torch.randn(5, dtype=torch.float64, requires_grad=True)
passed = gradcheck(MyReLU.apply, (x_check,), eps=1e-6)
print(f"\nGradcheck passed: {passed}")


print("\n" + "=" * 70)
print("PART 2: CUSTOM FUNCTION WITH MULTIPLE INPUTS")
print("=" * 70)

class WeightedAdd(Function):
    """f(x, y, alpha) = alpha * x + (1 - alpha) * y.
    Computes gradients for x and y but NOT alpha (treated as constant).
    """

    @staticmethod
    def forward(ctx, x, y, alpha):
        ctx.save_for_backward(x, y)
        ctx.alpha = alpha  # Non-tensor data stored directly on ctx
        return alpha * x + (1 - alpha) * y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        alpha = ctx.alpha

        # Must return one gradient per forward input
        grad_x = grad_output * alpha       # d_out/d_x = alpha
        grad_y = grad_output * (1 - alpha)  # d_out/d_y = (1 - alpha)
        grad_alpha = None                    # Not differentiable w.r.t. alpha

        return grad_x, grad_y, grad_alpha

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
alpha = 0.3

result = WeightedAdd.apply(x, y, alpha)
print(f"x = {x.tolist()}")
print(f"y = {y.tolist()}")
print(f"alpha = {alpha}")
print(f"Result = alpha*x + (1-alpha)*y = {result.tolist()}")

result.sum().backward()
print(f"\nx.grad = {x.grad.tolist()} (should all be {alpha})")
print(f"y.grad = {y.grad.tolist()} (should all be {1-alpha})")


print("\n" + "=" * 70)
print("PART 3: CUSTOM SIGMOID WITH EFFICIENT BACKWARD")
print("=" * 70)

class MySigmoid(Function):
    """Custom sigmoid that saves the OUTPUT (not input) for efficiency.

    sigmoid(x) = 1 / (1 + exp(-x))
    d_sigmoid/dx = sigmoid(x) * (1 - sigmoid(x))

    By saving the output, we avoid recomputing exp(-x) during backward.
    """

    @staticmethod
    def forward(ctx, input):
        output = 1 / (1 + torch.exp(-input))
        ctx.save_for_backward(output)  # Save output, not input
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        # d_sigmoid/dx = output * (1 - output)
        return grad_output * output * (1 - output)

x = torch.linspace(-3, 3, 7, requires_grad=True)
y = MySigmoid.apply(x)
y.sum().backward()

print(f"x:       {x.detach().tolist()}")
print(f"sigmoid: {y.detach().round(decimals=4).tolist()}")
print(f"grad:    {x.grad.round(decimals=4).tolist()}")

# Compare with built-in
x_builtin = torch.linspace(-3, 3, 7, requires_grad=True)
y_builtin = torch.sigmoid(x_builtin)
y_builtin.sum().backward()
print(f"\nBuilt-in sigmoid grad: {x_builtin.grad.round(decimals=4).tolist()}")
print(f"Match: {torch.allclose(x.grad, x_builtin.grad)}")

# Verify with gradcheck (use float64 for numerical precision)
x_check = torch.randn(10, dtype=torch.float64, requires_grad=True)
passed = gradcheck(MySigmoid.apply, (x_check,))
print(f"Gradcheck passed: {passed}")


print("\n" + "=" * 70)
print("PART 4: STRAIGHT-THROUGH ESTIMATOR (STE)")
print("=" * 70)

class StraightThroughEstimator(Function):
    """Binarize in forward, pass gradient straight through in backward.

    This is used in quantization-aware training and binary neural networks.
    Forward: output = sign(input)  (non-differentiable!)
    Backward: pretend it was the identity function (gradient = 1)
    """

    @staticmethod
    def forward(ctx, input):
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # Pass gradient through unchanged

x = torch.tensor([-0.5, -0.1, 0.0, 0.3, 0.8], requires_grad=True)
y = StraightThroughEstimator.apply(x)
loss = y.sum()
loss.backward()

print(f"Input:    {x.tolist()}")
print(f"Binarized: {y.tolist()}")
print(f"Gradient:  {x.grad.tolist()}")
print("(sign() has zero gradient everywhere, but STE pretends it's 1)")
print("This trick enables training networks with discrete operations.")


print("\n" + "=" * 70)
print("PART 5: GUMBEL-SOFTMAX TEMPERATURE SCALING")
print("=" * 70)

class TemperatureScaledSoftmax(Function):
    """Softmax with temperature: softmax(x / temperature).

    At low temperature, approaches argmax (hard selection).
    At high temperature, approaches uniform distribution.
    """

    @staticmethod
    def forward(ctx, logits, temperature):
        scaled = logits / temperature
        probs = torch.softmax(scaled, dim=-1)
        ctx.save_for_backward(probs)
        ctx.temperature = temperature
        return probs

    @staticmethod
    def backward(ctx, grad_output):
        probs, = ctx.saved_tensors
        T = ctx.temperature

        # Jacobian of softmax: diag(p) - p*p^T
        # d_softmax/d_logits = (diag(p) - p*p^T) / T
        inner = (grad_output * probs).sum(dim=-1, keepdim=True)
        grad_logits = (grad_output * probs - probs * inner) / T

        return grad_logits, None  # No gradient for temperature

logits = torch.tensor([2.0, 1.0, 0.5], requires_grad=True)

print(f"Logits: {logits.tolist()}\n")
for temp in [0.1, 0.5, 1.0, 2.0, 10.0]:
    probs = TemperatureScaledSoftmax.apply(logits.detach().requires_grad_(True), temp)
    print(f"T={temp:>4.1f}: probs={probs.detach().round(decimals=3).tolist()}")


print("\n" + "=" * 70)
print("PART 6: CUSTOM FUNCTION WITH NEEDS_INPUT_GRAD")
print("=" * 70)

class ConditionalLinear(Function):
    """Linear transformation: output = input @ weight + bias.
    Only computes gradients for inputs that actually need them.
    """

    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)
        return input @ weight + bias

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output @ weight.T
        if ctx.needs_input_grad[1]:
            grad_weight = input.T @ grad_output
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

# Test with frozen weight (no gradient needed)
x = torch.randn(4, 3, requires_grad=True)
w = torch.randn(3, 2, requires_grad=False)  # Frozen!
b = torch.randn(2, requires_grad=True)

y = ConditionalLinear.apply(x, w, b)
y.sum().backward()

print(f"x.grad exists: {x.grad is not None} (shape: {x.grad.shape if x.grad is not None else 'N/A'})")
print(f"w.grad exists: {w.grad is not None} (frozen — no gradient computed)")
print(f"b.grad exists: {b.grad is not None} (shape: {b.grad.shape if b.grad is not None else 'N/A'})")
print("\nneeds_input_grad avoids computing unnecessary gradients")


print("\n" + "=" * 70)
print("PART 7: USING GRADCHECK TO FIND BUGS")
print("=" * 70)

class BuggySquare(Function):
    """Intentionally buggy: wrong gradient."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input ** 2

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * 3 * input  # BUG: should be 2*input, not 3*input

class CorrectSquare(Function):
    """Correct implementation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input ** 2

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * 2 * input

x = torch.randn(3, dtype=torch.float64, requires_grad=True)

print("Testing BuggySquare:")
try:
    passed = gradcheck(BuggySquare.apply, (x,), raise_exception=True)
    print(f"  Gradcheck passed: {passed}")
except Exception as e:
    error_lines = str(e).split('\n')
    print(f"  FAILED: {error_lines[0][:80]}...")
    print("  gradcheck detected the incorrect gradient!")

print("\nTesting CorrectSquare:")
passed = gradcheck(CorrectSquare.apply, (x,))
print(f"  Gradcheck passed: {passed}")


print("\n" + "=" * 70)
print("PART 8: PRACTICAL EXAMPLE — CUSTOM LOSS FUNCTION")
print("=" * 70)

class FocalLoss(Function):
    """Focal loss for addressing class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    The (1-p_t)^gamma factor down-weights easy examples and
    focuses training on hard examples.
    """

    @staticmethod
    def forward(ctx, logits, targets, gamma=2.0, alpha=0.25):
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        focal_weight = (1 - p_t) ** gamma
        loss = -alpha_t * focal_weight * torch.log(p_t + 1e-8)

        ctx.save_for_backward(logits, targets, p_t, alpha_t, focal_weight)
        ctx.gamma = gamma

        return loss.sum()

    @staticmethod
    def backward(ctx, grad_output):
        logits, targets, p_t, alpha_t, focal_weight = ctx.saved_tensors
        gamma = ctx.gamma

        probs = torch.sigmoid(logits)
        dp_dx = probs * (1 - probs)  # sigmoid derivative

        # d/dp_t[-alpha_t * (1-p_t)^gamma * log(p_t)]
        # = -alpha_t * [-gamma*(1-p_t)^(gamma-1)*log(p_t) + (1-p_t)^gamma/p_t]
        dp_t_dx = (2 * targets - 1) * dp_dx

        term1 = gamma * (1 - p_t) ** (gamma - 1) * torch.log(p_t + 1e-8) * dp_t_dx
        term2 = -(1 - p_t) ** gamma / (p_t + 1e-8) * dp_t_dx

        grad_logits = grad_output * (-alpha_t) * (term1 + term2)

        return grad_logits, None, None, None

# Test focal loss
torch.manual_seed(42)
logits = torch.randn(5, requires_grad=True)
targets = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0])

loss = FocalLoss.apply(logits, targets)
loss.backward()

print(f"Logits:  {logits.detach().round(decimals=3).tolist()}")
print(f"Targets: {targets.tolist()}")
print(f"Focal loss: {loss.item():.4f}")
print(f"Gradients: {logits.grad.round(decimals=4).tolist()}")

print("\n" + "=" * 70)
print("Custom autograd functions demonstration complete!")
print("=" * 70)
