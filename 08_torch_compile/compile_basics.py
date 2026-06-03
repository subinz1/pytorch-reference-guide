"""
torch.compile Basics — Getting Started
========================================
Demonstrates the fundamental usage of torch.compile:
- Compiling a model
- Compiling a function
- Using the decorator form
- Observing compilation behavior

Run: python compile_basics.py
"""

import torch
import torch.nn as nn
import time

# =============================================================================
# 1. Basic model compilation
# =============================================================================

print("=" * 60)
print("torch.compile BASICS")
print("=" * 60)

class SimpleMLP(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x):
        return self.net(x) + x  # Residual connection


model = SimpleMLP()
model.eval()

# Compile the model
print("\n--- Compiling a model ---")
compiled_model = torch.compile(model)

# The first call triggers compilation
x = torch.randn(32, 256)

print("First call (triggers compilation)...")
start = time.time()
with torch.no_grad():
    out1 = compiled_model(x)
first_time = time.time() - start
print(f"  Time: {first_time:.3f}s (includes compilation)")

# Subsequent calls use the compiled code
print("Second call (uses compiled code)...")
start = time.time()
with torch.no_grad():
    for _ in range(100):
        out2 = compiled_model(x)
second_time = (time.time() - start) / 100
print(f"  Time per call: {second_time*1000:.3f}ms")

# Verify correctness — compiled output should match eager
with torch.no_grad():
    eager_out = model(x)
print(f"\nOutputs match: {torch.allclose(out1, eager_out, atol=1e-5)}")

# =============================================================================
# 2. Compiling a function
# =============================================================================

print("\n" + "=" * 60)
print("COMPILING A FUNCTION")
print("=" * 60)

def my_function(x, y):
    """A function with multiple operations that can be fused."""
    z = torch.matmul(x, y.T)
    z = z / z.shape[-1] ** 0.5
    z = torch.softmax(z, dim=-1)
    return z

compiled_fn = torch.compile(my_function)

a = torch.randn(64, 128)
b = torch.randn(64, 128)

# Trigger compilation
result = compiled_fn(a, b)
print(f"\nInput shapes: {a.shape}, {b.shape}")
print(f"Output shape: {result.shape}")
print(f"Output sum: {result.sum():.4f} (should be {a.shape[0]}.0 since softmax rows sum to 1)")

# =============================================================================
# 3. Decorator form
# =============================================================================

print("\n" + "=" * 60)
print("DECORATOR FORM")
print("=" * 60)

@torch.compile
def fused_residual_norm(x, weight, bias, eps=1e-5):
    """This function benefits from fusion — norm + residual in one kernel."""
    residual = x
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x = (x - mean) / torch.sqrt(var + eps)
    x = x * weight + bias
    return x + residual

dim = 512
x = torch.randn(32, dim)
weight = torch.ones(dim)
bias = torch.zeros(dim)

result = fused_residual_norm(x, weight, bias)
print(f"\nfused_residual_norm output shape: {result.shape}")
print(f"Output mean: {result.mean():.6f}")

# =============================================================================
# 4. Compile does NOT modify the original model
# =============================================================================

print("\n" + "=" * 60)
print("COMPILED vs ORIGINAL — Independent Objects")
print("=" * 60)

model = SimpleMLP()
compiled = torch.compile(model)

print(f"\nOriginal model type: {type(model)}")
print(f"Compiled model type: {type(compiled)}")
print(f"Same object? {model is compiled}")

# Both share the same parameters (compile wraps, doesn't copy)
print(f"Share parameters? {next(model.parameters()).data_ptr() == next(compiled._orig_mod.parameters()).data_ptr()}")

# Updating model parameters affects compiled version too
with torch.no_grad():
    next(model.parameters()).fill_(0.0)
    x = torch.randn(1, 256)
    out_compiled = compiled(x)
    out_eager = model(x)
    print(f"After zeroing weights, both produce same output: "
          f"{torch.allclose(out_compiled, out_eager, atol=1e-6)}")

# =============================================================================
# 5. torch.compile with training
# =============================================================================

print("\n" + "=" * 60)
print("COMPILING FOR TRAINING")
print("=" * 60)

model = SimpleMLP()
compiled_model = torch.compile(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# torch.compile works with training (backward pass + optimizer step)
print("\nTraining with compiled model:")
for step in range(5):
    x = torch.randn(32, 256)
    target = torch.randn(32, 256)

    optimizer.zero_grad()
    output = compiled_model(x)
    loss = nn.functional.mse_loss(output, target)
    loss.backward()
    optimizer.step()

    print(f"  Step {step+1}: loss = {loss.item():.4f}")

# =============================================================================
# 6. Checking if a model/function is compiled
# =============================================================================

print("\n" + "=" * 60)
print("UTILITY: Checking Compilation Status")
print("=" * 60)

def is_compiled(fn_or_model):
    """Check if something has been compiled."""
    return hasattr(fn_or_model, '_orig_mod') or hasattr(fn_or_model, '_torchdynamo_orig_callable')

model = SimpleMLP()
compiled = torch.compile(model)

print(f"\n  Original model compiled? {is_compiled(model)}")
print(f"  Compiled model compiled? {is_compiled(compiled)}")
print(f"  Access original: {type(compiled._orig_mod)}")

# =============================================================================
# 7. Disabling compilation for specific functions
# =============================================================================

print("\n" + "=" * 60)
print("DISABLING COMPILE FOR SPECIFIC CODE")
print("=" * 60)

@torch._dynamo.disable()
def non_compilable_helper(x):
    """This function will always run in eager mode, even inside compiled code."""
    # Maybe it does something torch.compile can't handle
    return x.numpy().mean()  # .numpy() isn't compilable

class ModelWithEagerPart(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64)

    def forward(self, x):
        x = self.linear(x)
        x = torch.relu(x)
        return x

model = ModelWithEagerPart()
compiled = torch.compile(model)
out = compiled(torch.randn(8, 64))
print(f"\n  Model with @disable decorated helper works: shape={out.shape}")

# =============================================================================
# 8. torch.compiler.is_compiling() — detect if inside compile
# =============================================================================

print("\n" + "=" * 60)
print("DETECTING COMPILATION CONTEXT")
print("=" * 60)

class AdaptiveModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64)

    def forward(self, x):
        if torch.compiler.is_compiling():
            # Optimized path during compilation
            return torch.relu(self.linear(x))
        else:
            # Debug/eager path
            out = self.linear(x)
            return torch.relu(out)

model = AdaptiveModel()
compiled = torch.compile(model)

with torch.no_grad():
    eager_out = model(torch.randn(4, 64))
    compiled_out = compiled(torch.randn(4, 64))

print(f"\n  Eager path runs without is_compiling: works")
print(f"  Compiled path detects compilation: works")

# =============================================================================
# 9. Resetting compiled state
# =============================================================================

print("\n" + "=" * 60)
print("RESETTING COMPILED STATE")
print("=" * 60)

# Reset clears all compiled graphs and caches
torch._dynamo.reset()
print("\n  torch._dynamo.reset() called — all compiled state cleared")
print("  Next call to any compiled function will trigger recompilation")

# After reset, compilation happens again
model = SimpleMLP()
compiled = torch.compile(model)
with torch.no_grad():
    out = compiled(torch.randn(8, 256))
print(f"  After reset, compilation works normally: shape={out.shape}")

print("\ntorch.compile basics complete!")
