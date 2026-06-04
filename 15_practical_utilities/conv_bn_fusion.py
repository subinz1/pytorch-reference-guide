"""
Conv-BN Fusion & Inference Optimization Utilities
===================================================
Learn how to fuse Conv+BatchNorm for faster inference, and other
practical model optimization utilities.
"""

import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_linear_bn_eval
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.nn.utils.init import skip_init
import time

print("=" * 65)
print("1. WHY FUSE Conv + BatchNorm?")
print("=" * 65)

print("""
During training, BatchNorm computes:
    y = gamma * (x - mean) / sqrt(var + eps) + beta

During inference (eval mode), mean and var are FIXED (running stats).
So BatchNorm becomes a simple affine transform: y = scale * x + bias

This can be folded into the preceding Conv's weights:
    Conv: y = W * x + b
    BN:   z = gamma * (y - mean) / sqrt(var + eps) + beta

Combined:
    z = (gamma / sqrt(var + eps)) * W * x + adjusted_bias

Result: ONE layer instead of TWO, with IDENTICAL output.
""")

print("=" * 65)
print("2. CONV-BN FUSION IN PRACTICE")
print("=" * 65)

# Create a conv + batchnorm pair
conv = nn.Conv2d(3, 64, 3, padding=1, bias=True)
bn = nn.BatchNorm2d(64)

# Simulate training (BN needs running stats)
conv.train()
bn.train()
for _ in range(10):
    x = torch.randn(8, 3, 32, 32)
    _ = bn(conv(x))

# Switch to eval mode (required for fusion)
conv.eval()
bn.eval()

# Fuse!
fused_conv = fuse_conv_bn_eval(conv, bn)

# Verify outputs match
x_test = torch.randn(4, 3, 32, 32)
with torch.no_grad():
    out_separate = bn(conv(x_test))
    out_fused = fused_conv(x_test)

print(f"Separate (conv+bn) output shape: {out_separate.shape}")
print(f"Fused conv output shape:         {out_fused.shape}")
print(f"Outputs match: {torch.allclose(out_separate, out_fused, atol=1e-5)}")

print("\n" + "=" * 65)
print("3. SPEED COMPARISON")
print("=" * 65)

# Create a small model
class ConvBNBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        return x

# Train briefly for running stats
model = ConvBNBlock()
model.train()
for _ in range(5):
    _ = model(torch.randn(4, 3, 32, 32))

model.eval()

# Create fused version
fused_model = ConvBNBlock()
fused_model.load_state_dict(model.state_dict())
fused_model.eval()

# Fuse conv+bn pairs
fused_model.conv1 = fuse_conv_bn_eval(fused_model.conv1, fused_model.bn1)
fused_model.bn1 = nn.Identity()
fused_model.conv2 = fuse_conv_bn_eval(fused_model.conv2, fused_model.bn2)
fused_model.bn2 = nn.Identity()
fused_model.conv3 = fuse_conv_bn_eval(fused_model.conv3, fused_model.bn3)
fused_model.bn3 = nn.Identity()

# Verify correctness
x_test = torch.randn(4, 3, 32, 32)
with torch.no_grad():
    out1 = model(x_test)
    out2 = fused_model(x_test)
print(f"Outputs match after fusion: {torch.allclose(out1, out2, atol=1e-5)}")

# Benchmark
N = 50
x_bench = torch.randn(8, 3, 64, 64)

with torch.no_grad():
    start = time.perf_counter()
    for _ in range(N):
        model(x_bench)
    separate_time = (time.perf_counter() - start) / N * 1000

    start = time.perf_counter()
    for _ in range(N):
        fused_model(x_bench)
    fused_time = (time.perf_counter() - start) / N * 1000

print(f"\nSeparate (3x Conv+BN): {separate_time:.2f} ms")
print(f"Fused (3x Conv):       {fused_time:.2f} ms")
print(f"Speedup: {separate_time/fused_time:.2f}x")

# Count parameters
params_sep = sum(p.numel() for p in model.parameters())
params_fused = sum(p.numel() for p in fused_model.parameters())
print(f"\nParameters separate: {params_sep:,}")
print(f"Parameters fused:   {params_fused:,}")

print("\n" + "=" * 65)
print("4. LINEAR-BN FUSION")
print("=" * 65)

linear = nn.Linear(256, 128)
bn = nn.BatchNorm1d(128)

# Train for running stats
linear.train()
bn.train()
for _ in range(10):
    _ = bn(linear(torch.randn(16, 256)))

linear.eval()
bn.eval()

fused_linear = fuse_linear_bn_eval(linear, bn)

x_test = torch.randn(8, 256)
with torch.no_grad():
    out_sep = bn(linear(x_test))
    out_fused = fused_linear(x_test)

print(f"Linear+BN fusion outputs match: {torch.allclose(out_sep, out_fused, atol=1e-5)}")

print("\n" + "=" * 65)
print("5. parameters_to_vector — Flatten All Parameters")
print("=" * 65)

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.Linear(20, 5),
)

# Flatten all parameters into a single vector
vec = parameters_to_vector(model.parameters())
print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
print(f"Flattened vector shape: {vec.shape}")

# Useful for: model comparison, L-BFGS, evolutionary optimization
model2 = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 5))
vec2 = parameters_to_vector(model2.parameters())
distance = (vec - vec2).norm()
print(f"Distance between two random models: {distance:.2f}")

# Write a modified vector back to model
vec_modified = vec * 0.5  # Halve all weights
vector_to_parameters(vec_modified, model.parameters())
vec_after = parameters_to_vector(model.parameters())
print(f"After halving: distance from original = {(vec - vec_after).norm():.2f}")

print("\n" + "=" * 65)
print("6. skip_init — Fast Module Creation")
print("=" * 65)

# Normal creation: allocates memory + initializes weights
start = time.perf_counter()
big_linear = nn.Linear(5000, 5000)
normal_time = (time.perf_counter() - start) * 1000

# skip_init: allocates memory WITHOUT initializing
start = time.perf_counter()
big_linear_fast = skip_init(nn.Linear, 5000, 5000)
skip_time = (time.perf_counter() - start) * 1000

print(f"Normal nn.Linear(5000,5000): {normal_time:.2f} ms")
print(f"skip_init nn.Linear(5000,5000): {skip_time:.2f} ms")
print(f"Speedup: {normal_time/max(skip_time, 0.01):.1f}x")

print("""
When to use skip_init:
  - Loading a pretrained model (init is immediately overwritten)
  - Creating very large models on meta device then materializing
  - Any time you don't need the default initialization
""")

print("Done!")
