"""
torch.compile Basics — JIT Compilation for Performance
========================================================
Covers: basic usage, modes, dynamic shapes, debugging, graph breaks.
Runs on CPU (no GPU required).
"""

import torch
import torch.nn as nn
import time

print("=" * 60)
print("1. BASIC torch.compile USAGE")
print("=" * 60)

# Simple function compilation
def my_fn(x, y):
    return (x + y).relu().mul(2)

compiled_fn = torch.compile(my_fn)

x = torch.randn(1000, 1000)
y = torch.randn(1000, 1000)

# First call triggers compilation
result = compiled_fn(x, y)
print(f"Compiled function output shape: {result.shape}")

# Verify correctness
expected = my_fn(x, y)
print(f"Matches eager: {torch.allclose(result, expected)}")

print("\n" + "=" * 60)
print("2. DECORATOR SYNTAX")
print("=" * 60)

@torch.compile
def fast_gelu(x):
    return x * torch.sigmoid(1.702 * x)

result = fast_gelu(torch.randn(100))
print(f"Compiled GELU output shape: {result.shape}")

print("\n" + "=" * 60)
print("3. COMPILING A MODEL")
print("=" * 60)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = SimpleModel()
compiled_model = torch.compile(model)

x = torch.randn(32, 256)
output = compiled_model(x)
print(f"Compiled model output: {output.shape}")

print("\n" + "=" * 60)
print("4. COMPILATION MODES")
print("=" * 60)

# Different modes trade compile time for runtime performance
modes = {
    "default": "Balanced compilation",
    "reduce-overhead": "CUDA graphs (best for small models)",
    "max-autotune": "Maximum optimization (slowest compile)",
}

for mode, desc in modes.items():
    print(f"  mode='{mode}': {desc}")

# Example
model_default = torch.compile(SimpleModel(), mode="default")
print(f"\nDefault mode output: {model_default(x).shape}")

print("\n" + "=" * 60)
print("5. FULLGRAPH MODE")
print("=" * 60)

# fullgraph=True errors on graph breaks
@torch.compile(fullgraph=True)
def no_breaks(x):
    y = x.sin()
    z = y.cos()
    return z + x

result = no_breaks(torch.randn(10))
print(f"fullgraph output: {result.shape}")

print("\n" + "=" * 60)
print("6. DYNAMIC SHAPES")
print("=" * 60)

@torch.compile(dynamic=True)
def dynamic_fn(x):
    return x.sum(dim=-1)

# One compilation handles all shapes
for size in [16, 32, 64, 128]:
    result = dynamic_fn(torch.randn(size, 256))
    print(f"  Input ({size}, 256) -> Output {result.shape}")

print("\n" + "=" * 60)
print("7. UNDERSTANDING GRAPH BREAKS")
print("=" * 60)

# These cause graph breaks (Dynamo falls back to eager):
# - print() inside compiled function
# - Unsupported Python constructs
# - Data-dependent control flow

def has_break(x):
    y = x + 1
    # print(y)  # This would cause a graph break!
    z = y * 2
    return z

compiled = torch.compile(has_break)
print(f"Function with potential break: {compiled(torch.randn(5))}")

print("\n" + "=" * 60)
print("8. EXPLAIN COMPILATION")
print("=" * 60)

def example_fn(x):
    return x.sin() + x.cos()

explanation = torch._dynamo.explain(example_fn)(torch.randn(10))
print(f"Explanation:\n{explanation}")

print("\n" + "=" * 60)
print("9. BENCHMARKING (CPU)")
print("=" * 60)

model = SimpleModel()
compiled = torch.compile(model)
x = torch.randn(64, 256)

# Warmup
for _ in range(3):
    compiled(x)

# Benchmark
N = 100
start = time.perf_counter()
for _ in range(N):
    model(x)
eager_time = (time.perf_counter() - start) / N * 1000

start = time.perf_counter()
for _ in range(N):
    compiled(x)
compiled_time = (time.perf_counter() - start) / N * 1000

print(f"Eager:    {eager_time:.2f} ms/iter")
print(f"Compiled: {compiled_time:.2f} ms/iter")
print(f"Speedup:  {eager_time/compiled_time:.2f}x")

# Reset dynamo state
torch._dynamo.reset()

print("\nDone!")
