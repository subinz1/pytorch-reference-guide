"""
Profiling — PyTorch Profiler Usage
====================================
Covers: basic profiling, timing, memory tracking.
"""

import torch
import torch.nn as nn
import time

print("=" * 60)
print("1. BASIC TIMING")
print("=" * 60)

model = nn.Sequential(
    nn.Linear(1000, 2000),
    nn.ReLU(),
    nn.Linear(2000, 1000),
    nn.ReLU(),
    nn.Linear(1000, 100),
)
x = torch.randn(64, 1000)

# Simple timing
N = 100
start = time.perf_counter()
for _ in range(N):
    with torch.no_grad():
        model(x)
elapsed = (time.perf_counter() - start) / N * 1000
print(f"Average forward pass: {elapsed:.2f} ms")

print("\n" + "=" * 60)
print("2. TORCH PROFILER")
print("=" * 60)

from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU],
    record_shapes=True,
    profile_memory=True,
) as prof:
    with record_function("model_forward"):
        for _ in range(10):
            output = model(x)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

print("\n" + "=" * 60)
print("3. CUSTOM ANNOTATIONS")
print("=" * 60)

class AnnotatedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(1000, 512)
        self.decoder = nn.Linear(512, 100)

    def forward(self, x):
        with record_function("encoder"):
            x = torch.relu(self.encoder(x))
        with record_function("decoder"):
            x = self.decoder(x)
        return x

model2 = AnnotatedModel()
with profile(activities=[ProfilerActivity.CPU]) as prof:
    for _ in range(10):
        model2(x)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

print("\n" + "=" * 60)
print("4. MEMORY PROFILING")
print("=" * 60)

print(f"Tensor allocation tracking:")
x = torch.randn(1000, 1000)
print(f"  1000x1000 float32: {x.numel() * 4 / 1e6:.1f} MB")

x = torch.randn(1000, 1000, dtype=torch.float16)
print(f"  1000x1000 float16: {x.numel() * 2 / 1e6:.1f} MB")

x = torch.randn(1000, 1000, dtype=torch.bfloat16)
print(f"  1000x1000 bfloat16: {x.numel() * 2 / 1e6:.1f} MB")

# Model memory estimation
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.Linear(512, 256),
    nn.Linear(256, 10),
)
param_mem = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"\nModel parameter memory: {param_mem / 1e6:.2f} MB")

# Using meta device for analysis
meta_model = nn.Sequential(
    nn.Linear(784, 512),
    nn.Linear(512, 256),
    nn.Linear(256, 10),
).to('meta')

total_params = sum(p.numel() for p in meta_model.parameters())
print(f"Meta device analysis: {total_params:,} params, "
      f"{total_params * 4 / 1e6:.2f} MB (float32)")

print("\nDone!")
