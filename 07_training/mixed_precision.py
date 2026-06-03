"""
Mixed Precision Training — AMP with float16 and bfloat16
=========================================================
Demonstrates automatic mixed precision (AMP) training on CPU using bfloat16,
and shows the GradScaler pattern that would be used with float16 on GPU.

Run: python mixed_precision.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

# =============================================================================
# 1. Model definition — a reasonably sized MLP
# =============================================================================

class MediumMLP(nn.Module):
    """Medium-size MLP to demonstrate mixed precision benefits."""

    def __init__(self, input_dim=256, hidden_dim=512, output_dim=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


# =============================================================================
# 2. Create synthetic data
# =============================================================================

torch.manual_seed(42)

num_samples = 5000
input_dim = 256
num_classes = 10

X = torch.randn(num_samples, input_dim)
y = torch.randint(0, num_classes, (num_samples,))

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# =============================================================================
# 3. Training WITHOUT mixed precision (baseline)
# =============================================================================

print("=" * 60)
print("MIXED PRECISION TRAINING COMPARISON")
print("=" * 60)

def train_epoch(model, loader, optimizer, loss_fn, use_amp=False, amp_dtype=None):
    """Train for one epoch, optionally with AMP."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for inputs, targets in loader:
        optimizer.zero_grad(set_to_none=True)

        if use_amp and amp_dtype is not None:
            with torch.amp.autocast(device_type='cpu', dtype=amp_dtype):
                output = model(inputs)
                loss = loss_fn(output, targets)
        else:
            output = model(inputs)
            loss = loss_fn(output, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


print("\n--- Baseline: float32 training ---")
model_fp32 = MediumMLP()
optimizer_fp32 = optim.Adam(model_fp32.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

start = time.time()
for epoch in range(5):
    loss = train_epoch(model_fp32, dataloader, optimizer_fp32, loss_fn)
fp32_time = time.time() - start
print(f"  5 epochs in {fp32_time:.3f}s, final loss: {loss:.4f}")

# =============================================================================
# 4. Training WITH bfloat16 mixed precision
# =============================================================================

print("\n--- BFloat16 mixed precision ---")
model_bf16 = MediumMLP()
optimizer_bf16 = optim.Adam(model_bf16.parameters(), lr=1e-3)

start = time.time()
for epoch in range(5):
    loss = train_epoch(model_bf16, dataloader, optimizer_bf16, loss_fn,
                       use_amp=True, amp_dtype=torch.bfloat16)
bf16_time = time.time() - start
print(f"  5 epochs in {bf16_time:.3f}s, final loss: {loss:.4f}")
print(f"  Speedup: {fp32_time / bf16_time:.2f}x")

# =============================================================================
# 5. Demonstrating what autocast does to dtypes
# =============================================================================

print("\n" + "=" * 60)
print("WHAT AUTOCAST DOES TO TENSOR DTYPES")
print("=" * 60)

model = MediumMLP()
model.eval()
sample = torch.randn(1, 256)

print(f"\nInput dtype: {sample.dtype}")

# Without autocast
with torch.no_grad():
    out_fp32 = model(sample)
print(f"Output without autocast: dtype={out_fp32.dtype}")

# With autocast
with torch.no_grad():
    with torch.amp.autocast(device_type='cpu', dtype=torch.bfloat16):
        out_bf16 = model(sample)
print(f"Output with autocast (bfloat16): dtype={out_bf16.dtype}")

# Show per-layer dtypes inside autocast
print("\nPer-layer dtypes inside autocast context:")
x = sample
with torch.no_grad():
    with torch.amp.autocast(device_type='cpu', dtype=torch.bfloat16):
        for i, layer in enumerate(model.layers):
            x = layer(x)
            if hasattr(layer, 'weight'):
                print(f"  After layer {i} ({layer.__class__.__name__}): "
                      f"output dtype={x.dtype}")

# =============================================================================
# 6. GradScaler pattern (for float16 on GPU — shown structurally on CPU)
# =============================================================================

print("\n" + "=" * 60)
print("GRADSCALER PATTERN (float16 / GPU style)")
print("=" * 60)
print("\nGradScaler is needed for float16 because of its limited range.")
print("This shows the code pattern — actual speedup requires GPU.\n")

model_scaler = MediumMLP()
optimizer_scaler = optim.Adam(model_scaler.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler(enabled=True)

model_scaler.train()
sample_input = torch.randn(32, 256)
sample_target = torch.randint(0, 10, (32,))

# The float16 + GradScaler training pattern
optimizer_scaler.zero_grad()

# autocast casts ops to float16 where safe
# On CPU, float16 autocast is limited, so we demonstrate bfloat16 instead
# but show the GradScaler pattern which is specific to float16/GPU
with torch.amp.autocast(device_type='cpu', dtype=torch.bfloat16):
    output = model_scaler(sample_input)
    loss = loss_fn(output, sample_target)

# Scale loss to prevent gradient underflow (float16 specific)
scaler.scale(loss).backward()

# Unscale gradients, then clip
scaler.unscale_(optimizer_scaler)
torch.nn.utils.clip_grad_norm_(model_scaler.parameters(), max_norm=1.0)

# Step (may skip if inf/nan detected in gradients)
scaler.step(optimizer_scaler)

# Update the scale factor for next iteration
scaler.update()

print("GradScaler training step completed successfully.")
print(f"  Current scale: {scaler.get_scale()}")
print(f"  Loss value: {loss.item():.4f}")

# =============================================================================
# 7. Comparing numerical differences between precisions
# =============================================================================

print("\n" + "=" * 60)
print("NUMERICAL PRECISION COMPARISON")
print("=" * 60)

model = MediumMLP()
model.eval()
test_input = torch.randn(16, 256)

with torch.no_grad():
    # Full precision
    out_fp32 = model(test_input)

    # bfloat16
    with torch.amp.autocast(device_type='cpu', dtype=torch.bfloat16):
        out_bf16 = model(test_input)

# Compare outputs
diff = (out_fp32.float() - out_bf16.float()).abs()
print(f"\nfloat32 output range: [{out_fp32.min():.4f}, {out_fp32.max():.4f}]")
print(f"bfloat16 output range: [{out_bf16.min():.4f}, {out_bf16.max():.4f}]")
print(f"Max absolute difference: {diff.max().item():.6f}")
print(f"Mean absolute difference: {diff.mean().item():.6f}")
print(f"Relative error: {(diff / (out_fp32.abs() + 1e-8)).mean().item():.6f}")

# =============================================================================
# 8. Memory usage comparison
# =============================================================================

print("\n" + "=" * 60)
print("MEMORY USAGE OF DIFFERENT DTYPES")
print("=" * 60)

size = (1000, 1000)
fp32_tensor = torch.randn(*size, dtype=torch.float32)
bf16_tensor = fp32_tensor.to(torch.bfloat16)
fp16_tensor = fp32_tensor.to(torch.float16)

print(f"\nTensor shape: {size}")
print(f"  float32: {fp32_tensor.element_size() * fp32_tensor.nelement() / 1024:.1f} KB")
print(f"  bfloat16: {bf16_tensor.element_size() * bf16_tensor.nelement() / 1024:.1f} KB")
print(f"  float16: {fp16_tensor.element_size() * fp16_tensor.nelement() / 1024:.1f} KB")
print(f"  Memory reduction: {1 - bf16_tensor.element_size() / fp32_tensor.element_size():.0%}")

# =============================================================================
# 9. Which operations stay in float32 during autocast?
# =============================================================================

print("\n" + "=" * 60)
print("OPERATIONS THAT STAY IN FLOAT32 DURING AUTOCAST")
print("=" * 60)
print("""
Operations automatically kept in float32 for numerical stability:
  - Loss functions (cross_entropy, mse_loss, etc.)
  - Softmax and log_softmax
  - Layer normalization
  - Batch normalization (accumulation)
  - Operations on small tensors
  - Reductions (sum, mean) in some contexts

Operations cast to lower precision (faster):
  - Linear layers (matrix multiply)
  - Convolutions
  - BMM (batch matrix multiply)
  - Most element-wise operations
""")

print("Done! Mixed precision training demonstration complete.")
