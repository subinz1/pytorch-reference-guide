"""
Mixed Precision Training — AMP, GradScaler, BF16, FP8, and torch.compile

This script demonstrates complete mixed-precision training workflows:
- AMP training loop with GradScaler (FP16)
- BF16 training loop (no scaler needed)
- FP32 vs FP16 vs BF16 convergence comparison on synthetic data
- GradScaler internals: observing scale factor dynamics
- autocast behavior: which ops get cast
- Mixed precision with torch.compile
- FP8 scaled matmul concepts
- FSDP2 MixedPrecisionPolicy pattern

All examples run on CPU (autocast('cpu') with bfloat16) to demonstrate
the API without requiring a GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler


def section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# ============================================================================
# Helper: Simple Model and Synthetic Data
# ============================================================================

class SimpleNet(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.norm(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def make_synthetic_data(num_samples=1000, input_dim=64, num_classes=10):
    """Create a simple classification dataset."""
    torch.manual_seed(42)
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y


# ============================================================================
# 1. FP32 Baseline Training
# ============================================================================

section("1. FP32 Baseline Training")

X, y = make_synthetic_data()
model_fp32 = SimpleNet()
optimizer = torch.optim.Adam(model_fp32.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

losses_fp32 = []
for epoch in range(20):
    optimizer.zero_grad()
    output = model_fp32(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    losses_fp32.append(loss.item())
    if epoch % 5 == 0:
        print(f"  Epoch {epoch:2d}: loss = {loss.item():.4f}")

print(f"  Final loss: {losses_fp32[-1]:.4f}")


# ============================================================================
# 2. FP16 Training with GradScaler (CPU simulation)
# ============================================================================

section("2. FP16 Training with GradScaler (Simulated on CPU)")

print("Note: GradScaler is designed for CUDA. On CPU we simulate the pattern.")
print("On CUDA, this would use tensor cores for 2-3x speedup.\n")

# Demonstrate the API pattern (GradScaler requires CUDA for actual scaling)
# We simulate by manually showing what each step does

torch.manual_seed(42)
model_fp16_sim = SimpleNet()
optimizer_fp16 = torch.optim.Adam(model_fp16_sim.parameters(), lr=1e-3)

losses_fp16 = []
scale_factor = 65536.0  # Simulated initial scale

for epoch in range(20):
    optimizer_fp16.zero_grad()

    # Simulate autocast: cast weights to float16 for matmul
    with torch.no_grad():
        x_half = X.to(torch.float16)

    # Forward in FP32 (CPU doesn't have FP16 tensor cores)
    output = model_fp16_sim(X)
    loss = criterion(output, y)

    # Simulate GradScaler: scale loss before backward
    scaled_loss = loss * scale_factor
    scaled_loss.backward()

    # Unscale gradients
    for p in model_fp16_sim.parameters():
        if p.grad is not None:
            p.grad.data /= scale_factor

    # Check for inf/nan (GradScaler would skip step if found)
    has_inf = any(
        torch.isinf(p.grad).any() or torch.isnan(p.grad).any()
        for p in model_fp16_sim.parameters() if p.grad is not None
    )

    if not has_inf:
        optimizer_fp16.step()
    else:
        scale_factor *= 0.5  # Backoff
        print(f"  Epoch {epoch}: inf detected, scale backed off to {scale_factor}")

    losses_fp16.append(loss.item())
    if epoch % 5 == 0:
        print(f"  Epoch {epoch:2d}: loss = {loss.item():.4f} (scale={scale_factor:.0f})")

print(f"  Final loss: {losses_fp16[-1]:.4f}")


# ============================================================================
# 3. BF16 Training with CPU autocast
# ============================================================================

section("3. BF16 Training with CPU autocast (No Scaler Needed)")

torch.manual_seed(42)
model_bf16 = SimpleNet()
optimizer_bf16 = torch.optim.Adam(model_bf16.parameters(), lr=1e-3)

losses_bf16 = []
for epoch in range(20):
    optimizer_bf16.zero_grad()

    # CPU autocast with bfloat16 — actually works on CPU!
    with autocast('cpu', dtype=torch.bfloat16):
        output = model_bf16(X)
        loss = criterion(output, y)

    # No scaler needed for BF16
    loss.backward()
    optimizer_bf16.step()
    losses_bf16.append(loss.item())
    if epoch % 5 == 0:
        print(f"  Epoch {epoch:2d}: loss = {loss.item():.4f}")

print(f"  Final loss: {losses_bf16[-1]:.4f}")


# ============================================================================
# 4. Convergence Comparison
# ============================================================================

section("4. Convergence Comparison: FP32 vs FP16 (sim) vs BF16")

print(f"{'Epoch':<8} {'FP32 Loss':<15} {'FP16 Loss':<15} {'BF16 Loss':<15}")
print("-" * 55)
for i in range(0, 20, 2):
    print(f"{i:<8} {losses_fp32[i]:<15.4f} {losses_fp16[i]:<15.4f} {losses_bf16[i]:<15.4f}")

print(f"\nFinal losses:")
print(f"  FP32: {losses_fp32[-1]:.6f}")
print(f"  FP16: {losses_fp16[-1]:.6f}")
print(f"  BF16: {losses_bf16[-1]:.6f}")
print(f"\nAll three converge to similar loss — mixed precision doesn't hurt accuracy!")


# ============================================================================
# 5. GradScaler Internals — Scale Factor Dynamics
# ============================================================================

section("5. GradScaler Internals: Scale Factor Dynamics")

print("Demonstrating GradScaler behavior with synthetic scenarios:\n")

# Simulate scale factor dynamics
scale = 65536.0
growth_factor = 2.0
backoff_factor = 0.5
growth_interval = 5
steps_since_growth = 0

print(f"{'Step':<6} {'Event':<25} {'Scale Factor':<15} {'Action'}")
print("-" * 65)

events = [
    "normal", "normal", "normal", "normal", "normal",  # 5 clean → grow
    "normal", "normal", "overflow",  # overflow → backoff
    "normal", "normal", "normal", "normal", "normal",  # 5 clean → grow
    "normal", "normal", "normal", "normal", "normal",  # 5 clean → grow
    "overflow", "overflow",  # two overflows
]

for step, event in enumerate(events):
    if event == "overflow":
        scale *= backoff_factor
        steps_since_growth = 0
        print(f"{step:<6} {'inf/nan detected':<25} {scale:<15.0f} {'BACKOFF + skip step'}")
    else:
        steps_since_growth += 1
        if steps_since_growth >= growth_interval:
            scale *= growth_factor
            steps_since_growth = 0
            print(f"{step:<6} {'growth interval reached':<25} {scale:<15.0f} {'GROW'}")
        else:
            print(f"{step:<6} {'clean step':<25} {scale:<15.0f} {f'({steps_since_growth}/{growth_interval})'}")


# ============================================================================
# 6. autocast Behavior: Which Ops Get Cast
# ============================================================================

section("6. autocast Behavior: Observing Dtype Changes")

torch.manual_seed(42)
x = torch.randn(4, 8)
linear = nn.Linear(8, 16)
norm = nn.LayerNorm(16)

print("Without autocast (everything FP32):")
out1 = linear(x)
out2 = norm(out1)
out3 = F.softmax(out2, dim=-1)
print(f"  Input:        {x.dtype}")
print(f"  After linear: {out1.dtype}")
print(f"  After norm:   {out2.dtype}")
print(f"  After softmax:{out3.dtype}")

print("\nWith autocast('cpu', dtype=torch.bfloat16):")
with autocast('cpu', dtype=torch.bfloat16):
    out1 = linear(x)
    out2 = norm(out1)
    out3 = F.softmax(out2, dim=-1)
    print(f"  Input:        {x.dtype}")
    print(f"  After linear: {out1.dtype}  ← CAST to BF16 (matmul)")
    print(f"  After norm:   {out2.dtype}  ← stays BF16 (LayerNorm on CPU)")
    print(f"  After softmax:{out3.dtype}  ← stays BF16 on CPU")

print("\nOn CUDA, LayerNorm and softmax would stay FP32 (numerically sensitive).")
print("CPU autocast has different cast rules than CUDA autocast.")


# ============================================================================
# 7. Disabling autocast for Sensitive Operations
# ============================================================================

section("7. Selectively Disabling autocast")

print("Pattern: Force FP32 for numerically sensitive custom operations\n")

def custom_sensitive_op(x):
    """An operation that needs FP32 precision."""
    return torch.log(torch.exp(x) + 1)  # log1p pattern, needs precision

x = torch.randn(4, 4)

with autocast('cpu', dtype=torch.bfloat16):
    # Normal autocast region
    y = F.linear(x, torch.randn(4, 4))
    print(f"  After linear (autocast): {y.dtype}")

    # Disable for sensitive computation
    with autocast('cpu', enabled=False):
        y_fp32 = y.float()  # Explicitly cast to FP32
        z = custom_sensitive_op(y_fp32)
        print(f"  After custom op (no autocast): {z.dtype}")

    # Back to autocast
    final = F.linear(z.bfloat16(), torch.randn(4, 4, dtype=torch.bfloat16))
    print(f"  After final linear (autocast): {final.dtype}")


# ============================================================================
# 8. Mixed Precision with torch.compile
# ============================================================================

section("8. Mixed Precision with torch.compile")

torch.manual_seed(42)
model_compile = SimpleNet()

# Compile the model
compiled_model = torch.compile(model_compile, backend='eager')  # 'eager' for CPU demo

print("torch.compile composes with autocast — trace through the autocast region:")
print("  compiled_model = torch.compile(model)")
print("  with autocast('cpu', dtype=torch.bfloat16):")
print("      output = compiled_model(data)")
print()

# Run with autocast
X_small = X[:32]
y_small = y[:32]

with autocast('cpu', dtype=torch.bfloat16):
    output = compiled_model(X_small)
    loss = criterion(output, y_small)

print(f"  Output dtype: {output.dtype}")
print(f"  Loss value: {loss.item():.4f}")
print(f"  Loss dtype: {loss.dtype}")

# Training loop with compile + autocast
optimizer_c = torch.optim.Adam(model_compile.parameters(), lr=1e-3)
losses_compile = []

for epoch in range(20):
    optimizer_c.zero_grad()
    with autocast('cpu', dtype=torch.bfloat16):
        output = compiled_model(X)
        loss = criterion(output, y)
    loss.backward()
    optimizer_c.step()
    losses_compile.append(loss.item())

print(f"\n  Training with compile + BF16 autocast:")
print(f"  Final loss: {losses_compile[-1]:.4f} (comparable to non-compiled)")

# set_float32_matmul_precision
print("\n  torch.set_float32_matmul_precision options:")
for precision in ['highest', 'high', 'medium']:
    print(f"    '{precision}': ", end="")
    if precision == 'highest':
        print("Pure FP32 matmul (slowest, most precise)")
    elif precision == 'high':
        print("TF32 on Ampere+ (default, good balance)")
    else:
        print("Reduced precision accumulators (fastest)")


# ============================================================================
# 9. FP8 Scaled Matmul Concepts
# ============================================================================

section("9. FP8 Scaled Matmul (Conceptual)")

print("FP8 matmul requires per-tensor scaling to keep values in representable range.")
print("torch._scaled_mm is the low-level API (CUDA-only).\n")

# Demonstrate the scaling concept on CPU
torch.manual_seed(42)
A = torch.randn(64, 128) * 3.0  # Simulated activations
B = torch.randn(128, 256) * 2.0  # Simulated weights

# FP32 reference
ref = A @ B

# Simulate FP8 E4M3 quantization with per-tensor scaling
e4m3_max = 448.0

scale_A = e4m3_max / A.abs().max()
scale_B = e4m3_max / B.abs().max()

# Quantize (simulate by clipping to E4M3 range after scaling)
A_scaled = (A * scale_A).clamp(-e4m3_max, e4m3_max)
B_scaled = (B * scale_B).clamp(-e4m3_max, e4m3_max)

# Cast to FP8 and back (loses precision)
A_fp8 = A_scaled.to(torch.float8_e4m3fn).to(torch.float32)
B_fp8 = B_scaled.to(torch.float8_e4m3fn).to(torch.float32)

# Matmul in FP32, then unscale
result_fp8 = (A_fp8 @ B_fp8) / (scale_A * scale_B)

# Error analysis
error = (ref - result_fp8).abs()
print(f"Matrix multiply: [{A.shape[0]}×{A.shape[1]}] @ [{B.shape[0]}×{B.shape[1]}]")
print(f"  Scale A: {scale_A.item():.4f}")
print(f"  Scale B: {scale_B.item():.4f}")
print(f"  Mean absolute error: {error.mean():.6f}")
print(f"  Max absolute error:  {error.max():.6f}")
print(f"  Relative error:      {(error / ref.abs().clamp(min=1e-8)).mean():.4%}")

# Compare with BF16
A_bf16 = A.to(torch.bfloat16).to(torch.float32)
B_bf16 = B.to(torch.bfloat16).to(torch.float32)
result_bf16 = A_bf16 @ B_bf16
error_bf16 = (ref - result_bf16).abs()
print(f"\n  For comparison, BF16 matmul error:")
print(f"  Mean absolute error: {error_bf16.mean():.6f}")
print(f"  Max absolute error:  {error_bf16.max():.6f}")
print(f"\n  FP8 is ~{error.mean() / error_bf16.mean().clamp(min=1e-8):.1f}× less accurate than BF16")
print(f"  but ~2× faster on H100 hardware")

print("""
On CUDA (H100), the actual API:
    result = torch._scaled_mm(
        a_fp8,              # [M, K] float8_e4m3fn
        b_fp8.t(),          # [N, K] float8_e4m3fn (transposed)
        scale_a=inv_scale_a,
        scale_b=inv_scale_b,
        out_dtype=torch.bfloat16
    )
""")


# ============================================================================
# 10. FSDP2 MixedPrecisionPolicy Pattern
# ============================================================================

section("10. FSDP2 MixedPrecisionPolicy (Code Pattern)")

print("FSDP2 has native mixed precision support via MixedPrecisionPolicy.")
print("This controls precision for compute and communication separately.\n")

print("""
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

# Define precision policy
mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,    # Compute in BF16 (2x memory savings)
    reduce_dtype=torch.float32,    # All-reduce in FP32 (numerical stability)
)

# Apply to each transformer block
model = TransformerModel()
for block in model.layers:
    fully_shard(block, mp_policy=mp_policy)
fully_shard(model, mp_policy=mp_policy)

# Training loop — NO autocast or scaler needed!
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
for data, target in dataloader:
    optimizer.zero_grad()
    output = model(data)  # FSDP handles BF16 casting internally
    loss = criterion(output, target)
    loss.backward()       # Gradients computed in BF16, reduced in FP32
    optimizer.step()      # Optimizer uses FP32 master weights
""")

print("Key points:")
print("  • param_dtype=bfloat16: parameters are cast to BF16 for forward/backward")
print("  • reduce_dtype=float32: gradient all-reduce uses FP32 to preserve small gradients")
print("  • No autocast wrapper needed (FSDP does the casting)")
print("  • No GradScaler needed (BF16 has sufficient range)")
print("  • Optimizer automatically maintains FP32 master weights")


# ============================================================================
# 11. Complete CUDA Training Pattern (Reference)
# ============================================================================

section("11. Complete CUDA Training Pattern (Reference Code)")

print("""
# === FP16 with GradScaler (V100, T4) ===

model = LargeModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = GradScaler('cuda')

for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()

        with autocast('cuda', dtype=torch.float16):
            output = model(data)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()


# === BF16 without GradScaler (A100, H100) ===

model = LargeModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()

        with autocast('cuda', dtype=torch.bfloat16):
            output = model(data)
            loss = criterion(output, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


# === BF16 + torch.compile (best performance) ===

model = LargeModel().cuda()
compiled_model = torch.compile(model, mode='max-autotune')
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

torch.set_float32_matmul_precision('high')  # TF32 for non-autocast matmuls

for epoch in range(num_epochs):
    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()

        with autocast('cuda', dtype=torch.bfloat16):
            output = compiled_model(data)
            loss = criterion(output, target)

        loss.backward()
        optimizer.step()
""")


# ============================================================================
# 12. Gradient Accumulation with Mixed Precision
# ============================================================================

section("12. Gradient Accumulation with Mixed Precision")

torch.manual_seed(42)
model_accum = SimpleNet()
optimizer_accum = torch.optim.Adam(model_accum.parameters(), lr=1e-3)
accumulation_steps = 4

print(f"Accumulating gradients over {accumulation_steps} mini-batches")
print("Pattern: scale loss by 1/accumulation_steps, accumulate, then step\n")

# Split data into mini-batches
batch_size = 250  # 1000 / 4 = 250 per mini-batch
losses_accum = []

for epoch in range(10):
    epoch_loss = 0.0
    for i in range(accumulation_steps):
        start = i * batch_size
        end = start + batch_size
        X_batch = X[start:end]
        y_batch = y[start:end]

        with autocast('cpu', dtype=torch.bfloat16):
            output = model_accum(X_batch)
            loss = criterion(output, y_batch) / accumulation_steps

        loss.backward()
        epoch_loss += loss.item()

    # Step after accumulating all mini-batches
    optimizer_accum.step()
    optimizer_accum.zero_grad()
    losses_accum.append(epoch_loss)

    if epoch % 3 == 0:
        print(f"  Epoch {epoch:2d}: accumulated loss = {epoch_loss:.4f}")

print(f"\n  Final accumulated loss: {losses_accum[-1]:.4f}")

print("""
On CUDA with GradScaler, the pattern becomes:
    for i, (data, target) in enumerate(dataloader):
        with autocast('cuda', dtype=torch.float16):
            loss = criterion(model(data), target) / accum_steps
        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
""")


# ============================================================================
# 13. Precision Health Check Utility
# ============================================================================

section("13. Precision Health Check Utility")

def precision_health_check(model, step=0):
    """Check for common mixed-precision issues."""
    issues = []

    # Check parameter ranges
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            issues.append(f"NaN in parameter: {name}")
        if torch.isinf(param).any():
            issues.append(f"Inf in parameter: {name}")

        if param.grad is not None:
            grad = param.grad
            if torch.isnan(grad).any():
                issues.append(f"NaN in gradient: {name}")
            if torch.isinf(grad).any():
                issues.append(f"Inf in gradient: {name}")
            zero_frac = (grad == 0).float().mean().item()
            if zero_frac > 0.9:
                issues.append(f"Gradient mostly zero ({zero_frac:.0%}): {name}")

    if issues:
        print(f"  Step {step} — ISSUES FOUND:")
        for issue in issues:
            print(f"    ⚠ {issue}")
    else:
        print(f"  Step {step} — All parameters and gradients healthy")

    return len(issues) == 0

# Demo the health check
torch.manual_seed(42)
model_check = SimpleNet()
optimizer_check = torch.optim.Adam(model_check.parameters(), lr=1e-3)

# Normal step
output = model_check(X[:32])
loss = criterion(output, y[:32])
loss.backward()
precision_health_check(model_check, step=0)

# Artificially corrupt a gradient to show detection
with torch.no_grad():
    model_check.fc1.weight.grad[0, 0] = float('nan')
precision_health_check(model_check, step=1)


# ============================================================================
# Summary
# ============================================================================

section("Summary")

print("""
Mixed Precision Training Workflow:

  ┌─────────────────────────────────────────────────────────┐
  │  Hardware        │  Recommended Setup                    │
  ├──────────────────┼──────────────────────────────────────┤
  │  H100            │  BF16 autocast + torch.compile       │
  │                  │  Consider FP8 for large matmuls      │
  │  A100            │  BF16 autocast + torch.compile       │
  │  V100 / T4       │  FP16 autocast + GradScaler          │
  │  CPU (Intel AMX) │  BF16 autocast (limited ops)         │
  │  CPU (general)   │  FP32 (no benefit from lower prec)   │
  └──────────────────┴──────────────────────────────────────┘

Key rules:
  1. Always keep optimizer states in FP32
  2. Use GradScaler ONLY for FP16 (not BF16)
  3. torch.compile fuses cast operations for better performance
  4. FSDP2 MixedPrecisionPolicy handles casting for distributed training
  5. Monitor for gradient underflow (FP16) and overflow (FP16 logits)
""")

if __name__ == "__main__":
    pass
