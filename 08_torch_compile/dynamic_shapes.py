"""
Dynamic Shapes — Handling Varying Input Sizes
===============================================
Demonstrates how torch.compile handles inputs with different shapes:
- Recompilation behavior
- dynamic=True
- mark_dynamic
- Automatic dynamic shape detection

Run: python dynamic_shapes.py
"""

import torch
import torch.nn as nn
import torch._dynamo

# =============================================================================
# 1. The recompilation problem
# =============================================================================

print("=" * 60)
print("DYNAMIC SHAPES IN torch.compile")
print("=" * 60)

print("\n--- The Recompilation Problem ---\n")

# Counter to track compilations
compile_count = 0

def counting_backend(gm, example_inputs):
    """Custom backend that counts how many times compilation occurs."""
    global compile_count
    compile_count += 1
    return gm

def simple_fn(x):
    return x.sin() + x.cos()

# Without dynamic shapes: each new shape triggers recompilation
torch._dynamo.reset()
compile_count = 0
compiled_fn = torch.compile(simple_fn, backend=counting_backend)

print("  Without dynamic=True:")
for batch_size in [8, 16, 32, 8, 16]:
    x = torch.randn(batch_size, 64)
    compiled_fn(x)
    print(f"    batch_size={batch_size:3d}: compilations so far = {compile_count}")

print(f"\n  Total compilations: {compile_count}")
print("  (Recompiled for each new shape, but cached for seen shapes)")

# =============================================================================
# 2. dynamic=True — compile once for any shape
# =============================================================================

print("\n" + "=" * 60)
print("--- dynamic=True ---")
print("=" * 60 + "\n")

torch._dynamo.reset()
compile_count = 0
compiled_dynamic = torch.compile(simple_fn, backend=counting_backend, dynamic=True)

print("  With dynamic=True:")
for batch_size in [8, 16, 32, 64, 128]:
    x = torch.randn(batch_size, 64)
    compiled_dynamic(x)
    print(f"    batch_size={batch_size:3d}: compilations so far = {compile_count}")

print(f"\n  Total compilations: {compile_count}")
print("  (Compiled ONCE with symbolic shapes, works for any batch size!)")

# =============================================================================
# 3. What dynamic=True does internally
# =============================================================================

print("\n" + "=" * 60)
print("--- How Dynamic Shapes Work ---")
print("=" * 60 + "\n")

print("""  With dynamic=True, TorchDynamo uses symbolic integers:
    - Instead of recording shape = [32, 64]
    - It records shape = [s0, s1] where s0, s1 are symbolic
    - Generated code handles any concrete value for s0, s1
    - Guards only check: dtype, device, rank (not exact sizes)
""")

# Demonstrate with a function that depends on shape
def shape_dependent_fn(x):
    """Function where behavior depends on shape (but compilable)."""
    batch, dim = x.shape
    # Scale by sqrt of dimension — dim is symbolic with dynamic=True
    return x / (dim ** 0.5)

torch._dynamo.reset()
compile_count = 0
compiled_shape = torch.compile(shape_dependent_fn, backend=counting_backend, dynamic=True)

# All these work without recompilation
for shape in [(8, 64), (16, 128), (32, 256), (4, 512)]:
    x = torch.randn(*shape)
    result = compiled_shape(x)

print(f"  Ran with 4 different shapes: {compile_count} compilation(s)")
print(f"  dynamic=True handles varying shapes in ALL dimensions")

# =============================================================================
# 4. mark_dynamic — fine-grained control
# =============================================================================

print("\n" + "=" * 60)
print("--- mark_dynamic: Per-Tensor Shape Control ---")
print("=" * 60 + "\n")

def matmul_fn(x, weight):
    """Matrix multiply where x batch varies but weight is fixed."""
    return x @ weight

torch._dynamo.reset()
compile_count = 0
compiled_matmul = torch.compile(matmul_fn, backend=counting_backend)

weight = torch.randn(64, 32)

print("  Using mark_dynamic on batch dimension only:")
for batch_size in [8, 16, 32, 64]:
    x = torch.randn(batch_size, 64)
    # Mark only dimension 0 (batch) as dynamic
    torch._dynamo.mark_dynamic(x, 0)
    result = compiled_matmul(x, weight)
    print(f"    batch={batch_size:3d}, output shape={result.shape}, "
          f"compilations={compile_count}")

print(f"\n  Only 1 compilation for varying batch sizes!")

# =============================================================================
# 5. Automatic dynamic shapes
# =============================================================================

print("\n" + "=" * 60)
print("--- Automatic Dynamic Shape Detection ---")
print("=" * 60 + "\n")

print("  PyTorch automatically marks dimensions as dynamic after")
print("  seeing recompilation on the same dimension.\n")

torch._dynamo.reset()
compile_count = 0
compiled_auto = torch.compile(simple_fn, backend=counting_backend)

# First few calls may recompile, then PyTorch auto-detects dynamic dims
batch_sizes = [8, 16, 32, 64, 128, 7, 13, 99]
for bs in batch_sizes:
    x = torch.randn(bs, 64)
    compiled_auto(x)

print(f"  Ran with {len(batch_sizes)} different batch sizes")
print(f"  Total compilations: {compile_count}")
print(f"  After initial recompilations, Dynamo marks dim 0 as dynamic")

# =============================================================================
# 6. Dynamic shapes with models
# =============================================================================

print("\n" + "=" * 60)
print("--- Dynamic Shapes with nn.Module ---")
print("=" * 60 + "\n")

class SequenceModel(nn.Module):
    """Model that processes variable-length sequences."""

    def __init__(self, dim=128):
        super().__init__()
        self.embed = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, dim)

    def forward(self, x):
        # x shape: [batch, seq_len, dim]
        x = self.embed(x)
        x = self.norm(x)
        # Mean pooling over sequence dimension
        x = x.mean(dim=1)
        return self.output(x)


model = SequenceModel()
model.eval()

torch._dynamo.reset()
compile_count = 0
compiled_model = torch.compile(model, backend=counting_backend, dynamic=True)

print("  Variable batch size AND sequence length:")
with torch.no_grad():
    for batch, seq_len in [(4, 10), (8, 20), (16, 5), (2, 100)]:
        x = torch.randn(batch, seq_len, 128)
        out = compiled_model(x)
        print(f"    Input: [{batch:3d}, {seq_len:3d}, 128] -> Output: {list(out.shape)}, "
              f"compilations={compile_count}")

print(f"\n  All shapes handled with {compile_count} compilation(s)!")

# =============================================================================
# 7. When dynamic shapes DON'T help
# =============================================================================

print("\n" + "=" * 60)
print("--- Limitations of Dynamic Shapes ---")
print("=" * 60 + "\n")

print("  Dynamic shapes don't help when:")
print("  1. Rank (number of dimensions) changes — always recompiles")
print("  2. Dtype changes — always recompiles")
print("  3. Device changes — always recompiles")
print("  4. Certain shape-dependent operations have constraints\n")

# Example: rank change forces recompilation
torch._dynamo.reset()
compile_count = 0

def rank_sensitive(x):
    return x.sum()

compiled_rank = torch.compile(rank_sensitive, backend=counting_backend, dynamic=True)

compiled_rank(torch.randn(8, 64))       # 2D
compiled_rank(torch.randn(8, 64, 32))   # 3D — must recompile!
compiled_rank(torch.randn(8))            # 1D — must recompile!

print(f"  Rank changes: {compile_count} compilations for 3 different ranks")

# Example: dtype change forces recompilation
torch._dynamo.reset()
compile_count = 0

compiled_dtype = torch.compile(rank_sensitive, backend=counting_backend, dynamic=True)
compiled_dtype(torch.randn(8, 64))                           # float32
compiled_dtype(torch.randn(8, 64, dtype=torch.float64))     # float64 — recompile!
compiled_dtype(torch.randint(0, 10, (8, 64)))               # int64 — recompile!

print(f"  Dtype changes: {compile_count} compilations for 3 different dtypes")

# =============================================================================
# 8. Best practices
# =============================================================================

print("\n" + "=" * 60)
print("BEST PRACTICES FOR DYNAMIC SHAPES")
print("=" * 60)
print("""
  1. Use dynamic=True when batch sizes vary (most common case)

  2. Use mark_dynamic for fine-grained control:
     - Mark batch dim as dynamic, keep feature dim static
     - Helps the compiler generate better code for known dimensions

  3. Pad sequences to a few fixed lengths to reduce recompilation:
     - Instead of 1-512 different lengths
     - Bucket into [32, 64, 128, 256, 512]

  4. Use torch.compiler.set_stance("fail_on_recompile") in production
     to catch unexpected recompilations

  5. Profile to verify: sometimes recompilation for a few fixed sizes
     is faster than dynamic code for all sizes
""")

print("Dynamic shapes demonstration complete!")
