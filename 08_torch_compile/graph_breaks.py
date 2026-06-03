"""
Graph Breaks — Examples and Fixes
==================================
Demonstrates what causes graph breaks in torch.compile and how to fix them.

Run: python graph_breaks.py
"""

import torch
import torch.nn as nn
import torch._dynamo

# =============================================================================
# 1. What is a graph break?
# =============================================================================

print("=" * 60)
print("GRAPH BREAKS IN torch.compile")
print("=" * 60)
print("""
A graph break splits compiled code into multiple segments.
Between segments, execution falls back to Python (slow).

Goal: Minimize graph breaks for maximum optimization.
""")

# =============================================================================
# 2. Example: print() causes a graph break
# =============================================================================

print("--- Example 1: print() causes a graph break ---\n")

class ModelWithPrint(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(64, 64)
        self.linear2 = nn.Linear(64, 64)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        print(f"Intermediate shape: {x.shape}")  # GRAPH BREAK!
        x = self.linear2(x)
        return x

model_break = ModelWithPrint()
x = torch.randn(8, 64)

# Use explain() to see graph breaks
explanation = torch._dynamo.explain(model_break)(x)
print(f"  Number of graph segments: {explanation.graph_count}")
print(f"  Graph break reasons:")
for i, reason in enumerate(explanation.break_reasons):
    print(f"    Break {i+1}: {reason.reason}")

# FIX: Remove the print
class ModelWithoutPrint(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(64, 64)
        self.linear2 = nn.Linear(64, 64)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x

model_fixed = ModelWithoutPrint()
explanation = torch._dynamo.explain(model_fixed)(x)
print(f"\n  Fixed model — graph segments: {explanation.graph_count} (single graph = optimal)")

# =============================================================================
# 3. Example: Data-dependent control flow
# =============================================================================

print("\n" + "=" * 60)
print("--- Example 2: Data-dependent control flow ---")
print("=" * 60 + "\n")

def bad_relu(x):
    """BAD: Uses Python if on tensor value — causes graph break."""
    if x.sum().item() > 0:  # Can't compile: value depends on data!
        return x
    else:
        return torch.zeros_like(x)

def good_relu_alternative(x):
    """GOOD: Uses torch.where — no graph break."""
    return torch.where(x.sum() > 0, x, torch.zeros_like(x))

x = torch.randn(8, 64)

explanation_bad = torch._dynamo.explain(bad_relu)(x)
print(f"  Data-dependent if: {explanation_bad.graph_count} graph segments")

explanation_good = torch._dynamo.explain(good_relu_alternative)(x)
print(f"  torch.where fix:   {explanation_good.graph_count} graph segment (no break)")

# =============================================================================
# 4. Example: Calling .item() or .numpy()
# =============================================================================

print("\n" + "=" * 60)
print("--- Example 3: .item() and .numpy() ---")
print("=" * 60 + "\n")

def fn_with_item(x):
    """BAD: .item() forces graph break (converts tensor to Python scalar)."""
    val = x.mean().item()  # Forces synchronization + Python scalar
    return x * val

def fn_without_item(x):
    """GOOD: Keep everything as tensors."""
    val = x.mean()  # Stays as a tensor
    return x * val

x = torch.randn(8, 64)

explanation = torch._dynamo.explain(fn_with_item)(x)
print(f"  With .item():    {explanation.graph_count} graph segments")

explanation = torch._dynamo.explain(fn_without_item)(x)
print(f"  Without .item(): {explanation.graph_count} graph segment")

# =============================================================================
# 5. Example: Python builtins on tensors
# =============================================================================

print("\n" + "=" * 60)
print("--- Example 4: Python builtins ---")
print("=" * 60 + "\n")

def fn_with_builtin(x):
    """Potentially problematic: some builtins may cause breaks."""
    # len() on a tensor is fine (known at compile time via guard)
    n = x.shape[0]
    return x / n

def fn_with_list_comprehension(x):
    """This is fine — static Python that Dynamo can handle."""
    chunks = torch.chunk(x, 4, dim=-1)
    processed = [c.relu() for c in chunks]
    return torch.cat(processed, dim=-1)

x = torch.randn(8, 64)

explanation = torch._dynamo.explain(fn_with_builtin)(x)
print(f"  Shape-based control: {explanation.graph_count} graph segment (OK!)")

explanation = torch._dynamo.explain(fn_with_list_comprehension)(x)
print(f"  List comprehension: {explanation.graph_count} graph segment (Dynamo handles it)")

# =============================================================================
# 6. Using fullgraph=True to catch breaks
# =============================================================================

print("\n" + "=" * 60)
print("--- fullgraph=True: Catching breaks at compile time ---")
print("=" * 60 + "\n")

@torch.compile(fullgraph=True)
def must_be_single_graph(x):
    """This will error if there's any graph break."""
    return x.sin() + x.cos()

# This works fine — no graph breaks
result = must_be_single_graph(torch.randn(8))
print(f"  Clean function with fullgraph=True: works! result shape={result.shape}")

# This would fail:
print("\n  Attempting fullgraph=True with graph break...")
try:
    @torch.compile(fullgraph=True)
    def will_break(x):
        x = x.sin()
        print("debug")  # Graph break!
        return x.cos()

    will_break(torch.randn(8))
except Exception as e:
    error_msg = str(e)[:100]
    print(f"  Caught error: {error_msg}...")
    print(f"  (fullgraph=True surfaces the break as an error)")

# =============================================================================
# 7. Skipping problematic functions with torch._dynamo.disable
# =============================================================================

print("\n" + "=" * 60)
print("--- Selectively disabling compilation ---")
print("=" * 60 + "\n")

@torch._dynamo.disable()
def non_compilable_logging(x):
    """This function opts out of compilation entirely."""
    # Can do anything here — won't cause graph breaks in callers
    val = x.mean().item()
    return val

class ModelWithLogging(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64)

    def forward(self, x):
        x = self.linear(x)
        x = torch.relu(x)
        # The disabled function is treated as opaque
        non_compilable_logging(x)
        return x

model = ModelWithLogging()
compiled = torch.compile(model)
out = compiled(torch.randn(8, 64))
print(f"  Model with @disable'd helper compiles: shape={out.shape}")

# =============================================================================
# 8. Common patterns that DO work (no graph breaks)
# =============================================================================

print("\n" + "=" * 60)
print("--- Patterns that DO work (no graph breaks) ---")
print("=" * 60 + "\n")

def pattern_shape_math(x):
    """Shape-based math is fine — shapes are known at compile time."""
    batch, dim = x.shape
    return x / dim ** 0.5

def pattern_torch_ops(x):
    """All standard torch operations work."""
    return torch.nn.functional.gelu(x) * torch.sigmoid(x)

def pattern_indexing(x):
    """Standard tensor indexing works."""
    first_half = x[:, :x.shape[1] // 2]
    second_half = x[:, x.shape[1] // 2:]
    return first_half + second_half

def pattern_inplace(x):
    """In-place operations generally work."""
    x = x.clone()
    x.relu_()
    x.mul_(2.0)
    return x

patterns = [
    ("Shape-based math", pattern_shape_math),
    ("Standard torch ops", pattern_torch_ops),
    ("Tensor indexing", pattern_indexing),
    ("In-place operations", pattern_inplace),
]

x = torch.randn(8, 64)
for name, fn in patterns:
    torch._dynamo.reset()
    explanation = torch._dynamo.explain(fn)(x)
    status = "OK" if explanation.graph_count == 1 else f"BREAKS ({explanation.graph_count} graphs)"
    print(f"  {name:25s}: {status}")

# =============================================================================
# 9. Summary of graph break causes and fixes
# =============================================================================

print("\n" + "=" * 60)
print("GRAPH BREAK SUMMARY")
print("=" * 60)
print("""
Common Causes & Fixes:

  Cause                    | Fix
  -------------------------+------------------------------------------
  print(tensor)            | Remove or use @torch._dynamo.disable
  tensor.item()            | Keep as tensor (use tensor directly)
  tensor.numpy()           | Keep as tensor
  if tensor.value > x:    | Use torch.where() or torch.cond()
  Custom C extensions      | Register as custom op
  Unsupported builtin      | Rewrite with torch operations
  Global variable mutation | Avoid or use @torch._dynamo.disable

  Debugging tools:
    torch._dynamo.explain(fn)(input)  — shows break reasons
    fullgraph=True                     — errors on any break
    TORCH_LOGS="graph_breaks"         — log breaks at runtime
""")

print("Graph breaks demonstration complete!")
