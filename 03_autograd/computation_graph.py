"""
Module 03: Computation Graph
==============================
Exploring PyTorch's dynamic computation graph: how it's built, traversed,
and destroyed during forward and backward passes.

Run: python computation_graph.py
"""

import torch

print("=" * 70)
print("PART 1: GRAPH CONSTRUCTION")
print("=" * 70)

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# Each operation creates a node in the computation graph
a = x * y        # MulBackward node
b = a + x        # AddBackward node
c = b ** 2       # PowBackward node

print("Computation: c = (x*y + x)^2")
print(f"x = {x.item()}, y = {y.item()}")
print(f"a = x*y = {a.item()}")
print(f"b = a+x = {b.item()}")
print(f"c = b^2 = {c.item()}")

print("\n--- Graph structure (via grad_fn) ---")
print(f"x.grad_fn = {x.grad_fn}  (None: leaf tensor)")
print(f"y.grad_fn = {y.grad_fn}  (None: leaf tensor)")
print(f"a.grad_fn = {a.grad_fn}")
print(f"b.grad_fn = {b.grad_fn}")
print(f"c.grad_fn = {c.grad_fn}")

print("\n--- Traversing the graph backward ---")
node = c.grad_fn
print(f"c created by: {node.name()}")
for i, (fn, idx) in enumerate(node.next_functions):
    if fn is not None:
        print(f"  input {i}: {fn.name()}")
        for j, (fn2, idx2) in enumerate(fn.next_functions):
            if fn2 is not None:
                print(f"    input {j}: {fn2.name()}")


print("\n" + "=" * 70)
print("PART 2: LEAF vs INTERMEDIATE TENSORS")
print("=" * 70)

x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = x * y + x ** 2

print(f"x: is_leaf={x.is_leaf}, requires_grad={x.requires_grad}, grad_fn={x.grad_fn}")
print(f"y: is_leaf={y.is_leaf}, requires_grad={y.requires_grad}, grad_fn={y.grad_fn}")
print(f"z: is_leaf={z.is_leaf}, requires_grad={z.requires_grad}, grad_fn={z.grad_fn}")

z.backward()
print(f"\nAfter backward:")
print(f"x.grad = {x.grad}  (leaf — gradient stored)")
print(f"y.grad = {y.grad}  (leaf — gradient stored)")
print(f"z.grad = {z.grad}  (intermediate — gradient NOT stored by default)")

# Using retain_grad to keep intermediate gradients
print("\n--- retain_grad() ---")
x = torch.tensor(1.0, requires_grad=True)
y = x ** 2
y.retain_grad()
z = y * 3
z.backward()
print(f"y.grad (with retain_grad): {y.grad}")

# Tensors without requires_grad don't participate in the graph
data = torch.randn(3)
w = torch.randn(3, requires_grad=True)
out = (data * w).sum()
print(f"\ndata.requires_grad: {data.requires_grad}")
print(f"w.requires_grad: {w.requires_grad}")
print(f"out.requires_grad: {out.requires_grad}")
print("(If ANY input requires grad, the output does too)")


print("\n" + "=" * 70)
print("PART 3: GRAPH LIFECYCLE — CREATION AND DESTRUCTION")
print("=" * 70)

x = torch.tensor(2.0, requires_grad=True)

print("Step 1: Forward pass (graph is built)")
y = x ** 2
print(f"  y.grad_fn: {y.grad_fn}")

print("\nStep 2: backward() (graph is traversed and destroyed)")
y.backward()
print(f"  x.grad: {x.grad}")

print("\nStep 3: Trying backward again...")
try:
    y = x ** 2  # Need new forward pass
    y.backward()
    print(f"  x.grad: {x.grad}  (accumulated! old 4.0 + new 4.0 = 8.0)")
except RuntimeError as e:
    print(f"  ERROR: {e}")

print("\n--- retain_graph=True ---")
x.grad.zero_()
y = x ** 2
y.backward(retain_graph=True)
print(f"First backward: x.grad = {x.grad}")
x.grad.zero_()
y.backward()
print(f"Second backward (retained graph): x.grad = {x.grad}")


print("\n" + "=" * 70)
print("PART 4: DYNAMIC GRAPHS — DIFFERENT PATHS EACH FORWARD")
print("=" * 70)

def dynamic_function(x, use_square=True):
    """Different computation graph depending on input."""
    if use_square:
        return x ** 2
    else:
        return x ** 3

x = torch.tensor(3.0, requires_grad=True)

# Path 1: f(x) = x^2
y = dynamic_function(x, use_square=True)
y.backward()
print(f"x^2 path: dy/dx at x=3 is {x.grad.item()} (expected 6)")
x.grad.zero_()

# Path 2: f(x) = x^3
y = dynamic_function(x, use_square=False)
y.backward()
print(f"x^3 path: dy/dx at x=3 is {x.grad.item()} (expected 27)")

# Data-dependent control flow
print("\n--- Data-dependent control flow ---")
def conditional_function(x):
    if x.sum() > 0:
        return x * 2
    else:
        return x * 3

for val in [1.0, -1.0]:
    x = torch.tensor(val, requires_grad=True)
    y = conditional_function(x)
    y.backward()
    print(f"x={val}: path={'x*2' if val > 0 else 'x*3'}, dy/dx={x.grad.item()}")

# Loop-dependent computation
print("\n--- Variable-length computation ---")
x = torch.tensor(0.5, requires_grad=True)
result = x
iterations = 0
while result.abs() < 100:
    result = result * x + 1
    iterations += 1
result.backward()
print(f"After {iterations} iterations: result={result.item():.4f}, dx={x.grad.item():.4f}")


print("\n" + "=" * 70)
print("PART 5: GRADIENT FLOW THROUGH OPERATIONS")
print("=" * 70)

# Demonstrating which operations preserve gradient flow
x = torch.tensor(2.0, requires_grad=True)

operations = [
    ("x + 1", lambda x: x + 1),
    ("x * 2", lambda x: x * 2),
    ("x ** 2", lambda x: x ** 2),
    ("sin(x)", lambda x: torch.sin(x)),
    ("exp(x)", lambda x: torch.exp(x)),
    ("relu(x)", lambda x: torch.relu(x)),
    ("sigmoid(x)", lambda x: torch.sigmoid(x)),
    ("clamp(x, 0, inf)", lambda x: torch.clamp(x, min=0)),
    ("abs(x)", lambda x: torch.abs(x)),
]

print(f"x = {x.item()}")
print(f"{'Operation':<25} {'Output':>10} {'Gradient':>10}")
print("-" * 48)

for name, op in operations:
    x_copy = torch.tensor(2.0, requires_grad=True)
    y = op(x_copy)
    y.backward()
    print(f"{name:<25} {y.item():10.4f} {x_copy.grad.item():10.4f}")


print("\n" + "=" * 70)
print("PART 6: OPERATIONS THAT BREAK GRADIENT FLOW")
print("=" * 70)

x = torch.tensor(2.5, requires_grad=True)

# .item() extracts a Python scalar — no gradient
val = x.item()
print(f".item() returns Python float: {val}, type: {type(val)}")
print(f"  Cannot compute gradients through .item()")

# Integer casting loses gradients
y = x.int()
print(f"\n.int() → {y}, requires_grad: {y.requires_grad}")
print(f"  Integer types don't support gradients")

# .data bypasses autograd (dangerous!)
z = x.data
print(f"\n.data → {z}, requires_grad: {z.requires_grad}")
print(f"  .data gives raw tensor without autograd — avoid using it!")
print(f"  Use .detach() instead for safety")

# Comparison operations return booleans
mask = x > 2
print(f"\nx > 2 → {mask}, requires_grad: {mask.requires_grad}")
print(f"  Boolean tensors don't have gradients")


print("\n" + "=" * 70)
print("PART 7: MULTIPLE BACKWARD PASSES")
print("=" * 70)

x = torch.tensor(1.0, requires_grad=True)
w1 = torch.tensor(2.0, requires_grad=True)
w2 = torch.tensor(3.0, requires_grad=True)

# Two losses that share the same input
loss1 = (x * w1) ** 2
loss2 = (x * w2) ** 2

print("Two losses sharing input x:")
print(f"  loss1 = (x*w1)^2 = {loss1.item()}")
print(f"  loss2 = (x*w2)^2 = {loss2.item()}")

# Must retain_graph for first backward since graph is shared
loss1.backward(retain_graph=True)
print(f"\nAfter loss1.backward():")
print(f"  x.grad = {x.grad.item()}")
print(f"  w1.grad = {w1.grad.item()}")

loss2.backward()
print(f"\nAfter loss2.backward() (gradients accumulated):")
print(f"  x.grad = {x.grad.item()} (sum of both gradients)")
print(f"  w2.grad = {w2.grad.item()}")

print("\nThis is how multi-task learning works:")
print("  total_grad = grad_from_task1 + grad_from_task2")


print("\n" + "=" * 70)
print("PART 8: GRADIENT ACCUMULATION FOR LARGE BATCHES")
print("=" * 70)

torch.manual_seed(42)
model_param = torch.randn(5, requires_grad=True)
data = torch.randn(20, 5)  # 20 samples
labels = torch.randn(20)

# Simulating gradient accumulation for effective batch size = 20
# using 4 mini-batches of 5
accumulation_steps = 4
mini_batch_size = 5

# Reset gradient
if model_param.grad is not None:
    model_param.grad.zero_()

for i in range(accumulation_steps):
    start = i * mini_batch_size
    end = start + mini_batch_size
    mini_data = data[start:end]
    mini_labels = labels[start:end]

    pred = mini_data @ model_param
    loss = ((pred - mini_labels) ** 2).mean() / accumulation_steps
    loss.backward()
    print(f"  Mini-batch {i}: loss={loss.item():.4f}, grad norm={model_param.grad.norm().item():.4f}")

print(f"\nFinal accumulated gradient norm: {model_param.grad.norm().item():.4f}")
print("(Equivalent to using full batch of 20 samples)")

# Compare with full batch
model_param2 = model_param.detach().clone().requires_grad_(True)
pred_full = data @ model_param2
loss_full = ((pred_full - labels) ** 2).mean()
loss_full.backward()
print(f"Full batch gradient norm:        {model_param2.grad.norm().item():.4f}")
print(f"Match: {torch.allclose(model_param.grad, model_param2.grad, atol=1e-5)}")


print("\n" + "=" * 70)
print("PART 9: GRAPH INSPECTION UTILITY")
print("=" * 70)

def trace_graph(tensor, depth=0):
    """Recursively trace the computation graph."""
    indent = "  " * depth
    if tensor.grad_fn is None:
        name = "leaf" if tensor.requires_grad else "constant"
        print(f"{indent}[{name}] shape={tensor.shape}")
        return

    print(f"{indent}{tensor.grad_fn.name()} → shape={tensor.shape}")
    for fn, _ in tensor.grad_fn.next_functions:
        if fn is not None:
            # Create a dummy tensor to trace
            dummy = torch.tensor(0.0)
            dummy.grad_fn = fn  # This doesn't actually work, but we can still traverse
            pass

# Simple trace
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = torch.sin(x * y) + torch.exp(x)

print("Expression: z = sin(x * y) + exp(x)")
print(f"z.grad_fn: {z.grad_fn}")
print("\nGraph traversal:")

node = z.grad_fn
print(f"  {node.name()}")
for fn, _ in node.next_functions:
    if fn:
        print(f"    ├── {fn.name()}")
        for fn2, _ in fn.next_functions:
            if fn2:
                print(f"    │   ├── {fn2.name()}")
                for fn3, _ in fn2.next_functions:
                    if fn3:
                        print(f"    │   │   └── {fn3.name()}")
                    else:
                        print(f"    │   │   └── [leaf tensor]")

print("\n" + "=" * 70)
print("Computation graph demonstration complete!")
print("=" * 70)
