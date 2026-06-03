"""
Debugging Techniques for PyTorch
================================

Practical debugging tools and patterns:
1. Anomaly detection: find the source of NaN/Inf gradients
2. Gradient flow checking: verify gradients propagate through your model
3. Shape debugging: systematic approach to shape mismatches
4. Common errors and fixes
5. Model sanity checks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# 1. Anomaly Detection
# ===========================================================================

def demo_anomaly_detection():
    """detect_anomaly() pinpoints the operation that produces NaN/Inf."""
    print("=" * 60)
    print("ANOMALY DETECTION")
    print("=" * 60)

    # A model with a potential NaN-producing operation
    class ProblematicModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)

        def forward(self, x):
            h = self.linear(x)
            # sqrt of negative numbers produces NaN
            # This is the kind of bug anomaly detection catches
            return torch.sqrt(torch.abs(h))  # fixed: abs() prevents NaN

    model = ProblematicModel()
    x = torch.randn(4, 10)

    # Enable anomaly detection (SLOW — only for debugging)
    with torch.autograd.detect_anomaly():
        output = model(x)
        loss = output.sum()
        loss.backward()

    print("  Anomaly detection ran without errors (model is correct)")
    print("  Note: If a NaN/Inf were produced, you'd see a traceback")
    print("  pointing to the exact operation that caused it.")

    # Demonstrate what triggers anomaly detection
    print("\n  Common NaN sources:")
    print("    - sqrt(negative): use abs() first")
    print("    - log(0): add epsilon, log(x + 1e-8)")
    print("    - 0/0: check for zero denominators")
    print("    - Very large values overflowing: use gradient clipping")


# ===========================================================================
# 2. Gradient Flow Checking
# ===========================================================================

def demo_gradient_flow():
    """Check that gradients flow through all layers of your model."""
    print("\n" + "=" * 60)
    print("GRADIENT FLOW CHECKING")
    print("=" * 60)

    class DeepModel(nn.Module):
        def __init__(self, num_layers=10):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(32, 32) for _ in range(num_layers)
            ])
            self.output = nn.Linear(32, 1)

        def forward(self, x):
            for layer in self.layers:
                x = F.relu(layer(x))
            return self.output(x)

    model = DeepModel(num_layers=10)
    x = torch.randn(8, 32)

    # Forward + backward
    output = model(x)
    loss = output.sum()
    loss.backward()

    # Check gradient statistics for each layer
    print("  Gradient flow through layers:")
    print(f"  {'Layer':<20s} {'Mean':>10s} {'Std':>10s} {'Max':>10s} {'Zero%':>8s}")
    print("  " + "-" * 60)

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            zero_pct = (grad == 0).float().mean().item() * 100
            print(f"  {name:<20s} {grad.mean():>10.6f} {grad.std():>10.6f} "
                  f"{grad.abs().max():>10.6f} {zero_pct:>7.1f}%")
        else:
            print(f"  {name:<20s} {'NO GRADIENT':>30s}")

    # Diagnose potential issues
    print("\n  Diagnosis:")
    all_grads = [p.grad.abs().mean().item() for p in model.parameters() if p.grad is not None]
    if all_grads[-1] / (all_grads[0] + 1e-10) < 0.01:
        print("  WARNING: Possible vanishing gradients (early layers have much smaller grads)")
    elif all_grads[-1] / (all_grads[0] + 1e-10) > 100:
        print("  WARNING: Possible exploding gradients (early layers have much larger grads)")
    else:
        print("  Gradient flow looks healthy")


# ===========================================================================
# 3. Shape Debugging
# ===========================================================================

def demo_shape_debugging():
    """Systematic approach to finding and fixing shape mismatches."""
    print("\n" + "=" * 60)
    print("SHAPE DEBUGGING")
    print("=" * 60)

    # A model where shapes might go wrong
    class ShapeDebugModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(32, 10)

        def forward(self, x, debug=False):
            if debug:
                print(f"    Input:      {list(x.shape)}")

            x = F.relu(self.conv1(x))
            if debug:
                print(f"    After conv1: {list(x.shape)}")

            x = F.max_pool2d(x, 2)
            if debug:
                print(f"    After pool:  {list(x.shape)}")

            x = F.relu(self.conv2(x))
            if debug:
                print(f"    After conv2: {list(x.shape)}")

            x = self.pool(x)
            if debug:
                print(f"    After GAP:   {list(x.shape)}")

            x = x.flatten(1)
            if debug:
                print(f"    After flat:  {list(x.shape)}")

            x = self.fc(x)
            if debug:
                print(f"    Output:      {list(x.shape)}")

            return x

    model = ShapeDebugModel()
    x = torch.randn(2, 3, 32, 32)

    print("  Shape trace through model:")
    with torch.no_grad():
        output = model(x, debug=True)

    # Hook-based shape debugging (non-invasive)
    print("\n  Hook-based shape monitoring:")
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            def make_hook(n):
                def hook(mod, inp, out):
                    print(f"    {n:15s}: {list(inp[0].shape)} -> {list(out.shape)}")
                return hook
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        output = model(x)

    for h in hooks:
        h.remove()


# ===========================================================================
# 4. Common Errors and Fixes
# ===========================================================================

def demo_common_errors():
    """Demonstrate common PyTorch errors and their fixes."""
    print("\n" + "=" * 60)
    print("COMMON ERRORS AND FIXES")
    print("=" * 60)

    # Error 1: In-place operation on a tensor that requires grad
    print("\n  1. In-place operation error:")
    x = torch.randn(3, requires_grad=True)
    # BAD: x += 1  # RuntimeError: in-place operation
    y = x + 1      # GOOD: out-of-place
    print(f"    Fixed: use 'y = x + 1' instead of 'x += 1'")

    # Error 2: Backward through graph twice
    print("\n  2. Double backward error:")
    x = torch.randn(3, requires_grad=True)
    y = x ** 2
    loss = y.sum()
    loss.backward()
    # BAD: loss.backward()  # Error: graph already freed
    # FIX 1: use retain_graph=True (first time)
    # FIX 2: recompute the forward pass
    y2 = x ** 2
    loss2 = y2.sum()
    loss2.backward()
    print(f"    Fixed: recompute forward pass or use retain_graph=True")

    # Error 3: Tensor on wrong device (simulated)
    print("\n  3. Device mismatch:")
    a = torch.randn(3)  # CPU
    b = torch.randn(3)  # CPU (would be .cuda() to trigger error)
    c = a + b
    print(f"    Fix: ensure all tensors are on the same device with .to(device)")

    # Error 4: Forgetting model.eval() / model.train()
    print("\n  4. Forgetting model.eval():")
    model = nn.Sequential(nn.Linear(10, 5), nn.BatchNorm1d(5), nn.Dropout(0.5))
    x = torch.randn(8, 10)

    model.eval()
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    same = torch.equal(out1, out2)
    print(f"    model.eval(): same output for same input = {same}")

    model.train()
    out1 = model(x)
    out2 = model(x)
    same = torch.equal(out1, out2)
    print(f"    model.train(): same output for same input = {same} (dropout is random)")

    # Error 5: Accumulating loss tensors (memory leak)
    print("\n  5. Memory leak from accumulating tensors:")
    model = nn.Linear(10, 1)
    losses_bad = []
    losses_good = []
    for _ in range(5):
        x = torch.randn(4, 10)
        loss = model(x).sum()
        # BAD: losses_bad.append(loss)  # keeps computation graph alive
        losses_good.append(loss.item())  # just stores the float
    print(f"    Fix: use loss.item() to store scalar values, not the tensor")

    # Error 6: Not zeroing gradients
    print("\n  6. Forgetting optimizer.zero_grad():")
    model = nn.Linear(5, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for i in range(3):
        x = torch.randn(4, 5)
        loss = model(x).sum()
        loss.backward()
        grad_norm = model.weight.grad.norm().item()
        if i == 0:
            print(f"    Step {i}: grad_norm = {grad_norm:.4f} (first step)")
        else:
            print(f"    Step {i}: grad_norm = {grad_norm:.4f} "
                  f"(ACCUMULATING — forgot zero_grad!)")

    print(f"    Fix: call optimizer.zero_grad() before loss.backward()")


# ===========================================================================
# 5. Model Sanity Checks
# ===========================================================================

def demo_sanity_checks():
    """Quick checks to verify your model is working correctly."""
    print("\n" + "=" * 60)
    print("MODEL SANITY CHECKS")
    print("=" * 60)

    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 5),
    )

    # Check 1: Can it overfit a tiny batch?
    print("\n  Check 1: Overfit a single batch")
    x = torch.randn(4, 10)
    y = torch.tensor([0, 1, 2, 3])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for step in range(200):
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    accuracy = (model(x).argmax(dim=1) == y).float().mean().item()
    print(f"    Final loss: {final_loss:.6f}")
    print(f"    Accuracy:   {accuracy * 100:.0f}%")
    print(f"    {'PASS' if accuracy == 1.0 else 'FAIL'}: "
          f"{'model can overfit small batch' if accuracy == 1.0 else 'model cannot overfit — check architecture'}")

    # Check 2: Output shape
    print("\n  Check 2: Output shapes")
    model.eval()
    for batch_size in [1, 4, 16]:
        x = torch.randn(batch_size, 10)
        out = model(x)
        expected = (batch_size, 5)
        match = tuple(out.shape) == expected
        print(f"    batch_size={batch_size:2d}: output={list(out.shape)}, "
              f"expected={list(expected)}, {'PASS' if match else 'FAIL'}")

    # Check 3: Parameter count
    print("\n  Check 3: Parameter count")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    print(f"    Total:     {total:,}")
    print(f"    Trainable: {trainable:,}")
    print(f"    Frozen:    {frozen:,}")

    # Check 4: NaN/Inf detection
    print("\n  Check 4: NaN/Inf in parameters")
    has_nan = any(torch.isnan(p).any() for p in model.parameters())
    has_inf = any(torch.isinf(p).any() for p in model.parameters())
    print(f"    NaN in parameters: {has_nan}")
    print(f"    Inf in parameters: {has_inf}")
    print(f"    {'PASS' if not (has_nan or has_inf) else 'FAIL'}")

    # Check 5: Unused parameters (they won't get gradients)
    print("\n  Check 5: Unused parameters")
    model.zero_grad()
    x = torch.randn(4, 10)
    loss = model(x).sum()
    loss.backward()
    unused = []
    for name, param in model.named_parameters():
        if param.grad is None or (param.grad == 0).all():
            unused.append(name)
    if unused:
        print(f"    WARNING: Unused parameters: {unused}")
    else:
        print(f"    All parameters receive gradients — PASS")


if __name__ == "__main__":
    demo_anomaly_detection()
    demo_gradient_flow()
    demo_shape_debugging()
    demo_common_errors()
    demo_sanity_checks()
    print("\n" + "=" * 60)
    print("All debugging demos completed successfully!")
    print("=" * 60)
