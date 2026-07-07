"""
Module 38 — Compiled Backward
==============================

Demonstrates Compiled Autograd: compiling the backward pass alongside the
forward for maximum training throughput.

Runnable on CPU — no GPU required. Timing differences are small on CPU;
the README explains the GPU gains.

Usage:
    python compiled_backward.py
"""

import time
import torch
import torch.nn as nn


# ============================================================
# Part 1: Enable Compiled Autograd
# ============================================================

def demo_compiled_autograd():
    """Show the basic API for enabling compiled autograd."""
    print("=" * 70)
    print("PART 1: Enabling Compiled Autograd")
    print("=" * 70)

    class SmallMLP(nn.Module):
        def __init__(self, d_in, d_hidden, d_out):
            super().__init__()
            self.fc1 = nn.Linear(d_in, d_hidden)
            self.fc2 = nn.Linear(d_hidden, d_hidden)
            self.fc3 = nn.Linear(d_hidden, d_out)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)

    torch._dynamo.config.compiled_autograd = True

    model = torch.compile(SmallMLP(32, 64, 10))
    x = torch.randn(16, 32)
    target = torch.randn(16, 10)

    # Forward + backward — both compiled
    loss = nn.functional.mse_loss(model(x), target)
    loss.backward()

    print(f"Loss: {loss.item():.6f}")
    print(f"fc1.weight.grad norm: {model._orig_mod.fc1.weight.grad.norm().item():.6f}")
    print("Compiled autograd: forward AND backward are compiled.")

    torch._dynamo.config.compiled_autograd = False
    print()


# ============================================================
# Part 2: Training Loop — Eager vs Compiled
# ============================================================

def train_loop(model, optimizer, x_data, y_data, n_steps):
    """Run a training loop and return losses."""
    losses = []
    for step in range(n_steps):
        optimizer.zero_grad()
        pred = model(x_data)
        loss = nn.functional.mse_loss(pred, y_data)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def compare_training():
    """Compare eager vs compiled training — loss should be identical."""
    print("=" * 70)
    print("PART 2: Training Comparison — Eager vs Compiled")
    print("=" * 70)

    torch.manual_seed(42)

    d_in, d_hidden, d_out = 32, 64, 10
    n_steps = 50
    batch_size = 16

    x_data = torch.randn(batch_size, d_in)
    y_data = torch.randn(batch_size, d_out)

    # Eager training
    torch.manual_seed(42)
    model_eager = nn.Sequential(
        nn.Linear(d_in, d_hidden), nn.ReLU(),
        nn.Linear(d_hidden, d_hidden), nn.ReLU(),
        nn.Linear(d_hidden, d_out),
    )
    opt_eager = torch.optim.SGD(model_eager.parameters(), lr=0.01)
    losses_eager = train_loop(model_eager, opt_eager, x_data, y_data, n_steps)

    # Compiled training (with compiled autograd)
    torch.manual_seed(42)
    model_compiled = nn.Sequential(
        nn.Linear(d_in, d_hidden), nn.ReLU(),
        nn.Linear(d_hidden, d_hidden), nn.ReLU(),
        nn.Linear(d_hidden, d_out),
    )
    model_compiled = torch.compile(model_compiled)

    torch._dynamo.config.compiled_autograd = True
    opt_compiled = torch.optim.SGD(model_compiled.parameters(), lr=0.01)
    losses_compiled = train_loop(model_compiled, opt_compiled, x_data, y_data, n_steps)
    torch._dynamo.config.compiled_autograd = False

    # Compare
    print(f"\n  {'Step':<8} {'Eager Loss':>12} {'Compiled Loss':>14} {'Diff':>12}")
    print(f"  {'─' * 8} {'─' * 12} {'─' * 14} {'─' * 12}")
    for i in range(0, n_steps, 10):
        diff = abs(losses_eager[i] - losses_compiled[i])
        print(f"  {i:<8} {losses_eager[i]:>12.6f} {losses_compiled[i]:>14.6f} {diff:>12.2e}")

    max_diff = max(abs(e - c) for e, c in zip(losses_eager, losses_compiled))
    print(f"\n  Max loss difference: {max_diff:.2e}")
    print(f"  Final eager loss:    {losses_eager[-1]:.6f}")
    print(f"  Final compiled loss: {losses_compiled[-1]:.6f}")
    print(f"  Losses match: {max_diff < 1e-4}")
    print()


# ============================================================
# Part 3: Timing Comparison
# ============================================================

def timing_comparison():
    """Time eager vs compiled training."""
    print("=" * 70)
    print("PART 3: Timing — Eager vs Compiled (CPU)")
    print("=" * 70)

    torch.manual_seed(0)
    d_in, d_hidden, d_out = 128, 256, 64
    batch_size = 64
    n_warmup = 10
    n_steps = 100

    x_data = torch.randn(batch_size, d_in)
    y_data = torch.randn(batch_size, d_out)

    def make_model():
        torch.manual_seed(0)
        return nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )

    # Eager timing
    model_e = make_model()
    opt_e = torch.optim.Adam(model_e.parameters())
    for _ in range(n_warmup):
        opt_e.zero_grad()
        nn.functional.mse_loss(model_e(x_data), y_data).backward()
        opt_e.step()

    t0 = time.perf_counter()
    for _ in range(n_steps):
        opt_e.zero_grad()
        nn.functional.mse_loss(model_e(x_data), y_data).backward()
        opt_e.step()
    eager_time = time.perf_counter() - t0

    # Compiled timing
    model_c = torch.compile(make_model())
    torch._dynamo.config.compiled_autograd = True
    opt_c = torch.optim.Adam(model_c.parameters())
    for _ in range(n_warmup):
        opt_c.zero_grad()
        nn.functional.mse_loss(model_c(x_data), y_data).backward()
        opt_c.step()

    t0 = time.perf_counter()
    for _ in range(n_steps):
        opt_c.zero_grad()
        nn.functional.mse_loss(model_c(x_data), y_data).backward()
        opt_c.step()
    compiled_time = time.perf_counter() - t0
    torch._dynamo.config.compiled_autograd = False

    print(f"\n  Eager:    {eager_time:.3f}s ({n_steps} steps)")
    print(f"  Compiled: {compiled_time:.3f}s ({n_steps} steps)")
    ratio = eager_time / compiled_time if compiled_time > 0 else float("inf")
    print(f"  Speedup:  {ratio:.2f}x")
    print("""
  Note: On CPU, the speedup from compiled autograd is modest because
  CPU kernel launch overhead is already low. On GPU, the gains are
  significant (1.2-1.5x for typical models) because GPU kernel launches
  have higher overhead, and Inductor's fusion eliminates many of them.
""")


# ============================================================
# Part 4: Backward Ops Are Fused
# ============================================================

def show_backward_fusion():
    """Demonstrate that compiled autograd fuses backward operations."""
    print("=" * 70)
    print("PART 4: Backward Ops Are Fused")
    print("=" * 70)

    from torch._functorch.aot_autograd import aot_function

    bw_op_count = {}

    def counting_bw_compiler(gm, example_inputs):
        call_nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
        bw_op_count["total"] = len(call_nodes)

        op_types = {}
        for n in call_nodes:
            name = n.target.__name__ if hasattr(n.target, "__name__") else str(n.target)
            short_name = name.split(".")[-1]
            op_types[short_name] = op_types.get(short_name, 0) + 1
        bw_op_count["breakdown"] = op_types
        return gm

    def model_fn(x, w1, b1, w2, b2):
        h = torch.relu(x @ w1 + b1)
        h = torch.sigmoid(h @ w2 + b2)
        return h.sum()

    compiled = aot_function(
        model_fn,
        fw_compiler=lambda gm, _: gm,
        bw_compiler=counting_bw_compiler,
    )

    x = torch.randn(8, 16, requires_grad=True)
    w1 = torch.randn(16, 12, requires_grad=True)
    b1 = torch.randn(12, requires_grad=True)
    w2 = torch.randn(12, 8, requires_grad=True)
    b2 = torch.randn(8, requires_grad=True)

    out = compiled(x, w1, b1, w2, b2)
    out.backward()

    print(f"\n  Backward graph total ops: {bw_op_count.get('total', 'N/A')}")
    if "breakdown" in bw_op_count:
        print("  Op breakdown:")
        for op, count in sorted(bw_op_count["breakdown"].items()):
            print(f"    {op}: {count}")
    print("""
  In eager mode, each of these ops launches as a separate kernel.
  With Inductor, element-wise backward ops (sigmoid_backward,
  threshold_backward, add, mul) are fused into fewer kernels.
  Matmul backward ops (mm) remain separate (already efficient).
""")


# ============================================================
# Part 5: CompileCounter — Verify Backward is Compiled
# ============================================================

def verify_backward_compiled():
    """Use CompileCounter to verify the backward pass is being compiled."""
    print("=" * 70)
    print("PART 5: CompileCounter — Verify Backward is Compiled")
    print("=" * 70)

    compile_count = {"forward": 0, "backward": 0}

    def counting_backend(gm, example_inputs):
        return gm

    model = nn.Sequential(
        nn.Linear(16, 32), nn.ReLU(),
        nn.Linear(32, 8),
    )

    # With torch.compile, we can check _dynamo frame count
    torch._dynamo.reset()
    compiled_model = torch.compile(model, backend="aot_eager")

    x = torch.randn(4, 16)
    target = torch.randn(4, 8)

    # Run one step to trigger compilation
    loss = nn.functional.mse_loss(compiled_model(x), target)
    loss.backward()

    # Check compile metrics
    metrics = torch._dynamo.utils.counters
    frame_count = metrics.get("frames", {}).get("ok", 0)

    print(f"\n  Dynamo frame count: {frame_count}")
    print(f"  (Each frame = one compiled graph region)")

    # With compiled autograd
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    torch._dynamo.config.compiled_autograd = True

    compiled_model2 = torch.compile(
        nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8)),
        backend="aot_eager",
    )

    loss = nn.functional.mse_loss(compiled_model2(x), target)
    loss.backward()

    frame_count_ca = torch._dynamo.utils.counters.get("frames", {}).get("ok", 0)
    torch._dynamo.config.compiled_autograd = False

    print(f"  With compiled autograd frame count: {frame_count_ca}")
    print(f"  (Higher frame count = backward is also compiled)")
    print()


# ============================================================
# Part 6: Disabling and Debugging Compiled Autograd
# ============================================================

def disable_and_debug():
    """Show how to disable and debug compiled autograd."""
    print("=" * 70)
    print("PART 6: Disabling & Debugging Compiled Autograd")
    print("=" * 70)

    print("""
  Enabling:
    torch._dynamo.config.compiled_autograd = True

  Disabling:
    torch._dynamo.config.compiled_autograd = False

  Debugging logs:
    TORCH_LOGS="compiled_autograd" python script.py
    TORCH_LOGS="compiled_autograd_verbose" python script.py

  Programmatic:
    import logging
    torch._logging.set_logs(compiled_autograd=logging.DEBUG)
""")

    # Demonstrate enable/disable cycle
    model = torch.compile(nn.Linear(8, 4))
    x = torch.randn(2, 8)

    torch._dynamo.config.compiled_autograd = True
    loss = model(x).sum()
    loss.backward()
    print("  With compiled autograd: backward completed ✓")

    torch._dynamo.config.compiled_autograd = False
    torch._dynamo.reset()
    model2 = torch.compile(nn.Linear(8, 4))
    loss = model2(x).sum()
    loss.backward()
    print("  Without compiled autograd: backward completed ✓")
    print()


# ============================================================
# Part 7: Common Issues and Fixes
# ============================================================

def common_issues():
    """Document common issues with compiled autograd and their fixes."""
    print("=" * 70)
    print("PART 7: Common Issues and Fixes")
    print("=" * 70)

    print("""
  Issue 1: "graph break in backward"
  -----------------------------------
  Cause: Unsupported operation in the backward pass (e.g., custom
         autograd.Function without compiled autograd support).
  Fix:   Implement the function using standard ops, or add
         setup_context / backward classmethod for compiled autograd.

  Issue 2: "recompilation on every step"
  --------------------------------------
  Cause: Dynamic shapes or changing graph structure between iterations.
  Fix:   Use mark_dynamic() on tensors with varying shapes, or ensure
         batch size and sequence length are consistent.

  Issue 3: "compiled autograd doesn't help on CPU"
  ------------------------------------------------
  Cause: CPU kernel launch overhead is already low; fusion gains are
         smaller compared to GPU.
  Fix:   This is expected. The main benefit is on GPU.

  Issue 4: "hooks not called"
  ---------------------------
  Cause: Some autograd hooks may be skipped or behave differently
         under compiled autograd.
  Fix:   Check hook compatibility. Use torch.autograd.graph hooks
         rather than tensor-level hooks when possible.

  Issue 5: "NaN gradients with compiled autograd"
  -----------------------------------------------
  Cause: Numerical differences from op fusion or recomputation order.
  Fix:   Compare with eager. If differences exist, check for
         operations sensitive to evaluation order (e.g., reductions).
         Use torch.autograd.set_detect_anomaly(True) for debugging.
""")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print()
    print("Module 38: Compiled Backward")
    print("============================")
    print("Compiling the backward pass for maximum training throughput.\n")

    demo_compiled_autograd()
    compare_training()
    timing_comparison()
    show_backward_fusion()
    verify_backward_compiled()
    disable_and_debug()
    common_issues()

    print("=" * 70)
    print("Done! Compiled Autograd brings the backward pass into the")
    print("compilation pipeline, enabling fusion and optimization for")
    print("the entire training step — not just the forward pass.")
    print("=" * 70)
