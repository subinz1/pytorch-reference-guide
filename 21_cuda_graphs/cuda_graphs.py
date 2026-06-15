"""
Module 21: CUDA Graphs — Eliminating CPU Launch Overhead

Demonstrates CUDA Graph capture, replay, static inputs, warmup,
benchmarking, torch.compile reduce-overhead, and make_graphed_callables.

Run: python cuda_graphs.py

If no CUDA GPU is available, the script prints concept explanations
and skips GPU-dependent examples.

Part of the PyTorch Complete Learning Guide.
"""

import torch
import torch.nn as nn
import time
import sys

print("=" * 70)
print("MODULE 21: CUDA GRAPHS — ELIMINATING CPU LAUNCH OVERHEAD")
print("=" * 70)
print(f"\nPyTorch version: {torch.__version__}")

HAS_CUDA = torch.cuda.is_available()

if HAS_CUDA:
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("No CUDA GPU detected — running in explanation-only mode.")
    print("GPU-dependent examples will print descriptions instead of running.\n")

# ============================================================================
# Section 1: What Are CUDA Graphs?
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 1: WHAT ARE CUDA GRAPHS?")
print("=" * 70)

print("""
CUDA Graphs capture a sequence of GPU operations (kernel launches, memory
copies) into a graph structure, then replay the entire graph with a single
CPU launch. This eliminates per-kernel CPU launch overhead.

Normal execution (CPU launches each kernel individually):
  CPU: launch K1, wait, launch K2, wait, launch K3, wait, ...
  GPU: [K1]  gap  [K2]  gap  [K3]  gap  ...

CUDA Graph replay (single CPU launch):
  CPU: launch graph
  GPU: [K1][K2][K3][K4][K5]   (no gaps, back-to-back)

Key insight: The graph captures MEMORY ADDRESSES, not tensor values.
This is why inputs must be pre-allocated "static" tensors.
""")


# ============================================================================
# Section 2: Basic CUDAGraph Capture and Replay
# ============================================================================

print("=" * 70)
print("SECTION 2: BASIC CUDA GRAPH CAPTURE AND REPLAY")
print("=" * 70)

if HAS_CUDA:
    model = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).cuda().eval()

    static_input = torch.randn(64, 512, device="cuda")

    # Warmup (mandatory before capture)
    with torch.no_grad():
        for _ in range(3):
            _ = model(static_input)
    torch.cuda.synchronize()

    # Capture
    g = torch.cuda.CUDAGraph()
    with torch.no_grad():
        with torch.cuda.graph(g):
            static_output = model(static_input)

    print(f"Graph captured successfully.")
    print(f"Static input shape:  {static_input.shape}")
    print(f"Static output shape: {static_output.shape}")

    # Replay with new data
    new_data = torch.randn(64, 512, device="cuda")
    static_input.copy_(new_data)
    g.replay()
    torch.cuda.synchronize()

    # Verify correctness
    with torch.no_grad():
        eager_output = model(new_data)

    max_diff = (static_output - eager_output).abs().max().item()
    print(f"Max diff (graph vs eager): {max_diff:.2e}")
    print(f"Results match: {max_diff < 1e-5}")
else:
    print("""
[Explanation] Basic capture/replay pattern:

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_output = model(static_input)

    # For each new input:
    static_input.copy_(new_data)
    g.replay()
    # static_output now contains the result
""")


# ============================================================================
# Section 3: The Static Inputs Requirement
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 3: THE STATIC INPUTS REQUIREMENT")
print("=" * 70)

print("""
CUDA Graphs capture memory addresses. During replay, the GPU reads from
and writes to those exact addresses. You must:

  1. Pre-allocate input tensors BEFORE capture
  2. copy_() new data INTO them before each replay
  3. Read results from the pre-allocated output tensor

WRONG:  new_tensor = data.cuda()   -> New address each time!
RIGHT:  static_input.copy_(data)   -> Same address, new values
""")

if HAS_CUDA:
    static_in = torch.zeros(32, 256, device="cuda")
    model_small = nn.Linear(256, 64).cuda().eval()

    with torch.no_grad():
        for _ in range(3):
            _ = model_small(static_in)

    g2 = torch.cuda.CUDAGraph()
    with torch.no_grad():
        with torch.cuda.graph(g2):
            static_out = model_small(static_in)

    print("Processing 5 batches through the same graph:")
    for i in range(5):
        batch = torch.randn(32, 256, device="cuda")
        static_in.copy_(batch)
        g2.replay()
        torch.cuda.synchronize()
        print(f"  Batch {i}: output norm = {static_out.norm().item():.4f}")

    del g2, static_in, static_out, model_small


# ============================================================================
# Section 4: Warmup Pattern
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 4: WARMUP — WHY IT'S MANDATORY")
print("=" * 70)

print("""
Before capturing, run the model several times to trigger lazy init:
  - cuDNN algorithm selection
  - CUDA context creation
  - Memory allocator pool building
  - cuBLAS handle creation
  - JIT kernel compilation (for some ops)

Without warmup, capture records these one-time operations, bloating
the graph or causing RuntimeError from dynamic allocations.

Standard warmup pattern:
    with torch.no_grad():
        for _ in range(3):
            _ = model(static_input)
    torch.cuda.synchronize()
""")

if HAS_CUDA:
    conv_model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 10),
    ).cuda().eval()

    static_img = torch.randn(8, 3, 32, 32, device="cuda")

    print("Warmup iterations:")
    with torch.no_grad():
        for i in range(3):
            out = conv_model(static_img)
            torch.cuda.synchronize()
            print(f"  Warmup {i + 1}: output shape {out.shape}")

    g3 = torch.cuda.CUDAGraph()
    with torch.no_grad():
        with torch.cuda.graph(g3):
            static_conv_out = conv_model(static_img)

    print(f"Conv model graph captured. Output shape: {static_conv_out.shape}")
    del g3, conv_model, static_img


# ============================================================================
# Section 5: Benchmarking — Eager vs CUDA Graph Replay
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 5: BENCHMARKING — EAGER vs CUDA GRAPH REPLAY")
print("=" * 70)

if HAS_CUDA:

    class SmallMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 10),
            )

        def forward(self, x):
            return self.layers(x)

    bench_model = SmallMLP().cuda().eval()
    bench_input = torch.randn(16, 256, device="cuda")
    num_iters = 1000

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = bench_model(bench_input)
    torch.cuda.synchronize()

    # Benchmark eager execution
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iters):
            _ = bench_model(bench_input)
    torch.cuda.synchronize()
    eager_time = time.perf_counter() - start

    # Capture graph
    g_bench = torch.cuda.CUDAGraph()
    with torch.no_grad():
        with torch.cuda.graph(g_bench):
            bench_output = bench_model(bench_input)

    # Benchmark graph replay
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        g_bench.replay()
    torch.cuda.synchronize()
    graph_time = time.perf_counter() - start

    print(f"Small MLP (batch=16, {num_iters} iterations):")
    print(f"  Eager:       {eager_time * 1000:.2f} ms  ({eager_time / num_iters * 1e6:.1f} us/iter)")
    print(f"  CUDA Graph:  {graph_time * 1000:.2f} ms  ({graph_time / num_iters * 1e6:.1f} us/iter)")
    speedup = eager_time / graph_time if graph_time > 0 else float("inf")
    print(f"  Speedup:     {speedup:.2f}x")

    del g_bench, bench_model, bench_input
else:
    print("""
[Explanation] Typical benchmark results for a small MLP (batch=16):

  Eager:       ~150 ms for 1000 iters  (~150 us/iter)
  CUDA Graph:  ~30 ms  for 1000 iters  (~30 us/iter)
  Speedup:     ~5x

The smaller the model and batch, the larger the relative speedup,
because CPU launch overhead dominates GPU compute time.
""")


# ============================================================================
# Section 6: torch.compile with reduce-overhead Mode
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 6: torch.compile WITH reduce-overhead MODE")
print("=" * 70)

print("""
The easiest way to use CUDA Graphs — torch.compile handles everything:

    compiled = torch.compile(model, mode="reduce-overhead")
    output = compiled(input_tensor)

'reduce-overhead' uses Inductor's cudagraph_trees internally:
  - Automatic warmup
  - Per-region graph capture (partial graphs OK)
  - Kernel fusion PLUS graph replay combined
  - Shape-variant caching
""")

if HAS_CUDA:

    class DemoModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(512, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 10)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)

    demo_model = DemoModel().cuda().eval()
    demo_input = torch.randn(32, 512, device="cuda")

    compiled_model = torch.compile(demo_model, mode="reduce-overhead")

    # First call triggers compilation + graph capture
    with torch.no_grad():
        out = compiled_model(demo_input)
    torch.cuda.synchronize()
    print(f"torch.compile(mode='reduce-overhead') output shape: {out.shape}")

    # Benchmark compiled model
    with torch.no_grad():
        for _ in range(10):
            _ = compiled_model(demo_input)
    torch.cuda.synchronize()

    num_iters = 500
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iters):
            _ = compiled_model(demo_input)
    torch.cuda.synchronize()
    compile_time = time.perf_counter() - start
    print(f"Compiled (reduce-overhead), {num_iters} iters: {compile_time * 1000:.2f} ms")
    print(f"  Per iteration: {compile_time / num_iters * 1e6:.1f} us")

    del compiled_model, demo_model
else:
    print("""
[Explanation] torch.compile(mode='reduce-overhead') automatically:
  1. Traces the model with Dynamo
  2. Optimizes with AOTAutograd + Inductor (kernel fusion)
  3. Wraps compiled regions in CUDA Graphs via cudagraph_trees
  4. Manages warmup, capture, and replay transparently

Advantages over manual CUDAGraph:
  - Handles input management automatically
  - Supports partial capture (graph breaks fall back to eager)
  - Combines kernel fusion with graph replay
""")


# ============================================================================
# Section 7: make_graphed_callables
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 7: torch.cuda.make_graphed_callables")
print("=" * 70)

print("""
make_graphed_callables wraps an nn.Module or callable with automatic
warmup, capture, and static-input management:

    graphed = torch.cuda.make_graphed_callables(
        model, sample_args=(sample_input,), num_warmup_iters=3
    )
    output = graphed(input_tensor)
""")

if HAS_CUDA:
    mgc_model = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    ).cuda().eval()

    sample = torch.randn(16, 128, device="cuda")

    with torch.no_grad():
        graphed_callable = torch.cuda.make_graphed_callables(
            mgc_model,
            sample_args=(sample,),
            num_warmup_iters=3,
        )

    with torch.no_grad():
        result = graphed_callable(sample)
    print(f"make_graphed_callables output shape: {result.shape}")

    with torch.no_grad():
        eager_result = mgc_model(sample)
    max_diff = (result - eager_result).abs().max().item()
    print(f"Max diff vs eager: {max_diff:.2e}")

    del graphed_callable, mgc_model


# ============================================================================
# Section 8: CUDA Graphs with AMP
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 8: CUDA GRAPHS WITH AMP (MIXED PRECISION)")
print("=" * 70)

print("""
Autocast works inside CUDA Graph capture — the graph records the
mixed-precision kernel variants. At replay time, no autocast context
is needed (dtypes are baked in).

    with torch.cuda.amp.autocast():
        with torch.cuda.graph(g):
            output = model(static_input)

    # Replay — no autocast needed:
    g.replay()
""")

if HAS_CUDA:
    amp_model = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    ).cuda().eval()

    amp_input = torch.randn(32, 512, device="cuda")

    # Warmup with AMP
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for _ in range(3):
            _ = amp_model(amp_input)
    torch.cuda.synchronize()

    # Capture with AMP
    g_amp = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.amp.autocast("cuda"):
        with torch.cuda.graph(g_amp):
            amp_output = amp_model(amp_input)

    print(f"AMP graph output dtype: {amp_output.dtype}")
    print(f"AMP graph output shape: {amp_output.shape}")

    # Replay
    amp_input.copy_(torch.randn(32, 512, device="cuda"))
    g_amp.replay()
    torch.cuda.synchronize()
    print(f"Replay successful. Output norm: {amp_output.norm().item():.4f}")

    del g_amp, amp_model


# ============================================================================
# Section 9: Limitations Demo
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 9: LIMITATIONS — WHAT BREAKS INSIDE CUDA GRAPHS")
print("=" * 70)

print("""
Operations that CANNOT be captured in a CUDA Graph:
  - print(tensor)         -> requires CPU sync
  - tensor.item()         -> transfers to CPU
  - tensor.cpu()          -> cross-device copy
  - torch.tensor([...])   -> CPU tensor creation
  - Dynamic shapes        -> graph hardcodes dimensions
  - Data-dependent flow   -> condition frozen at capture time
  - torch.cuda.synchronize() -> stream synchronization

Operations that silently produce WRONG results:
  - Using non-static tensors as input
  - Data-dependent if/else (branch frozen at capture)
  - Random ops without seed control (same random values each replay)
""")

if HAS_CUDA:
    print("Demonstrating what happens with data-dependent control flow:\n")

    class ConditionalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(64, 64)

        def forward(self, x):
            out = self.linear(x)
            # This condition is FROZEN at capture time
            if out.sum() > 0:
                return out * 2
            else:
                return out * 0.5

    cond_model = ConditionalModel().cuda().eval()
    cond_input = torch.randn(4, 64, device="cuda")

    with torch.no_grad():
        for _ in range(3):
            _ = cond_model(cond_input)

    g_cond = torch.cuda.CUDAGraph()
    with torch.no_grad():
        with torch.cuda.graph(g_cond):
            cond_output = cond_model(cond_input)

    with torch.no_grad():
        eager_result = cond_model(cond_input)
    graph_result_first = cond_output.clone()

    # Change input drastically
    cond_input.copy_(torch.randn(4, 64, device="cuda") * 100)
    g_cond.replay()
    torch.cuda.synchronize()
    graph_result_second = cond_output.clone()

    with torch.no_grad():
        eager_result_second = cond_model(cond_input)

    diff = (graph_result_second - eager_result_second).abs().max().item()
    print(f"  After changing input, graph vs eager max diff: {diff:.4f}")
    if diff > 0.01:
        print("  -> Branch was FROZEN at capture time (expected behavior)")
    else:
        print("  -> Results happened to match (same branch taken)")

    del g_cond, cond_model


# ============================================================================
# Section 10: Graph Pool Sharing
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 10: GRAPH POOL SHARING")
print("=" * 70)

print("""
Multiple graphs can share a memory pool to reduce memory usage.
Use pool sharing when graphs execute sequentially (never concurrently).

    with torch.cuda.graph(g1):
        out1 = model(input1)

    with torch.cuda.graph(g2, pool=g1.pool()):
        out2 = model(input2)
""")

if HAS_CUDA:
    pool_model = nn.Linear(256, 64).cuda().eval()
    input_a = torch.randn(8, 256, device="cuda")
    input_b = torch.randn(16, 256, device="cuda")

    with torch.no_grad():
        for _ in range(3):
            _ = pool_model(input_a)
            _ = pool_model(input_b)

    g_a = torch.cuda.CUDAGraph()
    with torch.no_grad():
        with torch.cuda.graph(g_a):
            out_a = pool_model(input_a)

    g_b = torch.cuda.CUDAGraph()
    with torch.no_grad():
        with torch.cuda.graph(g_b, pool=g_a.pool()):
            out_b = pool_model(input_b)

    g_a.replay()
    torch.cuda.synchronize()
    print(f"Graph A output shape: {out_a.shape}")

    g_b.replay()
    torch.cuda.synchronize()
    print(f"Graph B output shape: {out_b.shape}")
    print("Pool sharing: both graphs share memory (sequential use only)")

    del g_a, g_b, pool_model


# ============================================================================
# Section 11: Inference Server Pattern
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 11: PRACTICAL PATTERN — INFERENCE SERVER")
print("=" * 70)

if HAS_CUDA:

    class GraphedInferenceServer:
        """Wraps a model with CUDA Graph for high-throughput inference."""

        def __init__(self, model, batch_size, input_dim):
            self.model = model.cuda().eval()
            self.static_input = torch.zeros(
                batch_size, input_dim, device="cuda"
            )
            self.graph = torch.cuda.CUDAGraph()

            with torch.no_grad():
                for _ in range(3):
                    _ = self.model(self.static_input)
            torch.cuda.synchronize()

            with torch.no_grad():
                with torch.cuda.graph(self.graph):
                    self.static_output = self.model(self.static_input)

        def predict(self, input_tensor):
            self.static_input.copy_(input_tensor)
            self.graph.replay()
            return self.static_output.clone()

    server_model = nn.Sequential(
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 10),
    )
    server = GraphedInferenceServer(server_model, batch_size=8, input_dim=128)

    print("Simulating inference server with 10 requests:")
    for req_id in range(10):
        fake_request = torch.randn(8, 128, device="cuda")
        result = server.predict(fake_request)
        if req_id < 3 or req_id == 9:
            print(f"  Request {req_id}: prediction shape {result.shape}, "
                  f"argmax = {result.argmax(dim=1).tolist()}")
        elif req_id == 3:
            print("  ...")

    del server
else:
    print("""
[Explanation] Inference server pattern:

    class GraphedInferenceServer:
        def __init__(self, model, batch_size, input_dim):
            # Pre-allocate static input/output on GPU
            # Warmup, then capture graph

        def predict(self, input_tensor):
            self.static_input.copy_(input_tensor)
            self.graph.replay()
            return self.static_output.clone()

This eliminates per-request CPU overhead. For a small model, this
can push throughput from ~10K to ~50K+ predictions per second.
""")


# ============================================================================
# Section 12: Decision Guide
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 12: WHEN TO USE CUDA GRAPHS — DECISION GUIDE")
print("=" * 70)

print("""
Decision tree:

  1. Running on NVIDIA GPU?
     No  -> CUDA Graphs not applicable

  2. Fixed input shapes (or small set of fixed shapes)?
     No  -> Use torch.compile (handles dynamic shapes)

  3. CPU launch overhead is the bottleneck?
     (Small/medium model, many kernels, high throughput needed)
     No  -> torch.compile for kernel fusion is more impactful

  4. Inference only?
     Yes -> CUDA Graphs ideal!
            Use torch.compile(mode="reduce-overhead") or manual API
     No  -> Partial graph capture (forward+backward, optimizer eager)
            or torch.compile(mode="reduce-overhead")

Summary:
  +---------------------------+-------------------------------------------+
  | Scenario                  | Recommendation                            |
  +---------------------------+-------------------------------------------+
  | Inference, fixed shapes   | Manual CUDAGraph or reduce-overhead       |
  | Inference, variable shapes| torch.compile(mode="default")             |
  | Training, single GPU      | torch.compile(mode="reduce-overhead")     |
  | Training, multi-GPU       | torch.compile(mode="default") [NCCL compat]|
  | Data-dependent control    | torch.compile with graph breaks           |
  | Quick experiment          | make_graphed_callables                    |
  +---------------------------+-------------------------------------------+
""")


# ============================================================================
# Summary
# ============================================================================

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Key takeaways:

  1. CUDA Graphs capture GPU operations into a replayable graph,
     eliminating per-kernel CPU launch overhead.

  2. Inputs must be STATIC (pre-allocated). Use .copy_() to load
     new data, then .replay() the graph.

  3. Always WARMUP before capture (3+ forward passes).

  4. torch.compile(mode="reduce-overhead") is the easiest path —
     it handles warmup, capture, pools, and partial graphs.

  5. Limitations: no dynamic shapes, no CPU sync, no data-dependent
     control flow, no dynamic memory allocation inside the graph.

  6. Best for: inference servers with fixed batch sizes and small/
     medium models where CPU overhead dominates.

  7. For training or dynamic shapes, prefer torch.compile with
     default mode instead.
""")

if HAS_CUDA:
    torch.cuda.empty_cache()
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    print(f"GPU memory reserved:  {torch.cuda.memory_reserved() / 1e6:.1f} MB")

print("\nDone! See the README for full documentation.")
