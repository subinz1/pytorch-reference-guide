"""
Module 28 — torch.utils.benchmark Advanced
============================================
torch.compile benchmarking, shape sweeps, Fuzzer, dtype comparison,
model comparison end-to-end.

All examples run on CPU (compile examples use CPU backend).

Usage:
    python benchmark_advanced.py
"""

import torch
import torch.nn as nn
from torch.utils.benchmark import Timer, Compare

print("=" * 70)
print("MODULE 28: torch.utils.benchmark — Advanced")
print("=" * 70)

# ============================================================
# 1. Benchmark torch.compile — eager vs compiled
# ============================================================
print("\n" + "=" * 70)
print("1. torch.compile — eager vs compiled (CPU)")
print("=" * 70)


class SimpleFFN(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        return self.net(x)


dim = 512
batch = 64
model = SimpleFFN(dim)
x = torch.randn(batch, dim)

eager_timer = Timer(
    stmt="model(x)",
    globals={"model": model, "x": x},
    label="FFN forward",
    sub_label=f"dim={dim}, batch={batch}",
    description="eager",
    num_threads=1,
)

compiled_model = torch.compile(model, backend="inductor")
print("Warming up compiled model...")
for _ in range(5):
    compiled_model(x)
print("Warmup complete.")

compiled_timer = Timer(
    stmt="fn(x)",
    globals={"fn": compiled_model, "x": x},
    label="FFN forward",
    sub_label=f"dim={dim}, batch={batch}",
    description="compiled",
    num_threads=1,
)

compile_results = [
    eager_timer.blocked_autorange(min_run_time=1.0),
    compiled_timer.blocked_autorange(min_run_time=1.0),
]

compare = Compare(compile_results)
compare.colorize()
compare.print()

speedup = compile_results[0].median / compile_results[1].median
print(f"Compile speedup: {speedup:.2f}x")

# ============================================================
# 2. Shape sweep — vary matrix size, build comparison table
# ============================================================
print("\n" + "=" * 70)
print("2. Shape sweep — matrix multiply scaling")
print("=" * 70)

sizes = [128, 256, 512, 1024, 2048]
sweep_results = []

for n in sizes:
    a = torch.randn(n, n)
    b = torch.randn(n, n)

    for desc, stmt in [
        ("mm", "a @ b"),
        ("addmm", "torch.addmm(bias, a, b)"),
    ]:
        globs = {"a": a, "b": b, "torch": torch, "bias": torch.randn(n)}
        t = Timer(
            stmt=stmt,
            globals=globs,
            label="Matmul variants",
            sub_label=f"[{n}x{n}]",
            description=desc,
            num_threads=1,
        )
        sweep_results.append(t.blocked_autorange(min_run_time=0.5))

compare = Compare(sweep_results)
compare.trim_significant_figures()
compare.print()

# ============================================================
# 3. Fuzzer — random configurations
# ============================================================
print("\n" + "=" * 70)
print("3. Fuzzer — random tensor configurations")
print("=" * 70)

from torch.utils.benchmark import Fuzzer, FuzzedParameter, FuzzedTensor

fuzzer = Fuzzer(
    parameters=[
        FuzzedParameter("k", minval=4, maxval=10, distribution="uniform"),
        FuzzedParameter("m", minval=4, maxval=10, distribution="uniform"),
        FuzzedParameter("n", minval=4, maxval=10, distribution="uniform"),
    ],
    tensors=[
        FuzzedTensor(
            "x",
            size=("k", "m"),
            probability_contiguous=0.75,
        ),
        FuzzedTensor(
            "y",
            size=("m", "n"),
            probability_contiguous=0.75,
        ),
    ],
    seed=2026,
)

fuzz_results = []
for i, (tensors, tensor_params, params) in enumerate(fuzzer.take(8)):
    k, m, n = int(params["k"]), int(params["m"]), int(params["n"])
    x_contig = tensors["x"].is_contiguous()
    y_contig = tensors["y"].is_contiguous()

    t = Timer(
        stmt="x @ y",
        globals=tensors,
        label="Fuzzed matmul",
        sub_label=f"[{k}x{m}] @ [{m}x{n}] "
        f"(contig: x={'Y' if x_contig else 'N'}, y={'Y' if y_contig else 'N'})",
        description="mm",
        num_threads=1,
    )
    fuzz_results.append(t.blocked_autorange(min_run_time=0.2))

Compare(fuzz_results).print()

# ============================================================
# 4. Benchmarking different dtypes
# ============================================================
print("\n" + "=" * 70)
print("4. Dtype comparison — fp32 vs fp16 vs bf16")
print("=" * 70)

n = 1024
dtype_results = []

for dtype_name, dtype in [
    ("float32", torch.float32),
    ("float16", torch.float16),
    ("bfloat16", torch.bfloat16),
]:
    a = torch.randn(n, n, dtype=dtype)
    b = torch.randn(n, n, dtype=dtype)

    t = Timer(
        stmt="a @ b",
        globals={"a": a, "b": b},
        label="Dtype matmul",
        sub_label=f"[{n}x{n}]",
        description=dtype_name,
        num_threads=1,
    )
    dtype_results.append(t.blocked_autorange(min_run_time=0.5))

compare = Compare(dtype_results)
compare.colorize()
compare.print()

fp32_time = dtype_results[0].median
for i, (name, _) in enumerate(
    [("float32", None), ("float16", None), ("bfloat16", None)]
):
    ratio = fp32_time / dtype_results[i].median
    print(f"  {name}: {ratio:.2f}x vs float32")

# ============================================================
# 5. Benchmark custom functions
# ============================================================
print("\n" + "=" * 70)
print("5. Benchmark custom functions")
print("=" * 70)


def softmax_naive(x):
    e = torch.exp(x - x.max(dim=-1, keepdim=True).values)
    return e / e.sum(dim=-1, keepdim=True)


def softmax_logsumexp(x):
    return torch.exp(x - torch.logsumexp(x, dim=-1, keepdim=True))


x_sm = torch.randn(256, 1024)

custom_results = []
for desc, fn in [
    ("torch.softmax", torch.nn.functional.softmax),
    ("naive", softmax_naive),
    ("logsumexp", softmax_logsumexp),
]:
    if desc == "torch.softmax":
        t = Timer(
            stmt="fn(x, dim=-1)",
            globals={"fn": fn, "x": x_sm},
            label="Softmax",
            sub_label="[256x1024]",
            description=desc,
            num_threads=1,
        )
    else:
        t = Timer(
            stmt="fn(x)",
            globals={"fn": fn, "x": x_sm},
            label="Softmax",
            sub_label="[256x1024]",
            description=desc,
            num_threads=1,
        )
    custom_results.append(t.blocked_autorange(min_run_time=0.5))

compare = Compare(custom_results)
compare.colorize()
compare.print()

# ============================================================
# 6. Full recipe: compare two model implementations
# ============================================================
print("\n" + "=" * 70)
print("6. Recipe — compare two model implementations end-to-end")
print("=" * 70)


class ModelReLU(nn.Module):
    def __init__(self, dim, depth=3):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend([nn.Linear(dim, dim), nn.ReLU()])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ModelGELU(nn.Module):
    def __init__(self, dim, depth=3):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend([nn.Linear(dim, dim), nn.GELU()])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ModelSiLU(nn.Module):
    def __init__(self, dim, depth=3):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend([nn.Linear(dim, dim), nn.SiLU()])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


model_dim = 256
criterion = nn.MSELoss()
models = {
    "ReLU": ModelReLU(model_dim),
    "GELU": ModelGELU(model_dim),
    "SiLU": ModelSiLU(model_dim),
}

e2e_results = []
for batch_size in [16, 64, 256]:
    inp = torch.randn(batch_size, model_dim)
    tgt = torch.randn(batch_size, model_dim)

    for name, mdl in models.items():
        # Forward only
        t_fwd = Timer(
            stmt="model(x)",
            globals={"model": mdl, "x": inp},
            label="Activation comparison",
            sub_label=f"batch={batch_size}",
            description=f"{name} fwd",
            num_threads=1,
        )
        e2e_results.append(t_fwd.blocked_autorange(min_run_time=0.5))

        # Forward + backward
        t_bwd = Timer(
            stmt="""
y = model(x)
loss = criterion(y, target)
loss.backward()
""",
            globals={
                "model": mdl,
                "criterion": criterion,
                "x": inp,
                "target": tgt,
            },
            label="Activation comparison",
            sub_label=f"batch={batch_size}",
            description=f"{name} fwd+bwd",
            num_threads=1,
        )
        e2e_results.append(t_bwd.blocked_autorange(min_run_time=0.5))

compare = Compare(e2e_results)
compare.trim_significant_figures()
compare.colorize()
compare.print()

# ============================================================
# 7. Compile mode comparison
# ============================================================
print("\n" + "=" * 70)
print("7. torch.compile mode comparison")
print("=" * 70)

compile_model = ModelGELU(256)
compile_x = torch.randn(64, 256)

mode_results = []

eager_t = Timer(
    stmt="model(x)",
    globals={"model": compile_model, "x": compile_x},
    label="compile modes",
    sub_label="GELU-FFN [64,256]",
    description="eager",
    num_threads=1,
)
mode_results.append(eager_t.blocked_autorange(min_run_time=1.0))

for mode in ["default", "reduce-overhead"]:
    compiled = torch.compile(compile_model, mode=mode)
    print(f"Warming up mode='{mode}'...")
    for _ in range(5):
        compiled(compile_x)

    t = Timer(
        stmt="fn(x)",
        globals={"fn": compiled, "x": compile_x},
        label="compile modes",
        sub_label="GELU-FFN [64,256]",
        description=mode,
        num_threads=1,
    )
    mode_results.append(t.blocked_autorange(min_run_time=1.0))

compare = Compare(mode_results)
compare.colorize()
compare.print()

eager_median = mode_results[0].median
for i, desc in enumerate(["eager", "default", "reduce-overhead"]):
    ratio = eager_median / mode_results[i].median
    print(f"  {desc}: {ratio:.2f}x vs eager")

# ============================================================
# 8. Comprehensive shape + dtype sweep
# ============================================================
print("\n" + "=" * 70)
print("8. Comprehensive sweep — shapes x dtypes")
print("=" * 70)

comprehensive_results = []
for n in [256, 512, 1024]:
    for dtype_name, dtype in [("fp32", torch.float32), ("bf16", torch.bfloat16)]:
        a = torch.randn(n, n, dtype=dtype)
        b = torch.randn(n, n, dtype=dtype)

        t = Timer(
            stmt="a @ b",
            globals={"a": a, "b": b},
            label="Shape x Dtype",
            sub_label=f"[{n}x{n}]",
            description=dtype_name,
            num_threads=1,
        )
        comprehensive_results.append(t.blocked_autorange(min_run_time=0.5))

compare = Compare(comprehensive_results)
compare.trim_significant_figures()
compare.colorize()
compare.print()

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Advanced benchmarking takeaways:
  1. Always warmup torch.compile before measuring
  2. Shape sweeps reveal scaling behavior and algorithmic complexity
  3. Fuzzer catches performance on non-standard inputs (non-contiguous, odd sizes)
  4. Dtype comparison shows hardware-specific speedups (fp16/bf16 on supported CPUs)
  5. Compare tables with colorize() make winners/losers immediately visible
  6. End-to-end model comparison should include both forward and forward+backward
  7. Different compile modes have different overhead/speedup tradeoffs
""")
