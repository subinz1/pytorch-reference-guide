"""
Module 20: torch.backends — Performance Tuning
================================================
Demonstrates all major backend settings for performance optimization.
All examples run on CPU; GPU-specific features print explanatory text.

Usage:
    python backends_tuning.py
"""

import time
import torch
import torch.nn as nn


def print_section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# ==========================================================================
# Section 1: Print All Backend Settings
# ==========================================================================

def print_all_backend_settings():
    """Print a comprehensive summary of all backend configuration."""
    print_section("All Backend Settings")

    print("--- cuDNN ---")
    print(f"  cudnn.enabled:        {torch.backends.cudnn.enabled}")
    print(f"  cudnn.benchmark:      {torch.backends.cudnn.benchmark}")
    print(f"  cudnn.deterministic:  {torch.backends.cudnn.deterministic}")
    print(f"  cudnn.allow_tf32:     {torch.backends.cudnn.allow_tf32}")
    print(f"  cudnn.is_available(): {torch.backends.cudnn.is_available()}")
    print()

    print("--- CUDA matmul ---")
    print(f"  cuda.matmul.allow_tf32:                            {torch.backends.cuda.matmul.allow_tf32}")
    print(f"  cuda.matmul.allow_fp16_reduced_precision_reduction: {torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction}")
    print(f"  cuda.matmul.allow_bf16_reduced_precision_reduction: {torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}")
    print()

    print("--- CUDA SDPA ---")
    print(f"  cuda.flash_sdp_enabled():         {torch.backends.cuda.flash_sdp_enabled()}")
    print(f"  cuda.mem_efficient_sdp_enabled():  {torch.backends.cuda.mem_efficient_sdp_enabled()}")
    print(f"  cuda.math_sdp_enabled():           {torch.backends.cuda.math_sdp_enabled()}")
    print()

    print("--- MKL-DNN (oneDNN) ---")
    print(f"  mkldnn.enabled:        {torch.backends.mkldnn.enabled}")
    print(f"  mkldnn.is_available(): {torch.backends.mkldnn.is_available()}")
    print()

    print("--- OpenMP ---")
    print(f"  num_threads: {torch.get_num_threads()}")
    print()

    print("--- opt_einsum ---")
    print(f"  opt_einsum.enabled:  {torch.backends.opt_einsum.enabled}")
    if torch.backends.opt_einsum.enabled:
        print(f"  opt_einsum.strategy: {torch.backends.opt_einsum.strategy}")
    print()

    print("--- MPS (Apple Metal) ---")
    print(f"  mps.is_available(): {torch.backends.mps.is_available()}")
    print(f"  mps.is_built():     {torch.backends.mps.is_built()}")
    print()

    print(f"--- float32_matmul_precision ---")
    print(f"  Current: {torch.get_float32_matmul_precision()}")


# ==========================================================================
# Section 2: cuDNN Benchmark Mode
# ==========================================================================

def demo_cudnn_benchmark():
    """Demonstrate cuDNN benchmark mode behavior."""
    print_section("cuDNN Benchmark Mode")

    if not torch.cuda.is_available():
        print("GPU not available. Explaining cuDNN benchmark behavior:")
        print()
        print("  torch.backends.cudnn.benchmark = True")
        print()
        print("  When enabled, the first forward pass at each input shape triggers")
        print("  algorithm auto-tuning. cuDNN tries multiple convolution algorithms")
        print("  (e.g., Winograd, FFT, implicit GEMM) and caches the fastest one.")
        print()
        print("  Example timing pattern (hypothetical GPU):")
        print("    Benchmark=False: each forward pass ~5ms (consistent)")
        print("    Benchmark=True:  first pass ~50ms (tuning), subsequent ~3ms (faster)")
        print()
        print("  Best for: fixed input sizes (CNNs with constant batch/image size)")
        print("  Avoid for: variable sizes (NLP, detection with varying dimensions)")
        return

    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
    ).cuda()

    x = torch.randn(32, 3, 224, 224, device="cuda")

    # Without benchmark
    torch.backends.cudnn.benchmark = False
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        _ = model(x)
    torch.cuda.synchronize()
    time_no_bench = time.perf_counter() - t0

    # With benchmark
    torch.backends.cudnn.benchmark = True
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        _ = model(x)
    torch.cuda.synchronize()
    time_with_bench = time.perf_counter() - t0

    print(f"  Without benchmark: {time_no_bench*1000:.1f}ms (10 iterations)")
    print(f"  With benchmark:    {time_with_bench*1000:.1f}ms (10 iterations)")
    print(f"  Speedup:           {time_no_bench/time_with_bench:.2f}x")


# ==========================================================================
# Section 3: TF32 Precision
# ==========================================================================

def demo_tf32_precision():
    """Demonstrate TF32 precision toggle and its precision impact."""
    print_section("TF32 Precision")

    if not torch.cuda.is_available():
        print("GPU not available. Explaining TF32 precision:")
        print()
        print("  TF32 is a 19-bit format: 1 sign + 8 exponent + 10 mantissa")
        print("  Used internally by Ampere+ tensor cores for FP32 operations.")
        print()
        print("  Toggle for convolutions:")
        print("    torch.backends.cudnn.allow_tf32 = True/False")
        print()
        print("  Toggle for matrix multiplication:")
        print("    torch.backends.cuda.matmul.allow_tf32 = True/False")
        print()
        print("  Precision comparison (typical):")
        print("    FP32 matmul result: 1.0000000")
        print("    TF32 matmul result: 0.9999847  (relative error ~1.5e-5)")
        print()
        print("  Speedup on A100: ~2-3x for matmul, ~2x for convolutions")
        return

    A = torch.randn(1024, 1024, device="cuda")
    B = torch.randn(1024, 1024, device="cuda")

    # Full precision
    torch.backends.cuda.matmul.allow_tf32 = False
    result_fp32 = torch.mm(A, B)

    # TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    result_tf32 = torch.mm(A, B)

    diff = (result_fp32 - result_tf32).abs()
    print(f"  Max absolute difference:  {diff.max().item():.6e}")
    print(f"  Mean absolute difference: {diff.mean().item():.6e}")
    rel_err = diff / result_fp32.abs().clamp(min=1e-8)
    print(f"  Mean relative error:      {rel_err.mean().item():.6e}")


# ==========================================================================
# Section 4: torch.set_float32_matmul_precision
# ==========================================================================

def demo_matmul_precision_levels():
    """Demonstrate the three matmul precision levels."""
    print_section("torch.set_float32_matmul_precision()")

    levels = ["highest", "high", "medium"]

    for level in levels:
        torch.set_float32_matmul_precision(level)
        current = torch.get_float32_matmul_precision()
        print(f"  Level '{level}': set successfully (get returns '{current}')")

    print()
    print("  Level explanations:")
    print("    'highest' — Full FP32, no TF32. Maximum precision.")
    print("    'high'    — TF32 on Ampere+ GPUs. Good speed/precision balance.")
    print("    'medium'  — TF32 + reduced precision accumulation. Maximum speed.")
    print()
    print("  Recommended: 'high' for training, 'highest' for debugging")

    # Reset to high
    torch.set_float32_matmul_precision("high")


# ==========================================================================
# Section 5: OpenMP Thread Configuration
# ==========================================================================

def demo_openmp_threads():
    """Demonstrate OpenMP thread count configuration and its effect."""
    print_section("OpenMP Thread Configuration")

    original_threads = torch.get_num_threads()
    print(f"  Default thread count: {original_threads}")
    print()

    # Benchmark matmul with different thread counts
    A = torch.randn(2048, 2048)
    B = torch.randn(2048, 2048)

    thread_counts = [1, 2, 4, max(original_threads, 4)]
    # Remove duplicates while preserving order
    thread_counts = list(dict.fromkeys(thread_counts))

    print("  Timing CPU matmul (2048x2048) with different thread counts:")
    for n_threads in thread_counts:
        torch.set_num_threads(n_threads)

        # Warmup
        _ = torch.mm(A, B)

        t0 = time.perf_counter()
        for _ in range(5):
            _ = torch.mm(A, B)
        elapsed = time.perf_counter() - t0

        print(f"    {n_threads} thread(s): {elapsed*1000:.1f}ms (5 iterations)")

    # Restore
    torch.set_num_threads(original_threads)
    print(f"\n  Restored to {original_threads} threads")
    print()
    print("  Tips:")
    print("    - Set to physical cores (not hyperthreads) for best throughput")
    print("    - For serving: reduce threads to allow concurrent requests")
    print("    - Environment var: OMP_NUM_THREADS (set before importing torch)")


# ==========================================================================
# Section 6: opt_einsum Strategy
# ==========================================================================

def demo_opt_einsum():
    """Demonstrate opt_einsum optimization strategies."""
    print_section("opt_einsum Strategy")

    print(f"  opt_einsum available: {torch.backends.opt_einsum.enabled}")

    if not torch.backends.opt_einsum.enabled:
        print("  (opt_einsum not installed — pip install opt-einsum)")
        print("  Without it, torch.einsum uses a naive contraction order.")
        return

    # Multi-tensor contraction where order matters
    A = torch.randn(100, 50)
    B = torch.randn(50, 80)
    C = torch.randn(80, 100)
    D = torch.randn(100, 60)

    # Test different strategies
    strategies = ["auto", "greedy", "optimal"]
    print()
    print("  Timing einsum 'ij,jk,kl,lm->im' with different strategies:")

    for strategy in strategies:
        torch.backends.opt_einsum.strategy = strategy

        # Warmup
        _ = torch.einsum("ij,jk,kl,lm->im", A, B, C, D)

        t0 = time.perf_counter()
        for _ in range(100):
            _ = torch.einsum("ij,jk,kl,lm->im", A, B, C, D)
        elapsed = time.perf_counter() - t0

        print(f"    strategy='{strategy}': {elapsed*1000:.1f}ms (100 iterations)")

    # Reset
    torch.backends.opt_einsum.strategy = "auto"


# ==========================================================================
# Section 7: torch.backends.flags() Context Manager
# ==========================================================================

def demo_backends_flags():
    """Demonstrate the flags() context manager for temporary overrides."""
    print_section("torch.backends.flags() Context Manager")

    print("  Before context manager:")
    print(f"    cudnn.benchmark = {torch.backends.cudnn.benchmark}")
    print(f"    cudnn.deterministic = {torch.backends.cudnn.deterministic}")

    # Save originals for display
    orig_bench = torch.backends.cudnn.benchmark
    orig_det = torch.backends.cudnn.deterministic

    with torch.backends.flags(
        cudnn_benchmark=True,
        cudnn_deterministic=True,
    ):
        print()
        print("  Inside context manager:")
        print(f"    cudnn.benchmark = {torch.backends.cudnn.benchmark}")
        print(f"    cudnn.deterministic = {torch.backends.cudnn.deterministic}")

    print()
    print("  After context manager (restored):")
    print(f"    cudnn.benchmark = {torch.backends.cudnn.benchmark}")
    print(f"    cudnn.deterministic = {torch.backends.cudnn.deterministic}")

    assert torch.backends.cudnn.benchmark == orig_bench
    assert torch.backends.cudnn.deterministic == orig_det
    print()
    print("  Use cases:")
    print("    - Temporarily disable TF32 for a validation step")
    print("    - Enable benchmark mode for a specific model component")
    print("    - Tests that need isolated backend state")


# ==========================================================================
# Section 8: Optimal Configuration Functions
# ==========================================================================

def configure_for_training(fixed_input_sizes: bool = True):
    """Configure backends for maximum training speed."""
    torch.backends.cudnn.benchmark = fixed_input_sizes
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    return {
        "cudnn.benchmark": fixed_input_sizes,
        "cudnn.deterministic": False,
        "cudnn.allow_tf32": True,
        "cuda.matmul.allow_tf32": True,
        "float32_matmul_precision": "high",
    }


def configure_for_inference(num_cores: int = 4):
    """Configure backends for maximum inference throughput."""
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(num_cores)
    return {
        "cudnn.benchmark": True,
        "cudnn.deterministic": False,
        "cudnn.allow_tf32": True,
        "cuda.matmul.allow_tf32": True,
        "float32_matmul_precision": "high",
        "num_threads": num_cores,
    }


def configure_for_reproducibility(seed: int = 42):
    """Configure backends for exact reproducibility."""
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "seed": seed,
        "cudnn.benchmark": False,
        "cudnn.deterministic": True,
        "cudnn.allow_tf32": False,
        "cuda.matmul.allow_tf32": False,
        "float32_matmul_precision": "highest",
        "deterministic_algorithms": True,
    }


def demo_configurations():
    """Show the three configuration presets."""
    print_section("Configuration Presets")

    print("  1. Training (maximum speed):")
    settings = configure_for_training(fixed_input_sizes=True)
    for k, v in settings.items():
        print(f"     {k}: {v}")

    print()
    print("  2. Inference (maximum throughput):")
    settings = configure_for_inference(num_cores=4)
    for k, v in settings.items():
        print(f"     {k}: {v}")

    print()
    print("  3. Reproducibility (exact results):")
    settings = configure_for_reproducibility(seed=42)
    for k, v in settings.items():
        print(f"     {k}: {v}")

    # Reset deterministic algorithms for remaining demos
    torch.use_deterministic_algorithms(False)


# ==========================================================================
# Main
# ==========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  Module 20: torch.backends — Performance Tuning")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available:  {torch.cuda.is_available()}")
    print("=" * 70)

    print_all_backend_settings()
    demo_cudnn_benchmark()
    demo_tf32_precision()
    demo_matmul_precision_levels()
    demo_openmp_threads()
    demo_opt_einsum()
    demo_backends_flags()
    demo_configurations()

    print_section("Done!")
    print("  All backend demonstrations complete.")
    print("  See README.md for full documentation on each setting.")
