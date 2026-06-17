"""
Custom Triton Kernels — PyTorch Integration
============================================
Module 25: Register Triton kernels as PyTorch custom ops, add autograd
support, use with torch.compile, and autotune for peak performance.

Requirements: pip install torch triton (NVIDIA GPU required for kernel execution)
"""

import torch

# ---------------------------------------------------------------------------
# Check Triton availability
# ---------------------------------------------------------------------------
HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    pass

HAS_CUDA = torch.cuda.is_available()

print("=" * 70)
print("Module 25: Triton Kernels — PyTorch Integration")
print("=" * 70)
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {HAS_CUDA}")
print(f"Triton available: {HAS_TRITON}")
if HAS_CUDA:
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
print("=" * 70)


# ===================================================================
# SECTION 1: Register a Triton Kernel as a PyTorch Custom Op
# ===================================================================
print("\n" + "=" * 70)
print("SECTION 1: torch.library.custom_op — Registering Triton Kernels")
print("=" * 70)

print("""
The torch.library API lets you register custom operations that work
seamlessly with PyTorch's autograd, torch.compile, and export.

Pipeline:
  1. Write the Triton kernel (@triton.jit)
  2. Wrap in @torch.library.custom_op (defines the eager implementation)
  3. Register a "fake" (Meta) implementation for shape inference
  4. Register autograd (forward + backward)
  5. Use with torch.compile — it just works
""")

if HAS_TRITON and HAS_CUDA:

    # ---------------------------------------------------------------
    # Step 1: Define the Triton kernel — fused GELU activation
    # ---------------------------------------------------------------
    @triton.jit
    def _gelu_forward_kernel(
        x_ptr, out_ptr, n,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n

        x = tl.load(x_ptr + offsets, mask=mask)

        # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        k = 0.7978845608028654  # sqrt(2/pi)
        inner = k * (x + 0.044715 * x * x * x)
        out = 0.5 * x * (1.0 + tl.math.tanh(inner))

        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _gelu_backward_kernel(
        grad_out_ptr, x_ptr, grad_in_ptr, n,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n

        grad_out = tl.load(grad_out_ptr + offsets, mask=mask)
        x = tl.load(x_ptr + offsets, mask=mask)

        # GELU derivative
        k = 0.7978845608028654
        x3 = x * x * x
        inner = k * (x + 0.044715 * x3)
        tanh_inner = tl.math.tanh(inner)
        sech2 = 1.0 - tanh_inner * tanh_inner
        gelu_grad = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * k * (1.0 + 3.0 * 0.044715 * x * x)

        tl.store(grad_in_ptr + offsets, grad_out * gelu_grad, mask=mask)

    # ---------------------------------------------------------------
    # Step 2: Register as a custom op
    # ---------------------------------------------------------------
    @torch.library.custom_op("mylib::triton_gelu", mutates_args=())
    def triton_gelu(x: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(x)
        n = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        _gelu_forward_kernel[grid](x, output, n, BLOCK_SIZE=BLOCK_SIZE)
        return output

    # ---------------------------------------------------------------
    # Step 3: Meta (Fake Tensor) implementation for shape inference
    # torch.compile needs this to trace without running the kernel.
    # ---------------------------------------------------------------
    @triton_gelu.register_fake
    def triton_gelu_fake(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    # ---------------------------------------------------------------
    # Step 4: Autograd support
    # ---------------------------------------------------------------
    def triton_gelu_setup_context(ctx, inputs, output):
        (x,) = inputs
        ctx.save_for_backward(x)

    def triton_gelu_backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = torch.empty_like(x)
        n = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        _gelu_backward_kernel[grid](grad_output, x, grad_input, n, BLOCK_SIZE=BLOCK_SIZE)
        return grad_input

    triton_gelu.register_autograd(
        triton_gelu_backward,
        setup_context=triton_gelu_setup_context,
    )

    # ---------------------------------------------------------------
    # Test the full pipeline
    # ---------------------------------------------------------------
    print("Testing custom op: torch.ops.mylib.triton_gelu")

    x = torch.randn(1024, device="cuda")
    y = torch.ops.mylib.triton_gelu(x)
    y_ref = torch.nn.functional.gelu(x, approximate="tanh")
    print(f"  Forward:  max error = {(y - y_ref).abs().max().item():.2e}")

    x_grad = torch.randn(256, device="cuda", requires_grad=True)
    y_grad = torch.ops.mylib.triton_gelu(x_grad)
    y_grad.sum().backward()
    x_ref = x_grad.detach().clone().requires_grad_(True)
    y_ref = torch.nn.functional.gelu(x_ref, approximate="tanh")
    y_ref.sum().backward()
    print(f"  Backward: max error = {(x_grad.grad - x_ref.grad).abs().max().item():.2e}")
    print("  Autograd: works ✓")

else:
    print("""
[CPU-only mode] Registration pattern:

  @torch.library.custom_op("mylib::my_op", mutates_args=())
  def my_op(x: torch.Tensor) -> torch.Tensor:
      output = torch.empty_like(x)
      my_triton_kernel[grid](x, output, n, BLOCK_SIZE=1024)
      return output

  @my_op.register_fake
  def my_op_fake(x):
      return torch.empty_like(x)   # shape inference only

  def my_op_backward(ctx, grad_output):
      ...                           # compute gradient
  my_op.register_autograd(my_op_backward, setup_context=...)
""")


# ===================================================================
# SECTION 2: Using Custom Ops with torch.compile
# ===================================================================
print("\n" + "=" * 70)
print("SECTION 2: Custom Triton Ops + torch.compile")
print("=" * 70)

if HAS_TRITON and HAS_CUDA:

    @torch.compile
    def compiled_model(x):
        return torch.ops.mylib.triton_gelu(x)

    x = torch.randn(2048, device="cuda")
    y = compiled_model(x)
    y_ref = torch.nn.functional.gelu(x, approximate="tanh")
    print(f"  torch.compile + custom op: max error = {(y - y_ref).abs().max().item():.2e}")
    print("  Works with torch.compile ✓")

    # In a real model
    class ModelWithCustomGELU(torch.nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.linear1 = torch.nn.Linear(d_model, d_model * 4)
            self.linear2 = torch.nn.Linear(d_model * 4, d_model)

        def forward(self, x):
            x = self.linear1(x)
            x = torch.ops.mylib.triton_gelu(x)
            return self.linear2(x)

    model = ModelWithCustomGELU(512).cuda()
    compiled_model = torch.compile(model)
    x = torch.randn(32, 512, device="cuda")
    out = compiled_model(x)
    print(f"  Compiled model output shape: {out.shape}")
    print("  Custom op in nn.Module + torch.compile ✓")

else:
    print("""
[CPU-only mode] Using with torch.compile:

  @torch.compile
  def model_forward(x):
      return torch.ops.mylib.triton_gelu(x)  # seamlessly compiled

  # The register_fake implementation tells the compiler the output
  # shape/dtype without running the actual kernel during tracing.
  # Inductor can schedule your kernel alongside its generated ones.
""")


# ===================================================================
# SECTION 3: Autotuning with @triton.autotune
# ===================================================================
print("\n" + "=" * 70)
print("SECTION 3: Autotuning — Finding the Best Block Size")
print("=" * 70)

print("""
Different GPUs and problem sizes have different optimal BLOCK_SIZEs.
@triton.autotune benchmarks multiple configs and picks the fastest.

  @triton.autotune(
      configs=[
          triton.Config({'BLOCK_SIZE': 128}),
          triton.Config({'BLOCK_SIZE': 256}),
          triton.Config({'BLOCK_SIZE': 512}),
          triton.Config({'BLOCK_SIZE': 1024}),
      ],
      key=['n'],  # re-tune when n changes
  )
  @triton.jit
  def my_kernel(..., BLOCK_SIZE: tl.constexpr):
      ...

How it works:
  1. First call → benchmarks all configs, picks fastest for this 'n'
  2. Same n again → uses cached best config (no re-benchmark)
  3. Different n → re-benchmarks (optimal config may differ)
""")

if HAS_TRITON and HAS_CUDA:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 128}),
            triton.Config({"BLOCK_SIZE": 256}),
            triton.Config({"BLOCK_SIZE": 512}),
            triton.Config({"BLOCK_SIZE": 1024}),
            triton.Config({"BLOCK_SIZE": 2048}),
        ],
        key=["n"],
    )
    @triton.jit
    def add_kernel_autotuned(
        x_ptr, y_ptr, out_ptr, n,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)

    def autotuned_add(x, y):
        output = torch.empty_like(x)
        n = x.numel()
        # Lambda grid: BLOCK_SIZE is chosen by autotuner at runtime
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        add_kernel_autotuned[grid](x, y, output, n)
        return output

    print("Running autotuned vector addition (first call triggers benchmarking)...")
    x = torch.randn(1_000_000, device="cuda")
    y = torch.randn(1_000_000, device="cuda")
    z = autotuned_add(x, y)
    assert torch.allclose(z, x + y, atol=1e-6)
    print(f"  Autotuned result correct ✓")

    # Show which config was selected
    if hasattr(add_kernel_autotuned, "best_config"):
        print(f"  Best config: {add_kernel_autotuned.best_config}")
    print()


# ===================================================================
# SECTION 4: Viewing Inductor-Generated Triton Code
# ===================================================================
print("\n" + "=" * 70)
print("SECTION 4: How TorchInductor Uses Triton")
print("=" * 70)

print("""
When you torch.compile a function, TorchInductor generates Triton code:

  TORCH_LOGS="output_code" python my_script.py

Or programmatically:
  import torch._logging
  torch._logging.set_logs(output_code=True)

The generated code shows exactly what Inductor fuses. For example,
torch.compile(lambda x, y: torch.relu(x + y)) generates a single
Triton kernel that fuses the add and relu — the same optimization
you'd write by hand.

Your custom_op kernels appear in the Inductor schedule alongside
generated kernels. Inductor can fuse ops before/after your kernel.
""")

if HAS_CUDA:
    print("Demonstrating torch.compile code generation...")
    print("(Set TORCH_LOGS='output_code' to see full generated Triton)\n")

    @torch.compile
    def simple_fused(x, y):
        return torch.relu(x + y)

    x = torch.randn(1024, device="cuda")
    y = torch.randn(1024, device="cuda")
    result = simple_fused(x, y)
    expected = torch.relu(x + y)
    print(f"  torch.compile fused add+relu: max error = {(result - expected).abs().max().item():.2e}")
    print("  Inductor automatically fused add+relu into one Triton kernel ✓")
    print()
    print("  To inspect generated code, run:")
    print("    TORCH_LOGS='output_code' python triton_with_pytorch.py")


# ===================================================================
# SECTION 5: Complete Integration Example
# ===================================================================
print("\n" + "=" * 70)
print("SECTION 5: Complete Example — Fused RMSNorm")
print("=" * 70)

if HAS_TRITON and HAS_CUDA:

    @triton.jit
    def _rmsnorm_fwd_kernel(
        x_ptr, weight_ptr, out_ptr,
        n_cols, eps,
        x_row_stride, out_row_stride,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        x_start = x_ptr + row_idx * x_row_stride
        x = tl.load(x_start + col_offsets, mask=mask, other=0.0).to(tl.float32)

        # RMS = sqrt(mean(x^2) + eps)
        x_sq = x * x
        mean_sq = tl.sum(x_sq, axis=0) / n_cols
        rms = tl.math.rsqrt(mean_sq + eps)

        # Normalize and scale
        weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
        out = x * rms * weight

        out_start = out_ptr + row_idx * out_row_stride
        tl.store(out_start + col_offsets, out, mask=mask)

    @torch.library.custom_op("mylib::triton_rmsnorm", mutates_args=())
    def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        n_rows, n_cols = x.shape
        output = torch.empty_like(x)
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        grid = (n_rows,)
        _rmsnorm_fwd_kernel[grid](
            x, weight, output,
            n_cols, eps,
            x.stride(0), output.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return output

    @triton_rmsnorm.register_fake
    def triton_rmsnorm_fake(x, weight, eps=1e-6):
        return torch.empty_like(x)

    # Test against PyTorch reference
    def pytorch_rmsnorm(x, weight, eps=1e-6):
        rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
        return (x.float() * rms * weight).to(x.dtype)

    M, K = 512, 768
    x = torch.randn(M, K, device="cuda")
    w = torch.ones(K, device="cuda")

    out_triton = torch.ops.mylib.triton_rmsnorm(x, w)
    out_ref = pytorch_rmsnorm(x, w)
    print(f"  Fused RMSNorm ({M}x{K}): max error = {(out_triton - out_ref).abs().max().item():.2e}")
    print("  Custom op works ✓")

    # Works with torch.compile
    @torch.compile
    def compiled_rmsnorm(x, w):
        return torch.ops.mylib.triton_rmsnorm(x, w)

    out_compiled = compiled_rmsnorm(x, w)
    print(f"  + torch.compile:           max error = {(out_compiled - out_ref).abs().max().item():.2e}")
    print("  Works with torch.compile ✓")

else:
    print("""
[CPU-only mode] Complete integration pattern:

  1. @triton.jit kernel   — the GPU code
  2. @custom_op           — wraps kernel as a PyTorch op
  3. .register_fake       — shape inference for torch.compile
  4. .register_autograd   — backward pass (optional, for training)
  5. torch.compile(model) — Inductor schedules your kernel

  Example: Fused RMSNorm
    - Load row, compute mean(x^2), rsqrt, scale by weight
    - All in one kernel (no intermediate tensors)
    - Registered as torch.ops.mylib.triton_rmsnorm
""")


# ===================================================================
# SECTION 6: Summary
# ===================================================================
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("""
What we covered:
  1. torch.library.custom_op — register Triton kernels as PyTorch ops
  2. register_fake — shape inference for torch.compile tracing
  3. register_autograd — backward pass for training
  4. @triton.autotune — automatically find the best block size
  5. TorchInductor — how torch.compile generates Triton code
  6. Complete example — fused RMSNorm as a custom op

Integration pipeline:
  Triton kernel → custom_op → register_fake → register_autograd
  → works with torch.compile, torch.export, and autograd

Key points:
  - Always provide a register_fake impl (returns empty_like)
  - Autograd is optional (only needed for training, not inference)
  - torch.compile can schedule your kernels with Inductor-generated ones
  - Use TORCH_LOGS="output_code" to inspect what Inductor generates
""")
