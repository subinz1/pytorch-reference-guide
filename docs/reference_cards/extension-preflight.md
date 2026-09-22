# C++/CUDA extension preflight

Use this card before diagnosing an extension failure as a kernel or autograd
bug.

## Compatibility checklist

- [ ] Python ABI matches the PyTorch installation.
- [ ] Compiler and CUDA toolkit match the supported PyTorch build matrix.
- [ ] `torch.version.cuda` matches the runtime expectation.
- [ ] `TORCH_CUDA_ARCH_LIST` includes the target GPU when building ahead of
  time.
- [ ] A clean build directory is used after changing ABI-sensitive settings.
- [ ] The extension imports before it is tested through a larger model.

Build a tiny operator first, test CPU behavior where applicable, then add CUDA
and autograd coverage. Keep compiler command lines and the first linker error;
later errors are often consequences of the first missing symbol.
