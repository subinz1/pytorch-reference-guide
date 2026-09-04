"""
Module 50: Sparse Tensor Basics
===============================
COO/CSR construction, coalesce, dense conversion, sparse matmul,
and a small autograd example with sparse weights.

Run: python sparse_basics.py
"""

import torch


# ---------------------------------------------------------------------------
# PART 1: COO construction and coalesce
# ---------------------------------------------------------------------------
def demo_coo_basics():
    print("=" * 70)
    print("PART 1: COO tensors")
    print("=" * 70)
    # Duplicate index (0, 2) appears twice — coalesce sums values
    indices = torch.tensor([[0, 1, 0, 1],
                            [2, 0, 2, 2]])
    values = torch.tensor([3.0, 4.0, 1.0, 5.0])
    s = torch.sparse_coo_tensor(indices, values, size=(2, 3))
    print(f"is_coalesced before: {s.is_coalesced()}")
    s = s.coalesce()
    print(f"is_coalesced after:  {s.is_coalesced()}")
    print(f"indices:\n{s.indices()}")
    print(f"values:  {s.values().tolist()}")
    print(f"dense:\n{s.to_dense()}")


# ---------------------------------------------------------------------------
# PART 2: CSR layout and row slicing benefits
# ---------------------------------------------------------------------------
def demo_csr():
    print("\n" + "=" * 70)
    print("PART 2: CSR layout")
    print("=" * 70)
    dense = torch.tensor([[0.0, 0.0, 3.0],
                          [4.0, 0.0, 5.0]])
    csr = dense.to_sparse_csr()
    print(f"crow_indices: {csr.crow_indices().tolist()}")
    print(f"col_indices:  {csr.col_indices().tolist()}")
    print(f"values:       {csr.values().tolist()}")
    print(f"layout: {csr.layout}, nnz={csr._nnz()}")


# ---------------------------------------------------------------------------
# PART 3: Sparse @ dense matmul
# ---------------------------------------------------------------------------
def demo_sparse_mm():
    print("\n" + "=" * 70)
    print("PART 3: torch.sparse.mm")
    print("=" * 70)
    i = torch.tensor([[0, 1, 1],
                      [0, 0, 2]])
    v = torch.tensor([1.0, 2.0, 3.0])
    a = torch.sparse_coo_tensor(i, v, (2, 3)).coalesce().to_sparse_csr()
    b = torch.tensor([[1.0, 0.0],
                      [0.0, 1.0],
                      [1.0, 1.0]])
    y = torch.sparse.mm(a, b)
    print(f"A (dense):\n{a.to_dense()}")
    print(f"B:\n{b}")
    print(f"A @ B:\n{y}")
    print(f"ref dense mm:\n{a.to_dense() @ b}")


# ---------------------------------------------------------------------------
# PART 4: Autograd through sparse → dense path
# ---------------------------------------------------------------------------
def demo_autograd():
    print("\n" + "=" * 70)
    print("PART 4: Autograd with sparse weights")
    print("=" * 70)
    i = torch.tensor([[0, 1],
                      [1, 0]])
    v = torch.tensor([0.5, -1.0], requires_grad=True)
    w_sparse = torch.sparse_coo_tensor(i, v, (2, 2)).coalesce()
    x = torch.tensor([[1.0], [2.0]])
    # Convert for a simple dense matmul that still tracks v
    y = w_sparse.to_dense() @ x
    loss = y.square().sum()
    loss.backward()
    print(f"y:\n{y.detach()}")
    print(f"grad w.r.t. sparse values: {v.grad.tolist()}")


# ---------------------------------------------------------------------------
# PART 5: Sparsity stats helper
# ---------------------------------------------------------------------------
def sparsity_report(t: torch.Tensor) -> dict:
    """Return density stats for a dense or sparse tensor."""
    if t.is_sparse:
        nnz = t._nnz()
        total = t.numel()
    else:
        nnz = int((t != 0).sum())
        total = t.numel()
    density = nnz / total if total else 0.0
    return {"shape": tuple(t.shape), "nnz": nnz, "density": density}


def demo_sparsity_stats():
    print("\n" + "=" * 70)
    print("PART 5: Sparsity report")
    print("=" * 70)
    dense = torch.zeros(100, 100)
    dense[torch.randint(0, 100, (50,)), torch.randint(0, 100, (50,))] = 1.0
    sparse = dense.to_sparse()
    for name, t in [("dense", dense), ("coo", sparse)]:
        stats = sparsity_report(t)
        print(f"{name}: shape={stats['shape']}, nnz={stats['nnz']}, "
              f"density={stats['density']:.4%}")


# ---------------------------------------------------------------------------
# PART 6: When sparse wins (micro-benchmark heuristic)
# ---------------------------------------------------------------------------
def demo_when_sparse_helps():
    print("\n" + "=" * 70)
    print("PART 6: Density heuristic")
    print("=" * 70)
    print("Rule of thumb:")
    print("  density < ~1%  → strong candidate for sparse kernels")
    print("  1–5%           → measure; layout + hardware matter")
    print("  > 5–10%        → dense + mask often faster")
    for density in [0.001, 0.01, 0.05, 0.2]:
        n = 500
        nnz = int(density * n * n)
        idx = torch.randint(0, n, (2, nnz))
        val = torch.randn(nnz)
        s = torch.sparse_coo_tensor(idx, val, (n, n)).coalesce()
        print(f"  built COO n={n}, target_density={density:.1%}, "
              f"actual={s._nnz() / (n * n):.3%}")


if __name__ == "__main__":
    demo_coo_basics()
    demo_csr()
    demo_sparse_mm()
    demo_autograd()
    demo_sparsity_stats()
    demo_when_sparse_helps()
    print("\nAll sparse demos finished.")
