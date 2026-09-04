# Module 50: Sparse Tensors & Sparse Ops

## Overview

PyTorch **sparse tensors** store only nonzero values (plus indices), saving memory
and compute when data is highly sparse — recommendation matrices, graph
adjacency, embeddings with large vocabularies. This module surveys layouts
(`coo`, `csr`, `csc`, `csc`), conversion, and the sparse operator surface.

## Key Concepts

### Layouts

| Layout | Best for | Structure |
|--------|----------|-----------|
| **COO** | Construction, irregular sparsity | indices `[ndim, nnz]` + values |
| **CSR** | Fast row slices, SpMM on CPU/CUDA | crow_indices, col_indices, values |
| **CSC** | Fast column slices | ccol_indices, row_indices, values |
| **BSR / BSC** | Block-sparse (structured) | block indices + dense blocks |

Create with `torch.sparse_coo_tensor` / `torch.sparse_csr_tensor`, or
`dense.to_sparse()` / `dense.to_sparse_csr()`.

### Hybrid Tensors

A tensor can be sparse in some leading dims and dense in others (batched sparse).
Example: batched graph adjacency `[B, N, N]` stored as sparse batches.

### Sparse Ops

Common ops with sparse support (version-dependent):

- Arithmetic: `+`, `*`, `sparse.addmm` / `torch.sparse.mm`
- Reductions: `sum`, `softmax` (layout-specific)
- Conversions: `to_dense()`, `coalesce()` (merge duplicate COO indices)

Always **`coalesce()`** COO tensors before relying on unique indices.

### Gradients

Sparse tensors can participate in autograd when ops support it. Gradients w.r.t.
sparse parameters may be sparse or dense depending on the op — check docs and
test with small examples.

## Examples

```python
import torch

i = torch.tensor([[0, 1, 1],
                  [2, 0, 2]])
v = torch.tensor([3.0, 4.0, 5.0])
s = torch.sparse_coo_tensor(i, v, (2, 3)).coalesce()
print(s.to_dense())

# CSR SpMM-style matmul with a dense vector
csr = s.to_sparse_csr()
x = torch.randn(3)
y = torch.sparse.mm(csr, x.unsqueeze(1)).squeeze(1)
```

## When to Use

- Density below ~1–5% and ops you need are sparse-aware.
- Graph NNs, sparse attention masks, large embedding bags.
- Prefer dense + masking when sparsity is moderate or ops lack sparse kernels.
- Structured 2:4 sparsity for Ampere+ Tensor Cores — see Module 31 (torchao).

## Files in This Module

| File | Description |
|------|-------------|
| `sparse_basics.py` | COO/CSR creation, coalesce, matmul, autograd sketch |

## References

- [torch.sparse docs](https://pytorch.org/docs/stable/sparse.html)
- [Sparse semi-structured](https://pytorch.org/docs/stable/sparse.html#sparse-semi-structured-tensors)
- Module 13: [Advanced](../13_advanced/) · Module 31: [torchao](../31_torchao/)
