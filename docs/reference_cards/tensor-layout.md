# Tensor Layout: Views, Strides, and Contiguity

## The question

Why did `view()` fail, or why did a harmless-looking reshape allocate memory?

A tensor is described by storage, shape, strides, and storage offset. A view
changes metadata only; a copy allocates new storage.

```python
import torch

x = torch.arange(24).reshape(2, 3, 4)
y = x.permute(2, 0, 1)  # shape [4, 2, 3], non-contiguous view

print(y.is_contiguous())  # False
z = y.reshape(4, 6)      # works; may allocate when a view is impossible
w = y.contiguous().view(4, 6)  # explicit copy, then metadata-only view
```

## Checklist

- Inspect `tensor.shape`, `tensor.stride()`, and `tensor.is_contiguous()`.
- Prefer `reshape()` when either a view or a copy is acceptable.
- Use `view()` only when a view is required and layout is known.
- Call `contiguous()` only at an operation boundary that requires it.
- Do not assume a transpose or `permute()` copied data; it usually did not.

A layout conversion is a real performance cost. Keep tensors in the layout that
the next expensive operation expects instead of repeatedly converting them.

