# Shapes and Broadcasting

## Make dimensions part of the design

Name the semantic role of every axis before writing an operation. A common
convention is `[batch, sequence, features]`; broadcasting then becomes
predictable rather than accidental.

```python
import torch

scores = torch.randn(8, 16, 32)   # [B, T, C]
scale = torch.randn(32)           # [C]
bias = torch.randn(1, 1, 32)      # explicit broadcast axes

normalized = scores * scale + bias
per_token_max = scores.amax(dim=-1, keepdim=True)  # [B, T, 1]
```

## Checklist

- Use `keepdim=True` when the reduced result will be combined with the input.
- Add singleton dimensions deliberately with `unsqueeze` or indexing such as
  `x[:, None]`.
- Assert important shapes near module boundaries.
- Prefer named variables for dimensions over unexplained positional `dim`
  arguments.
- Test with more than one batch size and sequence length.

Broadcasting aligns trailing dimensions. If an operation is surprising, write
both operand shapes next to it and align them from the right.

