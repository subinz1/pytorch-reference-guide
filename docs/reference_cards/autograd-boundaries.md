# Autograd Boundaries

## Control graphs explicitly

Autograd records operations on tensors that require gradients. Use the right
boundary mechanism for the intent instead of mutating tensors through
`.data`.

```python
import torch

x = torch.randn(4, requires_grad=True)
loss = (x.square()).mean()
loss.backward()

with torch.no_grad():
    x.add_(-0.1 * x.grad)  # update that must not become part of a graph

features = x.detach()      # shares storage, has no gradient history
```

## Checklist

- Use `torch.no_grad()` for evaluation-time updates and parameter updates.
- Use `detach()` when a downstream computation must not backpropagate into an
  upstream tensor.
- Use `clone()` as well when the detached tensor will be modified in place.
- Never use `.data`; it can silently invalidate autograd's version tracking.
- Clear gradients at the start of each optimizer step.

If a tensor unexpectedly has no gradient, inspect `requires_grad`,
`grad_fn`, and whether it crossed a detach or no-grad boundary.

