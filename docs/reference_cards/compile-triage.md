# `torch.compile` Triage

## Establish a correct baseline first

Compilation should preserve results before it improves performance. Start with a
small representative input and compare eager and compiled outputs.

```python
import torch

model = model.eval().cuda()
compiled = torch.compile(model, mode="default")

x = torch.randn(8, 128, device="cuda")
with torch.inference_mode():
    eager = model(x)
    optimized = compiled(x)

torch.testing.assert_close(optimized, eager)
```

## Checklist

- Measure warm-up separately from steady-state execution.
- Use `fullgraph=True` temporarily to locate graph breaks.
- Use `dynamic=True` only when varying shapes are a real workload property.
- Reduce Python-side mutation and data-dependent control flow in hot paths.
- Keep an eager fallback while investigating compiler issues.

A compile failure is useful evidence: minimize it to the smallest module and
input that reproduces the issue before changing application logic.

