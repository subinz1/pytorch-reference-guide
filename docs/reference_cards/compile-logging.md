# `torch.compile` debugging logs

Use this card when a compile failure or graph break needs a small, shareable
reproducer.

```bash
TORCH_LOGS="graph_breaks,recompiles" python reproducer.py
```

## Escalation path

1. Reduce the model to the smallest failing function and input.
2. Record the PyTorch version, device, dtype, input shapes, and first error.
3. Start with `graph_breaks` and `recompiles`; add more log channels only when
   they answer a specific question.
4. Use `torch._dynamo.explain(fn)(*inputs)` to inspect capture boundaries.
5. Verify the eager path works before filing a compiler issue.

Avoid attaching secrets, production tensors, or full environment dumps to an
issue. A deterministic synthetic input is more useful than a large trace.
