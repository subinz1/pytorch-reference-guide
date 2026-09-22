# `torch.compile` correctness

Use this card when a compiled model is faster but you need evidence that it is
still producing the same result as eager execution.

```python
torch.testing.assert_close(compiled_output, eager_output, rtol=1e-4, atol=1e-5)
```

## Minimal workflow

1. Fix the random seed and switch the model to `eval()`.
2. Compare eager and compiled outputs on representative shapes, dtypes, and
   devices.
3. Compare gradients separately when the workload trains.
4. Include boundary inputs: zeros, unusually long batches, and non-contiguous
   views when the model accepts them.
5. Call `torch._dynamo.reset()` between isolated experiments so compile-cache
   state does not hide a reproducer.

Do not compare training-mode outputs with Dropout active unless the random
state is deliberately synchronized. See
[`compile_correctness_harness.py`](../../08_torch_compile/compile_correctness_harness.py)
for a complete example.
