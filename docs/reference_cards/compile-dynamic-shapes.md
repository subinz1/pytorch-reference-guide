# `torch.compile` and dynamic shapes

Use this card when batch or sequence lengths vary and compilation count grows
unexpectedly.

## Decide first

| Observation | First action |
|---|---|
| a small known set of shapes | warm each shape and measure cache reuse |
| arbitrary batch/sequence length | try `dynamic=True` and measure again |
| one dimension changes while others remain fixed | mark only that dimension dynamic when appropriate |
| compile time dominates | bucket inputs or reduce shape diversity |

## Measure, do not infer

Use a tiny backend that counts graph compilations. Run a known shape sequence
twice: the second pass should not add graphs unless guards genuinely differ.

`dynamic=True` is not a universal performance switch. It trades specialization
for fewer recompiles, so benchmark representative production shapes after
correctness checks. See
[`compile_shape_cache.py`](../../08_torch_compile/compile_shape_cache.py).
