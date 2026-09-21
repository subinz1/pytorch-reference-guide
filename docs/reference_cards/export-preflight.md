# Export Preflight

## Verify the model boundary before deployment

`torch.export` captures a constrained program. Give it representative inputs,
make dynamic dimensions explicit, and compare exported execution with eager
behavior.

```python
import torch
from torch.export import Dim, export

batch = Dim("batch", min=1, max=64)
example = (torch.randn(2, 16),)

model = model.eval()
exported = export(
    model,
    example,
    dynamic_shapes={"x": {0: batch}},
)
```

## Checklist

- Put the model in evaluation mode before exporting.
- Start with one known-good example input.
- Declare only dimensions that are genuinely dynamic.
- Avoid data-dependent Python control flow in the exported region.
- Compare eager and exported outputs on more than one allowed shape.

Export failures are often valuable design feedback. Isolate the smallest
failing submodule before introducing custom decompositions or operators.

