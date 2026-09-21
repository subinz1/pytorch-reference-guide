# Device and Dtype Policy

## Keep model and inputs aligned

Most device and dtype errors come from moving only one side of an operation.
Choose a device and dtype at the application boundary, then derive input
placement from the model.

```python
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.Linear(16, 4).to(device=device, dtype=torch.bfloat16)

inputs = torch.randn(8, 16, device=device, dtype=torch.bfloat16)
outputs = model(inputs)
```

## Checklist

- Move a module with `model.to(device=..., dtype=...)`, not parameter by
  parameter.
- Register persistent tensors with `register_buffer()` so module moves include
  them.
- Create temporary tensors from an existing tensor when possible:
  `torch.zeros_like(x)`, `x.new_zeros(...)`.
- Keep integer index tensors integral; do not cast token IDs to a floating dtype.
- Use a single device-selection policy instead of scattered `.cuda()` calls.

For mixed precision, let autocast choose eligible operation dtypes instead of
casting every intermediate manually.

