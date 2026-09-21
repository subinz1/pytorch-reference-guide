# Reproducibility

## Make the experiment state explicit

A seed alone is not a complete reproducibility policy. Record software,
hardware, data, configuration, and randomness boundaries.

```python
import random
import numpy as np
import torch

seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
```

## Checklist

- Store the complete training configuration with every checkpoint.
- Seed Python, NumPy, PyTorch CPU, and all CUDA devices.
- Seed DataLoader workers and document sampler behavior.
- Record the PyTorch version, CUDA version, device model, and git revision.
- Expect deterministic algorithms to reject or slow some kernels.

Use deterministic mode to diagnose a discrepancy. For production training,
choose the reproducibility/performance tradeoff deliberately and document it.

