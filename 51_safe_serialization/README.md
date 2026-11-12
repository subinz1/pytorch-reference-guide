# Module 51: Safe Model Serialization

## Overview

A checkpoint is an input boundary. Treat it with the same care as data received
from a network: know its source, load only the state you expect, validate it,
and keep the artifact format separate from executable Python objects.

For normal training and inference, save a model's `state_dict()` and load it
with `weights_only=True`. This is the portable, inspectable path for tensors and
simple Python containers. Do not load an untrusted checkpoint with
`weights_only=False`: pickle can execute arbitrary code during deserialization.

## The Recommended Contract

| Concern | Recommended practice | Why |
|---|---|---|
| Model weights | Save `model.state_dict()` | Keeps architecture in reviewed source code |
| Optimizer state | Save it separately with an explicit schema version | Makes resume behavior intentional |
| Loading | Pass `weights_only=True` and `map_location` explicitly | Restricts deserialization and avoids surprise device allocation |
| Compatibility | Record format version, PyTorch version, dtype, and model config | Gives loaders an actionable compatibility contract |
| Integrity | Verify a trusted artifact digest before loading | Detects corruption and unexpected replacement |
| Custom classes | Allowlist only reviewed types with `safe_globals` | Avoids broadly weakening the loading policy |

## Save State, Not Objects

Saving an entire `nn.Module` captures a Python object graph. It couples the
checkpoint to import paths and requires pickle support for every referenced
class. A state dictionary instead contains parameters and persistent buffers;
the receiving application constructs its reviewed model class, then applies the
state.

```python
model = TinyClassifier()
torch.save(model.state_dict(), "classifier-state.pt")

restored = TinyClassifier()
state = torch.load(
    "classifier-state.pt",
    map_location="cpu",
    weights_only=True,
)
restored.load_state_dict(state, strict=True)
restored.eval()
```

## Validate Before Using a Checkpoint

`weights_only=True` narrows the deserialization surface but is not a general
security guarantee. It does not establish who produced the file, prevent denial
of service from very large data, or prove that a tensor shape is suitable for a
particular model. Verify provenance and a digest before loading, then validate
the expected keys, shapes, and dtypes through `load_state_dict(strict=True)`.

For a checkpoint that needs a non-default class or function, inspect it first
with `torch.serialization.get_unsafe_globals_in_checkpoint()` when available.
Allowlist the smallest reviewed set with `torch.serialization.safe_globals()`;
do not switch the entire application to `weights_only=False` as a workaround.

## Device Placement and Memory

Use `map_location="cpu"` when loading an artifact from a GPU environment into a
CPU process, or before moving a reviewed model to its intended device. This
avoids an unexpected GPU allocation during load. For large tensor-only
checkpoints, `mmap=True` can defer storage reads until tensor data is accessed.

```python
state = torch.load(
    "classifier-state.pt",
    map_location="cpu",
    mmap=True,
    weights_only=True,
)
```

## A Versioned Training Checkpoint

Training resume state is more than model weights. Make the schema explicit and
keep it to tensor-friendly values and primitive metadata.

```python
checkpoint = {
    "format_version": 1,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "step": step,
    "model_config": {"hidden_size": 256, "num_classes": 10},
}
torch.save(checkpoint, "resume.pt")
```

On load, reject unsupported format versions before applying any state. A
versioned schema lets you add migrations deliberately instead of relying on
implicit assumptions about key names or optimizer internals.

## Files in This Module

| File | Description |
|---|---|
| `safe_checkpoint.py` | State-dict round trip, checksum verification, explicit load validation, and safe-global inspection |

## References

- [PyTorch serialization semantics](https://docs.pytorch.org/docs/stable/notes/serialization.html)
- [`torch.load`](https://docs.pytorch.org/docs/stable/generated/torch.load.html)
- [Saving and loading models tutorial](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
