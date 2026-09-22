# Checkpoint compatibility

Use this card when a checkpoint must survive code changes, device moves, or a
production rollback.

## Save deliberately

```python
torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "step": step,
    "config": config,
}, path)
```

Prefer state dictionaries over pickled whole modules. They make code ownership
explicit and allow controlled migration when module structure changes.

## Load defensively

1. Use `map_location` for the target device.
2. Validate expected keys and shapes; do not ignore unexpected keys by default.
3. Version the application config or migration path alongside long-lived
   checkpoints.
4. Test a save/load round trip in CI for representative models.

See [`test_serialization_contract.py`](../../14_testing/test_serialization_contract.py)
for a small contract test.
