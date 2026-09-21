# Checkpointing

## Save enough state to resume, not merely reload

A model-only checkpoint is useful for inference. A resumable training checkpoint
also needs optimizer, scheduler, scaler, counters, and configuration state.

```python
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "scaler": scaler.state_dict(),
    "epoch": epoch,
    "global_step": global_step,
    "config": vars(args),
}
torch.save(checkpoint, "checkpoint.pt")
```

## Checklist

- Save after a completed optimizer step, not mid-update.
- Use `map_location` when loading across CPU/GPU environments.
- Keep “last” and “best validation metric” checkpoints separately.
- Store the metric and data split used to decide “best.”
- Test resume by stopping a short run, loading, and comparing the next step.

Treat checkpoint formats as an interface. Version the dictionary when training
code is expected to evolve.

