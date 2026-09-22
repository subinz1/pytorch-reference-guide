# Randomness audit

Use this card when results change across runs, workers, or ranks.

## Seed the relevant generators

```python
torch.manual_seed(seed)
generator = torch.Generator().manual_seed(seed)
```

Also account for Python's `random`, NumPy when present, DataLoader workers, and
each distributed rank. A seed does not guarantee bitwise equality across every
device, library version, or nondeterministic kernel.

## Audit questions

- Which generator draws the random value?
- Is the generator state restored when resuming from a checkpoint?
- Does each worker/rank get a deliberate derived seed?
- Is the test asserting a distribution/property rather than one random value?
- Is deterministic mode required, and is its performance cost acceptable?

For a test, record the failing seed in the assertion message so the failure can
be reproduced rather than merely observed.
