# Validation

The guide has lightweight checks that run from the repository root and use the
active Python environment.

## Content inventory

```bash
python tools/check_content_inventory.py
```

This verifies the curriculum sequence, operational guides, notebook count,
reference-card count, and Python-example count.

## Representative CPU smoke test

```bash
python tools/run_cpu_smoke.py
```

The smoke manifest intentionally covers a small, fast set of CPU-safe examples
across foundational modules and the operational guides. It is not a replacement
for module-specific or accelerator validation.

List the current examples without executing them:

```bash
python tools/run_cpu_smoke.py --list
```

Run one manifest entry while debugging:

```bash
python tools/run_cpu_smoke.py --only 03_autograd/gradient_basics.py
```

The command uses the active interpreter. Install the guide's supported PyTorch
environment first and follow the project [requirements](../README.md#requirements).
