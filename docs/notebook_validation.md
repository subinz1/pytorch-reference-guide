# Notebook Validation

Validate the full notebook collection from the repository root:

```bash
python tools/validate_notebooks.py
```

This verifies the notebook sequence, `python3` kernel metadata, Python
language metadata, code-cell presence, and a clean committed state with no
outputs or execution counts.

Validate one notebook while editing it:

```bash
python tools/validate_notebooks.py --path notebooks/01_tensors_masterclass.ipynb
```

Execute a notebook only in an environment that satisfies its runtime contract.
Write execution output outside the repository or clear it before committing.
Accelerator-specific notebooks must not be used as CPU validation.
