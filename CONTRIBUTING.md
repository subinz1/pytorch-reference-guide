# Contributing to PyTorch Reference Guide

Thank you for your interest in contributing! This guide covers how to add
new modules, fix issues, and maintain consistency across the repository.

## Repository Structure

```
NN_module_name/
├── README.md              # Module overview, key concepts, references
├── example_script.py      # Runnable Python examples (80-150 lines)
└── ...
notebooks/
├── NN_module_name.ipynb   # Interactive Jupyter notebook
```

Each module is prefixed with a two-digit number for ordering (e.g., `45_torch_profiler/`).

## Adding a New Module

1. **Create the directory**: `mkdir NN_topic_name` using the next available number.
2. **Write `README.md`**: Include sections for Overview, Key Concepts, Examples,
   Files in This Module, and References.
3. **Add Python scripts**: Educational, well-documented code with docstrings.
   Target 80-150 lines per script. Each function should demonstrate one concept.
4. **Add a notebook**: Create `notebooks/NN_topic_name.ipynb` with 4-6 cells
   mixing markdown explanations and runnable code.
5. **Update dependencies**: If your examples require new packages, add them
   to `requirements.txt`.

## Code Style

- **Docstrings**: Every module-level and function-level docstring should explain
  *what* and *why*, not just *how*.
- **Self-contained**: Each script should run independently (`python script.py`).
- **Type hints**: Use type annotations for function signatures.
- **No secrets**: Never commit API keys, tokens, or credentials.

## Python Scripts

```python
"""
Module Title
============

Brief description of what this script demonstrates.
"""

import torch

def example_function(x: torch.Tensor) -> torch.Tensor:
    """One-line summary of what this function demonstrates."""
    ...

if __name__ == "__main__":
    example_function(torch.randn(4, 4))
```

## Notebooks

- Use the standard `.ipynb` format with 4-6 cells.
- Start with a markdown cell introducing the topic.
- End with a "Next Steps" markdown cell pointing to related modules.
- Keep outputs cleared before committing.

## Commit Messages

- Use short, descriptive messages: `Add module 45: Torch Profiler deep dive`
- One logical change per commit.
- Do not batch unrelated changes.

## Submitting Changes

1. Fork the repository and create a feature branch.
2. Make your changes following the guidelines above.
3. Test that all Python scripts run without errors.
4. Open a pull request with a clear description of what you added or changed.

## Reporting Issues

Open a GitHub issue for:
- Broken code examples
- Missing or outdated content
- Requests for new modules
- Typos or unclear explanations

## License

By contributing, you agree that your contributions will be licensed under the
same license as this repository.
