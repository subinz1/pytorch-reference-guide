# Targeted Test Selection

Running the full PyTorch test suite takes hours. **Targeted test
selection** identifies which tests are affected by a code change and
runs only those, reducing CI time from hours to minutes.

## Approaches

### 1. File-Path Heuristic (targeted_tests.py)

Maps changed source files to test files using submodule-level rules:

```python
SUBMODULE_MAP = {
    "torch/nn/":        [("test/test_nn.py", None)],
    "torch/optim/":     [("test/test_optim.py", None)],
    "torch/cuda/":      [("test/test_cuda.py", None)],
    "torch/fx/":        [("test/test_fx.py", None)],
    "aten/src/ATen/":   [("test/test_torch.py", "test_type")],
}
```

**Usage:**
```bash
python targeted_tests.py OLD_SHA NEW_SHA --pytorch-dir /pytorch --category cpu --commands-only
# Output:
# python test/run_test.py -i test_nn
# python test/run_test.py -i test_optim -k "test_adam"
```

### 2. Structural Analysis (TorchTalk)

For C++ changes, file-path heuristics miss transitive callers.
TorchTalk uses libclang to:
1. Parse `compile_commands.json` for accurate compilation flags
2. Extract changed C++ symbols
3. Walk the call graph to find all callers
4. Map callers through pybind11/TORCH_LIBRARY bindings
5. Resolve to Python test files and classes

```bash
python torchtalk_tests.py OLD_SHA NEW_SHA --pytorch-dir /pytorch --commands-only
```

### 3. Unified Merger

Combines both approaches by taking their **union**:

```bash
python merge_test_results.py OLD_SHA NEW_SHA --pytorch-dir /pytorch --category cpu --commands-only
```

If TorchTalk is unavailable, gracefully falls back to heuristic only.

## Test Categories

Tests are classified into 4 categories for parallel execution:

| Category | Prefix patterns | Runner requirement |
|----------|----------------|-------------------|
| cpu | `test/test_*.py` (default) | CPU only |
| inductor | `test/inductor/`, `test/dynamo/`, `test/export/` | CPU (some GPU) |
| sgpu | `test/test_cuda*` | 1 GPU |
| mgpu | `test/distributed/` | 2+ GPUs |

## run_test.py Format

PyTorch's test runner (`test/run_test.py`) handles path resolution,
environment setup, and timeout management:

```bash
# Run entire test file
python test/run_test.py -i test_torch

# Run with keyword filter
python test/run_test.py -i test_nn -k "test_linear or test_conv2d"

# Subdirectory tests use / separator
python test/run_test.py -i nn/test_multihead_attention

# Inductor tests
python test/run_test.py -i inductor/test_torchinductor
```

## Full-Suite Triggers

Some changes are too broad for targeted selection and trigger the
full test suite:

```python
FULL_SUITE_TRIGGERS = [
    "setup.py",
    "CMakeLists.txt",
    "torch/__init__.py",
    "torch/csrc/",        # Core C++ runtime
    "c10/",               # Core library
    "caffe2/",            # Legacy but affects build
    ".ci/",               # CI infrastructure
]
```

## CI Integration Example

```yaml
determine-tests:
  steps:
    - name: Resolve tests
      run: |
        for cat in cpu inductor sgpu mgpu; do
          CMDS=$(python merge_test_results.py $PREV_SHA $HEAD_SHA \
            --pytorch-dir /pytorch --category $cat --commands-only)
          echo "${cat}_tests=$(echo "$CMDS" | base64 -w0)" >> $GITHUB_OUTPUT
        done

cpu-tests:
  needs: determine-tests
  steps:
    - name: Run
      run: |
        COMMANDS=$(echo "${{ needs.determine-tests.outputs.cpu_tests }}" | base64 -d)
        while IFS= read -r cmd; do
          eval "$cmd"
        done <<< "$COMMANDS"
```

## Key Repositories

- [pytorch-targeted-tests](https://github.com/subinz1/pytorch-targeted-tests) — Heuristic engine
- [TorchTalk](https://github.com/TorchedHat/torchtalk) — Structural C++ analysis
- [pytorch-redhat-ci](https://github.com/TorchedHat/pytorch-redhat-ci) — Integration example

## Performance Metrics

| Approach | Avg. tests selected | Time saved vs full suite |
|----------|-------------------|-------------------------|
| File-path heuristic | ~5-15% of suite | 70-85% wall time |
| Structural (TorchTalk) | ~3-10% of suite | 80-90% wall time |
| Merged (union) | ~8-20% of suite | 65-80% wall time |
| Full suite | 100% | Baseline (~4-6 hours) |

## Extending the Mapping

To add a new test mapping for a source directory:

```python
# In targeted_tests.py, add to SUBMODULE_MAP:
SUBMODULE_MAP = {
    # existing mappings...
    "torch/my_feature/": [
        ("test/test_my_feature.py", None),        # run entire file
        ("test/test_related.py", "test_my_func"),  # run specific test
    ],
}
```

The tuple format is `(test_file, optional_keyword_filter)`. When the keyword
is `None`, the entire test file runs. Otherwise, it becomes a `-k` filter
passed to `run_test.py`.

## FAQ

**Q: What if targeted tests miss a regression?**
Nightly full-suite CI catches regressions that slip through targeted selection.
The merge of heuristic + structural analysis minimizes false negatives.

**Q: Can I run targeted tests locally?**
Yes: `python targeted_tests.py HEAD~1 HEAD --pytorch-dir . --category cpu --commands-only`
prints the commands you can paste into your terminal.

**Q: How does TorchTalk handle header-only changes?**
Header changes in `c10/` or `aten/src/ATen/core/` trigger the full suite
via `FULL_SUITE_TRIGGERS`, since their call graph is too broad to scope.
