# CI failure reproducer

Use this card when a PyTorch job fails remotely and you need the smallest local
command that retains the failure.

## Record the contract

- exact commit SHA and submodule state
- test target and any `-k` expression
- device type, Python version, and relevant environment variables
- test seed, shard, and timeout when the runner exposes them
- first failure rather than all cascade failures

## Reduce in order

1. Run the single test target using `test/run_test.py`.
2. Narrow to a test class or test name only after the full target reproduces.
3. Remove unrelated environment variables one at a time.
4. Replace large input artifacts with deterministic synthetic tensors.
5. Preserve the failing command in the issue or PR description.

A passing ad-hoc Python snippet is not equivalent to a passing CI test: it may
skip distributed initialization, dtype coverage, or runner setup.
