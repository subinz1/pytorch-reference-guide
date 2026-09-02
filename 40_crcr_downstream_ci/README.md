# Cross-Repository CI Relay (CRCR)

PyTorch's **Cross-Repository CI Relay** enables downstream backends
(Intel XPU, AMD ROCm, Apple MPS, Red Hat RHEL, etc.) to run their own
CI against upstream PyTorch PRs and report results back to the
[PyTorch HUD](https://hud.pytorch.org).

## Architecture

```
pytorch/pytorch (PR merged or updated)
       │
       ▼  repository_dispatch
downstream/backend-ci
       │
       ├─ Build PyTorch from source at dispatched SHA
       ├─ Run backend-specific tests
       └─ POST callback → CRCR Lambda → ClickHouse → HUD
```

## Registration (Allowlist)

Downstream repos register in `pytorch/pytorch/.github/allowlist.yml`:

```yaml
L2:
  - intel/torch-xpu-ops
  - TorchedHat/pytorch-redhat-ci
L3:
  - some-org/experimental-backend
```

**Levels:**
- **L1**: Receive dispatches, no HUD reporting
- **L2**: Report to HUD, non-blocking
- **L3**: Report to HUD, visible but non-blocking with distinct treatment
- **L4**: Report to HUD, blocking (viable/strict)

## Receiving Dispatches

```yaml
# .github/workflows/crcr-ci.yml
on:
  repository_dispatch:
    types: [pull_request]

jobs:
  test:
    runs-on: self-hosted
    steps:
      - name: Get PR info
        run: |
          PR_NUM="${{ github.event.client_payload.pr_number }}"
          SHA="${{ github.event.client_payload.head_sha }}"
          echo "Testing PR #${PR_NUM} at ${SHA}"

      - name: Build PyTorch
        run: |
          git clone https://github.com/pytorch/pytorch --depth 1
          cd pytorch && git fetch origin ${SHA} && git checkout ${SHA}
          git submodule update --init --recursive
          pip install -e . -v --no-build-isolation
```

## Reporting Results (Callback Action)

```yaml
      - name: Report to CRCR
        if: always()
        uses: pytorch/test-infra/.github/actions/cross-repo-ci-relay-callback@main
        with:
          conclusion: ${{ job.status }}
```

The callback action:
1. Mints an OIDC token (proves identity)
2. Builds a JSON payload (job name, conclusion, URL, timing)
3. POSTs to the CRCR Lambda endpoint
4. Lambda validates JWT, writes to ClickHouse

## Nightly & Periodic CI

For nightly builds, downstream repos use `schedule` triggers and
build from the `pytorch/pytorch` nightly branch:

```yaml
on:
  schedule:
    - cron: "0 4 * * *"

jobs:
  nightly:
    steps:
      - name: Get nightly SHA
        run: |
          # Nightly commits reference the source main SHA in their message
          NIGHTLY_SHA=$(curl -fsSL \
            "https://api.github.com/repos/pytorch/pytorch/commits?sha=nightly&per_page=1" \
            | jq -r '.[0].sha')
          COMMIT_MSG=$(curl -fsSL \
            "https://api.github.com/repos/pytorch/pytorch/commits/${NIGHTLY_SHA}" \
            | jq -r '.commit.message')
          SOURCE_SHA=$(echo "$COMMIT_MSG" | grep -oP '\(([a-f0-9]{40})\)' | tr -d '()')
          echo "Building from main@${SOURCE_SHA}"
```

## HUD Integration

CRCR results appear on:
- **PR pages**: As a distinct "CRCR" section showing downstream results
- **Commit pages**: Same CRCR section for commits with associated PRs
- **CRCR Summary page**: Aggregated health metrics across all backends
- **Main HUD grid**: CRCR columns alongside in-tree CI (grouped by level)

## Quick Start: Adding a New Downstream Backend

1. **Fork the template**: Start from the
   [CRCR starter workflow](https://github.com/pytorch/test-infra/tree/main/.github/actions/cross-repo-ci-relay-callback).
2. **Register**: Open a PR to add your repo to `.github/allowlist.yml` under L2.
3. **Configure dispatch handler**: Add a `repository_dispatch` workflow that
   builds PyTorch from the dispatched SHA and runs your backend tests.
4. **Add the callback step**: Include `pytorch/test-infra/.github/actions/cross-repo-ci-relay-callback@main`
   as the final step with `if: always()`.
5. **Verify on HUD**: After merge, trigger a test dispatch and check
   [hud.pytorch.org](https://hud.pytorch.org) for your results.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Dispatch never received | Repo not in allowlist | Add to `.github/allowlist.yml` |
| OIDC token mint fails | Missing `id-token: write` permission | Add `permissions: id-token: write` to workflow |
| Results not on HUD | Callback URL wrong or Lambda down | Check callback action logs; verify endpoint |
| Build fails at dispatched SHA | Submodules out of sync | Run `git submodule update --init --recursive` |
| Nightly SHA resolution fails | Commit message format changed | Update grep pattern for SHA extraction |

## Environment Variables

The dispatch payload sets these environment variables for your workflow:

| Variable | Description |
|----------|-------------|
| `github.event.client_payload.pr_number` | PR number that triggered the dispatch |
| `github.event.client_payload.head_sha` | Git SHA to build and test against |
| `github.event.client_payload.base_sha` | Base branch SHA for diff context |
| `github.event.client_payload.sender` | GitHub user who authored the PR |

## Key Resources

- [CRCR Blog Post](https://pytorch.org/blog/introducing-cross-repository-ci-relay-scalable-ci-for-pytorchs-out-of-tree-backends/)
- [Lambda Source](https://github.com/pytorch/test-infra/tree/main/aws/lambda/cross_repo_ci_relay)
- [Callback Action](https://github.com/pytorch/test-infra/tree/main/.github/actions/cross-repo-ci-relay-callback)
- [RFC 98: Nightly & Periodic](https://github.com/pytorch/rfcs/pull/98)
- [Allowlist schema](https://github.com/pytorch/pytorch/blob/main/.github/allowlist.yml)
