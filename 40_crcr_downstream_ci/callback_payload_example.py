"""
Module 40 (CRCR): Callback Payload Examples
===========================================
Educational examples of JSON payload shapes used when downstream CI
reports results back through the Cross-Repository CI Relay callback.

These are illustrative — field names track the public CRCR callback
action concepts (conclusion, job metadata, timing, OIDC identity).
Adapt to the current pytorch/test-infra action if schemas evolve.

Run: python callback_payload_example.py
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Payload building blocks
# ---------------------------------------------------------------------------
@dataclass
class TimingInfo:
    """Wall-clock timing for a downstream job."""

    started_at: str
    finished_at: str
    duration_seconds: float


@dataclass
class JobIdentity:
    """Who ran the job and which upstream PR/SHA it tested."""

    repository: str
    workflow: str
    job_name: str
    run_id: int
    run_attempt: int
    html_url: str
    upstream_pr: int | None
    upstream_head_sha: str
    upstream_base_sha: str | None = None


@dataclass
class CRCRCallbackPayload:
    """
    Top-level callback body posted to the CRCR Lambda after OIDC auth.

    conclusion: success | failure | cancelled | skipped | neutral
    level: allowlist level (L2/L3/L4) when known
    """

    schema_version: str
    conclusion: str
    identity: JobIdentity
    timing: TimingInfo
    level: str = "L2"
    backend: str = "example-backend"
    test_summary: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


def iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Example factories
# ---------------------------------------------------------------------------
def example_success_payload() -> CRCRCallbackPayload:
    start = "2026-09-04T10:00:00Z"
    end = "2026-09-04T10:42:15Z"
    return CRCRCallbackPayload(
        schema_version="1.0",
        conclusion="success",
        level="L2",
        backend="xpu-ops",
        identity=JobIdentity(
            repository="org/downstream-ci",
            workflow="crcr-ci.yml",
            job_name="xpu-smoke / linux",
            run_id=123456789,
            run_attempt=1,
            html_url="https://github.com/org/downstream-ci/actions/runs/123456789",
            upstream_pr=160001,
            upstream_head_sha="a" * 40,
            upstream_base_sha="b" * 40,
        ),
        timing=TimingInfo(start, end, duration_seconds=2535.0),
        test_summary={
            "total": 420,
            "passed": 418,
            "failed": 0,
            "skipped": 2,
            "category": "cpu",
        },
        extras={"pytorch_build": "source", "cuda": None, "dtype_smoke": "float32"},
    )


def example_failure_payload() -> CRCRCallbackPayload:
    payload = example_success_payload()
    payload.conclusion = "failure"
    payload.timing = TimingInfo(
        "2026-09-04T11:00:00Z",
        "2026-09-04T11:18:02Z",
        duration_seconds=1082.0,
    )
    payload.test_summary = {
        "total": 420,
        "passed": 401,
        "failed": 3,
        "skipped": 16,
        "failed_tests": [
            "test_nn.py::test_linear_xpu",
            "test_ops.py::test_add_broadcast",
            "inductor/test_smoke.py::test_compile_basic",
        ],
    }
    payload.extras["error_fingerprint"] = "RuntimeError: native backend mismatch"
    return payload


def example_nightly_payload() -> CRCRCallbackPayload:
    """Nightly jobs may omit upstream_pr and report a main SHA only."""
    payload = example_success_payload()
    payload.identity.upstream_pr = None
    payload.identity.job_name = "nightly / full-suite"
    payload.identity.upstream_head_sha = "c" * 40
    payload.extras["trigger"] = "schedule"
    payload.extras["suite"] = "nightly-full"
    return payload


def to_json(payload: CRCRCallbackPayload) -> str:
    """Serialize nested dataclasses to JSON."""
    return json.dumps(asdict(payload), indent=2)


# ---------------------------------------------------------------------------
# Validation helpers (educational)
# ---------------------------------------------------------------------------
REQUIRED_TOP_LEVEL = ("schema_version", "conclusion", "identity", "timing")
VALID_CONCLUSIONS = {"success", "failure", "cancelled", "skipped", "neutral"}


def validate_payload_dict(data: dict[str, Any]) -> list[str]:
    """Return a list of human-readable problems (empty if OK)."""
    problems: list[str] = []
    for key in REQUIRED_TOP_LEVEL:
        if key not in data:
            problems.append(f"missing top-level key: {key}")
    if data.get("conclusion") not in VALID_CONCLUSIONS:
        problems.append(f"invalid conclusion: {data.get('conclusion')!r}")
    identity = data.get("identity") or {}
    for key in ("repository", "job_name", "upstream_head_sha", "html_url"):
        if not identity.get(key):
            problems.append(f"identity.{key} is required")
    sha = identity.get("upstream_head_sha", "")
    if sha and len(sha) != 40:
        problems.append("upstream_head_sha should be a 40-char hex SHA")
    timing = data.get("timing") or {}
    if timing.get("duration_seconds", -1) < 0:
        problems.append("timing.duration_seconds must be >= 0")
    return problems


def demo():
    print("=" * 70)
    print("CRCR callback payload — success")
    print("=" * 70)
    success = example_success_payload()
    print(to_json(success))

    print("\n" + "=" * 70)
    print("Validation")
    print("=" * 70)
    for label, payload in [
        ("success", success),
        ("failure", example_failure_payload()),
        ("nightly", example_nightly_payload()),
    ]:
        data = asdict(payload)
        issues = validate_payload_dict(data)
        status = "OK" if not issues else "; ".join(issues)
        print(f"  {label}: {status}")

    print("\n" + "=" * 70)
    print("Minimal GitHub Actions sketch")
    print("=" * 70)
    print(
        """
      - name: Report to CRCR
        if: always()
        uses: pytorch/test-infra/.github/actions/cross-repo-ci-relay-callback@main
        with:
          conclusion: ${{ job.status }}
        # Action mints OIDC JWT, builds payload, POSTs to CRCR Lambda
""".rstrip()
    )


if __name__ == "__main__":
    demo()
    print("\nPayload examples finished.")
