"""Validate a minimal, sanitized callback payload before sending it.

Run: python local_callback_validation.py
"""

from __future__ import annotations

from typing import Any


TERMINAL_CONCLUSIONS = {"success", "failure", "cancelled", "skipped", "neutral", "timed_out"}


def validate(payload: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    for field in ("delivery_id", "job_name", "status", "workflow_run_url"):
        if not payload.get(field):
            problems.append(f"missing {field}")
    if payload.get("status") == "completed" and payload.get("conclusion") not in TERMINAL_CONCLUSIONS:
        problems.append("completed callbacks need a recognized conclusion")
    if payload.get("event_type") not in {"pull_request", "nightly", "periodic"}:
        problems.append("event_type is not recognized")
    return problems


def main() -> None:
    payload = {
        "delivery_id": "a" * 40,
        "job_name": "nightly / cpu",
        "status": "completed",
        "conclusion": "success",
        "event_type": "nightly",
        "workflow_run_url": "https://github.com/example/backend/actions/runs/1",
    }
    print("payload valid" if not validate(payload) else validate(payload))


if __name__ == "__main__":
    main()
