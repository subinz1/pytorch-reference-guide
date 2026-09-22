"""Model safe retry identity for a downstream callback sender.

Run: python callback_retry_idempotency.py
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CallbackIdentity:
    delivery_id: str
    workflow_run_id: int
    job_name: str
    run_attempt: int

    def key(self) -> tuple[str, int, str, int]:
        return (self.delivery_id, self.workflow_run_id, self.job_name, self.run_attempt)


class Sender:
    def __init__(self) -> None:
        self.sent: set[tuple[str, int, str, int]] = set()

    def send_once(self, identity: CallbackIdentity) -> bool:
        if identity.key() in self.sent:
            return False
        self.sent.add(identity.key())
        return True


def main() -> None:
    sender = Sender()
    identity = CallbackIdentity("a" * 40, 12, "nightly / cpu", 1)
    print(f"first delivery accepted: {sender.send_once(identity)}")
    print(f"retry suppressed locally: {not sender.send_once(identity)}")
    print("A real receiver still validates identity and owns final deduplication.")


if __name__ == "__main__":
    main()
