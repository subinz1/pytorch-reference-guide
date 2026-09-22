"""Illustrate the identity used to group scheduled CRCR results by PyTorch SHA.

Run: python nightly_delivery_identity.py
"""

from __future__ import annotations

from dataclasses import dataclass
import re


SHA40 = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class NightlyDelivery:
    backend_repo: str
    pytorch_sha: str
    workflow_run_id: int
    run_attempt: int

    @property
    def delivery_id(self) -> str:
        return self.pytorch_sha

    def validate(self) -> None:
        if not SHA40.fullmatch(self.pytorch_sha):
            raise ValueError("pytorch_sha must be a lowercase 40-character SHA")
        if self.run_attempt < 1:
            raise ValueError("run_attempt must be positive")


def main() -> None:
    delivery = NightlyDelivery("example-org/backend-ci", "a" * 40, 12345, 1)
    delivery.validate()
    print(f"HUD grouping identity: {delivery.delivery_id}")
    print("Use the upstream PyTorch SHA, not the downstream repository commit.")


if __name__ == "__main__":
    main()
