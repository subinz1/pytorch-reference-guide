"""Explain when targeted selection should yield to a broader test strategy.

Run: python risk_budget_selection.py
"""

from __future__ import annotations


def choose_strategy(changed_files: list[str], touched_public_api: bool, release_blocker: bool) -> str:
    if release_blocker or any(path.startswith(("torch/csrc/", "c10/", ".ci/")) for path in changed_files):
        return "full suite: runtime or infrastructure risk is broad"
    if touched_public_api:
        return "targeted suite plus public-API smoke coverage"
    return "targeted suite: verify mapping and inspect uncovered files"


def main() -> None:
    cases = [
        (["torch/nn/modules/linear.py"], False, False),
        (["torch/nn/modules/module.py"], True, False),
        (["torch/csrc/autograd/engine.cpp"], False, False),
    ]
    for files, public_api, blocker in cases:
        print(f"{files}: {choose_strategy(files, public_api, blocker)}")


if __name__ == "__main__":
    main()
