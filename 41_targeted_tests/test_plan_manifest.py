"""Produce a reviewable targeted-test manifest from changed paths.

Run: python test_plan_manifest.py
"""

from __future__ import annotations

import json

from test_selection_heuristics import select_tests


def build_manifest(changed_files: list[str]) -> dict[str, object]:
    selection = select_tests(changed_files)
    if selection == "FULL_SUITE":
        return {"changed_files": changed_files, "strategy": "full_suite", "commands": []}
    return {
        "changed_files": changed_files,
        "strategy": "targeted",
        "commands": [command.to_run_test() for command in selection],
    }


def main() -> None:
    manifest = build_manifest(["torch/nn/modules/linear.py", "torch/optim/adamw.py"])
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
