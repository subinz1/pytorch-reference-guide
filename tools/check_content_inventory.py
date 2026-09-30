#!/usr/bin/env python3
"""Validate the stable content inventory for the reference guide."""

from collections import Counter
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
OPERATIONAL_GUIDES = {
    "40_crcr_downstream_ci",
    "41_targeted_tests",
}
EXPECTED_MODULE_NUMBERS = list(range(1, 51))
EXPECTED_REFERENCE_CARDS = 28
EXPECTED_MODULE_SCRIPTS = 150


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def numbered_directories() -> list[Path]:
    return sorted(
        path
        for path in REPO_ROOT.iterdir()
        if path.is_dir()
        and re.fullmatch(r"\d{2}_[a-z0-9_]+", path.name)
        and (path / "README.md").is_file()
    )


def main() -> None:
    modules = [
        path for path in numbered_directories() if path.name not in OPERATIONAL_GUIDES
    ]
    module_numbers = Counter(int(path.name[:2]) for path in modules)
    expected_numbers = Counter(EXPECTED_MODULE_NUMBERS)
    if module_numbers != expected_numbers:
        fail(
            "curriculum module numbers do not match 01 through 50: "
            f"found {sorted(module_numbers.elements())}"
        )

    missing_guides = OPERATIONAL_GUIDES.difference(
        path.name for path in numbered_directories()
    )
    if missing_guides:
        fail(f"missing operational guides: {', '.join(sorted(missing_guides))}")

    notebooks = sorted((REPO_ROOT / "notebooks").glob("*.ipynb"))
    notebook_numbers = [int(notebook.name[:2]) for notebook in notebooks]
    if notebook_numbers != EXPECTED_MODULE_NUMBERS:
        fail("notebook numbers do not match 01 through 50")

    cards = [
        path
        for path in (REPO_ROOT / "docs" / "reference_cards").glob("*.md")
        if path.name != "README.md"
    ]
    if len(cards) != EXPECTED_REFERENCE_CARDS:
        fail(
            f"expected {EXPECTED_REFERENCE_CARDS} reference cards, found {len(cards)}"
        )

    module_scripts = [
        script for module in modules + [REPO_ROOT / guide for guide in OPERATIONAL_GUIDES]
        for script in module.rglob("*.py")
    ]
    if len(module_scripts) != EXPECTED_MODULE_SCRIPTS:
        fail(
            f"expected {EXPECTED_MODULE_SCRIPTS} module scripts, found {len(module_scripts)}"
        )

    print(f"curriculum modules: {len(modules)}")
    print(f"operational guides: {len(OPERATIONAL_GUIDES)}")
    print(f"notebooks: {len(notebooks)}")
    print(f"reference cards: {len(cards)}")
    print(f"module scripts: {len(module_scripts)}")


if __name__ == "__main__":
    main()
