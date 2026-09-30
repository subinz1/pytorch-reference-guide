#!/usr/bin/env python3
"""Validate notebook metadata and clean execution state."""

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
EXPECTED_NUMBERS = list(range(1, 51))


def validate_notebook(path: Path) -> list[str]:
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return [f"invalid JSON: {error}"]

    errors = []
    if notebook.get("nbformat") != 4:
        errors.append("nbformat must be 4")
    notebook_minor_version = notebook.get("nbformat_minor")
    if (
        not isinstance(notebook_minor_version, int)
        or notebook_minor_version < 4
    ):
        errors.append("nbformat_minor must be at least 4")
    requires_cell_ids = (
        isinstance(notebook_minor_version, int) and notebook_minor_version >= 5
    )

    metadata = notebook.get("metadata")
    if not isinstance(metadata, dict):
        errors.append("metadata must be an object")
        return errors
    if metadata.get("kernelspec", {}).get("name") != "python3":
        errors.append("kernelspec.name must be python3")
    if metadata.get("language_info", {}).get("name") != "python":
        errors.append("language_info.name must be python")

    cells = notebook.get("cells")
    if not isinstance(cells, list) or not cells:
        errors.append("cells must be a non-empty list")
        return errors
    if cells[0].get("cell_type") != "markdown":
        errors.append("first cell must be markdown")
    if not any(cell.get("cell_type") == "code" for cell in cells):
        errors.append("must contain at least one code cell")

    for index, cell in enumerate(cells, start=1):
        if requires_cell_ids and not cell.get("id"):
            errors.append(f"cell {index} is missing an id")
        if cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None:
            errors.append(f"code cell {index} has an execution count")
        if cell.get("outputs"):
            errors.append(f"code cell {index} has committed output")
    return errors


def notebook_paths(path: Path | None) -> list[Path]:
    if path is None:
        notebooks = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
        numbers = [int(notebook.name[:2]) for notebook in notebooks]
        if numbers != EXPECTED_NUMBERS:
            print("error: notebook numbers do not match 01 through 50", file=sys.stderr)
            raise SystemExit(2)
        return notebooks

    candidate = path.resolve()
    if not candidate.is_file():
        print(f"error: notebook does not exist: {candidate}", file=sys.stderr)
        raise SystemExit(2)
    try:
        candidate.relative_to(NOTEBOOKS_DIR)
    except ValueError:
        print("error: notebook must be inside notebooks/", file=sys.stderr)
        raise SystemExit(2)
    return [candidate]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path)
    args = parser.parse_args()

    failures = []
    for path in notebook_paths(args.path):
        errors = validate_notebook(path)
        if errors:
            failures.append(path)
            for error in errors:
                print(f"error: {path.relative_to(REPO_ROOT)}: {error}", file=sys.stderr)
        else:
            print(f"PASS {path.relative_to(REPO_ROOT)}")

    if failures:
        print(f"{len(failures)} notebook(s) failed validation", file=sys.stderr)
        raise SystemExit(1)
    print("Notebook metadata and clean state are valid")


if __name__ == "__main__":
    main()
