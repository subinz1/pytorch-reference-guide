"""Render targeted test commands as a stable review comment or CI summary.

Run: python selection_result_formatter.py
"""

from __future__ import annotations

from collections import defaultdict

from test_selection_heuristics import select_tests


def format_selection(changed_files: list[str]) -> str:
    selection = select_tests(changed_files)
    if selection == "FULL_SUITE":
        return "## Test plan\n\nFull suite required because a broad trigger changed."
    groups: dict[str, list[str]] = defaultdict(list)
    for command in selection:
        groups[command.category].append(command.to_run_test())
    lines = ["## Test plan"]
    for category in sorted(groups):
        lines.extend([f"\n### {category}", *[f"- `{command}`" for command in groups[category]]])
    return "\n".join(lines)


def main() -> None:
    print(format_selection(["torch/_dynamo/convert_frame.py", "torch/nn/modules/linear.py"]))


if __name__ == "__main__":
    main()
