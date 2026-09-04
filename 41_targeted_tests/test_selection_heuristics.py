"""
Module 41: Test Selection Heuristics
====================================
Practical heuristics for choosing which PyTorch tests to run for a diff.
Educational stand-alone version of the ideas behind targeted_tests mapping.

Run: python test_selection_heuristics.py
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Mapping tables
# ---------------------------------------------------------------------------
# source prefix → list of (test path, optional -k filter)
SUBMODULE_MAP: dict[str, list[tuple[str, str | None]]] = {
    "torch/nn/": [("test/test_nn.py", None)],
    "torch/optim/": [("test/test_optim.py", None)],
    "torch/cuda/": [("test/test_cuda.py", None)],
    "torch/fx/": [("test/test_fx.py", None)],
    "torch/distributed/": [("test/distributed/", None)],
    "torch/_dynamo/": [("test/dynamo/", None)],
    "torch/_inductor/": [("test/inductor/", None)],
    "torch/export/": [("test/export/", None)],
    "torch/ao/quantization/": [("test/quantization/", None)],
    "aten/src/ATen/native/": [("test/test_ops.py", None), ("test/test_torch.py", None)],
    "aten/src/ATen/": [("test/test_torch.py", "test_type")],
}

FULL_SUITE_TRIGGERS = [
    "setup.py",
    "CMakeLists.txt",
    "torch/__init__.py",
    "torch/csrc/",
    "c10/",
    "caffe2/",
    ".ci/",
    "torch/csrc/jit/",
]

CATEGORY_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("mgpu", ("test/distributed/",)),
    ("sgpu", ("test/test_cuda", "test/test_matmul_cuda")),
    ("inductor", ("test/inductor/", "test/dynamo/", "test/export/")),
    ("cpu", ()),  # default bucket
]


@dataclass(frozen=True)
class TestCommand:
    test_path: str
    keyword: str | None = None
    category: str = "cpu"

    def to_run_test(self) -> str:
        # Normalize to run_test.py -i form
        rel = self.test_path.removeprefix("test/").removesuffix(".py")
        cmd = f"python test/run_test.py -i {rel}"
        if self.keyword:
            cmd += f' -k "{self.keyword}"'
        return cmd


# ---------------------------------------------------------------------------
# Core heuristics
# ---------------------------------------------------------------------------
def triggers_full_suite(changed_files: list[str]) -> bool:
    """Return True if any changed path is too broad for targeted selection."""
    for path in changed_files:
        for trigger in FULL_SUITE_TRIGGERS:
            if path == trigger or path.startswith(trigger):
                return True
    return False


def map_file_to_tests(path: str) -> list[tuple[str, str | None]]:
    """Longest-prefix match against SUBMODULE_MAP."""
    matches: list[tuple[str, list[tuple[str, str | None]]]] = []
    for prefix, tests in SUBMODULE_MAP.items():
        if path.startswith(prefix):
            matches.append((prefix, tests))
    if not matches:
        return []
    matches.sort(key=lambda x: len(x[0]), reverse=True)
    return list(matches[0][1])


def categorize(test_path: str) -> str:
    for category, prefixes in CATEGORY_RULES:
        if category == "cpu":
            continue
        if any(test_path.startswith(p) or p in test_path for p in prefixes):
            return category
    return "cpu"


def select_tests(changed_files: list[str]) -> list[TestCommand] | str:
    """
    Return TestCommand list, or the string 'FULL_SUITE' when heuristics
    cannot safely narrow the run.
    """
    if triggers_full_suite(changed_files):
        return "FULL_SUITE"

    selected: dict[tuple[str, str | None], TestCommand] = {}
    for path in changed_files:
        for test_path, keyword in map_file_to_tests(path):
            key = (test_path, keyword)
            if key not in selected:
                selected[key] = TestCommand(test_path, keyword, categorize(test_path))
    return sorted(selected.values(), key=lambda c: (c.category, c.test_path, c.keyword or ""))


def group_by_category(cmds: list[TestCommand]) -> dict[str, list[TestCommand]]:
    out: dict[str, list[TestCommand]] = {}
    for cmd in cmds:
        out.setdefault(cmd.category, []).append(cmd)
    return out


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------
def print_plan(title: str, files: list[str]) -> None:
    print("=" * 70)
    print(title)
    print("=" * 70)
    print("changed:")
    for f in files:
        print(f"  - {f}")
    result = select_tests(files)
    if result == "FULL_SUITE":
        print("plan: FULL_SUITE (broad infra / runtime change)")
        return
    grouped = group_by_category(result)
    if not grouped:
        print("plan: no mapping — consider adding SUBMODULE_MAP entry or run smoke tests")
        return
    for category, cmds in grouped.items():
        print(f"category [{category}]:")
        for cmd in cmds:
            print(f"  {cmd.to_run_test()}")


def demo():
    print_plan(
        "Case A: nn + optim Python change",
        ["torch/nn/modules/linear.py", "torch/optim/adamw.py"],
    )
    print_plan(
        "Case B: inductor change",
        ["torch/_inductor/scheduler.py", "torch/_dynamo/convert_frame.py"],
    )
    print_plan(
        "Case C: full-suite trigger",
        ["torch/csrc/autograd/engine.cpp"],
    )
    print_plan(
        "Case D: unmapped path",
        ["docs/source/conf.py"],
    )

    print("\n" + "=" * 70)
    print("Heuristic tips")
    print("=" * 70)
    tips = [
        "Union heuristic + structural (TorchTalk) results when available.",
        "Prefer longest-prefix map matches to avoid overly broad tests.",
        "Keep FULL_SUITE_TRIGGERS conservative — false positives cost CI minutes;",
        "  false negatives cost nightlies (acceptable if nightlies are strong).",
        "Always print run_test.py commands for copy-paste local debugging.",
    ]
    for t in tips:
        print(f"  • {t}")


if __name__ == "__main__":
    demo()
    print("\nTest selection heuristic demos finished.")
