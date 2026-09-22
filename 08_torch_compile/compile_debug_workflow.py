"""Show a compact workflow for inspecting torch.compile graph breaks.

Run: TORCH_LOGS="graph_breaks,recompiles" python compile_debug_workflow.py
"""

from __future__ import annotations

import torch


def data_dependent_branch(x: torch.Tensor) -> torch.Tensor:
    if x.sum().item() > 0:
        return x.relu()
    return -x


def main() -> None:
    sample = torch.randn(4, 4)
    explanation = torch._dynamo.explain(data_dependent_branch)(sample)
    print(f"graphs captured: {explanation.graph_count}")
    for reason in explanation.break_reasons:
        print(f"graph break: {reason.reason}")

    compiled = torch.compile(data_dependent_branch)
    print(compiled(sample))
    print("Replace data-dependent Python control flow with tensor operations or torch.cond when appropriate.")


if __name__ == "__main__":
    main()
