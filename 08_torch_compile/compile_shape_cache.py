"""Count compilations for a controlled shape sequence.

Run: python compile_shape_cache.py
"""

from __future__ import annotations

import torch


class CompilationCounter:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self, graph_module: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
        self.count += 1
        return graph_module


def run(dynamic: bool) -> int:
    counter = CompilationCounter()
    compiled = torch.compile(lambda x: x.sin() + x.cos(), backend=counter, dynamic=dynamic)
    for batch_size in (2, 4, 8, 2, 4):
        compiled(torch.randn(batch_size, 16))
    return counter.count


def main() -> None:
    torch._dynamo.reset()
    static_count = run(dynamic=False)
    torch._dynamo.reset()
    dynamic_count = run(dynamic=True)
    print(f"static graphs: {static_count}")
    print(f"dynamic graphs: {dynamic_count}")


if __name__ == "__main__":
    main()
