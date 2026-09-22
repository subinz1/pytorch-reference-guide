"""Compare eager and compiled outputs over a small, explicit input matrix.

Run: python compile_correctness_harness.py
"""

from __future__ import annotations

import torch


class ResidualMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj_in = torch.nn.Linear(16, 32)
        self.proj_out = torch.nn.Linear(32, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.proj_out(torch.nn.functional.gelu(self.proj_in(x)))


def assert_compiled_matches_eager(model: torch.nn.Module, sample: torch.Tensor) -> None:
    eager = model(sample)
    compiled = torch.compile(model)(sample)
    torch.testing.assert_close(compiled, eager, rtol=1e-4, atol=1e-5)


def main() -> None:
    torch.manual_seed(0)
    model = ResidualMLP().eval()
    for shape in ((1, 16), (4, 16), (9, 16)):
        with torch.no_grad():
            assert_compiled_matches_eager(model, torch.randn(shape))
        print(f"passed shape={shape}")


if __name__ == "__main__":
    main()
