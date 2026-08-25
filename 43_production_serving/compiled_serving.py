"""
torch.compile + CUDA Graphs for low-latency inference serving.

Demonstrates how to combine torch.compile with CUDA graph capture
to eliminate kernel launch overhead for fixed-shape workloads.
"""

import torch
import torch.nn as nn
import time
from contextlib import contextmanager
from typing import Optional


class CompiledModelServer:
    """Wraps a model with torch.compile and optional CUDA graph capture."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        compile_mode: str = "reduce-overhead",
        use_cuda_graphs: bool = True,
        warmup_iterations: int = 10,
    ):
        self.device = device
        self.use_cuda_graphs = use_cuda_graphs and device.type == "cuda"
        self.warmup_iterations = warmup_iterations
        self._is_warmed_up = False

        model = model.to(device).eval()

        self.compiled_model = torch.compile(model, mode=compile_mode)

        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._static_input: Optional[torch.Tensor] = None
        self._static_output: Optional[torch.Tensor] = None

    def warmup(self, sample_input: torch.Tensor):
        """Warm up the compiled model and optionally capture CUDA graph."""
        sample_input = sample_input.to(self.device)

        print(f"Warming up with {self.warmup_iterations} iterations...")
        for _ in range(self.warmup_iterations):
            with torch.no_grad():
                _ = self.compiled_model(sample_input)

        if self.use_cuda_graphs:
            self._capture_cuda_graph(sample_input)

        self._is_warmed_up = True
        print("Warmup complete.")

    def _capture_cuda_graph(self, sample_input: torch.Tensor):
        """Capture the forward pass as a CUDA graph for replay."""
        self._static_input = sample_input.clone()
        self._graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(self._graph):
            self._static_output = self.compiled_model(self._static_input)

        print(f"CUDA graph captured for input shape {sample_input.shape}")

    def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run inference using the optimal path (CUDA graph or compiled)."""
        input_tensor = input_tensor.to(self.device)

        if self._graph is not None and input_tensor.shape == self._static_input.shape:
            self._static_input.copy_(input_tensor)
            self._graph.replay()
            return self._static_output.clone()

        with torch.no_grad():
            return self.compiled_model(input_tensor)

    @property
    def is_warmed_up(self) -> bool:
        return self._is_warmed_up


class MultiShapeServer:
    """Handles multiple input shapes by maintaining separate CUDA graphs."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        bucket_sizes: list[int],
        compile_mode: str = "reduce-overhead",
    ):
        self.device = device
        self.bucket_sizes = sorted(bucket_sizes)

        model = model.to(device).eval()
        self.compiled_model = torch.compile(model, mode=compile_mode)

        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._static_inputs: dict[int, torch.Tensor] = {}
        self._static_outputs: dict[int, torch.Tensor] = {}

    def warmup(self, feature_dim: int, batch_size: int = 1, iterations: int = 5):
        """Warm up and capture graphs for each bucket size."""
        for size in self.bucket_sizes:
            sample = torch.randn(batch_size, size, feature_dim, device=self.device)
            for _ in range(iterations):
                with torch.no_grad():
                    _ = self.compiled_model(sample)

            if self.device.type == "cuda":
                self._static_inputs[size] = sample.clone()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self._static_outputs[size] = self.compiled_model(self._static_inputs[size])
                self._graphs[size] = graph

        print(f"Warmed up {len(self.bucket_sizes)} bucket sizes: {self.bucket_sizes}")

    def _find_bucket(self, seq_len: int) -> int:
        """Find the smallest bucket that fits the sequence length."""
        for size in self.bucket_sizes:
            if size >= seq_len:
                return size
        return self.bucket_sizes[-1]

    def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run inference, padding to nearest bucket size if using CUDA graphs."""
        input_tensor = input_tensor.to(self.device)
        seq_len = input_tensor.size(1)
        bucket = self._find_bucket(seq_len)

        if bucket in self._graphs:
            padded = torch.zeros_like(self._static_inputs[bucket])
            padded[:, :seq_len] = input_tensor[:, :seq_len]
            self._static_inputs[bucket].copy_(padded)
            self._graphs[bucket].replay()
            return self._static_outputs[bucket][:, :seq_len].clone()

        with torch.no_grad():
            return self.compiled_model(input_tensor)


@contextmanager
def inference_mode_context():
    """Context manager combining inference mode with optimal settings."""
    prev_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    with torch.inference_mode():
        yield
    torch.set_float32_matmul_precision(prev_precision)


def benchmark_latency(
    fn,
    input_tensor: torch.Tensor,
    n_iterations: int = 100,
    warmup: int = 10,
) -> dict[str, float]:
    """Benchmark inference latency with warmup."""
    for _ in range(warmup):
        fn(input_tensor)

    if input_tensor.is_cuda:
        torch.cuda.synchronize()

    latencies = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        fn(input_tensor)
        if input_tensor.is_cuda:
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

    latencies.sort()
    return {
        "p50_ms": latencies[len(latencies) // 2],
        "p95_ms": latencies[int(len(latencies) * 0.95)],
        "p99_ms": latencies[int(len(latencies) * 0.99)],
        "mean_ms": sum(latencies) / len(latencies),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.GELU(),
        nn.Linear(256, 256),
        nn.GELU(),
        nn.Linear(256, 64),
    )

    server = CompiledModelServer(
        model=model,
        device=device,
        compile_mode="reduce-overhead",
        use_cuda_graphs=(device.type == "cuda"),
    )

    sample = torch.randn(1, 128)
    server.warmup(sample)

    with inference_mode_context():
        result = server.predict(torch.randn(1, 128))
        print(f"Prediction shape: {result.shape}")

        stats = benchmark_latency(
            server.predict,
            torch.randn(1, 128, device=device),
            n_iterations=50,
        )
        print(f"\nLatency stats (n=50):")
        for k, v in stats.items():
            print(f"  {k}: {v:.3f}")

    print(f"\n--- Multi-shape server ---")
    multi_server = MultiShapeServer(
        model=nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True),
            num_layers=2,
        ),
        device=device,
        bucket_sizes=[32, 64, 128, 256],
    )
    multi_server.warmup(feature_dim=64)

    test_input = torch.randn(1, 50, 64, device=device)
    output = multi_server.predict(test_input)
    print(f"Input seq_len=50, bucketed output shape: {output.shape}")


if __name__ == "__main__":
    main()
