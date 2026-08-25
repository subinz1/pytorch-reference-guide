"""
Monitoring and metrics for model serving.

Provides latency histograms, throughput counters, queue depth tracking,
and health check endpoints for production inference servers.
"""

import torch
import time
import threading
from dataclasses import dataclass, field
from collections import deque
from typing import Optional


@dataclass
class LatencyHistogram:
    """Fixed-bucket histogram for latency tracking."""
    buckets_ms: list[float] = field(
        default_factory=lambda: [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 5000]
    )
    counts: list[int] = field(default_factory=list)
    total: int = 0
    sum_ms: float = 0.0

    def __post_init__(self):
        if not self.counts:
            self.counts = [0] * (len(self.buckets_ms) + 1)

    def observe(self, latency_ms: float):
        self.total += 1
        self.sum_ms += latency_ms
        for i, boundary in enumerate(self.buckets_ms):
            if latency_ms <= boundary:
                self.counts[i] += 1
                return
        self.counts[-1] += 1

    @property
    def mean_ms(self) -> float:
        return self.sum_ms / self.total if self.total > 0 else 0.0

    def percentile(self, p: float) -> float:
        """Approximate percentile from histogram buckets."""
        target = self.total * p / 100.0
        cumulative = 0
        for i, count in enumerate(self.counts):
            cumulative += count
            if cumulative >= target:
                if i < len(self.buckets_ms):
                    return self.buckets_ms[i]
                return self.buckets_ms[-1] * 2
        return 0.0

    def summary(self) -> dict[str, float]:
        return {
            "count": self.total,
            "mean_ms": self.mean_ms,
            "p50_ms": self.percentile(50),
            "p95_ms": self.percentile(95),
            "p99_ms": self.percentile(99),
        }


@dataclass
class ThroughputCounter:
    """Sliding-window throughput counter."""
    window_seconds: float = 60.0
    _timestamps: deque = field(default_factory=deque)

    def record(self):
        now = time.monotonic()
        self._timestamps.append(now)
        self._prune(now)

    def _prune(self, now: float):
        cutoff = now - self.window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    @property
    def requests_per_second(self) -> float:
        now = time.monotonic()
        self._prune(now)
        count = len(self._timestamps)
        return count / self.window_seconds if count > 0 else 0.0

    @property
    def count_in_window(self) -> int:
        self._prune(time.monotonic())
        return len(self._timestamps)


class ModelMetrics:
    """Centralized metrics collection for a serving model."""

    def __init__(self, model_name: str = "default"):
        self.model_name = model_name
        self.inference_latency = LatencyHistogram()
        self.preprocessing_latency = LatencyHistogram()
        self.throughput = ThroughputCounter()
        self.error_count = 0
        self.queue_depth = 0
        self._lock = threading.Lock()
        self._start_time = time.monotonic()

    def record_inference(self, latency_ms: float):
        with self._lock:
            self.inference_latency.observe(latency_ms)
            self.throughput.record()

    def record_preprocessing(self, latency_ms: float):
        with self._lock:
            self.preprocessing_latency.observe(latency_ms)

    def record_error(self):
        with self._lock:
            self.error_count += 1

    def set_queue_depth(self, depth: int):
        self.queue_depth = depth

    def health_check(self) -> dict:
        """Return health status suitable for a /health endpoint."""
        with self._lock:
            uptime = time.monotonic() - self._start_time
            rps = self.throughput.requests_per_second
            return {
                "status": "healthy" if self.error_count == 0 or rps > 0 else "degraded",
                "model": self.model_name,
                "uptime_seconds": round(uptime, 1),
                "requests_per_second": round(rps, 2),
                "total_requests": self.inference_latency.total,
                "total_errors": self.error_count,
                "queue_depth": self.queue_depth,
                "latency": self.inference_latency.summary(),
            }

    def prometheus_metrics(self) -> str:
        """Format metrics in Prometheus exposition format."""
        lines = []
        name = self.model_name.replace("-", "_")

        lines.append(f"# HELP {name}_requests_total Total inference requests")
        lines.append(f"# TYPE {name}_requests_total counter")
        lines.append(f"{name}_requests_total {self.inference_latency.total}")

        lines.append(f"# HELP {name}_errors_total Total errors")
        lines.append(f"# TYPE {name}_errors_total counter")
        lines.append(f"{name}_errors_total {self.error_count}")

        lines.append(f"# HELP {name}_latency_ms Inference latency")
        lines.append(f"# TYPE {name}_latency_ms summary")
        summary = self.inference_latency.summary()
        lines.append(f'{name}_latency_ms{{quantile="0.5"}} {summary["p50_ms"]:.2f}')
        lines.append(f'{name}_latency_ms{{quantile="0.95"}} {summary["p95_ms"]:.2f}')
        lines.append(f'{name}_latency_ms{{quantile="0.99"}} {summary["p99_ms"]:.2f}')
        lines.append(f"{name}_latency_ms_sum {summary['mean_ms'] * summary['count']:.2f}")
        lines.append(f"{name}_latency_ms_count {int(summary['count'])}")

        lines.append(f"# HELP {name}_queue_depth Current queue depth")
        lines.append(f"# TYPE {name}_queue_depth gauge")
        lines.append(f"{name}_queue_depth {self.queue_depth}")

        return "\n".join(lines)


class InferenceTimer:
    """Context manager for timing inference and recording to metrics."""

    def __init__(self, metrics: ModelMetrics, phase: str = "inference"):
        self.metrics = metrics
        self.phase = phase
        self._start: float = 0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        if self.phase == "inference":
            self.metrics.record_inference(elapsed_ms)
        elif self.phase == "preprocessing":
            self.metrics.record_preprocessing(elapsed_ms)


def main():
    metrics = ModelMetrics(model_name="text-encoder")

    model = torch.nn.Linear(64, 32)
    model.eval()

    print("Simulating 100 inference requests...\n")
    for i in range(100):
        with InferenceTimer(metrics, phase="inference"):
            with torch.no_grad():
                _ = model(torch.randn(1, 64))
            time.sleep(0.001 * (i % 10))

    health = metrics.health_check()
    print("Health Check:")
    for k, v in health.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")

    print(f"\nPrometheus Metrics:\n{metrics.prometheus_metrics()}")


if __name__ == "__main__":
    main()
