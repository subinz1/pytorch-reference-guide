"""
End-to-end inference server combining all production serving patterns.

Ties together batched inference, dynamic batching, compiled model,
and monitoring into a complete serving example.
"""

import torch
import torch.nn as nn
import time
import threading
from dataclasses import dataclass
from queue import Queue, Empty
from typing import Optional

from batched_inference import pad_sequences, SimpleTransformerEncoder
from monitoring import ModelMetrics, InferenceTimer


@dataclass
class ServerConfig:
    model_dim: int = 256
    vocab_size: int = 10000
    max_batch_size: int = 16
    max_wait_ms: float = 50.0
    max_queue_depth: int = 128
    compile_mode: str = "reduce-overhead"
    warmup_iterations: int = 5
    num_workers: int = 1


@dataclass
class Request:
    input_ids: torch.Tensor
    result: Optional[torch.Tensor] = None
    event: threading.Event = None

    def __post_init__(self):
        if self.event is None:
            self.event = threading.Event()


class InferenceServer:
    """Production inference server with batching, compilation, and monitoring."""

    def __init__(self, config: ServerConfig = ServerConfig()):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics = ModelMetrics(model_name="transformer-server")
        self._queue: Queue[Request] = Queue(maxsize=config.max_queue_depth)
        self._running = False
        self._workers: list[threading.Thread] = []

        self._model = self._build_model()
        self._warmup()

    def _build_model(self) -> nn.Module:
        model = SimpleTransformerEncoder(
            vocab_size=self.config.vocab_size,
            d_model=self.config.model_dim,
        ).to(self.device)
        model.eval()

        compiled = torch.compile(model, mode=self.config.compile_mode)
        return compiled

    def _warmup(self):
        """Pre-fill caches with dummy inputs."""
        print(f"Warming up model on {self.device}...")
        sample_ids = torch.randint(0, self.config.vocab_size, (1, 32), device=self.device)
        sample_mask = torch.ones(1, 32, dtype=torch.bool, device=self.device)

        for _ in range(self.config.warmup_iterations):
            with torch.no_grad():
                _ = self._model(sample_ids, sample_mask)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        print("Warmup complete.")

    def start(self):
        """Start the batch processing workers."""
        self._running = True
        for i in range(self.config.num_workers):
            worker = threading.Thread(target=self._process_loop, name=f"worker-{i}", daemon=True)
            worker.start()
            self._workers.append(worker)
        print(f"Server started with {self.config.num_workers} worker(s)")

    def stop(self):
        """Stop workers gracefully."""
        self._running = False
        for w in self._workers:
            w.join(timeout=5.0)
        self._workers.clear()

    def predict(self, input_ids: torch.Tensor, timeout: float = 5.0) -> Optional[torch.Tensor]:
        """Submit a request and wait for result."""
        if self._queue.qsize() >= self.config.max_queue_depth:
            self.metrics.record_error()
            raise RuntimeError("Server overloaded: queue full")

        request = Request(input_ids=input_ids)
        self._queue.put(request)
        self.metrics.set_queue_depth(self._queue.qsize())

        if request.event.wait(timeout=timeout):
            return request.result
        else:
            self.metrics.record_error()
            return None

    def _process_loop(self):
        """Worker loop: collect batch, run inference, fan out results."""
        while self._running:
            batch = self._collect_batch()
            if not batch:
                continue

            try:
                self._run_batch(batch)
            except Exception as e:
                self.metrics.record_error()
                for req in batch:
                    req.event.set()

    def _collect_batch(self) -> list[Request]:
        """Collect requests up to max_batch_size or max_wait_ms."""
        batch: list[Request] = []
        deadline = time.monotonic() + self.config.max_wait_ms / 1000.0

        try:
            first = self._queue.get(timeout=0.1)
            batch.append(first)
        except Empty:
            return []

        while len(batch) < self.config.max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                req = self._queue.get(timeout=remaining)
                batch.append(req)
            except Empty:
                break

        return batch

    def _run_batch(self, batch: list[Request]):
        """Execute batched inference and distribute results."""
        sequences = [req.input_ids for req in batch]
        padded = pad_sequences(sequences)

        input_ids = padded.input_ids.to(self.device)
        attention_mask = padded.attention_mask.to(self.device)

        with InferenceTimer(self.metrics, phase="inference"):
            with torch.no_grad():
                output = self._model(input_ids, attention_mask)

        output = output.cpu()
        for i, req in enumerate(batch):
            length = padded.original_lengths[i]
            req.result = output[i, :length]
            req.event.set()

        self.metrics.set_queue_depth(self._queue.qsize())


def main():
    config = ServerConfig(
        max_batch_size=8,
        max_wait_ms=30.0,
        warmup_iterations=3,
    )

    server = InferenceServer(config)
    server.start()

    print("\nSending 20 requests with varying sequence lengths...")
    import random
    random.seed(42)

    results = []
    for i in range(20):
        seq_len = random.randint(5, 64)
        input_ids = torch.randint(0, config.vocab_size, (seq_len,))
        result = server.predict(input_ids, timeout=10.0)
        if result is not None:
            results.append((seq_len, result.shape))

    server.stop()

    print(f"\nCompleted {len(results)}/20 requests")
    for i, (input_len, output_shape) in enumerate(results[:5]):
        print(f"  req[{i}]: input_len={input_len}, output={output_shape}")

    print("\n--- Server Metrics ---")
    health = server.metrics.health_check()
    print(f"  Status: {health['status']}")
    print(f"  Total requests: {health['total_requests']}")
    print(f"  Errors: {health['total_errors']}")
    print(f"  Latency p50: {health['latency']['p50_ms']:.2f}ms")
    print(f"  Latency p95: {health['latency']['p95_ms']:.2f}ms")


if __name__ == "__main__":
    main()
