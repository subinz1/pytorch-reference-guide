"""
Dynamic batching for inference serving.

Accumulates incoming requests over a configurable time window or until
a batch size threshold is reached, then dispatches them as a single
batched forward pass for throughput efficiency.
"""

import torch
import torch.nn as nn
import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class InferenceRequest:
    input_tensor: torch.Tensor
    created_at: float = field(default_factory=time.monotonic)
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())


@dataclass
class BatchConfig:
    max_batch_size: int = 32
    max_wait_ms: float = 50.0
    max_queue_depth: int = 256


class DynamicBatcher:
    """Async dynamic batcher that groups requests for efficient GPU utilization."""

    def __init__(
        self,
        model_fn: Callable[[torch.Tensor], torch.Tensor],
        config: BatchConfig = BatchConfig(),
        device: torch.device = torch.device("cpu"),
    ):
        self.model_fn = model_fn
        self.config = config
        self.device = device
        self._queue: list[InferenceRequest] = []
        self._dispatch_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

        self.stats = BatcherStats()

    async def start(self):
        """Start the dispatch loop."""
        self._dispatch_task = asyncio.create_task(self._dispatch_loop())

    async def stop(self):
        """Stop the dispatch loop and flush remaining requests."""
        if self._dispatch_task:
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass
        if self._queue:
            await self._process_batch(self._queue)
            self._queue.clear()

    async def submit(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Submit a single inference request; returns when batch completes."""
        if len(self._queue) >= self.config.max_queue_depth:
            raise QueueFullError(
                f"Queue depth {len(self._queue)} exceeds limit {self.config.max_queue_depth}"
            )

        request = InferenceRequest(input_tensor=input_tensor)
        async with self._lock:
            self._queue.append(request)

        return await request.future

    async def _dispatch_loop(self):
        """Continuously check if a batch is ready to dispatch."""
        while True:
            await asyncio.sleep(self.config.max_wait_ms / 1000.0)

            async with self._lock:
                if not self._queue:
                    continue
                batch = self._queue[: self.config.max_batch_size]
                self._queue = self._queue[self.config.max_batch_size :]

            await self._process_batch(batch)

    async def _process_batch(self, batch: list[InferenceRequest]):
        """Run model inference on a batch and resolve futures."""
        if not batch:
            return

        self.stats.batches_processed += 1
        self.stats.total_requests += len(batch)
        self.stats.last_batch_size = len(batch)

        try:
            inputs = torch.stack([r.input_tensor for r in batch]).to(self.device)

            with torch.no_grad():
                outputs = self.model_fn(inputs)

            outputs = outputs.cpu()
            for i, request in enumerate(batch):
                if not request.future.done():
                    request.future.set_result(outputs[i])

        except Exception as e:
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(e)


@dataclass
class BatcherStats:
    batches_processed: int = 0
    total_requests: int = 0
    last_batch_size: int = 0

    @property
    def avg_batch_size(self) -> float:
        if self.batches_processed == 0:
            return 0.0
        return self.total_requests / self.batches_processed


class QueueFullError(Exception):
    pass


class SyncDynamicBatcher:
    """Synchronous dynamic batcher for simpler use cases (no asyncio)."""

    def __init__(
        self,
        model_fn: Callable[[torch.Tensor], torch.Tensor],
        max_batch_size: int = 32,
        device: torch.device = torch.device("cpu"),
    ):
        self.model_fn = model_fn
        self.max_batch_size = max_batch_size
        self.device = device

    def process_all(self, requests: list[torch.Tensor]) -> list[torch.Tensor]:
        """Process all requests in optimal batch sizes."""
        results = []
        for start in range(0, len(requests), self.max_batch_size):
            batch_inputs = requests[start : start + self.max_batch_size]
            batch_tensor = torch.stack(batch_inputs).to(self.device)
            with torch.no_grad():
                batch_output = self.model_fn(batch_tensor)
            results.extend(batch_output.cpu().unbind(0))
        return results


def main():
    torch.manual_seed(42)

    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 32),
    )
    model.eval()

    batcher = SyncDynamicBatcher(
        model_fn=model,
        max_batch_size=8,
    )

    requests = [torch.randn(64) for _ in range(25)]
    print(f"Processing {len(requests)} requests with max_batch_size=8")

    results = batcher.process_all(requests)

    print(f"Got {len(results)} results")
    print(f"Batches used: {(len(requests) + 7) // 8}")
    for i in range(min(3, len(results))):
        print(f"  result[{i}] shape: {results[i].shape}, norm: {results[i].norm():.4f}")

    print("\n--- Async batcher config ---")
    config = BatchConfig(max_batch_size=16, max_wait_ms=25.0, max_queue_depth=128)
    print(f"  max_batch_size: {config.max_batch_size}")
    print(f"  max_wait_ms: {config.max_wait_ms}")
    print(f"  max_queue_depth: {config.max_queue_depth}")


if __name__ == "__main__":
    main()
