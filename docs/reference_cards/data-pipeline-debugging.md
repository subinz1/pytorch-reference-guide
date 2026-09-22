# Data-pipeline debugging

Use this card when training accelerators are idle, batches are malformed, or a
DataLoader hangs only with workers enabled.

## Narrow the fault

| Experiment | Interpretation |
|---|---|
| `num_workers=0` passes | worker serialization or worker-side state is suspect |
| fixed synthetic dataset passes | data decoding/transforms are suspect |
| no pinning improves stability | host-to-device transfer path is suspect |
| one batch repeats | sampler/epoch seeding is suspect |

## Safe progression

1. Assert batch shapes, dtypes, and devices at the model boundary.
2. Seed the sampler and worker initialization path when reproducibility matters.
3. Increase workers only after a single-worker baseline is correct.
4. Profile data wait time separately from model compute time.
5. Avoid carrying open handles or CUDA tensors into worker processes.

Throughput changes should be benchmarked over enough batches to amortize worker
startup.
