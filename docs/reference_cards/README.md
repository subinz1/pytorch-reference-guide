# PyTorch Reference Cards

Short, practical reminders for decisions that come up while building and
debugging PyTorch programs. Each card is intentionally independent: read the
one that matches the problem in front of you, then follow its minimal example
and checklist.

| Card | Use it when you need to… |
| --- | --- |
| [Tensor layout](tensor-layout.md) | reason about views, strides, and contiguity |
| [Shapes and broadcasting](shape-broadcasting.md) | make tensor dimensions explicit |
| [Device and dtype](device-dtype.md) | move models and inputs safely |
| [Autograd boundaries](autograd-boundaries.md) | control gradient tracking without `.data` |
| [Training and evaluation modes](module-training-modes.md) | handle Dropout and BatchNorm correctly |
| [Optimizer lifecycle](optimizer-lifecycle.md) | order zeroing, backward, clipping, and stepping |
| [DataLoader performance](dataloader-performance.md) | remove input-pipeline bottlenecks |
| [Mixed precision](mixed-precision.md) | use autocast and gradient scaling safely |
| [Compile triage](compile-triage.md) | investigate `torch.compile` behavior |
| [Profiler workflow](profiler-workflow.md) | capture an actionable performance trace |
| [Out-of-memory triage](memory-oom.md) | distinguish leaks from peak-memory pressure |
| [Reproducibility](reproducibility.md) | make an experiment repeatable |
| [Checkpointing](checkpointing.md) | resume training faithfully |
| [Distributed environment](distributed-environment.md) | validate a `torchrun` launch |
| [Tensor testing](tensor-testing.md) | test numerical PyTorch code reliably |
| [Export preflight](export-preflight.md) | validate a model before `torch.export` |
| [Inference batching](inference-batching.md) | build a correct, efficient inference path |

The longer module guides explain the concepts in depth; these cards focus on
the operational details that are easy to forget under time pressure.
