# Performance regression triage

Use this card when a change is correct but makes a PyTorch workload slower.

## Establish a valid comparison

- pin the commit, hardware, driver, environment, and input shapes
- warm up eager and compiled paths before timing
- synchronize accelerator work before reading elapsed time
- report median and spread, not one wall-clock sample
- check memory peak and compilation time separately from steady-state latency

## Diagnose by shape of regression

| Pattern | Likely next tool |
|---|---|
| CPU time grows | profiler operator table |
| GPU idle gaps | trace timeline and DataLoader metrics |
| compile slowdown | graph/recompile logs and compile-cache count |
| memory increase | memory snapshot or allocation history |
| distributed slowdown | rank timeline and collective diagnostics |

Keep the benchmark small enough to rerun during review, but representative
enough that the changed path is exercised.
