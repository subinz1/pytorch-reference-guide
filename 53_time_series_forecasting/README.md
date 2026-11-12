# Module 53: Time-Series Forecasting for Operational Demand

## Overview

Forecasting answers an operational question: *given what was known at a cutoff,
what demand should we expect over the next horizon?* Time makes this different
from generic regression. Validation must respect chronology, feature generation
must not look beyond the cutoff, and the prediction horizon must match the
decision it supports.

The example trains a direct seven-day demand forecaster. It uses a GRU over the
last 28 observations plus calendar and promotion signals that are known for the
future horizon. The data is synthetic and CPU-runnable, but the contract maps
to inventory, capacity, incident-volume, and usage forecasting.

## Forecasting Data Contract

| Feature type | Available when predicting day *t*? | Example |
|---|---:|---|
| Historical target | Yes, through *t - 1* | Demand, traffic, queue depth |
| Known future covariate | Yes | Calendar, planned promotion, published price |
| Observed future covariate | No | Realized weather, actual redemption |

Only historical values and known future covariates belong in an online
forecast. A realized future feature creates leakage: offline metrics improve
while production accuracy degrades.

## Split by Time, Not at Random

Random train-test splits leak adjacent observations across both sets. Split by
the forecast target time, fit scaling statistics on the training period only,
and exclude windows whose target horizon crosses the split boundary. The
example records each window's target start to make this explicit.

## Direct Versus Recursive Horizons

For a seven-day horizon, a direct model returns all seven values at once. This
avoids recursively feeding a prediction back as input, which compounds error.
Recursive models remain useful when horizons vary; choose the strategy that
matches the serving contract and evaluate it the same way.

## Production Checklist

- Version the cutoff, feature definitions, model weights, and forecast output
  together.
- Monitor missing inputs, distribution drift, and delayed ground truth.
- Compare against a seasonal-naive baseline at several historical cutoffs.
- Define a safe fallback forecast before deployment.

## Files in This Module

| File | Description |
|---|---|
| `demand_forecasting.py` | Leakage-safe windows, direct multi-horizon GRU forecasting, and MAE in original demand units |

## References

- [PyTorch `nn.GRU`](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html)
- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)
