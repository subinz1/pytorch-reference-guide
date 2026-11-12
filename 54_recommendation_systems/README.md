# Module 54: Recommendation Systems — Retrieval and Ranking Foundations

## Overview

A production recommender is a funnel, not a single model:

1. **Retrieve** a few hundred plausible candidates from a large catalog.
2. **Rank** them with richer context and more expensive features.
3. **Apply policy** for availability, safety, diversity, freshness, and user
   controls.

This module implements the retrieval stage with a two-tower model in core
PyTorch. User and item towers produce vectors in a shared space; their dot
product is a retrieval score. It trains with in-batch negatives, a scalable
baseline for implicit-feedback data.

## Interaction Data Is a Contract

An interaction row is not automatically a positive preference. Preserve event
type, timestamp, exposure context, source, and any reversal such as a quick
dismissal. A purchase, completed watch, save, and impression all imply different
labels. Use a chronological split: random splits let a model learn from future
behavior.

## Two-Tower Retrieval

For user representation `u` and item representation `v`, use a normalized dot
product with temperature `τ`:

```python
logits = user_vectors @ item_vectors.T / temperature
loss = cross_entropy(logits, positive_positions)
```

Each row's paired item is positive; other items in the batch are negatives.
That is efficient, but only valid when accidental positives are rare. Mature
systems deliberately sample negatives and model multiple positive items.

## Retrieval Is Not Ranking

Retrieval should maximize recall: did the relevant item reach the candidate
set? The ranker can use detailed history, freshness, and cross-features that
are too expensive to compute across the full catalog. Evaluate retrieval with
Recall@K before interpreting ranker or click metrics.

## Operational Checklist

- Exclude unavailable and already-consumed items before presentation.
- Refresh item embeddings when catalog attributes change; version the index with
  the model checkpoint.
- Segment Recall@K by cohort, catalog age, and traffic source.
- Keep business constraints explicit rather than letting them become accidental
  artifacts of the training data.

## Files in This Module

| File | Description |
|---|---|
| `two_tower_retrieval.py` | Embedding towers, in-batch contrastive training, Recall@K, and filtering seen items before recommendation |

## References

- [PyTorch `nn.Embedding`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
- [PyTorch `cross_entropy`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html)
