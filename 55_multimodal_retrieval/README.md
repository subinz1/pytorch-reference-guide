# Module 55: Multimodal Retrieval with Dual Encoders

## Overview

Multimodal retrieval aligns different inputs — images and captions, products and
text queries, or documents and diagrams — in one vector space. At serving time,
encode a query once, search a precomputed index of the other modality, and then
apply product policy or a heavier ranker to the candidates.

This module builds a compact CLIP-style dual encoder in core PyTorch. It encodes
a numeric image representation and tokenized text, normalizes both, and learns
their alignment with a symmetric contrastive objective.

## Paired Data Is a Contract

Each pair must identify exactly what it claims to align. Preserve a stable sample
identifier, source version, image or document provenance, caption language, and
normalization version. Also track duplicate, near-duplicate, and multi-positive
relationships.

Duplicates matter because ordinary contrastive loss treats every off-diagonal
pair as negative. Two truthful captions for one image violate that assumption
and produce conflicting gradients unless the loss knows about both positives.

## Symmetric Contrastive Loss

Given normalized image matrix `I` and text matrix `T`, compute every pairwise
similarity in the batch:

```python
logits = I @ T.T / temperature
loss = (cross_entropy(logits, targets) + cross_entropy(logits.T, targets)) / 2
```

The first term asks each image to retrieve its text; the second asks each text
to retrieve its image. Normalization prevents vector magnitude from becoming an
unintended confidence score.

## Evaluation and Serving

Measure Recall@K in both directions on a deduplicated, time-aware held-out set.
In production, version item embeddings, the index, encoders, tokenizer, and
preprocessing together. Updating only one makes retrieval behavior impossible
to reproduce.

At large batch sizes, distributed contrastive training needs an all-gather of
embeddings and a target mapping that preserves global pair positions. Treat that
mapping as part of the training contract, not incidental glue code.

## Files in This Module

| File | Description |
|---|---|
| `dual_encoder_retrieval.py` | Masked text pooling, image and text projections, symmetric contrastive loss, and bidirectional Recall@K |

## References

- [PyTorch `normalize`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html)
- [PyTorch distributed collectives](https://docs.pytorch.org/docs/stable/distributed.html)
