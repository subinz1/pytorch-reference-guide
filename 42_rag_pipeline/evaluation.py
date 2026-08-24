"""
RAG Evaluation Metrics

Measures retrieval quality and generation quality for RAG systems.
Includes both retrieval metrics (recall, MRR, NDCG) and generation
metrics (faithfulness, relevance).

Usage:
    python evaluation.py
"""

import torch
import torch.nn.functional as F
import math


# ============================================================
# Retrieval Metrics
# ============================================================


def recall_at_k(
    retrieved_ids: list[str], relevant_ids: list[str], k: int
) -> float:
    """Proportion of relevant documents found in top-k results."""
    if not relevant_ids:
        return 0.0
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(retrieved_set & relevant_set) / len(relevant_set)


def precision_at_k(
    retrieved_ids: list[str], relevant_ids: list[str], k: int
) -> float:
    """Proportion of top-k results that are relevant."""
    if k == 0:
        return 0.0
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(retrieved_set & relevant_set) / k


def mean_reciprocal_rank(
    retrieved_ids: list[str], relevant_ids: list[str]
) -> float:
    """Reciprocal of the rank of the first relevant document."""
    relevant_set = set(relevant_ids)
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(
    retrieved_ids: list[str], relevant_ids: list[str], k: int
) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    relevant_set = set(relevant_ids)

    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in relevant_set:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because ranks start at 1

    # Ideal DCG
    ideal_relevant = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_relevant))

    return dcg / idcg if idcg > 0 else 0.0


# ============================================================
# Generation Quality Metrics
# ============================================================


def context_relevance_score(
    query_embedding: torch.Tensor,
    context_embeddings: torch.Tensor,
) -> float:
    """Average cosine similarity between query and retrieved contexts."""
    similarities = F.cosine_similarity(
        query_embedding.unsqueeze(0).expand(context_embeddings.size(0), -1),
        context_embeddings,
    )
    return similarities.mean().item()


def answer_faithfulness(
    answer_embedding: torch.Tensor,
    context_embeddings: torch.Tensor,
) -> float:
    """How well the answer is grounded in the retrieved context.
    Higher = answer is more faithful to context (less hallucination)."""
    similarities = F.cosine_similarity(
        answer_embedding.unsqueeze(0).expand(context_embeddings.size(0), -1),
        context_embeddings,
    )
    return similarities.max().item()


def answer_relevance(
    answer_embedding: torch.Tensor,
    query_embedding: torch.Tensor,
) -> float:
    """How relevant the answer is to the original query."""
    return F.cosine_similarity(
        answer_embedding.unsqueeze(0), query_embedding.unsqueeze(0)
    ).item()


# ============================================================
# Batch Evaluation
# ============================================================


class RAGEvaluator:
    """Evaluates a RAG pipeline on a test set of queries with ground truth."""

    def __init__(self, encoder=None):
        self.encoder = encoder

    def evaluate_retrieval(
        self,
        queries: list[str],
        retrieved_per_query: list[list[str]],
        relevant_per_query: list[list[str]],
        k_values: list[int] = [1, 3, 5, 10],
    ) -> dict[str, float]:
        """Evaluate retrieval quality across all queries."""
        metrics = {}

        for k in k_values:
            recalls = [
                recall_at_k(ret, rel, k)
                for ret, rel in zip(retrieved_per_query, relevant_per_query)
            ]
            precisions = [
                precision_at_k(ret, rel, k)
                for ret, rel in zip(retrieved_per_query, relevant_per_query)
            ]
            ndcgs = [
                ndcg_at_k(ret, rel, k)
                for ret, rel in zip(retrieved_per_query, relevant_per_query)
            ]
            metrics[f"recall@{k}"] = sum(recalls) / len(recalls)
            metrics[f"precision@{k}"] = sum(precisions) / len(precisions)
            metrics[f"ndcg@{k}"] = sum(ndcgs) / len(ndcgs)

        mrrs = [
            mean_reciprocal_rank(ret, rel)
            for ret, rel in zip(retrieved_per_query, relevant_per_query)
        ]
        metrics["mrr"] = sum(mrrs) / len(mrrs)

        return metrics

    def evaluate_generation(
        self,
        queries: list[str],
        answers: list[str],
        contexts: list[list[str]],
    ) -> dict[str, float]:
        """Evaluate generation quality using embedding similarity."""
        if self.encoder is None:
            raise ValueError("Encoder required for generation evaluation")

        faithfulness_scores = []
        relevance_scores = []

        for query, answer, ctx_docs in zip(queries, answers, contexts):
            query_emb = self.encoder.encode([query])
            answer_emb = self.encoder.encode([answer])
            context_embs = self.encoder.encode(ctx_docs)

            faithfulness_scores.append(
                answer_faithfulness(answer_emb.squeeze(0), context_embs)
            )
            relevance_scores.append(
                answer_relevance(answer_emb.squeeze(0), query_emb.squeeze(0))
            )

        return {
            "faithfulness": sum(faithfulness_scores) / len(faithfulness_scores),
            "answer_relevance": sum(relevance_scores) / len(relevance_scores),
        }


# ============================================================
# Demo
# ============================================================


def main():
    print("=" * 60)
    print("RAG Evaluation Metrics Demo")
    print("=" * 60)

    # Simulated retrieval results
    retrieved = ["doc_1", "doc_3", "doc_5", "doc_2", "doc_7"]
    relevant = ["doc_1", "doc_2", "doc_4"]

    print("\nRetrieval Metrics:")
    print(f"  Retrieved: {retrieved}")
    print(f"  Relevant:  {relevant}")
    print()

    for k in [1, 3, 5]:
        r = recall_at_k(retrieved, relevant, k)
        p = precision_at_k(retrieved, relevant, k)
        n = ndcg_at_k(retrieved, relevant, k)
        print(f"  @{k}: Recall={r:.3f}, Precision={p:.3f}, NDCG={n:.3f}")

    mrr = mean_reciprocal_rank(retrieved, relevant)
    print(f"  MRR: {mrr:.3f}")

    # Batch evaluation
    print(f"\n{'─' * 60}")
    print("\nBatch Evaluation (3 queries):")

    queries = ["What is PyTorch?", "How does compile work?", "What is FSDP?"]
    retrieved_per_query = [
        ["doc_1", "doc_2", "doc_3"],
        ["doc_4", "doc_5", "doc_1"],
        ["doc_6", "doc_4", "doc_7"],
    ]
    relevant_per_query = [
        ["doc_1", "doc_3"],
        ["doc_4", "doc_5"],
        ["doc_6"],
    ]

    evaluator = RAGEvaluator()
    metrics = evaluator.evaluate_retrieval(
        queries, retrieved_per_query, relevant_per_query, k_values=[1, 3]
    )

    print("\n  Aggregated Retrieval Metrics:")
    for name, value in sorted(metrics.items()):
        print(f"    {name}: {value:.3f}")

    # Embedding-based evaluation demo (without model loading)
    print(f"\n{'─' * 60}")
    print("\nEmbedding-Based Metrics (simulated):")

    torch.manual_seed(42)
    query_emb = F.normalize(torch.randn(384), dim=0)
    answer_emb = F.normalize(query_emb + 0.1 * torch.randn(384), dim=0)
    context_embs = F.normalize(torch.randn(5, 384), dim=0)
    context_embs[0] = F.normalize(answer_emb + 0.05 * torch.randn(384), dim=0)

    faith = answer_faithfulness(answer_emb, context_embs)
    relev = answer_relevance(answer_emb, query_emb)
    ctx_rel = context_relevance_score(query_emb, context_embs)

    print(f"  Faithfulness: {faith:.3f} (answer grounded in context)")
    print(f"  Answer Relevance: {relev:.3f} (answer matches query)")
    print(f"  Context Relevance: {ctx_rel:.3f} (retrieved docs match query)")

    print(f"\n{'=' * 60}")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
