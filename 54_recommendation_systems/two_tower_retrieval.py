"""A compact two-tower retrieval baseline with in-batch negatives."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TwoTower(nn.Module):
    """Map user and item identifiers into a normalized retrieval space."""

    def __init__(self, users: int, items: int, embedding_dim: int = 32) -> None:
        super().__init__()
        self.users = nn.Embedding(users, embedding_dim)
        self.items = nn.Embedding(items, embedding_dim)
        self.user_projection = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.Tanh())
        self.item_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def encode_users(self, user_ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.user_projection(self.users(user_ids)), dim=-1)

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item_projection(self.items(item_ids)), dim=-1)

    def in_batch_loss(
        self, user_ids: torch.Tensor, positive_item_ids: torch.Tensor, temperature: float = 0.1
    ) -> torch.Tensor:
        user_vectors = self.encode_users(user_ids)
        item_vectors = self.encode_items(positive_item_ids)
        logits = user_vectors @ item_vectors.T / temperature
        positions = torch.arange(user_ids.numel(), device=user_ids.device)
        return F.cross_entropy(logits, positions)


def recall_at_k(
    scores: torch.Tensor, positive_item_ids: torch.Tensor, candidate_item_ids: torch.Tensor, k: int
) -> torch.Tensor:
    """Return the fraction of rows whose positive item appears in the top-k."""
    top_items = candidate_item_ids[scores.topk(k, dim=-1).indices]
    return (top_items == positive_item_ids.unsqueeze(-1)).any(dim=-1).float().mean()


def recommend(
    model: TwoTower, user_id: int, item_ids: torch.Tensor, seen_item_ids: torch.Tensor, k: int
) -> torch.Tensor:
    """Retrieve top-k unseen catalog items for one user."""
    with torch.no_grad():
        user_vector = model.encode_users(torch.tensor([user_id]))
        scores = (user_vector @ model.encode_items(item_ids).T).squeeze(0)
        scores[torch.isin(item_ids, seen_item_ids)] = -torch.inf
        return item_ids[scores.topk(k).indices]


def main() -> None:
    torch.manual_seed(0)
    users, items = 32, 128
    model = TwoTower(users=users, items=items)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    # A stand-in for one positive implicit-feedback item per user. Real data
    # should be split chronologically and include exposure-aware labels.
    user_ids = torch.arange(users)
    positive_item_ids = (user_ids * 3 + 7) % items
    for _ in range(250):
        optimizer.zero_grad()
        loss = model.in_batch_loss(user_ids, positive_item_ids)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        candidates = torch.arange(items)
        scores = model.encode_users(user_ids) @ model.encode_items(candidates).T
        print(f"Recall@10: {recall_at_k(scores, positive_item_ids, candidates, k=10):.1%}")
    print("Recommendations for user 0:", recommend(model, 0, candidates, positive_item_ids[:1], 5).tolist())


if __name__ == "__main__":
    main()
