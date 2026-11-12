"""A CLIP-style dual-encoder retrieval example using only core PyTorch."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DualEncoder(nn.Module):
    """Project image features and tokenized text into a shared vector space."""

    def __init__(self, image_features: int, vocabulary: int, embedding_dim: int = 32) -> None:
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Linear(image_features, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.tokens = nn.Embedding(vocabulary, embedding_dim, padding_idx=0)
        self.text_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.image_encoder(images), dim=-1)

    def encode_text(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_vectors = self.tokens(token_ids)
        weights = attention_mask.unsqueeze(-1).to(token_vectors.dtype)
        pooled = (token_vectors * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return F.normalize(self.text_projection(pooled), dim=-1)


def symmetric_contrastive_loss(
    image_vectors: torch.Tensor, text_vectors: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:
    """Optimize image-to-text and text-to-image retrieval jointly."""
    if image_vectors.shape[0] != text_vectors.shape[0]:
        raise ValueError("each image must have a paired text row")
    targets = torch.arange(image_vectors.shape[0], device=image_vectors.device)
    logits = image_vectors @ text_vectors.T / temperature
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2


def recall_at_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Compute paired retrieval Recall@K for a square similarity matrix."""
    targets = torch.arange(logits.shape[0], device=logits.device)
    return (logits.topk(k, dim=-1).indices == targets.unsqueeze(-1)).any(dim=-1).float().mean()


def make_demo_batch(
    batch_size: int = 8, image_features: int = 12
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create paired synthetic image features and padded caption token IDs."""
    if batch_size > 8:
        raise ValueError("the synthetic demo supports at most eight unique concepts")
    concepts = torch.arange(batch_size)
    concept_vectors = F.one_hot(concepts, num_classes=8).float()
    images = torch.cat((concept_vectors, 0.05 * torch.randn(batch_size, image_features - 8)), dim=-1)
    tokens = torch.stack((concepts + 1, (concepts + 3) % 8 + 1, torch.zeros_like(concepts)), dim=-1)
    attention_mask = tokens.ne(0)
    return images, tokens, attention_mask


def main() -> None:
    torch.manual_seed(0)
    images, token_ids, attention_mask = make_demo_batch()
    model = DualEncoder(image_features=images.shape[-1], vocabulary=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    for _ in range(300):
        optimizer.zero_grad()
        image_vectors = model.encode_images(images)
        text_vectors = model.encode_text(token_ids, attention_mask)
        loss = symmetric_contrastive_loss(image_vectors, text_vectors)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        similarities = model.encode_images(images) @ model.encode_text(token_ids, attention_mask).T
        print(f"Image-to-text Recall@5: {recall_at_k(similarities, 5):.1%}")
        print(f"Text-to-image Recall@5: {recall_at_k(similarities.T, 5):.1%}")


if __name__ == "__main__":
    main()
