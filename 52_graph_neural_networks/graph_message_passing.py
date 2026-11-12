"""A dependency-free graph message-passing example using core PyTorch."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def validate_edge_index(edge_index: torch.Tensor, num_nodes: int) -> None:
    """Check the edge-index layout used by this module."""
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if edge_index.dtype not in (torch.int32, torch.int64):
        raise TypeError("edge_index must contain integer node identifiers")
    if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= num_nodes):
        raise ValueError("edge_index contains a node identifier outside the graph")


def add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Append one self-loop per node to a source-to-destination edge list."""
    nodes = torch.arange(num_nodes, device=edge_index.device, dtype=edge_index.dtype)
    return torch.cat([edge_index, torch.stack([nodes, nodes])], dim=1)


def mean_aggregate(features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Average incoming source features for every destination node."""
    validate_edge_index(edge_index, features.shape[0])
    source, destination = edge_index
    aggregate = torch.zeros_like(features)
    aggregate.index_add_(0, destination, features[source])

    degree = features.new_zeros(features.shape[0])
    degree.index_add_(0, destination, torch.ones_like(destination, dtype=features.dtype))
    return aggregate / degree.clamp_min(1).unsqueeze(-1)


class GraphConv(nn.Module):
    """A residual mean-aggregation graph-convolution layer."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.message = nn.Linear(in_features, out_features, bias=False)
        self.residual = (
            nn.Identity()
            if in_features == out_features
            else nn.Linear(in_features, out_features, bias=False)
        )

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        messages = self.message(features)
        return self.residual(features) + mean_aggregate(messages, edge_index)


class NodeClassifier(nn.Module):
    """Two graph-convolution layers followed by per-node classification."""

    def __init__(self, input_features: int, hidden_features: int, classes: int) -> None:
        super().__init__()
        self.first = GraphConv(input_features, hidden_features)
        self.second = GraphConv(hidden_features, classes)

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.first(features, edge_index))
        return self.second(hidden, edge_index)


def build_toy_graph() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a small labelled graph with a train mask."""
    features = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0], [0.1, 0.9], [0.2, 0.8]]
    )
    edges = torch.tensor(
        [[0, 1, 1, 2, 3, 4, 4, 5], [1, 0, 2, 1, 4, 3, 5, 4]],
        dtype=torch.long,
    )
    edge_index = add_self_loops(edges, features.shape[0])
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    train_mask = torch.tensor([True, True, False, True, True, False])
    return features, edge_index, labels, train_mask


def main() -> None:
    torch.manual_seed(0)
    features, edge_index, labels, train_mask = build_toy_graph()
    model = NodeClassifier(input_features=2, hidden_features=8, classes=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=1e-4)

    for _ in range(150):
        optimizer.zero_grad()
        logits = model(features, edge_index)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        predictions = model(features, edge_index).argmax(dim=-1)
        accuracy = (predictions == labels).float().mean()
    print(f"Toy-graph accuracy: {accuracy:.1%}")


if __name__ == "__main__":
    main()
