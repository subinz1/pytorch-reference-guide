# Module 52: Graph Neural Networks with Core PyTorch

## Overview

Graph neural networks (GNNs) learn from entities and relationships. A graph is
represented by node features `x` with shape `[num_nodes, feature_dim]` and an
`edge_index` tensor with shape `[2, num_edges]`. Each edge maps a source node to
a destination node. Message passing collects source features at each destination
and combines them with the destination's current representation.

This module uses only core PyTorch. It is useful when you want to understand the
tensor operations behind higher-level graph libraries, prototype a graph model
without an extra dependency, or test an operator contract precisely.

## Message Passing in Four Steps

For an edge `source -> destination`:

1. **Gather** source features: `messages = x[source]`.
2. **Transform** messages with a learned projection.
3. **Aggregate** messages at each destination with `index_add_`.
4. **Update** each node using its current features and the aggregated message.

```python
source, destination = edge_index
messages = x[source]
aggregated = torch.zeros_like(x)
aggregated.index_add_(0, destination, messages)
```

Repeated destinations are expected: `index_add_` accumulates every incoming
message. Divide by an incoming-degree tensor for mean aggregation, guarding
against zero-degree nodes with `clamp_min(1)`.

## Edge Direction Is Part of the Contract

`edge_index[0]` is the source and `edge_index[1]` is the destination in this
guide. Reversing them does not raise an error; it silently changes the model.
For an undirected graph, explicitly include both directions. Add self-loops when
a layer should preserve a node's own features through aggregation.

```python
reverse_edges = edge_index.flip(0)
undirected_edges = torch.cat([edge_index, reverse_edges], dim=1)
```

## A Minimal Graph Convolution

The `GraphConv` layer in `graph_message_passing.py` uses a learned linear
projection, mean aggregation, and a residual connection. It keeps the example
small enough to inspect while still exposing the important contracts:

- `edge_index` must be integral and shaped `[2, E]`.
- Node IDs must be within `[0, num_nodes)`.
- Aggregation output always has one row per node, including isolated nodes.
- CUDA `index_add_` can be nondeterministic when the same destination receives
  concurrent updates; select deterministic algorithms only when the performance
  cost is acceptable.

## Training for Node Classification

For a single graph, labels often exist only for selected nodes. Use boolean masks
to express the split rather than creating separate graph copies:

```python
logits = model(node_features, edge_index)
loss = torch.nn.functional.cross_entropy(logits[train_mask], labels[train_mask])
accuracy = (logits[val_mask].argmax(dim=-1) == labels[val_mask]).float().mean()
```

This pattern generalizes to transductive node classification. For inductive
workloads, ensure validation and test graphs do not leak training-only features
or edges into the message-passing neighborhood.

## Scaling the Representation

The dense `[2, E]` edge-index form is clear and flexible, but it still stores one
entry per edge. At larger scales, sample neighborhoods, partition the graph, or
use a sparse representation when the operation supports it. Profile end-to-end:
host-to-device transfer, feature lookup, gather/scatter, and synchronization can
matter more than the linear layers themselves.

## Files in This Module

| File | Description |
|---|---|
| `graph_message_passing.py` | Edge validation, self-loops, mean aggregation, a residual graph-convolution layer, and a toy node-classification loop |

## References

- [`torch.Tensor.index_add_`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.index_add_.html)
- [PyTorch sparse tensors](https://docs.pytorch.org/docs/stable/sparse.html)
- [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)
