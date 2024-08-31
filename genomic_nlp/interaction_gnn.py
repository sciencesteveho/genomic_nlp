# sourcery skip: avoid-single-character-names-variables
#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""GNN model architectures for link prediction. We train link-prediction GNNs to
compare the rich semantic capture capabilites of different language models.
Because the goal of the GNN is to evaluate the quality of the embeddings, we
default to a simpler GNN architecture to avoid overfitting and complexity that
may hinder the evaluation of the embeddings."""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv  # type: ignore
from torch_geometric.nn import GraphNorm


class LinkPredictionGNN(nn.Module):
    """Simple GNN for link prediction, trained on a base graph initialized with
    trained embeddings and evaluated on a separate graph of experimentally
    derived gene-gene interactions.

    The model defaults to two convolutional layers with graph normalization and
    ReLU for its activation function. Additionally, we implement dropouut and a
    dense skip connection.
    """

    def __init__(
        self,
        in_channels: int,
        embedding_size: int,
        out_channels: int,
        dropout: float = 0.2,
    ) -> None:
        """Instantiate the model."""
        super(LinkPredictionGNN, self).__init__()
        self.activation = F.relu
        self.dropout = nn.Dropout(dropout)

        # convolutional layers
        self.conv1 = GCNConv(in_channels, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)

        # normalization layers
        self.norm1 = GraphNorm(embedding_size)
        self.norm2 = GraphNorm(embedding_size)

        # skip connection + linear projection
        self.residual = nn.Linear(in_channels, out_channels)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encode the input graph."""
        x1 = self.dropout(self.activation(self.norm1(self.conv1(x, edge_index))))
        x2 = self.dropout(self.activation(self.norm2(self.conv2(x1, edge_index))))
        return self.residual(x1 + x2)

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Decode the input graph."""
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""
        return self.encode(x, edge_index)
