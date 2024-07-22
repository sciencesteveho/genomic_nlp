# sourcery skip: avoid-single-character-names-variables
#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Train a link-prediction GNN on a base graph initialized with node embedding
data trained from a language model (word2vec or an LLM). Testing is done on a
separate graph of experimentally derived gene-gene (or protein-protein)
interactions for which the training portion never sees.

All of our derived graphs only represent positive data. To ameliorate this, we
use a negative sampling strategy to create negative samples for training (i.e.,
pairs of nodes that are not connected in the graph)."""

import json
from typing import List, Tuple

import numpy as np
from sklearn.metrics import average_precision_score  # type: ignore
from sklearn.metrics import roc_auc_score  # type: ignore
import torch
import torch.nn as nn
from torch_geometric.data import Data  # type: ignore
from torch_geometric.utils import negative_sampling  # type: ignore
from torch_geometric.utils import train_test_split_edges  # type: ignore

from models import LinkPredictionGNN


def initialize_graph(node_features: torch.Tensor, edge_index: torch.Tensor) -> Data:
    """Create a PyG Data object from node features and edge index."""
    return Data(x=node_features, edge_index=edge_index)


def generate_negative_edges(
    edge_index: torch.Tensor, num_nodes: int, num_neg_samples: int
) -> torch.Tensor:
    """Generate negative samples for training."""
    return negative_sampling(
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_neg_samples,
        method="sparse",
    )


def train_model(
    model: nn.Module, optimizer: torch.optim.Optimizer, data: Data
) -> float:
    """Link-prediction GNN training loop."""
    model.train()
    optimizer.zero_grad()
    z = model(data.x, data.train_pos_edge_index)
    pos_out = model.decode(z, data.train_pos_edge_index)
    pos_loss = -torch.log(torch.sigmoid(pos_out) + 1e-15).mean()

    neg_out = model.decode(z, data.train_neg_edge_index)
    neg_loss = -torch.log(1 - torch.sigmoid(neg_out) + 1e-15).mean()

    loss = pos_loss + neg_loss
    loss.backward()
    optimizer.step()
    return loss.item()


def test_model(model: nn.Module, data: Data) -> float:
    """Link-prediction GNN eval loop"""
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.test_pos_edge_index)
        pos_out = model.decode(z, data.test_pos_edge_index).sigmoid()
        neg_out = model.decode(z, data.test_neg_edge_index).sigmoid()

        pred = torch.cat([pos_out, neg_out], dim=0).cpu().numpy()
        true = np.zeros(pred.shape[0])
        true[: pos_out.shape[0]] = 1

        auc = roc_auc_score(true, pred)

    return auc


def evaluate_and_rank_predictions(
    model: nn.Module, data: Data, k: int = 100
) -> Tuple[float, float, List[Tuple[int, int, float]]]:
    """Evaluate the model on the test set and provide K top-ranked
    predictions."""
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)

        # Generate all possible edges
        num_nodes = data.num_nodes
        all_edges = torch.combinations(torch.arange(num_nodes), r=2).t()

        # Predict scores for all possible edges
        all_scores = model.decode(z, all_edges).sigmoid()

        # Sort edges by predicted scores
        sorted_indices = torch.argsort(all_scores, descending=True)
        top_k_indices = sorted_indices[:k]

        # Get top k predicted edges and their scores
        top_k_edges = all_edges[:, top_k_indices].t().tolist()
        top_k_scores = all_scores[top_k_indices].tolist()

        # Combine edges and scores
        ranked_predictions = [
            (int(edge[0]), int(edge[1]), float(score))
            for edge, score in zip(top_k_edges, top_k_scores)
        ]

        # Calculate metrics
        true_edges = set(map(tuple, data.edge_index.t().tolist()))
        pred_edges = set(map(tuple, all_edges.t().tolist()))

        y_true = [1 if edge in true_edges else 0 for edge in pred_edges]
        y_scores = all_scores.tolist()

        auc = roc_auc_score(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)

    return auc, ap, ranked_predictions


def save_model_and_performance(
    model: nn.Module, performances: List[Tuple[str, float]], model_name: str
):
    # Save model
    torch.save(model.state_dict(), f"{model_name}.pt")

    # Save performances
    with open(f"{model_name}_performance.json", "w") as f:
        json.dump(performances, f)


def main():
    """Train a link prediction GNN before evaluating on experimentally validated
    links."""
    # Load data
    node_features, edge_index = load_data("path_to_your_data_file")

    # Initialize graph
    data = initialize_graph(node_features, edge_index)

    # Split edges for training and testing
    data = train_test_split_edges(data)

    # Generate negative edges for training and testing
    num_neg_samples = data.train_pos_edge_index.size(1)
    data.train_neg_edge_index = generate_negative_edges(
        data.train_pos_edge_index, data.num_nodes, num_neg_samples
    )
    data.test_neg_edge_index = generate_negative_edges(
        data.test_pos_edge_index, data.num_nodes, num_neg_samples
    )

    # initialize model and optimizer
    model = LinkPredictionGNN(
        in_channels=node_features.size(1), hidden_channels=256, out_channels=128
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00 - 1)

    # training loop
    best_auc = 0
    for epoch in range(100):
        loss = train_model(model, optimizer, data)
        auc = test_model(model, data)
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), "best_model.pth")
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}")

    print(f"Best AUC: {best_auc:.4f}")

    # load best model for evaluation on external graphs
    model.load_state_dict(torch.load("best_model.pth"))

    # evaluate on external gene-gene graphs
    # external_graphs = (
    #     load_external_graphs()
    # )  # This function should be implemented to load your external graphs
    for i, external_graph in enumerate(external_graphs):
        external_data = initialize_graph(node_features, external_graph)
        num_neg_samples = external_data.edge_index.size(1)
        external_data.test_pos_edge_index = external_data.edge_index
        external_data.test_neg_edge_index = generate_negative_edges(
            external_data.edge_index, external_data.num_nodes, num_neg_samples
        )
        external_auc = test_model(model, external_data)
        print(f"AUC on external graph {i}: {external_auc:.4f}")


if __name__ == "__main__":
    main()
