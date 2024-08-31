# sourcery skip: avoid-single-character-names-variables
#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Train a link-prediction GNN on a base graph initialized with node embedding
data trained from a language model. Testing is done on a separate graph of
experimentally derived gene-gene (or protein-protein) interactions for which the
training portion never sees.

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

from interaction_gnn import LinkPredictionGNN


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


def evaluate_and_rank_validated_predictions(
    model: nn.Module, data: Data, validated_edges: torch.Tensor
) -> Tuple[float, float, List[Tuple[int, int, float]]]:
    """Evaluate the model on experimentally validated edges and rank them."""
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)

        # predict scores for validated edges
        scores = model.decode(z, validated_edges).sigmoid()

        # sort edges by predicted scores
        sorted_indices = torch.argsort(scores, descending=True)

        # get ranked edges and their scores
        ranked_edges = validated_edges[:, sorted_indices].t().tolist()
        ranked_scores = scores[sorted_indices].tolist()

        # combine edges and scores
        ranked_predictions = [
            (int(edge[0]), int(edge[1]), float(score))
            for edge, score in zip(ranked_edges, ranked_scores)
        ]

        # calculate metrics
        y_true = np.ones(len(scores))  # all edges are true positives
        y_scores = scores.cpu().numpy()

        auc = roc_auc_score(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)

    return auc, ap, ranked_predictions


def save_model_and_performance(
    model: nn.Module, performances: List[Tuple[str, float]], model_name: str
):
    # save model
    torch.save(model.state_dict(), f"{model_name}.pt")

    # save performances
    with open(f"{model_name}_performance.json", "w") as f:
        json.dump(performances, f)


def main():
    """Train a link prediction GNN before evaluating on experimentally validated
    links."""
    # load data
    node_features, edge_index = load_data("path_to_your_data_file")

    # initialize graph
    data = initialize_graph(node_features, edge_index)

    # split edges for training and testing
    data = train_test_split_edges(data)

    # generate negative edges for training and testing
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

    # load the trained model
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    # evaluate on external gene-gene graphs
    external_graphs = load_external_graphs()
    for i, external_graph in enumerate(external_graphs):
        external_data = initialize_graph(node_features, external_graph["edge_index"])

        auc, ap, ranked_predictions = evaluate_and_rank_validated_predictions(
            model, external_data, external_graph["edge_index"]
        )

        print(f"Evaluation on external graph {i}:")
        print(f"AUC: {auc:.4f}")
        print(f"Average Precision: {ap:.4f}")
        print("Top 10 ranked validated links:")
        for rank, (node1, node2, score) in enumerate(ranked_predictions[:10], 1):
            if "edge_names" in external_graph:
                node1_name = external_graph["edge_names"][node1]
                node2_name = external_graph["edge_names"][node2]
                print(f"Rank {rank}: ({node1_name}, {node2_name}) - Score: {score:.4f}")
            else:
                print(f"Rank {rank}: ({node1}, {node2}) - Score: {score:.4f}")
        print("\n")

        # Save all ranked predictions to a file
        with open(f"ranked_validated_predictions_graph_{i}.txt", "w") as f:
            for node1, node2, score in ranked_predictions:
                if "edge_names" in external_graph:
                    node1_name = external_graph["edge_names"][node1]
                    node2_name = external_graph["edge_names"][node2]
                    f.write(f"{node1_name}\t{node2_name}\t{score:.4f}\n")
                else:
                    f.write(f"{node1}\t{node2}\t{score:.4f}\n")


if __name__ == "__main__":
    main()
