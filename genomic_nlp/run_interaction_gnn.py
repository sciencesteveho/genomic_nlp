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


import argparse
import json
from typing import List, Tuple

import numpy as np
from sklearn.metrics import average_precision_score  # type: ignore
from sklearn.metrics import roc_auc_score  # type: ignore
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data  # type: ignore
from torch_geometric.nn import GCNConv  # type: ignore
from torch_geometric.nn import GraphNorm
from tqdm import tqdm

from interaction_models import LinkPredictionGNN
from model_data_preprocessor import GNNDataPreprocessor

# helpers
EPOCHS = 100
PATIENCE = 15


def train_model(
    model: nn.Module, optimizer: torch.optim.Optimizer, data: Data, epoch: int
) -> float:
    """Training function for the link-prediction GNN."""
        pbar = tqdm(total=len(data_loader))
        pbar.set_description(
            f"\nEvaluating {self.model.__class__.__name__} model @ epoch: {epoch}"
        )
    model.train()
    optimizer.zero_grad()
    z = model(data.x, data.edge_index)  # forward pass

    # positive edges
    pos_out = model.decode(z, data.train_pos_edge_index)
    pos_loss = -torch.log(torch.sigmoid(pos_out) + 1e-15).mean()

    # negative edges
    neg_out = model.decode(z, data.train_neg_edge_index)
    neg_loss = -torch.log(1 - torch.sigmoid(neg_out) + 1e-15).mean()

    loss = pos_loss + neg_loss
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data: Data,
) -> float:
    """Get ROC AUC score for the model on the test set."""
    model.eval()
    z = model(data.x, data.edge_index)
    pos_out = model.decode(z, data.val_pos_edge_index).sigmoid()
    neg_out = model.decode(z, data.val_neg_edge_index).sigmoid()
    pred = torch.cat([pos_out, neg_out], dim=0).cpu().numpy()
    true = np.zeros(pred.shape[0])
    true[: pos_out.shape[0]] = 1
    return roc_auc_score(true, pred)


@torch.no_grad()
def evaluate_and_rank_validated_predictions(
    model: nn.Module,
    data: Data,
    positive_test_edges: torch.Tensor,
    negative_test_edges: torch.Tensor,
) -> Tuple[float, float, List[Tuple[int, int, float]]]:
    """Evaluate the model on experimentally validated edges and rank them."""
    model.eval()
    z = model(data.x, data.edge_index)

    # combine positive and negative edges
    all_test_edges = torch.cat([positive_test_edges, negative_test_edges], dim=1)

    # predict scores
    raw_scores = model.decode(z, all_test_edges).sigmoid()

    # get labels
    y_true = torch.cat(
        [
            torch.ones(positive_test_edges.shape[1]),
            torch.zeros(negative_test_edges.shape[1]),
        ]
    )

    # sort edges by predicted scores and rank
    sorted_indices = torch.argsort(raw_scores, descending=True)
    ranked_edges = all_test_edges[:, sorted_indices].t().tolist()
    ranked_scores = raw_scores[sorted_indices].tolist()

    # combine edges and scores
    ranked_predictions = [
        (int(edge[0]), int(edge[1]), float(score))
        for edge, score in zip(ranked_edges, ranked_scores)
    ]

    # calculate metrics
    scores = raw_scores.cpu().numpy()
    labels = y_true.cpu().numpy()
    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)

    return auc, ap, ranked_predictions


def save_model_and_performance(
    model: nn.Module, performances: List[Tuple[str, float]], model_name: str
) -> None:
    """Save model and metrics."""
    # save model
    torch.save(model.state_dict(), f"{model_name}.pt")

    # save performances
    with open(f"{model_name}_performance.json", "w") as f:
        json.dump(performances, f)


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate a GNN for link prediction."
    )
    parser.add_argument("--embeddings", type=str, help="Path to embeddings file")
    parser.add_argument("--edge_file", type=str, help="Path to edge list file")
    parser.add_argument(
        "--positive_pairs_file", type=str, help="Path to positive pairs file"
    )
    args = parser.parse_args()

    # preprocess data
    preprocessor = GNNDataPreprocessor(args)
    data, positive_test_edges, negative_test_edges = preprocessor.preprocess_data()

    # initialize model and optimizer
    model = LinkPredictionGNN(
        in_channels=data.num_node_features, embedding_size=256, out_channels=256
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # training loop
    best_auc = float("-inf")
    patience_count = 0
    for epoch in range(EPOCHS):
        loss = train_model(model=model, optimizer=optimizer, data=data)
        auc = evaluate_model(model=model, data=data)
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    print(f"Best AUC: {best_auc:.4f}")

    # Load the best model for final evaluation
    model.load_state_dict(torch.load("best_model.pth"))

    # Evaluate on test set
    auc, ap, ranked_predictions = evaluate_and_rank_validated_predictions(
        model=model,
        data=data,
        positive_test_edges=positive_test_edges,
        negative_test_edges=negative_test_edges,
    )

    print("Final Evaluation:")
    print(f"AUC: {auc:.4f}")
    print(f"Average Precision: {ap:.4f}")
    print("Top 10 ranked validated links:")
    for rank, (node1, node2, score) in enumerate(ranked_predictions[:10], 1):
        print(f"Rank {rank}: ({node1}, {node2}) - Score: {score:.4f}")

    # Save all ranked predictions to a file
    with open("ranked_validated_predictions.txt", "w") as f:
        for node1, node2, score in ranked_predictions:
            f.write(f"{node1}\t{node2}\t{score:.4f}\n")

    # Save model and performance
    save_model_and_performance(model, [("AUC", auc), ("AP", ap)], "final_model")


if __name__ == "__main__":
    main()
