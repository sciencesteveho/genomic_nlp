# sourcery skip: avoid-single-character-names-variables
#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Train a link-prediction GNN on a base graph initialized with node embedding
data trained from a language model. Testing is done on a separate graph of
experimentally derived gene-gene (or protein-protein) interactions for which the
training portion never sees.

All of our derived graphs only represent positive data. To ameliorate this, we
use a negative sampling strategy to create negative samples for training (i.e.,
pairs of nodes that are not connected in the graph). See the GNNDataPreprocessor
class for more details."""


import argparse
import json
from pathlib import Path
from typing import List, Tuple

from sklearn.metrics import average_precision_score  # type: ignore
from sklearn.metrics import roc_auc_score  # type: ignore
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch_geometric  # type: ignore
from torch_geometric.data import Data  # type: ignore
from torch_geometric.loader import DataLoader  # type: ignore
from tqdm import tqdm  # type: ignore

from interaction_models import LinkPredictionGNN
from model_data_preprocessor import GNNDataPreprocessor

# helpers
EPOCHS = 100
PATIENCE = 15


def create_edge_loader(
    edge_index: torch.Tensor,
    batch_size: int,
    shuffle: bool = False,
) -> torch_geometric.data.DataLoader:
    """Create a DataLoader for edge pairs."""
    edge_dataset = edge_index.t()  # transpose to get pairs of nodes
    return DataLoader(edge_dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data: Data,
    pos_edge_loader: torch_geometric.data.DataLoader,
    neg_edge_loader: torch_geometric.data.DataLoader,
    device: torch.device,
    epoch: int,
) -> float:
    """Train the link prediction model."""
    model.train()
    total_loss = 0.0
    num_batches = len(pos_edge_loader)

    pbar = tqdm(
        total=num_batches,
        desc=f"Training Epoch {epoch}",
    )

    for pos_edges, neg_edges in zip(pos_edge_loader, neg_edge_loader):
        optimizer.zero_grad()

        z = model(data.x.to(device), data.edge_index.to(device))  # forward pass
        pos_out = model.decode(z, pos_edges.to(device))
        neg_out = model.decode(z, neg_edges.to(device))

        loss = (
            -torch.log(pos_out.sigmoid() + 1e-15).mean()
            - torch.log(1 - neg_out.sigmoid() + 1e-15).mean()
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / (pbar.n + 1)

        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
        pbar.update(1)

    pbar.close()
    return total_loss / num_batches


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data: Data,
    pos_edge_loader: torch_geometric.data.DataLoader,
    neg_edge_loader: torch_geometric.data.DataLoader,
    device: torch.device,
) -> float:
    """Evaluate the link prediction model on the validation split."""
    model.eval()
    z = model(data.x.to(device), data.edge_index.to(device))

    pos_scores: List[torch.Tensor] = []
    neg_scores: List[torch.Tensor] = []

    num_batches = len(pos_edge_loader)
    pbar = tqdm(
        total=num_batches,
        desc="Evaluating",
    )

    for pos_edges, neg_edges in zip(pos_edge_loader, neg_edge_loader):
        pos_scores.append(model.decode(z, pos_edges.to(device)).sigmoid().cpu())
        neg_scores.append(model.decode(z, neg_edges.to(device)).sigmoid().cpu())
        pbar.update(1)

    pbar.close()

    pos_scores_tensor = torch.cat(pos_scores, dim=0)
    neg_scores_tensor = torch.cat(neg_scores, dim=0)

    scores = torch.cat([pos_scores_tensor, neg_scores_tensor]).numpy()
    labels = torch.cat(
        [torch.ones(pos_scores_tensor.size(0)), torch.zeros(neg_scores_tensor.size(0))]
    ).numpy()

    return roc_auc_score(labels, scores)


@torch.no_grad()
def evaluate_and_rank_validated_predictions(
    model: nn.Module,
    data: Data,
    pos_edge_loader: torch_geometric.data.DataLoader,
    neg_edge_loader: torch_geometric.data.DataLoader,
    device: torch.device,
) -> Tuple[float, float, List[Tuple[int, int, float]]]:
    """Evaluate the model on experimentally validated edges and rank them."""
    model.eval()
    model = model.to(device)
    z = model(data.x.to(device), data.edge_index.to(device))

    all_scores, all_edges, y_true = [], [], []

    pbar = tqdm(
        total=len(pos_edge_loader) + len(neg_edge_loader), desc="Evaluating edges"
    )

    # process positive edges
    for pos_edges in pos_edge_loader:
        pos_edges = pos_edges.to(device)
        scores = model.decode(z, pos_edges.t()).sigmoid()
        all_scores.append(scores.cpu())
        all_edges.append(pos_edges.cpu())
        y_true.append(torch.ones(len(pos_edges)))
        pbar.update(1)

    # process negative edges
    for neg_edges in neg_edge_loader:
        neg_edges = neg_edges.to(device)
        scores = model.decode(z, neg_edges.t()).sigmoid()
        all_scores.append(scores.cpu())
        all_edges.append(neg_edges.cpu())
        y_true.append(torch.zeros(len(neg_edges)))
        pbar.update(1)

    pbar.close()

    # concatenate results
    all_scores = torch.cat(all_scores)  # type: ignore
    all_edges = torch.cat(all_edges)  # type: ignore
    y_true = torch.cat(y_true)  # type: ignore

    # sort edges by predicted scores and rank
    sorted_indices = torch.argsort(all_scores, descending=True)  # type: ignore
    ranked_edges = all_edges[sorted_indices].tolist()
    ranked_scores = all_scores[sorted_indices].tolist()

    # combine edges and scores
    ranked_predictions = [
        (int(edge[0]), int(edge[1]), float(score))
        for edge, score in zip(ranked_edges, ranked_scores)
    ]

    # calculate metrics
    scores = all_scores.numpy()  # type: ignore
    labels = y_true.numpy()  # type: ignore
    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)

    return auc, ap, ranked_predictions


def save_model_and_performance(
    model: nn.Module,
    performances: List[Tuple[str, float]],
    model_name: str,
    save_dir: Path,
) -> None:
    """Save model and metrics."""
    # save model
    torch.save(model.state_dict(), save_dir / f"{model_name}.pt")

    # save performances
    with open(save_dir / f"{model_name}_performance.json", "w") as f:
        json.dump(performances, f)


def create_loaders(
    data: Data,
    positive_test_edges: torch.Tensor,
    negative_test_edges: torch.Tensor,
    batch_size: int,
) -> Tuple[
    torch_geometric.data.DataLoader,
    torch_geometric.data.DataLoader,
    torch_geometric.data.DataLoader,
    torch_geometric.data.DataLoader,
    torch_geometric.data.DataLoader,
    torch_geometric.data.DataLoader,
]:
    """Set up dataloaders for training, validation, and testing."""
    train_pos_loader = create_edge_loader(
        data.train_pos_edge_index, batch_size=batch_size, shuffle=True
    )
    train_neg_loader = create_edge_loader(
        data.train_neg_edge_index, batch_size=batch_size, shuffle=True
    )
    val_pos_loader = create_edge_loader(data.val_pos_edge_index, batch_size=batch_size)
    val_neg_loader = create_edge_loader(data.val_neg_edge_index, batch_size=batch_size)
    test_pos_loader = create_edge_loader(positive_test_edges, batch_size=batch_size)
    test_neg_loader = create_edge_loader(negative_test_edges, batch_size=batch_size)

    return (
        train_pos_loader,
        train_neg_loader,
        val_pos_loader,
        val_neg_loader,
        test_pos_loader,
        test_neg_loader,
    )


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate a GNN for link prediction."
    )
    parser.add_argument("--embeddings", type=str, help="Path to embeddings file")
    parser.add_argument("--text_edges_file", type=str, help="Path to edge list file")
    parser.add_argument(
        "--positive_pairs_file", type=str, help="Path to positive pairs file"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    save_dir = Path("/ocean/projects/bio210019p/stevesho/genomic_nlp/models/gnn")

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # preprocess data
    preprocessor = GNNDataPreprocessor(args)
    data, positive_test_edges, negative_test_edges = preprocessor.preprocess_data()

    # loaders
    (
        train_pos_loader,
        train_neg_loader,
        val_pos_loader,
        val_neg_loader,
        test_pos_loader,
        test_neg_loader,
    ) = create_loaders(
        data=data,
        positive_test_edges=positive_test_edges,
        negative_test_edges=negative_test_edges,
        batch_size=args.batch_size,
    )

    # initialize model, optimizer, and scheduler
    model = LinkPredictionGNN(
        in_channels=data.num_node_features, embedding_size=256, out_channels=256
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=True
    )

    # training loop
    best_auc = float("-inf")
    patience_counter = 0
    for epoch in range(EPOCHS):
        loss = train_model(
            model=model,
            optimizer=optimizer,
            data=data,
            pos_edge_loader=train_pos_loader,
            neg_edge_loader=train_neg_loader,
            device=device,
            epoch=epoch,
        )
        auc = evaluate_model(
            model=model,
            data=data,
            pos_edge_loader=val_pos_loader,
            neg_edge_loader=val_neg_loader,
            device=device,
        )
        scheduler.step(auc)
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

    # load the best model for final evaluation
    model.load_state_dict(torch.load("best_model.pth"))

    # evaluate on test set
    auc, ap, ranked_predictions = evaluate_and_rank_validated_predictions(
        model=model,
        data=data,
        pos_edge_loader=test_pos_loader,
        neg_edge_loader=test_neg_loader,
        device=device,
    )

    print("Final Evaluation:")
    print(f"AUC: {auc:.4f}")
    print(f"Average Precision: {ap:.4f}")
    print("Top 10 ranked validated links:")
    for rank, (node1, node2, score) in enumerate(ranked_predictions[:10], 1):
        print(f"Rank {rank}: ({node1}, {node2}) - Score: {score:.4f}")

    # save all ranked predictions to a file
    with open(save_dir / "ranked_validated_predictions.txt", "w") as f:
        for node1, node2, score in ranked_predictions:
            f.write(f"{node1}\t{node2}\t{score:.4f}\n")

    # save model and performance
    save_model_and_performance(
        model, [("AUC", auc), ("AP", ap)], "final_model", save_dir
    )


if __name__ == "__main__":
    main()
