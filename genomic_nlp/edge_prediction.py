# sourcery skip: avoid-single-character-names-variables
#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Train a link-prediction GNN on a base graph initialized with node embedding
data trained from node2vec. The graph structure is derived from co-occurrence in
abstracts.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

from gensim.models import Word2Vec  # type: ignore
import numpy as np
from sklearn.metrics import average_precision_score  # type: ignore
from sklearn.metrics import roc_auc_score  # type: ignore
from sklearn.metrics import roc_curve  # type: ignore
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch_geometric  # type: ignore
from torch_geometric.data import Data  # type: ignore
from torch_geometric.loader import DataLoader  # type: ignore
from tqdm import tqdm  # type: ignore

from genomic_nlp.gda_data_preprocessor import GDADataPreprocessor
from genomic_nlp.models.edge_prediction_gnn import LinkPredictionGNN

# helpers
EPOCHS = 20
PATIENCE = 3


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
    global_step: int,
    warmup_steps: int,
    base_lr: float,
    pos_weight: float = 1.0,
) -> Tuple[float, int]:
    """Train the link prediction model."""
    model.train()
    total_loss = 0.0
    num_batches = len(pos_edge_loader)

    pbar = tqdm(
        total=num_batches,
        desc=f"Training Epoch {epoch}",
        leave=False,
    )

    for pos_edges, neg_edges in zip(pos_edge_loader, neg_edge_loader):
        optimizer.zero_grad()

        # warmup learning rate for 10% training steps
        if global_step < warmup_steps:
            # linearly scale from 0 -> base_lr over warmup_steps
            warmup_lr = base_lr * float(global_step + 1) / float(warmup_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr

        optimizer.zero_grad()

        # forward pass
        z = model(data.x.to(device), data.edge_index.to(device))
        pos_logits = model.decode(z, pos_edges.to(device))
        neg_logits = model.decode(z, neg_edges.to(device))

        loss_pos = -torch.log(torch.sigmoid(pos_logits) + 1e-15).mean()
        loss_neg = -torch.log(1 - torch.sigmoid(neg_logits) + 1e-15).mean()
        loss = pos_weight * loss_pos + loss_neg

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / (pbar.n + 1)

        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
        pbar.update(1)
        global_step += 1

    pbar.close()
    epoch_loss = total_loss / num_batches
    return epoch_loss, global_step


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

    num_batches = len(pos_edge_loader) + len(neg_edge_loader)
    pbar = tqdm(
        total=num_batches,
        desc="Evaluating",
    )

    for pos_edge in pos_edge_loader:
        pos_scores.append(torch.sigmoid(model.decode(z, pos_edge.to(device))).cpu())
        pbar.update(1)

    for neg_edge in neg_edge_loader:
        neg_scores.append(torch.sigmoid(model.decode(z, neg_edge.to(device))).cpu())
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
) -> Tuple[float, float, np.ndarray, np.ndarray]:
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
        scores = torch.sigmoid(model.decode(z, pos_edges)).cpu()
        all_scores.append(scores)
        all_edges.append(pos_edges.cpu())
        y_true.append(torch.ones(pos_edges.size(0)))
        pbar.update(1)

    # process negative edges
    for neg_edges in neg_edge_loader:
        neg_edges = neg_edges.to(device)
        scores = torch.sigmoid(model.decode(z, neg_edges)).cpu()
        all_scores.append(scores)
        all_edges.append(neg_edges.cpu())
        y_true.append(torch.zeros(neg_edges.size(0)))
        pbar.update(1)

    pbar.close()

    # concatenate results
    all_scores = torch.cat(all_scores)  # type: ignore
    all_edges = torch.cat(all_edges)  # type: ignore
    y_true = torch.cat(y_true)  # type: ignore

    # sort edges by predicted scores and rank
    sorted_indices = torch.argsort(all_scores, descending=True)  # type: ignore
    ranked_scores = all_scores[sorted_indices].tolist()
    sorted_labels = y_true[sorted_indices].numpy()

    # calculate metrics
    scores = all_scores.numpy()  # type: ignore
    labels = y_true.numpy()  # type: ignore
    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)

    return auc, ap, sorted_labels, np.array(ranked_scores)


@torch.no_grad()
def predict_links(
    model: nn.Module,
    data: Data,
    device: torch.device,
) -> List[Tuple[int, int, float]]:
    """Predict links in the graph."""
    model.eval()
    z = model(data.x.to(device), data.edge_index.to(device))

    # predict all possible links
    all_edges = torch_geometric.utils.to_undirected(data.edge_index)
    all_scores = torch.sigmoid(model.decode(z, all_edges)).cpu()

    # sort edges by predicted scores and rank
    sorted_indices = torch.argsort(all_scores, descending=True)
    ranked_edges = all_edges[:, sorted_indices].t().tolist()
    ranked_scores = all_scores[sorted_indices].tolist()

    return [
        (int(edge[0]), int(edge[1]), float(score))
        for edge, score in zip(ranked_edges, ranked_scores)
    ]


def evaluate_predictions(
    predictions: List[Tuple[int, int, float]],
    test_pos_edges: torch.Tensor,
    k: int = 1000,
) -> Tuple[float, float, float, float]:
    """Evaluate predicted links by seeing how many test edges are in the top K."""
    test_edges = set(map(tuple, test_pos_edges.t().tolist()))

    # calculate precision and recall
    true_positives = sum((edge[0], edge[1]) in test_edges for edge in predictions[:k])
    precision = true_positives / k
    recall = true_positives / len(test_edges)

    # calculate AUC and AP
    y_true = [1 if (edge[0], edge[1]) in test_edges else 0 for edge in predictions]
    y_scores = [score for _, _, score in predictions]
    auc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    return precision, recall, auc, ap


def save_model_and_performance(
    model: nn.Module,
    performances: List[Tuple[str, float]],
    model_name: str,
    save_dir: Path,
    year: str,
) -> None:
    """Save model and metrics."""
    # save model
    torch.save(model.state_dict(), save_dir / f"{model_name}.pt")

    # save performances
    with open(save_dir / f"{model_name}_performance_{year}.json", "w") as f:
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

    # verify that the loaders are correct
    print("Train pos loader:", len(train_pos_loader))
    print("Train neg loader:", len(train_neg_loader))
    print("Val   pos loader:", len(val_pos_loader))
    print("Val   neg loader:", len(val_neg_loader))
    print("Test  pos loader:", len(test_pos_loader))
    print("Test  neg loader:", len(test_neg_loader))

    return (
        train_pos_loader,
        train_neg_loader,
        val_pos_loader,
        val_neg_loader,
        test_pos_loader,
        test_neg_loader,
    )


def save_roc_data(
    y_true: np.ndarray, y_scores: np.ndarray, save_dir: Path, year: str
) -> None:
    """Save ROC curve data."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    np.savez(save_dir / f"{year}_roc_data.npz", fpr=fpr, tpr=tpr, thresholds=thresholds)


def save_loss_data(losses: List[float], save_dir: Path, year: str) -> None:
    """Save loss curve data."""
    np.savetxt(save_dir / f"{year}_loss_data.txt", losses)


@torch.no_grad()
def predict_gene_disease_links(
    model: nn.Module,
    data: Data,
    device: torch.device,
    chunk_size: int = 2048,
) -> List[Tuple[int, int, float]]:
    """Predict gene-disease links in the graph by generating all possible
    pairs.
    """
    model.eval()

    # get latent representation
    z = model(data.x.to(device), data.edge_index.to(device))

    # get gene/disease node indices on device
    gene_nodes = data.gene_nodes.to(device)
    disease_nodes = data.disease_nodes.to(device)
    inv_map = data.inv_node_mapping

    all_pairs_cpu = []
    all_scores_cpu = []
    genes = gene_nodes.tolist()
    diseases = disease_nodes.tolist()
    total_pairs = len(genes) * len(diseases)
    print(f"Total candidate gene-disease pairs: {total_pairs:,}")

    start = 0
    while start < len(genes):
        end = min(start + chunk_size, len(genes))
        chunk_gene_nodes = torch.tensor(genes[start:end], device=device)

        # cartesian_prod => shape [chunk_size * len(diseases), 2]
        chunk_pairs = torch.cartesian_prod(chunk_gene_nodes, disease_nodes)
        print(
            f"  cartesian_prod: chunk_pairs.shape = {tuple(chunk_pairs.shape)} "
            f"(should be [N, 2])"
        )

        chunk_scores = torch.sigmoid(model.decode(z, chunk_pairs)).cpu()

        all_pairs_cpu.append(chunk_pairs.cpu())
        all_scores_cpu.append(chunk_scores)

        print(f"Processed chunk of {end - start} genes → {chunk_scores.size(0)} edges")
        start = end

    # concatenate
    cat_pairs = torch.cat(all_pairs_cpu, dim=0)
    cat_scores = torch.cat(all_scores_cpu, dim=0)
    print("cat_pairs final shape =", cat_pairs.shape)
    print("cat_scores final shape =", cat_scores.shape)

    # sort descending
    sorted_idx = torch.argsort(cat_scores, descending=True)
    cat_pairs = cat_pairs[sorted_idx]
    cat_scores = cat_scores[sorted_idx]

    # only keep scores > 0.5
    cat_pairs = cat_pairs[cat_scores > 0.5]

    return [
        (inv_map[g_idx], inv_map[d_idx], float(score))
        for (g_idx, d_idx), score in zip(cat_pairs.tolist(), cat_scores.tolist())
    ]


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate a GNN for link prediction."
    )
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--year", type=int, help="Year of data to use", default=2008)
    args = parser.parse_args()

    save_dir = Path("/ocean/projects/bio210019p/stevesho/genomic_nlp/models/gnn")
    embedding_path = "/ocean/projects/bio210019p/stevesho/genomic_nlp/models/w2v"
    text_path = "/ocean/projects/bio210019p/stevesho/genomic_nlp/training_data/disease"
    w2vmodel_file = (
        f"{embedding_path}/{args.year}/word2vec_300_dimensions_{args.year}.model"
    )
    text_edges_file = f"{text_path}/gda_co_occurence_{args.year}.tsv"

    # load w2v model and get embeddings (dictionary of word: embedding)
    w2v_model = Word2Vec.load(w2vmodel_file)
    embeddings = {word: w2v_model.wv[word] for word in w2v_model.wv.index_to_key}

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # preprocess data
    preprocessor = GDADataPreprocessor(
        text_edges_file=text_edges_file,
        embeddings=embeddings,
    )
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
    base_lr = 0.0001
    model = LinkPredictionGNN(
        in_channels=data.num_node_features, embedding_size=300, out_channels=300
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    # training loop
    total_steps = EPOCHS * len(train_pos_loader)
    warmup_steps = int(total_steps * 0.1)

    global_step = 0
    best_auc = float("-inf")
    patience_counter = 0
    losses = []

    # training loop!
    for epoch in range(EPOCHS):
        loss, global_step = train_model(
            model=model,
            optimizer=optimizer,
            data=data,
            pos_edge_loader=train_pos_loader,
            neg_edge_loader=train_neg_loader,
            device=device,
            epoch=epoch,
            global_step=global_step,
            warmup_steps=warmup_steps,
            base_lr=base_lr,
        )
        losses.append(loss)

        auc = evaluate_model(
            model=model,
            data=data,
            pos_edge_loader=val_pos_loader,
            neg_edge_loader=val_neg_loader,
            device=device,
        )

        if global_step >= warmup_steps:
            scheduler.step(auc)

        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), f"{save_dir}/best_model_{args.year}.pth")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    print(f"Best AUC: {best_auc:.4f}")
    save_loss_data(losses, save_dir, str(args.year))

    # load the best model for final evaluation
    model.load_state_dict(
        torch.load(f"{save_dir}/best_model_{args.year}.pth", map_location=device)
    )

    # evaluate on test set
    auc, ap, y_true_sorted, y_scores_sorted = evaluate_and_rank_validated_predictions(
        model=model,
        data=data,
        pos_edge_loader=test_pos_loader,
        neg_edge_loader=test_neg_loader,
        device=device,
    )

    save_roc_data(y_true_sorted, y_scores_sorted, save_dir, str(args.year))

    print("Final Evaluation:")
    print(f"AUC: {auc:.4f}")
    print(f"Average Precision: {ap:.4f}")

    # save model and performance
    save_model_and_performance(
        model,
        [("AUC", auc), ("AP", ap)],
        f"final_model_{args.year}",
        save_dir,
        args.year,
    )

    # predict gene-disease links
    predicted_gdas = predict_gene_disease_links(model, data, device)

    print("Top 10 predicted gene-disease associations:")
    for rank, (gene_idx, disease_idx, score) in enumerate(predicted_gdas[:10], 1):
        print(
            f"Rank {rank}: Gene {gene_idx} -- Disease {disease_idx} | Score: {score:.4f}"
        )

    with open(save_dir / f"predicted_gdas_{args.year}.txt", "w") as f:
        for gene_idx, disease_idx, score in predicted_gdas:
            f.write(f"{gene_idx}\t{disease_idx}\t{score:.4f}\n")


if __name__ == "__main__":
    main()
