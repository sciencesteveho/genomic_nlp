#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Embedding visualization."""


from pathlib import Path
import pickle
import random
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from gensim.models import Word2Vec  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import phate  # type: ignore
from pybedtools import BedTool  # type: ignore
from sklearn.cluster import KMeans  # type: ignore
from sklearn.metrics import auc  # type: ignore
from sklearn.metrics import roc_curve  # type: ignore

from genomic_nlp.utils.constants import census_oncogenes
from genomic_nlp.utils.constants import RANDOM_STATE
from genomic_nlp.visualization import set_matplotlib_publication_parameters

# import umap  # type: ignore


def load_roc_data(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """Open a file and load the ROC data."""
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data[0]


def plot_roc_curves(emb_model: str) -> None:
    """Plot ROC curves for all models."""
    _set_matplotlib_publication_parameters()
    models = [
        ("Logistic Regression", "logistic_regression_final_roc.pkl"),
        ("MLP", "mlp_final_roc.pkl"),
        ("SVM", "svm_final_roc.pkl"),
        ("XGBoost", "xgboost_final_roc.pkl"),
    ]

    for model_name, filename in models:
        y_true, y_score = load_roc_data(filename)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic (ROC) Curves for {emb_model}")
    plt.legend(loc="lower right")
    plt.show()


def load_tokens(filename: str) -> Set[str]:
    """Load gene tokens from a file."""
    with open(filename, "r") as f:
        return {line.strip().lower() for line in f}


def _set_matplotlib_publication_parameters() -> None:
    """Set matplotlib parameters for publication-quality plots."""
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 300,
            "figure.figsize": (4, 3),
            "font.sans-serif": "Helvetica",
        }
    )


def plot_loss(file_path):
    # read the loss values from the file
    _set_matplotlib_publication_parameters()

    # adjust figure size
    plt.figure(figsize=(3, 2))

    with open(file_path, "r") as file:
        loss_values = [float(line.strip()) for line in file]

    # create an array of epochs (assuming one loss value per epoch)
    epochs = np.arange(1, len(loss_values) + 1)

    # create the plot
    plt.plot(epochs, loss_values, "b-")
    plt.title("Simple link prediction GNN training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # pptionally, use logarithmic scale for y-axis if loss values vary greatly
    # plt.yscale('log')

    # display the plot
    plt.xlim(left=0)
    plt.show()


def compute_phate(
    word_embeddings: Dict[str, np.ndarray], n_components: int = 2
) -> Tuple[np.ndarray, List]:
    """Apply PHATE to reduce dimensionality of word embeddings."""
    words = list(word_embeddings.keys())
    vectors = np.array(list(word_embeddings.values()))
    phate_operator = phate.PHATE(n_components=n_components)
    reduced_vectors = phate_operator.fit_transform(vectors)
    return reduced_vectors, words


# def compute_umap(
#     word_embeddings: Dict[str, np.ndarray], n_components: int = 2
# ) -> Tuple[np.ndarray, List]:
#     """Apply UMAP to reduce dimensionality of word embeddings."""
#     words = list(word_embeddings.keys())
#     vectors = np.array(list(word_embeddings.values()))
#     reducer = umap.UMAP(
#         min_dist=0.05, n_components=n_components, random_state=RANDOM_STATE
#     )
#     reduced_vectors = reducer.fit_transform(vectors)
#     return reduced_vectors, words


def plot_reduction_with_clusters(
    reduced_vectors: np.ndarray,
    words: List[str],
    n_clusters: int = 5,
    year: int = 2023,
    words_to_annotate: Optional[List[str]] = None,
    cmap: str = "viridis",
    reduction: str = "PHATE",
) -> None:
    """Plot PHATE-reduced embeddings with KMeans clustering."""
    _set_matplotlib_publication_parameters()
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(reduced_vectors)

    # create mask for words to annotate
    highlight_mask = np.zeros(len(words), dtype=bool)
    if words_to_annotate:
        for word in words_to_annotate:
            if word in words:
                highlight_mask[words.index(word)] = True

    # plot non-highlighted words
    plt.scatter(
        reduced_vectors[~highlight_mask, 0],
        reduced_vectors[~highlight_mask, 1],
        c=clusters[~highlight_mask],
        cmap=cmap,
        alpha=0.6,
        s=0.2,
    )

    # plot highlighted words in red
    plt.scatter(
        reduced_vectors[highlight_mask, 0],
        reduced_vectors[highlight_mask, 1],
        c="red",
        alpha=1,
        s=0.2,
    )

    # add a colorbar for the clusters
    # scatter = plt.scatter(
    #     reduced_vectors[:, 0],
    #     reduced_vectors[:, 1],
    #     c=clusters,
    #     cmap=cmap,
    #     alpha=0,
    #     s=0,
    # )
    # plt.colorbar(scatter, label="Cluster")
    plt.title(
        f"{reduction} Visualization with KMeans Clustering (n_clusters={n_clusters})"
    )
    plt.xlabel(f"{reduction} 1")
    plt.ylabel(f"{reduction} 2")
    plt.savefig(f"{reduction.lower()}_kmeans_clusters_{year}.png", dpi=450)
    plt.close()


def visualize_word_embeddings(
    word_embeddings: Dict[str, np.ndarray],
    n_components: int = 2,
    n_clusters: int = 5,
    words_to_annotate: Optional[List[str]] = None,
    reduction: str = "PHATE",
    year: int = 2023,
) -> None:
    """
    Full pipeline to visualize word embeddings using PHATE and KMeans clustering.

    Parameters:
        word_embeddings (dict): Dictionary with words as keys and vectors as values.
        n_components (int): Number of dimensions to reduce to (2 or 3).
        n_clusters (int): Number of clusters for KMeans.
        n_annotate (int): Number of words to annotate on the plot.

    Returns:
        None
    """
    if reduction == "PHATE":
        reduced_vectors, words = compute_phate(word_embeddings, n_components)
    elif reduction == "UMAP":
        # reduced_vectors, words = compute_umap(word_embeddings, n_components)
        raise NotImplementedError("UMAP not implemented.")
    else:
        raise ValueError("Invalid reduction method specified. Use 'PHATE' or 'UMAP'.")
    plot_reduction_with_clusters(
        reduced_vectors, words, n_clusters, year, words_to_annotate, reduction=reduction
    )


def get_genes(
    gencode_bed: Path,
) -> List[str]:
    """Filter rna_seq data by TPM"""
    gencode = BedTool(gencode_bed)
    return [
        feature[3] for feature in gencode if feature[0] not in ["chrX", "chrY", "chrM"]
    ]


def load_gencode_lookup(filepath: str) -> Dict[str, str]:
    """Load the Gencode-to-gene-symbol lookup table."""
    gencode_to_symbol = {}
    with open(filepath, "r") as f:
        for line in f:
            gencode, symbol = line.strip().split("\t")
            gencode_to_symbol[symbol] = gencode
    return gencode_to_symbol


def main() -> None:
    """Main function."""
    set_matplotlib_publication_parameters()

    working_dir = "/Users/steveho/genomic_nlp/development/expression"
    gencode_bed = "/Users/steveho/ogl/development/recap/gencode_v26_genes_only_with_GTEx_targets.bed"
    avg_activity_pkl = f"{working_dir}/gtex_tpm_median_across_all_tissues.pkl"
    protein_coding_bed = f"{working_dir}/gencode_v26_protein_coding.bed"
    gencode_to_genesymbol = "/Users/steveho/gnn_plots/graph_resources/local/gencode_to_genesymbol_lookup_table.txt"

    gencode_lookup = load_gencode_lookup(gencode_to_genesymbol)
    symbol_lookup = {v: k for k, v in gencode_lookup.items()}
    genes = get_genes(Path(gencode_bed))
    genes = [gene for gene in genes if "_" not in gene]

    working_dir = "/Users/steveho/genomic_nlp/development/expression"
    gencode_bed = "/Users/steveho/ogl/development/recap/gencode_v26_genes_only_with_GTEx_targets.bed"

    genes = get_genes(Path(gencode_bed))
    genes = [gene for gene in genes if "_" not in gene]
    # change genes to genesymbols
    genes = [symbol_lookup[gene].casefold() for gene in genes if gene in symbol_lookup]

    # embedding_file = "w2v_filtered_embeddings.pkl"
    # embedding_file = "n2v_embeddings.pkl"
    # embedding_file = "genept_embeddings.pkl"
    # embedding_file = "biowordvec_embeddings.pkl"

    # model = Word2Vec.load("word2vec_300_dimensions_2023.model")

    # with open(embedding_file, "rb") as file:
    #     word_embeddings = pickle.load(file)
    # load w2v model
    for year in range(2003, 2024):
        w2v = Word2Vec.load(
            f"/Users/steveho/genomic_nlp/development/models/word2vec_300_dimensions_{year}.model"
        )

        # get genes in the model
        word_embeddings = {gene: w2v.wv[gene] for gene in genes if gene in w2v.wv}

        words = list(word_embeddings.keys())
        # oncogenes = [gene.upper() for gene in census_oncogenes if gene.upper() in words]
        oncogenes = [gene for gene in census_oncogenes if gene in words]

        visualize_word_embeddings(
            word_embeddings=word_embeddings,
            n_clusters=6,
            words_to_annotate=oncogenes,
            reduction="PHATE",
            year=year,
        )
