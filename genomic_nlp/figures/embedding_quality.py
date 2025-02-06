# sourcery skip: name-type-suffix
#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Run a series of analyses to evaluate the quality of gene-centric word
embeddings.
"""


import argparse
import csv
import pickle
import random
import sys
from typing import Dict, List, Set, Tuple

from gensim.models import Word2Vec  # type: ignore
from matplotlib import colors as mcolors  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pybedtools  # type: ignore
from scipy.stats import t  # type: ignore
from scipy.stats import ttest_rel  # type: ignore
from tqdm import tqdm  # type: ignore

from genomic_nlp.utils.common import gene_symbol_from_gencode
from genomic_nlp.utils.common import hgnc_ncbi_genes
from genomic_nlp.visualization import set_matplotlib_publication_parameters


def gencode_genes(gtf: str) -> Set[str]:
    """Get gene symbols from a gencode gtf file."""
    gtf = pybedtools.BedTool(gtf)
    genes = list(gene_symbol_from_gencode(gtf))
    return set(genes)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Take the sigmoid of x."""
    return 1 / (1 + np.exp(-x))


def add_significance_bar(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    p_value: float,
    bar_height: float = 0.5,
    text_offset: float = -0.65,
) -> None:
    """Add a bar with p-value significance between two boxplots."""
    # draw vertical lines that connect to horizontal bar
    ax.plot([x1, x1], [y, y + bar_height], color="black", lw=0.5)
    ax.plot([x2, x2], [y, y + bar_height], color="black", lw=0.5)
    ax.plot([x1, x2], [y + bar_height, y + bar_height], color="black", lw=0.5)

    # format the p-value for display
    if p_value < 0.001:
        p_text = "***"
    elif p_value < 0.01:
        p_text = "**"
    elif p_value < 0.05:
        p_text = "*"
    else:
        p_text = "ns"  # not significant

    # add significance text
    ax.text(
        (x1 + x2) * 0.5,
        y + bar_height + text_offset,
        p_text,
        ha="center",
        va="bottom",
        fontsize=7,
    )


def plot_embedding_quality(
    sim_1: np.ndarray, sim_2: np.ndarray, comparison: str = "GO"
) -> None:
    """Plot the quality of the model's embeddings."""
    # get p-value
    t_stat, p_value = ttest_rel(sim_1, sim_2)
    if p_value == 0:
        p_value_title = f"$p$ < {sys.float_info.min:.2e}"
    else:
        p_value_title = f"$p$ = {p_value:.4e}"

    if comparison == "GO":
        colors = ["#CAD178", "#4B7F52"]
        title = f"GO pair context probabilities\n$t$ = {t_stat:.2f}, {p_value_title}"
    elif comparison == "coessential":
        colors = ["#78aad1", "#4b557f"]
        title = f"Coessential pair context probabilities\n$t$ = {t_stat:.2f}, {p_value_title}"
    else:
        raise ValueError(
            f"Unknown comparison: {comparison}. Must be 'GO' or 'coessential'."
        )

    # prepare the plot
    fig, ax = plt.subplots(constrained_layout=True)

    # create violin plots
    violin_parts = ax.violinplot(
        [sim_1, sim_2],
        positions=[1, 2],
        widths=0.8,
        showmeans=False,
        showextrema=False,
        showmedians=True,
    )

    # customize visuals
    for i, pc in enumerate(violin_parts["bodies"]):  # type: ignore
        pc.set_facecolor(colors[i])
        pc.set_edgecolor("black")
        pc.set_alpha(0.7)
        pc.set_linewidth(0.5)

    # customize median lines
    for partname in ("cmedians",):
        vp = violin_parts[partname]
        vp.set_edgecolor("black")
        vp.set_linewidth(0.5)

    # add internal boxplot
    boxprops = dict(linestyle="-", linewidth=0.5, color="black")
    whiskerprops = dict(linestyle="-", linewidth=0.5, color="black")
    medianprops = dict(linestyle="-", linewidth=0.5, color="black")

    ax.boxplot(
        [sim_1, sim_2],
        positions=[1, 2],
        widths=0.05,
        whiskerprops=whiskerprops,
        boxprops=boxprops,
        showfliers=False,
    )

    # adjust boxplot line width
    for line in ax.get_lines():
        line.set_linewidth(0.5)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["2023", "2003"])
    ax.set_ylabel("Probability")
    ax.set_title(title)

    # add significance bar
    y_max = max(max(sim_1), max(sim_2))
    add_significance_bar(ax, 1, 2, y_max + 0.45, p_value)

    plt.tight_layout()
    fig.set_size_inches(1.2, 1.95)
    plt.savefig(
        f"{comparison}_dotprod.png",
        dpi=450,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)


def prepare_gene_matrices(gv: Dict[str, Dict[str, np.ndarray]]) -> Tuple:
    """Prepares input and output matrices and gene index mappings.

    Args:
        gv (dict): Gene vectors with 'input' and 'output' embeddings.

    Returns:
        tuple: (genes, gene_to_idx, input_matrix, output_matrix)
    """
    genes = list(gv.keys())
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
    input_matrix = np.array([gv[gene]["input"] for gene in genes])
    output_matrix = np.array([gv[gene]["output"] for gene in genes])
    return genes, gene_to_idx, input_matrix, output_matrix


def get_context_probabilities(
    positive_pairs: List[Tuple[str, str]],
    genes: List[str],
    gene_to_idx: Dict[str, int],
    input_matrix: np.ndarray,
    output_matrix: np.ndarray,
    sigmoid_transform: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the context probabilities for positive and random pairs.

    Args:
        positive_pairs (list of tuples): List of (gene1, gene2) tuples.
        genes (list): List of gene names.
        gene_to_idx (dict): Mapping from gene names to indices.
        input_matrix (np.ndarray): Matrix of input vectors.
        output_matrix (np.ndarray): Matrix of output vectors.
        sigmoid_transform (bool): Whether to apply sigmoid to dot products.

    Returns:
        tuple: Arrays of positive and random scores.
    """
    num_pairs = len(positive_pairs)
    gene_indices = np.array(
        [gene_to_idx[gene] for pair in positive_pairs for gene in pair]
    ).reshape(num_pairs, 2)
    gene1_indices = gene_indices[:, 0]
    gene2_indices = gene_indices[:, 1]

    # compute positive scores
    positive_vec1 = input_matrix[gene1_indices]
    positive_vec2 = output_matrix[gene2_indices]
    positive_scores = np.einsum("ij,ij->i", positive_vec1, positive_vec2)

    if sigmoid_transform:
        positive_scores = sigmoid(positive_scores)

    # for each pair, exclude gene1 and gene2 from possible random genes
    # create a mask where valid genes are True
    mask = np.ones((num_pairs, len(genes)), dtype=bool)
    mask[np.arange(num_pairs)[:, None], gene1_indices] = False
    mask[np.arange(num_pairs)[:, None], gene2_indices] = False

    # count of possible random genes per pair
    valid_counts = mask.sum(axis=1)

    # select random offsets
    random_offsets = np.random.randint(0, valid_counts)

    # get the indices of valid random genes
    valid_gene_indices = np.argsort(~mask, axis=1)  # sort true values first
    random_gene_indices = valid_gene_indices[np.arange(num_pairs), random_offsets]

    # compute random scores
    random_vec = output_matrix[random_gene_indices]
    random_scores = np.einsum("ij,ij->i", positive_vec1, random_vec)

    if sigmoid_transform:
        random_scores = sigmoid(random_scores)

    return positive_scores, random_scores


def context_probability_experiment(
    model_name, gv, positive_pairs, num_runs=5, samples=10000, sigmoid_transform=False
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the context probability analysis for a given model n times and
    calculate statistical significance.

    Args:
        model_name (str): Name of the model (for printing purposes).
        gv (dict): Gene vectors with 'input' and 'output' embeddings.
        positive_pairs (list of tuples): List of (gene1, gene2) tuples.
        num_runs (int): Number of runs for robustness.
        sigmoid_transform (bool): Whether to apply sigmoid to dot products.

    Returns:
        tuple: Lists of positive and random scores aggregated over runs.
    """
    genes, gene_to_idx, input_matrix, output_matrix = prepare_gene_matrices(gv)
    all_t_stats = []
    all_p_values = []
    aggregated_positive_scores: List[float] = []
    aggregated_random_scores: List[float] = []

    for i in range(num_runs):
        # resample positive pairs
        sampled_pairs = random.sample(positive_pairs, samples)

        print(f"Run {i + 1}/{num_runs} for {model_name}")
        positive_scores, random_scores = get_context_probabilities(
            sampled_pairs,
            genes,
            gene_to_idx,
            input_matrix,
            output_matrix,
            sigmoid_transform,
        )
        aggregated_positive_scores.extend(positive_scores)
        aggregated_random_scores.extend(random_scores)
        t_stat, p_value = ttest_rel(positive_scores, random_scores)
        all_t_stats.append(t_stat)
        all_p_values.append(p_value)

    # calculate statistics
    mean_t_stat = np.mean(all_t_stats)
    std_t_stat = np.std(all_t_stats, ddof=1)
    mean_p_value = np.mean(all_p_values)
    std_p_value = np.std(all_p_values, ddof=1)

    # calculate 95% confidence interval for t-statistic
    ci_low, ci_high = t.interval(
        0.95,
        len(all_t_stats) - 1,
        loc=mean_t_stat,
        scale=std_t_stat / np.sqrt(len(all_t_stats)),
    )

    # print results
    print(f"\n=== {model_name} Analysis Results ===")
    print(
        f"T-statistic: Mean = {mean_t_stat:.2f}, Std = {std_t_stat:.2f}, "
        f"95% CI = [{ci_low:.2f}, {ci_high:.2f}]"
    )
    print(f"P-value: Mean = {mean_p_value:.4e}, Std = {std_p_value:.4e}\n")

    return np.array(aggregated_positive_scores), np.array(aggregated_random_scores)


def parse_arguments() -> argparse.Namespace:
    """Parse args for embedding valuation on GO and coessential pairs."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model1",
        type=str,
        help="Path to the first Word2Vec model.",
        default="word2vec_300_dimensions_2023.model",
    )
    parser.add_argument(
        "--model2",
        type=str,
        help="Path to the second Word2Vec model.",
        default="word2vec_300_dimensions_2003.model",
    )
    parser.add_argument(
        "--coess",
        type=str,
        help="Path to the coessential pairs file.",
        default="/Users/steveho/genomic_nlp/development/training_data/coessential_pairs.txt",
    )
    parser.add_argument(
        "--gencode",
        type=str,
        help="Path to the gencode GTF file.",
        default="gencode.v45.basic.annotation.gtf",
    )
    parser.add_argument(
        "--hgnc",
        type=str,
        help="Path to the HGNC gene file.",
        default="hgnc_complete_set.txt",
    )
    parser.add_argument(
        "--ncbi",
        type=str,
        help="Path to the NCBI gene file.",
        default="ncbi_genes.tsv",
    )
    parser.add_argument(
        "--go",
        type=str,
        help="Path to the GO graph file.",
        default="go_graph.pkl",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=5,
        help="Number of runs for robustness.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    """Code for analyzing gene coessentiality

    1. For all gene symbols in gencode, we collect present gene vectors
    2. We filter the dataset of interest for pairs of genes that are present in
       both models
    3. Given each dataset pair, we calculate the context probability and keep all
       pairs where a dataset pair is the top hit
    4. We plot ROC
    """
    set_matplotlib_publication_parameters()
    # # parse arguments
    # args = parse_arguments()

    # # set runs and random seed
    # num_runs = args.num_runs
    # random.seed(args.seed)
    # np.random.seed(args.seed)

    num_runs = 5
    random.seed(42)
    np.random.seed(42)

    go_graph_file = "go_graph.pkl"
    coess = "/Users/steveho/genomic_nlp/development/training_data/coessential_pairs.txt"
    model1_path = "word2vec_300_dimensions_2023.model"
    model2_path = "word2vec_300_dimensions_2003.model"

    # load genes
    gencode = gencode_genes(gtf="gencode.v45.basic.annotation.gtf")
    hgnc = hgnc_ncbi_genes("hgnc_complete_set.txt", hgnc=True)
    ncbi = hgnc_ncbi_genes("ncbi_genes.tsv")
    genes = gencode.union(hgnc).union(ncbi)
    genes = {gene.casefold() for gene in genes}

    with open(go_graph_file, "rb") as file:
        go = pickle.load(file)

    go = {(k[0].casefold(), k[1].casefold()) for k in go}
    genes = {k[0] for k in go}.union({k[1] for k in go})

    # load Word2Vec models
    print("Loading Word2Vec models...")
    model1 = Word2Vec.load(model1_path)
    model2 = Word2Vec.load(model2_path)

    # ensure that syn1neg attribute exists
    if not hasattr(model1, "syn1neg"):
        print(
            "Model1 does not have syn1neg attribute. Please ensure it was trained with negative sampling."
        )
        sys.exit(1)
    if not hasattr(model2, "syn1neg"):
        print(
            "Model2 does not have syn1neg attribute. Please ensure it was trained with negative sampling."
        )
        sys.exit(1)

    # extract gene vectors for both models
    print("Extracting gene vectors for Model 1...")
    gv1 = {}
    for gene in genes:
        if gene in model1.wv.key_to_index:
            idx = model1.wv.key_to_index[gene]
            gv1[gene] = {"input": model1.wv[gene], "output": model1.syn1neg[idx]}
    print(f"Genes in Model 1: {len(gv1)}")

    print("Extracting gene vectors for Model 2...")
    gv2 = {}
    for gene in genes:
        if gene in model2.wv.key_to_index:
            idx = model2.wv.key_to_index[gene]
            gv2[gene] = {"input": model2.wv[gene], "output": model2.syn1neg[idx]}
    print(f"Genes in Model 2: {len(gv2)}")

    # identify common genes present in both models
    common_genes = set(gv1.keys()).intersection(gv2.keys())
    print(f"Common genes in both models: {len(common_genes)}")

    # filter go pairs and retain only those present in both models
    print("Filtering go pairs present in both models...")
    pairs = {
        (gene1, gene2)
        for gene1, gene2 in go
        if gene1 in common_genes and gene2 in common_genes
    }
    positive_pairs = list(pairs)

    # run analysis for both models with sigmoid transformation
    print("Starting analysis for Model 1...")
    go_2023, _ = context_probability_experiment(
        "Model1", gv1, positive_pairs, samples=10000
    )

    print("Starting analysis for Model 2...")
    go_2003, _ = context_probability_experiment(
        "Model2", gv2, positive_pairs, samples=10000
    )
    # plot embedding quality
    plot_embedding_quality(
        sim_1=go_2023,
        sim_2=go_2003,
        comparison="GO",
    )

    # read coessential pairs and retain only those present in both models
    print("Filtering coessential pairs present in both models...")
    coess_pairs = set()
    with open(coess, "r") as file:
        reader = csv.reader(file, delimiter="\t")
        for line in reader:
            if len(line) < 3:
                continue
            gene_1, gene_2, relation = line
            gene_1_cf = gene_1.casefold()
            gene_2_cf = gene_2.casefold()
            if gene_1_cf in common_genes and gene_2_cf in common_genes:
                coess_pairs.add((gene_1_cf, gene_2_cf, relation))

    print(f"Total coessential pairs after filtering: {len(pairs)}")

    # extract positive pairs
    coess_positive_pairs = [
        (gene1, gene2)
        for gene1, gene2, relation in coess_pairs
        if relation.lower() == "pos"
    ]
    print(f"Total positive pairs: {len(coess_positive_pairs)}")

    # run analysis for both models with sigmoid transformation
    print("Starting analysis for Model 1...")
    coess_2023, _ = context_probability_experiment(
        "Model1", gv1, coess_positive_pairs, sigmoid_transform=False
    )

    print("Starting analysis for Model 2...")
    coess_2003, _ = context_probability_experiment(
        "Model2", gv2, coess_positive_pairs, sigmoid_transform=False
    )

    print("Analysis completed for both models.")
    set_matplotlib_publication_parameters()
    plot_embedding_quality(
        sim_1=coess_2023,
        sim_2=coess_2003,
        comparison="coessential",
    )
