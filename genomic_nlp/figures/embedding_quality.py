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

from genomic_nlp.utils.common import gene_symbol_from_gencode
from genomic_nlp.visualization import set_matplotlib_publication_parameters


def gencode_genes(gtf: str) -> Set[str]:
    """Get gene symbols from a gencode gtf file."""
    gtf = pybedtools.BedTool(gtf)
    genes = list(gene_symbol_from_gencode(gtf))
    return set(genes)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Take the sigmoid of x."""
    return 1 / (1 + np.exp(-x))


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
):
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
    positive_vec1 = input_matrix[gene1_indices]  # Shape: (num_pairs, vector_dim)
    positive_vec2 = output_matrix[gene2_indices]  # Shape: (num_pairs, vector_dim)
    positive_scores = np.einsum("ij,ij->i", positive_vec1, positive_vec2)  # Dot product

    if sigmoid_transform:
        positive_scores = sigmoid(positive_scores)

    # prepare for random gene selection
    all_gene_indices = np.arange(len(genes))

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
    valid_gene_indices = np.argsort(~mask, axis=1)  # Sort True values first
    random_gene_indices = valid_gene_indices[np.arange(num_pairs), random_offsets]

    # compute random scores
    random_vec = output_matrix[random_gene_indices]
    random_scores = np.einsum("ij,ij->i", positive_vec1, random_vec)

    if sigmoid_transform:
        random_scores = sigmoid(random_scores)

    return positive_scores, random_scores


def context_probability_experiment(
    model_name, gv, positive_pairs, num_runs=5, sigmoid_transform=False
):
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
    aggregated_positive_scores = []
    aggregated_random_scores = []

    for i in range(num_runs):
        print(f"Run {i + 1}/{num_runs} for {model_name}")
        positive_scores, random_scores = get_context_probabilities(
            positive_pairs,
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
        required=True,
        help="Path to the first Word2Vec model.",
        default="word2vec_300_dimensions_2023.model",
    )
    parser.add_argument(
        "--model2",
        type=str,
        required=True,
        help="Path to the second Word2Vec model.",
        default="word2vec_300_dimensions_2003.model",
    )
    parser.add_argument(
        "--coess",
        type=str,
        required=True,
        help="Path to the coessential pairs file.",
        default="coessential_pairs.txt",
    )
    parser.add_argument(
        "--gencode",
        type=str,
        required=True,
        help="Path to the gencode GTF file.",
        default="gencode.v45.basic.annotation.gtf",
    )
    parser.add_argument(
        "--hgnc",
        type=str,
        required=True,
        help="Path to the HGNC gene file.",
        default="hgnc_complete_set.txt",
    )
    parser.add_argument(
        "--ncbi",
        type=str,
        required=True,
        help="Path to the NCBI gene file.",
        default="ncbi_genes.tsv",
    )
    parser.add_argument(
        "--go",
        type=str,
        required=True,
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
    # parse arguments
    args = parse_arguments()

    # set runs and random seed
    num_runs = args.num_runs
    random.seed(args.seed)
    np.random.seed(args.seed)

    # gencode = gencode_genes(gtf="gencode.v45.basic.annotation.gtf")
    # hgnc = hgnc_ncbi_genes("hgnc_complete_set.txt", hgnc=True)
    # ncbi = hgnc_ncbi_genes("ncbi_genes.tsv")
    # genes = gencode.union(hgnc).union(ncbi)
    # genes = {gene.casefold() for gene in genes}

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

    # read coessential pairs and retain only those present in both models
    # print("Filtering coessential pairs present in both models...")
    # pairs = set()
    # with open(coess, "r") as file:
    #     reader = csv.reader(file, delimiter="\t")
    #     for line in reader:
    #         if len(line) < 3:
    #             continue  # Skip malformed lines
    #         gene_1, gene_2, relation = line
    #         gene_1_cf = gene_1.casefold()
    #         gene_2_cf = gene_2.casefold()
    #         if gene_1_cf in common_genes and gene_2_cf in common_genes:
    #             pairs.add((gene_1_cf, gene_2_cf, relation))

    # print(f"Total coessential pairs after filtering: {len(pairs)}")

    # filter go pairs and retain only those present in both models
    print("Filtering go pairs present in both models...")
    pairs = set()
    for gene1, gene2 in go:
        if gene1 in common_genes and gene2 in common_genes:
            pairs.add((gene1, gene2))

    # extract positive pairs
    # positive_pairs = [
    #     (gene1, gene2) for gene1, gene2, relation in pairs if relation.lower() == "pos"
    # ]
    # print(f"Total positive pairs: {len(positive_pairs)}")
    positive_pairs = list(pairs)
    positive_pairs = random.sample(positive_pairs, 500000)

    # run analysis for both models with sigmoid transformation
    print("Starting analysis for Model 1...")
    m1_pos, m1_random = context_probability_experiment(
        "Model1", gv1, positive_pairs, num_runs=num_runs, sigmoid_transform=False
    )

    print("Starting analysis for Model 2...")
    m2_pos, m2_random = context_probability_experiment(
        "Model2", gv2, positive_pairs, num_runs=num_runs, sigmoid_transform=False
    )

    print("Analysis completed for both models.")

    set_matplotlib_publication_parameters()

    positive_sims = m1_pos
    random_sims = m1_random
    model_year = 2023

    # positive_sims = m2_pos
    # random_sims = m2_random
    # model_year = 2003

    t_stat, p_value = ttest_rel(positive_sims, random_sims)
    if p_value == 0:
        p_value_str = f"p < {sys.float_info.min:.2e}"
    else:
        p_value_str = f"p = {p_value:.4e}"

    # prepare the plot
    fig, ax = plt.subplots(constrained_layout=True)
    colors = ["#D43F3A", "#5CB85C"]  # Red for positive_sims, Green for random_sims

    # create violin plots
    violin_parts = ax.violinplot(
        [positive_sims, random_sims],
        positions=[1, 2],
        widths=0.8,  # adjust the width of the violins
        showmeans=False,
        showextrema=False,
        showmedians=True,
    )

    # customize the violins
    for i, pc in enumerate(violin_parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor("black")
        pc.set_alpha(0.7)

    # customize median lines
    for partname in ("cmedians",):
        vp = violin_parts[partname]
        vp.set_edgecolor("black")
        vp.set_linewidth(1.5)

    # add box plots inside violins
    boxprops = dict(linestyle="-", linewidth=1, color="black")
    whiskerprops = dict(linestyle="-", linewidth=1, color="black")
    medianprops = dict(linestyle="-", linewidth=1, color="black")

    ax.boxplot(
        [positive_sims, random_sims],
        positions=[1, 2],
        widths=0.05,  # Width of the boxes
        whiskerprops=whiskerprops,
        boxprops=boxprops,
        medianprops=medianprops,
        showfliers=False,
    )

    # # customize box colors
    # colors = ["#4C72B0", "#55A868"]
    # for patch, color in zip(bp["boxes"], colors):
    #     patch.set_facecolor(color)

    # # set labels and title
    # ax.set_xticks([1, 2])
    # ax.set_xticklabels(["Coessential", "Random"])
    # ax.set_ylabel("Probability")
    # ax.set_title(
    #     f"Comparison of context probabilities\nModel through year {model_year}\nt = {t_stat:.2f}, {p_value_str}"
    # )

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["GO", "Random"])
    ax.set_ylabel("Probability")
    ax.set_title(
        f"Comparison of context probabilities\nModel through year {model_year}\nt = {t_stat:.2f}, {p_value_str}"
    )

    # add significance bar function
    def add_significance_bar(ax, x1, x2, y, p_value, bar_height=0.1, text_offset=-0.3):
        """Add a bar with p-value significance between two boxplots."""
        bar_x = [x1, x1, x2, x2]
        bar_y = [y, y + bar_height, y + bar_height, y]
        ax.plot(bar_x, bar_y, color="black", lw=0.5)

        # format the p-value for display
        if p_value < 0.001:
            p_text = "***"
        elif p_value < 0.01:
            p_text = "**"
        elif p_value < 0.05:
            p_text = "*"
        else:
            p_text = "ns"  # not significant

        ax.text(
            (x1 + x2) * 0.5,
            y + bar_height + text_offset,
            p_text,
            ha="center",
            va="bottom",
            fontsize=7,
        )

    y_max = max(max(positive_sims), max(random_sims))
    add_significance_bar(ax, 1, 2, y_max, p_value)

    # adjust layout for better aesthetics
    plt.tight_layout()

    # save the figure
    fig.set_size_inches(1.75, 3)
    # plt.savefig(
    #     f"coessentiality_dotprod_{model_year}.png",
    #     dpi=450,
    #     bbox_inches="tight",
    #     pad_inches=0.02,
    # )
    plt.savefig(
        f"go_dotprod_{model_year}.png",
        dpi=450,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)
