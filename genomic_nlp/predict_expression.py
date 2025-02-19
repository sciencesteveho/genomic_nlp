# sourcery skip: lambdas-should-be-short
#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""
1. Get the array of average activity - transform log2(TPM + 0.25)
2. Get the gene - chr map
3. Load embeddings (w2v and n2v)
4. Filter embeddings for genes
4. Filter gene - chr map for genes
5. Use 8,9 as holdout test set. 10 for validation
6. Print number of genes for train, test, val
6. Run XGBoost model, train on rest of chrs. Test on 8,9
"""


from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from gensim.models import Word2Vec  # type: ignore
from matplotlib.colors import LogNorm  # type: ignore
from matplotlib.figure import Figure  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
from pybedtools import BedTool  # type: ignore
from scipy import stats  # type: ignore
from scipy.stats import pearsonr  # type: ignore
import seaborn as sns  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
from sklearn.model_selection import GridSearchCV  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
import xgboost as xgb

from genomic_nlp.visualization import set_matplotlib_publication_parameters

PSEUDOCOUNT = 0.25


def xavier_uniform_initialization(shape: Tuple[int, int]) -> np.ndarray:
    """Initializes a weight matrix using xavier uniform initialization."""
    in_dim, out_dim = shape
    limit = np.sqrt(6.0 / (in_dim + out_dim))
    return np.random.uniform(-limit, limit, size=(in_dim, out_dim))


def get_genes(
    gencode_bed: Path,
) -> List[str]:
    """Filter rna_seq data by TPM"""
    gencode = BedTool(gencode_bed)
    return [
        feature[3]
        for feature in gencode
        if "protein_coding" in str(feature)
        and feature[0] not in ["chrX", "chrY", "chrM"]
    ]


def gene_chr_map(gencode_bed: Path) -> Dict[str, List[str]]:
    """Create a dictionary of chromosome: gene list"""
    gencode = BedTool(gencode_bed)
    chr_map: Dict[str, List[str]] = {}
    for feature in gencode:
        gene = feature[3]
        chrom = feature[0]
        if chrom not in chr_map:
            chr_map[chrom] = []
        chr_map[chrom].append(gene)
    return chr_map


def read_encode_rna_seq_data(
    rna_seq_file: str,
) -> pd.DataFrame:
    """Read an ENCODE rna-seq tsv, keep only ENSG genes"""
    df = pd.read_table(rna_seq_file, index_col=0, header=[0])
    return df[df.index.str.contains("ENSG")]


def plot_predicted_versus_expected(
    predicted: np.ndarray,
    expected: np.ndarray,
    savetitle: str,
) -> None:
    """Plots predicted versus expected values for a given model"""
    set_matplotlib_publication_parameters()

    # calculate Pearson and Spearman correlations
    pearson_r, _ = stats.pearsonr(expected, predicted)

    # jointplot - hexbin scatter with marginal histograms
    plot = sns.jointplot(
        x=predicted,
        y=expected,
        kind="scatter",
        height=3,
        ratio=4,
        space=0,
        s=5,
        edgecolor=None,
        linewidth=0,
        alpha=0.75,
        marginal_kws={
            "element": "step",
            "color": "lightsteelblue",
            "edgecolors": "lightslategray",
            "linewidth": 0,
        },
    )
    # set labels and title
    x_label = r"Predicted Log$_2$ TPM"
    y_label = r"Median Log$_2$ TPM in GTEx v8"

    plot.ax_joint.set_xlabel(x_label)
    plot.ax_joint.set_ylabel(y_label)

    # add best fit line
    slope, intercept = np.polyfit(predicted, expected, 1)
    x_range = np.linspace(min(predicted), max(predicted), 100)
    plot.ax_joint.plot(
        x_range,
        slope * x_range + intercept,
        "r-",
        linewidth=0.75,
        color="darkred",
        linestyle="--",
    )

    # adjust axis limits to include all data points
    plot.ax_joint.set_xlim(min(predicted), max(predicted))
    plot.ax_joint.set_ylim(min(expected), max(expected))

    # add pearson R
    plot.ax_joint.text(
        0.85,
        0.15,
        r"$\mathit{r}$ = " + f"{pearson_r:.4f}",
        transform=plot.ax_joint.transAxes,
        fontsize=7,
        verticalalignment="top",
    )
    plt.tight_layout()
    plt.savefig(savetitle, dpi=450, bbox_inches="tight")
    plt.clf()
    plt.close()


def assign_split(row_id: str, chr_map: Dict[str, List[str]]) -> str:
    """Assigns a split to a gene based on chromosome"""
    test_chrs = ["chr1"]
    val_chrs = [""]

    gene_chr = next(
        (chr_name for chr_name, genes in chr_map.items() if row_id in genes),
        None,
    )

    # assign split based on chromosome
    if gene_chr in test_chrs:
        return "test"
    elif gene_chr in val_chrs:
        return "val"
    else:
        return "train"


def load_gencode_lookup(filepath: str) -> Dict[str, str]:
    """Load the Gencode-to-gene-symbol lookup table."""
    gencode_to_symbol = {}
    with open(filepath, "r") as f:
        for line in f:
            gencode, symbol = line.strip().split("\t")
            gencode_to_symbol[symbol] = gencode
    return gencode_to_symbol


def main() -> None:
    """Main function"""
    working_dir = "/Users/steveho/genomic_nlp/development/expression"
    gencode_bed = "/Users/steveho/ogl/development/recap/gencode_v26_genes_only_with_GTEx_targets.bed"
    avg_activity_pkl = f"{working_dir}/gtex_tpm_median_across_all_tissues.pkl"
    protein_coding_bed = f"{working_dir}/gencode_v26_protein_coding.bed"
    gencode_to_genesymbol = "/Users/steveho/gnn_plots/graph_resources/local/gencode_to_genesymbol_lookup_table.txt"

    gencode_lookup = load_gencode_lookup(gencode_to_genesymbol)
    genes = get_genes(Path(gencode_bed))
    genes = [gene for gene in genes if "_" not in gene]
    chr_map = gene_chr_map(Path(gencode_bed))

    # reverse gencode lookup
    symbol_lookup = {v: k for k, v in gencode_lookup.items()}

    # load average activity data
    df = pd.read_pickle(avg_activity_pkl)

    # add pseudocount to TPM values and log2 transform
    # to "average" column
    df["all_tissues"] = np.log2(df["all_tissues"] + PSEUDOCOUNT)

    # load n2v embeddings
    # with open("input_embeddings.pkl", "rb") as f:
    #     embeddings = pickle.load(f)
    # # upper case all gene names
    # embeddings = {gene.upper(): embedding for gene, embedding in embeddings.items()}
    # embedding_genes = {gene.upper() for gene in embeddings}

    # load w2v model
    w2v = Word2Vec.load("word2vec_300_dimensions_2023.model")

    # add symbols to df
    df["genesymbol"] = df.index.map(lambda x: symbol_lookup.get(x, x))
    genes_symbols = {gene.lower() for gene in list(df["genesymbol"])}

    # filter for valid genes
    valid_genes = [gene for gene in genes_symbols if gene in w2v.wv]
    w2v_genes = {gene: w2v.wv[gene] for gene in valid_genes}

    # assign train, test, val based on chr_map
    df["split"] = df.index.map(lambda x: assign_split(x, chr_map))

    # filter for protein coding
    gene_bed = BedTool(protein_coding_bed)
    genes = [feature[3] for feature in gene_bed if feature[3] in genes]
    df = df[df.index.isin(genes)]

    # convert to genesymbol and filter for embeddings
    df["genesymbol"] = df.index.map(lambda x: symbol_lookup.get(x, x))

    # df = df[df["genesymbol"].isin(embedding_genes)]
    # convert to lower first
    df["genesymbol"] = df["genesymbol"].map(lambda x: x.lower())
    df = df[df["genesymbol"].isin(w2v_genes)]

    # add embeddings to df
    # df["embedding"] = df["genesymbol"].map(lambda x: embeddings.get(x))
    df["embedding"] = df["genesymbol"].map(lambda x: w2v_genes.get(x))

    # check how many train, test, val
    split_counts = df["split"].value_counts()

    print("Distribution of genes across splits:")
    print(f"Train: {split_counts['train']:,} genes")
    print(f"Test:  {split_counts['test']:,} genes")
    # print(f"Val:   {split_counts['val']:,} genes")

    # split the dataframe into training and test sets based on the 'split' column.
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    # get vectors for model training
    X_train = np.array(train_df["embedding"].tolist())
    y_train = train_df["all_tissues"].values

    X_test = np.array(test_df["embedding"].tolist())
    y_test = test_df["all_tissues"].values

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "xgb",
                xgb.XGBRegressor(
                    objective="reg:squarederror",
                    random_state=42,
                    eval_metric="rmse",
                    tree_method="hist",
                ),
            ),
        ]
    )

    param_grid = {
        "xgb__n_estimators": [500, 550],
        "xgb__max_depth": [4],
        "xgb__learning_rate": [
            0.06,
            0.065,
            0.07,
        ],
        "xgb__subsample": [0.9, 0.95],
        "xgb__colsample_bytree": [0.7],
        "xgb__min_child_weight": [5],
        "xgb__gamma": [0.05, 0.1],
        "xgb__reg_alpha": [1, 1.2, 1.4],
        "xgb__reg_lambda": [1, 1.5, 2],
    }

    # set up grid search with 5-fold cv
    grid_search = GridSearchCV(
        pipe, param_grid, cv=5, scoring="r2", verbose=1, n_jobs=8
    )
    grid_search.fit(X_train, y_train)

    print("Best parameters found:")
    print(grid_search.best_params_)

    predictions = grid_search.predict(X_test)
    pearson_corr, _ = pearsonr(predictions, y_test)
    print(f"Improved Pearson Correlation: {pearson_corr:.4f}")

    spearman_corr, _ = stats.spearmanr(predictions, y_test)
    print(f"Improved Spearman Correlation: {spearman_corr:.4f}")

    plot_predicted_versus_expected(
        predictions, y_test, savetitle="w2v_embedding_expression_prediction.png"
    )

    # run random model with best params
    best_params = {
        "xgb__colsample_bytree": 0.7,
        "xgb__gamma": 0.05,
        "xgb__learning_rate": 0.06,
        "xgb__max_depth": 4,
        "xgb__min_child_weight": 5,
        "xgb__n_estimators": 550,
        "xgb__reg_alpha": 1,
        "xgb__reg_lambda": 1,
        "xgb__subsample": 0.9,
    }

    # get random vectors using xavier uniform init
    embedding_dimension = X_train.shape[1]
    num_train_genes = X_train.shape[0]
    X_train_random = xavier_uniform_initialization(
        (num_train_genes, embedding_dimension)
    )

    pipe_random_embedding = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "xgb",
                xgb.XGBRegressor(
                    objective="reg:squarederror",
                    random_state=42,
                    eval_metric="rmse",
                    tree_method="hist",
                    **best_params,  # Use the best parameters here!
                ),
            ),
        ]
    )

    print("Training XGBoost with RANDOM embeddings...")
    pipe_random_embedding.fit(X_train_random, y_train)

    print("Making predictions with RANDOM embeddings...")
    predictions_random = pipe_random_embedding.predict(X_test)

    pearson_corr_random, _ = pearsonr(predictions_random, y_test)
    print(f"Pearson Correlation with RANDOM Embeddings: {pearson_corr_random:.4f}")

    spearman_corr_random, _ = stats.spearmanr(predictions_random, y_test)
    print(f"Spearman Correlation with RANDOM Embeddings: {spearman_corr_random:.4f}")

    plot_predicted_versus_expected(
        predictions_random,
        y_test,
        savetitle="random_embedding_expression_prediction.png",
    )


# Best parameters found:
# {'xgb__colsample_bytree': 0.7, 'xgb__gamma': 0.05, 'xgb__learning_rate': 0.06, 'xgb__max_depth': 4, 'xgb__min_child_weight': 5, 'xgb__n_estimators': 550, 'xgb__reg_alpha': 1, 'xgb__reg_lambda': 1, 'xgb__subsample': 0.9}
# Improved Pearson Correlation: 0.6016
# Improved Spearman Correlation: 0.5971


if __name__ == "__main__":
    main()
