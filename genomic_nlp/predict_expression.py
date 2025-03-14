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

from adjustText import adjust_text  # type: ignore
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
import shap  # type: ignore
from sklearn.cluster import KMeans  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
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

    # load w2v model
    w2v = Word2Vec.load(
        "/Users/steveho/genomic_nlp/development/models/word2vec_300_dimensions_2023.model"
    )

    # load genePT
    with open(
        "/Users/steveho/genomic_nlp/development/plots/GenePT_emebdding_v2/GenePT_gene_embedding_ada_text.pickle",
        "rb",
    ) as f:
        gene_pt_raw = pickle.load(f)

    gene_pt = {
        gene.casefold(): np.array(embedding) for gene, embedding in gene_pt_raw.items()
    }

    # load attention embeddings
    with open("averaged_embeddings.pkl", "rb") as f:
        avg_embeddings = pickle.load(f)

    # add symbols to df and convert to lowercase
    df["genesymbol"] = df.index.map(lambda x: symbol_lookup.get(x, x))
    df["genesymbol"] = df["genesymbol"].map(lambda x: x.lower())
    genes_symbols = set(df["genesymbol"])

    # assign train, test, val based on chr_map
    # and filter for protein coding genes
    df["split"] = df.index.map(lambda x: assign_split(x, chr_map))
    gene_bed = BedTool(protein_coding_bed)
    protein_coding_genes = {feature[3] for feature in gene_bed if feature[3] in genes}
    df = df[df.index.isin(protein_coding_genes)]

    # filter for valid genes
    w2v_genes = {gene for gene in genes_symbols if gene in w2v.wv}
    gene_pt_genes = {gene for gene in genes_symbols if gene in gene_pt}
    attn_genes = {gene for gene in genes_symbols if gene in avg_embeddings}

    shared_genes = w2v_genes.intersection(gene_pt_genes).intersection(attn_genes)
    print(f"Total genes with all three embeddings: {len(shared_genes):,}")

    valid_genes = [gene for gene in genes_symbols if gene in w2v.wv]
    w2v_genes = {gene: w2v.wv[gene] for gene in valid_genes}

    # filter for valid genes in genePT
    valid_genes = [gene for gene in genes_symbols if gene in gene_pt]
    gene_pt_genes = {gene: gene_pt[gene] for gene in valid_genes}

    # fitler for valid genes in attention embeddings
    valid_genes = [gene for gene in genes_symbols if gene in avg_embeddings]
    attn_genes = {gene: avg_embeddings[gene] for gene in valid_genes}

    # filter for protein coding
    gene_bed = BedTool(protein_coding_bed)
    genes = [feature[3] for feature in gene_bed if feature[3] in genes]
    df = df[df.index.isin(genes)]

    # filter for shared genes
    df = df[df["genesymbol"].isin(shared_genes)]

    # check split distribution
    split_counts = df["split"].value_counts()
    print("Distribution of genes across splits after filtering for shared genes:")
    print(f"Train: {split_counts.get('train', 0):,} genes")
    print(f"Test:  {split_counts.get('test', 0):,} genes")

    # split the dataframe
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    # map embeddings
    w2v_embeddings = {gene: w2v.wv[gene] for gene in shared_genes}
    genept_embeddings = {gene: gene_pt[gene] for gene in shared_genes}
    attn_embeddings = {gene: avg_embeddings[gene] for gene in shared_genes}

    # get vectors for model training
    X_train = np.array(train_df["embedding"].tolist())
    y_train = train_df["all_tissues"].values

    X_test = np.array(test_df["embedding"].tolist())
    y_test = test_df["all_tissues"].values

    # pipe = Pipeline(
    #     [
    #         ("scaler", StandardScaler()),
    #         (
    #             "xgb",
    #             xgb.XGBRegressor(
    #                 objective="reg:squarederror",
    #                 random_state=42,
    #                 eval_metric="rmse",
    #                 tree_method="hist",
    #             ),
    #         ),
    #     ]
    # )

    # param_grid = {
    #     "xgb__n_estimators": [500, 550],
    #     "xgb__max_depth": [4],
    #     "xgb__learning_rate": [
    #         0.06,
    #         0.065,
    #         0.07,
    #     ],
    #     "xgb__subsample": [0.9, 0.95],
    #     "xgb__colsample_bytree": [0.7],
    #     "xgb__min_child_weight": [5],
    #     "xgb__gamma": [0.05, 0.1],
    #     "xgb__reg_alpha": [1, 1.2, 1.4],
    #     "xgb__reg_lambda": [1, 1.5, 2],
    # }

    # # set up grid search with 5-fold cv
    # grid_search = GridSearchCV(
    #     pipe, param_grid, cv=5, scoring="r2", verbose=1, n_jobs=8
    # )
    # grid_search.fit(X_train, y_train)

    # print("Best parameters found:")
    # print(grid_search.best_params_)

    # predictions = grid_search.predict(X_test)
    # pearson_corr, _ = pearsonr(predictions, y_test)
    # print(f"Improved Pearson Correlation: {pearson_corr:.4f}")

    # spearman_corr, _ = stats.spearmanr(predictions, y_test)
    # print(f"Improved Spearman Correlation: {spearman_corr:.4f}")

    # plot_predicted_versus_expected(
    #     predictions, y_test, savetitle="w2v_embedding_expression_prediction.png"
    # )

    # Best parameters found:
    # {'xgb__colsample_bytree': 0.7, 'xgb__gamma': 0.05, 'xgb__learning_rate': 0.06, 'xgb__max_depth': 4, 'xgb__min_child_weight': 5, 'xgb__n_estimators': 550, 'xgb__reg_alpha': 1, 'xgb__reg_lambda': 1, 'xgb__subsample': 0.9}
    # Improved Pearson Correlation: 0.6016
    # Improved Spearman Correlation: 0.5971

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
    xgb_params = {k.replace("xgb__", ""): v for k, v in best_params.items()}

    # # get random vectors using xavier uniform init
    # embedding_dimension = X_train.shape[1]
    # num_train_genes = X_train.shape[0]
    # X_train_random = xavier_uniform_initialization(
    #     (num_train_genes, embedding_dimension)
    # )

    # pipe_random_embedding = Pipeline(
    #     [
    #         ("scaler", StandardScaler()),
    #         (
    #             "xgb",
    #             xgb.XGBRegressor(
    #                 objective="reg:squarederror",
    #                 random_state=42,
    #                 eval_metric="rmse",
    #                 tree_method="hist",
    #                 **xgb_params,
    #             ),
    #         ),
    #     ]
    # )

    # print("Training XGBoost with RANDOM embeddings...")
    # pipe_random_embedding.fit(X_train_random, y_train)

    # print("Making predictions with RANDOM embeddings...")
    # predictions_random = pipe_random_embedding.predict(X_test)

    # pearson_corr_random, _ = pearsonr(predictions_random, y_test)
    # print(f"Pearson Correlation with RANDOM Embeddings: {pearson_corr_random:.4f}")

    # spearman_corr_random, _ = stats.spearmanr(predictions_random, y_test)
    # print(f"Spearman Correlation with RANDOM Embeddings: {spearman_corr_random:.4f}")

    # plot_predicted_versus_expected(
    #     predictions_random,
    #     y_test,
    #     savetitle="random_embedding_expression_prediction.png",
    # )
    # predictions_genept = final_model.predict(X_test)
    # predictions_attn = final_model.predict(X_test)  # type: ignore

    final_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "xgb",
                xgb.XGBRegressor(
                    objective="reg:squarederror",
                    random_state=42,
                    eval_metric="rmse",
                    tree_method="hist",
                    **xgb_params,
                ),
            ),
        ]
    )

    final_model.fit(X_train, y_train)
    pearson_corr_final, _ = pearsonr(final_model.predict(X_test), y_test)
    print(f"Pearson Correlation with FINAL model: {pearson_corr_final:.4f}")

    shap.initjs()
    try:
        explainer = shap.TreeExplainer(final_model.named_steps["xgb"])
        X_train_scaled = final_model.named_steps["scaler"].transform(X_train)
        shap_values = explainer.shap_values(X_train_scaled)
        print("[SHAP] Generating summary plot...")

        set_matplotlib_publication_parameters()
        plt.figure()
        shap.summary_plot(shap_values, X_train_scaled, show=False, plot_size=(3, 3.5))
        fig = plt.gcf()
        for ax in fig.get_axes():
            ax.tick_params(axis="both", which="major", labelsize=7)
            ax.tick_params(axis="both", which="minor", labelsize=7)
            ax.xaxis.label.set_fontsize(7)
            ax.yaxis.label.set_fontsize(7)
            if ax.get_title():
                ax.title.set_fontsize(7)
        # fig.set_size_inches(4, 3)
        plt.tight_layout()
        plt.savefig("shap_summary_plot.png", dpi=450)
        plt.close()
    except Exception as e:
        print(f"[SHAP] Error: {e}")

    with open("xgboost_final_2023.pkl", "wb") as f:
        pickle.dump(final_model, f)

    # save training features and labels
    np.save("x_train.npy", X_train)
    np.save("y_train.npy", y_train)

    # save shap vals
    if "shap_values" in locals() and shap_values is not None:
        np.save("shap_values.npy", shap_values)

    # get 100 best predicted genes >=1 log2 TPM
    test_df = df[df["split"] == "test"].copy()
    predictions_test = final_model.predict(X_test)
    test_df["predicted_log2TPM"] = predictions_test
    test_df["abs_error"] = abs(test_df["predicted_log2TPM"] - test_df["all_tissues"])

    filtered_df = test_df[test_df["predicted_log2TPM"] >= 1]
    top100_best = filtered_df.sort_values(by="abs_error", ascending=True).head(100)

    # fully capitalize the gene symbols
    top100_best["genesymbol"] = top100_best["genesymbol"].map(lambda x: x.upper())

    # print top 100 genes
    print("Top 100 best predicted genes:")
    for gene in top100_best["genesymbol"]:
        print(gene)

    # 1. Compute average absolute SHAP importances from your expression model's training data
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)  # shape: (n_features,)
    top_k = 10  # number of top features
    top_features_idx = np.argsort(mean_abs_shap)[::-1][:top_k]
    print("Top feature indices (from expression model SHAP):", top_features_idx)
    print("Corresponding mean absolute SHAP values:", mean_abs_shap[top_features_idx])

    # 2. Compute a reference vector from the expression training set using these top dimensions
    reference_vector = np.mean(X_train_scaled[:, top_features_idx], axis=0).reshape(
        1, -1
    )

    # 3. Get all embeddings from the full Word2Vec model
    all_embeddings = {gene: w2v.wv[gene] for gene in w2v.wv.index_to_key}
    genes_full = list(all_embeddings.keys())
    X_full = np.array(list(all_embeddings.values()))
    print(f"Total embeddings in full Word2Vec model: {len(genes_full)}")

    # 4. Scale the full embeddings using the scaler from your final model
    X_full_scaled = final_model.named_steps["scaler"].transform(X_full)

    # 5. Subset the scaled full embeddings to only the top features
    X_full_top = X_full_scaled[:, top_features_idx]

    # 6. Compute cosine similarity between each gene's vector and the reference vector
    similarity_scores = cosine_similarity(X_full_top, reference_vector).flatten()

    # Diagnostic: Print range and a simple histogram of similarity scores
    print(
        "Similarity scores range: min =",
        similarity_scores.min(),
        ", max =",
        similarity_scores.max(),
    )
    plt.figure(figsize=(6, 4))
    sns.histplot(similarity_scores, bins=50, kde=True)
    plt.title("Distribution of Cosine Similarity Scores")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("cosine_similarity_scores.png", dpi=450)
    plt.close()

    # Option B: Alternatively, select the top-N genes by similarity
    top_N = 100  # select top 100 genes
    filtered_indices_topN = np.argsort(similarity_scores)[::-1][:top_N]
    print("Number of genes selected by top-N strategy:", len(filtered_indices_topN))

    # filter
    filtered_indices = filtered_indices_topN

    # Get filtered genes and their corresponding embeddings in the top-feature space
    filtered_genes = [genes_full[i] for i in filtered_indices]
    filtered_embeddings = X_full_top[filtered_indices]
    print(f"Number of genes filtered: {len(filtered_genes)}")

    # 7. Cluster the filtered embeddings using K-Means
    num_clusters = 3  # or choose based on your domain knowledge
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    filtered_cluster_labels = kmeans.fit_predict(filtered_embeddings)

    # 8. Visualize the filtered and clustered genes using PCA to reduce to 2D
    pca = PCA(n_components=2, random_state=42)
    filtered_pca = pca.fit_transform(filtered_embeddings)

    # names to annotate
    special_genes = [
        "microvascular_angina",
        "thalassemia",
        "foxb2",
        "apototic",
        "nfe2l1",
        "hudep",
        "linc01774",
        "linc00886",
        "immunopotentiators",
        "azoramide",
    ]

    plt.figure(figsize=(4, 3))
    sns.scatterplot(
        x=filtered_pca[:, 0],
        y=filtered_pca[:, 1],
        hue=filtered_cluster_labels,
        palette="viridis",
        alpha=1,
        s=10,
        edgecolor=None,
        legend=False,
    )

    # add text annotations
    texts = []
    texts.extend(
        plt.text(filtered_pca[i, 0], filtered_pca[i, 1], gene)
        for i, gene in enumerate(filtered_genes)
        if gene in special_genes
    )
    # Then let adjust_text do the work
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5))

    plt.title("Embeddings similar to expression model top features")
    plt.xlabel("First principal component")
    plt.ylabel("Second principal component")
    plt.tight_layout()
    plt.savefig("clusters_filtered_by_similarity.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
