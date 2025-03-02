# sourcery skip: avoid-single-character-names-variables, name-type-suffix, require-return-annotation
#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Quick and dirty script to train XGBoost and Logistic Regression models on the
GDA dataset."""

import argparse
import csv
from pathlib import Path
import pickle
import random
from typing import Dict, List, Set, Tuple

from gensim.models import Word2Vec  # type: ignore
import numpy as np
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import average_precision_score  # type: ignore
from sklearn.metrics import precision_recall_curve  # type: ignore
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve  # type: ignore
import torch
from torch_geometric.data import Data  # type: ignore
from xgboost import XGBClassifier

from genomic_nlp.gda_data_preprocessor import GDADataPreprocessor


def build_edge_features(
    edge_index: torch.Tensor,
    data: Data,
    embeddings: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given an edge_index of shape [2, E], extract each edge's gene + disease embeddings
    and concatenate them into a single feature vector. Returns X (features) and
    a dummy array.
    """
    edges_np = edge_index.numpy().T
    X = []
    for src, dst in edges_np:
        src_str = data.inv_node_mapping[src]
        dst_str = data.inv_node_mapping[dst]
        vec_src = embeddings[src_str]
        vec_dst = embeddings[dst_str]
        X.append(np.concatenate([vec_src, vec_dst]))
    X = np.array(X)  # type: ignore
    return X, np.zeros(len(X))  # type: ignore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--year", type=int, default=2007, help="Which year's data to load."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/ocean/projects/bio210019p/stevesho/genomic_nlp/models/disease",
    )
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = "/ocean/projects/bio210019p/stevesho/genomic_nlp/models/w2v"
    w2vmodel_file = (
        f"{embedding_path}/{args.year}/word2vec_300_dimensions_{args.year}.model"
    )
    w2v_model = Word2Vec.load(w2vmodel_file)
    embeddings = {word: w2v_model.wv[word] for word in w2v_model.wv.index_to_key}

    text_path = "/ocean/projects/bio210019p/stevesho/genomic_nlp/training_data/disease"
    text_edges_file = f"{text_path}/gda_co_occurence_{args.year}.tsv"
    test_file = f"{text_path}/gda_normalized_sorted.txt"
    preprocessor = GDADataPreprocessor(
        text_edges_file=text_edges_file,
        embeddings=embeddings,
    )
    data, test_pos_edge_index, test_neg_edge_index = preprocessor.preprocess_data()

    # build new test set from external file
    test_positive_pairs: List[Tuple[str, str]] = []
    with open(test_file, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            gene_raw, disease_raw, year_str = row[0], row[1], row[2]
            try:
                year = int(year_str)
            except ValueError:
                continue
            if year > args.year:
                gene = gene_raw.casefold()
                disease = disease_raw.casefold()
                if gene in embeddings and disease in embeddings:
                    test_positive_pairs.append((gene, disease))
    test_positive_pairs = list(set(test_positive_pairs))
    print(
        f"[INFO] After removing duplicates, {len(test_positive_pairs)} positive test pairs remain."
    )

    all_genes = [k for k in embeddings if k in preprocessor.available_genes]
    all_diseases = [k for k in embeddings if k in preprocessor.available_diseases]
    test_positive_set = set(test_positive_pairs)
    num_pos = len(test_positive_pairs)
    test_negative_pairs = []

    batch_size = 20000
    max_iter = 100000  # safety limit
    i_iter = 0
    while len(test_negative_pairs) < num_pos and i_iter < max_iter:
        i_iter += 1
        genes_random = random.choices(all_genes, k=batch_size)
        diseases_random = random.choices(all_diseases, k=batch_size)
        for g, d in zip(genes_random, diseases_random):
            if (g, d) not in test_positive_set:
                test_negative_pairs.append((g, d))
                if len(test_negative_pairs) >= num_pos:
                    break
    print(f"[INFO] Created {len(test_negative_pairs)} negative test pairs.")

    def build_features(pairs: List[Tuple[str, str]]) -> np.ndarray:
        X_list = []
        for g, d in pairs:
            vec_g = embeddings[g]
            vec_d = embeddings[d]
            X_list.append(np.concatenate([vec_g, vec_d]))
        return np.array(X_list)

    X_test_pos = build_features(test_positive_pairs)
    X_test_neg = build_features(test_negative_pairs)
    y_test_pos = np.ones(len(X_test_pos), dtype=int)
    y_test_neg = np.zeros(len(X_test_neg), dtype=int)
    X_test = np.concatenate([X_test_pos, X_test_neg], axis=0)
    y_test = np.concatenate([y_test_pos, y_test_neg], axis=0)

    # build training features from train+val splits
    train_pos = torch.cat([data.train_pos_edge_index, data.val_pos_edge_index], dim=1)
    train_neg = torch.cat([data.train_neg_edge_index, data.val_neg_edge_index], dim=1)
    X_pos, _ = build_edge_features(train_pos, data, embeddings)
    X_neg, _ = build_edge_features(train_neg, data, embeddings)
    y_pos = np.ones(len(X_pos), dtype=int)
    y_neg = np.zeros(len(X_neg), dtype=int)
    X_train = np.concatenate([X_pos, X_neg], axis=0)
    y_train = np.concatenate([y_pos, y_neg], axis=0)

    print(f"Train pos: {len(X_pos)}, neg: {len(X_neg)}; total train: {len(X_train)}")
    print(
        f"Test pos:  {len(X_test_pos)}, neg: {len(X_test_neg)}; total test:  {len(X_test)}"
    )

    # train XGBoost
    xgb_clf = XGBClassifier(
        eval_metric="aucpr",
        n_estimators=300,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        seed=42,
        reg_lambda=1,
    )
    print("Training XGBoost...")
    xgb_clf.fit(X_train, y_train)
    probs_test = xgb_clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs_test)
    ap = average_precision_score(y_test, probs_test)
    pr_curve = precision_recall_curve(y_test, probs_test)
    print(f"Test AUC = {auc:.4f}")
    print(f"Test Average Precision = {ap:.4f}")
    np.save(save_dir / f"pr_curve_{args.year}.npy", pr_curve)
    model_path = save_dir / f"xgboost_gda_{args.year}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(xgb_clf, f)
    print(f"Saved XGBoost model to {model_path}")
    probs_test_named = []
    for idx, (g, d) in enumerate(test_positive_pairs + test_negative_pairs):
        probs_test_named.append((g, d, probs_test[idx]))
    with open(save_dir / f"probs_test_{args.year}.tsv", "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["gene", "disease", "probability"])
        for row in probs_test_named:
            writer.writerow(row)

    # train Logistic Regression
    lr_clf = LogisticRegression()
    print("Training Logistic Regression...")
    lr_clf.fit(X_train, y_train)
    probs_test_lr = lr_clf.predict_proba(X_test)[:, 1]
    auc_lr = roc_auc_score(y_test, probs_test_lr)
    ap_lr = average_precision_score(y_test, probs_test_lr)
    pr_curve_lr = precision_recall_curve(y_test, probs_test_lr)
    print(f"Test AUC (LR) = {auc_lr:.4f}")
    print(f"Test Average Precision (LR) = {ap_lr:.4f}")
    np.save(save_dir / f"pr_curve_lr_{args.year}.npy", pr_curve_lr)
    model_path_lr = save_dir / f"lr_gda_{args.year}.pkl"
    with open(model_path_lr, "wb") as f:
        pickle.dump(lr_clf, f)
    print(f"Saved Logistic Regression model to {model_path_lr}")
    probs_test_named_lr = []
    for idx, (g, d) in enumerate(test_positive_pairs + test_negative_pairs):
        probs_test_named_lr.append((g, d, probs_test_lr[idx]))
    with open(save_dir / f"probs_test_lr_{args.year}.tsv", "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["gene", "disease", "probability"])
        for row in probs_test_named_lr:
            writer.writerow(row)

    # random baseline
    print("Running Random Baseline...")

    class RandomBaseline:
        """
        A random baseline classifier that returns random probabilities.
        Mimics the interface of a model with a predict_proba method.
        """

        def __init__(self, random_state: int = 42):
            self.rng = np.random.default_rng(random_state)

        def fit(self, X: np.ndarray, y: np.ndarray):
            return self

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            n_samples = X.shape[0]
            probs = self.rng.random(n_samples)
            return np.column_stack((1 - probs, probs))

    random_baseline = RandomBaseline(random_state=42).fit(X_train, y_train)
    probs_test_rand = random_baseline.predict_proba(X_test)[:, 1]
    auc_rand = roc_auc_score(y_test, probs_test_rand)
    ap_rand = average_precision_score(y_test, probs_test_rand)
    pr_curve_rand = precision_recall_curve(y_test, probs_test_rand)
    print(f"Test AUC (Random Baseline) = {auc_rand:.4f}")
    print(f"Test Average Precision (Random Baseline) = {ap_rand:.4f}")
    np.save(save_dir / f"pr_curve_rand_{args.year}.npy", pr_curve_rand)
    probs_test_named_rand = []
    for idx, (g, d) in enumerate(test_positive_pairs + test_negative_pairs):
        probs_test_named_rand.append((g, d, probs_test_rand[idx]))
    with open(save_dir / f"probs_test_rand_{args.year}.tsv", "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["gene", "disease", "probability"])
        for row in probs_test_named_rand:
            writer.writerow(row)


if __name__ == "__main__":
    main()
