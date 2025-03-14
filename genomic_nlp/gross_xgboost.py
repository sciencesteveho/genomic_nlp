#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Quick and dirty script to train XGBoost and Logistic Regression models on the
GDA dataset."""

import argparse
import csv
import json
from pathlib import Path
import pickle
import random
from typing import Any, Dict, List, Set, Tuple

from gensim.models import Word2Vec  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import shap  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import average_precision_score  # type: ignore
from sklearn.metrics import precision_recall_curve  # type: ignore
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve  # type: ignore
from sklearn.model_selection import KFold  # type: ignore
import torch
from torch_geometric.data import Data  # type: ignore
from xgboost import XGBClassifier

from genomic_nlp.gda_data_preprocessor import GDADataPreprocessor
from genomic_nlp.visualization import set_matplotlib_publication_parameters


class BaselineModel:
    """Wrapper for baseline models to use with SHAP."""

    def __init__(self, model):
        self.model = model


def run_shap(
    final_model: BaselineModel,
    train_features: np.ndarray,
    model_dir: str,
    model_name: str,
) -> np.ndarray:
    """Run SHAP analysis for a tree-based model."""
    explainer = shap.TreeExplainer(final_model.model)
    shap_values = explainer.shap_values(train_features)
    print("[SHAP] generating summary plot...")
    set_matplotlib_publication_parameters()
    shap.summary_plot(shap_values, train_features)
    plt.savefig(f"{model_dir}/{model_name}_shap_summary_plot.png", dpi=450)
    plt.close()
    return shap_values


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
        "--year", type=int, default=2023, help="Which year's data to load."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/ocean/projects/bio210019p/stevesho/genomic_nlp/models/disease",
    )
    parser.add_argument("--model", type=str, default="n2v")
    args = parser.parse_args()
    embedding_path = "/ocean/projects/bio210019p/stevesho/genomic_nlp/models/w2v"

    if args.model == "n2v":
        model_path = f"/ocean/projects/bio210019p/stevesho/genomic_nlp/models/n2v/disease/{args.year}/input_embeddings.pkl"
        with open(model_path, "rb") as f:
            embeddings = pickle.load(f)
    elif args.model == "w2v":
        w2vmodel_file = (
            f"{embedding_path}/{args.year}/word2vec_300_dimensions_{args.year}.model"
        )
        w2v_model = Word2Vec.load(w2vmodel_file)
        embeddings = {word: w2v_model.wv[word] for word in w2v_model.wv.index_to_key}
    elif args.model == "bert":
        model_path = "/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings/averaged_embeddings.pkl"
        with open(model_path, "rb") as f:
            embeddings = pickle.load(f)

    save_dir = f"{args.save_dir}/{args.model}"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    text_path = "/ocean/projects/bio210019p/stevesho/genomic_nlp/training_data/disease"
    text_edges_file = f"{text_path}/gda_co_occurence_{args.year}.tsv"
    test_file = f"{text_path}/gda_normalized_sorted.txt"
    preprocessor = GDADataPreprocessor(
        text_edges_file=text_edges_file,
        embeddings=embeddings,
    )
    data, test_pos_edge_index, test_neg_edge_index = preprocessor.preprocess_data()

    # special handling for year 2023
    if args.year == 2023:
        # for 2023, use 5-fold CV instead of a test set
        # build training features from all available data
        all_pos = torch.cat(
            [data.train_pos_edge_index, data.val_pos_edge_index, test_pos_edge_index],
            dim=1,
        )
        all_neg = torch.cat(
            [data.train_neg_edge_index, data.val_neg_edge_index, test_neg_edge_index],
            dim=1,
        )
        X_pos, _ = build_edge_features(all_pos, data, embeddings)
        X_neg, _ = build_edge_features(all_neg, data, embeddings)
        y_pos = np.ones(len(X_pos), dtype=int)
        y_neg = np.zeros(len(X_neg), dtype=int)
        X_all = np.concatenate([X_pos, X_neg], axis=0)
        y_all = np.concatenate([y_pos, y_neg], axis=0)

        print(
            f"Total data for CV: pos: {len(X_pos)}, neg: {len(X_neg)}; total: {len(X_all)}"
        )

        # # 5-fold cross-validation
        # kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # # results dictionary
        # cv_results = {"xgb": {"auc": [], "ap": []}, "lr": {"auc": [], "ap": []}}

        # print("Running 5-fold cross-validation...")
        # for fold, (train_idx, val_idx) in enumerate(kf.split(X_all)):
        #     X_train_fold, X_val_fold = X_all[train_idx], X_all[val_idx]
        #     y_train_fold, y_val_fold = y_all[train_idx], y_all[val_idx]

        #     # Train XGBoost
        #     xgb_clf = XGBClassifier(
        #         eval_metric="aucpr",
        #         n_estimators=300,
        #         learning_rate=0.05,
        #         subsample=0.8,
        #         colsample_bytree=0.8,
        #         seed=42,
        #         reg_lambda=1,
        #     )

        #     print(f"Training XGBoost (fold {fold+1}/5)...")
        #     xgb_clf.fit(X_train_fold, y_train_fold)
        #     probs_val = xgb_clf.predict_proba(X_val_fold)[:, 1]
        #     auc = roc_auc_score(y_val_fold, probs_val)
        #     ap = average_precision_score(y_val_fold, probs_val)
        #     cv_results["xgb"]["auc"].append(auc)
        #     cv_results["xgb"]["ap"].append(ap)
        #     print(f"Fold {fold+1} - XGB AUC = {auc:.4f}, AP = {ap:.4f}")

        #     # # Train Logistic Regression
        #     # lr_clf = LogisticRegression(max_iter=1000)
        #     # print(f"Training Logistic Regression (fold {fold+1}/5)...")
        #     # lr_clf.fit(X_train_fold, y_train_fold)
        #     # probs_val_lr = lr_clf.predict_proba(X_val_fold)[:, 1]
        #     # auc_lr = roc_auc_score(y_val_fold, probs_val_lr)
        #     # ap_lr = average_precision_score(y_val_fold, probs_val_lr)
        #     # cv_results["lr"]["auc"].append(auc_lr)
        #     # cv_results["lr"]["ap"].append(ap_lr)
        #     # print(f"Fold {fold+1} - LR AUC = {auc_lr:.4f}, AP = {ap_lr:.4f}")

        # # calculate average performance across folds
        # cv_results["xgb"]["mean_auc"] = np.mean(cv_results["xgb"]["auc"])
        # cv_results["xgb"]["std_auc"] = np.std(cv_results["xgb"]["auc"])
        # cv_results["xgb"]["mean_ap"] = np.mean(cv_results["xgb"]["ap"])
        # cv_results["xgb"]["std_ap"] = np.std(cv_results["xgb"]["ap"])

        # # cv_results["lr"]["mean_auc"] = np.mean(cv_results["lr"]["auc"])
        # # cv_results["lr"]["std_auc"] = np.std(cv_results["lr"]["auc"])
        # # cv_results["lr"]["mean_ap"] = np.mean(cv_results["lr"]["ap"])
        # # cv_results["lr"]["std_ap"] = np.std(cv_results["lr"]["ap"])

        # # save CV results as JSON
        # with open(save_dir / f"cv_results_{args.year}.json", "w") as f:
        #     json.dump(cv_results, f, indent=4)

        # print("CV Results:")
        # print(
        #     f"XGB - Mean AUC: {cv_results['xgb']['mean_auc']:.4f} ± {cv_results['xgb']['std_auc']:.4f}"
        # )
        # print(
        #     f"XGB - Mean AP: {cv_results['xgb']['mean_ap']:.4f} ± {cv_results['xgb']['std_ap']:.4f}"
        # )
        # print(
        #     f"LR - Mean AUC: {cv_results['lr']['mean_auc']:.4f} ± {cv_results['lr']['std_auc']:.4f}"
        # )
        # print(
        #     f"LR - Mean AP: {cv_results['lr']['mean_ap']:.4f} ± {cv_results['lr']['std_ap']:.4f}"
        # )

        # Train final models on all data
        print("Training final XGBoost model on all data...")
        final_xgb = XGBClassifier(
            eval_metric="aucpr",
            n_estimators=300,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            seed=42,
            reg_lambda=1,
        )
        final_xgb.fit(X_all, y_all)

        # save final XGBoost model
        model_path = save_dir / f"xgboost_gda_{args.year}_final_notbert.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(final_xgb, f)
        print(f"Saved XGBoost model to {model_path}")

        # run SHAP analysis on XGBoost
        xgb_wrapped = BaselineModel(final_xgb)
        shap_values = run_shap(xgb_wrapped, X_all, save_dir, f"xgboost_gda_{args.year}")

        if shap_values is not None:
            shap_path = save_dir / f"gda_shap_values_{args.year}.npy"
            np.save(shap_path, shap_values)
            print(f"Saved SHAP values to {shap_path}")

        # # train final Logistic Regression model
        # print("Training final Logistic Regression model on all data...")
        # final_lr = LogisticRegression(max_iter=1000)
        # final_lr.fit(X_all, y_all)

        # # save final Logistic Regression model
        # model_path_lr = save_dir / f"lr_gda_{args.year}.pkl"
        # with open(model_path_lr, "wb") as f:
        #     pickle.dump(final_lr, f)
        # print(f"Saved Logistic Regression model to {model_path_lr}")

    else:
        # code for years other than 2023
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

        # remove any test pairs that are in the training data
        train_pos_edges = data.train_pos_edge_index.numpy().T
        train_neg_edges = data.train_neg_edge_index.numpy().T
        train_edges = np.concatenate([train_pos_edges, train_neg_edges], axis=0)
        train_edges_set = {
            (data.inv_node_mapping[src], data.inv_node_mapping[dst])
            for src, dst in train_edges
        }
        test_positive_pairs = [
            (g, d) for g, d in test_positive_pairs if (g, d) not in train_edges_set
        ]

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
        train_pos = torch.cat(
            [data.train_pos_edge_index, data.val_pos_edge_index], dim=1
        )
        train_neg = torch.cat(
            [data.train_neg_edge_index, data.val_neg_edge_index], dim=1
        )
        X_pos, _ = build_edge_features(train_pos, data, embeddings)
        X_neg, _ = build_edge_features(train_neg, data, embeddings)
        y_pos = np.ones(len(X_pos), dtype=int)
        y_neg = np.zeros(len(X_neg), dtype=int)
        X_train = np.concatenate([X_pos, X_neg], axis=0)
        y_train = np.concatenate([y_pos, y_neg], axis=0)

        print(
            f"Train pos: {len(X_pos)}, neg: {len(X_neg)}; total train: {len(X_train)}"
        )
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
        precision, recall, thresholds = pr_curve
        print(f"Test AUC = {auc:.4f}")
        print(f"Test Average Precision = {ap:.4f}")

        np.savez(
            save_dir / f"pr_curve_{args.year}.npz",
            precision=precision,
            recall=recall,
            thresholds=thresholds,
        )

        model_path = save_dir / f"xgboost_gda_{args.year}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(xgb_clf, f)

        print(f"Saved XGBoost model to {model_path}")

        # probs_test_named = []
        # for idx, (g, d) in enumerate(test_positive_pairs + test_negative_pairs):
        #     probs_test_named.append((g, d, probs_test[idx]))
        # with open(save_dir / f"probs_test_{args.year}.tsv", "w") as f:
        #     writer = csv.writer(f, delimiter="\t")
        #     writer.writerow(["gene", "disease", "probability"])
        #     for row in probs_test_named:
        #         writer.writerow(row)

        # # train Logistic Regression
        # lr_clf = LogisticRegression()
        # print("Training Logistic Regression...")

        # lr_clf.fit(X_train, y_train)
        # probs_test_lr = lr_clf.predict_proba(X_test)[:, 1]
        # auc_lr = roc_auc_score(y_test, probs_test_lr)
        # ap_lr = average_precision_score(y_test, probs_test_lr)
        # pr_curve_lr = precision_recall_curve(y_test, probs_test_lr)
        # precision_lr, recall_lr, thresholds_lr = pr_curve_lr

        # print(f"Test AUC (LR) = {auc_lr:.4f}")
        # print(f"Test Average Precision (LR) = {ap_lr:.4f}")
        # np.savez(
        #     save_dir / f"pr_curve_lr_{args.year}.npz",
        #     precision=precision_lr,
        #     recall=recall_lr,
        #     thresholds=thresholds_lr,
        # )

        # model_path_lr = save_dir / f"lr_gda_{args.year}.pkl"
        # with open(model_path_lr, "wb") as f:
        #     pickle.dump(lr_clf, f)

        # print(f"Saved Logistic Regression model to {model_path_lr}")
        # probs_test_named_lr = []
        # for idx, (g, d) in enumerate(test_positive_pairs + test_negative_pairs):
        #     probs_test_named_lr.append((g, d, probs_test_lr[idx]))

        # with open(save_dir / f"probs_test_lr_{args.year}.tsv", "w") as f:
        #     writer = csv.writer(f, delimiter="\t")
        #     writer.writerow(["gene", "disease", "probability"])
        #     for row in probs_test_named_lr:
        #         writer.writerow(row)

        # # random baseline
        # print("Running Random Baseline...")

        # class RandomBaseline:
        #     """
        #     A random baseline classifier that returns random probabilities.
        #     Mimics the interface of a model with a predict_proba method.
        #     """

        #     def __init__(self, random_state: int = 42):
        #         self.rng = np.random.default_rng(random_state)

        #     def fit(self, X: np.ndarray, y: np.ndarray):
        #         return self

        #     def predict_proba(self, X: np.ndarray) -> np.ndarray:
        #         n_samples = X.shape[0]
        #         probs = self.rng.random(n_samples)
        #         return np.column_stack((1 - probs, probs))

        # random_baseline = RandomBaseline(random_state=42).fit(X_train, y_train)
        # probs_test_rand = random_baseline.predict_proba(X_test)[:, 1]
        # auc_rand = roc_auc_score(y_test, probs_test_rand)
        # ap_rand = average_precision_score(y_test, probs_test_rand)
        # pr_curve_rand = precision_recall_curve(y_test, probs_test_rand)
        # precision_rand, recall_rand, thresholds_rand = pr_curve_rand

        # print(f"Test AUC (Random Baseline) = {auc_rand:.4f}")
        # print(f"Test Average Precision (Random Baseline) = {ap_rand:.4f}")
        # np.savez(
        #     save_dir / f"pr_curve_rand_{args.year}.npz",
        #     precision=precision_rand,
        #     recall=recall_rand,
        #     thresholds=thresholds_rand,
        # )

        # probs_test_named_rand = []
        # for idx, (g, d) in enumerate(test_positive_pairs + test_negative_pairs):
        #     probs_test_named_rand.append((g, d, probs_test_rand[idx]))

        # with open(save_dir / f"probs_test_rand_{args.year}.tsv", "w") as f:
        #     writer = csv.writer(f, delimiter="\t")
        #     writer.writerow(["gene", "disease", "probability"])
        #     for row in probs_test_named_rand:
        #         writer.writerow(row)


if __name__ == "__main__":
    main()
