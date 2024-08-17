#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to run baseline models for gene interaction prediction. Models take gene
embeddings as input and predict whether the pair of genes interacts. Models can
make probability based predictions or binary predictions based on a probability
threshold."""

import argparse
import os
import pickle
import random
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
from scipy import stats  # type: ignore
from scipy.stats import mannwhitneyu  # type: ignore
from scipy.stats import spearmanr  # type: ignore
from sklearn.metrics import average_precision_score  # type: ignore
from sklearn.metrics import roc_auc_score  # type: ignore
from sklearn.model_selection import StratifiedKFold  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

from baseline_models import BaselineModel
from baseline_models import CosineSimilarity
from baseline_models import LogisticRegressionModel
from baseline_models import MLP
from baseline_models import RandomForest
from baseline_models import XGBoost

RANDOM_SEED = 42


def _unpickle_dict(
    pickle_file: str,
) -> Union[Dict[str, np.ndarray], List[Tuple[str, str]]]:
    """Simple wrapper to unpickle embedding or pair pkls."""
    with open(pickle_file, "rb") as file:
        return pickle.load(file)


def balance_filtered_pairs(
    filtered_positive_pairs: List[Tuple[str, str]],
    filtered_negative_pairs: List[Tuple[str, str]],
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Adjust pairs to have the same size."""
    if len(filtered_positive_pairs) > len(filtered_negative_pairs):
        filtered_positive_pairs = random.sample(
            filtered_positive_pairs, len(filtered_negative_pairs)
        )
    elif len(filtered_negative_pairs) > len(filtered_positive_pairs):
        filtered_negative_pairs = random.sample(
            filtered_negative_pairs, len(filtered_positive_pairs)
        )
    return filtered_positive_pairs, filtered_negative_pairs


def filter_pairs_for_embeddings(
    gene_embeddings: Dict[str, np.ndarray],
    positive_pairs: List[Tuple[str, str]],
    negative_pairs: List[Tuple[str, str]],
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Filter pairs to only include those with embeddings."""
    filtered_positive_pairs = [
        pair for pair in positive_pairs if all(gene in gene_embeddings for gene in pair)
    ]
    filtered_negative_pairs = [
        pair for pair in negative_pairs if all(gene in gene_embeddings for gene in pair)
    ]

    # adjust pairs to have same size
    return balance_filtered_pairs(filtered_positive_pairs, filtered_negative_pairs)


def prepare_data_and_targets(
    gene_embeddings: Dict[str, np.ndarray],
    positive_pairs: List[Tuple[str, str]],
    negative_pairs: List[Tuple[str, str]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Create feature data and target labels from gene pairs."""
    data = []
    targets = []
    for pair in positive_pairs + negative_pairs:
        gene1, gene2 = pair
        vec1 = gene_embeddings[gene1]
        vec2 = gene_embeddings[gene2]
        data.append(np.concatenate([vec1, vec2]))
        targets.append(1 if pair in positive_pairs else 0)
    return np.array(data), np.array(targets)


def format_training_data(
    gene_embeddings: Dict[str, np.ndarray],
    positive_pairs: List[Tuple[str, str]],
    negative_pairs: List[Tuple[str, str]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare input data for model training."""
    # filter pairs to only include those with embeddings
    positive_pairs, negative_pairs = filter_pairs_for_embeddings(
        gene_embeddings, positive_pairs, negative_pairs
    )

    # create feature data and target labels
    return prepare_data_and_targets(gene_embeddings, positive_pairs, negative_pairs)


def train_model(
    model_class: Callable[[], BaselineModel],
    features: np.ndarray,
    labels: np.ndarray,
) -> BaselineModel:
    """Train a model on given features and labels."""
    model = model_class()
    model.train(feature_data=features, target_labels=labels)
    return model


def evaluate_model(
    model: BaselineModel, features: np.ndarray, labels: np.ndarray
) -> float:
    """Evaluate a model and return its AUC score."""
    predicted_probabilities = model.predict_probability(features)
    return roc_auc_score(labels, predicted_probabilities)


def perform_cross_validation(
    model_class: Callable[[], BaselineModel],
    gene_pairs: np.ndarray,
    targets: np.ndarray,
    n_splits: int,
) -> List[float]:
    """Perform stratified k-fold cross-validation and return AUC scores."""
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    auc_scores = []

    for k_fold, (train_index, test_index) in enumerate(
        folds.split(gene_pairs, targets), 1
    ):
        train_features, test_features = gene_pairs[train_index], gene_pairs[test_index]
        train_labels, test_labels = targets[train_index], targets[test_index]

        model = train_model(model_class, train_features, train_labels)
        auc_score = evaluate_model(model, test_features, test_labels)
        auc_scores.append(auc_score)

        print(f"Fold {k_fold} AUC: {auc_score:.4f}")

    return auc_scores


def train_and_evaluate_baseline_models(
    model_class: Callable[[], BaselineModel],
    gene_pairs: np.ndarray,
    targets: np.ndarray,
    n_splits: int = 5,
) -> Tuple[BaselineModel, float, float, float]:
    """Train a model and evaluate its performance using stratified k-fold cross-validation."""
    auc_scores = perform_cross_validation(model_class, gene_pairs, targets, n_splits)

    mean_auc = float(np.mean(auc_scores))
    std_auc = float(np.std(auc_scores))
    print(f"Mean AUC: {mean_auc:.4f} (+/- {std_auc:.4f})")

    # train a final model on all data
    final_model = train_model(model_class, gene_pairs, targets)

    # get final model metrics
    final_predictions = final_model.predict_probability(gene_pairs)
    final_auc = roc_auc_score(targets, final_predictions)
    print(f"Final AUC: {final_auc:.4f}")

    return final_model, mean_auc, std_auc, final_auc


def bootstrap_evaluation(
    models: Dict[str, Any],
    gene_pairs: np.ndarray,
    targets: np.ndarray,
    eval_func: Callable,
    n_iterations: int = 1000,
    sample_size: float = 0.8,
    confidence_level: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """Perform bootstrapped evaluation of the cosine similarity model.

    Args:
        models: Dictionary of models to evaluate (key: model name, value: model
        object)
    """
    n_samples = int(len(gene_pairs) * sample_size)
    bootstrap_results: Dict[str, Dict[Any, Any]] = {
        model_name: {} for model_name in models
    }

    for _ in range(n_iterations):
        # generate bootstrap sample
        indices = np.random.choice(len(gene_pairs), size=n_samples, replace=True)
        boot_gene_pairs = gene_pairs[indices]
        boot_targets = targets[indices]

        # evaluate each model on bootstrap sample
        for model_name, model in models.items():
            results = eval_func(model, boot_gene_pairs, boot_targets)

            for metric, value in results.items():
                if metric not in bootstrap_results[model_name]:
                    bootstrap_results[model_name][metric] = []
                bootstrap_results[model_name][metric].append(value)

    # compute statistics
    bootstrap_stats: Dict[str, Dict[str, Any]] = {}
    for model_name in models:
        bootstrap_stats[model_name] = {}
        for metric, values in bootstrap_results[model_name].items():
            mean = np.mean(values)
            std_err = np.std(values, ddof=1)
            ci_lower, ci_upper = stats.t.interval(
                confidence_level, len(values) - 1, loc=mean, scale=std_err
            )

            bootstrap_stats[model_name][metric] = {
                "mean": mean,
                "std_error": std_err,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }

    return bootstrap_stats


# def single_evaluation(model, pairs, targets):
#     # This function should contain your evaluation logic
#     # and return a dictionary of metrics
#     similarities = model.predict_probability(pairs)
#     relative_performance = np.mean(similarities[targets == 1]) / np.mean(
#         similarities[targets == 0]
#     )
#     correlation, _ = stats.spearmanr(targets, similarities)
#     return {"relative_performance": relative_performance, "correlation": correlation}


def main() -> None:
    """Main function to run baseline models for gene interaction prediction."""
    parser = argparse.ArgumentParser(
        description="Run baseline models for gene interaction prediction."
    )
    parser.add_argument(
        "--embeddings", type=str, help="Path to gene embeddings pickle file."
    )
    parser.add_argument(
        "--positive_pairs_file",
        type=str,
        help="Path to positive gene interaction pairs pickle file.",
    )
    parser.add_argument(
        "--negative_pairs_file",
        type=str,
        help="Path to negative gene interaction pairs pickle file.",
    )
    args = parser.parse_args()

    # load data and embeddings
    gene_embeddings = _unpickle_dict(args.embeddings)
    positive_pairs = _unpickle_dict(args.positive_pairs_file)
    negative_pairs = _unpickle_dict(args.negative_pairs_file)

    print(f"No. of positive pairs: {len(positive_pairs)}")
    print(f"No. of negative pairs: {len(negative_pairs)}")

    # type checking to mypy doesn't complain
    if not isinstance(gene_embeddings, dict):
        raise ValueError("Gene embeddings must be a dictionary.")
    if not isinstance(positive_pairs, list):
        raise ValueError("Positive pairs must be a list.")
    if not isinstance(negative_pairs, list):
        raise ValueError("Negative pairs must be a list.")

    # get names for saving
    embeddings_name = (
        args.embeddings.split("/")[-1].split(".")[0].replace("_embeddings", "")
    )
    model_base = "/ocean/projects/bio210019p/stevesho/genomic_nlp/models/baseline"
    model_dir = f"{model_base}/{embeddings_name}"
    os.makedirs(model_dir, exist_ok=True)

    # format data and embeddings for model training
    gene_pairs, targets = format_training_data(
        gene_embeddings=gene_embeddings,
        positive_pairs=positive_pairs,
        negative_pairs=negative_pairs,
    )

    # define models
    models = {
        "logistic_regression": LogisticRegressionModel,
        "random_forest": RandomForest,
        "xgboost": XGBoost,
        "mlp": MLP,
    }

    # train and evaluate models
    results = {}
    for name, model_initializer in models.items():
        print(f"\nTraining and evaluating {name}:")
        final_model, mean_auc, std_auc, final_auc = train_and_evaluate_baseline_models(
            model_class=model_initializer, gene_pairs=gene_pairs, targets=targets
        )
        results[name] = (final_model, mean_auc, std_auc)
        print(f"{name} - Mean AUC: {mean_auc:.4f} (+/- {std_auc:.4f})")
        print(f"{name} - Final AUC: {final_auc:.4f}")

    example_pair_features = np.random.rand(1, gene_pairs.shape[1])

    # save models
    for name, (model, _, _) in results.items():
        model.save_model(f"{model_dir}/{name}_model.pkl")

    # train and evaluate cosine similarity model
    # print("\nEvaluating Cosine Similarity model:")
    # cosine_model = CosineSimilarity()
    # cosine_results = evaluate_cosine_similarity(
    #     cosine_model=cosine_model, gene_pairs=gene_pairs, targets=targets
    # )
    # print(
    #     "Mean Relative Performance: "
    #     f"{cosine_results['mean_relative_performance']:.4f}"
    # )

    # bootstrap_results = bootstrap_multi_model_evaluation(
    #     models, gene_pairs, targets, single_evaluation
    # )

    # # Print results
    # for model_name, model_stats in bootstrap_results.items():
    #     print(f"\nResults for {model_name}:")
    #     for metric, stats in model_stats.items():
    #         print(f"  {metric}:")
    #         print(f"    Mean: {stats['mean']:.4f}")
    #         print(f"    Std Error: {stats['std_error']:.4f}")
    #         print(f"    95% CI: ({stats['ci_lower']:.4f}, {stats['ci_upper']:.4f})")

    # # Perform statistical tests between models
    # for metric in bootstrap_results[list(models.keys())[0]]:
    #     print(f"\nStatistical tests for {metric}:")
    #     model_names = list(models.keys())
    #     for i in range(len(model_names)):
    #         for j in range(i + 1, len(model_names)):
    #             model1 = model_names[i]
    #             model2 = model_names[j]
    #             t_stat, p_value = stats.ttest_ind(
    #                 bootstrap_results[model1][metric], bootstrap_results[model2][metric]
    #             )
    #             print(
    #                 f"  {model1} vs {model2}: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}"
    #             )


if __name__ == "__main__":
    main()
