#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to run baseline models for gene interaction prediction. Models take gene
embeddings as input and predict whether the pair of genes interacts. Models can
make probability based predictions or binary predictions based on a probability
threshold."""


from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from scipy import stats  # type: ignore
from scipy.stats import mannwhitneyu  # type: ignore
from scipy.stats import spearmanr  # type: ignore
from sklearn.metrics import average_precision_score  # type: ignore
from sklearn.metrics import roc_auc_score  # type: ignore
from sklearn.model_selection import StratifiedKFold  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

from baseline_models import BaselineInteractionPredictor
from baseline_models import CosineSimilarity
from baseline_models import LogisticRegressionModel
from baseline_models import MLP
from baseline_models import RandomForest
from baseline_models import XGBoost

RANDOM_SEED = 42


def format_training_data(
    gene_embeddings: Dict[str, np.ndarray],
    positive_pairs: List[Tuple[str, str]],
    negative_pairs: List[Tuple[str, str]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare input data for model training."""
    data = []
    targets = []
    for pair in positive_pairs + negative_pairs:
        gene1, gene2 = pair
        vec1 = gene_embeddings[gene1]
        vec2 = gene_embeddings[gene2]
        data.append(np.concatenate([vec1, vec2]))
        targets.append(1 if pair in positive_pairs else 0)
    return np.array(data), np.array(targets)


def train_and_evaluate_baseline_models(
    model_class: Callable[[], BaselineInteractionPredictor],
    gene_pairs: np.ndarray,
    targets: np.ndarray,
    n_splits: int = 5,
) -> Tuple[BaselineInteractionPredictor, float, float]:
    """Train a model and evaluate its performance using stratified k-fold
    cross-validation.
    """
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    auc_scores = []

    for k_fold, (train_index, test_index) in enumerate(
        folds.split(gene_pairs, targets), 1
    ):
        train_features, test_features = (
            gene_pairs[train_index],
            gene_pairs[test_index],
        )
        train_labels, test_labels = (
            targets[train_index],
            targets[test_index],
        )

        model = model_class()
        model.train(feature_data=train_features, target_labels=train_labels)
        predicted_probabilities = model.predict_probability(test_features)
        auc_score = roc_auc_score(test_labels, predicted_probabilities)
        auc_scores.append(auc_score)
        print(f"Fold {k_fold} AUC: {auc_score:.4f}")

    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    print(f"Mean AUC: {mean_auc:.4f} (+/- {std_auc:.4f})")

    # Train a final model on all data
    final_model = model_class()
    final_model.train(feature_data=gene_pairs, target_labels=targets)

    return final_model, mean_auc, std_auc


def evaluate_cosine_similarity(
    cosine_model: CosineSimilarity,
    gene_pairs: np.ndarray,
    targets: np.ndarray,
    n_splits: int = 5,
) -> Dict[str, float]:
    """Evaluate CosineSimilarity model using stratified k-fold
    cross-validation.
    """
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    metrics: Dict[str, List[Any]] = {
        "relative_performance": [],
        "correlation": [],
        "effect_size": [],
    }

    for k_fold, (_, test_index) in enumerate(folds.split(gene_pairs, targets), 1):
        test_pairs = gene_pairs[test_index]
        test_targets = targets[test_index]
        predicted_similarities = cosine_model.predict_probability(test_pairs)

        positive_similarities = predicted_similarities[test_targets == 1]
        negative_similarities = predicted_similarities[test_targets == 0]

        # relative performance metric
        relative_performance = np.mean(positive_similarities) / np.mean(
            negative_similarities
        )

        # spearman correlation
        correlation, _ = spearmanr(test_targets, predicted_similarities)

        # effect size (Cohen's d)
        effect_size = (
            np.mean(positive_similarities) - np.mean(negative_similarities)
        ) / np.sqrt(
            (
                np.std(positive_similarities, ddof=1) ** 2
                + np.std(negative_similarities, ddof=1) ** 2
            )
            / 2
        )

        # Mann-Whitney U test
        _, p_value = mannwhitneyu(
            positive_similarities, negative_similarities, alternative="two-sided"
        )

        metrics["relative_performance"].append(relative_performance)
        metrics["correlation"].append(correlation)
        metrics["effect_size"].append(effect_size)

        print(
            f"Fold {k_fold} - "
            f"Relative Performance: {relative_performance:.4f}, "
            f"Correlation: {correlation:.4f}, "
            f"Effect Size: {effect_size:.4f}, "
            f"p-value: {p_value:.4f}"
        )

    # calculate means and standard deviations
    results = {}
    for key in metrics:
        results[f"mean_{key}"] = np.mean(metrics[key])
        results[f"std_{key}"] = np.std(metrics[key])

    print(
        f"Mean Relative Performance: {results['mean_relative_performance']:.4f} "
        f"(+/- {results['std_relative_performance']:.4f})"
    )
    print(
        f"Mean Correlation: {results['mean_correlation']:.4f} "
        f"(+/- {results['std_correlation']:.4f})"
    )
    print(
        f"Mean Effect Size: {results['mean_effect_size']:.4f} "
        f"(+/- {results['std_effect_size']:.4f})"
    )

    return results


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


def main():
    """Main function to run baseline models for gene interaction prediction."""

    ### load data and embeddings
    gene_embeddings: Dict[str, np.ndarray] = {}  # gene_id -> embedding vector
    positive_pairs: List[Tuple[str, str]] = (
        []
    )  # list of (gene1_id, gene2_id) for known interactions
    negative_pairs: List[Tuple[str, str]] = (
        []
    )  # list of (gene1_id, gene2_id) for known non-interactions

    # format data and embeddings for model training
    gene_pairs, targets = format_training_data(
        gene_embeddings=gene_embeddings,
        positive_pairs=positive_pairs,
        negative_pairs=negative_pairs,
    )

    models = {
        "Logistic Regression": LogisticRegressionModel,
        "Random Forest": RandomForest,
        "XGBoost": XGBoost,
        "MLP": MLP,
    }

    # train and evaluate models
    results = {}
    for name, model_initializer in models.items():
        print(f"\nTraining and evaluating {name}:")
        final_model, mean_auc, std_auc = train_and_evaluate_baseline_models(
            model_class=model_initializer, gene_pairs=gene_pairs, targets=targets
        )
        results[name] = (final_model, mean_auc, std_auc)
        print(f"{name} - Mean AUC: {mean_auc:.4f} (+/- {std_auc:.4f})")

    example_pair_features = np.random.rand(1, gene_pairs.shape[1])
    lr_model, _, _ = results["Logistic Regression"]
    lr_prediction = lr_model.predict_probability(example_pair_features)
    print(
        "Logistic Regression prediction for example gene pair: "
        f"{lr_prediction[0]:.4f}"
    )

    # train and evaluate cosine similarity model
    print("\nEvaluating Cosine Similarity model:")
    cosine_model = CosineSimilarity()
    cosine_results = evaluate_cosine_similarity(
        cosine_model=cosine_model, gene_pairs=gene_pairs, targets=targets
    )
    print(
        "Mean Relative Performance: "
        f"{cosine_results['mean_relative_performance']:.4f}"
    )

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
