#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Baseline models for prediction gene-gene interactions. We adopted a base
class for which the other models inherit from.

The following are implemented:
    (1) Pairwise similarity scoring based on cosine distance 
    (1) Logistic regression
    (2) Random forest
    (3) XGBoost
    (4) Multi-layer perceptron (MLP)"""


from typing import Optional, Tuple, Union


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
