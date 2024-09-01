#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to run baseline models for gene interaction prediction. Models take gene
embeddings as input and predict whether the pair of genes interacts. Models can
make probability based predictions or binary predictions based on a probability
threshold."""


import argparse
import multiprocessing as mp
import os
import pickle
from typing import Any, Callable, Dict, List

import numpy as np
import psutil  # type: ignore
from scipy import stats  # type: ignore
from sklearn.metrics import average_precision_score  # type: ignore
from sklearn.metrics import roc_auc_score  # type: ignore
from sklearn.model_selection import StratifiedKFold  # type: ignore

# from baseline_models import CosineSimilarity
from constants import RANDOM_STATE
from interaction_models import BaselineModel
from interaction_models import LogisticRegressionModel
from interaction_models import MLP
from interaction_models import SVM
from interaction_models import XGBoost
from model_data_preprocessor import InteractionDataPreprocessor
from utils import get_physical_cores
from visualizers import BaselineModelVisualizer


class GeneInterationPredictions:
    """Class used to train an individual gene interaction prediction model.
    Evaluates via 5-fold cross validation, then trains a final model on the
    entire training set and evaluates on the hold-out test set.
    """

    def __init__(
        self,
        model_class: Callable[..., BaselineModel],
        train_features: np.ndarray,
        train_targets: np.ndarray,
        test_features: np.ndarray,
        test_targets: np.ndarray,
        model_name: str,
        model_dir: str,
    ) -> None:
        """Instantiate a GeneInteractionPredictions object."""
        self.model_class = model_class
        self.train_features = train_features
        self.train_targets = train_targets
        self.test_features = test_features
        self.test_targets = test_targets
        self.model_name = model_name
        self.model_dir = model_dir

    def perform_cross_validation(
        self,
        n_splits: int,
        **kwargs,
    ) -> List[float]:
        """Perform stratified k-fold cross-validation and return AUC scores."""
        folds = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
        )
        return [
            self.evaluate_model(
                model=self.train_model(
                    model_class=self.model_class,
                    features=self.train_features[train_index],
                    labels=self.train_targets[train_index],
                    **kwargs,
                ),
                features=self.train_features[test_index],
                labels=self.train_targets[test_index],
            )
            for train_index, test_index in folds.split(
                X=self.train_features, y=self.train_targets
            )
        ]

    def train_and_evaluate_model(
        self,
    ) -> BaselineModel:
        """Train a model, evaluate, and save results."""
        print(f"\nTraining and evaluating {self.model_name}:")

        # perform 5-fold CV
        cv_scores = self.perform_cross_validation(n_splits=5)
        mean_cv_auc = np.mean(cv_scores)
        std_cv_auc = np.std(cv_scores)
        print(f"Cross-validation Mean AUC: {mean_cv_auc:.4f} (+/- {std_cv_auc:.4f})")

        # train final model on entire training set
        final_model = self.train_model(
            self.model_class, self.train_features, self.train_targets
        )

        # evaluate on training set
        train_auc = self.evaluate_model(
            final_model, self.train_features, self.train_targets
        )
        print(f"Training set AUC: {train_auc:.4f}")

        # evaluate on hold-out test set
        test_auc = self.evaluate_model(
            final_model, self.test_features, self.test_targets
        )
        print(f"Hold-out test set AUC: {test_auc:.4f}")

        # save model
        model_path = f"{self.model_dir}/{self.model_name}_model.pkl"
        try:
            with open(model_path, "wb") as f:
                pickle.dump(final_model, f)
            print(f"Model saved to {model_path}")
        except Exception as e:
            print(f"Error saving model {self.model_name}: {str(e)}")

        return final_model

    @staticmethod
    def train_model(
        model_class: Callable[..., BaselineModel],
        features: np.ndarray,
        labels: np.ndarray,
        **kwargs,
    ) -> BaselineModel:
        """Train a model on given features and labels."""
        model = model_class(**kwargs)
        model.train(feature_data=features, target_labels=labels)
        return model

    @staticmethod
    def evaluate_model(
        model: BaselineModel, features: np.ndarray, labels: np.ndarray
    ) -> float:
        """Evaluate a model and return its AUC score."""
        predicted_probabilities = model.predict_probability(features)
        return roc_auc_score(labels, predicted_probabilities)


class BootstrapEvaluator:
    """Class used to evaluate models using n out of n bootstrapping."""

    def __init__(
        self,
        models: Dict[str, BaselineModel],
        test_features: np.ndarray,
        test_targets: np.ndarray,
        n_iterations: int = 1000,
        confidence_level: float = 0.95,
    ) -> None:
        """Instantiate a BootstrapEvaluator object."""
        self.models = models
        self.test_features = test_features
        self.test_targets = test_targets
        self.n_iterations = n_iterations
        self.confidence_level = confidence_level

    def bootstrap_test_evaluation(
        self,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Perform bootstrapped evaluation of multiple models on the test set."""
        n_samples = len(self.test_features)

        # prepare arguments for multiprocessing
        mp_args = [
            (n_samples, self.test_features, self.test_targets, self.models)
            for _ in range(self.n_iterations)
        ]

        # use multiprocessing to run iterations in parallel
        with mp.Pool(processes=get_physical_cores()) as pool:
            results = pool.starmap(self._bootstrap_iteration, mp_args)

        # aggregate results
        bootstrap_results: Dict[str, Dict[str, List[float]]] = {
            model_name: {"auc": [], "auprc": []} for model_name in self.models
        }
        for result in results:
            for model_name, metrics in result.items():
                bootstrap_results[model_name]["auc"].append(metrics["auc"])
                bootstrap_results[model_name]["auprc"].append(metrics["auprc"])

        return self.compute_bootstrap_statistics(
            bootstrap_results=bootstrap_results, confidence_level=self.confidence_level
        )

    @staticmethod
    def _bootstrap_iteration(n_samples, test_features, test_targets, models):
        """Perform a single bootstrap iteration."""
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_features = test_features[indices]
        boot_targets = test_targets[indices]
        iteration_results = {}
        for model_name, model in models.items():
            predictions = model.predict_probability(boot_features)
            iteration_results[model_name] = {
                "auc": roc_auc_score(boot_targets, predictions),
                "auprc": average_precision_score(boot_targets, predictions),
            }
        return iteration_results

    @staticmethod
    def compute_bootstrap_statistics(
        bootstrap_results: Dict[str, Dict[str, List[float]]], confidence_level: float
    ) -> Dict[str, Any]:
        """Compute statistics from bootstrap results."""
        bootstrap_stats: Dict[str, Any] = {}

        for model_name, metrics in bootstrap_results.items():
            bootstrap_stats[model_name] = {}
            for metric, values in metrics.items():
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


def _setup_model_dir(args: argparse.Namespace) -> str:
    """Setup model directory."""
    embeddings_name = (
        args.embeddings.split("/")[-1].split(".")[0].replace("_embeddings", "")
    )
    model_dir = os.path.join(
        "/ocean/projects/bio210019p/stevesho/genomic_nlp/models/baseline",
        embeddings_name,
    )
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
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
    parser.add_argument(
        "--text_edges_file",
        type=str,
        help="Path to text edges pickle file.",
    )
    # parser.add_argument(
    #     "--no_train",
    #     action="store_true",
    #     help="Do not train models, only plot results.",
    # )
    return parser.parse_args()


def main() -> None:
    """Run baseline models for gene interaction prediction."""
    args = parse_args()

    # process training data
    data_preprocessor = InteractionDataPreprocessor(args=args)
    (
        positive_pairs,
        negative_pairs,
        train_features,
        train_targets,
        test_features,
        test_targets,
        pos_test,
        neg_test,
    ) = data_preprocessor.load_and_preprocess_data()

    print(f"No. of positive pairs: {len(positive_pairs)}")
    print(f"No. of negative pairs: {len(negative_pairs)}")
    print(f"Train features shape: {train_features.shape}")
    print(f"Train targets shape: {train_targets.shape}")
    print(f"Test features shape: {test_features.shape}")
    print(f"Test targets shape: {test_targets.shape}")

    # prepare stratified test data
    stratified_test_data = data_preprocessor.prepare_stratified_test_data(
        pos_test=pos_test, test_features=test_features, neg_test=neg_test
    )

    # setup model directory
    model_dir = _setup_model_dir(args=args)

    # define models
    models = {
        "logistic_regression": LogisticRegressionModel,
        "svm": SVM,
        "xgboost": XGBoost,
        "mlp": MLP,
    }

    # train and evaluate models
    trained_models = {}
    train_results = {}
    test_results = {}
    stratified_results: Dict[str, Dict[Any, Any]] = {model: {} for model in models}

    for name, model_class in models.items():
        gene_interaction_predictor = GeneInterationPredictions(
            model_class=model_class,
            train_features=train_features,
            train_targets=train_targets,
            test_features=test_features,
            test_targets=test_targets,
            model_name=name,
            model_dir=model_dir,
        )
        trained_model = gene_interaction_predictor.train_and_evaluate_model()
        trained_models[name] = trained_model

        # collect results
        train_results[name] = gene_interaction_predictor.evaluate_model(
            trained_model, train_features, train_targets
        )
        test_results[name] = gene_interaction_predictor.evaluate_model(
            trained_model, test_features, test_targets
        )

        # evaluate on stratified test sets
        for source, data in stratified_test_data.items():
            stratified_results[name][source] = (
                gene_interaction_predictor.evaluate_model(
                    model=trained_model,
                    features=data["features"],
                    labels=data["targets"],
                )
            )

    visualizer = BaselineModelVisualizer(output_path=model_dir)
    visualizer.plot_model_performances(
        train_results=train_results, test_results=test_results
    )
    visualizer.plot_stratified_performance(stratified_results=stratified_results)
    visualizer.plot_roc_curve(
        models=trained_models, test_features=test_features, test_labels=test_targets
    )

    # # bootstrap evaluation
    # print("\nBootstrapping evaluation:")
    # bootstrap_evaluator = BootstrapEvaluator(
    #     models=trained_models,
    #     test_features=test_features,
    #     test_targets=test_targets,
    # )
    # bootstrap_results = bootstrap_evaluator.bootstrap_test_evaluation()

    # # plot results
    # visualizer.plot_bootstrap_results(bootstrap_stats=bootstrap_results)


if __name__ == "__main__":
    main()
