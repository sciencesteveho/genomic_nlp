#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to run baseline models for gene interaction prediction. Models take gene
embeddings as input and predict whether the pair of genes interacts. Models can
make probability based predictions or binary predictions based on a probability
threshold."""


import argparse
import csv
import multiprocessing as mp
import os
import pickle
from typing import Any, Callable, Dict, List, Set, Tuple, Union

from gensim.models import Word2Vec  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import psutil  # type: ignore
from scipy import stats  # type: ignore
import shap  # type: ignore
from sklearn.metrics import average_precision_score  # type: ignore
from sklearn.metrics import roc_auc_score  # type: ignore
from sklearn.model_selection import StratifiedKFold  # type: ignore

from genomic_nlp.interaction_data_preprocessor import InteractionDataPreprocessor
from genomic_nlp.models.interaction_models import BaselineModel
from genomic_nlp.models.interaction_models import LogisticRegressionModel
from genomic_nlp.models.interaction_models import MLP
from genomic_nlp.models.interaction_models import RandomBaseline
from genomic_nlp.models.interaction_models import SVM
from genomic_nlp.models.interaction_models import XGBoost
from genomic_nlp.run_cancer_models import _extract_gene_vectors
from genomic_nlp.utils.common import get_physical_cores
from genomic_nlp.utils.constants import RANDOM_STATE
from genomic_nlp.visualization import set_matplotlib_publication_parameters
from genomic_nlp.visualization.visualizers import BaselineModelVisualizer


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
        test_gene_pairs: List[Tuple[str, str]],
        model_name: str,
        model_dir: str,
    ) -> None:
        """Instantiate a GeneInteractionPredictions object."""
        self.model_class = model_class
        self.train_features = train_features
        self.train_targets = train_targets
        self.test_features = test_features
        self.test_targets = test_targets
        self.test_gene_pairs = test_gene_pairs
        self.model_name = model_name
        self.model_dir = model_dir

    def train_and_evaluate_model(
        self,
    ) -> BaselineModel:
        """Train a model, evaluate, and save results."""
        print(f"\nTraining and evaluating {self.model_name}:")

        # train final model on entire training set
        final_model = self.train_model(
            self.model_class, self.train_features, self.train_targets
        )

        # evaluate on training set
        train_ap = self.evaluate_model(
            final_model,
            self.train_features,
            self.train_targets,
        )
        test_ap_plot = self.evaluate_model(
            final_model,
            self.test_features,
            self.test_targets,
        )
        print(f"Training set AUC: {train_ap:.4f}")

        # evaluate on hold-out test set
        test_ap, test_predictions = self.evaluate_model(
            final_model, self.test_features, self.test_targets, return_predictions=True
        )
        print(f"Hold-out test set AUC: {test_ap:.4f}")

        # SHAP analysis for tree-based model
        shap_values = None
        if self.model_name == "xgboost":
            try:
                self.run_shap(final_model)
            except Exception as e:
                print(f"Error generating SHAP summary plot: {str(e)}")

        # save model
        model_path = f"{self.model_dir}/{self.model_name}_model.pkl"
        try:
            with open(model_path, "wb") as f:
                pickle.dump(final_model, f)
            print(f"Model saved to {model_path}")
        except Exception as e:
            print(f"Error saving model {self.model_name}: {str(e)}")

        # save training features and labels
        train_features_path = f"{self.model_dir}/{self.model_name}_train_features.npy"
        train_targets_path = f"{self.model_dir}/{self.model_name}_train_targets.npy"
        np.save(train_features_path, self.train_features)
        np.save(train_targets_path, self.train_targets)

        # save SHAP vals
        if shap_values is not None:
            shap_path = f"{self.model_dir}/{self.model_name}_shap_values.npy"
            np.save(shap_path, shap_values)

        predictions_out = {
            pair: test_predictions[i] for i, pair in enumerate(self.test_gene_pairs)
        }
        predictions_path = f"{self.model_dir}/{self.model_name}_test_predictions.pkl"
        with open(predictions_path, "wb") as f:
            pickle.dump(predictions_out, f)

        print(f"Test set predictions saved to {predictions_path}")

        return final_model

    def run_shap(self, final_model: BaselineModel) -> None:
        """Run SHAP analysis for a tree-based model."""
        explainer = shap.TreeExplainer(final_model.model)
        shap_values = explainer.shap_values(self.train_features)
        print("[SHAP) generating summary plot...")
        set_matplotlib_publication_parameters()
        shap.summary_plot(shap_values, self.train_features)
        plt.savefig(
            f"{self.model_dir}/{self.model_name}_shap_summary_plot.png", dpi=450
        )
        plt.close()

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
        model: BaselineModel,
        features: np.ndarray,
        labels: np.ndarray,
        return_predictions: bool = False,
    ) -> Union[float, Tuple[float, np.ndarray]]:
        """Evaluate a model and return its AUC score."""
        predicted_probabilities = model.predict_probability(features)
        ap_score = average_precision_score(labels, predicted_probabilities)
        if return_predictions:
            return ap_score, predicted_probabilities
        return ap_score


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
        "--ppi_directory",
        type=str,
        default="/ocean/projects/bio210019p/stevesho/genomic_nlp/ppi",
    )
    parser.add_argument(
        "--test_file_dir",
        type=str,
        default="/ocean/projects/bio210019p/stevesho/genomic_nlp/training_data",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="String to save the model with.",
        default="/ocean/projects/bio210019p/stevesho/genomic_nlp/models/interaction",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to word2vec model.",
        default="/ocean/projects/bio210019p/stevesho/genomic_nlp/models",
    )
    parser.add_argument(
        "--gene_names",
        type=str,
        help="Path to gene names file.",
        default="/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings/gene_synonyms.pkl",
    )
    parser.add_argument(
        "--n2v_type",
        type=str,
        help="Type of n2v embeddings to use.",
        choices=["ppi", "disease"],
        default="ppi",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        help="Type of embedding model to use.",
        choices=["w2v", "n2v"],
    )
    parser.add_argument(
        "--year", type=int, help="Year of the model to use.", default=2019
    )
    return parser.parse_args()


def get_gene_embeddings(
    args: argparse.Namespace, gene_names: Set[str]
) -> Tuple[Dict[str, np.ndarray], str]:
    """Get gene embeddings based on the model type."""
    model_dir = f"{args.model_path}/{args.model_type}"

    if args.model_type == "w2v":
        model = Word2Vec.load(
            f"{model_dir}/{args.year}/word2vec_300_dimensions_{args.year}.model"
        )
        gene_embeddings = _extract_gene_vectors(model, gene_names)
        save_path = f"{args.save_path}/w2v/{args.year}"
        os.makedirs(save_path, exist_ok=True)
    elif args.model_type == "n2v":
        model_path = f"{model_dir}/{args.n2v_type}/{args.year}/input_embeddings.pkl"
        with open(model_path, "rb") as f:
            embeddings = pickle.load(f)
        gene_embeddings = _extract_gene_vectors(embeddings, gene_names)
        save_path = f"{args.save_path}/n2v/{args.n2v_type}/{args.year}"
        os.makedirs(save_path, exist_ok=True)

    return gene_embeddings, save_path


def get_training_files(
    args: argparse.Namespace,
) -> Tuple[str, str, str, str]:
    """Get training file names."""
    positive_train_file = f"{args.ppi_directory}/gene_co_occurence_{args.year}.tsv"
    negative_train_file = f"{args.ppi_directory}/negative_edges_training.pkl"
    positive_test_file = f"{args.test_file_dir}/experimentally_derived_edges.pkl"
    negative_test_file = f"{args.test_file_dir}/negative_edges.pkl"
    return (
        positive_train_file,
        negative_train_file,
        positive_test_file,
        negative_test_file,
    )


def main() -> None:
    """Run baseline models for gene interaction prediction."""
    args = parse_args()
    print(f"Running {args.model_type} models for {args.year} embeddings.")

    # load test files
    (
        positive_train_file,
        negative_train_file,
        positive_test_file,
        negative_test_file,
    ) = get_training_files(args=args)

    # load gene names
    with open(args.gene_names, "rb") as f:
        gene_names = pickle.load(f)

    gene_names = set(gene_names.keys())

    # load embeddings and out path
    gene_embeddings, out_path = get_gene_embeddings(args=args, gene_names=gene_names)

    # process training data
    data_preprocessor = InteractionDataPreprocessor(
        embeddings=gene_embeddings,
        positive_train_file=positive_train_file,
        negative_train_file=negative_train_file,
        positive_test_file=positive_test_file,
        negative_test_file=negative_test_file,
    )
    (
        train_features,
        train_targets,
        test_features,
        test_targets,
        test_gene_pairs,
    ) = data_preprocessor.load_and_preprocess_data()

    print(f"No. of training examples: {len(train_features)}")
    print(f"Train features shape: {train_features.shape}")
    print(f"Train targets shape: {train_targets.shape}")
    print(f"Test features shape: {test_features.shape}")
    print(f"Test targets shape: {test_targets.shape}")

    # prepare stratified test data
    stratified_test_data = data_preprocessor.prepare_stratified_test_data(
        pos_test=data_preprocessor.positive_test_pairs,
        test_features=test_features,
        neg_test=data_preprocessor.negative_test_pairs,
    )

    # define models
    models = {
        "xgboost": XGBoost,
        "random_baseline": RandomBaseline,
        "logistic_regression": LogisticRegressionModel,
        # "svm": SVM,
        # "mlp": MLP,
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
            test_gene_pairs=test_gene_pairs,
            model_name=name,
            model_dir=out_path,
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

    visualizer = BaselineModelVisualizer(output_path=out_path)
    visualizer.plot_model_performances(
        train_results=train_results, test_results=test_results
    )
    visualizer.plot_stratified_performance(stratified_results=stratified_results)
    visualizer.plot_pr_curve(
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
