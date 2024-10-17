#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to run models to predict cancer genes. Models take in gene embeddings
that are known to be cancer associated (COSMIC + NCG) along with negatively
sampled embeddings and are trained on binary classification with
cross-validation, as there is no "true" test set. The models are then used to
predict cancer relatedness for all gene embeddings in the dataset."""


import argparse
import os
from pathlib import Path
import pickle
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score  # type: ignore
from sklearn.metrics import roc_curve  # type: ignore
from sklearn.model_selection import StratifiedKFold  # type: ignore

from cancer_models import CancerBaseModel
from cancer_models import LogisticRegressionModel
from cancer_models import MLP
from cancer_models import SVM
from cancer_models import XGBoost
from constants import RANDOM_STATE
from model_data_preprocessor import CancerGeneDataPreprocessor

# from utils import get_physical_cores


class CancerGenePrediction:
    """Class used to train and evaluate oncogenicity prediction models.
    Performs 5-fold cross-validation and predicts cancer relatedness for all
    gene embeddings.
    """

    def __init__(
        self,
        model_class: Callable[..., CancerBaseModel],
        features: np.ndarray,
        targets: np.ndarray,
        gene_embeddings: Dict[str, np.ndarray],
        model_name: str,
        model_dir: Path,
    ) -> None:
        """Initialize an OncogenicityPredictionTrainer object."""
        self.model_class = model_class
        self.features = features
        self.targets = targets
        self.gene_embeddings = gene_embeddings
        self.model_name = model_name
        self.model_dir = model_dir

    def perform_cross_validation(
        self, n_splits: int = 5, **kwargs
    ) -> Tuple[List[float], np.ndarray, List[CancerBaseModel]]:
        """Perform stratified 5-fold cross-validation, return AUC scores,
        combined probabilities for soft voting, and trained models."""
        folds = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
        )
        cv_scores = []
        cv_val_probabilities = np.zeros(len(self.targets))
        trained_models = []
        all_val_indices = np.array([], dtype=int)

        for fold_num, (train_index, val_index) in enumerate(
            folds.split(self.features, self.targets), 1
        ):
            train_features, val_features = (
                self.features[train_index],
                self.features[val_index],
            )
            train_targets, val_targets = (
                self.targets[train_index],
                self.targets[val_index],
            )

            model = self.train_model(
                model_class=self.model_class,
                features=train_features,
                labels=train_targets,
                **kwargs,
            )

            probabilities = model.predict_probability(val_features)
            cv_scores.append(roc_auc_score(val_targets, probabilities))
            cv_val_probabilities[val_index] = probabilities
            trained_models.append(model)
            all_val_indices = np.concatenate([all_val_indices, val_index])
            print(f"Fold {fold_num} ROC AUC: {cv_scores[-1]:.4f}")

        # Save the ROC curve data
        fpr, tpr, thresholds = roc_curve(
            self.targets[all_val_indices], cv_val_probabilities[all_val_indices]
        )
        roc_data = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}
        self.save_data(roc_data, "roc_curve_data")

        self.save_data(cv_scores, "cv_scores")
        self.save_data(cv_val_probabilities, "cv_val_probabilities")
        self.save_data(trained_models, "trained_models")
        return cv_scores, cv_val_probabilities, trained_models

    # def train_and_evaluate_model(self) -> CancerBaseModel:
    #     """Train a model using 5-fold CV and return the final model."""
    #     print(f"\nTraining and evaluating {self.model_name}:")

    #     # perform 5-fold CV
    #     cv_scores = self.perform_cross_validation(n_splits=5)
    #     mean_cv_auc = np.mean(cv_scores)
    #     std_cv_auc = np.std(cv_scores)
    #     print(
    #         f"Cross-validation Mean ROC AUC: {mean_cv_auc:.4f} (+/- {std_cv_auc:.4f})"
    #     )

    #     # train final model on entire dataset
    #     final_model = self.train_model(self.model_class, self.features, self.targets)
    #     final_probas = final_model.predict_probability(self.features)
    #     final_auc = roc_auc_score(self.targets, final_probas)
    #     self.save_data([(self.targets, final_probas)], "final_roc")
    #     print(f"Final model ROC AUC on entire dataset: {final_auc:.4f}")

    #     # save model
    #     self.save_data(final_model, "model")
    #     return final_model

    def predict_all_genes(self, model: CancerBaseModel) -> Dict[str, float]:
        """Infer cancer relatedness for all gene embeddings."""
        all_genes = list(self.gene_embeddings.keys())
        all_embeddings = np.array([self.gene_embeddings[gene] for gene in all_genes])
        probabilities = model.predict_probability(all_embeddings)  # inference
        return dict(zip(all_genes, probabilities))

    @staticmethod
    def train_model(
        model_class: Callable[..., CancerBaseModel],
        features: np.ndarray,
        labels: np.ndarray,
        **kwargs,
    ) -> CancerBaseModel:
        """Train a model on given features and labels."""
        model = model_class(**kwargs)
        model.train(feature_data=features, target_labels=labels)
        return model

    def save_data(self, data: Any, data_type: str) -> None:
        """Save data to the model directory."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.model_dir / f"{self.model_name}_{data_type}.pkl"

        try:
            with open(file_path, "wb") as f:
                pickle.dump(data, f)
            print(f"{data_type.capitalize()} saved to {file_path}")
        except Exception as e:
            print(f"Error saving {data_type} for {self.model_name}: {str(e)}")


def prepare_data(
    args: argparse.Namespace,
) -> Tuple[
    np.ndarray, np.ndarray, Path, Dict[str, np.ndarray], CancerGeneDataPreprocessor
]:
    """Prepare data and directories for model training."""
    preprocessor = CancerGeneDataPreprocessor(args)
    features, targets = preprocessor.preprocess_data()
    save_dir = (
        Path("/ocean/projects/bio210019p/stevesho/genomic_nlp/models/cancer")
        / args.save_str
    )
    os.makedirs(save_dir, exist_ok=True)
    return features, targets, save_dir, preprocessor.gene_embeddings, preprocessor


def define_models() -> Dict[str, Callable[..., CancerBaseModel]]:
    """Define the models to be used in the ensemble."""
    return {
        "logistic_regression": LogisticRegressionModel,
        "xgboost": XGBoost,
        "svm": SVM,
        "mlp": MLP,
    }


def main() -> None:
    """Main function to run cancer gene prediction models."""
    # prep training data
    parser = argparse.ArgumentParser(
        description="Run baseline models for gene interaction prediction."
    )
    parser.add_argument(
        "--embeddings", type=str, help="Path to gene embeddings pickle file."
    )
    parser.add_argument("--save_str", type=str, help="String to save the model with.")
    args = parser.parse_args()

    # prepare data
    features, targets, save_dir, gene_embeddings, preprocessor = prepare_data(args)

    # define models
    models = define_models()

    # initialize vars for 5-fold cv and soft voting ensemble
    ensemble_val_probabilities = np.zeros(len(targets))
    ensemble_final_probabilities = np.zeros(len(preprocessor.gene_embeddings))

    print("Running models.")

    for name, model_class in models.items():
        print(f"\nRunning {name} model.")
        trainer = CancerGenePrediction(
            model_class=model_class,
            features=features,
            targets=targets,
            gene_embeddings=preprocessor.gene_embeddings,
            model_name=name,
            model_dir=save_dir,
        )

        # perform cross-validation
        _, cv_val_probs, trained_models = trainer.perform_cross_validation(n_splits=5)

        # Soft voting on the training labels
        ensemble_val_probabilities += cv_val_probs

        # predict on the entire dataset
        all_genes = list(preprocessor.gene_embeddings.keys())
        all_embeddings = np.array(
            [preprocessor.gene_embeddings[gene] for gene in all_genes]
        )
        model_predictions = np.zeros(len(all_genes))

        for model in trained_models:
            probas = model.predict_probability(all_embeddings)
            model_predictions += probas

        # soft vote
        model_predictions /= len(trained_models)
        ensemble_final_probabilities += model_predictions
        trainer.save_data(model_predictions, f"{name}_final_probas")

    # average votes
    num_models = len(models)
    ensemble_val_probabilities /= num_models
    ensemble_final_probabilities /= num_models

    # evaluate ensemble
    ensemble_cv_auc = roc_auc_score(targets, ensemble_val_probabilities)
    print(f"\nEnsemble Cross-validation ROC AUC: {ensemble_cv_auc:.4f}")

    # Save the ensemble ROC curve data
    ensemble_fpr, ensemble_tpr, ensemble_thresholds = roc_curve(
        targets, ensemble_val_probabilities
    )
    ensemble_roc_data = {
        "fpr": ensemble_fpr,
        "tpr": ensemble_tpr,
        "thresholds": ensemble_thresholds,
    }
    ensemble_roc_path = save_dir / "ensemble_roc_curve_data.pkl"
    with open(ensemble_roc_path, "wb") as f:
        pickle.dump(ensemble_roc_data, f)
    print(f"Ensemble ROC curve data saved to {ensemble_roc_path}")

    # predict cancer relatedness for all genes using ensemble
    ensemble_final_predictions = dict(zip(all_genes, ensemble_final_probabilities))

    # save probas and predictions
    ensemble_cv_path = save_dir / "ensemble_cv_probabilities.pkl"
    with open(ensemble_cv_path, "wb") as f:
        pickle.dump(ensemble_val_probabilities, f)
    print(f"Ensemble cross-validation probabilities saved to {ensemble_cv_path}")

    ensemble_predictions_path = save_dir / "ensemble_final_predictions.pkl"
    with open(ensemble_predictions_path, "wb") as f:
        pickle.dump(ensemble_final_predictions, f)
    print(f"Ensemble final predictions saved to {ensemble_predictions_path}")


if __name__ == "__main__":
    main()
