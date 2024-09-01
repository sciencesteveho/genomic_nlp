#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to run models to predict cancer genes. Models take in gene embeddings
that are known to be cancer associated (COSMIC + NCG) along with negatively
sampled embeddings and are trained on binary classification with
cross-validation, as there is no "true" test set. The models are then used to
predict cancer relatedness for all gene embeddings in the dataset."""


import argparse
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
        model_dir: str,
    ) -> None:
        """Initialize an OncogenicityPredictionTrainer object."""
        self.model_class = model_class
        self.features = features
        self.targets = targets
        self.gene_embeddings = gene_embeddings
        self.model_name = model_name
        self.model_dir = Path(model_dir)

    def perform_cross_validation(self, n_splits: int = 5, **kwargs) -> List[float]:
        """Perform stratified 5-fold cross-validation and return F1 scores and thresholds."""
        folds = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
        )
        cv_scores = []
        cv_data = []

        for train_index, val_index in folds.split(X=self.features, y=self.targets):
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
            cv_data.append((val_targets, probabilities))

        self.save_data(cv_data, "cv")
        return cv_scores

    def train_and_evaluate_model(self) -> CancerBaseModel:
        """Train a model using 5-fold CV and return the final model."""
        print(f"\nTraining and evaluating {self.model_name}:")

        # perform 5-fold CV
        cv_scores = self.perform_cross_validation(n_splits=5)
        mean_cv_auc = np.mean(cv_scores)
        std_cv_auc = np.std(cv_scores)
        print(
            f"Cross-validation Mean ROC AUC: {mean_cv_auc:.4f} (+/- {std_cv_auc:.4f})"
        )

        # train final model on entire dataset
        final_model = self.train_model(self.model_class, self.features, self.targets)
        final_probas = final_model.predict_probability(self.features)
        final_auc = roc_auc_score(self.targets, final_probas)
        self.save_data([(self.targets, final_probas)], "final_roc")
        print(f"Final model ROC AUC on entire dataset: {final_auc:.4f}")

        # save model
        self.save_data(final_model, "model")
        return final_model

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


def main() -> None:
    """Main function to run cancer gene prediction models."""
    # prep training data
    parser = argparse.ArgumentParser(
        description="Run baseline models for gene interaction prediction."
    )
    parser.add_argument(
        "--embeddings", type=str, help="Path to gene embeddings pickle file."
    )
    args = parser.parse_args()
    preprocessor = CancerGeneDataPreprocessor(args)
    features, targets = preprocessor.preprocess_data()
    save_dir = "/ocean/projects/bio210019p/stevesho/genomic_nlp/models/cancer"

    # define models
    models = {
        "logistic_regression": LogisticRegressionModel,
        "svm": SVM,
        "xgboost": XGBoost,
        "mlp": MLP,
    }

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

        # final model on all train data
        final_model = trainer.train_and_evaluate_model()

        # predict cancer relatedness for all genes
        all_gene_predictions = trainer.predict_all_genes(final_model)
        print(f"Predicted cancer relatedness for genes in model {name}.")

        # save the predictions
        trainer.save_data(all_gene_predictions, "predictions")


if __name__ == "__main__":
    main()
