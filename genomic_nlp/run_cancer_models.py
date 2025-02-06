#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to run models to predict cancer genes.
"""


import argparse
import os
from pathlib import Path
import pickle
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from gensim.models import Word2Vec  # type: ignore
import numpy as np
from sklearn.metrics import average_precision_score  # type: ignore
from sklearn.metrics import precision_recall_curve  # type: ignore

from genomic_nlp.cancer_data_preprocessor import CancerGeneDataPreprocessor
from genomic_nlp.models.cancer_models import CancerBaseModel
from genomic_nlp.models.cancer_models import LogisticRegressionModel
from genomic_nlp.models.cancer_models import MLP
from genomic_nlp.models.cancer_models import SVM
from genomic_nlp.models.cancer_models import XGBoost
from genomic_nlp.utils.constants import RANDOM_STATE


class CancerGenePrediction:
    """Class used to train and evaluate oncogenicity prediction models.
    Trains a model on known cancer genes up to a year threshold on a model
    trained on text until that year. Then predicts potential cancer genes
    (identified cancer genes after the year threshold) using the trained model.
    """

    def __init__(
        self,
        model_class: Callable[..., CancerBaseModel],
        train_features: np.ndarray,
        train_targets: np.ndarray,
        test_features: np.ndarray,
        test_targets: np.ndarray,
        gene_embeddings: Dict[str, np.ndarray],
        model_name: str,
        model_dir: Path,
        year: int,
        cancer_genes: Set[str],
    ) -> None:
        """Initialize an OncogenicityPredictionTrainer object."""
        self.model_class = model_class
        self.train_features = train_features
        self.train_targets = train_targets
        self.test_features = test_features
        self.test_targets = test_targets
        self.gene_embeddings = gene_embeddings
        self.model_name = model_name
        self.model_dir = model_dir
        self.year = year
        self.cancer_genes = cancer_genes

        self.model: Optional[CancerBaseModel] = None

    def train_and_evaluate_once(self, **kwargs) -> None:
        """Train on (train_features, train_targets) and evaluate once on
        (test_features, test_targets). Save model artifacts/metrics.
        """
        # train model
        self.model = self.train_model(
            model_class=self.model_class,
            features=self.train_features,
            labels=self.train_targets,
            **kwargs,
        )

        # predict test set
        test_probabilities = self.model.predict_probability(self.test_features)

        # calculate PR AUC
        pr_auc = average_precision_score(self.test_targets, test_probabilities)
        print(f"Single train/test PR AUC: {pr_auc:.4f}")

        # save PR curve data
        precision, recall, thresholds = precision_recall_curve(
            self.test_targets, test_probabilities
        )
        pr_data = {"precision": precision, "recall": recall, "thresholds": thresholds}
        self.save_data(pr_data, f"pr_curve_data_{self.year}")

        # save model
        self.save_data(self.model, f"trained_model_{self.year}")

    def predict_all_genes(self) -> Dict[str, float]:
        """Predict cancer relatedness for all gene embeddings using the single
        trained model. Focus is on genes not already known to be cancer
        genes.
        """
        if not self.model:
            raise ValueError(
                "Model is not trained yet. Call train_and_evaluate_once first."
            )

        all_genes = set(self.gene_embeddings.keys())

        # remove known cancer genes
        predict_genes = list(all_genes - self.cancer_genes)
        all_embeddings = np.array(
            [self.gene_embeddings[gene] for gene in predict_genes]
        )

        predictions = self.model.predict_probability(all_embeddings)
        return dict(zip(all_genes, predictions))

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
    gene_embeddings: Dict[str, np.ndarray],
    year: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Path,
    Dict[str, np.ndarray],
    Set[str],
]:
    """Prepare data and directories for model training."""
    # load preprocessor
    preprocessor = CancerGeneDataPreprocessor(gene_embeddings=gene_embeddings)

    # load data
    train_features, train_targets, test_features, test_targets = (
        preprocessor.preprocess_data_by_year(year=year)
    )

    save_dir = (
        Path("/ocean/projects/bio210019p/stevesho/genomic_nlp/models/cancer")
        / args.save_str
    )
    os.makedirs(save_dir, exist_ok=True)
    return (
        train_features,
        train_targets,
        test_features,
        test_targets,
        save_dir,
        preprocessor.gene_embeddings,
        preprocessor.cancer_genes,
    )


def define_models() -> Dict[str, Callable[..., CancerBaseModel]]:
    """Define the models to be used in the ensemble."""
    return {
        "logistic_regression": LogisticRegressionModel,
        "xgboost": XGBoost,
        "svm": SVM,
        "mlp": MLP,
    }


def _extract_gene_vectors(model: Word2Vec, genes: List[str]) -> Dict[str, np.ndarray]:
    """Extract gene vectors from a word2vec model."""
    return {gene: model.wv[gene] for gene in genes if gene in model.wv.key_to_index}


def main() -> None:
    """Main function to run cancer gene prediction models."""
    # prep training data
    parser = argparse.ArgumentParser(
        description="Run baseline models for gene interaction prediction."
    )
    parser.add_argument("--save_str", type=str, help="String to save the model with.")
    parser.add_argument(
        "--w2v_model_path",
        type=str,
        help="Path to word2vec model.",
        default="/ocean/projects/bio210019p/stevesho/genomic_nlp/models/w2v",
    )
    parser.add_argument(
        "--gene_names",
        type=str,
        help="Path to gene names file.",
        default="/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings/gene_synonyms.pkl",
    )
    args = parser.parse_args()

    # load gene names
    with open(args.gene_names, "rb") as f:
        gene_names = pickle.load(f)

    gene_names = set(gene_names.keys())

    # train and test models via temporal split
    for year in range(2002, 2023):
        print(f"Running models for year {year}...")

        # load w2v model
        model = Word2Vec.load(
            f"{args.w2v_model_path}/{year}/word2vec_300_dimensions_{year}.model"
        )

        # extract gene vectors
        gene_embeddings = _extract_gene_vectors(model, gene_names)

        # prepare targets
        (
            train_features,
            train_targets,
            test_features,
            test_targets,
            save_dir,
            gene_embeddings,
            cancer_genes,
        ) = prepare_data(args=args, gene_embeddings=gene_embeddings, year=year)
        print(f"Total number of genes in training data: {len(train_features)}")
        print(f"Total number of genes in test data: {len(test_features)}")

        # define models
        models = define_models()

        print("Running models (single train/test).")
        for name, model_class in models.items():
            print(f"\nRunning {name} model...")

            # initialize trainer
            trainer = CancerGenePrediction(
                model_class=model_class,
                train_features=train_features,
                train_targets=train_targets,
                test_features=test_features,
                test_targets=test_targets,
                gene_embeddings=gene_embeddings,
                model_name=name,
                model_dir=save_dir,
                year=year,
                cancer_genes=cancer_genes,
            )

            # train and evaluate
            trainer.train_and_evaluate_once()

            # predict all genes
            final_predictions = trainer.predict_all_genes()
            trainer.save_data(final_predictions, f"final_predictions_{year}")

    print("All models have been processed!.")


if __name__ == "__main__":
    main()
