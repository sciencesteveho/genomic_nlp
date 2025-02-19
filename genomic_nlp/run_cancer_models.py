#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to run models to predict cancer genes.
"""


import argparse
import os
from pathlib import Path
import pickle
import random
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from gensim.models import Word2Vec  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import shap  # type: ignore
from sklearn.metrics import average_precision_score  # type: ignore
from sklearn.metrics import precision_recall_curve  # type: ignore
from sklearn.model_selection import StratifiedKFold  # type: ignore

from genomic_nlp.cancer_data_preprocessor import CancerGeneDataPreprocessor
from genomic_nlp.models.cancer_models import CancerBaseModel
from genomic_nlp.models.cancer_models import LogisticRegressionModel
from genomic_nlp.models.cancer_models import MLP
from genomic_nlp.models.cancer_models import RandomBaseline
from genomic_nlp.models.cancer_models import SVM
from genomic_nlp.models.cancer_models import XGBoost
from genomic_nlp.utils.constants import RANDOM_STATE
from genomic_nlp.visualization import set_matplotlib_publication_parameters


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
        save_dir: Path,
        year: int,
        cancer_genes: Set[str],
        horizon: Optional[int] = None,
    ) -> None:
        """Initialize an OncogenicityPredictionTrainer object."""
        self.model_class = model_class
        self.train_features = train_features
        self.train_targets = train_targets
        self.test_features = test_features
        self.test_targets = test_targets
        self.gene_embeddings = gene_embeddings
        self.model_name = model_name
        self.save_dir = save_dir
        self.year = year
        self.cancer_genes = cancer_genes
        self.horizon = horizon

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
        save_pr_name = (
            f"pr_curve_data_{self.year}_horizon_{self.horizon}"
            if self.horizon
            else f"pr_curve_data_{self.year}"
        )
        self.save_data(pr_data, save_pr_name)

        # save model
        save_model_name = (
            f"trained_model_{self.year}_horizon_{self.horizon}"
            if self.horizon
            else f"trained_model_{self.year}"
        )
        self.save_data(self.model, save_model_name)

        # report what percentage of predicted genes are cancer genes
        predicted_genes = {
            gene
            for gene, prob in zip(self.gene_embeddings.keys(), test_probabilities)
            if prob > 0.5
        }
        cancer_genes = set(self.cancer_genes)
        cancer_gene_count = len(predicted_genes & cancer_genes)
        total_gene_count = len(predicted_genes)
        print(
            f"Percentage of predicted genes that are cancer genes: {cancer_gene_count / total_gene_count:.4f}"
        )

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
        self.save_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.save_dir / f"{self.model_name}_{data_type}.pkl"

        try:
            with open(file_path, "wb") as f:
                pickle.dump(data, f)
            print(f"{data_type.capitalize()} saved to {file_path}")
        except Exception as e:
            print(f"Error saving {data_type} for {self.model_name}: {str(e)}")


def prepare_data(
    save_path: str,
    gene_embeddings: Dict[str, np.ndarray],
    year: int,
    horizon: Optional[int],
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
        preprocessor.preprocess_data_by_year(year=year, horizon=horizon)
    )

    # create save directory
    save_dir = Path(save_path) / str(year)
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
        # "logistic_regression": LogisticRegressionModel,
        "xgboost": XGBoost,
        "svm": SVM,
        "random_baseline": RandomBaseline,
        # "mlp": MLP,
    }


def _extract_gene_vectors(
    model: Union[Word2Vec, Dict[str, np.ndarray]], gene_names: Set[str]
) -> Dict[str, np.ndarray]:
    """Extract gene vectors from a word2vec model."""
    if type(model) == Word2Vec:
        return {
            gene: model.wv[gene] for gene in gene_names if gene in model.wv.key_to_index
        }
    elif type(model) == dict:
        return {gene: model[gene] for gene in gene_names if gene in model}
    else:
        raise ValueError("Model must be a Word2Vec model or a dictionary.")


def get_gene_embeddings(
    args: argparse.Namespace, gene_names: Set[str], year: int
) -> Tuple[Dict[str, np.ndarray], str]:
    """Get gene embeddings based on the model type."""
    model_dir = f"{args.model_path}/{args.model_type}"

    if args.model_type == "w2v":
        model = Word2Vec.load(
            f"{model_dir}/{year}/word2vec_300_dimensions_{year}.model"
        )
        gene_embeddings = _extract_gene_vectors(model, gene_names)
        save_path = args.save_path
    elif args.model_type == "n2v":
        model_path = f"{model_dir}/{args.n2v_type}/{year}/input_embeddings.pkl"
        with open(model_path, "rb") as f:
            embeddings = pickle.load(f)
        gene_embeddings = _extract_gene_vectors(embeddings, gene_names)
        save_path = f"{args.save_path}/n2v/{args.n2v_type}"

    return gene_embeddings, save_path


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Ensure that n2v embedding type is specified."""
    if args.model_type == "n2v" and not args.n2v_type:
        parser.error("n2v_type must be specified when using n2v embeddings.")


def run_final_model(
    args: argparse.Namespace,
    gene_names: Set[str],
) -> None:
    """Build a final dataset for 2023 using all known cancer genes. Run 5-fold
    CV + final model. Predict on all unlabeled data and run SHAP for feature
    importance.
    """
    # using xgboost for enhanced interpretability
    model_class = XGBoost

    # load embeddings
    gene_embeddings, save_path = get_gene_embeddings(
        args=args, gene_names=gene_names, year=2023
    )

    save_dir = Path(save_path) / "final_2023"
    save_dir.mkdir(parents=True, exist_ok=True)

    # initialize preprocessor
    preprocessor = CancerGeneDataPreprocessor(gene_embeddings=gene_embeddings)

    # get all positives
    all_known_cancer = set(preprocessor.cancer_genes)
    all_known_cancer = all_known_cancer & set(gene_embeddings.keys())

    # get all negatives
    all_genes = set(gene_embeddings.keys())
    unlabeled = all_genes - all_known_cancer

    # 1:1 negative sampling
    random.seed(RANDOM_STATE)
    all_positives = sorted(list(all_known_cancer))
    all_unlabeled = sorted(list(unlabeled))

    sample_size = int(1 * len(all_positives))
    sample_size = min(sample_size, len(all_unlabeled))
    negative_samples = random.sample(all_unlabeled, sample_size)

    # build training data
    train_genes = all_positives + negative_samples
    y_labels = [1] * len(all_positives) + [0] * len(negative_samples)
    X_train = np.array([gene_embeddings[g] for g in train_genes])
    y_train = np.array(y_labels, dtype=int)

    X_unlabeled = np.array([gene_embeddings[g] for g in all_unlabeled])

    # 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        # train
        fold_model = model_class()
        fold_model.train(feature_data=X_tr, target_labels=y_tr)

        # eval
        val_probs = fold_model.predict_probability(X_val)
        ap = average_precision_score(y_val, val_probs)
        fold_scores.append(ap)
        print(f"5-fold CV Fold {fold_idx+1}, AP={ap:.4f}")

    mean_ap = np.mean(fold_scores)
    print(f"[2023 5-fold CV] Mean AP: {mean_ap:.4f}")

    # train final model on all data
    final_model = model_class()
    final_model.train(feature_data=X_train, target_labels=y_train)

    # predict on unlabeled data
    unlabeled_probs_final = final_model.predict_probability(X_unlabeled)

    # shap tree explainer
    try:
        explainer = shap.TreeExplainer(final_model.model)
        shap_values = explainer.shap_values(X_train)
        print("[SHAP] Generating summary plot...")
        set_matplotlib_publication_parameters()  # set publication parameters
        shap.summary_plot(shap_values, X_train)
        plt.savefig(save_dir / "shap_summary_plot.png", dpi=450)
        plt.close()
    except Exception as e:
        print(f"[SHAP] Error: {e}")

    # save final model, features, predictions, and shap values
    with open(save_dir / "xgboost_final_2023.pkl", "wb") as f:
        pickle.dump(final_model, f)

    # save predictions
    unlabeled_predictions = dict(zip(all_unlabeled, unlabeled_probs_final))
    with open(save_dir / "unlabeled_predictions_2023.pkl", "wb") as f:
        pickle.dump(unlabeled_predictions, f)

    # save training features and labels
    np.save(save_dir / "X_train.npy", X_train)
    np.save(save_dir / "y_train.npy", y_train)

    # save shap values
    if shap_values is not None:
        np.save(save_dir / "shap_values.npy", shap_values)


def main() -> None:
    """Main function to run cancer gene prediction models."""
    # prep training data
    parser = argparse.ArgumentParser(
        description="Run baseline models for gene interaction prediction."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="String to save the model with.",
        default="/ocean/projects/bio210019p/stevesho/genomic_nlp/models/cancer",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to word2vec model.",
        default="/ocean/projects/bio210019p/stevesho/genomic_nlp/models",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        help="Type of embedding model to use.",
        default="n2v",
        choices=["w2v", "n2v"],
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
        "--year", type=int, help="Year of the model to use.", default=2019
    )
    args = parser.parse_args()
    _validate_args(parser=parser, args=args)

    # load gene names
    with open(args.gene_names, "rb") as f:
        gene_names = pickle.load(f)

    gene_names = set(gene_names.keys())

    # # train and test models via temporal split with horizon
    # for year in range(2003, 2016):
    #     if year == 2004:
    #         continue  # little data for 2004
    #     print(f"Running models for year {year}... with horizon")

    #     # load gene embeddings
    #     gene_embeddings, save_path = get_gene_embeddings(
    #         args=args, gene_names=gene_names, year=year
    #     )

    #     # prepare targets
    #     (
    #         train_features,
    #         train_targets,
    #         test_features,
    #         test_targets,
    #         save_dir,
    #         gene_embeddings,
    #         cancer_genes,
    #     ) = prepare_data(
    #         save_path=save_path,
    #         gene_embeddings=gene_embeddings,
    #         year=year,
    #         horizon=3,
    #     )
    #     print(f"Total number of genes in training data: {len(train_features)}")
    #     print(f"Total number of genes in test data: {len(test_features)}")

    #     # define models
    #     models = define_models()

    #     print("Running models (single train/test).")
    #     for name, model_class in models.items():
    #         print(f"\nRunning {name} model...")

    #         # initialize trainer
    #         trainer = CancerGenePrediction(
    #             model_class=model_class,
    #             train_features=train_features,
    #             train_targets=train_targets,
    #             test_features=test_features,
    #             test_targets=test_targets,
    #             gene_embeddings=gene_embeddings,
    #             model_name=name,
    #             save_dir=save_dir,
    #             year=year,
    #             cancer_genes=cancer_genes,
    #         )

    #         # train and evaluate
    #         trainer.train_and_evaluate_once()

    #         # predict all genes
    #         final_predictions = trainer.predict_all_genes()
    #         trainer.save_data(final_predictions, f"final_predictions_{year}_horizon")

    # # train and test models via temporal split without horizon
    # for year in range(2003, 2020):
    #     print(f"Running models for year {year}...")

    #     # load gene embeddings
    #     gene_embeddings, save_path = get_gene_embeddings(
    #         args=args, gene_names=gene_names, year=year
    #     )

    #     # prepare
    #     (
    #         train_features,
    #         train_targets,
    #         test_features,
    #         test_targets,
    #         save_dir,
    #         gene_embeddings,
    #         cancer_genes,
    #     ) = prepare_data(
    #         save_path=save_path,
    #         gene_embeddings=gene_embeddings,
    #         year=year,
    #         horizon=None,
    #     )
    #     print(f"Total number of genes in training data: {len(train_features)}")
    #     print(f"Total number of genes in test data: {len(test_features)}")

    #     # define models
    #     models = define_models()

    #     print("Running models (single train/test).")
    #     for name, model_class in models.items():
    #         print(f"\nRunning {name} model...")

    #         # initialize trainer
    #         trainer = CancerGenePrediction(
    #             model_class=model_class,
    #             train_features=train_features,
    #             train_targets=train_targets,
    #             test_features=test_features,
    #             test_targets=test_targets,
    #             gene_embeddings=gene_embeddings,
    #             model_name=name,
    #             save_dir=save_dir,
    #             year=year,
    #             cancer_genes=cancer_genes,
    #         )

    #         # train and evaluate
    #         trainer.train_and_evaluate_once()

    #         # predict all genes
    #         final_predictions = trainer.predict_all_genes()
    #         trainer.save_data(final_predictions, f"final_predictions_{year}")

    # run final model for 2023
    run_final_model(args=args, gene_names=gene_names)
    print("All models have been processed!.")


if __name__ == "__main__":
    main()
