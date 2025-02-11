# sourcery skip: snake-case-arguments, upper-camel-case-classes
#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Baseline models for predicting potential oncogenic and tumor supressor genes.
We adopted a base class for which the other models inherit from.

The following are implemented:
    (0) Pairwise similarity scoring based on cosine distance 
    (1) Logistic regression
    (2) SVM
    (3) XGBoost
    (4) Multi-layer perceptron (MLP)
"""


from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.base import ClassifierMixin  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.multiclass import OneVsRestClassifier  # type: ignore
from sklearn.neural_network import MLPClassifier  # type: ignore
from sklearn.svm import SVC  # type: ignore
from xgboost import XGBClassifier

from genomic_nlp.utils.constants import RANDOM_STATE


class CancerBaseModel(BaseEstimator, ClassifierMixin):
    """Base class for oncogenicity prediction models."""

    def __init__(self, model: BaseEstimator, threshold: float = 0.5) -> None:
        """Initialize a cancer gene prediction model."""
        self.model = model
        self.threshold = threshold
        self.input_dim: Optional[int] = None

    def train(
        self, feature_data: np.ndarray, target_labels: np.ndarray
    ) -> "CancerBaseModel":
        """Train the model."""
        self.input_dim = feature_data.shape[1]
        self.model.fit(feature_data, target_labels)
        return self

    def predict_probability(self, input_features: np.ndarray) -> np.ndarray:
        """Predict the probability of a gene being cancer-related."""
        processed_features = self._process_input(input_features)
        return self.model.predict_proba(processed_features)[:, 1]

    def predict(self, input_features: np.ndarray) -> np.ndarray:
        """Predict whether a gene is cancer-related (1) or not (0)."""
        probabilities = self.predict_probability(input_features)
        return (probabilities >= self.threshold).astype(int)

    def _process_input(self, input_features: np.ndarray) -> np.ndarray:
        """Process input features to match the model's input dimension."""
        if self.input_dim is None:
            raise ValueError("Model has not been trained yet.")
        if input_features.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, got {input_features.shape[1]}"
            )
        return input_features


class RandomBaseline(CancerBaseModel):
    """
    A baseline model that ignores the input features
    and returns random probabilities for the positive class.
    """

    def __init__(self) -> None:
        """Initialize a random baseline model."""
        super().__init__(model=None)

    def train(
        self, feature_data: np.ndarray, target_labels: np.ndarray
    ) -> CancerBaseModel:
        """Do nothing, as this is a baseline model."""
        feature_data = feature_data
        target_labels = target_labels
        return self

    def predict_probability(self, feature_data: np.ndarray) -> np.ndarray:
        """Return random probabilities between 0 and 1."""
        return np.random.rand(len(feature_data))


class LogisticRegressionModel(CancerBaseModel):
    """Logistic regression for cancer prediction."""

    def __init__(self, **kwargs):
        model = LogisticRegression(**kwargs)
        super().__init__(model)


class SVM(CancerBaseModel):
    """Support vector machine - non-linear margin-based classification model."""

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 5,
        **kwargs,
    ):
        model = SVC(
            kernel=kernel,
            C=C,
            probability=True,
            random_state=RANDOM_STATE,
            gamma="scale",
            **kwargs,
        )
        super().__init__(model)


class XGBoost(CancerBaseModel):
    """XGBoost for cancer prediction."""

    def __init__(self, **kwargs):
        model = XGBClassifier(
            eval_metric="aucpr",
            n_estimators=300,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            seed=RANDOM_STATE,
            reg_lambda=1,
            **kwargs,
        )
        super().__init__(model)


class MLP(CancerBaseModel):
    """Multi-layer perceptron for cancer prediction."""

    def __init__(
        self, hidden_layer_sizes=(256, 256), max_iter=1000, random_state=42, **kwargs
    ):
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            **kwargs,
        )
        super().__init__(model)
