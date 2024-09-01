# sourcery skip: snake-case-arguments
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
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.neural_network import MLPClassifier  # type: ignore
from sklearn.svm import SVC  # type: ignore
from xgboost import XGBClassifier

from constants import RANDOM_STATE


class OncogenicityBaseModel(BaseEstimator, ClassifierMixin):
    """Base class for oncogenicity prediction models."""

    def __init__(self, model: BaseEstimator, binary_threshold: float = 0.5):
        self.model = model
        self.binary_threshold = binary_threshold
        self.input_dim: Optional[int] = None

    def train(
        self, feature_data: np.ndarray, target_labels: np.ndarray
    ) -> "OncogenicityBaseModel":
        """Train the model."""
        self.input_dim = feature_data.shape[1]
        self.model.fit(feature_data, target_labels)
        return self

    def predict_probability(self, input_features: np.ndarray) -> np.ndarray:
        """Predict the probability of a gene being oncogenic."""
        processed_features = self._process_input(input_features)
        return self.model.predict_proba(processed_features)[:, 1]

    def predict_binary(self, input_features: np.ndarray) -> np.ndarray:
        """Predict binary oncogenicity based on threshold."""
        probabilities = self.predict_probability(input_features)
        return (probabilities >= self.binary_threshold).astype(int)

    def _process_input(self, input_features: np.ndarray) -> np.ndarray:
        """Process input features to match the model's input dimension."""
        if self.input_dim is None:
            raise ValueError("Model has not been trained yet.")
        if input_features.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, got {input_features.shape[1]}"
            )
        return input_features


class OncogenicityLogisticRegression(OncogenicityBaseModel):
    """Logistic regression for oncogenicity prediction."""

    def __init__(self, **kwargs):
        model = LogisticRegression(**kwargs)
        super().__init__(model)


class OncogenicitySVM(OncogenicityBaseModel):
    """Support vector machine - non-linear margin-based classification model."""

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        random_state: int = RANDOM_STATE,
        **kwargs,
    ):
        model = SVC(
            kernel=kernel, C=C, probability=True, random_state=random_state, **kwargs
        )
        super().__init__(model)


class OncogenicityRandomForest(OncogenicityBaseModel):
    """Random forest for oncogenicity prediction."""

    def __init__(
        self, n_estimators: int = 100, random_state: int = RANDOM_STATE, **kwargs
    ):
        model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state, **kwargs
        )
        super().__init__(model)


class OncogenicityXGBoost(OncogenicityBaseModel):
    """XGBoost for oncogenicity prediction."""

    def __init__(self, **kwargs):
        model = XGBClassifier(eval_metric="logloss", **kwargs)
        super().__init__(model)


class OncogenicityMLP(OncogenicityBaseModel):
    """Multi-layer perceptron for oncogenicity prediction."""

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
