# sourcery skip: avoid-single-character-names-variables, snake-case-arguments, upper-camel-case-classes
#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Baseline models for predicting gene-gene interactions. We adopted a base
class for which the other models inherit from.

The following are implemented:
    (0) Pairwise similarity scoring based on cosine distance 
    (1) Logistic regression
    (2) Random forest
    (3) XGBoost
    (4) Multi-layer perceptron (MLP)
"""


from typing import Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
from sklearn.neural_network import MLPClassifier  # type: ignore
from sklearn.svm import SVC  # type: ignore
from xgboost import XGBClassifier

from genomic_nlp.utils.constants import RANDOM_STATE


class BaselineModel(BaseEstimator, ClassifierMixin):
    """Base class for gene interaction prediction models. Defines input
    processing, model training, and prediction methods.
    """

    def __init__(self, model: BaseEstimator, binary_threshold: float = 0.7):
        self.model = model
        self.binary_threshold = binary_threshold
        self.input_dim: Optional[int] = None

    def train(
        self, feature_data: np.ndarray, target_labels: np.ndarray
    ) -> "BaselineModel":
        """Call model.fit() to train the model."""
        if feature_data.shape[1] % 2 != 0:
            raise ValueError("Input dimension must be even for paired vectors")
        self.input_dim = feature_data.shape[1] // 2
        self.model.fit(feature_data, target_labels)
        return self

    def predict_probability(
        self, input_features: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        """Given input features, return the probability of interaction."""
        processed_features = self._process_input(input_features)
        return self.model.predict_proba(processed_features)[:, 1]

    def predict_binary(
        self, input_features: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        """Given input features, return binary prediction based on threshold."""
        processed_features = self._process_input(input_features)
        probabilities = self.model.predict_proba(processed_features)[:, 1]
        return (probabilities >= self.binary_threshold).astype(int)

    def _process_input(
        self, input_features: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        """Process input features to match the model's input dimension. Most of
        the models require a concatenated vector of two input vectors, but
        pairwise similarity does not."""
        if isinstance(input_features, tuple):
            return np.hstack(input_features)
        elif (
            self.input_dim is not None and input_features.shape[1] == self.input_dim * 2
        ):
            return input_features
        elif input_features.shape[1] == self.input_dim:
            return np.hstack([input_features, input_features])
        else:
            raise ValueError("Input dimension mismatch")


class LogisticRegressionModel(BaselineModel):
    """Logistic regression - linear classification model."""

    def __init__(self, **kwargs):
        model = LogisticRegression(**kwargs)
        super().__init__(model)


class SVM(BaselineModel):
    """Support vector machine - non-linear margin-based classification model."""

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        random_state: int = RANDOM_STATE,
        **kwargs
    ):
        model = SVC(
            kernel=kernel, C=C, probability=True, random_state=random_state, **kwargs
        )
        super().__init__(model)


class RandomForest(BaselineModel):
    """Random forest - ensemble decision tree model."""

    def __init__(self, n_estimators: int = 100, random_state: int = 42, **kwargs):
        model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state, **kwargs
        )
        super().__init__(model)


class XGBoost(BaselineModel):
    """XGBoost - gradient boosting decision tree model."""

    def __init__(self, **kwargs):
        model = XGBClassifier(eval_metric="logloss", **kwargs)
        super().__init__(model)


class MLP(BaselineModel):
    """Multi-layer perceptron - feedforward neural network model."""

    def __init__(
        self, hidden_layer_sizes=(256, 256), max_iter=1000, random_state=42, **kwargs
    ):
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            **kwargs
        )
        super().__init__(model)


class CosineSimilarity:
    """Use pairwise similarity as if were a typical learning model."""

    def __init__(self):
        self.input_dim = None

    def predict_probability(self, input_features: np.ndarray) -> np.ndarray:
        """Compute pairwise similarity scores."""
        if self.input_dim is None:
            self.input_dim = input_features.shape[1] // 2
        scores = []
        for pair in input_features:
            vec1, vec2 = pair[: self.input_dim], pair[self.input_dim :]
            scores.append(self.pairwise_similarity_score(vec1, vec2))
        return np.array(scores)

    @staticmethod
    def pairwise_similarity_score(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Similarity score based on cosine distance between two vectors."""
        return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
