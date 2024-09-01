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
    
Also included is GNN model architectures for link prediction. We train
link-prediction GNNs to compare the rich semantic capture capabilites of
different language models. Because the goal of the GNN is to evaluate the
quality of the embeddings, we default to a simpler GNN architecture to avoid
overfitting and complexity that may hinder the evaluation of the embeddings."""


from typing import Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
from sklearn.neural_network import MLPClassifier  # type: ignore
from sklearn.svm import SVC  # type: ignore
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv  # type: ignore
from torch_geometric.nn import GraphNorm
from xgboost import XGBClassifier

from constants import RANDOM_STATE


class LinkPredictionGNN(nn.Module):
    """Simple GNN for link prediction, trained on a base graph initialized with
    trained embeddings and evaluated on a separate graph of experimentally
    derived gene-gene interactions.

    The model defaults to two convolutional layers with graph normalization and
    ReLU for its activation function. Additionally, we implement dropouut and a
    dense skip connection.
    """

    def __init__(
        self,
        in_channels: int,
        embedding_size: int,
        out_channels: int,
        dropout: float = 0.2,
    ) -> None:
        """Instantiate the model."""
        super(LinkPredictionGNN, self).__init__()
        self.activation = F.relu
        self.dropout = nn.Dropout(dropout)

        # convolutional layers
        self.conv1 = GCNConv(in_channels, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)

        # normalization layers
        self.norm1 = GraphNorm(embedding_size)
        self.norm2 = GraphNorm(embedding_size)

        # skip connection
        self.residual = nn.Linear(in_channels, out_channels)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encode the input graph."""
        x1 = self.dropout(self.activation(self.norm1(self.conv1(x, edge_index))))
        x2 = self.dropout(self.activation(self.norm2(self.conv2(x1, edge_index))))
        return self.residual(x) + x2

    def decode(self, z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        """Decode the input graph via dot product of node embeddings."""
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_label_index: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the network to predict links."""
        z = self.encode(x=x, edge_index=edge_index)
        return self.decode(z=z, edge_label_index=edge_label_index)


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
