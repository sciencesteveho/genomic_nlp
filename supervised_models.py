import random
from typing import Dict, List, Tuple, Union

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


class SupervisedGGInteractionPredictor:
    def __init__(self, embeddings: np.ndarray, gene_ids: List[str]):
        self.embeddings: np.ndarray = embeddings
        self.gene_ids: List[str] = gene_ids
        self.gene_to_index: Dict[str, int] = {
            gene: idx for idx, gene in enumerate(gene_ids)
        }
        self.scaler: StandardScaler = StandardScaler()

    def prepare_data(
        self,
        positive_pairs: List[Tuple[str, str]],
        negative_pairs: Union[List[Tuple[str, str]], None] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        X: List[np.ndarray] = []
        y: List[int] = []

        # Process positive pairs
        for gene1, gene2 in positive_pairs:
            emb1 = self.embeddings[self.gene_to_index[gene1]]
            emb2 = self.embeddings[self.gene_to_index[gene2]]
            X.append(np.concatenate([emb1, emb2, np.abs(emb1 - emb2)]))
            y.append(1)

        # Process negative pairs if provided, otherwise generate random negative pairs
        if negative_pairs is None:
            num_negative = len(positive_pairs)
            all_pairs = set(
                (g1, g2)
                for i, g1 in enumerate(self.gene_ids)
                for g2 in self.gene_ids[i + 1 :]
            )
            negative_pairs = random.sample(
                all_pairs - set(positive_pairs), num_negative
            )

        for gene1, gene2 in negative_pairs:
            emb1 = self.embeddings[self.gene_to_index[gene1]]
            emb2 = self.embeddings[self.gene_to_index[gene2]]
            X.append(np.concatenate([emb1, emb2, np.abs(emb1 - emb2)]))
            y.append(0)

        X_array: np.ndarray = np.array(X)
        y_array: np.ndarray = np.array(y)

        # Scale features
        X_scaled: np.ndarray = self.scaler.fit_transform(X_array)

        return X_scaled, y_array

    def train_and_evaluate(
        self, X: np.ndarray, y: np.ndarray, model_type: str = "logistic", cv: int = 5
    ) -> Dict[str, float]:
        if model_type == "logistic":
            model = LogisticRegression(random_state=42)
        elif model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == "xgboost":
            model = xgb.XGBClassifier(
                use_label_encoder=False, eval_metric="logloss", random_state=42
            )
        else:
            raise ValueError("Unsupported model type")

        # Perform cross-validation
        cv_scores: np.ndarray = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

        # Train on full dataset for final evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model.fit(X_train, y_train)
        y_pred: np.ndarray = model.predict_proba(X_test)[:, 1]

        return {
            "cv_auc_roc": float(np.mean(cv_scores)),
            "test_auc_roc": float(roc_auc_score(y_test, y_pred)),
            "test_avg_precision": float(average_precision_score(y_test, y_pred)),
        }

    def evaluate_embeddings(
        self,
        positive_pairs: List[Tuple[str, str]],
        model_type: str = "logistic",
        cv: int = 5,
    ) -> Dict[str, float]:
        X, y = self.prepare_data(positive_pairs)
        return self.train_and_evaluate(X, y, model_type, cv)


# Example usage
embeddings: np.ndarray = np.random.rand(1000, 300)  # Replace with actual embeddings
gene_ids: List[str] = [f"gene_{i}" for i in range(1000)]
positive_pairs: List[Tuple[str, str]] = [
    ("gene_1", "gene_2"),
    ("gene_3", "gene_4"),
]  # Replace with actual positive pairs

predictor = SupervisedGGInteractionPredictor(embeddings, gene_ids)

results_logistic: Dict[str, float] = predictor.evaluate_embeddings(
    positive_pairs, model_type="logistic"
)
results_rf: Dict[str, float] = predictor.evaluate_embeddings(
    positive_pairs, model_type="random_forest"
)
results_xgb: Dict[str, float] = predictor.evaluate_embeddings(
    positive_pairs, model_type="xgboost"
)

print("Logistic Regression Results:", results_logistic)
print("Random Forest Results:", results_rf)
print("XGBoost Results:", results_xgb)
