"""Shared training/evaluation logic for the classical ML candidates.

Kept independent of any CLI/script orchestration so both `scripts/*.py` and
the test suite can call the exact same fitting/evaluation code paths that
end up inside the saved model artifact.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_SEED = 42

# Whether each candidate needs standardized features. Random Forest is
# scale-invariant; LR and SVM (especially with an RBF kernel) are not.
NEEDS_SCALING: dict[str, bool] = {
    "LogisticRegression": True,
    "SVM": True,
    "RandomForest": False,
}


def build_candidate_models(seed: int = RANDOM_SEED) -> dict[str, Any]:
    """Sensible-default estimators. Intentionally not heavily tuned: the
    project goal is credible generalization on a small dataset, not chasing
    accuracy through a large hyperparameter search (see CLAUDE.md section 5.2).
    """
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            random_state=seed,
            class_weight="balanced",
        ),
        "SVM": SVC(
            kernel="rbf",
            probability=True,
            random_state=seed,
            class_weight="balanced",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=20,
            random_state=seed,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }


@dataclass
class TrainedModel:
    name: str
    estimator: Any
    scaler: StandardScaler | None
    train_seconds: float


def fit_model(
    name: str,
    estimator: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    needs_scaling: bool,
) -> TrainedModel:
    scaler: StandardScaler | None = None
    x_fit = x_train
    if needs_scaling:
        scaler = StandardScaler()
        x_fit = scaler.fit_transform(x_train)

    start = time.perf_counter()
    estimator.fit(x_fit, y_train)
    elapsed = time.perf_counter() - start
    return TrainedModel(name=name, estimator=estimator, scaler=scaler, train_seconds=elapsed)


def predict_proba(trained: TrainedModel, x: np.ndarray) -> np.ndarray:
    x_in = trained.scaler.transform(x) if trained.scaler is not None else x
    return trained.estimator.predict_proba(x_in)[:, 1]


@dataclass
class EvalMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: list[list[int]]
    n_samples: int
    n_positive: int
    n_negative: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "confusion_matrix": self.confusion_matrix,
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
        }


def evaluate(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> EvalMetrics:
    y_true = np.asarray(y_true)
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    classes_present = set(np.unique(y_true).tolist())
    roc_auc = float(roc_auc_score(y_true, y_proba)) if len(classes_present) > 1 else float("nan")

    return EvalMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=roc_auc,
        confusion_matrix=cm.tolist(),
        n_samples=int(len(y_true)),
        n_positive=int((y_true == 1).sum()),
        n_negative=int((y_true == 0).sum()),
    )
