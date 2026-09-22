from pathlib import Path

import joblib
import numpy as np

from app.ml.training import (
    NEEDS_SCALING,
    build_candidate_models,
    evaluate,
    fit_model,
    predict_proba,
)


def _separable_dataset(n_per_class: int = 60, n_features: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(loc=-2.0, scale=1.0, size=(n_per_class, n_features))
    x1 = rng.normal(loc=2.0, scale=1.0, size=(n_per_class, n_features))
    x = np.vstack([x0, x1]).astype(np.float32)
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)]).astype(int)
    return x, y


def test_build_candidate_models_returns_all_three() -> None:
    models = build_candidate_models(seed=42)
    assert set(models.keys()) == {"LogisticRegression", "SVM", "RandomForest"}


def test_fit_and_predict_on_separable_data_is_near_perfect() -> None:
    x, y = _separable_dataset()
    models = build_candidate_models(seed=42)
    for name, estimator in models.items():
        trained = fit_model(name, estimator, x, y, needs_scaling=NEEDS_SCALING[name])
        proba = predict_proba(trained, x)
        metrics = evaluate(y, proba)
        assert metrics.accuracy > 0.9, f"{name} underfit an easily separable dataset"


def test_random_forest_does_not_require_scaler() -> None:
    x, y = _separable_dataset()
    trained = fit_model("RandomForest", build_candidate_models()["RandomForest"], x, y, needs_scaling=False)
    assert trained.scaler is None


def test_logistic_regression_gets_a_scaler() -> None:
    x, y = _separable_dataset()
    trained = fit_model(
        "LogisticRegression", build_candidate_models()["LogisticRegression"], x, y, needs_scaling=True
    )
    assert trained.scaler is not None


def test_evaluate_on_random_labels_is_near_chance() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=400)
    y_proba = rng.random(400)  # uninformative predictions
    metrics = evaluate(y_true, y_proba)
    assert 0.35 <= metrics.accuracy <= 0.65
    assert 0.35 <= metrics.roc_auc <= 0.65


def test_evaluate_confusion_matrix_shape_and_totals() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.6, 0.9])
    metrics = evaluate(y_true, y_proba)
    cm = np.array(metrics.confusion_matrix)
    assert cm.shape == (2, 2)
    assert cm.sum() == 4
    assert metrics.n_positive == 2
    assert metrics.n_negative == 2


def test_model_artifact_round_trip(tmp_path: Path) -> None:
    x, y = _separable_dataset()
    trained = fit_model(
        "RandomForest", build_candidate_models()["RandomForest"], x, y, needs_scaling=False
    )
    bundle = {
        "model": trained.estimator,
        "model_name": "RandomForest",
        "model_version": "v1-test",
        "feature_schema_version": "v1",
        "feature_names": [f"f{i}" for i in range(x.shape[1])],
        "random_seed": 42,
        "scaler": trained.scaler,
        "metrics": {"accuracy": 1.0},
    }
    path = tmp_path / "model.joblib"
    joblib.dump(bundle, path)

    loaded = joblib.load(path)
    assert loaded["model_name"] == "RandomForest"
    assert loaded["feature_schema_version"] == "v1"
    proba = loaded["model"].predict_proba(x)[:, 1]
    assert proba.shape == (x.shape[0],)
