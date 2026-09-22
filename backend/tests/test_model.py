from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from app.ml.model import ModelBundle


def _make_bundle(tmp_path: Path, *, with_scaler: bool) -> Path:
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(-3, 1, (30, 4)), rng.normal(3, 1, (30, 4))])
    y = np.array([0] * 30 + [1] * 30)

    scaler = None
    x_fit = x
    if with_scaler:
        scaler = StandardScaler()
        x_fit = scaler.fit_transform(x)

    model = LogisticRegression().fit(x_fit, y)
    bundle = {
        "model": model,
        "model_name": "LogisticRegression",
        "model_version": "v1-test",
        "feature_schema_version": "v1",
        "scaler": scaler,
    }
    path = tmp_path / "bundle.joblib"
    joblib.dump(bundle, path)
    return path


def test_predict_score_without_scaler(tmp_path: Path) -> None:
    path = _make_bundle(tmp_path, with_scaler=False)
    bundle = ModelBundle(str(path))
    bundle.load()
    assert bundle.loaded
    score = bundle.predict_score(np.array([3.0, 3.0, 3.0, 3.0], dtype=np.float32))
    assert 0.0 <= score <= 1.0
    assert score > 0.5  # matches the positive-class cluster


def test_predict_score_applies_scaler_when_present(tmp_path: Path) -> None:
    path = _make_bundle(tmp_path, with_scaler=True)
    bundle = ModelBundle(str(path))
    bundle.load()
    score = bundle.predict_score(np.array([3.0, 3.0, 3.0, 3.0], dtype=np.float32))
    assert 0.0 <= score <= 1.0
    assert score > 0.5


def test_metadata_reports_feature_schema_version(tmp_path: Path) -> None:
    path = _make_bundle(tmp_path, with_scaler=False)
    bundle = ModelBundle(str(path))
    bundle.load()
    meta = bundle.metadata()
    assert meta["name"] == "LogisticRegression"
    assert meta["feature_schema_version"] == "v1"


def test_unloaded_bundle_reports_unavailable_metadata() -> None:
    bundle = ModelBundle("/nonexistent/path/model.joblib")
    bundle.load()
    assert not bundle.loaded
    assert bundle.metadata() == {"name": "unavailable", "version": "none"}
