"""Shared fixtures for the API tests.

The core integration tests use the REAL frozen Phase 1 artifact
(data/models/stegoshield_rf.joblib, git-ignored). If it is absent they are
skipped with an explicit reason rather than silently faked. Failure-path
tests use a tiny synthetic model bundle built in a temp directory.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import warnings
from pathlib import Path

import joblib
import numpy as np
import pytest
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

from app.core.config import DEFAULT_MODEL_PATH, FROZEN_MODEL_SHA256, Settings
from app.ml.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION

warnings.filterwarnings("ignore", message=".*httpx.*starlette.testclient.*")

REAL_MODEL_AVAILABLE = DEFAULT_MODEL_PATH.is_file()
requires_real_model = pytest.mark.skipif(
    not REAL_MODEL_AVAILABLE, reason="frozen Phase 1 artifact data/models/stegoshield_rf.joblib is not present locally"
)


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def jpeg_bytes(arr: np.ndarray, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def noise(shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=shape, dtype=np.uint8)


def make_settings(**overrides) -> Settings:
    return dataclasses.replace(Settings(), **overrides)


def write_tiny_bundle(path: Path, *, feature_names: list[str] | None = None, schema: str = FEATURE_SCHEMA_VERSION,
                      drop_key: str | None = None) -> str:
    """Write a small synthetic model bundle in the Phase 1 artifact format; return its SHA-256."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, len(FEATURE_NAMES)))
    y = (x[:, 0] + rng.normal(scale=0.5, size=200) > 0).astype(int)
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(x, y)
    bundle = {
        "model": model, "model_name": "TinyTestForest", "model_version": "t1",
        "feature_schema_version": schema, "feature_names": feature_names or list(FEATURE_NAMES),
        "scaler": None, "training_dataset_id": "synthetic", "random_seed": 0,
        "metrics": {"test": {"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5, "roc_auc": 0.5, "n_samples": 10}},
    }
    if drop_key:
        bundle.pop(drop_key)
    joblib.dump(bundle, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="session")
def frozen_model_hash() -> str:
    return FROZEN_MODEL_SHA256
