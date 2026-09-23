"""Loads the frozen model ONCE and exposes it to the API.

Security: joblib is pickle-based, so deserialising an unverified file can run
arbitrary code. The artifact's bytes are therefore hashed and compared with
the pinned SHA-256 BEFORE being deserialised, and deserialisation uses those
same bytes (no second read, so no time-of-check/time-of-use gap). The model
path comes only from server configuration; nothing here accepts a path or
file from a request.

The existing Phase 1 `ModelBundle` is reused unchanged for inference; this
class only verifies, loads and describes it.
"""

from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.core.config import EXPECTED_FEATURE_SCHEMA_VERSION, Settings
from app.ml.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from app.ml.model import ModelBundle

logger = logging.getLogger("stegoshield.model")

_REQUIRED_KEYS = ("model", "model_name", "model_version", "feature_schema_version", "feature_names")


class ModelUnavailableError(RuntimeError):
    """The frozen model could not be loaded or verified. `reason` is a short
    machine-readable tag safe to expose; details stay in the server log."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ModelMetadata:
    name: str
    version: str
    feature_schema_version: str
    feature_count: int
    artifact_sha256: str
    training_dataset_id: str | None
    random_seed: int | None
    reference_performance: dict[str, Any] | None


class ModelService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._bundle: ModelBundle | None = None
        self._metadata: ModelMetadata | None = None
        self._importances: np.ndarray | None = None
        self._failure: str | None = "not_loaded"

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        """Verify and load the artifact. Never raises: failure is recorded and
        surfaced through `ready`/`failure_reason` so the API can start and
        report not-ready instead of crashing or silently degrading.
        """
        self._bundle = None
        self._metadata = None
        self._importances = None
        try:
            self._load()
            self._failure = None
            logger.info("model loaded: %s %s (schema %s)", self._metadata.name, self._metadata.version, self._metadata.feature_schema_version)
        except ModelUnavailableError as exc:
            self._failure = exc.reason
            logger.error("model unavailable: %s", exc.reason)
        except Exception:  # noqa: BLE001
            self._failure = "load_failed"
            logger.exception("model failed to load")

    def _load(self) -> None:
        path: Path = self._settings.model_path
        if not path.is_file():
            raise ModelUnavailableError("artifact_missing")

        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        expected = self._settings.model_sha256
        if expected is not None and digest != expected:
            raise ModelUnavailableError("artifact_hash_mismatch")

        raw = joblib.load(io.BytesIO(data))  # only reached after the hash check
        if not isinstance(raw, dict) or any(k not in raw for k in _REQUIRED_KEYS):
            raise ModelUnavailableError("artifact_malformed")
        if raw["feature_schema_version"] != EXPECTED_FEATURE_SCHEMA_VERSION or FEATURE_SCHEMA_VERSION != EXPECTED_FEATURE_SCHEMA_VERSION:
            raise ModelUnavailableError("feature_schema_version_mismatch")
        if list(raw["feature_names"]) != list(FEATURE_NAMES):
            raise ModelUnavailableError("feature_names_mismatch")
        if int(getattr(raw["model"], "n_features_in_", -1)) != len(FEATURE_NAMES):
            raise ModelUnavailableError("feature_count_mismatch")

        bundle = ModelBundle(str(path))
        bundle.bundle = raw  # reuse the unchanged Phase 1 inference wrapper
        self._bundle = bundle
        test_metrics = (raw.get("metrics") or {}).get("test")
        self._metadata = ModelMetadata(
            name=str(raw["model_name"]),
            version=str(raw["model_version"]),
            feature_schema_version=str(raw["feature_schema_version"]),
            feature_count=len(FEATURE_NAMES),
            artifact_sha256=digest,
            training_dataset_id=raw.get("training_dataset_id"),
            random_seed=raw.get("random_seed"),
            reference_performance=(
                {k: test_metrics[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "n_samples") if k in test_metrics}
                if isinstance(test_metrics, dict) else None
            ),
        )
        importances = getattr(raw["model"], "feature_importances_", None)
        self._importances = np.asarray(importances, dtype=np.float64) if importances is not None else None

    # ---------------------------------------------------------------- state
    @property
    def ready(self) -> bool:
        return self._failure is None and self._bundle is not None

    @property
    def failure_reason(self) -> str | None:
        return self._failure

    @property
    def metadata(self) -> ModelMetadata:
        if self._metadata is None:
            raise ModelUnavailableError(self._failure or "not_loaded")
        return self._metadata

    @property
    def feature_importances(self) -> np.ndarray | None:
        return self._importances

    # ------------------------------------------------------------- inference
    def predict_score(self, vector: np.ndarray) -> float:
        """Raw model output in [0, 1] (uncalibrated). Uses the Phase 1 ModelBundle."""
        if not self.ready or self._bundle is None:
            raise ModelUnavailableError(self._failure or "not_loaded")
        if vector.shape != (len(FEATURE_NAMES),):
            raise ValueError("feature vector has the wrong length")
        return self._bundle.predict_score(vector)
