from pathlib import Path
from typing import Any

import joblib
import numpy as np


class ModelBundle:
    """Small inference wrapper. Artifact schema is documented in System-Design/ml."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.bundle: dict[str, Any] | None = None

    def load(self) -> None:
        if self.path.exists():
            self.bundle = joblib.load(self.path)

    @property
    def loaded(self) -> bool:
        return self.bundle is not None

    def predict_score(self, vector: np.ndarray) -> float:
        if not self.bundle:
            raise RuntimeError("No trained model artifact is loaded")
        model = self.bundle["model"]
        x = vector.reshape(1, -1)
        scaler = self.bundle.get("scaler")
        if scaler is not None:
            x = scaler.transform(x)
        probability = model.predict_proba(x)[0, 1]
        return float(probability)

    def metadata(self) -> dict[str, str]:
        if not self.bundle:
            return {"name": "unavailable", "version": "none"}
        return {
            "name": self.bundle.get("model_name", "unknown"),
            "version": self.bundle.get("model_version", "unknown"),
            "feature_schema_version": self.bundle.get("feature_schema_version", "unknown"),
        }
