"""Reference distributions that give the raw model score an honest scale.

The frozen Random Forest emits an UNCALIBRATED score that clusters near 0.5
(Phase 1: at a 0.5 threshold 71.9% of clean held-out BOSSBase covers already
score >= 0.5). A fixed 0.5 cut-off would therefore flag most clean images, so
risk is instead defined against an empirical reference distribution of clean
images:

  * cover_percentile = fraction of reference clean images whose score is <= this score
  * risk 'elevated'  = at or above the 90th reference-cover percentile
  * risk 'high'      = at or above the 99th reference-cover percentile

These bands are operating points expressed as false-alarm rates on the
reference set. They are NOT probabilities and are not tuned on any results.

The reference file is generated (scripts/build_reference_scores.py) from the
Phase 1A dataset with the frozen model: cover scores come from the VALIDATION
split; per-feature reference statistics (used only for explanations) come from
the TRAIN split. Flag rates on the held-out TEST split are stored for
description only. The file is bound to the model by SHA-256; a mismatch
disables it rather than mislead.
"""

from __future__ import annotations

import bisect
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

logger = logging.getLogger("stegoshield.reference")

ELEVATED_PERCENTILE = 0.90
HIGH_PERCENTILE = 0.99
REFERENCE_FORMAT_VERSION = 1


def risk_from_percentile(percentile: float) -> str:
    if percentile >= HIGH_PERCENTILE:
        return "high"
    if percentile >= ELEVATED_PERCENTILE:
        return "elevated"
    return "low"


@dataclass(frozen=True)
class FeatureReference:
    cover_mean: float
    cover_std: float
    stego_mean: float


@dataclass(frozen=True)
class ReferenceData:
    model_sha256: str
    feature_schema_version: str
    description: str
    payload_note: str
    cover_scores_sorted: tuple[float, ...]
    features: dict[str, FeatureReference]
    operating_points: dict[str, dict]

    def cover_percentile(self, score: float) -> float:
        """Empirical CDF of the reference clean-cover scores at `score` (0..1)."""
        return bisect.bisect_right(self.cover_scores_sorted, score) / len(self.cover_scores_sorted)


def load_reference(path: Path, *, model_sha256: str, feature_names: list[str], feature_schema_version: str) -> ReferenceData | None:
    """Load and verify the reference; return None (with a logged reason) if it is
    missing or does not belong to the loaded model, instead of guessing."""
    if not path.is_file():
        logger.warning("reference file not found; risk levels disabled")
        return None
    try:
        raw = json.loads(path.read_text())
        if raw.get("format_version") != REFERENCE_FORMAT_VERSION:
            raise ValueError("unsupported reference format")
        if raw["model_sha256"] != model_sha256:
            logger.warning("reference file is bound to a different model artifact; risk levels disabled")
            return None
        if raw["feature_schema_version"] != feature_schema_version or list(raw["feature_names"]) != list(feature_names):
            logger.warning("reference file feature schema differs from the model; risk levels disabled")
            return None
        scores = tuple(sorted(float(x) for x in raw["val_cover_scores"]))
        if len(scores) < 100:
            raise ValueError("reference has too few cover scores")
        feats = {n: FeatureReference(**raw["features"][n]) for n in feature_names}
        return ReferenceData(
            model_sha256=raw["model_sha256"], feature_schema_version=raw["feature_schema_version"],
            description=raw["description"], payload_note=raw["payload_note"], cover_scores_sorted=scores,
            features=feats, operating_points=raw["operating_points"],
        )
    except Exception:  # noqa: BLE001 - a bad reference must not take the API down
        logger.exception("reference file invalid; risk levels disabled")
        return None


# ---------------------------------------------------------------------------
# Builder (used by scripts/build_reference_scores.py; pure so it can be tested)
# ---------------------------------------------------------------------------


def build_reference(
    *,
    model_sha256: str,
    feature_schema_version: str,
    feature_names: list[str],
    predict_proba: Callable[[np.ndarray], np.ndarray],
    train_x_cover: np.ndarray,
    train_x_stego: np.ndarray,
    val_x_cover: np.ndarray,
    val_x_stego: np.ndarray,
    test_x_cover: np.ndarray,
    test_x_stego: np.ndarray,
    payload_note: str,
    description: str,
) -> dict:
    val_cover = np.sort(np.asarray(predict_proba(val_x_cover), dtype=np.float64))

    def percentile(scores: np.ndarray) -> np.ndarray:
        return np.searchsorted(val_cover, scores, side="right") / len(val_cover)

    def rate(scores: np.ndarray, threshold: float) -> float:
        return float((percentile(scores) >= threshold).mean())

    val_stego = np.asarray(predict_proba(val_x_stego), dtype=np.float64)
    test_cover = np.asarray(predict_proba(test_x_cover), dtype=np.float64)
    test_stego = np.asarray(predict_proba(test_x_stego), dtype=np.float64)

    operating_points = {}
    for level, threshold in (("elevated", ELEVATED_PERCENTILE), ("high", HIGH_PERCENTILE)):
        eligible = val_cover[percentile(val_cover) >= threshold]
        operating_points[level] = {
            "cover_percentile_threshold": threshold,
            "min_score": float(eligible.min()),
            "flag_rate": {
                "validation_clean": rate(val_cover, threshold),
                "validation_lsb_stego": rate(val_stego, threshold),
                "test_clean": rate(test_cover, threshold),
                "test_lsb_stego": rate(test_stego, threshold),
            },
        }

    features = {}
    for j, name in enumerate(feature_names):
        c = train_x_cover[:, j].astype(np.float64)
        s = train_x_stego[:, j].astype(np.float64)
        features[name] = {"cover_mean": float(c.mean()), "cover_std": float(c.std(ddof=1)), "stego_mean": float(s.mean())}

    return {
        "format_version": REFERENCE_FORMAT_VERSION,
        "model_sha256": model_sha256,
        "feature_schema_version": feature_schema_version,
        "feature_names": list(feature_names),
        "description": description,
        "payload_note": payload_note,
        "n": {"train_cover": len(train_x_cover), "val_cover": len(val_x_cover), "val_stego": len(val_x_stego),
              "test_cover": len(test_x_cover), "test_stego": len(test_x_stego)},
        "val_cover_scores": [float(x) for x in val_cover],
        "features": features,
        "operating_points": operating_points,
    }
