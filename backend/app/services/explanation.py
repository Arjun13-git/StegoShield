"""Model-based evidence for a result, described honestly.

Indicators rank features by
    (global Random Forest importance) x (|deviation of the image's value from
    the reference clean-cover distribution|, in reference standard deviations),
    listing features that separate clean from LSB-stego in the reference data
    before those that do not.
The direction says whether the deviation points the same way as the difference
between clean and LSB-stego reference images. This is a heuristic summary of
measured statistics; it is NOT a causal or per-prediction attribution, and a
feature deviating does not prove hidden data is present.
"""

from __future__ import annotations

import numpy as np

from app.schemas.api import FeatureIndicator
from app.services.reference import ReferenceData

TOP_N = 5
DEVIATION_THRESHOLD = 0.5  # reference standard deviations
MIN_CLASS_SEPARATION = 0.05  # reference-stego mean shift, in cover std, below which a feature is "informational"

EXPLANATION_METHOD = (
    "Features ranked by global model importance times deviation from the reference clean-image distribution. "
    "Heuristic model-based evidence, not a causal or per-prediction attribution."
)

_STAT = {
    "mean": "Mean intensity",
    "std": "Intensity standard deviation",
    "lsb_ratio": "Share of pixels with least-significant bit = 1",
    "lsb_transition": "LSB transition rate",
    "lsb_local_variance": "Block-to-block variance of local LSB density",
    "entropy": "Intensity histogram entropy",
}
_CHANNEL = {"r": "red channel", "g": "green channel", "b": "blue channel"}
_GLOBAL = {
    "neighbor_diff_mean": "Mean absolute neighbouring-pixel difference",
    "neighbor_diff_std": "Spread of neighbouring-pixel differences",
    "highpass_mean_abs": "Mean absolute high-pass residual",
    "highpass_std": "High-pass residual standard deviation",
}


def feature_label(name: str) -> str:
    if name in _GLOBAL:
        return _GLOBAL[name]
    stat, _, channel = name.rpartition("_")
    if stat in _STAT and channel in _CHANNEL:
        return f"{_STAT[stat]} ({_CHANNEL[channel]})"
    return name


def build_indicators(
    vector: np.ndarray,
    names: list[str] | tuple[str, ...],
    importances: np.ndarray | None,
    reference: ReferenceData | None,
    top_n: int = TOP_N,
) -> list[FeatureIndicator]:
    values = [float(v) for v in vector]
    imp = importances if importances is not None else np.zeros(len(names))

    if reference is None:
        order = sorted(range(len(names)), key=lambda j: -imp[j])[:top_n]
        return [
            FeatureIndicator(name=names[j], label=feature_label(names[j]), value=values[j], direction="informational",
                             deviation_from_reference_cover=None, model_importance=float(imp[j]) if importances is not None else None)
            for j in order
        ]

    def separation(name: str) -> float:
        ref = reference.features[name]
        std = ref.cover_std if ref.cover_std > 1e-12 else 1e-12
        return (ref.stego_mean - ref.cover_mean) / std  # signed clean->stego shift, in cover std

    scored = []
    for j, name in enumerate(names):
        ref = reference.features[name]
        std = ref.cover_std if ref.cover_std > 1e-12 else 1e-12
        z = (values[j] - ref.cover_mean) / std
        # Features that do not separate clean from LSB-stego in the reference data are ranked last: a large
        # deviation there says the image differs from the reference set, not that it resembles stego.
        discriminative = abs(separation(name)) >= MIN_CLASS_SEPARATION
        scored.append((discriminative, imp[j] * abs(z), j, z))
    scored.sort(key=lambda t: (not t[0], -t[1]))

    out = []
    for _, _, j, z in scored[:top_n]:
        name = names[j]
        shift = separation(name)
        if abs(shift) < MIN_CLASS_SEPARATION:
            direction = "informational"
        else:
            alignment = float(np.sign(shift)) * z
            direction = ("increases_stego_signal" if alignment > DEVIATION_THRESHOLD
                         else "decreases_stego_signal" if alignment < -DEVIATION_THRESHOLD else "informational")
        out.append(FeatureIndicator(name=name, label=feature_label(name), value=values[j], direction=direction,
                                    deviation_from_reference_cover=round(float(z), 3),
                                    model_importance=round(float(imp[j]), 5) if importances is not None else None))
    return out
