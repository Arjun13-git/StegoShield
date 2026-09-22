"""Feature extraction contract.

Keep this module deterministic and versioned. Training and inference MUST call the
same extractor to avoid feature-order drift.
"""

from collections.abc import Sequence

import numpy as np
from PIL import Image

FEATURE_NAMES: tuple[str, ...] = (
    "mean_r", "mean_g", "mean_b",
    "std_r", "std_g", "std_b",
    "lsb_ratio_r", "lsb_ratio_g", "lsb_ratio_b",
    "lsb_transition_r", "lsb_transition_g", "lsb_transition_b",
    "entropy_r", "entropy_g", "entropy_b",
    "neighbor_diff_mean",
    "neighbor_diff_std",
    "highpass_mean_abs",
    "highpass_std",
)


def _entropy(values: np.ndarray) -> float:
    hist = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    probs = hist[hist > 0] / values.size
    return float(-(probs * np.log2(probs)).sum())


def _lsb_transition_rate(channel: np.ndarray) -> float:
    bits = channel & 1
    horizontal = np.mean(bits[:, 1:] != bits[:, :-1])
    vertical = np.mean(bits[1:, :] != bits[:-1, :])
    return float((horizontal + vertical) / 2)


def extract_features(image: Image.Image) -> tuple[np.ndarray, Sequence[str]]:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    channels = [arr[..., i] for i in range(3)]

    values: list[float] = []
    for channel in channels:
        values.extend([float(channel.mean()), float(channel.std())])
    for channel in channels:
        values.append(float(np.mean(channel & 1)))
    for channel in channels:
        values.append(_lsb_transition_rate(channel))
    for channel in channels:
        values.append(_entropy(channel))

    gray = arr.mean(axis=2)
    dx = np.diff(gray, axis=1)
    dy = np.diff(gray, axis=0)
    residual = gray - (
        np.roll(gray, 1, axis=0) + np.roll(gray, -1, axis=0) +
        np.roll(gray, 1, axis=1) + np.roll(gray, -1, axis=1)
    ) / 4
    values.extend([
        float((np.abs(dx).mean() + np.abs(dy).mean()) / 2),
        float((np.abs(dx).std() + np.abs(dy).std()) / 2),
        float(np.abs(residual).mean()),
        float(residual.std()),
    ])

    vector = np.asarray(values, dtype=np.float32)
    if len(vector) != len(FEATURE_NAMES):
        raise RuntimeError("Feature vector/schema mismatch")
    return vector, FEATURE_NAMES
