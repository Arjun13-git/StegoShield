"""Feature extraction contract.

Keep this module deterministic and versioned. Training and inference MUST call the
same extractor to avoid feature-order drift.
"""

from collections.abc import Sequence

import numpy as np
from PIL import Image

FEATURE_SCHEMA_VERSION = "v1"

FEATURE_NAMES: tuple[str, ...] = (
    "mean_r", "mean_g", "mean_b",
    "std_r", "std_g", "std_b",
    "lsb_ratio_r", "lsb_ratio_g", "lsb_ratio_b",
    "lsb_transition_r", "lsb_transition_g", "lsb_transition_b",
    "lsb_local_variance_r", "lsb_local_variance_g", "lsb_local_variance_b",
    "entropy_r", "entropy_g", "entropy_b",
    "neighbor_diff_mean",
    "neighbor_diff_std",
    "highpass_mean_abs",
    "highpass_std",
)

_LSB_BLOCK_SIZE = 8


def _entropy(values: np.ndarray) -> float:
    hist = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    probs = hist[hist > 0] / values.size
    return float(-(probs * np.log2(probs)).sum())


def _lsb_transition_rate(channel: np.ndarray) -> float:
    bits = channel & 1
    horizontal = np.mean(bits[:, 1:] != bits[:, :-1])
    vertical = np.mean(bits[1:, :] != bits[:-1, :])
    return float((horizontal + vertical) / 2)


def _lsb_local_variance(channel: np.ndarray, block: int = _LSB_BLOCK_SIZE) -> float:
    """Variance, across non-overlapping blocks, of the local mean LSB value.

    A natural image's LSB plane retains some correlation with local texture,
    so block-to-block LSB density varies. LSB embedding pushes each block's
    bit density toward an independent Bernoulli(0.5), which *lowers* this
    block-to-block variance. This is the "local LSB randomness" indicator
    called out in the ML design doc, distinct from the global lsb_ratio and
    the pairwise lsb_transition rate above.
    """
    bits = (channel & 1).astype(np.float64)
    height, width = bits.shape
    h_crop = (height // block) * block
    w_crop = (width // block) * block
    if h_crop == 0 or w_crop == 0:
        return 0.0
    cropped = bits[:h_crop, :w_crop]
    blocks = cropped.reshape(h_crop // block, block, w_crop // block, block)
    block_means = blocks.mean(axis=(1, 3))
    return float(block_means.var())


def extract_features(image: Image.Image) -> tuple[np.ndarray, Sequence[str]]:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    channels = [arr[..., i] for i in range(3)]

    values: list[float] = []
    for channel in channels:
        values.append(float(channel.mean()))
    for channel in channels:
        values.append(float(channel.std()))
    for channel in channels:
        values.append(float(np.mean(channel & 1)))
    for channel in channels:
        values.append(_lsb_transition_rate(channel))
    for channel in channels:
        values.append(_lsb_local_variance(channel))
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
