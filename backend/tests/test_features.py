import numpy as np
from PIL import Image

from app.ml.features import FEATURE_NAMES, extract_features


def _random_rgb(seed: int = 0, size: tuple[int, int] = (64, 64)) -> Image.Image:
    arr = np.random.default_rng(seed).integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_feature_schema_is_stable() -> None:
    image = Image.new("RGB", (128, 128), color=(120, 90, 60))
    vector, names = extract_features(image)
    assert len(vector) == len(FEATURE_NAMES)
    assert tuple(names) == FEATURE_NAMES


def test_feature_vector_length_matches_schema() -> None:
    image = _random_rgb()
    vector, _ = extract_features(image)
    assert vector.shape == (len(FEATURE_NAMES),)


def test_feature_extraction_is_deterministic() -> None:
    image = _random_rgb(seed=7)
    v1, _ = extract_features(image)
    v2, _ = extract_features(image)
    assert np.array_equal(v1, v2)


def test_feature_output_is_finite() -> None:
    image = _random_rgb(seed=3)
    vector, _ = extract_features(image)
    assert np.isfinite(vector).all()


def test_feature_output_finite_for_uniform_image() -> None:
    # Degenerate case: zero variance everywhere (entropy of a single value = 0).
    image = Image.new("RGB", (64, 64), color=(10, 10, 10))
    vector, _ = extract_features(image)
    assert np.isfinite(vector).all()


def test_grayscale_input_produces_matching_rgb_channels() -> None:
    arr = np.random.default_rng(5).integers(0, 256, size=(64, 64), dtype=np.uint8)
    gray_image = Image.fromarray(arr)
    vector, names = extract_features(gray_image)
    by_name = dict(zip(names, vector))
    # A grayscale source, once converted to RGB (R=G=B), must yield identical
    # per-channel statistics across r/g/b for every channel-wise feature.
    assert by_name["mean_r"] == by_name["mean_g"] == by_name["mean_b"]
    assert by_name["std_r"] == by_name["std_g"] == by_name["std_b"]
    assert by_name["lsb_ratio_r"] == by_name["lsb_ratio_g"] == by_name["lsb_ratio_b"]


def test_rgb_image_with_distinct_channels_yields_distinct_stats() -> None:
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    arr[..., 0] = 10
    arr[..., 1] = 120
    arr[..., 2] = 240
    image = Image.fromarray(arr)
    vector, names = extract_features(image)
    by_name = dict(zip(names, vector))
    assert by_name["mean_r"] != by_name["mean_g"] != by_name["mean_b"]


def test_lsb_local_variance_is_lower_for_random_lsb_plane() -> None:
    """lsb_local_variance_* measures block-to-block spread of local LSB
    density. An image whose LSB plane is clustered into all-0 / all-1
    regions has high block variance; an image with an independently random
    LSB plane (which is what LSB embedding approximates) should have much
    lower block variance, since every block's mean converges toward 0.5.
    """
    height, width = 64, 64
    base = np.full((height, width), 100, dtype=np.uint8)
    clustered = base.copy()
    clustered[: height // 2, :] &= 0xFE  # top half: LSB forced to 0
    clustered[height // 2 :, :] |= 0x01  # bottom half: LSB forced to 1

    rng = np.random.default_rng(11)
    randomized_lsb = (base & 0xFE) | rng.integers(0, 2, size=base.shape, dtype=np.uint8)

    v_clustered, names = extract_features(Image.fromarray(clustered))
    v_random, _ = extract_features(Image.fromarray(randomized_lsb))
    idx = names.index("lsb_local_variance_r")
    assert v_random[idx] < v_clustered[idx]
