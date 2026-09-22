import numpy as np
import pytest

from app.ml.lsb import embed_lsb, lsb_capacity_bits, sample_seed, validate_payload


def _pixels(seed: int = 0, shape: tuple[int, int] = (32, 32)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def test_capacity_equals_pixel_count() -> None:
    assert lsb_capacity_bits((32, 32)) == 1024
    assert lsb_capacity_bits((512, 512)) == 262144


def test_validate_payload_rejects_out_of_range() -> None:
    validate_payload(0.0)
    validate_payload(1.0)
    with pytest.raises(ValueError):
        validate_payload(-0.01)
    with pytest.raises(ValueError):
        validate_payload(1.01)


def test_embed_zero_payload_is_identity() -> None:
    pixels = _pixels()
    out = embed_lsb(pixels, 0.0, seed=1)
    assert np.array_equal(out, pixels)


def test_embed_is_deterministic_for_fixed_seed() -> None:
    pixels = _pixels()
    out1 = embed_lsb(pixels, 0.2, seed=42)
    out2 = embed_lsb(pixels, 0.2, seed=42)
    assert np.array_equal(out1, out2)


def test_embed_different_seed_changes_result() -> None:
    pixels = _pixels()
    out1 = embed_lsb(pixels, 0.2, seed=42)
    out2 = embed_lsb(pixels, 0.2, seed=43)
    assert not np.array_equal(out1, out2)


def test_embed_only_touches_lsb() -> None:
    pixels = _pixels()
    out = embed_lsb(pixels, 0.5, seed=7)
    # Upper 7 bits must be untouched everywhere; only the LSB may change.
    assert (pixels >> 1 == out >> 1).all()


def test_embed_touches_expected_pixel_count() -> None:
    pixels = _pixels(shape=(100, 100))
    out = embed_lsb(pixels, 0.3, seed=7)
    n_embed_positions = 3000  # round(0.3 * 10000)
    changed_or_candidate = (pixels != out).sum()
    # At most n_embed_positions pixels can differ (some embedded bits match
    # the original LSB by chance and cause no visible change).
    assert changed_or_candidate <= n_embed_positions


def test_embed_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError):
        embed_lsb(np.zeros((4, 4, 3), dtype=np.uint8), 0.1, seed=1)


def test_embed_rejects_non_uint8_input() -> None:
    with pytest.raises(ValueError):
        embed_lsb(np.zeros((4, 4), dtype=np.float32), 0.1, seed=1)


def test_sample_seed_is_stable_and_process_independent() -> None:
    s1 = sample_seed(42, "cover_0001", 0.1)
    s2 = sample_seed(42, "cover_0001", 0.1)
    assert s1 == s2
    assert isinstance(s1, int)


def test_sample_seed_varies_by_input() -> None:
    base = sample_seed(42, "cover_0001", 0.1)
    assert base != sample_seed(42, "cover_0002", 0.1)
    assert base != sample_seed(42, "cover_0001", 0.2)
    assert base != sample_seed(43, "cover_0001", 0.1)
