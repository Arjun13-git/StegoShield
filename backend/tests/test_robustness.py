import numpy as np
import pytest

from app.ml.lsb import embed_lsb
from app.ml.robustness import (
    add_gaussian_noise,
    build_suite_v1,
    crop_box,
    jpeg_recompress,
    resize_scale,
)


def _gray(seed: int = 0, shape: tuple[int, int] = (64, 64)) -> np.ndarray:
    return np.random.default_rng(seed).integers(20, 236, size=shape, dtype=np.uint8)


def test_jpeg_preserves_shape_dtype_and_changes_pixels() -> None:
    px = _gray()
    out = jpeg_recompress(px, 75)
    assert out.shape == px.shape and out.dtype == np.uint8
    assert not np.array_equal(out, px)


def test_jpeg_rgb_stays_three_channel() -> None:
    px = np.random.default_rng(1).integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    assert jpeg_recompress(px, 90).shape == (32, 32, 3)


def test_jpeg_is_deterministic() -> None:
    px = _gray()
    assert np.array_equal(jpeg_recompress(px, 90), jpeg_recompress(px, 90))


def test_jpeg_rejects_invalid_quality() -> None:
    with pytest.raises(ValueError):
        jpeg_recompress(_gray(), 0)


def test_jpeg_spreads_lsb_perturbation_beyond_original_positions() -> None:
    """The premise of a JPEG robustness test: an LSB-only perturbation is
    no longer confined to the originally modified pixels after JPEG (the
    DCT quantization redistributes it), so the pixel-level difference
    between transformed cover and transformed stego covers more positions.
    """
    cover = _gray(seed=3, shape=(128, 128))
    stego = embed_lsb(cover, 0.4, seed=5)
    before = int((cover != stego).sum())
    j_cover, j_stego = jpeg_recompress(cover, 75), jpeg_recompress(stego, 75)
    after = int((j_cover != j_stego).sum())
    assert before > 0
    assert after > before


def test_resize_output_shape_and_no_mutation() -> None:
    px = _gray(shape=(100, 100))
    original = px.copy()
    out = resize_scale(px, 0.9)
    assert out.shape == (90, 90)
    assert np.array_equal(px, original)


def test_resize_rejects_nonpositive_scale() -> None:
    with pytest.raises(ValueError):
        resize_scale(_gray(), 0.0)


def test_noise_same_seed_same_output_different_seed_differs() -> None:
    px = _gray()
    assert np.array_equal(add_gaussian_noise(px, 2.0, 7), add_gaussian_noise(px, 2.0, 7))
    assert not np.array_equal(add_gaussian_noise(px, 2.0, 7), add_gaussian_noise(px, 2.0, 8))


def test_noise_zero_sigma_is_identity() -> None:
    px = _gray()
    assert np.array_equal(add_gaussian_noise(px, 0.0, 1), px)


def test_noise_clips_to_valid_range() -> None:
    px = np.full((32, 32), 255, dtype=np.uint8)
    out = add_gaussian_noise(px, 50.0, 1)
    assert out.dtype == np.uint8 and out.max() <= 255


def test_same_noise_field_applied_to_cover_and_stego() -> None:
    """Same seed -> identical noise realization for a cover and its stego,
    so (transformed - original) matches away from clipping boundaries.
    """
    cover = _gray(seed=2, shape=(64, 64))
    stego = embed_lsb(cover, 0.2, seed=9)
    noise_c = add_gaussian_noise(cover, 2.0, 11).astype(int) - cover.astype(int)
    noise_s = add_gaussian_noise(stego, 2.0, 11).astype(int) - stego.astype(int)
    # Pixels differ by at most 1 level between cover/stego, so rounding can
    # differ by at most 1 as well.
    assert np.abs(noise_c - noise_s).max() <= 2


def test_crop_matches_slice_and_validates_bounds() -> None:
    px = _gray(shape=(50, 60))
    out = crop_box(px, (5, 4, 25, 34))
    assert np.array_equal(out, px[4:34, 5:25])
    with pytest.raises(ValueError):
        crop_box(px, (0, 0, 61, 50))


def test_suite_v1_declares_expected_transforms_and_runs_on_512() -> None:
    suite = build_suite_v1()
    assert [t.name for t in suite] == [
        "jpeg_qf90", "jpeg_qf75", "resize_0.9x", "gaussian_noise_sigma2", "crop_480_offset13",
    ]
    px = _gray(shape=(512, 512))
    shapes = {t.name: t.apply(px, 1).shape for t in suite}
    assert shapes["jpeg_qf90"] == (512, 512)
    assert shapes["resize_0.9x"] == (461, 461)
    assert shapes["crop_480_offset13"] == (480, 480)
    assert all(t.describe()["params"] for t in suite)
