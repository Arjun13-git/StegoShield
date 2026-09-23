"""Image transformations for the Phase 1B robustness stress test.

Every function is a pure, deterministic map from a uint8 pixel array to a
new uint8 pixel array (inputs are never mutated). Randomness (Gaussian
noise) is driven entirely by an explicit seed so the SAME transformation
can be applied to a cover and its stego derivative.

The suite parameters below are declared here, in code, BEFORE any
robustness run, and are not tuned against results. They follow the list in
System-Design/03-data-pipeline.md section 7 (JPEG QF90, JPEG QF75, mild
resize, mild Gaussian noise, small crop).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from PIL import Image


def jpeg_recompress(pixels: np.ndarray, quality: int) -> np.ndarray:
    """Encode to JPEG at `quality` with Pillow's encoder (libjpeg, default
    settings, no optimize) and decode back to a pixel array. Grayscale (2D)
    input stays single-channel, RGB (3D) input stays 3-channel.
    """
    if not 1 <= quality <= 100:
        raise ValueError(f"quality must be within [1, 100], got {quality}")
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    with Image.open(buf) as img:
        img.load()
        return np.asarray(img, dtype=np.uint8).copy()


def resize_scale(pixels: np.ndarray, scale: float) -> np.ndarray:
    """Resize both dimensions by `scale` with bicubic resampling (Pillow).
    The image is NOT resized back afterwards: the detector sees the smaller
    image, as it would if a platform downscaled an upload.
    """
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    height, width = pixels.shape[:2]
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = Image.fromarray(pixels).resize(new_size, resample=Image.Resampling.BICUBIC)
    return np.asarray(resized, dtype=np.uint8).copy()


def add_gaussian_noise(pixels: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """Add i.i.d. zero-mean Gaussian noise (std `sigma`, in 0-255 pixel
    units), round to the nearest integer, clip to [0, 255]. Same (sigma,
    seed, shape) always yields the identical noise field.
    """
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=pixels.shape)
    noisy = np.rint(pixels.astype(np.float64) + noise)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def crop_box(pixels: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Crop to (left, top, right, bottom) in pixel coordinates."""
    left, top, right, bottom = box
    height, width = pixels.shape[:2]
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError(f"crop box {box} is outside an image of size {width}x{height}")
    return pixels[top:bottom, left:right].copy()


@dataclass(frozen=True)
class RobustnessTransform:
    name: str
    kind: str
    params: dict[str, Any]
    apply: Callable[[np.ndarray, int], np.ndarray] = field(repr=False, compare=False)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "params": self.params}


def build_suite_v1() -> list[RobustnessTransform]:
    """The pre-declared Phase 1B robustness suite (version 1).

    `apply(pixels, seed)` -- the seed is only consumed by the noise
    transform; deterministic transforms ignore it.
    """
    return [
        RobustnessTransform(
            "jpeg_qf90", "jpeg", {"quality": 90, "encoder": "Pillow/libjpeg default"},
            lambda px, seed: jpeg_recompress(px, 90),
        ),
        RobustnessTransform(
            "jpeg_qf75", "jpeg", {"quality": 75, "encoder": "Pillow/libjpeg default"},
            lambda px, seed: jpeg_recompress(px, 75),
        ),
        RobustnessTransform(
            "resize_0.9x", "resize", {"scale": 0.9, "resample": "BICUBIC", "resized_back": False},
            lambda px, seed: resize_scale(px, 0.9),
        ),
        RobustnessTransform(
            "gaussian_noise_sigma2", "noise", {"sigma": 2.0, "unit": "8-bit pixel levels", "clip": "[0,255]"},
            lambda px, seed: add_gaussian_noise(px, 2.0, seed),
        ),
        RobustnessTransform(
            "crop_480_offset13", "crop", {"box_left_top_right_bottom": [13, 13, 493, 493]},
            lambda px, seed: crop_box(px, (13, 13, 493, 493)),
        ),
    ]
