"""Controlled LSB payload simulator for dataset generation.

This module builds reproducible cover/stego training pairs for the
steganalysis pipeline. It is NOT the user-facing message encoder that will
back the `/api/v1/encode` endpoint (that literal encode/decode feature is
Phase 2 scope); it exists solely to generate labelled statistical training
data for Phase 1.

Payload definition
-------------------
`payload` is the embedding rate in bits-per-pixel for a single-LSB scheme,
expressed as a fraction in [0, 1]. A payload of 0.10 means 10% of the
image's pixels are selected (without replacement, via a seeded RNG) and
their least-significant bit is overwritten with an independently drawn
pseudorandom bit. This is the standard way to simulate an embedded,
statistically-random (e.g. encrypted or compressed) payload without needing
a literal message: because the injected bit matches the original LSB about
half the time, roughly `payload / 2` of all pixels actually change value.
This definition matches the payload levels used in published BOSSBase
steganalysis benchmarks (0.01-0.40 bits per pixel).
"""

from __future__ import annotations

import hashlib

import numpy as np

DEFAULT_PAYLOAD_LEVELS: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20, 0.40)


def sample_seed(base_seed: int, source_id: str, payload: float) -> int:
    """Derive a stable per-sample seed from (base_seed, source_id, payload).

    Uses SHA-256 rather than Python's built-in hash(), which is salted per
    process (PYTHONHASHSEED) and would silently break reproducibility across
    runs and machines.
    """
    key = f"{base_seed}:{source_id}:{payload:.6f}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def lsb_capacity_bits(shape: tuple[int, int]) -> int:
    """Total number of pixels (= max embeddable bits at payload=1.0)."""
    height, width = shape
    return int(height * width)


def validate_payload(payload: float) -> None:
    if not (0.0 <= payload <= 1.0):
        raise ValueError(f"payload must be within [0, 1], got {payload}")


def embed_lsb(pixels: np.ndarray, payload: float, seed: int) -> np.ndarray:
    """Return a copy of a single-channel uint8 array with `payload` embedded.

    `payload` fraction of pixels (chosen via a seeded RNG, without
    replacement) have their LSB overwritten with an independently drawn
    pseudorandom bit. Deterministic for a fixed (pixels, payload, seed).
    """
    if pixels.ndim != 2:
        raise ValueError("embed_lsb expects a single-channel (2D) pixel array")
    if pixels.dtype != np.uint8:
        raise ValueError("embed_lsb expects a uint8 pixel array")
    validate_payload(payload)

    flat = pixels.reshape(-1).copy()
    n_pixels = flat.size
    n_embed = int(round(payload * n_pixels))
    if n_embed == 0:
        return flat.reshape(pixels.shape)

    rng = np.random.default_rng(seed)
    positions = rng.choice(n_pixels, size=n_embed, replace=False)
    bits = rng.integers(0, 2, size=n_embed, dtype=np.uint8)
    flat[positions] = (flat[positions] & 0xFE) | bits
    return flat.reshape(pixels.shape)
