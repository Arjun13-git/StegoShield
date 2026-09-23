"""Controlled LSB message encoder (educational demonstration).

This is the literal message embedder for `POST /api/v1/encode`. It is
separate from, and does not modify, the Phase 1 payload SIMULATOR in
`app.ml.lsb` (which embeds random bits to build training data).

Format of the embedded stream (sequential, starting at the first sample):
    4 bytes  magic  b"SSLB"
    4 bytes  message length, big-endian unsigned
    N bytes  UTF-8 message
Each bit replaces the least-significant bit of one 8-bit sample, in row-major
order over the colour/grey samples. The alpha channel is never modified. The
result is always written as PNG (lossless; JPEG would destroy the payload) and
carries no metadata from the input. Nothing is written to disk.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from app.core.errors import ApiError
from app.services.image_service import ValidatedImage

MAGIC = b"SSLB"
HEADER_BYTES = 8


def _samples(image: Image.Image) -> tuple[np.ndarray, str]:
    arr = np.asarray(image, dtype=np.uint8)
    if image.mode == "RGBA":
        return np.ascontiguousarray(arr[..., :3]), "RGBA"
    return np.ascontiguousarray(arr), image.mode


def capacity_bytes(image: Image.Image) -> int:
    """Message bytes that fit (excluding the 8-byte header)."""
    arr, _ = _samples(image)
    return max(0, arr.size // 8 - HEADER_BYTES)


def encode_message(validated: ValidatedImage, message: str, max_message_bytes: int) -> tuple[bytes, int, int]:
    """Return (png_bytes, embedded_message_bytes, capacity_bytes)."""
    if not message:
        raise ApiError(400, "missing_message", "A non-empty message is required.")
    data = message.encode("utf-8")
    if len(data) > max_message_bytes:
        raise ApiError(400, "message_too_large", f"The message exceeds the limit of {max_message_bytes} bytes.")

    samples, mode = _samples(validated.image)
    capacity = max(0, samples.size // 8 - HEADER_BYTES)
    if len(data) > capacity:
        raise ApiError(400, "payload_exceeds_capacity", f"The message needs {len(data)} bytes but this image can hold {capacity} bytes.")

    stream = MAGIC + len(data).to_bytes(4, "big") + data
    bits = np.unpackbits(np.frombuffer(stream, dtype=np.uint8))
    flat = samples.reshape(-1).copy()
    flat[: bits.size] = (flat[: bits.size] & 0xFE) | bits
    embedded = flat.reshape(samples.shape)

    if mode == "RGBA":
        out = np.asarray(validated.image, dtype=np.uint8).copy()
        out[..., :3] = embedded
    else:
        out = embedded

    buf = BytesIO()
    Image.fromarray(out).save(buf, format="PNG")
    return buf.getvalue(), len(data), capacity


def decode_message(image: Image.Image) -> str | None:
    """Inverse of `encode_message`; returns None if no valid stream is present."""
    samples, _ = _samples(image)
    flat = samples.reshape(-1)
    if flat.size < HEADER_BYTES * 8:
        return None
    header = np.packbits(flat[: HEADER_BYTES * 8] & 1).tobytes()
    if header[:4] != MAGIC:
        return None
    length = int.from_bytes(header[4:8], "big")
    if (HEADER_BYTES + length) * 8 > flat.size:
        return None
    body = np.packbits(flat[HEADER_BYTES * 8 : (HEADER_BYTES + length) * 8] & 1).tobytes()
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError:
        return None
