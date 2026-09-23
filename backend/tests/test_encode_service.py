import io

import numpy as np
import pytest
from PIL import Image

from app.core.errors import ApiError
from app.services.encode_service import HEADER_BYTES, capacity_bytes, decode_message, encode_message
from app.services.image_service import ImageLimits, validate_image
from conftest import jpeg_bytes, noise, png_bytes

LIMITS = ImageLimits(500_000, 8, 2000, 2000 * 2000)


def encode(arr: np.ndarray, message: str, max_bytes: int = 10_000, jpeg: bool = False):
    v = validate_image(jpeg_bytes(arr) if jpeg else png_bytes(arr), LIMITS)
    png, n, cap = encode_message(v, message, max_bytes)
    return v, Image.open(io.BytesIO(png)), n, cap


@pytest.mark.parametrize("shape, mode", [((64, 64), "L"), ((64, 64, 3), "RGB"), ((64, 64, 4), "RGBA")])
def test_round_trip_for_grey_rgb_rgba(shape, mode) -> None:
    _, out, n, cap = encode(noise(shape, 1), "attack at dawn ✓")
    assert out.format == "PNG" and out.mode == mode
    assert decode_message(out) == "attack at dawn ✓" and n == len("attack at dawn ✓".encode())


def test_only_least_significant_bits_change_and_alpha_is_untouched() -> None:
    cover = noise((64, 64, 4), 2)
    _, out, _, _ = encode(cover, "x" * 100)
    stego = np.asarray(out)
    assert np.array_equal(stego[..., 3], cover[..., 3])  # alpha channel never modified
    assert ((stego[..., :3] >> 1) == (cover[..., :3] >> 1)).all()  # upper 7 bits identical


def test_is_deterministic_and_does_not_mutate_input() -> None:
    cover = noise((64, 64), 3)
    v = validate_image(png_bytes(cover), LIMITS)
    before = np.asarray(v.image).copy()
    a = encode_message(v, "same", 100)[0]
    b = encode_message(v, "same", 100)[0]
    assert a == b and np.array_equal(np.asarray(v.image), before)


def test_capacity_and_limits() -> None:
    v = validate_image(png_bytes(noise((64, 64))), LIMITS)
    assert capacity_bytes(v.image) == 64 * 64 // 8 - HEADER_BYTES
    cap = capacity_bytes(v.image)
    encode_message(v, "a" * cap, 10_000)  # exactly full is allowed
    with pytest.raises(ApiError) as e:
        encode_message(v, "a" * (cap + 1), 10_000)
    assert e.value.code == "payload_exceeds_capacity"
    with pytest.raises(ApiError) as e:
        encode_message(v, "abc", 2)
    assert e.value.code == "message_too_large"
    with pytest.raises(ApiError) as e:
        encode_message(v, "", 100)
    assert e.value.code == "missing_message"


def test_jpeg_cover_yields_a_lossless_png_that_still_decodes() -> None:
    _, out, _, _ = encode(noise((64, 64, 3), 4), "from a jpeg cover", jpeg=True)
    assert out.format == "PNG" and decode_message(out) == "from a jpeg cover"


def test_decode_returns_none_when_nothing_is_embedded() -> None:
    assert decode_message(Image.fromarray(noise((64, 64), 5))) is None
