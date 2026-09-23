import io

import numpy as np
import pytest
from PIL import Image

from app.services.image_service import ImageLimits, ImageValidationError, validate_image, validate_upload
from conftest import jpeg_bytes, noise, png_bytes

LIMITS = ImageLimits(max_upload_bytes=500_000, min_image_side=64, max_image_side=1000, max_image_pixels=200 * 200)


def code_of(payload: bytes, limits: ImageLimits = LIMITS) -> str:
    with pytest.raises(ImageValidationError) as exc:
        validate_image(payload, limits)
    return exc.value.code


def test_valid_grayscale_rgb_rgba_png_and_rgb_jpeg() -> None:
    assert validate_image(png_bytes(noise((100, 100))), LIMITS).mode == "L"
    assert validate_image(png_bytes(noise((100, 100, 3))), LIMITS).mode == "RGB"
    rgba = validate_image(png_bytes(noise((100, 100, 4))), LIMITS)
    assert rgba.mode == "RGBA" and rgba.alpha_discarded and rgba.to_rgb().mode == "RGB"
    j = validate_image(jpeg_bytes(noise((100, 100, 3))), LIMITS)
    assert j.format == "JPEG" and j.width == j.height == 100


def test_grayscale_is_replicated_to_rgb_like_phase1() -> None:
    gray = noise((80, 80), 3)
    rgb = np.asarray(validate_image(png_bytes(gray), LIMITS).to_rgb())
    assert all(np.array_equal(rgb[..., c], gray) for c in range(3))


def test_extension_and_mime_do_not_matter_only_content() -> None:
    # a PNG is a PNG regardless of what it is called; validate_upload ignores the filename
    assert validate_upload(png_bytes(noise((100, 100))), "evil.exe", LIMITS).size == (100, 100)


@pytest.mark.parametrize("payload, code", [
    (b"", "empty_file"),
    (b"hello world", "invalid_image"),
    (b"GIF89a....", "unsupported_format"),
    (b"BM" + b"\x00" * 60, "unsupported_format"),
    (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "unsupported_format"),
    (b"\x89PNG\r\n\x1a\n" + b"garbage", "invalid_image"),
    (b"\xff\xd8\xff" + b"garbage", "invalid_image"),
])
def test_bad_content(payload: bytes, code: str) -> None:
    assert code_of(payload) == code


def test_size_dimension_mode_and_frame_limits() -> None:
    assert code_of(png_bytes(noise((100, 100))) + b"\x00" * 600_000) == "file_too_large"
    assert code_of(png_bytes(noise((32, 100)))) == "invalid_dimensions"   # a side below the minimum
    assert code_of(png_bytes(noise((300, 300)))) == "invalid_dimensions"  # over the pixel cap
    tall = ImageLimits(500_000, 64, 1000, 10_000_000)
    assert code_of(png_bytes(noise((1200, 100))), tall) == "invalid_dimensions"  # over the side cap
    buf = io.BytesIO(); Image.fromarray(noise((100, 100))).convert("P").save(buf, format="PNG")
    assert code_of(buf.getvalue()) == "unsupported_color_mode"
    buf = io.BytesIO(); Image.fromarray(noise((100, 100, 3))).convert("LA").save(buf, format="PNG")
    assert code_of(buf.getvalue()) == "unsupported_color_mode"
    frames = [Image.fromarray(noise((100, 100), s)) for s in range(3)]
    buf = io.BytesIO(); frames[0].save(buf, format="PNG", save_all=True, append_images=frames[1:])  # APNG
    assert code_of(buf.getvalue()) == "invalid_image"


def test_dimensions_are_checked_from_the_header_before_decoding() -> None:
    """A tiny PNG that CLAIMS a huge size is rejected without allocating pixels."""
    import struct, zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", 60000, 60000, 8, 0, 0, 0, 0)
    bomb = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(b"\x00" * 100)) + chunk(b"IEND", b"")
    assert len(bomb) < 200
    assert code_of(bomb) == "invalid_dimensions"


def test_error_is_also_a_value_error_for_backwards_compatibility() -> None:
    with pytest.raises(ValueError):
        validate_upload(b"", "x.png", LIMITS)
