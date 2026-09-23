"""Untrusted-image validation.

Order of checks (cheapest and safest first):
  1. size limits on the raw bytes;
  2. magic-byte sniffing (the filename and Content-Type are never trusted);
  3. Pillow is restricted to the PNG/JPEG plugins via `formats=`, so exotic
     decoders (TIFF, EPS, PDF, ...) are never invoked on attacker input;
  4. dimensions, pixel count, colour mode and frame count are checked from the
     HEADER, before any pixel data is decoded (decompression-bomb defence);
  5. only then is the image fully decoded, inside a guarded block.

Nothing is written to disk and no user-supplied name is used anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

from PIL import Image

from app.core.config import Settings
from app.core.errors import ApiError

ALLOWED_FORMATS = ("PNG", "JPEG")
ALLOWED_MODES = ("L", "RGB", "RGBA")

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"
_OTHER_IMAGE_MAGIC = (b"GIF87a", b"GIF89a", b"BM", b"II*\x00", b"MM\x00*", b"%PDF", b"\x00\x00\x01\x00", b"8BPS")


class ImageValidationError(ApiError, ValueError):
    """Client-safe validation failure (also a ValueError for backwards compatibility)."""


def _fail(status: int, code: str, detail: str) -> ImageValidationError:
    return ImageValidationError(status, code, detail)


@dataclass(frozen=True)
class ImageLimits:
    max_upload_bytes: int
    min_image_side: int
    max_image_side: int
    max_image_pixels: int

    @classmethod
    def from_settings(cls, settings: Settings) -> "ImageLimits":
        return cls(settings.max_upload_bytes, settings.min_image_side, settings.max_image_side, settings.max_image_pixels)


@dataclass(frozen=True)
class ValidatedImage:
    image: Image.Image  # fully decoded, in its ORIGINAL mode (L, RGB or RGBA)
    format: str  # "PNG" or "JPEG"
    width: int
    height: int
    mode: str
    size_bytes: int

    @property
    def alpha_discarded(self) -> bool:
        return self.mode == "RGBA"

    def to_rgb(self) -> Image.Image:
        """The representation the frozen detector is fed.

        L -> RGB replicates the grey channel, exactly as in Phase 1
        (`Image.fromarray(gray).convert("RGB")`). RGBA -> RGB DROPS the alpha
        channel (it is never treated as a colour channel); this is reported to
        the caller via `alpha_discarded`.
        """
        return self.image.convert("RGB")


def _sniff_format(payload: bytes) -> str:
    if payload.startswith(_PNG_MAGIC):
        return "PNG"
    if payload.startswith(_JPEG_MAGIC):
        return "JPEG"
    if payload.startswith(_OTHER_IMAGE_MAGIC) or (payload[:4] == b"RIFF" and payload[8:12] == b"WEBP"):
        raise _fail(400, "unsupported_format", "Only PNG and JPEG images are supported.")
    raise _fail(400, "invalid_image", "The uploaded file is not a valid PNG or JPEG image.")


def validate_image(payload: bytes, limits: ImageLimits) -> ValidatedImage:
    if not payload:
        raise _fail(400, "empty_file", "The uploaded file is empty.")
    if len(payload) > limits.max_upload_bytes:
        raise _fail(413, "file_too_large", "The uploaded file exceeds the configured size limit.")

    fmt = _sniff_format(payload)

    try:
        image = Image.open(BytesIO(payload), formats=[fmt])
    except Image.DecompressionBombError as exc:
        raise _fail(400, "invalid_dimensions", "Image dimensions exceed the allowed maximum.") from exc
    except Exception as exc:  # noqa: BLE001 - any parser failure on untrusted input is "invalid image"
        raise _fail(400, "invalid_image", "The uploaded file is not a valid PNG or JPEG image.") from exc

    width, height = image.size
    if min(width, height) < limits.min_image_side:
        raise _fail(400, "invalid_dimensions", f"Image is too small; each side must be at least {limits.min_image_side} px.")
    if max(width, height) > limits.max_image_side or width * height > limits.max_image_pixels:
        raise _fail(
            400,
            "invalid_dimensions",
            f"Image is too large; at most {limits.max_image_pixels} pixels and {limits.max_image_side} px per side are allowed.",
        )
    if image.mode not in ALLOWED_MODES:
        raise _fail(400, "unsupported_color_mode", "Only grayscale, RGB and RGBA images are supported.")
    if getattr(image, "n_frames", 1) > 1:
        raise _fail(400, "invalid_image", "Animated or multi-frame images are not supported.")

    try:
        image.load()  # full decode; truncated/corrupt data raises here
    except Exception as exc:  # noqa: BLE001
        raise _fail(400, "invalid_image", "The uploaded file is not a valid PNG or JPEG image.") from exc

    return ValidatedImage(image=image, format=fmt, width=width, height=height, mode=image.mode, size_bytes=len(payload))


def validate_upload(payload: bytes, filename: str = "upload", limits: ImageLimits | None = None) -> Image.Image:
    """Backwards-compatible helper returning the RGB image used by the detector.

    `filename` is accepted for compatibility and deliberately ignored.
    """
    limits = limits or ImageLimits.from_settings(Settings())
    return validate_image(payload, limits).to_rgb()
