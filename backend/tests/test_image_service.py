from io import BytesIO

from PIL import Image
import pytest

from app.services.image_service import validate_upload


def _png_bytes() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (128, 128), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def test_valid_image() -> None:
    image = validate_upload(_png_bytes(), "test.png")
    assert image.mode == "RGB"


def test_empty_upload_rejected() -> None:
    with pytest.raises(ValueError):
        validate_upload(b"", "empty.png")
