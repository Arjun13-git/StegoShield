from io import BytesIO

from PIL import Image, UnidentifiedImageError

from app.core.config import settings


ALLOWED_FORMATS = {"PNG", "JPEG"}


def validate_upload(payload: bytes, filename: str) -> Image.Image:
    if not payload:
        raise ValueError("Empty upload")
    if len(payload) > settings.max_upload_bytes:
        raise ValueError("Image exceeds the configured upload limit")

    try:
        image = Image.open(BytesIO(payload))
        image.verify()
        image = Image.open(BytesIO(payload))
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Uploaded file is not a valid image") from exc

    if image.format not in ALLOWED_FORMATS:
        raise ValueError("Only PNG and JPEG images are supported in the MVP")

    width, height = image.size
    if min(width, height) < settings.min_image_side:
        raise ValueError("Image is too small for reliable analysis")
    if max(width, height) > settings.max_image_side:
        raise ValueError("Image dimensions exceed the configured limit")

    return image.convert("RGB")
