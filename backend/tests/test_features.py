from PIL import Image

from app.ml.features import FEATURE_NAMES, extract_features


def test_feature_schema_is_stable() -> None:
    image = Image.new("RGB", (128, 128), color=(120, 90, 60))
    vector, names = extract_features(image)
    assert len(vector) == len(FEATURE_NAMES)
    assert tuple(names) == FEATURE_NAMES
