from PIL import Image

from app.core.config import settings
from app.ml.features import extract_features
from app.ml.model import ModelBundle
from app.schemas.api import AnalysisResponse, FeatureIndicator

_model = ModelBundle(settings.model_path)
_model.load()


def analyze_image(image: Image.Image, request_id: str) -> AnalysisResponse:
    vector, names = extract_features(image)

    if not _model.loaded:
        return AnalysisResponse(
            request_id=request_id,
            verdict="inconclusive",
            stego_score=0.0,
            model="not-loaded",
            model_version="none",
            feature_count=len(vector),
            indicators=[FeatureIndicator(name=n, value=float(v), direction="informational") for n, v in zip(names[:5], vector[:5])],
            limitations=["No trained model artifact is installed yet.", "This response is a skeleton contract, not a production prediction."],
        )

    score = _model.predict_score(vector)
    metadata = _model.metadata()
    verdict = "potential_steganography" if score >= 0.5 else "likely_cover"
    indicators = [
        FeatureIndicator(name=name, value=float(value), direction="informational")
        for name, value in zip(names[:8], vector[:8])
    ]
    return AnalysisResponse(
        request_id=request_id,
        verdict=verdict,
        stego_score=round(score * 100, 2),
        model=metadata["name"],
        model_version=metadata["version"],
        feature_count=len(vector),
        indicators=indicators,
        limitations=["Model scope is limited to the embedding/data distribution used during training."],
    )
