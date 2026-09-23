"""Orchestrates: validated image -> existing feature extractor -> frozen model
-> risk/explanation layer -> response schema.

No model is loaded or trained here; the ModelService (created once at
application start-up) is injected. Feature extraction is the unchanged
Phase 1 implementation.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from app.core.config import Settings
from app.core.errors import ApiError
from app.ml.features import FEATURE_NAMES, extract_features
from app.schemas.api import AnalysisResponse, ImageInfo
from app.services.explanation import EXPLANATION_METHOD, build_indicators
from app.services.image_service import ImageLimits, ValidatedImage, validate_image
from app.services.model_service import ModelService
from app.services.reference import ReferenceData, risk_from_percentile

logger = logging.getLogger("stegoshield.analysis")

SCORE_NOTE_LIMITATIONS = [
    "The score is an uncalibrated model output, not a probability that the image contains hidden data.",
    "The detector was built for pixel-domain LSB replacement on grayscale BOSSBase images; in-domain it is a weak detector "
    "and it was not found to detect the JPEG-domain steganography algorithms evaluated in Phase 1B (JMiPOD, J-UNIWARD, UERD).",
    "Its signal is removed by JPEG re-compression and by mild Gaussian noise (Phase 1B robustness evaluation).",
    "A low result is not evidence that an image is free of hidden data, and a high result is not evidence of malicious intent.",
]


def image_warnings(image: ValidatedImage) -> list[str]:
    warnings = []
    if image.format == "JPEG":
        warnings.append(
            "JPEG input: lossy compression rewrites pixel values and removes most pixel-domain LSB evidence, "
            "so a low result on a JPEG image is not informative."
        )
    if image.mode in ("RGB", "RGBA"):
        warnings.append("Colour image: the detector was trained on grayscale-derived images (grey replicated to RGB); results on colour images fall outside that setting.")
    if image.alpha_discarded:
        warnings.append("The alpha channel was ignored; only the colour channels were analysed.")
    if (image.width, image.height) != (512, 512):
        warnings.append("The image is not 512x512, the size used in the research evaluations.")
    return warnings


def _risk_context(risk: str, reference: ReferenceData | None) -> str:
    if reference is None:
        return "No reference distribution is available for this model, so the score cannot be placed on a scale."
    op = reference.operating_points
    e = op["elevated"]["flag_rate"]
    if risk == "low":
        return (f"Below the 'elevated' level (top 10% of reference clean images). In Phase 1 held-out testing that level flagged "
                f"{e['test_clean'] * 100:.1f}% of clean images and {e['test_lsb_stego'] * 100:.1f}% of 0.10 bits-per-pixel LSB stego images.")
    level = op[risk]["flag_rate"]
    pct = "10%" if risk == "elevated" else "1%"
    return (f"At or above the {risk} level (top {pct} of reference clean images). In Phase 1 held-out testing this level flagged "
            f"{level['test_clean'] * 100:.1f}% of clean images and {level['test_lsb_stego'] * 100:.1f}% of 0.10 bits-per-pixel LSB stego images.")


class AnalysisService:
    def __init__(self, settings: Settings, model: ModelService, reference: ReferenceData | None) -> None:
        self._limits = ImageLimits.from_settings(settings)
        self._model = model
        self._reference = reference

    def analyze(self, payload: bytes, request_id: str) -> AnalysisResponse:
        started = time.perf_counter()
        validated = validate_image(payload, self._limits)

        vector, names = extract_features(validated.to_rgb())
        if tuple(names) != FEATURE_NAMES or not np.isfinite(vector).all():
            raise ApiError(500, "analysis_failed", "Analysis failed.")

        score = self._model.predict_score(vector)  # raises ModelUnavailableError if not ready
        meta = self._model.metadata

        reference = self._reference
        if reference is None:
            percentile, risk, verdict = None, "not_assessed", "inconclusive"
        else:
            p = reference.cover_percentile(score)
            percentile, risk = round(p * 100, 2), risk_from_percentile(p)
            verdict = "likely_cover" if risk == "low" else "potential_steganography"

        response = AnalysisResponse(
            request_id=request_id,
            verdict=verdict,
            risk_level=risk,
            stego_score=round(score * 100, 2),
            cover_reference_percentile=percentile,
            risk_context=_risk_context(risk, reference),
            model=meta.name,
            model_version=meta.version,
            feature_schema_version=meta.feature_schema_version,
            feature_count=meta.feature_count,
            indicators=build_indicators(vector, list(names), self._model.feature_importances, reference),
            explanation_method=EXPLANATION_METHOD,
            image=ImageInfo(
                format=validated.format, width=validated.width, height=validated.height, original_mode=validated.mode,
                size_bytes=validated.size_bytes, alpha_discarded=validated.alpha_discarded,
                analyzed_as="RGB (grey replicated)" if validated.mode == "L" else "RGB",
            ),
            warnings=image_warnings(validated),
            limitations=list(SCORE_NOTE_LIMITATIONS),
        )
        logger.info(
            "request_id=%s analysis verdict=%s risk=%s fmt=%s size=%dx%d ms=%.0f",
            request_id, verdict, risk, validated.format, validated.width, validated.height,
            (time.perf_counter() - started) * 1000,
        )
        return response
