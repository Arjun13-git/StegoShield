from typing import Any, Literal

from pydantic import BaseModel, Field

Direction = Literal["increases_stego_signal", "decreases_stego_signal", "informational"]
Verdict = Literal["likely_cover", "potential_steganography", "inconclusive"]
RiskLevel = Literal["low", "elevated", "high", "not_assessed"]


class FeatureIndicator(BaseModel):
    name: str
    label: str
    value: float
    direction: Direction
    deviation_from_reference_cover: float | None = Field(
        default=None, description="Deviation from the reference clean-image mean, in reference standard deviations."
    )
    model_importance: float | None = Field(default=None, description="Global Random Forest feature importance.")


class ImageInfo(BaseModel):
    format: Literal["PNG", "JPEG"]
    width: int
    height: int
    original_mode: str
    size_bytes: int
    alpha_discarded: bool
    analyzed_as: str


class AnalysisResponse(BaseModel):
    request_id: str
    verdict: Verdict
    risk_level: RiskLevel
    stego_score: float = Field(
        ge=0, le=100,
        description="Uncalibrated model score (model output x 100). NOT a probability that the image contains hidden data.",
    )
    score_kind: Literal["uncalibrated_model_score"] = "uncalibrated_model_score"
    cover_reference_percentile: float | None = Field(
        default=None, ge=0, le=100,
        description="Percent of reference clean images (Phase 1 validation split) with a score at or below this one.",
    )
    risk_context: str
    model: str
    model_version: str
    feature_schema_version: str
    feature_count: int
    indicators: list[FeatureIndicator]
    explanation_method: str
    image: ImageInfo
    warnings: list[str]
    limitations: list[str]


class ModelInfoResponse(BaseModel):
    name: str
    version: str
    feature_schema: str
    scope: str
    feature_count: int
    artifact_sha256: str
    training_dataset_id: str | None
    random_seed: int | None
    reference_performance: dict[str, Any] | None = Field(
        description="Phase 1 held-out in-domain test metrics stored in the artifact (BOSSBase, controlled LSB)."
    )
    score_interpretation: str
    risk_levels: dict[str, Any] | None
    limitations: list[str]


class ErrorResponse(BaseModel):
    detail: str
    code: str
    request_id: str
