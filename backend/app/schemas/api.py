from typing import Literal
from pydantic import BaseModel, Field


class FeatureIndicator(BaseModel):
    name: str
    value: float
    direction: Literal["increases_stego_signal", "decreases_stego_signal", "informational"]


class AnalysisResponse(BaseModel):
    request_id: str
    verdict: Literal["likely_cover", "potential_steganography", "inconclusive"]
    stego_score: float = Field(ge=0, le=100)
    model: str
    model_version: str
    feature_count: int
    indicators: list[FeatureIndicator]
    limitations: list[str]
