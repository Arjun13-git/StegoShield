from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str


class ReadyResponse(BaseModel):
    status: str  # "ready" | "not_ready"
    model_loaded: bool
    reference_available: bool
    reason: str | None = None  # short machine tag only (e.g. "artifact_missing"); never a path
