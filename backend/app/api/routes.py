"""HTTP layer only: parse the request, call a service, shape the response.

No model training/loading, no image parsing and no file writing happens here.
Uploaded filenames and Content-Types are never used.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response

from app.api.deps import get_services
from app.core.errors import ApiError, request_id_of
from app.schemas.api import AnalysisResponse, ErrorResponse, ModelInfoResponse
from app.schemas.health import HealthResponse, ReadyResponse
from app.services.analysis_service import SCORE_NOTE_LIMITATIONS
from app.services.encode_service import encode_message
from app.services.image_service import ImageLimits, validate_image
from app.services.model_service import ModelUnavailableError

logger = logging.getLogger("stegoshield.api")

SERVICE_NAME = "stegoshield-api"
_ERRORS = {
    400: {"model": ErrorResponse, "description": "Invalid image, unsupported format/mode/dimensions, or bad message"},
    413: {"model": ErrorResponse, "description": "Upload too large"},
    422: {"model": ErrorResponse, "description": "Malformed request fields"},
    500: {"model": ErrorResponse, "description": "Internal failure (no details exposed)"},
    503: {"model": ErrorResponse, "description": "Detection model unavailable"},
}

# Root-level operational endpoints (liveness / readiness)
system_router = APIRouter(tags=["system"])
# Versioned API
router = APIRouter(tags=["api"])

_MODEL_UNAVAILABLE = ApiError(503, "model_unavailable", "The detection model is not available.")


def _model_unavailable() -> ApiError:
    return ApiError(_MODEL_UNAVAILABLE.status_code, _MODEL_UNAVAILABLE.code, _MODEL_UNAVAILABLE.detail)


@system_router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness: the process is up. Says nothing about whether inference works."""
    return HealthResponse(status="ok", service=SERVICE_NAME)


@system_router.get("/ready", response_model=ReadyResponse, responses={503: {"model": ReadyResponse}})
def ready(request: Request):
    """Readiness: the frozen model is verified and loaded, so inference can run."""
    svc = get_services(request)
    ok = svc.model.ready
    body = ReadyResponse(
        status="ready" if ok else "not_ready",
        model_loaded=ok,
        reference_available=svc.reference is not None,
        reason=None if ok else svc.model.failure_reason,
    )
    return JSONResponse(status_code=200 if ok else 503, content=body.model_dump())


@router.get("/health", response_model=HealthResponse)
def api_health() -> HealthResponse:
    return HealthResponse(status="ok", service=SERVICE_NAME)


@router.get("/model-info", response_model=ModelInfoResponse, responses={503: _ERRORS[503]})
def model_info(request: Request) -> ModelInfoResponse:
    svc = get_services(request)
    if not svc.model.ready:
        raise _model_unavailable()
    meta = svc.model.metadata
    ref = svc.reference
    risk_levels = None
    if ref is not None:
        risk_levels = {
            "definition": "Percentiles of reference clean-image scores (Phase 1 validation split); operating points, not probabilities.",
            "reference_stego": ref.payload_note,
            "levels": {name: {"cover_percentile_at_or_above": op["cover_percentile_threshold"], "flag_rate_held_out_test": {
                "clean": op["flag_rate"]["test_clean"], "lsb_stego": op["flag_rate"]["test_lsb_stego"]}}
                for name, op in ref.operating_points.items()},
        }
    return ModelInfoResponse(
        name=meta.name, version=meta.version, feature_schema=meta.feature_schema_version,
        scope="LSB-focused image steganalysis", feature_count=meta.feature_count, artifact_sha256=meta.artifact_sha256,
        training_dataset_id=meta.training_dataset_id, random_seed=meta.random_seed,
        reference_performance=meta.reference_performance,
        score_interpretation="stego_score is the model output x 100. It is uncalibrated and is not a probability.",
        risk_levels=risk_levels, limitations=list(SCORE_NOTE_LIMITATIONS),
    )


@router.post("/analyze", response_model=AnalysisResponse, responses=_ERRORS)
async def analyze(request: Request, file: UploadFile = File(...)) -> AnalysisResponse:
    svc = get_services(request)
    if not svc.model.ready:
        raise _model_unavailable()
    request_id = request_id_of(request)
    # Bounded read; the body-size middleware has already capped the whole request.
    data = await file.read(svc.settings.max_upload_bytes + 1)
    async with svc.concurrency:
        try:
            return await run_in_threadpool(svc.analysis.analyze, data, request_id)
        except ApiError:
            raise
        except ModelUnavailableError:
            raise _model_unavailable() from None
        except Exception:  # noqa: BLE001 - details go to the log, never to the client
            logger.exception("request_id=%s analysis failed", request_id)
            raise ApiError(500, "analysis_failed", "Analysis failed.") from None


@router.post("/encode", responses={200: {"content": {"image/png": {}}, "description": "PNG with the message embedded"}, **_ERRORS})
async def encode(request: Request, file: UploadFile = File(...), message: str = Form("")) -> Response:
    """Controlled LSB encoder (demonstration). Returns the stego image as PNG.

    An absent or empty `message` is a 400 `missing_message` (per the API design), enforced in the service.
    """
    svc = get_services(request)
    data = await file.read(svc.settings.max_upload_bytes + 1)
    limits = ImageLimits.from_settings(svc.settings)

    def work() -> tuple[bytes, int, int]:
        validated = validate_image(data, limits)
        return encode_message(validated, message, svc.settings.max_message_bytes)

    async with svc.concurrency:
        try:
            png, embedded, capacity = await run_in_threadpool(work)
        except ApiError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("request_id=%s encoding failed", request_id_of(request))
            raise ApiError(500, "encoding_failed", "Encoding failed.") from None
    logger.info("request_id=%s encode bytes=%d capacity=%d", request_id_of(request), embedded, capacity)
    return Response(
        content=png,
        media_type="image/png",
        headers={
            "Content-Disposition": 'attachment; filename="stegoshield-encoded.png"',
            "X-Embedded-Bytes": str(embedded),
            "X-Capacity-Bytes": str(capacity),
        },
    )
