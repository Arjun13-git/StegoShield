from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.api import AnalysisResponse
from app.services.analysis_service import analyze_image
from app.services.image_service import validate_upload

router = APIRouter(tags=["analysis"])


@router.get("/health")
def api_health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/model-info")
def model_info() -> dict[str, object]:
    return {
        "name": "RandomForest-StegoShield",
        "version": "placeholder",
        "feature_schema": "v1",
        "scope": "LSB-focused image steganalysis",
    }


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...)) -> AnalysisResponse:
    request_id = str(uuid4())
    payload = await file.read()
    try:
        image = validate_upload(payload, file.filename or "upload")
        return analyze_image(image, request_id=request_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/encode")
async def encode(file: UploadFile = File(...), message: str = "") -> dict[str, str]:
    # Implementation belongs in the service layer; kept as a contract in the skeleton.
    if not message:
        raise HTTPException(status_code=400, detail="message is required")
    payload = await file.read()
    validate_upload(payload, file.filename or "upload")
    return {"status": "not_implemented", "detail": "LSB encoder contract is ready for implementation."}
