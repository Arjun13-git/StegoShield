from __future__ import annotations

import asyncio
from dataclasses import dataclass

from fastapi import Request

from app.core.config import Settings
from app.services.analysis_service import AnalysisService
from app.services.model_service import ModelService
from app.services.reference import ReferenceData


@dataclass
class AppServices:
    """Objects created ONCE at start-up and shared by all requests."""

    settings: Settings
    model: ModelService
    reference: ReferenceData | None
    analysis: AnalysisService
    concurrency: asyncio.Semaphore


def get_services(request: Request) -> AppServices:
    return request.app.state.services
