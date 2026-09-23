"""Consistent, non-leaky API errors.

Every error response has the shape (superset of the design's contract):

    {"detail": "<safe human-readable message>", "code": "<machine code>", "request_id": "<uuid>"}

Messages are fixed strings chosen here; exception text, filesystem paths,
stack traces and environment details never reach the client. Details go to
the server log only.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("stegoshield.api")


class ApiError(Exception):
    """A deliberate, client-safe error."""

    def __init__(self, status_code: int, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.code = code
        self.detail = detail


def request_id_of(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def error_body(detail: str, code: str, request_id: str) -> dict[str, str]:
    return {"detail": detail, "code": code, "request_id": request_id}


def error_response(request: Request, status_code: int, code: str, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=error_body(detail, code, request_id_of(request)))


_HTTP_CODES = {404: ("not_found", "Resource not found."), 405: ("method_not_allowed", "Method not allowed.")}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        logger.info("request_id=%s api_error code=%s status=%s", request_id_of(request), exc.code, exc.status_code)
        return error_response(request, exc.status_code, exc.code, exc.detail)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Field names only; never echo submitted values.
        fields = sorted({str(e["loc"][-1]) for e in exc.errors() if e.get("loc")})
        logger.info("request_id=%s invalid_request fields=%s", request_id_of(request), fields)
        return error_response(request, 422, "invalid_request", "Malformed request: missing or invalid form fields.")

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code, detail = _HTTP_CODES.get(exc.status_code, ("http_error", "The request could not be processed."))
        return error_response(request, exc.status_code, code, detail)

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("request_id=%s unhandled_exception", request_id_of(request))
        return error_response(request, 500, "internal_error", "Internal error.")
