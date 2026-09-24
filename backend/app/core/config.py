"""Application configuration.

All limits are configurable through environment variables (same names as the
project's `.env.example`). Nothing here is read from request input.

`Settings.from_env()` is evaluated when the application is created, not at
import time, so tests and deployments can build differently-configured apps.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# SHA-256 of the frozen Phase 1A Random Forest artifact. The API refuses to
# serve a model whose bytes do not hash to this value unless the operator
# explicitly overrides it (MODEL_SHA256) or disables the check (empty value).
FROZEN_MODEL_SHA256 = "34707f83dc385e0846be3df7c56da930aad951a23fd3daa032a050449d02523d"
EXPECTED_FEATURE_SCHEMA_VERSION = "v1"

DEFAULT_MODEL_PATH = REPO_ROOT / "data" / "models" / "stegoshield_rf.joblib"
DEFAULT_REFERENCE_PATH = Path(__file__).resolve().parents[1] / "resources" / "reference_scores_v1.json"

# Explicit origins of the Next.js dev server (`npm run dev`, port 3000). The browser sends whichever
# host the user typed, so both spellings are listed. Never a wildcard; override with CORS_ORIGINS.
DEFAULT_CORS_ORIGINS = ("http://localhost:3000", "http://127.0.0.1:3000")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"Environment variable {name} must be positive")
    return value


def _env_path(name: str, default: Path) -> Path:
    """Relative paths are resolved against the repository root, never the CWD."""
    raw = os.getenv(name)
    path = Path(raw) if raw and raw.strip() else default
    return path if path.is_absolute() else REPO_ROOT / path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    api_version: str = "0.2.0"

    # --- upload / image limits -------------------------------------------------
    max_upload_bytes: int = 10 * 1024 * 1024
    min_image_side: int = 64
    max_image_side: int = 4096
    # Feature extraction peaks at roughly 55-60 bytes of RAM per pixel
    # (measured: 2048x2048 ~ 270 MB and ~0.33 s; 4096x4096 ~ 1 GB and ~1.3 s),
    # so the pixel count, not just the side length, is capped.
    max_image_pixels: int = 2048 * 2048
    max_message_bytes: int = 16 * 1024
    max_concurrent_analyses: int = 2

    # --- model -------------------------------------------------------------------
    model_path: Path = DEFAULT_MODEL_PATH
    model_sha256: str | None = FROZEN_MODEL_SHA256
    reference_path: Path = DEFAULT_REFERENCE_PATH

    # --- http ----------------------------------------------------------------
    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS
    enable_docs: bool = True

    @property
    def max_request_body_bytes(self) -> int:
        """Hard cap on the whole multipart body: file + message + envelope slack."""
        return self.max_upload_bytes + self.max_message_bytes + 64 * 1024

    @classmethod
    def from_env(cls) -> "Settings":
        model_sha = os.getenv("MODEL_SHA256")
        if model_sha is None:
            expected: str | None = FROZEN_MODEL_SHA256
        elif model_sha.strip() == "":
            expected = None  # integrity check explicitly disabled by the operator
        else:
            expected = model_sha.strip().lower()

        origins = os.getenv("CORS_ORIGINS")
        cors = tuple(o.strip() for o in origins.split(",") if o.strip()) if origins else DEFAULT_CORS_ORIGINS
        if "*" in cors:
            raise ValueError("CORS_ORIGINS must list explicit origins; '*' is not allowed")

        return cls(
            max_upload_bytes=_env_int("MAX_UPLOAD_BYTES", cls.max_upload_bytes),
            min_image_side=_env_int("MIN_IMAGE_SIDE", cls.min_image_side),
            max_image_side=_env_int("MAX_IMAGE_SIDE", cls.max_image_side),
            max_image_pixels=_env_int("MAX_IMAGE_PIXELS", cls.max_image_pixels),
            max_message_bytes=_env_int("MAX_MESSAGE_BYTES", cls.max_message_bytes),
            max_concurrent_analyses=_env_int("MAX_CONCURRENT_ANALYSES", cls.max_concurrent_analyses),
            model_path=_env_path("MODEL_PATH", DEFAULT_MODEL_PATH),
            model_sha256=expected,
            reference_path=_env_path("REFERENCE_PATH", DEFAULT_REFERENCE_PATH),
            cors_origins=cors,
            enable_docs=_env_bool("ENABLE_DOCS", True),
        )
