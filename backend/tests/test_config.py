from pathlib import Path

import pytest

from app.core.config import DEFAULT_CORS_ORIGINS, FROZEN_MODEL_SHA256, REPO_ROOT, Settings
from app.main import create_app

ENV_KEYS = ("MAX_UPLOAD_BYTES", "MIN_IMAGE_SIDE", "MAX_IMAGE_SIDE", "MAX_IMAGE_PIXELS", "MAX_MESSAGE_BYTES",
            "MAX_CONCURRENT_ANALYSES", "MODEL_PATH", "MODEL_SHA256", "REFERENCE_PATH", "CORS_ORIGINS", "ENABLE_DOCS")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


def test_defaults_are_conservative_and_pin_the_frozen_model() -> None:
    s = Settings.from_env()
    assert s.model_sha256 == FROZEN_MODEL_SHA256
    assert s.model_path == REPO_ROOT / "data" / "models" / "stegoshield_rf.joblib"
    assert s.max_image_pixels == 2048 * 2048 and s.max_concurrent_analyses == 2
    assert s.cors_origins == ("http://localhost:3000", "http://127.0.0.1:3000")
    assert s.max_request_body_bytes > s.max_upload_bytes


def test_env_overrides_and_relative_paths_resolve_against_repo_root_not_cwd(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODEL_PATH", "data/models/other.joblib")
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "12345")
    monkeypatch.setenv("CORS_ORIGINS", "http://a.example, http://b.example")
    s = Settings.from_env()
    assert s.model_path == REPO_ROOT / "data" / "models" / "other.joblib"
    assert s.max_upload_bytes == 12345 and s.cors_origins == ("http://a.example", "http://b.example")


def test_wildcard_cors_and_bad_numbers_are_refused(monkeypatch) -> None:
    monkeypatch.setenv("CORS_ORIGINS", "*")
    with pytest.raises(ValueError):
        Settings.from_env()
    monkeypatch.delenv("CORS_ORIGINS")
    for bad in ("abc", "0", "-5"):
        monkeypatch.setenv("MAX_UPLOAD_BYTES", bad)
        with pytest.raises(ValueError):
            Settings.from_env()


def test_integrity_pin_can_be_overridden_or_explicitly_disabled(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_SHA256", "ABCDEF")
    assert Settings.from_env().model_sha256 == "abcdef"
    monkeypatch.setenv("MODEL_SHA256", "")
    assert Settings.from_env().model_sha256 is None


def test_app_is_not_in_debug_mode_and_docs_can_be_disabled(monkeypatch) -> None:
    assert create_app(Settings()).debug is False
    monkeypatch.setenv("ENABLE_DOCS", "false")
    app = create_app(Settings.from_env())
    assert app.docs_url is None and app.openapi_url is None


def test_default_cors_origins_cover_the_next_dev_server_in_both_spellings_and_are_explicit() -> None:
    for settings in (Settings(), Settings.from_env()):
        assert "http://localhost:3000" in settings.cors_origins
        assert "http://127.0.0.1:3000" in settings.cors_origins
        assert "*" not in settings.cors_origins
    assert Settings().cors_origins == DEFAULT_CORS_ORIGINS


def test_env_example_documents_the_same_cors_defaults() -> None:
    line = next(l for l in (REPO_ROOT / ".env.example").read_text().splitlines() if l.startswith("CORS_ORIGINS="))
    assert tuple(o.strip() for o in line.split("=", 1)[1].split(",")) == DEFAULT_CORS_ORIGINS
