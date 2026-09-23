"""API integration tests. Core tests run against the REAL frozen artifact."""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import DEFAULT_MODEL_PATH, FROZEN_MODEL_SHA256
from app.main import create_app
from app.ml.features import FEATURE_NAMES, extract_features
from app.ml.model import ModelBundle
from conftest import jpeg_bytes, make_settings, noise, png_bytes, requires_real_model, write_tiny_bundle

LEAK_MARKERS = ("Traceback", "/home", "joblib", ".py", "stegoshield_rf", "site-packages", "Errno")


def assert_safe_error(resp, status: int, code: str) -> dict:
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert set(body) == {"detail", "code", "request_id"}
    assert body["code"] == code
    assert not any(m in resp.text for m in LEAK_MARKERS), resp.text
    return body


@pytest.fixture(scope="module")
def real_client():
    if not DEFAULT_MODEL_PATH.is_file():
        pytest.skip("frozen Phase 1 artifact not present locally")
    with TestClient(create_app(make_settings())) as c:
        yield c


def upload(client, data: bytes, name: str = "img.png", ctype: str = "image/png", path: str = "/api/v1/analyze"):
    return client.post(path, files={"file": (name, data, ctype)})


# ------------------------------------------------------------------ health / ready
def test_health_reports_liveness_only() -> None:
    with TestClient(create_app(make_settings(model_path=DEFAULT_MODEL_PATH.parent / "does_not_exist.joblib"))) as c:
        assert c.get("/health").json() == {"status": "ok", "service": "stegoshield-api"}
        assert c.get("/api/v1/health").json() == {"status": "ok", "service": "stegoshield-api"}


@requires_real_model
def test_ready_with_real_model(real_client) -> None:
    r = real_client.get("/ready")
    assert r.status_code == 200
    assert r.json() == {"status": "ready", "model_loaded": True, "reference_available": True, "reason": None}


@pytest.mark.parametrize(
    "case, reason",
    [("missing", "artifact_missing"), ("hash", "artifact_hash_mismatch"), ("malformed", "artifact_malformed"),
     ("schema", "feature_schema_version_mismatch"), ("names", "feature_names_mismatch")],
)
def test_not_ready_when_model_cannot_be_verified(tmp_path, case, reason) -> None:
    path = tmp_path / "m.joblib"
    if case == "missing":
        settings = make_settings(model_path=path)
    else:
        kwargs = {"malformed": {"drop_key": "model_name"}, "schema": {"schema": "v999"},
                  "names": {"feature_names": list(reversed(FEATURE_NAMES))}}.get(case, {})
        sha = write_tiny_bundle(path, **kwargs)
        settings = make_settings(model_path=path, model_sha256=("0" * 64) if case == "hash" else sha)
    with TestClient(create_app(settings)) as c:
        r = c.get("/ready")
        assert r.status_code == 503
        assert r.json()["status"] == "not_ready" and r.json()["model_loaded"] is False and r.json()["reason"] == reason
        assert str(tmp_path) not in r.text
        assert c.get("/health").status_code == 200  # alive but not ready
        assert_safe_error(upload(c, png_bytes(noise((128, 128)))), 503, "model_unavailable")
        assert_safe_error(c.get("/api/v1/model-info"), 503, "model_unavailable")


# ------------------------------------------------------------------ real-model analysis
@requires_real_model
def test_analyze_valid_image_schema_and_scale(real_client) -> None:
    r = upload(real_client, png_bytes(noise((256, 256), 1)))
    assert r.status_code == 200
    d = r.json()
    assert d["verdict"] in ("likely_cover", "potential_steganography")
    assert d["risk_level"] in ("low", "elevated", "high")
    assert 0 <= d["stego_score"] <= 100 and d["score_kind"] == "uncalibrated_model_score"
    assert 0 <= d["cover_reference_percentile"] <= 100
    assert d["model"] == "RandomForest" and d["model_version"] == "v1"
    assert d["feature_schema_version"] == "v1" and d["feature_count"] == 22
    assert 1 <= len(d["indicators"]) <= 5
    assert d["image"] == {"format": "PNG", "width": 256, "height": 256, "original_mode": "L", "size_bytes": len(png_bytes(noise((256, 256), 1))),
                          "alpha_discarded": False, "analyzed_as": "RGB (grey replicated)"}
    assert any("probability" in s for s in d["limitations"])
    assert r.headers["x-request-id"] == d["request_id"]
    assert r.headers["cache-control"] == "no-store"


@requires_real_model
def test_analysis_is_deterministic(real_client) -> None:
    data = png_bytes(noise((200, 200), 5))
    a, b = upload(real_client, data).json(), upload(real_client, data).json()
    a.pop("request_id"), b.pop("request_id")
    assert a == b


@requires_real_model
def test_api_score_equals_direct_phase1_pipeline(real_client) -> None:
    """The API must feed the frozen model exactly what Phase 1 did:
    grayscale -> RGB replication -> unchanged extractor -> unchanged ModelBundle.
    """
    gray = noise((160, 160), 9)
    vec, _ = extract_features(Image.fromarray(gray).convert("RGB"))
    bundle = ModelBundle(str(DEFAULT_MODEL_PATH))
    bundle.load()
    expected = round(bundle.predict_score(vec) * 100, 2)
    assert upload(real_client, png_bytes(gray)).json()["stego_score"] == expected


@requires_real_model
def test_filename_and_content_type_are_ignored(real_client) -> None:
    data = png_bytes(noise((128, 128), 3))
    a = upload(real_client, data, name="../../etc/passwd", ctype="text/plain").json()
    b = upload(real_client, data, name="ok.png").json()
    a.pop("request_id"), b.pop("request_id")
    assert a == b


@requires_real_model
def test_rgb_jpeg_and_rgba_inputs_are_handled_intentionally(real_client) -> None:
    rgb = noise((128, 128, 3), 4)
    j = upload(real_client, jpeg_bytes(rgb), "a.jpg", "image/jpeg").json()
    assert j["image"]["format"] == "JPEG" and any("JPEG input" in w for w in j["warnings"])
    rgba = np.dstack([rgb, noise((128, 128), 6)])
    a = upload(real_client, png_bytes(rgba)).json()
    assert a["image"]["alpha_discarded"] is True and any("alpha" in w for w in a["warnings"])
    # alpha must not influence the analysis: changing only alpha gives the same result
    rgba2 = np.dstack([rgb, noise((128, 128), 77)])
    b = upload(real_client, png_bytes(rgba2)).json()
    for d in (a, b):
        d.pop("request_id"), d["image"].pop("size_bytes")
    assert a == b


@requires_real_model
def test_model_info_matches_frozen_artifact(real_client) -> None:
    d = real_client.get("/api/v1/model-info").json()
    assert d["artifact_sha256"] == FROZEN_MODEL_SHA256
    assert d["feature_schema"] == "v1" and d["feature_count"] == 22 and d["scope"] == "LSB-focused image steganalysis"
    assert "not a probability" in d["score_interpretation"]
    assert set(d["risk_levels"]["levels"]) == {"elevated", "high"}
    assert not any(m in str(d) for m in ("/home", ".joblib"))


# ------------------------------------------------------------------ input rejection
def test_rejections_use_the_safe_error_contract(tmp_path) -> None:
    path = tmp_path / "m.joblib"
    sha = write_tiny_bundle(path)
    settings = make_settings(model_path=path, model_sha256=sha, max_upload_bytes=200_000, max_image_pixels=300 * 300)
    with TestClient(create_app(settings)) as c:
        assert_safe_error(upload(c, b""), 400, "empty_file")
        assert_safe_error(upload(c, b"definitely not an image"), 400, "invalid_image")
        assert_safe_error(upload(c, b"GIF89a" + b"\x00" * 50), 400, "unsupported_format")
        assert_safe_error(upload(c, png_bytes(noise((128, 128)))[:-40]), 400, "invalid_image")  # truncated PNG
        assert_safe_error(upload(c, png_bytes(noise((32, 32)))), 400, "invalid_dimensions")  # too small
        assert_safe_error(upload(c, png_bytes(noise((320, 320)))), 400, "invalid_dimensions")  # over pixel cap
        buf = io.BytesIO(); Image.fromarray(noise((128, 128))).convert("P").save(buf, format="PNG")
        assert_safe_error(upload(c, buf.getvalue()), 400, "unsupported_color_mode")
        buf = io.BytesIO(); Image.fromarray(noise((128, 128, 3))).convert("CMYK").save(buf, format="JPEG")
        assert_safe_error(upload(c, buf.getvalue()), 400, "unsupported_color_mode")
        assert_safe_error(upload(c, png_bytes(noise((300, 300))) + b"\x00" * 250_000), 413, "file_too_large")
        assert_safe_error(c.post("/api/v1/analyze"), 422, "invalid_request")  # missing file
        assert_safe_error(c.post("/api/v1/analyze", data={"file": "not-a-file"}), 422, "invalid_request")
        assert_safe_error(c.get("/api/v1/nope"), 404, "not_found")
        assert_safe_error(c.get("/api/v1/analyze"), 405, "method_not_allowed")


def test_oversized_bodies_rejected_before_parsing(tmp_path) -> None:
    path = tmp_path / "m.joblib"
    sha = write_tiny_bundle(path)
    settings = make_settings(model_path=path, model_sha256=sha, max_upload_bytes=1000, max_message_bytes=100)
    limit = settings.max_request_body_bytes
    with TestClient(create_app(settings)) as c:
        # declared Content-Length above the cap
        r = c.post("/api/v1/analyze", content=b"x" * (limit + 10), headers={"content-type": "multipart/form-data; boundary=b"})
        assert_safe_error(r, 413, "file_too_large")

        # chunked upload (no Content-Length) that streams past the cap
        def chunks():
            for _ in range(limit // 4096 + 3):
                yield b"x" * 4096

        r = c.post("/api/v1/analyze", content=chunks(), headers={"content-type": "multipart/form-data; boundary=b"})
        assert_safe_error(r, 413, "file_too_large")


def test_unexpected_failure_is_generic_and_not_leaked(tmp_path, monkeypatch) -> None:
    path = tmp_path / "m.joblib"
    sha = write_tiny_bundle(path)
    with TestClient(create_app(make_settings(model_path=path, model_sha256=sha))) as c:
        def boom(*a, **k):
            raise RuntimeError("secret /home/user/model.joblib token=abc123")

        monkeypatch.setattr(c.app.state.services.analysis, "analyze", boom)
        body = assert_safe_error(upload(c, png_bytes(noise((128, 128)))), 500, "analysis_failed")
        assert "abc123" not in str(body) and "secret" not in str(body)


def test_request_ids_are_server_generated_and_unique(tmp_path) -> None:
    path = tmp_path / "m.joblib"
    sha = write_tiny_bundle(path)
    with TestClient(create_app(make_settings(model_path=path, model_sha256=sha))) as c:
        r1 = c.get("/health", headers={"X-Request-ID": "attacker\nInjected: 1"})
        r2 = c.get("/health")
        assert r1.headers["x-request-id"] != r2.headers["x-request-id"]
        assert "attacker" not in r1.headers["x-request-id"]
        assert r1.headers["x-content-type-options"] == "nosniff"


def test_cors_is_restricted_to_configured_origins(tmp_path) -> None:
    path = tmp_path / "m.joblib"
    sha = write_tiny_bundle(path)
    with TestClient(create_app(make_settings(model_path=path, model_sha256=sha, cors_origins=("http://app.example",)))) as c:
        ok = c.options("/api/v1/analyze", headers={"Origin": "http://app.example", "Access-Control-Request-Method": "POST"})
        assert ok.headers.get("access-control-allow-origin") == "http://app.example"
        assert "access-control-allow-credentials" not in ok.headers
        bad = c.options("/api/v1/analyze", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"})
        assert "access-control-allow-origin" not in bad.headers


def test_inconclusive_when_reference_does_not_match_model(tmp_path) -> None:
    """A model other than the one the reference was built for must not get risk levels."""
    path = tmp_path / "m.joblib"
    sha = write_tiny_bundle(path)
    with TestClient(create_app(make_settings(model_path=path, model_sha256=sha))) as c:
        assert c.get("/ready").json()["reference_available"] is False
        d = upload(c, png_bytes(noise((128, 128)))).json()
        assert d["verdict"] == "inconclusive" and d["risk_level"] == "not_assessed" and d["cover_reference_percentile"] is None
        assert d["model"] == "TinyTestForest"


# ------------------------------------------------------------------ encode
def test_encode_endpoint_round_trip_and_analyzable_output(tmp_path) -> None:
    from app.services.encode_service import decode_message

    path = tmp_path / "m.joblib"
    sha = write_tiny_bundle(path)
    with TestClient(create_app(make_settings(model_path=path, model_sha256=sha))) as c:
        cover = noise((128, 128, 3), 2)
        r = c.post("/api/v1/encode", files={"file": ("cover.png", png_bytes(cover), "image/png")}, data={"message": "héllo ✓ secret"})
        assert r.status_code == 200 and r.headers["content-type"] == "image/png"
        assert r.headers["content-disposition"] == 'attachment; filename="stegoshield-encoded.png"'
        stego = Image.open(io.BytesIO(r.content))
        assert stego.format == "PNG" and stego.size == (128, 128)
        assert decode_message(stego) == "héllo ✓ secret"
        assert int(r.headers["x-embedded-bytes"]) == len("héllo ✓ secret".encode())
        assert int(r.headers["x-capacity-bytes"]) == 128 * 128 * 3 // 8 - 8
        # the generated image is itself a valid analysis input
        assert upload(c, r.content).status_code == 200


def test_encode_rejections(tmp_path) -> None:
    path = tmp_path / "m.joblib"
    sha = write_tiny_bundle(path)
    with TestClient(create_app(make_settings(model_path=path, model_sha256=sha, max_message_bytes=100))) as c:
        img = png_bytes(noise((64, 64)))
        post = lambda data: c.post("/api/v1/encode", files={"file": ("c.png", img, "image/png")}, data=data)
        assert_safe_error(post({"message": ""}), 400, "missing_message")
        assert_safe_error(post({"message": "x" * 101}), 400, "message_too_large")
        assert_safe_error(c.post("/api/v1/encode", files={"file": ("c.png", img, "image/png")}), 400, "missing_message")
        assert_safe_error(c.post("/api/v1/encode", data={"message": "hi"}), 422, "invalid_request")  # missing file
        small = png_bytes(noise((64, 64)))  # capacity 64*64//8 - 8 = 504 bytes; raise the message limit to reach it
    with TestClient(create_app(make_settings(model_path=path, model_sha256=sha, max_message_bytes=5000))) as c:
        r = c.post("/api/v1/encode", files={"file": ("c.png", small, "image/png")}, data={"message": "y" * 600})
        assert_safe_error(r, 400, "payload_exceeds_capacity")
        assert_safe_error(c.post("/api/v1/encode", files={"file": ("c.png", b"junk", "image/png")}, data={"message": "hi"}), 400, "invalid_image")
