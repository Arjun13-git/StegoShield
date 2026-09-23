"""Direct ASGI tests for the body-size limit: a real multi-message stream
(which the TestClient cannot produce) and dishonest/absent Content-Length.
"""

import asyncio
import json

from app.core.middleware import BodySizeLimitMiddleware, RequestContextMiddleware


async def _echo_length_app(scope, receive, send):
    total = 0
    while True:
        msg = await receive()
        if msg["type"] == "http.disconnect":
            return
        total += len(msg.get("body", b""))
        if not msg.get("more_body", False):
            break
    await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/plain")]})
    await send({"type": "http.response.body", "body": str(total).encode()})


def run(headers: list[tuple[bytes, bytes]], chunks: list[bytes], limit: int):
    sent: list[dict] = []
    queue = [{"type": "http.request", "body": c, "more_body": i < len(chunks) - 1} for i, c in enumerate(chunks)]
    calls = {"n": 0}

    async def receive():
        calls["n"] += 1
        return queue.pop(0) if queue else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    app = RequestContextMiddleware(BodySizeLimitMiddleware(_echo_length_app, max_body_bytes=limit))
    scope = {"type": "http", "method": "POST", "path": "/x", "headers": headers}
    asyncio.run(app(scope, receive, send))
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return status, body, calls["n"]


def test_stream_under_the_limit_passes_through_untouched() -> None:
    status, body, _ = run([], [b"a" * 100] * 3, limit=400)
    assert (status, body) == (200, b"300")


def test_stream_exceeding_the_limit_without_content_length_is_cut_off_with_413() -> None:
    status, body, receives = run([], [b"a" * 100] * 10, limit=250)
    assert status == 413
    assert json.loads(body)["code"] == "file_too_large" and "request_id" in json.loads(body)
    assert receives <= 4  # stopped reading soon after the cap instead of consuming all 10 chunks


def test_declared_content_length_over_the_limit_is_rejected_before_reading_anything() -> None:
    status, body, receives = run([(b"content-length", b"999999")], [b"a"], limit=1000)
    assert status == 413 and receives == 0


def test_lying_content_length_is_still_caught_by_byte_counting() -> None:
    status, _, _ = run([(b"content-length", b"10")], [b"a" * 100] * 10, limit=250)
    assert status == 413


def test_unparseable_content_length_falls_back_to_counting() -> None:
    assert run([(b"content-length", b"abc")], [b"a" * 100] * 10, limit=250)[0] == 413
    assert run([(b"content-length", b"abc")], [b"a" * 10], limit=250)[0] == 200
