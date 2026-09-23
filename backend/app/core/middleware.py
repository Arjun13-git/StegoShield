"""Pure-ASGI middleware (no BaseHTTPMiddleware: it buffers bodies and swallows
disconnects, which is exactly what a body-size limit must not do).
"""

from __future__ import annotations

import json
import uuid

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class RequestContextMiddleware:
    """Assign a server-generated request id (client-supplied ids are ignored,
    which prevents log injection) and add conservative response headers.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = str(uuid.uuid4())
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = request_id
                headers["X-Content-Type-Options"] = "nosniff"
                headers["Cache-Control"] = "no-store"
            await send(message)

        await self.app(scope, receive, send_with_headers)


class BodySizeLimitMiddleware:
    """Reject request bodies larger than `max_body_bytes` BEFORE the multipart
    parser buffers them (Starlette spools uploaded parts to disk without a
    size bound). Checks Content-Length up front and counts streamed bytes for
    chunked or dishonest requests.
    """

    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def _reject(self, scope: Scope, send: Send) -> None:
        request_id = scope.get("state", {}).get("request_id", "unknown")
        body = json.dumps(
            {"detail": "Request body exceeds the configured upload limit.", "code": "file_too_large", "request_id": request_id}
        ).encode()
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()),
                                (b"connection", b"close")]})
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None:
            try:
                if int(declared) > self.max_body_bytes:
                    await self._reject(scope, send)
                    return
            except ValueError:
                pass  # fall through to byte counting

        received = 0
        rejected = False

        async def limited_receive() -> Message:
            nonlocal received, rejected
            if rejected:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    rejected = True
                    await self._reject(scope, send)
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message: Message) -> None:
            if not rejected:  # after our 413, drop anything the app tries to send
                await send(message)

        await self.app(scope, limited_receive, guarded_send)
