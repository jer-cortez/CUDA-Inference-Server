"""Pure-ASGI bearer authentication for prediction requests."""

from __future__ import annotations

import hashlib
import hmac
import logging
import re
from typing import Any

from ..security.keys import ApiKeyRecord, KEY_ID_PATTERN, load_key_file
from .admission import ASGIApp, _json_response
from .observability import log_message

_PREDICTION_PATHS = {"/predict", "/predict/", "/predict/raw", "/predict/raw/"}
_SECRET_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_DUMMY_DIGEST = "0" * 64

security_logger = logging.getLogger("cuda_db.security")


class ApiKeyAuthenticator:
    """A restart-scoped immutable view of the configured key store."""

    def __init__(self, required: bool) -> None:
        self.required = required
        self._digests: dict[str, str] | None = {} if not required else None

    def load(self, path: str) -> None:
        if not self.required:
            self._digests = {}
            return
        records: tuple[ApiKeyRecord, ...] = load_key_file(path, require_nonempty=True)
        self._digests = {record.key_id: record.sha256 for record in records}

    def clear(self) -> None:
        self._digests = {} if not self.required else None

    def authenticate(self, headers: list[tuple[bytes, bytes]]) -> str | None:
        """Return the authenticated id, or ``None`` without revealing why."""
        if not self.required:
            return "-"
        if self._digests is None:
            return None

        authorization = [
            value for name, value in headers if name.lower() == b"authorization"
        ]
        if len(authorization) != 1:
            return None
        try:
            value = authorization[0].decode("ascii")
        except UnicodeDecodeError:
            return None
        pieces = value.split(" ")
        if len(pieces) != 2 or pieces[0].lower() != "bearer" or not pieces[1]:
            return None
        token = pieces[1]
        token_parts = token.split(".")
        if len(token_parts) != 2:
            return None
        key_id, secret = token_parts
        if KEY_ID_PATTERN.fullmatch(key_id) is None:
            return None
        if _SECRET_PATTERN.fullmatch(secret) is None:
            return None

        actual = hashlib.sha256(token.encode("ascii")).hexdigest()
        expected = self._digests.get(key_id, _DUMMY_DIGEST)
        if not hmac.compare_digest(actual, expected):
            return None
        return key_id


class AuthenticationMiddleware:
    """Authenticate before admission and before any request-body receive."""

    def __init__(self, app: ASGIApp, authenticator: ApiKeyAuthenticator) -> None:
        self.app = app
        self.authenticator = authenticator

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        protected = (
            scope["type"] == "http"
            and scope.get("method") == "POST"
            and scope.get("path") in _PREDICTION_PATHS
            and self.authenticator.required
        )
        if not protected:
            await self.app(scope, receive, send)
            return

        key_id = self.authenticator.authenticate(scope.get("headers", []))
        if key_id is None:
            security_logger.warning(
                log_message(
                    "authentication_failed",
                    request_id=scope.get("cuda_db.request_id", "-"),
                    endpoint=_canonical_endpoint(scope.get("path")),
                ),
                extra={
                    "request_id": scope.get("cuda_db.request_id", "-"),
                    "endpoint": _canonical_endpoint(scope.get("path")),
                },
            )
            await _json_response(
                send,
                401,
                {"detail": "authentication required"},
                headers=[(b"www-authenticate", b"Bearer")],
            )
            return

        scope["cuda_db.client_key_id"] = key_id
        await self.app(scope, receive, send)


def _canonical_endpoint(path: Any) -> str:
    if path in {"/predict", "/predict/"}:
        return "/predict"
    if path in {"/predict/raw", "/predict/raw/"}:
        return "/predict/raw"
    return "unknown"
