"""Gateway local-development and production JWT authentication tests."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from services.gateway.auth import authenticate_user, authorize_request_user
from services.gateway import main as gateway_main


JWT_SECRET = "test-secret-with-at-least-thirty-two-bytes"


def _segment(value: dict[str, object]) -> str:
    encoded = json.dumps(value, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(encoded).rstrip(b"=").decode("ascii")


def _token(payload: dict[str, object], secret: str = JWT_SECRET) -> str:
    header = _segment({"alg": "HS256", "typ": "JWT"})
    body = _segment(payload)
    signed = f"{header}.{body}"
    signature = hmac.new(secret.encode("utf-8"), signed.encode("ascii"), hashlib.sha256).digest()
    encoded_signature = base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")
    return f"{signed}.{encoded_signature}"


class GatewayAuthenticationTest(unittest.TestCase):
    def test_local_mode_requires_persistent_header_identity(self) -> None:
        with patch.dict("os.environ", {"GATEWAY_AUTH_MODE": "local"}, clear=True):
            self.assertEqual(authenticate_user(authorization=None, x_user_id="user-a"), "user-a")
            with self.assertRaises(HTTPException) as raised:
                authenticate_user(authorization=None, x_user_id="anonymous")
        self.assertEqual(raised.exception.status_code, 400)

    def test_local_mode_rejects_body_identity_by_default(self) -> None:
        with patch.dict("os.environ", {"GATEWAY_AUTH_MODE": "local"}, clear=True):
            with self.assertRaises(HTTPException) as raised:
                authorize_request_user("user-a", authorization=None, x_user_id=None)
        self.assertEqual(raised.exception.status_code, 400)

    def test_local_mode_body_identity_requires_explicit_development_switch(self) -> None:
        env = {"GATEWAY_AUTH_MODE": "local", "GATEWAY_ALLOW_BODY_USER_ID": "1"}
        with patch.dict("os.environ", env, clear=True):
            self.assertEqual(authorize_request_user("user-a", authorization=None, x_user_id=None), "user-a")

    def test_local_mode_is_rejected_in_production(self) -> None:
        env = {"GATEWAY_AUTH_MODE": "local", "NAWEI_ENV": "production"}
        with patch.dict("os.environ", env, clear=True), self.assertRaises(HTTPException) as raised:
            authenticate_user(authorization=None, x_user_id="user-a")
        self.assertEqual(raised.exception.status_code, 503)

    def test_jwt_mode_returns_verified_subject(self) -> None:
        token = _token(
            {
                "sub": "user-a",
                "exp": int(time.time()) + 300,
                "iss": "nawei-auth",
                "aud": "nawei-gateway",
            }
        )
        env = {
            "GATEWAY_AUTH_MODE": "jwt",
            "GATEWAY_JWT_SECRET": JWT_SECRET,
            "GATEWAY_JWT_ISSUER": "nawei-auth",
            "GATEWAY_JWT_AUDIENCE": "nawei-gateway",
        }
        with patch.dict("os.environ", env, clear=True):
            subject = authenticate_user(authorization=f"Bearer {token}", x_user_id="forged-user")
        self.assertEqual(subject, "user-a")

    def test_jwt_mode_rejects_expired_and_invalid_signatures(self) -> None:
        expired = _token({"sub": "user-a", "exp": int(time.time()) - 1})
        invalid = _token({"sub": "user-a", "exp": int(time.time()) + 300}, secret="x" * 32)
        env = {"GATEWAY_AUTH_MODE": "jwt", "GATEWAY_JWT_SECRET": JWT_SECRET}
        with patch.dict("os.environ", env, clear=True):
            for token in (expired, invalid):
                with self.subTest(token=token[-12:]), self.assertRaises(HTTPException) as raised:
                    authenticate_user(authorization=f"Bearer {token}", x_user_id=None)
                self.assertEqual(raised.exception.status_code, 401)

    def test_jwt_mode_fails_closed_when_secret_is_not_configured(self) -> None:
        with patch.dict("os.environ", {"GATEWAY_AUTH_MODE": "jwt"}, clear=True):
            with self.assertRaises(HTTPException) as raised:
                authenticate_user(authorization="Bearer invalid", x_user_id=None)
        self.assertEqual(raised.exception.status_code, 503)

    def test_completion_endpoint_rejects_body_user_impersonation(self) -> None:
        token = _token({"sub": "user-a", "exp": int(time.time()) + 300})
        request = {
            "request_id": "auth-request",
            "trace_id": "auth-trace",
            "session_id": "auth-session",
            "user_id": "user-b",
            "message": {"role": "user", "content": "test"},
            "stream": False,
        }
        env = {"GATEWAY_AUTH_MODE": "jwt", "GATEWAY_JWT_SECRET": JWT_SECRET}
        with patch.dict("os.environ", env, clear=True):
            response = TestClient(gateway_main.app).post(
                "/api/v1/text/completions",
                headers={"Authorization": f"Bearer {token}"},
                json=request,
            )
        self.assertEqual(response.status_code, 403)

    def test_jwt_rejects_non_finite_expiry(self) -> None:
        token = _token({"sub": "user-a", "exp": "NaN"})
        env = {"GATEWAY_AUTH_MODE": "jwt", "GATEWAY_JWT_SECRET": JWT_SECRET}
        with patch.dict("os.environ", env, clear=True), self.assertRaises(HTTPException) as raised:
            authenticate_user(authorization=f"Bearer {token}", x_user_id=None)
        self.assertEqual(raised.exception.status_code, 401)


if __name__ == "__main__":
    unittest.main()
