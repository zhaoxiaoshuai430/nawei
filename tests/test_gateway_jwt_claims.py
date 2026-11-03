"""Signed JWT claims must fail cleanly and preserve persistent identity rules."""

import base64
import hashlib
import hmac
import json
import time

import pytest
from fastapi import HTTPException

from services.gateway.auth import authenticate_user


def token(claims, secret):
    def segment(value):
        return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")

    signed = segment({"alg": "HS256"}) + "." + segment(claims)
    signature = hmac.new(secret.encode(), signed.encode(), hashlib.sha256).digest()
    return signed + "." + base64.urlsafe_b64encode(signature).decode().rstrip("=")


@pytest.fixture
def authorization(monkeypatch):
    secret = "test-jwt-signing-secret-32-bytes-long"
    monkeypatch.setenv("GATEWAY_AUTH_MODE", "jwt")
    monkeypatch.setenv("GATEWAY_JWT_SECRET", secret)
    monkeypatch.delenv("GATEWAY_JWT_ISSUER", raising=False)
    monkeypatch.delenv("GATEWAY_JWT_AUDIENCE", raising=False)
    return lambda **claims: "Bearer " + token({"sub": "owner", "exp": time.time() + 60, **claims}, secret)


@pytest.mark.parametrize("subject", ["anonymous", " anonymous ", "\tanonymous\n"])
def test_reserved_anonymous_subject_is_rejected_after_normalization(authorization, subject):
    with pytest.raises(HTTPException) as exc:
        authenticate_user(authorization(sub=subject), None)
    assert exc.value.status_code == 401


@pytest.mark.parametrize("audience", ["教育助手", ["教育助手"], ["其他", "教育助手"]])
@pytest.mark.parametrize("expected", ["education-api", "教育助手"])
def test_unicode_audience_is_compared_without_server_error(monkeypatch, authorization, audience, expected):
    monkeypatch.setenv("GATEWAY_JWT_AUDIENCE", expected)
    if expected == "教育助手":
        assert authenticate_user(authorization(aud=audience), None) == "owner"
    else:
        with pytest.raises(HTTPException) as exc:
            authenticate_user(authorization(aud=audience), None)
        assert exc.value.status_code == 401


@pytest.mark.parametrize("claim", ["exp", "nbf"])
@pytest.mark.parametrize("value", [10 ** 400, -(10 ** 400)], ids=["positive_overflow", "negative_overflow"])
def test_out_of_range_numeric_dates_are_rejected(authorization, claim, value):
    with pytest.raises(HTTPException) as exc:
        authenticate_user(authorization(**{claim: value}), None)
    assert exc.value.status_code == 401
