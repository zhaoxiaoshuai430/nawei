"""Internal authentication rejects malformed credentials without server errors."""

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from services.asr.src.service_security import require_asr_internal_api_key
from services.knowledge.auth import authorize_knowledge_admin_request, authorize_knowledge_internal_request
from services.llm.auth import authorize_llm_internal_request
from services.tts.app.service_security import require_tts_internal_api_key


@pytest.fixture(params=[
    ("ASR_INTERNAL_API_KEY", "X-Internal-API-Key", require_asr_internal_api_key),
    ("TTS_INTERNAL_API_KEY", "X-Internal-API-Key", require_tts_internal_api_key),
    ("LLM_INTERNAL_API_KEY", "X-Internal-API-Key", authorize_llm_internal_request),
    ("KNOWLEDGE_INTERNAL_API_KEY", "X-Internal-API-Key", authorize_knowledge_internal_request),
    ("KNOWLEDGE_ADMIN_API_KEY", "X-Admin-API-Key", authorize_knowledge_admin_request),
])
def auth_boundary(request, monkeypatch):
    name, header, authorize = request.param
    monkeypatch.setenv("NAWEI_ENV", "production")
    monkeypatch.setenv(name, "valid-internal-key-" * 2)
    return name, header, authorize


def test_non_ascii_header_returns_unauthorized(auth_boundary):
    _, header, authorize = auth_boundary
    app = FastAPI()

    @app.get("/protected", dependencies=[Depends(authorize)])
    def protected():
        return {"ok": True}

    with TestClient(app) as client:
        response = client.get("/protected", headers=[(header.encode(), b"invalid-\xe9-key")])
        assert response.status_code == 401
        assert client.get("/protected", headers={header: "valid-internal-key-" * 2}).status_code == 200


def test_configured_unicode_key_uses_byte_comparison(auth_boundary, monkeypatch):
    name, _, authorize = auth_boundary
    key = "\u00e9" * 32
    monkeypatch.setenv(name, key)
    authorize(key)
    with pytest.raises(HTTPException) as caught:
        authorize("wrong-key")
    assert caught.value.status_code == 401
