"""Exercise the ASR HTTP boundary without loading speech models."""

from __future__ import annotations

import base64
import importlib.util
import io
import sys
import types
import wave
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app_protocols import ASRResponse


@pytest.fixture
def asr_service(monkeypatch: pytest.MonkeyPatch):
    source = Path(__file__).resolve().parents[1] / "services" / "asr" / "src"
    monkeypatch.syspath_prepend(str(source))
    import runtime_tools

    monkeypatch.setattr(runtime_tools, "ensure_ffmpeg_on_path", lambda: None)
    monkeypatch.setattr(runtime_tools, "resolve_model_source", lambda *args, **kwargs: "test-model")
    monkeypatch.setenv("NAWEI_ENV", "development")
    monkeypatch.setenv("ASR_INTERNAL_API_KEY", "")
    monkeypatch.setenv("ASR_CORS_ORIGINS", "")
    monkeypatch.setenv("ASR_REALTIME_BACKEND", "sensevoice")
    monkeypatch.setitem(sys.modules, "model", types.ModuleType("model"))
    funasr = types.ModuleType("funasr")
    funasr.AutoModel = Mock(return_value=Mock())
    monkeypatch.setitem(sys.modules, "funasr", funasr)
    postprocess = types.ModuleType("funasr.utils.postprocess_utils")
    postprocess.rich_transcription_postprocess = lambda text: text
    monkeypatch.setitem(sys.modules, "funasr.utils.postprocess_utils", postprocess)
    spec = importlib.util.spec_from_file_location("asr_pcm16_test_service", source / "main.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def payload(content: bytes, **overrides) -> dict:
    return {
        "request_id": "pcm-test", "trace_id": "pcm-test", "session_id": "pcm-test",
        "audio_base64": base64.b64encode(content).decode("ascii"), "format": "pcm16",
        "sample_rate": 16000, "channels": 1, **overrides,
    }


@pytest.mark.parametrize(("sample_rate", "channels"), [(16000, 1), (24000, 2)])
def test_pcm16_upload_reaches_model_as_wav(asr_service, monkeypatch, sample_rate, channels):
    pcm = b"\x01\x00\xff\xff" * 320

    async def recognize(**kwargs):
        assert kwargs["suffix"] == ".wav"
        with wave.open(io.BytesIO(kwargs["content"]), "rb") as audio:
            assert audio.getframerate() == sample_rate
            assert audio.getnchannels() == channels
            assert audio.getsampwidth() == 2
            assert audio.readframes(audio.getnframes()) == pcm
        return {"text": "recognized", "language": "en"}

    monkeypatch.setattr(asr_service, "_recognize_content_async", recognize)
    with TestClient(asr_service.app) as client:
        response = client.post("/api/v1/audio/transcriptions/base64", json=payload(
            pcm, sample_rate=sample_rate, channels=channels, options={"filename": "upload.raw"},
        ))
    assert response.status_code == 200, response.text
    assert response.json()["text"] == "recognized"


@pytest.mark.parametrize(("content", "channels"), [(b"\x01", 1), (b"\x01\x00", 2)])
def test_incomplete_pcm16_frame_returns_bad_request(asr_service, monkeypatch, content, channels):
    async def recognize(**kwargs):
        raise AssertionError("Invalid PCM must be rejected before inference")

    monkeypatch.setattr(asr_service, "_recognize_content_async", recognize)
    with TestClient(asr_service.app) as client:
        response = client.post("/api/v1/audio/transcriptions/base64", json=payload(content, channels=channels))
    assert response.status_code == 400, response.text
    asr_service.model.generate.assert_not_called()


def test_asr_response_with_corrections_matches_gateway_contract(asr_service, monkeypatch):
    corrections = [{"source": "misheard", "replacement": "corrected", "occurrences": "1"}]

    async def recognize(**kwargs):
        return {"text": "corrected", "clean_text": "corrected", "language": "en", "corrections": corrections}

    monkeypatch.setattr(asr_service, "_recognize_content_async", recognize)
    with TestClient(asr_service.app) as client:
        response = client.post("/api/v1/audio/transcriptions/base64", json=payload(b"\x01\x00" * 320))

    assert response.status_code == 200
    parsed = ASRResponse.model_validate(response.json())
    assert parsed.text == "corrected"
    assert parsed.asr.corrections == corrections


def test_native_pcm16_uses_wav_inference_and_cleans_temporary_file(asr_service, monkeypatch):
    pcm = b"\x01\x00\xff\xff" * 320
    paths = []

    def generate(**kwargs):
        path = Path(kwargs["input"])
        paths.append(path)
        with wave.open(str(path), "rb") as audio:
            assert audio.getframerate() == 16000
            assert audio.getnchannels() == 1
            assert audio.getsampwidth() == 2
            assert audio.readframes(audio.getnframes()) == pcm
        return [{"text": "recognized"}]

    asr_service.model.generate.side_effect = generate
    decoder = Mock(side_effect=AssertionError("Native WAV must not invoke FFmpeg"))
    monkeypatch.setattr(asr_service, "_find_ffmpeg", decoder)
    with TestClient(asr_service.app) as client:
        response = client.post("/api/v1/asr/transcriptions", json=payload(pcm))
    assert response.status_code == 200, response.text
    assert response.json()["text"] == "recognized"
    assert len(paths) == 1
    assert not paths[0].exists()
    decoder.assert_not_called()


def test_native_realtime_dispatch_preserves_legacy_upload(asr_service, monkeypatch):
    marker = object()
    pool = types.SimpleNamespace(session=lambda: nullcontext(marker), snapshot=lambda: {"limit": 2})
    monkeypatch.setattr(asr_service, "STREAMING_MODEL", pool)

    async def serve(websocket, decoder):
        assert decoder is marker
        await websocket.accept()
        await websocket.send_json({"native": True})
        await websocket.close()

    monkeypatch.setattr(asr_service, "serve_streaming", serve)
    with TestClient(asr_service.app) as client:
        capabilities = client.get("/api/v1/capabilities").json()
        assert capabilities["keeps_decoder_state"] and capabilities["partial_results"]
        assert capabilities["http_streaming_mode"] == "independent-chunk"
        assert capabilities["streaming_concurrency"]["limit"] == 2
        assert client.get("/health").json()["model"] == asr_service.MODEL_ID
        with client.websocket_connect("/api/v1/asr/realtime") as connection:
            assert connection.receive_json() == {"native": True}
        assert any(route.path == "/api/v1/asr/realtime/sensevoice" for route in asr_service.app.routes)
