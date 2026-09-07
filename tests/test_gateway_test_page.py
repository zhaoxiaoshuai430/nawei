"""Gateway browser test-page regression tests."""

from fastapi.testclient import TestClient

from services.gateway import main as gateway_main


def test_gateway_serves_end_to_end_audio_test_page(monkeypatch) -> None:
    monkeypatch.setenv("NAWEI_ENV", "development")
    monkeypatch.setenv("GATEWAY_ENABLE_DIAGNOSTICS", "true")
    with TestClient(gateway_main.app) as client:
        response = client.get("/test")
        worklet = client.get("/test/pcm-capture-worklet.js")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"] == "no-store"
    assert worklet.status_code == 200
    assert worklet.headers["content-type"].startswith("application/javascript")
    assert "registerProcessor(\"pcm-capture-processor\"" in worklet.text
    assert "上传音频完整流程" in response.text
    assert "持续实时语音对话" in response.text
    assert "停顿都会自动转写并触发 AI 回复" in response.text
    assert "transcripts:new Map()" in response.text
    assert "appendLiveTranscript" in response.text
    assert "continuous_turns:true" in response.text
    assert "resumeLiveCaptureAfterPlayback" in response.text
    assert "up-asr" in response.text
    assert "live-ttfa" in response.text
    assert "/api/v1/voice/realtime" in response.text
    assert "/api/v1/audio/speech/stream" in response.text
    assert "/api/v1/text/completions/stream" in response.text
    assert "streamLlm" in response.text
    assert "requestAnimationFrame" in response.text
    assert "formatApiError" in response.text
    assert "references_hidden:true" in response.text
    assert "资料来源保留在文字回复中，但 TTS 不会朗读来源" in response.text
    assert "splitTtsText" in response.text
    assert "normalizeTtsText" in response.text
    assert "liveCaptureBufferSize=2048" in response.text
    assert "audioWorklet.addModule" in response.text
    assert "new window.AudioWorkletNode" in response.text
    assert "captureMode='audio-worklet'" in response.text
    assert "createStreamingTtsReferenceFilter" in response.text
    assert "extractStreamingTtsSegments" in response.text
    assert "ttsChain=Promise.resolve()" in response.text
    assert "llm_tts_parallel:true" in response.text


def test_gateway_hides_diagnostics_in_production(monkeypatch) -> None:
    monkeypatch.setenv("NAWEI_ENV", "production")
    monkeypatch.setenv("GATEWAY_ENABLE_DIAGNOSTICS", "true")

    with TestClient(gateway_main.app) as client:
        responses = [
            client.get(path)
            for path in ("/test", "/test/pcm-capture-worklet.js", "/docs", "/redoc", "/openapi.json")
        ]
        root = client.get("/")

    assert all(response.status_code == 404 for response in responses)
    assert "docs" not in root.json()
    assert "test" not in root.json()
