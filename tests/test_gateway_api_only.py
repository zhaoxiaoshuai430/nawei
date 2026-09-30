"""Robot ingress remains available after removing browser interfaces."""

import pytest
from fastapi.testclient import TestClient

from services.gateway import main as gateway
from services.gateway.xiaozhi import router


@pytest.mark.parametrize("environment", ["development", "production"])
def test_gateway_does_not_serve_browser_pages(monkeypatch, environment):
    monkeypatch.setenv("NAWEI_ENV", environment)
    monkeypatch.setenv("GATEWAY_ENABLE_DIAGNOSTICS", "true")
    monkeypatch.setenv("GATEWAY_STARTUP_WARMUP_ENABLED", "false")
    with TestClient(gateway.app) as client:
        for path in (
            "/chat", "/chat/config", "/chat/assets/chat.js", "/chat/assets/robot.jpg",
            "/devices", "/devices/config", "/test", "/test/pcm-capture-worklet.js",
            "/docs", "/redoc", "/docs/oauth2-redirect", "/api/v1/me",
        ):
            assert client.get(path).status_code == 404, path
        root = client.get("/")
        assert root.headers["content-type"] == "application/json"
        assert root.json()["robot_ota"] == "/xiaozhi/ota/"
        assert root.json()["robot_websocket"] == "/xiaozhi/v1/"
        if environment == "production":
            assert client.get("/openapi.json").status_code == 404
            assert "openapi" not in root.json()


def test_robot_can_activate_bind_and_connect_without_a_webpage(monkeypatch, tmp_path):
    monkeypatch.setenv("NAWEI_ENV", "development")
    monkeypatch.setenv("GATEWAY_AUTH_MODE", "local")
    monkeypatch.setenv("GATEWAY_STARTUP_WARMUP_ENABLED", "false")
    monkeypatch.setenv("XIAOZHI_ENABLED", "true")
    monkeypatch.setenv("XIAOZHI_DATABASE_PATH", str(tmp_path / "devices.sqlite3"))
    monkeypatch.setenv("XIAOZHI_PUBLIC_WEBSOCKET_URL", "ws://127.0.0.1:8000/xiaozhi/v1/")
    router.limiter.buckets.clear()
    identity = {"Device-Id": "aa:bb:cc:dd:ee:12", "Client-Id": "api-only-test-0001", "X-Device-Key": "a" * 64}
    with TestClient(gateway.app) as client:
        activation = client.post("/xiaozhi/ota/", headers=identity, json={})
        assert activation.status_code == 200
        binding = client.post("/api/v1/devices/bind", headers={"X-User-ID": "robot-owner"},
                              json={"code": activation.json()["activation"]["code"], "name": "Robot"})
        assert binding.status_code == 200
        config = client.post("/xiaozhi/ota/", headers=identity, json={}).json()["websocket"]
        headers = {**identity, "Authorization": "Bearer " + config["token"], "Protocol-Version": "1"}
        with client.websocket_connect("/xiaozhi/v1/", headers=headers) as socket:
            socket.send_json({"type": "hello", "version": 1, "transport": "websocket",
                              "audio_params": {"format": "opus", "sample_rate": 16000,
                                               "channels": 1, "frame_duration": 60}})
            assert socket.receive_json()["type"] == "hello"
