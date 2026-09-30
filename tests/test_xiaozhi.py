import asyncio
import base64
import json
import hmac
import math
import struct
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app_protocols import EventType, StreamEvent
from services.gateway import main as gateway
from services.gateway.xiaozhi import pipeline, router, websocket as xiaozhi_websocket
from services.gateway.xiaozhi.device_registry import DeviceRegistry
from services.gateway.xiaozhi.device_writer import DeviceWriter
from services.gateway.xiaozhi.opus_codec import OpusDecoder, OpusEncoder
from services.gateway.xiaozhi.session import DeviceSession


IDENTITY = {"Device-Id": "aa:bb:cc:dd:ee:01", "Client-Id": "test-client-00001", "X-Device-Key": "a" * 64}
HELLO = {"type": "hello", "version": 1, "transport": "websocket",
         "audio_params": {"format": "opus", "sample_rate": 16000, "channels": 1, "frame_duration": 60}}
USER = {"X-User-ID": "owner"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("XIAOZHI_ENABLED", "true")
    monkeypatch.setenv("XIAOZHI_DATABASE_PATH", str(tmp_path / "devices.sqlite3"))
    monkeypatch.setenv("GATEWAY_AUTH_MODE", "local")
    monkeypatch.setenv("NAWEI_ENV", "development")
    monkeypatch.setenv("XIAOZHI_HELLO_TIMEOUT_SECONDS", "0.2")
    router.limiter.buckets.clear()
    with TestClient(gateway.app) as test_client:
        yield test_client


def bound(client, identity=None):
    identity = identity or IDENTITY
    response = client.post("/xiaozhi/ota/", headers=identity, json={})
    assert response.status_code == 200
    code = response.json()["activation"]["code"]
    assert client.post("/api/v1/devices/bind", headers=USER, json={"code": code}).status_code == 200
    config = client.post("/xiaozhi/ota/", headers=identity, json={}).json()["websocket"]
    return {**identity, "Authorization": "Bearer " + config["token"], "Protocol-Version": "1"}, code


def test_activation_binding_and_persistence(client):
    first = client.post("/xiaozhi/ota/", headers=IDENTITY, json={})
    assert first.headers["cache-control"] == "no-store"
    assert "websocket" not in first.json()
    assert client.post("/xiaozhi/ota/activate", headers=IDENTITY).status_code == 202
    headers, code = bound(client)
    assert client.post("/xiaozhi/ota/activate", headers=IDENTITY).status_code == 200
    config = client.post("/xiaozhi/ota/", headers=IDENTITY).json()["websocket"]
    assert headers["Authorization"] == "Bearer " + config["token"]
    devices = client.get("/api/v1/devices", headers=USER).json()
    assert len(devices) == 1 and not devices[0]["online"]
    assert "token" not in json.dumps(devices) and "key_hash" not in json.dumps(devices)
    registry = DeviceRegistry(router.registry().path)
    assert registry.authenticate(IDENTITY["Device-Id"], IDENTITY["Client-Id"], config["token"])["user_id"] == "owner"
    assert client.post("/api/v1/devices/bind", headers=USER, json={"code": code}).status_code == 409


def test_invalid_expired_activation_and_identity_theft(client):
    result = client.post("/xiaozhi/ota/", headers=IDENTITY).json()
    code = result["activation"]["code"]
    with router.registry().transaction() as connection:
        connection.execute("UPDATE activations SET expires_at=? WHERE code=?", (time.time() - 1, code))
    assert client.post("/api/v1/devices/bind", headers=USER, json={"code": code}).status_code == 410
    assert client.post("/api/v1/devices/bind", headers=USER, json={"code": "999999"}).status_code == 404
    impostor = {**IDENTITY, "X-Device-Key": "b" * 64}
    assert client.post("/xiaozhi/ota/", headers=impostor).status_code == 401
    assert client.post("/xiaozhi/ota/", headers={"Device-Id": IDENTITY["Device-Id"], "Client-Id": IDENTITY["Client-Id"]}).status_code == 401


def test_explicit_legacy_device_mode_supports_firmware_without_device_key(client, monkeypatch):
    identity = {"Device-Id": "14:c1:9f:39:e5:74", "Client-Id": "legacy-client-00001"}
    monkeypatch.setenv("XIAOZHI_ALLOW_LEGACY_DEVICES", "true")
    monkeypatch.setenv("XIAOZHI_LEGACY_DEVICE_SECRET", "legacy-test-secret-that-is-at-least-32-characters")
    first = client.post("/xiaozhi/ota/", headers=identity, json={})
    assert first.status_code == 200
    assert "activation" in first.json()
    assert client.post("/xiaozhi/ota/", headers=identity, json={}).status_code == 200
    assert client.post("/xiaozhi/ota/", headers={**identity, "Client-Id": "legacy-client-00002"}, json={}).status_code == 401


def test_ownership_unbinding_and_disabled_device(client):
    headers, _ = bound(client)
    path = "/api/v1/devices/" + IDENTITY["Device-Id"]
    assert client.get(path, headers={"X-User-ID": "other"}).status_code == 404
    assert client.delete(path + "/binding", headers={"X-User-ID": "other"}).status_code == 404
    assert client.patch(path, headers=USER, json={"voice": "other", "role": "tutor"}).status_code == 200
    assert client.patch(path, headers=USER, json={"enabled": False}).status_code == 200
    assert client.post("/xiaozhi/ota/", headers=IDENTITY).status_code == 403
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/xiaozhi/v1/", headers=headers):
            pass
    assert client.patch(path, headers=USER, json={"enabled": True}).status_code == 200
    assert client.delete(path + "/binding", headers=USER).status_code == 204
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/xiaozhi/v1/", headers=headers):
            pass
    replacement, _ = bound(client)
    assert replacement["Authorization"] != headers["Authorization"]


@pytest.mark.parametrize("override", [{"Authorization": "Bearer invalid"}, {"Device-Id": "aa:bb:cc:dd:ee:02"},
                                      {"Client-Id": "test-client-00002"}, {"Protocol-Version": "2"}])
def test_websocket_rejects_wrong_identity(client, override):
    headers, _ = bound(client)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/xiaozhi/v1/", headers={**headers, **override}):
            pass


def test_hello_reconnect_disable_and_limits(client):
    headers, _ = bound(client)
    with client.websocket_connect("/xiaozhi/v1/", headers=headers) as first:
        first.send_json(HELLO)
        hello = first.receive_json()
        assert hello["audio_params"]["sample_rate"] == 24000
        assert hello["features"] == {"barge_in": True, "full_duplex_transport": True}
        assert client.get("/api/v1/devices", headers=USER).json()[0]["online"]
        with client.websocket_connect("/xiaozhi/v1/", headers=headers) as second:
            second.send_json(HELLO)
            assert second.receive_json()["session_id"] != hello["session_id"]
            with pytest.raises(WebSocketDisconnect):
                first.receive_json()
            second.send_json({"type": "ping"})
            assert second.receive_json()["type"] == "pong"
            client.patch("/api/v1/devices/" + IDENTITY["Device-Id"], headers=USER, json={"enabled": False})
            with pytest.raises(WebSocketDisconnect):
                second.receive_json()
    client.patch("/api/v1/devices/" + IDENTITY["Device-Id"], headers=USER, json={"enabled": True})
    with client.websocket_connect("/xiaozhi/v1/", headers=headers) as connection:
        with pytest.raises(WebSocketDisconnect):
            connection.receive_json()
    with client.websocket_connect("/xiaozhi/v1/", headers=headers) as connection:
        connection.send_json(HELLO)
        connection.receive_json()
        connection.send_bytes(bytes(1276))
        with pytest.raises(WebSocketDisconnect):
            connection.receive_json()


@pytest.mark.parametrize("rate", [16000, 24000])
def test_real_opus_roundtrip_and_resampling(rate):
    encoder = OpusEncoder(rate)
    pcm = b"".join(struct.pack("<h", int(4000 * math.sin(index * 2 * math.pi * 440 / 24000))) for index in range(24000))
    packets = encoder.feed(pcm[:12000], 24000) + encoder.feed(pcm[12000:], 24000) + encoder.finish()
    assert 16 <= len(packets) <= 19 and all(0 < len(packet) <= 1275 for packet in packets)
    decoder = OpusDecoder(rate)
    decoded = b"".join(decoder.decode(packet) for packet in packets)
    samples = struct.unpack("<" + "h" * (len(decoded) // 2), decoded)
    assert 0.95 <= len(samples) / rate <= 1.15
    assert max(samples) > 1000


async def test_writer_drops_old_generation():
    messages = []

    async def send(payload):
        messages.append(payload)

    writer = DeviceWriter(SimpleNamespace(send_json=send, send_bytes=send))
    await writer.send(b"old")
    writer.invalidate()
    await writer.send({"type": "tts", "state": "stop"})
    worker = asyncio.create_task(writer.run())
    try:
        await writer.flush()
        assert messages == [{"type": "tts", "state": "stop"}]
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


async def test_idle_robot_waits_for_transport_disconnect(monkeypatch):
    wait_for = asyncio.wait_for

    async def accelerated_wait(awaitable, timeout):
        return await wait_for(awaitable, 0.01 if timeout == 300 else timeout)

    async def receive():
        await asyncio.sleep(0.03)
        return {"type": "websocket.disconnect"}

    monkeypatch.setattr(asyncio, "wait_for", accelerated_wait)
    await asyncio.wait_for(
        xiaozhi_websocket._receive(SimpleNamespace(receive=receive), None, None, "client", "token"),
        timeout=1,
    )


@pytest.mark.parametrize("completed", [False, True])
async def test_listen_start_only_interrupts_unfinished_robot_output(monkeypatch, completed):
    sent = []
    old_turn_cancelled = asyncio.Event()
    new_turn_started = asyncio.Event()

    class Writer:
        generation = 0
        turn_complete = completed

        def invalidate(self):
            self.generation += 1

        async def send(self, payload, generation=None):
            sent.append(payload)

    class WebSocket:
        def __init__(self):
            self.messages = [
                {"type": "websocket.receive", "text": json.dumps({"type": "listen", "state": "start", "mode": "auto"})},
                {"type": "websocket.disconnect"},
            ]

        async def receive(self):
            return self.messages.pop(0)

    class Registry:
        def authenticate(self, device, client, token):
            return {"device_id": device, "client_id": client, "user_id": "owner", "voice": "Nana", "role": "student"}

    async def old_turn():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            old_turn_cancelled.set()
            raise

    async def new_turn(session, mode):
        new_turn_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(xiaozhi_websocket, "run_turn", new_turn)
    session = DeviceSession(
        {"device_id": "device", "client_id": "client", "user_id": "owner", "voice": "Nana", "role": "student"},
        Writer(),
        24000,
    )
    session.turn = asyncio.create_task(old_turn())
    await asyncio.sleep(0)
    await xiaozhi_websocket._receive(WebSocket(), session, Registry(), "client", "token")

    assert old_turn_cancelled.is_set()
    if completed:
        assert sent == []
    else:
        assert sent[0]["type"] == "tts"
        assert sent[0]["state"] == "stop"
        assert sent[0]["reason"] == "barge_in"
    await asyncio.wait_for(new_turn_started.wait(), timeout=1)
    session.turn.cancel()
    await asyncio.gather(session.turn, return_exceptions=True)


async def test_writer_marks_completion_before_peer_can_start_next_turn():
    delivered = asyncio.Event()
    release_send = asyncio.Event()

    async def send(payload):
        delivered.set()
        await release_send.wait()

    writer = DeviceWriter(SimpleNamespace(send_json=send))
    await writer.send({"type": "tts", "state": "stop"})
    assert not writer.turn_complete
    worker = asyncio.create_task(writer.run())
    try:
        await asyncio.wait_for(delivered.wait(), 1)
        assert writer.turn_complete
        writer.invalidate()
        assert not writer.turn_complete
        release_send.set()
        await writer.flush()
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


@pytest.mark.parametrize("audio_started", [False, True])
async def test_consecutive_barge_in_stops_each_active_turn(monkeypatch, audio_started):
    sent = []
    incoming = asyncio.Queue()
    started = asyncio.Queue()

    async def send(payload):
        sent.append(payload)

    writer = DeviceWriter(SimpleNamespace(send_json=send, send_bytes=send))
    session = DeviceSession({"device_id": "device"}, writer, 24000)

    async def turn(session, mode):
        if audio_started:
            await writer.send({"type": "tts", "state": "start"})
        started.put_nowait(True)
        await asyncio.Event().wait()

    monkeypatch.setattr(xiaozhi_websocket, "run_turn", turn)
    registry = SimpleNamespace(authenticate=lambda *args: session.device)
    worker = asyncio.create_task(writer.run())
    session.turn = asyncio.create_task(turn(session, "manual"))
    reader = asyncio.create_task(xiaozhi_websocket._receive(
        SimpleNamespace(receive=incoming.get), session, registry, "client", "token",
    ))
    try:
        async with asyncio.timeout(3):
            await started.get()
            for _ in range(2):
                incoming.put_nowait({"type": "websocket.receive", "text": json.dumps({"type": "listen", "state": "start"})})
                await started.get()
                await writer.flush()
                assert not writer.turn_complete
            assert sum(item.get("reason") == "barge_in" for item in sent) == 2
    finally:
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
        await session.cancel_turn()
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


@pytest.mark.parametrize("origin, expected", [
    ("http://127.0.0.1:8000", "ws://127.0.0.1:8000/xiaozhi/v1/"),
    ("http://192.168.10.177:8000", "ws://192.168.10.177:8000/xiaozhi/v1/"),
    ("https://voice.example.test", "wss://voice.example.test/xiaozhi/v1/"),
    ("http://[::1]:8000", "ws://[::1]:8000/xiaozhi/v1/"),
])
def test_local_auto_discovery_follows_request_origin(client, monkeypatch, origin, expected):
    monkeypatch.setenv("NAWEI_PUBLIC_ROBOT_URL", "auto")
    monkeypatch.setenv("XIAOZHI_PUBLIC_WEBSOCKET_URL", "auto")
    bound(client)
    # This Starlette TestClient version cannot parse an IPv6 transport netloc.
    request_origin = "http://testserver" if "[" in origin else origin
    result = client.post(request_origin + "/xiaozhi/ota/?unused=1",
                         headers={**IDENTITY, "Host": urlsplit(origin).netloc}).json()
    assert result["websocket"]["url"] == expected


def test_production_requires_explicit_robot_endpoint(monkeypatch):
    from services.gateway.xiaozhi.config import Settings

    monkeypatch.setenv("NAWEI_ENV", "production")
    monkeypatch.setenv("XIAOZHI_ENABLED", "true")
    monkeypatch.setenv("GATEWAY_AUTH_MODE", "jwt")
    monkeypatch.setenv("XIAOZHI_PUBLIC_WEBSOCKET_URL", "auto")
    with pytest.raises(ValueError, match="WSS"):
        Settings.load()


def test_twenty_voice_turns_use_bound_owner_and_opus(client, monkeypatch):
    requests = []

    async def transcribe(session, mode):
        assert await session.audio.get() == b"simulated-opus"
        assert await session.audio.get() is None
        return "Hello Nawei"

    async def response(request, message):
        requests.append(request)
        for kind, data in (
            (EventType.AUDIO_STARTED, {}),
            (EventType.AUDIO_DELTA, {"audio_base64": base64.b64encode(bytes(2880)).decode(),
                                    "format": "pcm16", "sample_rate": 24000, "channels": 1,
                                    "segment_index": 0, "text": "Hello"}),
            (EventType.AUDIO_DONE, {}),
        ):
            yield StreamEvent(event=kind, request_id=request.request_id, trace_id=request.trace_id,
                              session_id=request.session_id, data=data)

    monkeypatch.setattr(pipeline, "transcribe", transcribe)
    monkeypatch.setattr(gateway, "_stream_voice_response", response)
    headers, _ = bound(client)
    with client.websocket_connect("/xiaozhi/v1/", headers=headers) as connection:
        connection.send_json(HELLO)
        connection.receive_json()
        for turn in range(20):
            connection.send_json({"type": "listen", "state": "start", "mode": "manual"})
            connection.send_bytes(b"simulated-opus")
            connection.send_json({"type": "listen", "state": "stop"})
            events = []
            while True:
                message = connection.receive()
                assert message["type"] == "websocket.send"
                if "bytes" in message:
                    events.append("audio")
                else:
                    body = json.loads(message["text"])
                    events.append(body.get("state", body["type"]))
                    if body.get("state") == "stop":
                        break
            assert events.index("stt") < events.index("start") < events.index("audio") < events.index("stop")
        assert len(requests) == 20
        assert {request.user_id for request in requests} == {"owner"}
        assert len({request.request_id for request in requests}) == 20
        assert len({request.session_id for request in requests}) == 1


def test_body_limits_and_bind_rate_limit(client):
    assert client.post("/xiaozhi/ota/", headers=IDENTITY, content=bytes(16385)).status_code == 413
    for attempt in range(10):
        assert client.post("/api/v1/devices/bind", headers=USER, json={"code": "000000"}).status_code == 404
    assert client.post("/api/v1/devices/bind", headers=USER, json={"code": "000000"}).status_code == 429


async def test_real_opus_reaches_asr_pcm_and_cleans_up(monkeypatch):
    from websockets.asyncio.server import serve

    received = bytearray()
    finished = asyncio.Event()

    async def asr(upstream):
        try:
            start = json.loads(await upstream.recv())
            assert start["sample_rate"] == 16000 and start["channels"] == 1
            async for message in upstream:
                if isinstance(message, bytes):
                    received.extend(message)
                elif json.loads(message)["type"] == "stop":
                    await upstream.send(json.dumps({"type": "transcript.final", "text": "audio received"}))
                    await upstream.send(json.dumps({"type": "session.done"}))
                    return
        finally:
            finished.set()

    async with serve(asr, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        monkeypatch.setenv("GATEWAY_ASR_REALTIME_URL", f"ws://127.0.0.1:{port}")
        session = DeviceSession({}, DeviceWriter(None), 24000)
        encoder = OpusEncoder(16000)
        packets = encoder.feed(bytes(1920 * 3), 16000) + encoder.finish()
        for packet in packets:
            await session.audio.put(packet)
        await session.audio.put(None)
        assert await asyncio.wait_for(pipeline.transcribe(session, "manual"), 5) == "audio received"
        await asyncio.wait_for(finished.wait(), 2)
        assert len(received) >= 1920 * 3 - 64


async def test_cancel_backpressured_voice_producers(monkeypatch):
    from app_protocols import Message, MessageRole

    async def llm(request):
        for index in range(100):
            yield StreamEvent(event=EventType.MESSAGE_DELTA, request_id=request.request_id,
                              trace_id=request.trace_id, session_id=request.session_id, data={"text": "Hello."})

    monkeypatch.setattr(gateway.orchestrator.llm_provider, "stream_complete", llm)
    monkeypatch.setenv("GATEWAY_VOICE_EVENT_QUEUE_SIZE", "1")
    monkeypatch.setenv("GATEWAY_VOICE_TTS_SEGMENT_QUEUE_SIZE", "1")
    request = gateway.RealtimeVoiceRequest(request_id="cancel", trace_id="cancel", session_id="cancel", user_id="owner")
    stream = gateway._stream_voice_response(request, Message(role=MessageRole.USER, content="Hello"))
    await anext(stream)
    await asyncio.sleep(0.05)
    await asyncio.wait_for(stream.aclose(), 2)


async def test_robot_partial_snapshots_preserve_previous_final_segments(monkeypatch):
    from websockets.asyncio.server import serve

    seen = []

    async def send(payload):
        seen.append(payload)

    async def asr(upstream):
        await upstream.recv()
        await upstream.send(json.dumps({"type": "transcript.partial", "segment_index": 0, "text": "第一"}))
        await upstream.send(json.dumps({"type": "transcript.final", "segment_index": 0, "text": "第一句"}))
        await upstream.send(json.dumps({"type": "transcript.partial", "segment_index": 1, "text": "第二"}))
        await upstream.send(json.dumps({"type": "transcript.final", "segment_index": 1, "text": "第二句"}))
        await upstream.send(json.dumps({"type": "session.done"}))

    async with serve(asr, "127.0.0.1", 0) as server:
        monkeypatch.setenv("GATEWAY_ASR_REALTIME_URL", f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}")
        session = DeviceSession({}, SimpleNamespace(send=send), 24000)
        result = await asyncio.wait_for(pipeline.transcribe(session, "manual"), 5)
    assert result == "第一句 第二句"
    assert [event["text"] for event in seen] == ["第一", "第一句 第二"]
    assert all(event["final"] is False for event in seen)


def test_owner_can_release_device_for_nvs_reset(client):
    headers, _ = bound(client)
    replacement = {**IDENTITY, "Client-Id": "replacement-client-001", "X-Device-Key": "c" * 64}
    assert client.post("/xiaozhi/ota/", headers=replacement).status_code == 401
    client.delete("/api/v1/devices/" + IDENTITY["Device-Id"] + "/binding", headers=USER)
    new_headers, _ = bound(client, replacement)
    assert headers["Authorization"] != new_headers["Authorization"]


def test_jwt_user_and_device_credentials_are_separate(client, monkeypatch):
    from deploy.scripts.create_jwt import _base64url, _encode_json

    headers, _ = bound(client)
    second_headers, _ = bound(client, {**IDENTITY, "Device-Id": "aa:bb:cc:dd:ee:02"})
    assert headers["Authorization"] != second_headers["Authorization"]
    secret = "test-user-signing-secret-0123456789abcdef"
    monkeypatch.setenv("GATEWAY_AUTH_MODE", "jwt")
    monkeypatch.setenv("GATEWAY_JWT_SECRET", secret)
    monkeypatch.delenv("GATEWAY_JWT_ISSUER", raising=False)
    monkeypatch.delenv("GATEWAY_JWT_AUDIENCE", raising=False)
    signing_input = b".".join((_encode_json({"alg": "HS256"}), _encode_json({"sub": "owner", "exp": time.time() + 60})))
    signature = _base64url(hmac.new(secret.encode(), signing_input, "sha256").digest())
    token = signing_input.decode() + "." + signature
    assert client.get("/api/v1/devices", headers={"Authorization": "Bearer " + token}).status_code == 200
    assert client.get("/api/v1/devices", headers=headers).status_code == 401
    assert client.get("/api/v1/devices", headers=USER).status_code == 401
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/xiaozhi/v1/", headers={**headers, "Authorization": "Bearer " + token}):
            pass
