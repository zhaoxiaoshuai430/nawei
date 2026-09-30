"""Native decoder state, PCM continuity and WebSocket lifecycle regressions."""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
import threading

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


@pytest.fixture
def modules(monkeypatch):
    source = Path(__file__).resolve().parents[1] / "services/asr/src"
    monkeypatch.syspath_prepend(str(source))
    monkeypatch.setenv("NAWEI_ENV", "development")
    monkeypatch.setenv("ASR_INTERNAL_API_KEY", "")
    monkeypatch.setenv("ASR_REALTIME_SAMPLE_RATE", "16000")
    monkeypatch.setenv("ASR_REALTIME_RMS_THRESHOLD", "100")
    return tuple(importlib.import_module(name) for name in ("streaming_model", "streaming_realtime", "realtime"))


class Decoder:
    chunk_bytes = 19200

    def __init__(self):
        self.calls = []

    async def decode(self, pcm, cache, *, final):
        self.calls.append((pcm, cache, final))
        cache["count"] = cache.get("count", 0) + 1
        return "结束" if final else "你好"


@pytest.fixture
def model_pool(modules, monkeypatch, tmp_path):
    module = modules[0]
    original_model = module.ParaformerStreamingModel
    starts = []
    releases = []

    def create_model(path, *, device, threads):
        model = original_model.__new__(original_model)
        model.capacity = module.InferenceCapacity(limit=1, queue_limit=16, wait_timeout_seconds=1)
        index = len(starts)
        started, release = threading.Event(), threading.Event()
        starts.append(started)
        releases.append(release)

        def decode(pcm, cache, *, final):
            assert cache.setdefault("worker", index) == index
            started.set()
            if not release.wait(3):
                raise TimeoutError("Test did not release the model")
            return str(index)

        model._decode = decode
        return model

    monkeypatch.setattr(module, "ParaformerStreamingModel", create_model)
    pool = module.ParaformerStreamingPool(tmp_path, workers=2)
    yield pool, starts, releases
    for release in releases:
        release.set()


async def test_pool_runs_independent_models_concurrently_and_pins_sessions(model_pool):
    pool, starts, releases = model_pool
    with pool.session() as first, pool.session() as second:
        assert first is not second
        caches = [{}, {}]
        tasks = [asyncio.create_task(model.decode(b"", cache, final=False))
                 for model, cache in zip((first, second), caches)]
        try:
            async with asyncio.timeout(2):
                while not all(start.is_set() for start in starts):
                    await asyncio.sleep(0.01)
            assert pool.snapshot()["active"] == 2
            assert pool.snapshot()["sessions"] == 2
        finally:
            for release in releases:
                release.set()
            results = await asyncio.gather(*tasks)
        assert results == ["0", "1"]
        assert await first.decode(b"", caches[0], final=True) == "0"
        assert await second.decode(b"", caches[1], final=True) == "1"
    assert pool.snapshot()["sessions"] == pool.snapshot()["active"] == 0


@pytest.mark.parametrize("cancel_count", [1, 3])
async def test_cancelled_decode_keeps_worker_exclusive_until_thread_finishes(model_pool, cancel_count):
    pool, starts, releases = model_pool
    with pool.session() as model:
        first = asyncio.create_task(model.decode(b"", {}, final=False))
        second = None
        try:
            async with asyncio.timeout(2):
                while not starts[0].is_set():
                    await asyncio.sleep(0.01)
            for _ in range(cancel_count):
                first.cancel()
                await asyncio.sleep(0.01)
            assert not first.done(), "Cancellation released a running inference"
            second = asyncio.create_task(model.decode(b"", {}, final=True))
            async with asyncio.timeout(1):
                while model.capacity.snapshot().waiting != 1:
                    await asyncio.sleep(0.01)
            assert model.capacity.snapshot().active == 1
            assert not second.done()
        finally:
            releases[0].set()
            results = await asyncio.gather(first, *([second] if second else []), return_exceptions=True)
        assert isinstance(results[0], asyncio.CancelledError)
    assert pool.snapshot()["active"] == pool.snapshot()["waiting"] == 0


def test_pool_releases_session_assignment_on_error(model_pool):
    pool, _, _ = model_pool
    with pool.session() as first:
        with pytest.raises(RuntimeError):
            with pool.session() as second:
                assert second is not first
                raise RuntimeError("Disconnected")
        with pool.session() as replacement:
            assert replacement is second
    assert pool.snapshot()["sessions"] == 0


@pytest.mark.parametrize("workers", [0, -1])
def test_pool_rejects_invalid_worker_count(modules, tmp_path, workers):
    with pytest.raises(ValueError, match="ASR_STREAMING_WORKERS"):
        modules[0].ParaformerStreamingPool(tmp_path, workers=workers)


@pytest.fixture
def endpoint(modules):
    decoder = Decoder()
    app = FastAPI()
    app.state.streaming_finished = threading.Event()

    @app.websocket("/realtime")
    async def realtime(websocket: WebSocket):
        await modules[1].serve_streaming(websocket, decoder)
        app.state.streaming_finished.set()

    with TestClient(app) as client:
        yield client, decoder


def start(connection, **overrides):
    connection.send_json({"type": "start", "sample_rate": 16000, "channels": 1, **overrides})
    return connection.receive_json()


def receive_until(connection, kind):
    events = []
    for _ in range(30):
        event = connection.receive_json()
        events.append(event)
        if event["type"] == kind:
            return events
    raise AssertionError("Too many events before " + kind)


async def test_decoder_only_receives_new_audio_and_flushes_empty_tail(modules):
    decoder = Decoder()
    transcript = modules[0].StreamingTranscript(decoder)
    audio = b"\x01\x00" * 9600
    assert await transcript.feed(audio[:1000]) == []
    events = await transcript.feed(audio[1000:])
    assert events[0]["type"] == "transcript.partial"
    assert events[0]["text"] == "你好" and not events[0]["final"]
    cache = transcript.cache
    final = await transcript.finish()
    assert final["text"] == "你好结束" and final["final"]
    assert decoder.calls == [(audio, cache, False), (b"", cache, True)]
    assert transcript.cache is not cache and not transcript.cache


async def test_final_transcript_applies_contextual_correction_and_keeps_raw_text(modules):
    class CorrectionDecoder:
        chunk_bytes = 4

        async def decode(self, pcm, cache, *, final):
            return "请指回答结果" if final else ""

    transcript = modules[0].StreamingTranscript(CorrectionDecoder())
    await transcript.feed(b"\x01\x00\x01\x00")
    final = await transcript.finish()
    assert final["text"] == "请只回答结果"
    assert final["clean_text"] == "请只回答结果"
    assert final["raw_text"] == "请指回答结果"
    assert final["corrections"][0]["source"] == "请指回答结果"


async def test_short_audio_and_sessions_have_independent_caches(modules):
    decoder = Decoder()
    first = modules[0].StreamingTranscript(decoder)
    second = modules[0].StreamingTranscript(decoder)
    assert await first.feed(b"\x01\x00" * 50) == []
    await second.feed(bytes(decoder.chunk_bytes))
    assert (await first.finish())["text"] == "结束"
    assert decoder.calls[0][1] is not decoder.calls[1][1]
    assert decoder.calls[1][0] == b"\x01\x00" * 50
    with pytest.raises(ValueError):
        await first.feed(b"\x01")


def test_vad_native_audio_has_no_loss_or_duplication(modules):
    vad = modules[2]
    config = vad.RealtimeVadConfig(start_ms=40, end_silence_ms=60, min_speech_ms=40, rms_threshold=100)
    stream = vad.PCM16VadSegmenter(config, emit_chunks=True)
    voice = b"\xe8\x03" * 640
    silence = bytes(1920)
    events = stream.feed(voice[:700]) + stream.feed(voice[700:] + silence)
    assert b"".join(event.pcm for event in events if event.kind == "audio") == voice + silence
    assert [event.kind for event in events].count("segment") == 1
    assert next(event for event in events if event.kind == "segment").pcm == b""
    assert stream.flush() == []


def test_partial_arrives_while_client_is_still_speaking(endpoint):
    client, decoder = endpoint
    with client.websocket_connect("/realtime") as connection:
        ready = start(connection)
        assert ready["partial_results"] and ready["keeps_decoder_state"]
        connection.send_bytes(b"\xe8\x03" * 9600)
        events = receive_until(connection, "transcript.partial")
        assert not any(event["type"] == "transcript.final" for event in events)
        assert events[-1]["text"] == "你好"
        connection.send_bytes(b"\xe8\x03" * 500)
        connection.send_json({"type": "stop"})
        events = receive_until(connection, "session.done")
        final = next(event for event in events if event["type"] == "transcript.final")
        assert final["text"] == "你好结束"
        assert final["segment_index"] == 0
        assert len(b"".join(call[0] for call in decoder.calls)) == 20200
        assert decoder.calls[-1][2]


def test_vad_finalizes_each_turn_and_resets_cache(endpoint):
    client, decoder = endpoint
    with client.websocket_connect("/realtime") as connection:
        start(connection)
        for segment_index in range(2):
            connection.send_bytes(b"\xe8\x03" * 9600)
            connection.send_bytes(bytes(16000))
            events = receive_until(connection, "transcript.final")
            assert events[-1]["segment_index"] == segment_index
            assert events[-1]["text"] == "你好结束"
        connection.send_json({"type": "stop"})
        assert receive_until(connection, "session.done")[-1]["segments"] == 2
    assert decoder.calls[0][1] is not decoder.calls[2][1]


def test_silence_does_not_hallucinate_a_transcript(endpoint):
    client, decoder = endpoint
    with client.websocket_connect("/realtime") as connection:
        start(connection)
        connection.send_bytes(bytes(32000))
        connection.send_json({"type": "stop"})
        assert connection.receive_json() == {"type": "session.done", "segments": 0}
    assert not decoder.calls


@pytest.mark.parametrize("overrides", [{"sample_rate": 24000}, {"channels": 2}, {"language": "ja"}, {"type": "invalid"}])
def test_invalid_handshake_never_calls_model(endpoint, overrides):
    client, decoder = endpoint
    with client.websocket_connect("/realtime") as connection:
        assert start(connection, **overrides)["type"] == "error"
        with pytest.raises(WebSocketDisconnect):
            connection.receive_json()
    assert not decoder.calls


def test_internal_auth_is_required(endpoint, monkeypatch):
    monkeypatch.setenv("ASR_INTERNAL_API_KEY", "test-internal-key")
    client, decoder = endpoint
    with client.websocket_connect("/realtime") as connection:
        assert start(connection)["code"] == "unauthorized"
    assert not decoder.calls


@pytest.mark.parametrize("payload", ["[]", "invalid-json", " " * 4097], ids=["array", "malformed", "oversized"])
def test_first_command_is_bounded_json(endpoint, payload):
    client, decoder = endpoint
    with client.websocket_connect("/realtime") as connection:
        connection.send_text(payload)
        assert connection.receive_json()["code"] == "invalid_request"
    assert not decoder.calls


@pytest.mark.parametrize("packet", [b"x", bytes(32002)], ids=["incomplete-frame", "oversized-packet"])
def test_bad_audio_is_rejected(endpoint, packet):
    client, decoder = endpoint
    with client.websocket_connect("/realtime") as connection:
        start(connection)
        connection.send_bytes(packet)
        assert connection.receive_json()["type"] == "error"
    assert not decoder.calls


def test_inference_failure_terminates_instead_of_hanging(endpoint):
    client, decoder = endpoint

    async def fail(*args, **kwargs):
        raise RuntimeError("test decoder failure")

    decoder.decode = fail
    with client.websocket_connect("/realtime") as connection:
        start(connection)
        connection.send_bytes(b"\xe8\x03" * 9600)
        assert receive_until(connection, "error")[-1]["code"] == "recognition_failed"
        with pytest.raises(WebSocketDisconnect):
            connection.receive_json()


def test_disconnect_cancels_pending_inference(endpoint):
    client, decoder = endpoint
    cancelled = []
    started = threading.Event()

    async def wait(*args, **kwargs):
        try:
            started.set()
            await asyncio.sleep(60)
        finally:
            cancelled.append(True)

    decoder.decode = wait
    with client.websocket_connect("/realtime") as connection:
        start(connection)
        connection.send_bytes(b"\xe8\x03" * 9600)
        receive_until(connection, "speech.started")
        connection.send_json({"type": "ping"})
        receive_until(connection, "pong")
        assert started.wait(3), "Inference did not start"
        connection.close()
        assert client.app.state.streaming_finished.wait(3), "Disconnect cleanup did not finish"
    assert cancelled
