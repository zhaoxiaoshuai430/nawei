"""A disconnected ASGI stream must finish its worker and return capacity."""

import asyncio
import importlib
import io
import json
from pathlib import Path
import sys
import threading
import types
from unittest.mock import Mock

import pytest
from fastapi import HTTPException, UploadFile
from starlette.requests import ClientDisconnect
from starlette.requests import Request


@pytest.fixture
def handlers(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "services/tts"))
    model_module = types.ModuleType("faster_qwen3_tts")
    model_module.FasterQwen3TTS = object
    monkeypatch.setitem(sys.modules, "faster_qwen3_tts", model_module)
    audio_module = types.ModuleType("app.utils.audio")
    audio_module.concat_audio = lambda audio: audio
    audio_module.encode_audio_b64 = lambda *args, **kwargs: "AAAA"
    monkeypatch.setitem(sys.modules, "app.utils.audio", audio_module)
    original_modules = set(sys.modules)
    try:
        yield importlib.import_module("app.services.tts.handlers")
    finally:
        for name in set(sys.modules) - original_modules:
            if name == "app" or name.startswith("app."):
                module = sys.modules.pop(name, None)
                parent_name, _, attribute = name.rpartition(".")
                parent = sys.modules.get(parent_name)
                if parent is not None and getattr(parent, attribute, None) is module:
                    delattr(parent, attribute)


@pytest.mark.parametrize("provided", [None, "wrong", "\u00e9"])
def test_remote_model_load_rejects_invalid_admin_key(handlers, monkeypatch, provided):
    engine = types.ModuleType("app.services.tts.engine")
    engine.load_model_into_cache = Mock()
    monkeypatch.setitem(sys.modules, "app.services.tts.engine", engine)
    routes = importlib.import_module("app.api.routes")
    monkeypatch.setenv("TTS_ADMIN_API_KEY", "valid-admin-key")
    request = Request({"type": "http", "client": ("192.0.2.1", 1234)})
    with pytest.raises(HTTPException) as caught:
        routes._authorize_model_load(request, provided)
    assert caught.value.status_code == 403
    routes._authorize_model_load(request, "valid-admin-key")


@pytest.mark.parametrize("source", ["upload", "preset", "emotion"])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("failed", [False, True])
async def test_generation_preserves_shared_reference(handlers, monkeypatch, tmp_path, source, streaming, failed):
    reference = tmp_path / "shared.wav"
    content = b"shared-reference"
    reference.write_bytes(content)
    preset = {"path": str(reference), "ref_text": "reference transcript"}
    monkeypatch.setattr(handlers, "preset_refs", {"test": preset} if source == "preset" else {})
    monkeypatch.setattr(handlers.emotion_refs, "emotion_refs", {"test": preset} if source == "emotion" else {})
    monkeypatch.setattr(handlers, "get_cached_ref_path", lambda data: str(reference))
    calls = []

    class Model:
        def generate_voice_clone(self, **kwargs):
            calls.append(kwargs)
            assert reference.read_bytes() == content
            if failed:
                raise RuntimeError("generation failed")
            return bytes(100), 24000

        def generate_voice_clone_streaming(self, **kwargs):
            audio, rate = self.generate_voice_clone(**kwargs)
            yield audio, rate, {}

    monkeypatch.setattr(handlers.state, "active_model_name", "test")
    monkeypatch.setattr(handlers.state, "model_cache", {"test": Model()})
    upload = UploadFile(io.BytesIO(content), filename="reference.wav") if source == "upload" else None
    options = dict(text="test", language="English", mode="voice_clone", speaker="", ref_text="",
                   instruct="", xvec_only=True, temperature=0.9, top_k=50, repetition_penalty=1.05,
                   ref_preset="" if source == "upload" else "test", ref_audio=upload)
    try:
        if streaming:
            response = await handlers.generate_stream(chunk_size=25, **options)
            async with asyncio.timeout(3):
                events = "".join([event async for event in response.body_iterator])
            assert ('"type": "error"' if failed else '"type": "done"') in events
        elif failed:
            with pytest.raises(RuntimeError, match="generation failed"):
                await handlers.generate_non_streaming(**options)
        else:
            assert (await handlers.generate_non_streaming(**options)).status_code == 200
        assert calls[0]["ref_audio"] == str(reference)
        assert calls[0]["ref_text"] == ("" if source == "upload" else "reference transcript")
        assert reference.read_bytes() == content
        capacity = handlers.state.generation_capacity.snapshot()
        assert capacity.active == capacity.waiting == 0
    finally:
        if upload:
            await upload.close()


@pytest.mark.parametrize("content,accepted", [(b"", False), (b"1234", True), (b"12345", False)])
async def test_reference_upload_reads_only_limit_probe(handlers, monkeypatch, content, accepted):
    class Upload:
        filename = "reference.wav"
        requested = None

        async def read(self, size=-1):
            self.requested = size
            return content if size < 0 else content[:size]

    monkeypatch.setattr(handlers, "MAX_AUDIO_BYTES", 4)
    cached = Mock(return_value="cached.wav")
    monkeypatch.setattr(handlers, "get_cached_ref_path", cached)
    upload = Upload()
    if accepted:
        result = await handlers.resolve_ref_audio(ref_preset="", ref_audio=upload, ref_text="", required=True)
        assert result.path == "cached.wav"
        cached.assert_called_once_with(content)
    else:
        with pytest.raises(HTTPException) as caught:
            await handlers.resolve_ref_audio(ref_preset="", ref_audio=upload, ref_text="", required=True)
        assert caught.value.status_code == 400
        cached.assert_not_called()
    assert upload.requested == 5


async def test_reference_cache_write_runs_off_event_loop(handlers, monkeypatch):
    loop_thread = threading.get_ident()

    def cache_reference(content):
        assert threading.get_ident() != loop_thread
        return "cached.wav"

    monkeypatch.setattr(handlers, "get_cached_ref_path", cache_reference)
    upload = UploadFile(io.BytesIO(b"audio"), filename="reference.wav")
    try:
        result = await handlers.resolve_ref_audio(ref_preset="", ref_audio=upload, ref_text="")
        assert result.path == "cached.wav"
    finally:
        await upload.close()


async def test_cancelled_reference_resolution_waits_for_cache_writer(handlers, monkeypatch):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def cache_reference(content):
        entered.set()
        assert release.wait(3)
        finished.set()
        return "cached.wav"

    monkeypatch.setattr(handlers, "get_cached_ref_path", cache_reference)
    upload = UploadFile(io.BytesIO(b"audio"), filename="reference.wav")
    task = asyncio.create_task(handlers.resolve_ref_audio(ref_preset="", ref_audio=upload, ref_text=""))
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        assert finished.is_set()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await upload.close()


@pytest.mark.parametrize("disconnect_during_send", [False, True])
async def test_disconnect_waits_for_worker_then_releases_capacity(handlers, monkeypatch, disconnect_during_send):
    capacity = handlers.state.generation_capacity
    release_worker = threading.Event()
    second_chunk_started = threading.Event()
    first_chunk = asyncio.Event()
    disconnected = asyncio.Event()

    class Model:
        def generate_custom_voice_streaming(self, **kwargs):
            yield bytes(100), 24000, {}
            second_chunk_started.set()
            if not release_worker.wait(3):
                raise TimeoutError("Test did not release synthesis worker")
            yield bytes(100), 24000, {}

    monkeypatch.setattr(handlers.state, "active_model_name", "test")
    monkeypatch.setattr(handlers.state, "model_cache", {"test": Model()})
    response = await handlers.generate_stream(
        text="test", language="English", mode="custom", speaker="test", ref_text="",
        instruct="", xvec_only=True, chunk_size=25, temperature=0.9, top_k=50,
        repetition_penalty=1.05, ref_preset="", ref_audio=None,
    )

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            first_chunk.set()
            if disconnect_during_send:
                await asyncio.Event().wait()

    async def receive():
        await first_chunk.wait()
        assert await asyncio.to_thread(second_chunk_started.wait, 1)
        disconnected.set()
        return {"type": "http.disconnect"}

    async def finish_worker():
        await disconnected.wait()
        await asyncio.sleep(0.05)
        assert capacity.snapshot().active == 1
        release_worker.set()

    finishing = asyncio.create_task(finish_worker())
    try:
        async with asyncio.timeout(3):
            await response({"type": "http", "asgi": {"spec_version": "2.3"}}, receive, send)
            await finishing
        assert capacity.snapshot().active == capacity.snapshot().waiting == 0
        async with asyncio.timeout(1):
            async with await capacity.acquire():
                assert capacity.snapshot().active == 1
    finally:
        release_worker.set()
        finishing.cancel()
        await asyncio.gather(finishing, return_exceptions=True)


@pytest.mark.parametrize("close_mode", ["iterator", "acquired", "disconnect", "send_error"])
async def test_queued_stream_close_cleans_up_pending_or_acquired_lease(handlers, monkeypatch, close_mode):
    capacity = handlers.state.generation_capacity
    holder = await capacity.acquire()
    monkeypatch.setattr(handlers.state, "active_model_name", "test")
    monkeypatch.setattr(handlers.state, "model_cache", {"test": object()})
    response = await handlers.generate_stream(
        text="test", language="English", mode="custom", speaker="test", ref_text="",
        instruct="", xvec_only=True, chunk_size=25, temperature=0.9, top_k=50,
        repetition_penalty=1.05, ref_preset="", ref_audio=None,
    )
    sending = asyncio.Event()

    async def send(message):
        if message["type"] == "http.response.body":
            assert b'"queued"' in message["body"]
            sending.set()
            if close_mode == "send_error":
                raise OSError("Client disconnected during send")
            await asyncio.Event().wait()

    async def receive():
        await sending.wait()
        return {"type": "http.disconnect"}

    existing_tasks = asyncio.all_tasks()
    try:
        async with asyncio.timeout(2):
            if close_mode in {"iterator", "acquired"}:
                assert '"queued"' in await anext(response.body_iterator)
                if close_mode == "acquired":
                    await holder.release()
                    while capacity.snapshot().waiting:
                        await asyncio.sleep(0)
                await response.body_iterator.aclose()
            elif close_mode == "send_error":
                with pytest.raises(ClientDisconnect):
                    await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
            else:
                await response({"type": "http", "asgi": {"spec_version": "2.3"}}, receive, send)
            assert capacity.snapshot().waiting == 0
            await holder.release()
            assert capacity.snapshot().active == 0
            async with await capacity.acquire():
                assert capacity.snapshot().active == 1
    finally:
        await response.body_iterator.aclose()
        pending = asyncio.all_tasks() - existing_tasks
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        await holder.release()


async def test_disconnect_during_first_chunk_does_not_request_second(handlers, monkeypatch):
    started, release, closed = threading.Event(), threading.Event(), threading.Event()
    iterations = []

    def audio():
        try:
            started.set()
            assert release.wait(3)
            iterations.append(1)
            yield bytes(100), 24000, {}
            iterations.append(2)
            yield bytes(100), 24000, {}
        finally:
            closed.set()

    generator = audio()  # Retain it so garbage collection cannot hide missing close().
    model = types.SimpleNamespace(generate_custom_voice_streaming=lambda **kwargs: generator)
    monkeypatch.setattr(handlers.state, "active_model_name", "test")
    monkeypatch.setattr(handlers.state, "model_cache", {"test": model})
    response = await handlers.generate_stream(
        text="test", language="English", mode="custom", speaker="test", ref_text="",
        instruct="", xvec_only=True, chunk_size=25, temperature=0.9, top_k=50,
        repetition_penalty=1.05, ref_preset="", ref_audio=None,
    )
    task = asyncio.create_task(anext(response.body_iterator))
    try:
        assert await asyncio.to_thread(started.wait, 1)
        task.cancel()
        await asyncio.sleep(0.01)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        assert iterations == [1]
        assert closed.is_set()
        assert handlers.state.generation_capacity.snapshot().active == 0
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await response.body_iterator.aclose()
        generator.close()


async def test_encoding_failure_closes_retained_model_generator(handlers, monkeypatch):
    closed = threading.Event()

    def audio():
        try:
            yield bytes(100), 24000, {}
            yield bytes(100), 24000, {}
        finally:
            closed.set()

    generator = audio()
    monkeypatch.setattr(handlers.state, "active_model_name", "test")
    monkeypatch.setattr(handlers.state, "model_cache", {"test": types.SimpleNamespace(
        generate_custom_voice_streaming=lambda **kwargs: generator)})

    def fail_encoding(*args, **kwargs):
        raise ValueError("encoding failed")

    monkeypatch.setattr(handlers, "encode_audio_b64", fail_encoding)
    response = await handlers.generate_stream(
        text="test", language="English", mode="custom", speaker="test", ref_text="",
        instruct="", xvec_only=True, chunk_size=25, temperature=0.9, top_k=50,
        repetition_penalty=1.05, ref_preset="", ref_audio=None,
    )
    try:
        async with asyncio.timeout(2):
            events = "".join([event async for event in response.body_iterator])
        assert "encoding failed" in events
        assert closed.is_set()
        assert handlers.state.generation_capacity.snapshot().active == 0
    finally:
        await response.body_iterator.aclose()
        generator.close()


async def test_full_queue_disconnect_leaves_no_completion_put_task(handlers, monkeypatch):
    monkeypatch.setenv("TTS_STREAM_QUEUE_SIZE", "1")
    monkeypatch.setenv("TTS_STREAM_QUEUE_PUT_TIMEOUT_SECONDS", "0.05")
    closed = threading.Event()

    def audio(**kwargs):
        try:
            for _ in range(10):
                yield bytes(100), 24000, {}
        finally:
            closed.set()

    monkeypatch.setattr(handlers.state, "active_model_name", "test")
    monkeypatch.setattr(handlers.state, "model_cache", {"test": types.SimpleNamespace(generate_custom_voice_streaming=audio)})
    response = await handlers.generate_stream(
        text="test", language="English", mode="custom", speaker="test", ref_text="",
        instruct="", xvec_only=True, chunk_size=25, temperature=0.9, top_k=50,
        repetition_penalty=1.05, ref_preset="", ref_audio=None,
    )
    existing = asyncio.all_tasks()
    try:
        assert '"chunk"' in await asyncio.wait_for(anext(response.body_iterator), 1)
        assert await asyncio.to_thread(closed.wait, 1)
        await response.body_iterator.aclose()
        await asyncio.sleep(0)
        assert not (asyncio.all_tasks() - existing)
        assert handlers.state.generation_capacity.snapshot().active == 0
    finally:
        await response.body_iterator.aclose()
        pending = asyncio.all_tasks() - existing
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


@pytest.mark.parametrize("chunk_count", [0, 4])
async def test_completed_stream_drains_all_queued_events(handlers, monkeypatch, chunk_count):
    monkeypatch.setenv("TTS_STREAM_QUEUE_SIZE", "8")
    closed = threading.Event()

    def audio(**kwargs):
        try:
            for index in range(chunk_count):
                yield bytes(index + 1), 24000, {}
        finally:
            closed.set()

    monkeypatch.setattr(handlers, "encode_audio_b64", lambda audio, *args, **kwargs: str(len(audio)))
    monkeypatch.setattr(handlers.state, "active_model_name", "test")
    monkeypatch.setattr(handlers.state, "model_cache", {"test": types.SimpleNamespace(generate_custom_voice_streaming=audio)})
    response = await handlers.generate_stream(
        text="test", language="English", mode="custom", speaker="test", ref_text="",
        instruct="", xvec_only=True, chunk_size=25, temperature=0.9, top_k=50,
        repetition_penalty=1.05, ref_preset="", ref_audio=None,
    )
    existing = asyncio.all_tasks()
    try:
        async with asyncio.timeout(2):
            events = [await anext(response.body_iterator)]
            assert await asyncio.to_thread(closed.wait, 1)
            events.extend([event async for event in response.body_iterator])
        payloads = [json.loads(event.removeprefix("data: ")) for event in events]
        assert [payload["type"] for payload in payloads] == ["chunk"] * chunk_count + ["done"]
        assert [payload["audio_b64"] for payload in payloads[:-1]] == [str(i + 1) for i in range(chunk_count)]
        assert handlers.state.generation_capacity.snapshot().active == 0
        assert not (asyncio.all_tasks() - existing)
    finally:
        await response.body_iterator.aclose()
