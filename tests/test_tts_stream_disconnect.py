"""A disconnected ASGI stream must finish its worker and return capacity."""

import asyncio
import importlib
from pathlib import Path
import sys
import threading
import types

import pytest
from starlette.requests import ClientDisconnect


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


@pytest.mark.parametrize("disconnect_during_send", [False, True])
async def test_disconnect_waits_for_worker_then_releases_capacity(handlers, monkeypatch, disconnect_during_send):
    capacity = handlers.state.generation_capacity
    release_worker = threading.Event()
    first_chunk = asyncio.Event()
    disconnected = asyncio.Event()

    class Model:
        def generate_custom_voice_streaming(self, **kwargs):
            yield bytes(100), 24000, {}
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
