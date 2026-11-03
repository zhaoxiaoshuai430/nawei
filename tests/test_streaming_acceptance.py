"""Live acceptance must preserve segmented ASR output and reject incomplete sessions."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from scripts import check_native_streaming


class ASRSocket:
    def __init__(self, *, legacy, complete):
        self.legacy = legacy
        self.complete = complete
        self.stopped = asyncio.Event()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def send(self, data):
        if isinstance(data, str) and json.loads(data).get("type") == "stop":
            self.stopped.set()

    async def recv(self):
        return json.dumps({"type": "ready", "partial_results": not self.legacy})

    async def __aiter__(self):
        yield json.dumps({"type": "transcript.partial", "text": "first"})
        await self.stopped.wait()
        for index, text in enumerate(("first segment", "second segment")):
            yield json.dumps({"type": "transcript.final", "text": text, "segment_index": index})
        if self.complete:
            yield json.dumps({"type": "session.done", "segments": 2})


@pytest.mark.parametrize("kind", ["asr", "legacy"])
@pytest.mark.parametrize("complete", [True, False])
async def test_direct_asr_requires_completion_and_preserves_segments(monkeypatch, tmp_path, kind, complete):
    socket = ASRSocket(legacy=kind == "legacy", complete=complete)
    monkeypatch.setattr(check_native_streaming, "connect", lambda *args, **kwargs: socket)
    flow = SimpleNamespace(config={}, pcm=bytes(1920), output=tmp_path)
    if not complete:
        with pytest.raises(AssertionError, match="closed before session completion"):
            await check_native_streaming.exercise(flow, kind)
        assert not (tmp_path / f"{kind}-events.json").exists()
    else:
        result = await check_native_streaming.exercise(flow, kind)
        assert result["transcript"] == "first segment second segment"
        assert result["transcript_segments"] == ["first segment", "second segment"]
        report = json.loads((tmp_path / f"{kind}-events.json").read_text(encoding="utf-8"))
        finals = [event for event in report["events"] if event["type"] == "transcript.final"]
        assert [event["segment_index"] for event in finals] == [0, 1]
