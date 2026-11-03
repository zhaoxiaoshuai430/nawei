"""Native stateful Paraformer decoding; each connection owns its decoder cache."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Protocol

from inference_capacity import InferenceCapacity
from text_correction import correct_transcript


MODEL_ID = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"


class StreamingDecoder(Protocol):
    chunk_bytes: int

    async def decode(self, pcm: bytes, cache: dict[str, Any], *, final: bool) -> str: ...


class ParaformerStreamingModel:
    chunk_bytes = 9600 * 2

    def __init__(self, path: Path, *, device: str = "cpu", threads: int = 2) -> None:
        import yaml  # type: ignore[import-untyped]
        from funasr import AutoModel

        for name in ("model.pt", "config.yaml", "tokens.json", "am.mvn", "configuration.json"):
            if not (path / name).is_file():
                raise RuntimeError("Streaming weights missing; run scripts/prepare_streaming_asr.py --download first.")
        config = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8"))
        if config.get("model") != "ParaformerStreaming":
            raise ValueError("The realtime model must support native stateful streaming.")
        self.model = AutoModel(
            model=str(path), device=device, ncpu=threads, trust_remote_code=False,
            disable_update=True, disable_pbar=True, check_latest=False,
        )
        self.capacity = InferenceCapacity(limit=1, queue_limit=16, wait_timeout_seconds=15)
        self._decode(bytes(self.chunk_bytes), {}, final=True)

    def _decode(self, pcm: bytes, cache: dict[str, Any], *, final: bool) -> str:
        import numpy as np

        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        result = self.model.generate(
            input=samples, cache=cache, is_final=final, chunk_size=[0, 10, 5],
            encoder_chunk_look_back=4, decoder_chunk_look_back=1,
        )
        return "".join(str(item.get("text") or "") for item in result)

    async def decode(self, pcm: bytes, cache: dict[str, Any], *, final: bool) -> str:
        # Serialize access to FunASR's mutable model kwargs, without sharing per-session caches.
        async with await self.capacity.acquire():
            worker = asyncio.create_task(asyncio.to_thread(self._decode, pcm, cache, final=final))
            try:
                return await asyncio.shield(worker)
            except asyncio.CancelledError:
                # Repeated cancellation must not release the model while its thread runs.
                while not worker.done():
                    try:
                        await asyncio.shield(worker)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                if not worker.cancelled():
                    worker.exception()
                raise


class ParaformerStreamingPool:
    """Pin each connection to an independently loaded, serialized model."""

    def __init__(self, path: Path, *, workers: int = 1, device: str = "cpu", threads: int = 2) -> None:
        if workers < 1:
            raise ValueError("ASR_STREAMING_WORKERS must be at least 1.")
        self.models = [ParaformerStreamingModel(path, device=device, threads=threads) for _ in range(workers)]
        self.sessions = [0] * workers

    @contextmanager
    def session(self) -> Iterator[ParaformerStreamingModel]:
        # Prefer the least assigned worker, then avoid a worker that is already
        # decoding or has queued work. This keeps new connections away from a
        # temporarily busy model without changing per-session pinning.
        def load_score(item: int) -> tuple[int, int]:
            snapshot = self.models[item].capacity.snapshot()
            return self.sessions[item], snapshot.active + snapshot.waiting

        index = min(range(len(self.models)), key=load_score)
        self.sessions[index] += 1
        try:
            yield self.models[index]
        finally:
            self.sessions[index] -= 1

    def snapshot(self) -> dict[str, Any]:
        workers = [
            {"worker": index, "sessions": self.sessions[index], **vars(model.capacity.snapshot())}
            for index, model in enumerate(self.models)
        ]
        return {
            **{key: sum(worker[key] for worker in workers) for key in ("active", "waiting", "limit", "queue_limit", "sessions")},
            "workers": workers,
        }


class StreamingTranscript:
    def __init__(self, decoder: StreamingDecoder) -> None:
        self.decoder = decoder
        self.cache: dict[str, Any] = {}
        self.pending = bytearray()
        self.text = ""
        self.audio_bytes = 0
        self.started = False
        self.segment_index = 0

    def reset(self) -> None:
        self.cache = {}
        self.pending.clear()
        self.text = ""
        self.audio_bytes = 0
        self.started = False

    def _append(self, delta: str) -> bool:
        delta = delta.strip()
        if not delta:
            return False
        if self.text and self.text[-1].isascii() and self.text[-1].isalnum() and delta[0].isascii() and delta[0].isalnum():
            self.text += " "
        self.text += delta
        return True

    def _event(self, kind: str) -> dict[str, Any]:
        return {
            "type": kind, "segment_index": self.segment_index, "text": self.text,
            "language": "zh", "duration_ms": self.audio_bytes // 32,
            "model": MODEL_ID, "final": kind == "transcript.final",
        }

    async def feed(self, pcm: bytes) -> list[dict[str, Any]]:
        if len(pcm) % 2:
            raise ValueError("PCM16 requires complete sample frames.")
        self.audio_bytes += len(pcm)
        self.pending.extend(pcm)
        events = []
        while len(self.pending) >= self.decoder.chunk_bytes:
            chunk = bytes(self.pending[:self.decoder.chunk_bytes])
            del self.pending[:self.decoder.chunk_bytes]
            delta = await self.decoder.decode(chunk, self.cache, final=False)
            self.started = True
            if self._append(delta):
                events.append(self._event("transcript.partial"))
        return events

    async def finish(self) -> dict[str, Any]:
        # Flush even an empty tail after exact-sized chunks: the decoder may still hold tokens.
        if self.pending or self.started:
            self._append(await self.decoder.decode(bytes(self.pending), self.cache, final=True))
        event = self._event("transcript.final")
        correction = correct_transcript(self.text)
        event["raw_text"] = self.text
        event["clean_text"] = correction.text
        event["text"] = correction.text
        event["corrections"] = list(correction.applied)
        self.segment_index += 1
        self.reset()
        return event
