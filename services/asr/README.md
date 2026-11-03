# ASR Service

ASR service with SenseVoice for complete-audio recognition and Paraformer Online
for native Chinese streaming. Gateway uses `ASRRequest` / `ASRResponse` for HTTP
uploads and PCM16 WebSocket events for realtime recognition.

## Runtime

| Item | Value |
|---|---|
| Path | `./services/asr` |
| Default port | `8001` |
| App entry | `src\main.py` |
| Model | `iic/SenseVoiceSmall` |
| Code default device | `cuda:0` |
| Unified startup device | `cpu` |
| Local model path | Repository `models/asr/` (SenseVoiceSmall, fsmn-vad, paraformer-online) |
| Download cache | `services/asr/hf_download` |

## Start

```powershell
cd .\services\asr
.\.venv\Scripts\python.exe src\main.py
```

Health check:

```text
GET http://127.0.0.1:8001/health
```

## Standard API

Gateway uses this endpoint:

```text
POST http://127.0.0.1:8001/api/v1/audio/transcriptions/base64
```

Alias:

```text
POST http://127.0.0.1:8001/api/v1/asr/transcriptions
```

For raw little-endian PCM16, set `format` to `pcm16` and provide the actual
`sample_rate` and `channels`. The service wraps these samples in a WAV container
before decoding. Each payload must contain complete frames (two bytes per
channel); incomplete frames return HTTP 400.

Request body:

```json
{
  "request_id": "asr-1",
  "trace_id": "trace-1",
  "session_id": "session-1",
  "audio_base64": "UklGR...",
  "audio_url": null,
  "language": "auto",
  "return_timestamps": false,
  "format": "wav",
  "sample_rate": 16000,
  "channels": 1,
  "options": {
    "filename": "audio.wav",
    "use_itn": true
  }
}
```

Response body:

```json
{
  "request_id": "asr-1",
  "text": "你好呀。",
  "language": "zh",
  "confidence": null,
  "duration_ms": null,
  "segments": [],
  "asr": {
    "key": "audio.wav",
    "emotion": "SAD",
    "emotion_label": "悲伤",
    "emotion_emoji": "😔",
    "events": [],
    "emotion_hint": "情绪：悲伤",
    "raw_text": "<|zh|>...",
    "clean_text": "你好呀。"
  },
  "metadata": {
    "provider": "sensevoice",
    "model": "iic/SenseVoiceSmall"
  }
}
```

## Security

`GET /health` and `GET /api/v1/capabilities` are public. ASR transcription
endpoints require `X-Internal-API-Key` when `ASR_INTERNAL_API_KEY` is configured;
production requires a key of at least 32 bytes. The realtime WebSocket supplies
the same key as `internal_api_key` in its initial `start` message.

## Legacy Native APIs

These endpoints are kept for ASR debugging and old tools. They are not the Gateway standard path.

```text
POST /api/v1/asr
POST /api/v1/asr/base64
POST /api/v1/asr/stream
```

`/api/v1/asr/stream` recognizes each posted, independently decodable audio
chunk and returns one SSE result. SenseVoice does not retain decoder state
between requests, so this endpoint is not true frame-level realtime ASR. Check
`GET /api/v1/capabilities` and the `X-ASR-Streaming-Mode` response header before
selecting a client strategy.

## Realtime PCM Input

Gateway uses the following WebSocket endpoint for microphone streaming:

```text
WS /api/v1/asr/realtime
```

The first WebSocket message is JSON:

```json
{
  "type": "start",
  "sample_rate": 16000,
  "channels": 1,
  "language": "auto",
  "use_itn": true,
  "internal_api_key": "..."
}
```

With either backend, after `ready`, send little-endian mono PCM16 as binary messages. Send
`{"type":"stop"}` to flush the active utterance. With the legacy SenseVoice backend, the service performs
energy-based VAD, queues completed utterances through the normal inference
capacity limiter, and emits `speech.started`, `speech.ended`,
`transcript.processing`, `transcript.final`, and `session.done` events.

The SenseVoice backend provides realtime input with sentence-level final results. SenseVoice still
does not retain decoder state, so `partial` word/token results are intentionally
not advertised.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ASR_HOST` | `0.0.0.0` | Bind address |
| `ASR_PORT` | `8001` | Service port |
| `ASR_DEVICE` | `cuda:0` | Inference device |
| `ASR_MODEL_ID` | `iic/SenseVoiceSmall` | SenseVoice model |
| `ASR_MODEL_PATH` | auto-detected | Explicit local SenseVoice snapshot directory |
| `ASR_VAD_MODEL_ID` | `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` | VAD model |
| `ASR_VAD_MODEL_PATH` | auto-detected | Explicit local VAD snapshot directory |
| `ASR_MAX_CONCURRENT_INFERENCES` | `1` | Parallel model calls per process; keep `1` unless the model/GPU is validated for parallel inference |
| `ASR_MAX_QUEUE_SIZE` | `16` | Maximum queued inference requests per process |
| `ASR_QUEUE_TIMEOUT_SECONDS` | `30` | Maximum time a request waits for an inference slot |
| `ASR_REALTIME_RMS_THRESHOLD` | `500` | PCM16 RMS threshold used to classify a 20 ms frame as voiced |
| `ASR_REALTIME_START_MS` | `60` | Consecutive voiced duration required to start an utterance |
| `ASR_REALTIME_END_SILENCE_MS` | `420` | Silence duration that finalizes an utterance |
| `ASR_REALTIME_PRE_ROLL_MS` | `200` | Audio retained before the detected speech start |
| `ASR_REALTIME_MIN_SPEECH_MS` | `180` | Minimum finalized segment duration |
| `ASR_REALTIME_MAX_SEGMENT_SECONDS` | `20` | Maximum duration before a segment is forcibly finalized |
| `ASR_REALTIME_MAX_SESSION_SECONDS` | `3600` | Maximum PCM duration accepted by one WebSocket session; raise it for longer continuous dictation |
| `ASR_REALTIME_SEGMENT_QUEUE_SIZE` | `4` | Completed utterances buffered while inference is busy |
| `ASR_TEXT_CORRECTIONS_ENABLED` | `true` | Apply conservative corrections to finalized transcripts |
| `ASR_TEXT_CORRECTIONS_INCLUDE_DEFAULTS` | `true` | Include built-in corrections for observed command phrases |
| `ASR_TEXT_CORRECTIONS_JSON` | empty | JSON phrase map added after model decoding; use contextual phrases instead of single characters |

The service sets ModelScope and HuggingFace cache variables to project-local paths under `hf_download` so it does not write to the Windows user cache directory. Complete local snapshots are preferred automatically, and routine FunASR/ModelScope update checks are disabled during service startup.
# Native online ASR

The Windows launcher and production Compose default to `ASR_REALTIME_BACKEND=paraformer`.
The direct standalone Python entry defaults to `sensevoice` unless explicitly configured.
Prepare the additional model with `scripts/prepare_streaming_asr.py --download`
using the ASR interpreter from the project root.

- `ASR_STREAMING_MODEL_PATH`: local `models/asr/paraformer-online` directory.
- `ASR_STREAMING_DEVICE`: `cpu` by default, separate from the SenseVoice device.
- `ASR_STREAMING_THREADS`: `2` by default.
- `ASR_STREAMING_WORKERS`: direct code default `1`; Windows launcher and production
  template default `2`. These are independent model instances inside one ASR
  process, with per-connection pinning and serialized decoding per instance.
- Realtime input: 16 kHz, mono, little-endian PCM16; packets up to 32000 bytes.
- Native decoding: `[0, 10, 5]` chunks, 600 ms stride, connection-local caches.
- `transcript.partial.text` is the current segment's complete snapshot, not a delta.
- `transcript.final` replaces that snapshot and resets decoder state for the next segment.
- Final events preserve `raw_text` and expose corrected `text`/`clean_text` plus applied correction metadata.
- `stop` flushes residual PCM and decoder tokens before `session.done`.
- Explicit legacy route: `/api/v1/asr/realtime/sensevoice`.
- HTTP upload endpoints and their SenseVoice behavior remain unchanged.

`/health` and `/api/v1/capabilities` distinguish native WebSocket streaming from
the existing independent HTTP chunk mode. The online model targets Chinese and
does not provide SenseVoice emotion/event tags. Use the legacy path when those
capabilities or the original multilingual recognition are required.
