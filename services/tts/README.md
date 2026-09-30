# TTS Service

Standalone Qwen3 TTS service for the `liliya` voice. In the integrated project, Gateway calls this service through the standard `TTSRequest` / `TTSResponse` JSON protocol.

## Runtime

| Item | Value |
|---|---|
| Path | `./services/tts` |
| Default port | `19000` |
| App entry | `app.services.tts.server` |
| Default model | `voice_lab/liliya` |
| Python environment | `cuda_env` |

## Install

The local environment reuses the production dependency set and adds the
matching PyTorch runtime:

```powershell
cd .\services\tts
.\cuda_env\Scripts\python.exe -m pip install -r requirements-tts.txt
```

## Start

```powershell
cd .\services\tts
$env:TTS_SKIP_PRESET_FETCH="1"
.\cuda_env\Scripts\python.exe -m app.services.tts.server --model voice_lab/liliya --port 19000
```

When preset fetching is enabled, downloads require HTTPS and are installed
atomically after size and format validation. The limits can be adjusted with
`TTS_PRESET_DOWNLOAD_TIMEOUT_SECONDS`, `TTS_PRESET_MAX_AUDIO_BYTES`, and
`TTS_PRESET_MAX_TRANSCRIPT_BYTES`.

Health check:

```text
GET http://127.0.0.1:19000/api/v1/health
```

## Standard API

Gateway uses this endpoint:

```text
POST http://127.0.0.1:19000/api/v1/audio/speech
```

Alias:

```text
POST http://127.0.0.1:19000/api/v1/tts
```

Streaming endpoint and alias:

```text
POST http://127.0.0.1:19000/api/v1/audio/speech/stream
POST http://127.0.0.1:19000/api/v1/tts/stream
```

Both streaming routes accept the same JSON `TTSRequest` shape shown below and
return SSE audio events.

Request body:

```json
{
  "request_id": "tts-1",
  "trace_id": "trace-1",
  "session_id": "session-1",
  "text": "你好",
  "stream": false,
  "voice": {
    "mode": "custom",
    "speaker_id": "liliya",
    "language": "Chinese",
    "instruct": ""
  },
  "audio_config": {
    "format": "wav",
    "sample_rate": 24000,
    "channels": 1
  },
  "metadata": {
    "temperature": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.05
  }
}
```

Response body:

```json
{
  "request_id": "tts-1",
  "audio_url": null,
  "audio_base64": "UklGR...",
  "duration_ms": 2800,
  "format": "wav",
  "sample_rate": 24000,
  "rtf": 0.8,
  "metadata": {
    "provider": "http-tts",
    "tts_mode": "custom",
    "speaker": "liliya",
    "language": "Chinese"
  }
}
```

Supported `audio_config.format` values are `wav`, `pcm16`, and `mp3`. The
service resamples and adjusts channels so the encoded `audio_base64` bytes
match `format`, `sample_rate`, and `channels`. For `pcm16`, `audio_base64`
contains raw signed 16-bit little-endian interleaved samples.

## Legacy API Changes

The old form-data generation and prewarm APIs are removed:

```text
POST /api/v1/tts/generate
POST /api/v1/tts/prewarm
```

Use `/api/v1/audio/speech` or `/api/v1/audio/speech/stream`. The current
`/api/v1/tts/stream` compatibility alias accepts JSON, not the historical
form-data payload.

## Security

`GET /api/v1/health` and `GET /api/v1/status` are public. Speech generation,
streaming, and preset-reference APIs require `X-Internal-API-Key` when
`TTS_INTERNAL_API_KEY` is configured; production requires a key of at least 32
bytes.

## Other APIs

```text
GET  /api/v1/status
GET  /api/v1/preset_ref/{preset_id}
POST /api/v1/models/load
```

`POST /api/v1/models/load` only accepts model IDs returned by `GET /api/v1/status` in
`available_models` and receives `model_id` as a form field. Local loopback
requests work without credentials. Remote model management requires
`TTS_ADMIN_API_KEY` and the matching `X-TTS-Admin-Key` header.

Concurrent clients are admitted through a bounded single-GPU generation queue.
Configure `TTS_MAX_QUEUE_SIZE` (default `8`), `TTS_QUEUE_TIMEOUT_SECONDS`
(default `120`), `TTS_STREAM_QUEUE_SIZE` (default `8`), and
`TTS_STREAM_QUEUE_PUT_TIMEOUT_SECONDS` (default `5`). Scale TTS with one model
process per GPU after measuring the model's memory usage; do not add Uvicorn
workers on the same GPU because each worker loads another model copy.

The selected model is preloaded once and remains resident in the process-level
GPU cache across conversations. `TTS_OUTPUT_SAMPLE_RATE` can explicitly override
the checkpoint sample rate; otherwise the service reads local checkpoint metadata
and uses `24000` Hz only when no metadata is available.
