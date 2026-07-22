# TTS Service

Standalone Qwen3 TTS service for the `nana` voice. In the integrated project, Gateway calls this service through the standard `TTSRequest` / `TTSResponse` JSON protocol.

## Runtime

| Item | Value |
|---|---|
| Path | `D:\daima\nawei\TTS` |
| Default port | `19000` |
| App entry | `app.services.tts.server` |
| Default model | `voice_lab/nana` |
| Python environment | `cuda_env` |

## Start

```powershell
cd D:\daima\nawei\TTS
$env:TTS_SKIP_PRESET_FETCH="1"
.\cuda_env\Scripts\python.exe -m app.services.tts.server --model voice_lab/nana --port 19000
```

Health check:

```text
GET http://127.0.0.1:19000/api/v1/health
```

Demo page:

```text
http://127.0.0.1:19000/demo
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
    "speaker_id": "nana",
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
    "speaker": "nana",
    "language": "Chinese"
  }
}
```

Supported `audio_config.format` values are `wav`, `pcm16`, and `mp3`. The
service resamples and adjusts channels so the encoded `audio_base64` bytes
match `format`, `sample_rate`, and `channels`. For `pcm16`, `audio_base64`
contains raw signed 16-bit little-endian interleaved samples.

## Removed Legacy APIs

The old form-data APIs are intentionally removed from the standalone TTS service:

```text
POST /api/v1/tts/generate
POST /api/v1/tts/stream
POST /api/v1/tts/prewarm
```

Use `/api/v1/audio/speech` instead.

## Other APIs

```text
GET  /api/v1/status
GET  /api/v1/preset_ref/{preset_id}
POST /api/v1/models/load
```

`POST /api/v1/models/load` only accepts model IDs returned by `GET /api/v1/status` in
`available_models`. Local loopback requests work without credentials. Remote model
management requires `TTS_ADMIN_API_KEY` and the matching `X-TTS-Admin-Key` header.
