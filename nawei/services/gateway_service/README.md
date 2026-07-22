# Gateway Service

Gateway is the public API layer. It receives client requests and calls ASR, LLM, and TTS using the shared DTOs in `app_protocols`.

## Runtime

| Item | Value |
|---|---|
| Path | `D:\daima\nawei\nawei\services\gateway_service` |
| Default port | `8000` |
| App | `services.gateway_service.main:app` |

## Start

```powershell
cd D:\daima\nawei\nawei
.\.venv\Scripts\python.exe -m uvicorn services.gateway_service.main:app --host 0.0.0.0 --port 8000
```

Health check:

```text
GET http://127.0.0.1:8000/health
```

Expected service modes:

```json
{
  "status": "ok",
  "services": {
    "asr": "http",
    "llm": "http",
    "tts": "http"
  }
}
```

## Default Upstream URLs

| Variable | Default |
|---|---|
| `GATEWAY_ASR_URL` | `http://127.0.0.1:8001/api/v1/audio/transcriptions/base64` |
| `GATEWAY_LLM_URL` | `http://127.0.0.1:8002/api/v1/llm` |
| `GATEWAY_LLM_STREAM_URL` | `http://127.0.0.1:8002/api/v1/llm/stream` |
| `GATEWAY_TTS_URL` | `http://127.0.0.1:19000/api/v1/audio/speech` |
| `GATEWAY_TTS_STREAM_URL` | `http://127.0.0.1:19000/api/v1/audio/speech/stream` |
| `GATEWAY_ASR_TIMEOUT_SECONDS` | `300` |
| `GATEWAY_LLM_TIMEOUT_SECONDS` | `60` |
| `GATEWAY_TTS_TIMEOUT_SECONDS` | `300` |
| `GATEWAY_SESSION_MAX_COUNT` | `1000` |
| `GATEWAY_SESSION_MAX_MESSAGES` | `40` |
| `GATEWAY_SESSION_TTL_SECONDS` | `3600` |

## APIs

### ASR only

```text
POST http://127.0.0.1:8000/api/v1/audio/transcriptions/base64
```

Request body follows `AudioTranscriptionRequest`.

### LLM only

```text
POST http://127.0.0.1:8000/api/v1/text/completions
```

Request body follows `ChatCompletionRequest`.

### LLM then TTS

```text
POST http://127.0.0.1:8000/api/v1/text/speech-completions
```

Request body follows `ChatCompletionRequest`. Gateway ignores the incoming `output_audio` value for this route and always runs:

```text
Client text -> LLM -> TTS -> ChatCompletionResponse
```

The response includes the LLM reply in `text` and the synthesized audio in `audio_base64`.

### TTS only

```text
POST http://127.0.0.1:8000/api/v1/audio/speech
```

Request body follows `TTSRequest`.

### Full voice turn

```text
POST http://127.0.0.1:8000/api/v1/voice/completions
```

Request body:

```json
{
  "request_id": "voice-1",
  "trace_id": "trace-1",
  "session_id": "session-1",
  "audio_base64": "UklGR...",
  "language": "auto",
  "output_audio": true,
  "voice": {
    "mode": "custom",
    "speaker_id": "nana",
    "language": "Chinese",
    "instruct": ""
  },
  "input_audio_config": {
    "format": "wav",
    "sample_rate": 16000,
    "channels": 1
  },
  "output_audio_config": {
    "format": "wav",
    "sample_rate": 24000,
    "channels": 1
  },
  "metadata": {
    "filename": "audio.wav",
    "use_itn": true
  }
}
```

Response contains:

```text
transcription: AudioTranscriptionResponse
completion:    ChatCompletionResponse
metadata.path: asr+llm or asr+llm+tts
```

Exactly one of `audio_base64` and `audio_url` is required. The current HTTP
ASR adapter supports `audio_base64`; `audio_url` is rejected by the standalone
ASR service until URL ingestion is implemented.

`input_audio_config` describes audio sent to ASR. `output_audio_config`
describes audio produced by TTS. The deprecated `audio_config` field remains
accepted by itself and is applied to both directions for compatibility; it
cannot be combined with either split field.

Gateway keeps successful user and assistant messages by `session_id`. History
is bounded by the session settings above, refreshes its TTL on access, and
serializes concurrent turns for the same session. The default store is
in-process, so deployments with multiple Gateway workers or replicas must
replace it with a shared store to provide cross-worker conversation continuity.

### Streaming full voice turn

```text
POST http://127.0.0.1:8000/api/v1/voice/completions/stream
Accept: text/event-stream
```

The request body is the same `VoiceCompletionRequest` used by the non-streaming
endpoint. Gateway completes ASR first, streams LLM text deltas, and sends each
completed sentence to the standalone TTS streaming API. The canonical SSE event
order is:

```text
asr.started
asr.completed
message.started
message.delta       repeated
audio.started       when output_audio=true
audio.delta         repeated, includes audio_base64 and segment_index
message.done
audio.done
```

Failures after streaming starts are returned as an `error` SSE event. Browser
clients should use streaming `fetch()` because the endpoint is POST with a JSON
body; the native `EventSource` API only supports GET.

## Notes

Gateway no longer uses local ASR/TTS stubs by default. It calls the standalone ASR and TTS services over HTTP.
