# app_protocols

`app_protocols` defines the shared DTOs and provider boundaries used by the
Gateway, ASR, LLM, and TTS integration.

## Important DTOs

| DTO | Direction | Purpose |
|---|---|---|
| `AudioTranscriptionRequest` | Client -> Gateway | ASR-only public request |
| `ASRRequest` | Gateway -> ASR | Internal ASR request |
| `ASRResponse` | ASR -> Gateway | Normalized ASR result |
| `ChatCompletionRequest` | Client -> Gateway | LLM-only public request |
| `LLMRequest` | Gateway -> LLM | Internal LLM request |
| `LLMResponse` | LLM -> Gateway | Normalized LLM result |
| `TTSRequest` | Client/Gateway -> TTS | TTS request |
| `TTSResponse` | TTS -> Client/Gateway | Normalized TTS result |
| `RealtimeVoiceStart` | Client -> Gateway WebSocket | Realtime voice session start, defined in Gateway |

## Audio Source Rule

ASR requests must provide exactly one audio source:

```text
audio_base64 XOR audio_url
```

The current ASR HTTP provider supports `audio_base64`.

## TTS Response Rule

The project standard field is:

```text
audio_base64
```

The old standalone TTS field `audio_b64` is no longer exposed by the standard TTS endpoint.

## Main Schemas

`ASRRequest`:

```json
{
  "request_id": "req-1",
  "trace_id": "trace-1",
  "session_id": "session-1",
  "user_id": "student-1",
  "audio_base64": "UklGR...",
  "audio_url": null,
  "language": "auto",
  "return_timestamps": false,
  "format": "wav",
  "sample_rate": 16000,
  "channels": 1,
  "options": {}
}
```

`LLMRequest`:

```json
{
  "request_id": "req-1",
  "trace_id": "trace-1",
  "session_id": "session-1",
  "user_id": "student-1",
  "messages": [
    {"role": "user", "content": "你好", "name": null, "metadata": {}}
  ],
  "stream": false,
  "tools": [],
  "generation_config": {},
  "metadata": {}
}
```

`TTSRequest`:

```json
{
  "request_id": "req-1",
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
  "metadata": {}
}
```

## Source Files

| File | Role |
|---|---|
| `schemas.py` | Pydantic DTO definitions |
| `enums.py` | Shared enum values |
| `orchestrator.py` | ASR + LLM + optional TTS orchestration |
| `errors.py` | Shared error model helpers |
