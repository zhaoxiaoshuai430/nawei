# Gateway Service

Gateway is the public entry point on port `8000`.

```text
Gateway -> ASR  http://127.0.0.1:8001
Gateway -> LLM  http://127.0.0.1:8002
Gateway -> TTS  http://127.0.0.1:19000
```

## Start

```powershell
.\.venv\Scripts\python.exe -m services.gateway.run
```

## APIs

```text
GET  /health
GET  /api/v1/status
POST /api/v1/audio/transcriptions/base64
POST /api/v1/text/completions
POST /api/v1/text/speech-completions
POST /api/v1/audio/speech
POST /api/v1/audio/speech/stream
WS   /api/v1/voice/realtime
```

The voice assistant uses only `WS /api/v1/voice/realtime`: the client sends
PCM16 audio frames, Gateway relays them to realtime ASR, then streams LLM text
and optional sentence-level TTS audio events back on the same WebSocket. The
complete-audio voice completion HTTP routes have been removed.

Set `metadata.continuous_turns=true` in the initial `start` message to process
each VAD-finalized transcript as an independent conversation turn without
closing the microphone session. The same `session_id` is retained for memory.

## Configuration

| Variable | Default |
|---|---|
| `GATEWAY_ASR_URL` | `http://127.0.0.1:8001/api/v1/audio/transcriptions/base64` |
| `GATEWAY_ASR_REALTIME_URL` | `ws://127.0.0.1:8001/api/v1/asr/realtime` |
| `GATEWAY_LLM_URL` | `http://127.0.0.1:8002/api/v1/llm` |
| `GATEWAY_LLM_STREAM_URL` | `http://127.0.0.1:8002/api/v1/llm/stream` |
| `GATEWAY_TTS_URL` | `http://127.0.0.1:19000/api/v1/audio/speech` |
| `GATEWAY_TTS_STREAM_URL` | `http://127.0.0.1:19000/api/v1/audio/speech/stream` |
| `GATEWAY_ENABLE_DIAGNOSTICS` | `true` outside production |

## Authentication

`GET /`, `GET /health`, and `GET /api/v1/status` are public. Other HTTP APIs
require a persistent user identity. Local development uses `X-User-ID`; for
requests that also carry `user_id`, the header and body values must match.
Production requires `GATEWAY_AUTH_MODE=jwt`, an HS256 Bearer token, and a token
`sub` matching the request `user_id`. The realtime WebSocket carries the same
identity and optional access token in its initial `start` message.

The `/test`, `/docs`, `/redoc`, and `/openapi.json` diagnostic routes can be
disabled during development with `GATEWAY_ENABLE_DIAGNOSTICS=false` and are
always unavailable when `NAWEI_ENV=production`.

Gateway forwards authenticated `user_id`, `session_id`, the current message, and request metadata. It does not own conversation history, profiles, Agent tools, or long-term memory; those belong to LLM.
