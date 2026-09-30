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

Robot discovery, activation, device management and Opus v1 transport are documented
in [xiaozhi/README.md](xiaozhi/README.md). Set `XIAOZHI_ENABLED=true` to enable
robot ingress. Bind devices through the authenticated management API.
Gateway serves no HTML pages. Production deployment uses the cloud server
directly, with JWT for management APIs and HTTPS/WSS for robot access.

```text
GET  /health
GET  /api/v1/status
POST /xiaozhi/ota/
POST /xiaozhi/ota/activate
POST /api/v1/devices/bind
GET  /api/v1/devices
WS   /xiaozhi/v1/
POST /api/v1/audio/transcriptions/base64
POST /api/v1/text/completions
POST /api/v1/text/speech-completions
POST /api/v1/audio/speech
POST /api/v1/audio/speech/stream
WS   /api/v1/voice/realtime
```

The robot uses `WS /xiaozhi/v1/` with Opus audio and a device token. The generic
`WS /api/v1/voice/realtime` API remains available for backend protocol checks: the client sends
PCM16 audio frames, Gateway relays them to realtime ASR, then streams LLM text
and optional sentence-level TTS audio events back on the same WebSocket. The
complete-audio voice completion HTTP routes have been removed.

Set `metadata.continuous_turns=true` in the initial `start` message to process
each VAD-finalized transcript as an independent conversation turn without
closing the microphone session. The same `session_id` is retained for memory.
Set `metadata.barge_in=true` as well to cancel an active LLM/TTS response when
the next `speech.started` event arrives. Gateway emits `response.interrupted`
after the old response stream has been cancelled.

## Configuration

| Variable | Default |
|---|---|
| `GATEWAY_ASR_URL` | `http://127.0.0.1:8001/api/v1/audio/transcriptions/base64` |
| `GATEWAY_ASR_REALTIME_URL` | `ws://127.0.0.1:8001/api/v1/asr/realtime` |
| `GATEWAY_STARTUP_WARMUP_ENABLED` | `true` |
| `GATEWAY_LLM_URL` | `http://127.0.0.1:8002/api/v1/llm` |
| `GATEWAY_LLM_STREAM_URL` | `http://127.0.0.1:8002/api/v1/llm/stream` |
| `GATEWAY_TTS_URL` | `http://127.0.0.1:19000/api/v1/audio/speech` |
| `GATEWAY_TTS_STREAM_URL` | `http://127.0.0.1:19000/api/v1/audio/speech/stream` |
| `GATEWAY_KEYWORD_FILTER_ENABLED` | `false` |
| `GATEWAY_KEYWORD_FILTER_URL` | empty |
| `GATEWAY_KEYWORD_FILTER_FAILURE_MODE` | `closed` |
| `GATEWAY_ENABLE_DIAGNOSTICS` | `true` outside production |

## ASR keyword filtering

When enabled, the realtime voice path becomes `ASR -> keyword filter -> LLM Agent -> TTS`.
Gateway posts the final ASR transcript to the configured keyword API before creating an
LLM request. A blocked transcript bypasses the Agent and LLM completely, returns
`GATEWAY_KEYWORD_FILTER_FALLBACK_TEXT`, and sends that fallback text to TTS when audio
output is enabled.

The default request body is `{"text": "..."}` and the default response contract is:

```json
{
  "blocked": true,
  "matched_keywords": ["example"],
  "reason": "policy"
}
```

Use dotted field paths to adapt another provider without changing code. For example, a
response shaped as `{"data": {"hit": true, "terms": []}}` uses
`GATEWAY_KEYWORD_FILTER_BLOCKED_FIELD=data.hit` and
`GATEWAY_KEYWORD_FILTER_MATCHED_KEYWORDS_FIELD=data.terms`. If the provider returns an
allow flag instead, leave the blocked field empty and set
`GATEWAY_KEYWORD_FILTER_ALLOWED_FIELD=data.safe`.

`GATEWAY_KEYWORD_FILTER_FAILURE_MODE=closed` sends the fallback when the external API
times out, fails, or returns an invalid payload. Set it to `open` only when availability
is more important than blocking on filter failures.

## Authentication

`GET /`, `GET /health`, and `GET /api/v1/status` are public. Other HTTP APIs
require a persistent user identity. Local development uses `X-User-ID`; for
requests that also carry `user_id`, the header and body values must match.
Production requires `GATEWAY_AUTH_MODE=jwt`, an HS256 Bearer token, and a token
`sub` matching the request `user_id`. The realtime WebSocket carries the same
identity and optional access token in its initial `start` message.

The JSON `/openapi.json` schema can be disabled during development with
`GATEWAY_ENABLE_DIAGNOSTICS=false` and is always unavailable in production.

Gateway forwards authenticated `user_id`, `session_id`, the current message, and request metadata. It does not own conversation history, profiles, Agent tools, or long-term memory; those belong to LLM.
