# Services

This directory contains the main project services. In the current integration, Gateway calls:

```text
ASR: D:\daima\nawei\ASR        port 8001
LLM: D:\daima\nawei\nawei      port 8002
TTS: D:\daima\nawei\TTS        port 19000
```

## Current Chain

```text
Client
-> Gateway:8000 /api/v1/voice/completions
-> ASR:8001 /api/v1/audio/transcriptions/base64
-> LLM:8002 /api/v1/llm
-> TTS:19000 /api/v1/audio/speech
-> Gateway
-> Client
```

## Services in This Directory

| Directory | Current role |
|---|---|
| `gateway_service` | Public Gateway and orchestrator entry |
| `llm_service` | LLM service, supports stub and DeepSeek |
| `asr_service` | Older local stub ASR example, not used by the current Gateway defaults |
| `tts_service` | Older local stub TTS example, not used by the current Gateway defaults |

The active ASR and TTS services are the sibling directories `D:\daima\nawei\ASR` and `D:\daima\nawei\TTS`.

## Gateway APIs

| API | Purpose |
|---|---|
| `GET /health` | Gateway health |
| `POST /api/v1/audio/transcriptions/base64` | ASR only |
| `POST /api/v1/text/completions` | LLM only |
| `POST /api/v1/text/speech-completions` | LLM -> TTS |
| `POST /api/v1/audio/speech` | TTS only |
| `POST /api/v1/voice/completions` | Full ASR -> LLM -> optional TTS |

## Standalone APIs

| Service | URL |
|---|---|
| ASR | `http://127.0.0.1:8001/api/v1/audio/transcriptions/base64` |
| LLM | `http://127.0.0.1:8002/api/v1/llm` |
| TTS | `http://127.0.0.1:19000/api/v1/audio/speech` |

## Test

```powershell
cd D:\daima\nawei\nawei
.\.venv\Scripts\python.exe -m pytest -q
```
