# Nawei Voice AI Project

This workspace contains the current ASR + LLM + TTS integration.

## Directory Layout

```text
D:\daima\nawei
+-- nawei   Main project: shared DTOs, Gateway, LLM service, tests
+-- ASR     Standalone SenseVoice ASR service
+-- TTS     Standalone Qwen3 TTS service
```

## Current Standard Chain

Clients should call the Gateway first. Gateway then calls ASR, LLM, and TTS through the project DTOs.

```text
Client
-> Gateway:8000 POST /api/v1/voice/completions
-> ASR:8001     POST /api/v1/audio/transcriptions/base64
-> LLM:8002     POST /api/v1/llm
-> TTS:19000    POST /api/v1/audio/speech
-> Gateway
-> Client
```

## Ports

| Port | Service | Role |
|---:|---|---|
| 8000 | Gateway | Public entry and orchestration |
| 8001 | ASR | Speech to text |
| 8002 | LLM | Text generation, stub or DeepSeek |
| 19000 | TTS | Text to speech |

Use `127.0.0.1` in local integration scripts.

## Public Gateway APIs

| API | Purpose |
|---|---|
| `GET /health` | Gateway health |
| `POST /api/v1/audio/transcriptions/base64` | ASR only |
| `POST /api/v1/text/completions` | LLM only |
| `POST /api/v1/text/speech-completions` | LLM -> TTS |
| `POST /api/v1/audio/speech` | TTS only |
| `POST /api/v1/voice/completions` | Full ASR -> LLM -> optional TTS |
| `POST /api/v1/voice/completions/stream` | Full ASR -> streaming LLM -> sentence streaming TTS |

## Standalone Service APIs

| Service | Standard API |
|---|---|
| ASR | `POST http://127.0.0.1:8001/api/v1/audio/transcriptions/base64` |
| LLM | `POST http://127.0.0.1:8002/api/v1/llm` |
| TTS | `POST http://127.0.0.1:19000/api/v1/audio/speech` |

The old TTS form-data APIs `/api/v1/tts/generate`, `/api/v1/tts/stream`, and `/api/v1/tts/prewarm` have been removed from the standalone TTS service.

## Start Commands

Run each command in a separate terminal.

```powershell
# 启动 LLM 文本生成服务，监听 8002 端口。
cd D:\daima\nawei\nawei
.\.venv\Scripts\python.exe -m uvicorn services.llm_service.main:app --host 0.0.0.0 --port 8002
```

```powershell
# 启动独立的 ASR 语音转文字服务，监听 8001 端口。
cd D:\daima\nawei\ASR
.\.venv\Scripts\python.exe src\main.py
```

```powershell
# 启动独立的 TTS 文字转语音服务，监听 19000 端口。
cd D:\daima\nawei\TTS
# 本地启动时跳过预设数据的下载或刷新。
$env:TTS_SKIP_PRESET_FETCH="1"
.\cuda_env\Scripts\python.exe -m app.services.tts.server --model voice_lab/nana --port 19000
```

```powershell
# 在 ASR、LLM、TTS 都启动后，再启动 Gateway 网关服务，监听 8000 端口。
cd D:\daima\nawei\nawei
.\.venv\Scripts\python.exe -m uvicorn services.gateway_service.main:app --host 0.0.0.0 --port 8000
```

## Test

```powershell
# 在主 Python 项目目录下运行测试。
cd D:\daima\nawei\nawei
.\.venv\Scripts\python.exe -m pytest -q
```

Current expected result: `14 passed`.
