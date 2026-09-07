# Nawei Voice AI Project

Nawei is a five-service voice AI system:

```text
Client -> Gateway -> ASR -> LLM -> TTS -> Gateway -> Client
```

The LLM service owns the internal Agent runtime, tools, and persistent memory.
The current bundled TTS voice is `liliya`.

## Project Layout

```text
app_protocols/          Shared request, response, and streaming event models
services/gateway/       Public HTTP/WebSocket gateway, port 8000
services/asr/           SenseVoice speech recognition, port 8001
services/llm/           DeepSeek, Agent, tools, and memory, port 8002
services/knowledge/     Qdrant + Neo4j GraphRAG service, port 8003
services/tts/           Qwen3-TTS with the Liliya voice, port 19000
deploy/                 Production containers, Compose, and dependencies
models/                 Git LFS model manifests and split model files
scripts/                Start, stop, status, model restore, and project checks
tests/                  Cross-service protocol and deployment regression tests
```

## Requirements

- Windows PowerShell
- Python 3.11
- Git LFS for bundled model files
- Docker for local PostgreSQL/Redis/Qdrant/Neo4j infrastructure and production Compose
- Service-specific ASR and TTS Python environments

Install the main application and development dependencies:

```powershell
cd D:\daima\nawei
.\.venv\Scripts\python.exe -m pip install -r deploy\requirements-app.txt -r deploy\requirements-dev.txt
```

## Configuration

Create the local environment file:

```powershell
Copy-Item .env.example .env
```

The local file is ignored by Git. The DeepSeek configuration uses:

```dotenv
LLM_PROVIDER=deepseek
LLM_API_BASE_URL=https://api.deepseek.com
LLM_API_KEY=REPLACE_WITH_YOUR_DEEPSEEK_API_KEY
LLM_MODEL=deepseek-v4-pro
```

Production configuration starts from `deploy/.env.production.example`.

## Model Files

Download Git LFS objects and restore split model files before starting services:

```powershell
git lfs pull
.\.venv\Scripts\python.exe scripts\restore_models.py
```

The repository tracks the Liliya model split parts through Git LFS. The restored
`models/tts/liliya/checkpoint-epoch-1/model.safetensors` file is intentionally
ignored because it can be reconstructed from the validated parts in
`models/manifest.json`.

## Start Project

The unified startup script restores models, starts PostgreSQL, Redis, Qdrant,
and Neo4j, then starts ASR, Knowledge, LLM, TTS, and Gateway:

Start Docker Desktop before running the script. If the Docker CLI is installed
but its engine is unavailable, startup now stops with a direct diagnostic.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_project.ps1
```

Check or stop the running services:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\status_project.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\stop_project.ps1
```

## Browser Test Page

After Gateway starts in development mode, open the local end-to-end test console:

```text
http://127.0.0.1:8000/test
```

The page supports uploaded WAV/MP3 audio and realtime microphone conversation.
Both modes play TTS chunks as soon as the first streaming audio block arrives.
It displays Gateway dependency latency plus file-read, ASR, LLM, TTS, first-token,
first-audio, recording, and total elapsed times. Microphone access works on
`localhost` or HTTPS origins.

`GATEWAY_ENABLE_DIAGNOSTICS=true` enables `/test`, `/docs`, `/redoc`, and
`/openapi.json` during development. These routes are always disabled when
`NAWEI_ENV=production`.

Use `-SkipInfrastructure` only when externally managed PostgreSQL, Redis,
Qdrant, and Neo4j instances are already configured.

## Manual Start

Run each service in a separate terminal.

ASR:

```powershell
cd D:\daima\nawei\services\asr
.\.venv\Scripts\python.exe src\main.py
```

LLM:

```powershell
cd D:\daima\nawei
.\.venv\Scripts\python.exe -m services.llm.run
```

Knowledge GraphRAG:

```powershell
cd D:\daima\nawei
.\.venv\Scripts\python.exe -m services.knowledge.run
```

TTS:

```powershell
cd D:\daima\nawei\services\tts
$env:TTS_SKIP_PRESET_FETCH="1"
.\cuda_env\Scripts\python.exe -m app.services.tts.server --model voice_lab/liliya --port 19000
```

Gateway:

```powershell
cd D:\daima\nawei
.\.venv\Scripts\python.exe -m services.gateway.run
```

## Ports

| Port | Service | Purpose |
|---:|---|---|
| 8000 | Gateway | Public API and orchestration |
| 8001 | ASR | Speech recognition |
| 8002 | LLM | DeepSeek, Agent, tools, and memory |
| 8003 | Knowledge | Qdrant vector retrieval and Neo4j graph expansion |
| 19000 | TTS | Liliya speech synthesis |
| 5432 | PostgreSQL | Persistent LLM memory |
| 6379 | Redis | Session and profile cache |
| 6333 | Qdrant | Knowledge chunk vectors and payloads |
| 7474/7687 | Neo4j | Browser and Bolt knowledge graph access |

## Gateway APIs

| API | Purpose |
|---|---|
| `GET /health` | Gateway process health |
| `GET /api/v1/status` | Dependency and concurrency status |
| `GET /api/v1/status/details` | Authenticated detailed dependency status (`X-User-ID` locally) |
| `POST /api/v1/audio/transcriptions/base64` | ASR through Gateway |
| `POST /api/v1/text/completions` | LLM text completion |
| `POST /api/v1/text/speech-completions` | LLM response with synthesized audio |
| `POST /api/v1/audio/speech` | TTS through Gateway |
| `POST /api/v1/audio/speech/stream` | Streaming TTS through Gateway |
| `WS /api/v1/voice/realtime` | Realtime ASR, LLM, and optional TTS |

## Standalone APIs

| Service | Health | Main API |
|---|---|---|
| ASR | `GET http://127.0.0.1:8001/health` | `POST /api/v1/audio/transcriptions/base64` |
| LLM | `GET http://127.0.0.1:8002/health` | `POST /api/v1/llm` |
| Knowledge | `GET http://127.0.0.1:8003/health` | `POST /api/v1/knowledge/retrieve` |
| TTS | `GET http://127.0.0.1:19000/api/v1/status` | `POST /api/v1/audio/speech` |

Internal service APIs require their configured internal API keys when security
is enabled.

## Project Checks

Run all locally available checks:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\check_project.ps1
```

The script runs Python compilation, Ruff, Mypy, main tests, tracked-secret
scanning, ASR tests, TTS tests, PowerShell parsing, and Docker Compose
validation. When all four services are running, include health checks with:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\check_project.ps1 -Live
```

Run only the fast main test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Production Compose

Validate the production configuration:

```powershell
$env:NAWEI_ENV_FILE=".env.production.example"
docker compose --env-file deploy\.env.production.example -f deploy\docker-compose.production.yml config --quiet
```

Create a real `deploy/.env.production` with distinct production secrets before
starting the production stack.
