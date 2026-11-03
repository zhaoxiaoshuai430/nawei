# Nawei Voice AI Project

Nawei is a five-service voice AI system:

```text
Client -> Gateway -> ASR -> LLM -> TTS -> Gateway -> Client
                            |
                            +-> Knowledge (Qdrant + Neo4j)
```

The LLM service owns the internal Agent runtime, tools, and persistent memory.
The current bundled TTS voice is `liliya`.

The optional summary worker also runs inside LLM. Enable it only after the
explicit 001/002/003 migrations; deployment templates keep it disabled. See
[summary configuration](services/llm/summary/README.md) for evidence, budgets,
online use and deletion semantics.

The project exposes robot and backend APIs only. Browser chat, device pages,
audio test pages and public tunnel launchers have been removed. See
[robot integration](services/gateway/xiaozhi/README.md) for activation and binding
without a webpage. Cloud IP deployment still requires the server address and TLS
configuration; this cleanup does not deploy a server or flash the robot.

See [cloud deployment and robot onboarding](deploy/CLOUD_DEPLOYMENT.md) for
production configuration, offline models, HTTPS/WSS, migration and rollback.

## Native Streaming Voice

Windows startup now selects local Paraformer online for realtime ASR. Download
its additional weights once, using the existing ASR virtual environment:

```powershell
.\services\asr\.venv\Scripts\python.exe scripts/prepare_streaming_asr.py --download
```

The model resides in `models/asr/paraformer-online` and keeps per-session decoder
state. `/api/v1/asr/realtime` emits cumulative `transcript.partial` snapshots
while audio is arriving and `transcript.final` after endpointing or stop.
Clients replace a segment's partial text rather than appending each snapshot.
The 600 ms decoding chunk is not a guarantee of first-text latency.

Windows startup now defaults to `ASR_STREAMING_WORKERS=2`. Each worker loads its
own Paraformer model and processes one audio chunk at a time; each connection
stays with the same worker and retains its own decoder cache. `/health` exposes
`streaming_concurrency` separately from SenseVoice's `concurrency` counters.
Set `ASR_STREAMING_WORKERS=1` before startup to reduce model memory usage.
The production template exposes the same setting when Paraformer is enabled.

TTS still runs one generation per process to protect shared CUDA graph state.
Its default waiting queue is now 16; this increases burst tolerance, not GPU
throughput. Multiple TTS instances need sufficient GPU memory and routing.
This configuration does not claim capacity for 20 simultaneous voice users.

Gateway sends the first spoken segment immediately. Later segments already in
the queue are combined up to `GATEWAY_TTS_BATCH_MAX_CHARS` (default `80`, capped
at `450`) to reduce repeated synthesis setup for short sentences. Batching does
not wait for more model output. Set the limit to `1` to disable combining;
sentence-start events describe the combined spoken segment.

With local services running and two streaming workers enabled, run the live
two-robot check (uses real LLM requests and creates then removes test bindings):

```powershell
.\.venv\Scripts\python.exe scripts/check_robot_concurrency.py
```

The report in `.runtime/robot-concurrency/checks.json` records observed ASR/TTS
activity and per-robot results. First-audio timings start after microphone input
ends. This is a concurrency smoke check, not a sustained capacity benchmark.

Robot Opus upload, native ASR, LLM text streaming and sentence-based streaming
TTS form the realtime voice path. By default, answers are generated after a
finalized user turn. Optional voice preview can prepare an answer from stable
partials, but never delivers it before final validation. The generic realtime
API supports opt-in continuous turns and barge-in; robot transport can cancel
playback on a new listen command. These transport features do not establish
working acoustic echo cancellation or physical full-duplex performance. The
bundled firmware still needs OTA device-key adaptation; see the robot guide.

SenseVoice HTTP uploads and `/api/v1/asr/realtime/sensevoice` remain available,
including the original multilingual recognition. The new online model targets
Chinese. Robot firmware displays partial transcripts through the `stt` protocol.

Set `$env:ASR_REALTIME_BACKEND="sensevoice"` before starting/restarting ASR to
restore the original realtime backend. Production Compose defaults to Paraformer;
set `ASR_REALTIME_BACKEND=sensevoice` to use the original backend. Rebuild the ASR
image after downloading the online model. Model files are never fetched
automatically while serving requests.

Acceptance (local transport, real models, simulated microphone/device):

```powershell
.\.venv\Scripts\python.exe scripts/check_native_streaming.py
```

## Project Layout

```text
app_protocols/          Shared request, response, and streaming event models
services/gateway/       Public HTTP/WebSocket gateway, port 8000
services/asr/           Paraformer online + SenseVoice recognition, port 8001
services/llm/           DeepSeek, Agent, tools, and memory, port 8002
services/knowledge/     Qdrant + Neo4j GraphRAG service, port 8003
services/tts/           Qwen3-TTS with the Liliya voice, port 19000
deploy/                 Production containers, Compose, and dependencies
models/                 Local model assets, manifest, and split weight files
firmware/xiaozhi-esp32/ ESP-IDF firmware sources and board build tools
scripts/                Start, stop, status, model restore, and project checks
tests/                  Cross-service protocol and deployment regression tests
```

## Requirements

- Windows PowerShell
- Python 3.11
- Bundled local model files matching `models/manifest.json`
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

The local file contains private configuration and must not be shared. The DeepSeek configuration uses:

```dotenv
LLM_PROVIDER=deepseek
LLM_API_BASE_URL=https://api.deepseek.com
LLM_API_KEY=REPLACE_WITH_YOUR_DEEPSEEK_API_KEY
LLM_MODEL=deepseek-v4-pro
```

Production configuration starts from `deploy/.env.production.example`.

## Model Files

Validate local model files and restore available split weights before starting services:

```powershell
.\.venv\Scripts\python.exe scripts\restore_models.py
```

This local distribution does not require Git history or an LFS object cache.
Keep the complete `models/` directory, including the Liliya split weights used
by the TTS container build. Sizes and SHA-256 checksums are recorded in
`models/manifest.json`.

## Start Project

The unified startup script restores models, starts PostgreSQL, Redis, Qdrant,
and Neo4j, then launches ASR, TTS, Knowledge, LLM, and Gateway processes in that
order. Model loading continues concurrently; the script waits for all services:

Before reporting readiness, startup synthesizes a short local voice sample and
transcribes it through HTTP ASR and every online ASR worker. This moves initial
inference setup into startup without calling the LLM or creating conversation
history. Results are saved to `.runtime/voice-warmup.json`; `-SkipWarmup` skips
this step when needed.

For local robot discovery, `.env.robot` can set `NAWEI_PUBLIC_ROBOT_URL=auto`
and `XIAOZHI_PUBLIC_WEBSOCKET_URL=auto`. OTA then uses the address the client
connected to, so local and LAN clients receive their respective WebSocket URL.
The robot still needs a reachable initial OTA address. Production requires an
explicit WSS endpoint and JWT authentication.

Start Docker Desktop before running the script. If the Docker CLI is installed
but its engine is unavailable, startup now stops with a direct diagnostic.

Startup and service restart allow 900 seconds for cold model loading by default.
Pass `-TimeoutSeconds` to change that deadline. Windows readiness checks inspect
the JSON readiness fields, so an HTTP 200 with `degraded` or `loaded=false` does
not count as ready. Gateway dependency probes allow 10 seconds by default,
covering the LLM's nested memory and Knowledge checks; Windows callers wait 15
seconds, and the production Gateway container health check allows 20 seconds.
If overriding these budgets, keep each caller's deadline longer than its probe.
LLM health checks probe memory and Knowledge concurrently. PostgreSQL readiness
requires a successful `SELECT 1`, and `KNOWLEDGE_REQUIRED=true` makes a failed
Knowledge probe mark the LLM as degraded. The Knowledge client requires both
`status=ok` and `ready=true` and enforces a five-second total probe deadline.
Memory initialization publishes connections only after schema setup or Redis
ping succeeds. PostgreSQL connection and schema setup share a deadline of
`MEMORY_CONNECT_TIMEOUT_SECONDS + 0.5` seconds; Redis setup uses
`MEMORY_CONNECT_TIMEOUT_SECONDS`. Failed or cancelled setup releases its
candidate connections. Redis failures close the failed client and wait for
`MEMORY_RETRY_INTERVAL_SECONDS` (default 10 seconds) before reconnecting;
concurrent callers share the same initialization attempt. When memory is
required, LLM startup allows `LLM_MEMORY_STARTUP_TIMEOUT_SECONDS` (default 30
seconds) for PostgreSQL and Redis to recover before failing startup.

Gateway and LLM expose `/ready`, returning HTTP 503 when dependencies are not
ready. Monitoring uses these endpoints and also checks Knowledge; Gateway
`/health` remains a separate process liveness probe. WebSocket device registry
access runs in worker threads, and read-only SQLite queries do not take write locks.

History reads merge local messages still awaiting Redis persistence with shared
history by message ID. Already persisted local entries do not override newer
history from other workers. Background memory work is capped at
`MEMORY_POST_TURN_MAX_PENDING` (64 per worker); admission waits when full, with
`MEMORY_POST_TURN_TIMEOUT_SECONDS` (120 seconds) limiting admission and each task.
Memory extraction stays within the tracked task. Runtime diagnostics include
pending work, oldest age and failure count.

On Windows, the multi-worker LLM uses a selector loop with a nonblocking
cross-process accept mutex. Each callback rechecks listener readiness under
the mutex and accepts at most one connection, preventing competing workers
from blocking in `accept()` after another worker consumes the queued connection.

Windows stop/restart scripts verify process ownership and drain all LLM workers
before terminating them. A drain blocks new business requests and waits up to
150 seconds for active requests and background work. Worker files live under
the current runtime log directory. Missing worker reports, pending work or
recorded persistence failures prevent an automatic stop; a failed drain resumes
request admission. `-ForceStop` explicitly skips draining and can lose unfinished
writes; it is also needed for the one-time transition from an older running
process without drain support. Normal stops of updated processes need no flag.
Process records are replaced atomically, retain a `.bak` copy, and are saved
immediately after each launch so a readiness failure does not lose the new PID.
Container shutdown allows 45 seconds for requests, 130 seconds for background
memory work and a 210-second container grace period. These limits do not provide
crash-proof delivery or a transaction spanning PostgreSQL and Redis.

Completed turns now wait for recent conversation messages to be written before
returning completion. `MEMORY_HISTORY_WRITE_TIMEOUT_SECONDS` defaults to five
seconds; write failures are reported instead of acknowledging completion.
Long-term memory extraction still runs in the bounded background queue and uses
a separate lock, so slow extraction does not delay recent history writes.
Development mode without shared storage still only provides process-local history.

Gateway removes client-supplied age, birth-date and grade fields from metadata
before forwarding requests. Configure audience identity through the trusted LLM
profile interface. Stored profile age takes priority over internal evaluation hints.

Knowledge imports and deletes exclude concurrent operations on the same document.
Contention returns HTTP 503 and can be retried after the current operation finishes.
With PostgreSQL enabled, the exclusion also spans Knowledge processes. Cancelled
imports retain the guard while cleanup runs, with a 15-second cleanup deadline.
Forced termination can still leave partial indexes; these require retry or deletion.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_project.ps1
```

Check or stop the running services:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\status_project.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\stop_project.ps1
```

## Robot Acceptance

### Voice Preview

Voice preview is opt-in (`GATEWAY_VOICE_PREVIEW_ENABLED=true`). When enabled,
robot and generic realtime voice requests prepare an answer after an ASR
partial has remained unchanged for 250 ms (minimum six characters). Generation
runs while audio is still being received; no answer text or audio is sent before
the final transcript. Changed partials cancel the local preview. Only an exact
final-text match can reuse a preview, with keyword and LLM safety checks applied
again before delivery. Provisional answers do not write conversation history,
long-term memory, or pending weather-query state.

The default `GATEWAY_VOICE_PREVIEW_ENABLED=false` keeps streaming recognition
and generates the answer after the final transcript, avoiding unused requests.
The other
`GATEWAY_VOICE_PREVIEW_*` settings in `.env.example` control the debounce, minimum
length, per-turn attempt budget (four), and final wait budget (eight seconds).
Gateway and LLM each allow at most two concurrent previews per process. Preview
answers are buffered as text; TTS starts only after final validation. This can
increase model usage, and latency savings depend on how stable ASR output is
and how quickly the complete answer is generated. Timeout, missing cache,
capacity exhaustion, or changed text falls back to normal generation.

The LLM keeps up to 64 single-use preview results for 60 seconds. A different
LLM worker or a restart can cause a cache miss and normal regeneration. Restart
LLM and Gateway together when deploying this protocol change. Measure latency
on real recordings; `Robot turn timing` includes `voice_preview_ready` and
`voice_preview_attempts`, and reused LLM results report `voice_preview_used`.

TTS uses `GATEWAY_TTS_STREAM_CHUNK_SIZE=8` for the first segment and
`GATEWAY_TTS_CONTINUATION_CHUNK_SIZE=12` for subsequent segments. Explicit
request metadata `chunk_size` overrides both. Compare actual synthesis latency
and throughput before tuning these settings on a different GPU:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_tts_stream.py --sizes 8 12 16
```

The benchmark writes timing results and playable WAV files to `.runtime/tts-speed`.
It calls the real TTS service; run it without concurrent voice traffic.

Run protocol checks against the local services:

```powershell
.\.venv\Scripts\python.exe scripts/simulate_robot.py --rounds 20
.\.venv\Scripts\python.exe scripts/check_full_flow.py
```

`GATEWAY_ENABLE_DIAGNOSTICS=true` enables the JSON `/openapi.json` schema during
development. It is disabled in production. Gateway serves no HTML pages.

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
| 5432 | PostgreSQL | LLM memory, summary jobs/evidence, and Knowledge catalog |
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
validation. When all five services are running, include health checks with:

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
