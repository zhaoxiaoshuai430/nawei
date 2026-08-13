# Cross-Service Tests

This directory contains only deployment, shared-protocol, and service-boundary
tests. Unit tests that belong to one service live with that service:

```text
services/gateway/tests
services/asr/tests
services/llm/tests
services/tts/tests
```

`test_service_connectivity.py` verifies the four-service runtime chain:

```text
Gateway -> ASR -> LLM -> TTS
```

## Fast contract tests

The contract tests use deterministic service replacements. They cover health
checks, request and response mapping, internal authentication, the realtime
WebSocket route, and interleaved LLM/TTS events without loading models.

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_service_connectivity.py -k GatewayConnectivity
```

## Live service tests

Start the complete project first. The live suite calls the real ASR, LLM, and
TTS services and can consume model resources, so it is disabled by default.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_project.ps1
$env:NAWEI_RUN_LIVE_TESTS="1"
.\.venv\Scripts\python.exe -m pytest -q tests\test_service_connectivity.py -k LiveService
```

The default audio is `services/asr/tests/fixtures/test.wav` relative to the repository root.
Override endpoints, audio, or timeout when necessary:

```powershell
$env:NAWEI_GATEWAY_URL="http://127.0.0.1:8000"
$env:NAWEI_ASR_URL="http://127.0.0.1:8001"
$env:NAWEI_LLM_URL="http://127.0.0.1:8002"
$env:NAWEI_TTS_URL="http://127.0.0.1:19000"
$env:NAWEI_TEST_AUDIO="C:\audio\sample.wav"
$env:NAWEI_LIVE_TEST_TIMEOUT_SECONDS="300"
```

The live cases cover all four health endpoints, Gateway dependency readiness,
and each direct service route. Realtime voice protocol behavior is covered by
the Gateway WebSocket contract suite.

## Static checks

Install the main development tools once:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Run all locally available checks from the workspace root:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\check_project.ps1
```

Pass `-Live` after all four services are running to include real connectivity
calls. The script also validates Ruff, Mypy, ASR, TTS, and the production
Docker Compose file when Docker is installed.
