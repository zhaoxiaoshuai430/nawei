# Operations

## Local commands

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_project.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\status_project.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\stop_project.ps1
```

Expected services:

```text
gateway  8000
asr      8001
llm      8002
tts      19000
```

Gateway status: `GET http://127.0.0.1:8000/api/v1/status`.
LLM runtime status: `GET http://127.0.0.1:8002/api/v1/llm/runtime` with the configured internal key.

## Troubleshooting

- If LLM health is healthy but replies use fallback, inspect `LLM_PROVIDER`, API key, model, timeout, and upstream response.
- If LLM fails to start, inspect PostgreSQL/Redis health, `DATABASE_URL`, `REDIS_URL`, and `LLM_MEMORY_REQUIRED`.
- If realtime voice fails before text generation, inspect the Gateway and ASR WebSocket logs, then test ASR health separately.
- If text works without audio, test `/api/v1/audio/speech` and the TTS model directory.
- Runtime logs are written below `logs/runtime-*` by the PowerShell launcher.

LLM `/health` reports Agent, Skills, registered Tools, PostgreSQL, and Redis state. Production requires both stores to report `ok`.
