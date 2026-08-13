# Development

## Local setup

Create `.env` from `.env.example` in the repository root, then configure the LLM provider.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_project.ps1
```

The start script launches Redis/PostgreSQL with `deploy/docker-compose.llm-infra.yml`, then starts four business processes on ports `8000`, `8001`, `8002`, and `19000`. Pass `-SkipInfrastructure` only when `DATABASE_URL` and `REDIS_URL` point to external instances.

## Main tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check app_protocols services tests
.\.venv\Scripts\python.exe -m mypy app_protocols services
```

Use the following command for the main project, ASR, TTS, and deployment checks together:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\check_project.ps1
```

## Change locations

| Change | Location |
|---|---|
| Public API and orchestration | `services/gateway/main.py` |
| Realtime speech recognition | `services/asr/src/` |
| LLM API and inference | `services/llm/main.py` |
| Agent orchestration | `services/llm/agents/` |
| AgentSkills | `services/llm/skills/` |
| Runtime tools | `services/llm/tools/` |
| Redis/PostgreSQL Memory | `services/llm/memory.py` |
| Streaming speech synthesis | `services/tts/app/services/tts/` |
| Shared DTOs and events | `app_protocols/` |
| Service unit tests | `services/<service>/tests/` |
| Cross-service tests | `tests/` |
