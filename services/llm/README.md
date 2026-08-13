# LLM Service

The LLM service is the complete intelligent runtime boundary. Its inference
implementation comes from `w3280306004/agent02` commit
`99fc166561aeea580b7beb0799d23865bef4829b`; the same commit's Tutor Agent,
AgentSkills, runtime tools, and persistent Memory were adapted into this module.

```text
LLM API
  -> TutorGraph / AgentSkillManager
  -> MemorySkill + ConceptSkill + TutorSkill
  -> RuntimeToolAgent
  -> DeepSeek or Stub inference
  -> Redis/PostgreSQL post-turn persistence
```

## Start

Local startup from the repository root starts Redis/PostgreSQL automatically:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_project.ps1
```

Start only LLM after configuring external storage:

```powershell
.\.venv\Scripts\python.exe -m services.llm.run
```

## API

```text
GET    /health
GET    /api/v1/llm/runtime
POST   /api/v1/llm
POST   /api/v1/llm/stream
GET    /api/v1/llm/memory/{user_id}
POST   /api/v1/llm/memory/{user_id}/remember
POST   /api/v1/llm/memory/{user_id}/profile
DELETE /api/v1/llm/memory/{user_id}/items/{memory_id}
DELETE /api/v1/llm/memory/{user_id}
```

All `/api/v1/llm/*` endpoints require `X-Internal-API-Key` when
`LLM_INTERNAL_API_KEY` is configured.

`LLMRequest.user_id` is the stable Memory owner. The LLM response metadata
contains the selected Agent route, active Skills, tool results, recalled Memory
counts, timings, and post-turn persistence result.

## Configuration

```dotenv
LLM_PROVIDER=deepseek
LLM_API_BASE_URL=https://api.deepseek.com
LLM_API_KEY=your-key
LLM_MODEL=deepseek-chat
LLM_AGENT_ENABLED=true
LLM_AGENT_MODE=langgraph
LLM_TOOL_AGENT_MODE=langgraph
MEMORY_ENABLED=true
LLM_MEMORY_REQUIRED=true
DATABASE_URL=postgresql://nawei:password@postgres:5432/nawei
REDIS_URL=redis://redis:6379/0
```

Redis stores recent session context. PostgreSQL stores messages, profiles,
governed long-term memories, and learning-concept state. Production startup
fails when either dependency is unavailable. The module has no document
ingestion, knowledge base, embedding service, or Qdrant integration.

The registered runtime tools are calculator, dictionary, encyclopedia, local
time, unit conversion, weather, and Bocha web search. Network-backed tools can
be disabled or require their corresponding provider configuration.
