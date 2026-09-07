# LLM Service

## Agent02 Prompt And Skill Adaptation

The LLM service uses the Xiaowei persona, response policy, TutorGraph routing,
and reusable teaching Skills adapted from the `agent02` branch `05-no-app`. The
adaptation stays inside the existing LLM service boundary and does not replace
the project's persistence or GraphRAG layers.

- PostgreSQL and Redis remain owned by `services.llm.memory.MemoryStore`.
- Qdrant and Neo4j remain behind the standalone Knowledge service.
- `KnowledgeClient` injects retrieved chunks and graph relations into the new
  persona and teaching prompt with the existing citation rules.
- Gateway, ASR, TTS, public DTOs, ports, and deployment topology are unchanged.

The LLM service is the complete intelligent runtime boundary. Its inference
implementation is updated through `w3280306004/agent02` commit
`ee57da600283e95bd299a217d5621f7fab0f636c`; its persona v6, contextual
follow-up continuity, knowledge-weaving Skill, native tool improvements, and
lower-latency local-history path were adapted without replacing the current
persistence or GraphRAG layers.

```text
LLM API
  -> TutorGraph / AgentSkillManager
  -> MemorySkill + ConceptSkill + KnowledgeSkill + TutorSkill
  -> standalone Knowledge service (Qdrant + Neo4j)
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
LLM_MODEL=deepseek-v4-pro
LLM_AGENT_ENABLED=true
LLM_AGENT_MODE=langgraph
LLM_TOOL_AGENT_MODE=langgraph
LLM_TOOL_SELECTION_MODE=native
LLM_TOOL_PLANNER_MAX_CALLS=3
KNOWLEDGE_WEAVING_ENABLED=true
MEMORY_ENABLED=true
LLM_MEMORY_REQUIRED=true
DATABASE_URL=postgresql://nawei:password@postgres:5432/nawei
REDIS_URL=redis://redis:6379/0
```

## Deterministic Safety

The LLM service applies local rules before model invocation, after runtime tool
calls, and before non-streaming or streaming text reaches Gateway. The guard
masks private data, removes prompt-injection directives, blocks high-confidence
dangerous instructions, buffers stream fragments across chunk boundaries, and
writes changed decisions to the `nawei.llm.safety` audit logger.

```dotenv
SAFETY_ENABLED=true
SAFETY_INPUT_ENABLED=true
SAFETY_TOOL_RESULT_ENABLED=true
SAFETY_OUTPUT_ENABLED=true
SAFETY_PRIVACY_MASK_ENABLED=true
SAFETY_AUDIT_LOG_ENABLED=true
SAFETY_STREAM_BUFFER_CHARS=24
LLM_STREAM_CHUNK_CHARS=1
```

Redis stores recent session context. PostgreSQL stores messages, profiles,
governed long-term memories, and learning-concept state. The LLM module does
not own document ingestion or embeddings; it preserves the existing
`KnowledgeClient` boundary to the standalone Qdrant and Neo4j GraphRAG service.

The registered runtime tools are calculator, dictionary, encyclopedia, local
time, unit conversion, weather, and Bocha web search. In native mode the model
selects enabled tools through OpenAI-compatible function definitions, tool
results are sanitized before the second model round, and streaming retries a
stale pooled connection only before any upstream data has been emitted.
