# LLM Service

## Agent02 Prompt And Skill Adaptation

The LLM service uses the Xiaowei persona, response policy, TutorGraph routing,
and reusable teaching Skills adapted from the `agent02` branch `08全网+北京话`. The
adaptation stays inside the existing LLM service boundary and does not replace
the project's persistence or GraphRAG layers.

- PostgreSQL and Redis remain owned by `services.llm.memory.MemoryStore`.
- Qdrant and Neo4j remain behind the standalone Knowledge service.
- `KnowledgeClient` injects retrieved chunks and graph relations into the new
  persona and teaching prompt with the existing citation rules.
- Gateway, ASR, TTS, public DTOs, ports, and deployment topology are unchanged.

The LLM service is the complete intelligent runtime boundary. Its inference
implementation is updated through `w3280306004/agent02` commit
`9d0996cdad93714c3b4547f1a4ebbf23cd9a2e5e`; its persona v6, contextual
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
LLM_UPSTREAM_WARMUP_ENABLED=true
LLM_UPSTREAM_WARMUP_TIMEOUT_SECONDS=5
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

Startup warmup performs an authenticated `GET /models` through the persistent
HTTP client. It prepares DNS, TLS, and the connection pool without creating a
chat completion or writing conversation memory; failures are non-fatal.

## Deterministic Safety

### Aliyun Content Review

Set `ALIYUN_CONTENT_SAFETY_ENABLED=true` with `ALIBABA_CLOUD_ACCESS_KEY_ID`,
`ALIBABA_CLOUD_ACCESS_KEY_SECRET`, `ALIYUN_GREEN_REGION_ID=cn-shanghai`, and
`ALIYUN_GREEN_ENDPOINT=green-cip.cn-shanghai.aliyuncs.com` in the backend environment.
Install `deploy/requirements-app.txt` and restart LLM after configuration changes.
The production Compose template passes these variables to LLM only.

Both LLM completion endpoints review the current user message before Agent execution
and review the final answer before returning text to Gateway/TTS. This also covers
direct answers and model fallback answers. Agent-disabled requests still pass review.
Local safety rules run first, retaining their crisis-support and privacy behavior.
Leave `GATEWAY_KEYWORD_FILTER_ENABLED=false` to avoid duplicate external input review.
Direct TTS APIs and manual memory-management APIs are outside this conversation gate.

`content_safety.py` uses the official Aliyun SDK and the tested `chat_detection`
service. Medium/high risks block; low risks pass with decision metadata. Labels with
no risk level block conservatively. Unrecognized blocked labels use the generic reply.
Label mappings are exact aliases and must be checked against the enabled vendor policy;
their presence does not claim that every category is enabled or was tested live.

Edit `safety_replies.json` to change Chinese fallback wording. Every category has
`3_5`, `6_8`, `9_11`, `12_14`, `15_17`, `adult`, and `unknown` variants. Safety age
is read from request metadata (`age`, `user_age`, or a birth-date field), then from
the persistent profile; an existing grade is mapped to an age band when exact age is
unavailable. Age changes wording only, never whether content is allowed or blocked.
Category priority is self-harm support, violence, illegal activity, sexual content,
sensitive content, privacy, profanity, and advertising. Operational failures have
a separate age-aware reply. Local high-risk rules share the same reply table.
Blocked inputs skip Agent and persistence; externally blocked outputs skip persistence.
Metadata/logs contain decisions and vendor request IDs, never API secrets or risk words.

With review enabled, SSE generation is buffered until the whole answer has been
checked, then emitted as the existing delta/done protocol. This increases time to
first text/audio. Long text is checked in 500-character windows with 100-character
overlap; cross-window semantic detection remains limited by the vendor API.
The total review timeout is `ALIYUN_CONTENT_SAFETY_TIMEOUT_SECONDS=12` per stage.
Timeouts, invalid responses, missing SDK/credentials, and text over 20,000 characters
fail closed with the temporary-unavailable reply. No automatic retries are performed.
Cancellation while generating or reviewing closes the source without publishing or
persisting an unreviewed answer.

Regression checks (vendor mocked):

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_aliyun_content_safety.py tests/test_llm_safety.py -q
```

Live review smoke check (five paid vendor requests on the short sample path;
local ASGI HTTP/SSE entry points, stub inference, no database writes):

```powershell
.\.venv\Scripts\python.exe scripts/check_aliyun_content_safety_flow.py
```

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

The registered runtime tools are calculator, dictionary, local time, unit
conversion, weather, Bocha web search, memory_search, history_search,
profile_lookup, and concept_lookup. In native mode the model
selects enabled tools through OpenAI-compatible function definitions, tool
results are sanitized before the second model round, and streaming retries a
stale pooled connection only before any upstream data has been emitted.

## 08 Branch Fusion

Conversation orchestration is adapted from the target's Gateway into this LLM
service, not copied over the current Gateway. Model credentials, the configured
Pro model, authentication, robot protocols, PostgreSQL/Redis, Qdrant/Neo4j,
embedding configuration, existing documents, and citation delivery remain local.
The branch name does not imply a Beijing-dialect prompt: none exists in the
pinned target source.

Request options can be supplied through `metadata` or `generation_config`, with
`generation_config` taking precedence:

- `answer_mode`: `fast`, `standard` (default), or `deep`; temperature defaults are
  0.25, 0.32, and 0.35. An explicit temperature overrides the default.
- Agent requests no longer forward `max_tokens`, matching the target's unlimited
  answer-mode policy. Provider/model limits still apply. `agent_enabled=false`
  bypasses these changes and preserves raw inference parameters for extraction
  and comparison scripts.
- `memory_mode`: `auto`, `off`, `lightweight`, or `full`. Standard conversation
  loads profile and recent history; older memories and concepts are tools loaded
  on demand. Explicit `full` preloads those memories. Fast mode adapts to context
  continuity. Memory preparation has mode-dependent deadlines and degrades
  independently of knowledge retrieval. Explicit `off` also disables memory tools,
  but is not a request to delete data or disable post-turn history persistence.
- `tool_selection_mode`: `native` (default), `rules`, or `llm` (JSON planner with
  rules fallback). Native mode exposes all enabled target tools. Trusted user and
  session IDs are bound by the service, never supplied by model arguments.

Rules/planner results can answer directly when `LLM_DIRECT_TOOL_ANSWER_ENABLED`
is true and `LLM_SEND_TOOL_RESULTS_TO_LLM` is false. Native tool calls retain the
second model round. Weather without a location asks for a city; the next location
reply carries the original forecast day. Pending state is user/session isolated,
expires after five minutes, and is shared through Redis across LLM workers.
Temporal history questions use PostgreSQL or the local-cache fallback directly.

Follow-up invitations use the target's connector normalization, with streaming
holding the invitation while emitting the body. Existing reference blocks are
excluded from normalization. Final deltas, `message.done.text`, and persisted
assistant text agree; Gateway's screen-only citation/TTS filtering is unchanged.

`MEMORY_POST_TURN_ASYNC=true` caches history immediately and persists it in the
background, maintaining user/assistant order and draining pending work on close.
Fast turns default to `LLM_FAST_POST_TURN_MODE=history_only`; use `full` for memory
extraction as well, or `off` to skip those writes. Async responses report
`persistence.scheduled`, not a guarantee that the database write has finished.
Persistence failures are logged. `MEMORY_POST_TURN_ASYNC=false` waits for writes.

Focused verification: `.\.venv\Scripts\python.exe -m pytest tests/test_agent02_08_fusion.py`.
