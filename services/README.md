# Runtime Services

The application has five runtime services:

| Runtime | Port | Responsibility |
|---|---:|---|
| `gateway/` | 8000 | Public API and ASR/LLM/TTS orchestration |
| `asr/` | 8001 | SenseVoice speech recognition and emotion tags |
| `llm/` | 8002 | Agent, Skills, Tools, persistent Memory, DeepSeek-compatible inference and SSE |
| `knowledge/` | 8003 | Knowledge ingestion, Qdrant vectors, Neo4j graph expansion, and retrieval |
| `tts/` | 19000 | Qwen3-TTS inference, Liliya voice, and audio streaming |

Gateway calls LLM directly. Agent and personal Memory remain internal LLM packages backed by Redis and PostgreSQL. Public document knowledge is isolated in the Knowledge service, with Qdrant storing chunk vectors and Neo4j storing entities and relations.
