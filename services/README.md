# Runtime Services

The application has four runtime services:

| Runtime | Port | Responsibility |
|---|---:|---|
| `gateway/` | 8000 | Public API and ASR/LLM/TTS orchestration |
| `asr/` | 8001 | SenseVoice speech recognition and emotion tags |
| `llm/` | 8002 | Agent, Skills, Tools, persistent Memory, DeepSeek-compatible inference and SSE |
| `tts/` | 19000 | Qwen3-TTS inference, Nana voice, and audio streaming |

There are no separate Agent, Memory, or Knowledge services. Gateway calls LLM directly; Agent and Memory are internal LLM packages backed by Redis and PostgreSQL. No knowledge base is configured.
