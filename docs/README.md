# Documentation

These documents describe the current four-module runtime only: Gateway, ASR,
LLM, and TTS. Agent, Skills, Tools, and Redis/PostgreSQL Memory are internal to
LLM. Documents for standalone Agent/Memory/Knowledge services or vector stores
are obsolete and must not be added back as active architecture references.

| Document | Purpose |
|---|---|
| [Architecture](architecture.md) | Four-service boundaries and request paths |
| [Development](development.md) | Local workflow and tests |
| [Operations](operations.md) | Start, status, stop, and troubleshooting |
| [Deployment](deployment.md) | Production Compose deployment |

Service references:

- [Gateway](../services/gateway/README.md)
- [ASR](../services/asr/README.md)
- [LLM](../services/llm/README.md)
- [TTS](../services/tts/README.md)
