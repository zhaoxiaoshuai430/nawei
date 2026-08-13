# Architecture

## Runtime topology

```text
Device / application client
        |
        v
Gateway :8000
  |-> ASR :8001
  |-> LLM :8002
  |     |-> Tutor Agent / AgentSkills / Runtime Tools
  |     |-> Redis short-term context
  |     |-> PostgreSQL profile, memories, messages, concepts
  |     `-> DeepSeek or Stub inference
  `-> TTS :19000
```

Gateway is the only public application API. ASR, LLM, and TTS are private upstream services authenticated with distinct internal keys in production.

## Request paths

```text
Text:       Gateway -> LLM
Text audio: Gateway -> LLM -> TTS
ASR only:   Gateway -> ASR
TTS only:   Gateway -> TTS
Realtime:   Gateway WebSocket -> ASR WebSocket -> LLM SSE -> optional TTS SSE
```

The voice assistant exposes only the realtime WebSocket path. Complete-audio
voice completion HTTP endpoints are intentionally not part of the public API.

Gateway is stateless with respect to conversations. It forwards the authenticated `user_id`, current message, session id, and request metadata to LLM. LLM prepares AgentSkill context, executes selected tools, runs inference, and persists the completed turn. There is no document ingestion, knowledge base, or Qdrant.

## Ownership

| Boundary | Owner |
|---|---|
| Public validation, auth, concurrency, orchestration | Gateway |
| Audio transcription and emotion metadata | ASR |
| Agent, Skills, Tools, Memory, system prompt, inference, fallback, text SSE | LLM |
| Short-term session context | LLM-owned Redis |
| Messages, profiles, long-term memories, learning concepts | LLM-owned PostgreSQL |
| Voice generation and audio SSE | TTS |
