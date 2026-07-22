# LLM Service

Standalone LLM service used by Gateway. It accepts `LLMRequest` and returns `LLMResponse`.

## Runtime

| Item | Value |
|---|---|
| Path | `D:\daima\nawei\nawei\services\llm_service` |
| Default port | `8002` |
| App | `services.llm_service.main:app` |

## Start

```powershell
cd D:\daima\nawei\nawei
.\.venv\Scripts\python.exe -m uvicorn services.llm_service.main:app --host 0.0.0.0 --port 8002
```

Health check:

```text
GET http://127.0.0.1:8002/health
```

## API

```text
POST http://127.0.0.1:8002/api/v1/llm
```

Request body:

```json
{
  "request_id": "llm-1",
  "trace_id": "trace-1",
  "session_id": "session-1",
  "messages": [
    {"role": "user", "content": "你好", "name": null, "metadata": {}}
  ],
  "stream": false,
  "tools": [],
  "generation_config": {
    "temperature": 0.2
  }
}
```

Response body:

```json
{
  "request_id": "llm-1",
  "text": "你好！有什么可以帮你？",
  "finish_reason": "stop",
  "usage": {
    "input_tokens": 0,
    "output_tokens": 0
  },
  "metadata": {
    "provider": "deepseek",
    "model": "deepseek-v4-pro"
  }
}
```

### Streaming API

```text
POST http://127.0.0.1:8002/api/v1/llm/stream
Accept: text/event-stream
```

The request body is `LLMRequest` with `stream: true`. The service translates the
DeepSeek stream into canonical `message.delta`, `message.done`, and `error` SSE
events defined by `app_protocols.StreamEvent`.

## Modes

### Stub mode

Default mode if no real API configuration is provided.

```env
LLM_PROVIDER=stub
```

### DeepSeek mode

```env
LLM_PROVIDER=deepseek
LLM_API_BASE_URL=https://api.deepseek.com
LLM_API_KEY=your_api_key
LLM_MODEL=deepseek-v4-pro
LLM_TIMEOUT_SECONDS=60
```

The service calls the DeepSeek chat completions API:

```text
{LLM_API_BASE_URL}/chat/completions
```

If `LLM_API_BASE_URL` already ends with `/chat/completions`, it will not append the path again.
