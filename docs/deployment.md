# Production Deployment

Production Compose runs the four business modules, LLM-owned Redis/PostgreSQL, Caddy, and optional monitoring helpers.

```powershell
Copy-Item deploy\.env.production.example deploy\.env.production
docker compose --env-file deploy\.env.production -f deploy\docker-compose.production.yml config
docker compose --env-file deploy\.env.production -f deploy\docker-compose.production.yml up -d --build
```

Required production properties:

- `NAWEI_ENV=production`
- `GATEWAY_AUTH_MODE=jwt`
- `LLM_PROVIDER=deepseek`
- distinct ASR, LLM, and TTS internal keys of at least 32 bytes
- real DeepSeek API key
- strong PostgreSQL password
- `LLM_AGENT_ENABLED=true`, `MEMORY_ENABLED=true`, and `LLM_MEMORY_REQUIRED=true`
- Docker service names in all Gateway upstream URLs
- valid host TTS model directory

Redis and PostgreSQL are private LLM dependencies on the internal backend network. No Agent or Memory HTTP service is added. Back up the `postgres_data` volume for long-term memory and the `redis_data` volume when short-term session recovery is required. The deployment has no knowledge-base or Qdrant container.
