"""Production environment validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from deploy.scripts.validate_production_env import validate_environment


VALID_ENV = {
    "NAWEI_ENV": "production",
    "GATEWAY_AUTH_MODE": "jwt",
    "GATEWAY_ENABLE_DIAGNOSTICS": "false",
    "LLM_PROVIDER": "deepseek",
    "LLM_AGENT_ENABLED": "true",
    "MEMORY_ENABLED": "true",
    "LLM_MEMORY_REQUIRED": "true",
    "KNOWLEDGE_ENABLED": "true",
    "KNOWLEDGE_REQUIRED": "true",
    "GATEWAY_JWT_SECRET": "gateway-secret-00000000000000000001",
    "ASR_INTERNAL_API_KEY": "asr-secret-00000000000000000000001",
    "LLM_INTERNAL_API_KEY": "llm-secret-00000000000000000000001",
    "TTS_INTERNAL_API_KEY": "tts-secret-00000000000000000000001",
    "TTS_ADMIN_API_KEY": "tts-admin-secret-00000000000000000001",
    "LLM_API_KEY": "provider-test-key",
    "POSTGRES_PASSWORD": "postgres-secret-000000000000000001",
    "QDRANT_API_KEY": "qdrant-secret-0000000000000000001",
    "NEO4J_PASSWORD": "neo4j-secret-00000000000000000001",
    "KNOWLEDGE_INTERNAL_API_KEY": "knowledge-internal-000000000000000001",
    "KNOWLEDGE_ADMIN_API_KEY": "knowledge-admin-000000000000000000001",
    "EMBEDDING_API_KEY": "embedding-test-key",
    "EMBEDDING_PROVIDER": "openai_compatible",
    "GATEWAY_ASR_URL": "http://asr:8001/api/v1/audio/transcriptions/base64",
    "GATEWAY_ASR_REALTIME_URL": "ws://asr:8001/api/v1/asr/realtime",
    "GATEWAY_LLM_URL": "http://llm:8002/api/v1/llm",
    "GATEWAY_LLM_STREAM_URL": "http://llm:8002/api/v1/llm/stream",
    "GATEWAY_TTS_URL": "http://tts:19000/api/v1/audio/speech",
    "GATEWAY_TTS_STREAM_URL": "http://tts:19000/api/v1/audio/speech/stream",
    "DATABASE_URL": "postgresql://nawei:secret@postgres:5432/nawei",
    "REDIS_URL": "redis://redis:6379/0",
    "QDRANT_URL": "http://qdrant:6333",
    "NEO4J_URI": "neo4j://neo4j:7687",
}


def _set_valid_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in VALID_ENV.items():
        monkeypatch.setenv(name, value)


def test_valid_production_environment_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_valid_environment(monkeypatch)

    assert validate_environment() == []


def test_placeholder_secrets_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_valid_environment(monkeypatch)
    monkeypatch.setenv("GATEWAY_JWT_SECRET", "REPLACE_WITH_AT_LEAST_32_RANDOM_BYTES")

    errors = validate_environment()

    assert "GATEWAY_JWT_SECRET must be configured with a real value." in errors


def test_localhost_service_urls_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_valid_environment(monkeypatch)
    monkeypatch.setenv("GATEWAY_LLM_URL", "http://127.0.0.1:8002/api/v1/llm")

    errors = validate_environment()

    assert "GATEWAY_LLM_URL must use its Docker service name, not 127.0.0.1." in errors


def test_production_diagnostics_must_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_valid_environment(monkeypatch)
    monkeypatch.setenv("GATEWAY_ENABLE_DIAGNOSTICS", "true")

    errors = validate_environment()

    assert "GATEWAY_ENABLE_DIAGNOSTICS must be false." in errors


def test_production_compose_scopes_full_environment_to_config_check() -> None:
    compose = (Path(__file__).resolve().parents[1] / "deploy" / "docker-compose.production.yml").read_text(
        encoding="utf-8"
    )

    assert compose.count("env_file:") == 1
    assert "  config-check:\n    <<: *app-service\n    env_file:" in compose
    caddy = compose.split("  caddy:\n", 1)[1].split("  blackbox-exporter:\n", 1)[0]
    assert "env_file:" not in caddy
    assert "GATEWAY_JWT_SECRET" not in caddy
    assert "POSTGRES_PASSWORD" not in caddy

    gateway = compose.split("  gateway:\n", 1)[1].split("  caddy:\n", 1)[0]
    assert 'GATEWAY_ENABLE_DIAGNOSTICS: "false"' in gateway


def test_production_asr_requests_nvidia_gpu() -> None:
    compose = (Path(__file__).resolve().parents[1] / "deploy" / "docker-compose.production.yml").read_text(
        encoding="utf-8"
    )
    asr = compose.split("  asr:\n", 1)[1].split("  tts:\n", 1)[0]

    assert "ASR_DEVICE: ${ASR_DEVICE:-cuda:0}" in asr
    assert "driver: nvidia" in asr
    assert "capabilities: [gpu]" in asr


def test_deepseek_v4_pro_is_the_documented_default() -> None:
    project_root = Path(__file__).resolve().parents[1]
    paths = (
        ".env.example",
        "deploy/.env.production.example",
        "deploy/docker-compose.production.yml",
        "services/llm/README.md",
    )
    documentation = "\n".join((project_root / path).read_text(encoding="utf-8") for path in paths)

    assert "deepseek-v4-pro" in documentation
    assert "deepseek-chat" not in documentation


def test_readme_project_check_commands_exist() -> None:
    project_root = Path(__file__).resolve().parents[1]
    readme = (project_root / "README.md").read_text(encoding="utf-8")

    assert (project_root / "deploy" / "requirements-dev.txt").is_file()
    assert (project_root / "scripts" / "check_project.ps1").is_file()
    assert "deploy\\requirements-dev.txt" in readme
    assert ".\\scripts\\check_project.ps1" in readme


def test_local_asr_requirements_reuse_production_dependencies() -> None:
    project_root = Path(__file__).resolve().parents[1]
    local_requirements = (project_root / "services" / "asr" / "requirements.txt").read_text(encoding="utf-8")
    production_requirements = (project_root / "deploy" / "requirements-asr.txt").read_text(encoding="utf-8")

    assert "-r ../../deploy/requirements-asr.txt" in local_requirements
    assert "torch==2.1.0" in local_requirements
    assert "torchaudio==2.1.0" in local_requirements
    assert "websockets>=16,<17" in production_requirements
