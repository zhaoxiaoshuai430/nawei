"""Deployment credentials, environment isolation, and readiness regressions."""

from pathlib import Path

import pytest

from deploy.scripts.cloud import compose_environment, initialize, read_env
from deploy.scripts.healthcheck import is_ready
from deploy.scripts.validate_production_env import INTERNAL_KEYS, validate_environment


def production_env(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    path = tmp_path / ".env.production"
    initialize(path, "voice.nawei.cn", "ops@nawei.cn")
    env = read_env(path)
    env["LLM_API_KEY"] = "provider-test-key"
    return path, env


def test_generated_credentials_are_valid_distinct_and_not_printed(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    path, env = production_env(tmp_path)
    assert validate_environment(env) == []
    assert len({env[key] for key in INTERNAL_KEYS}) == len(INTERNAL_KEYS)
    assert env["POSTGRES_PASSWORD"] in env["DATABASE_URL"]
    assert env["GATEWAY_JWT_SECRET"] not in capsys.readouterr().out
    assert "REPLACE_WITH_YOUR_DEEPSEEK_API_KEY" in path.read_text(encoding="utf-8")
    assert any("LLM_API_KEY" in error for error in validate_environment(read_env(path)))


def test_init_never_rotates_existing_credentials(tmp_path: Path) -> None:
    path, _ = production_env(tmp_path)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        initialize(path, "voice.nawei.cn", "ops@nawei.cn")
    assert path.read_bytes() == original


@pytest.mark.parametrize("domain", ["localhost", "127.0.0.1", "voice.example.com", "https://voice.nawei.cn", "a..cn"])
def test_init_rejects_unusable_domain_without_creating_file(tmp_path: Path, domain: str) -> None:
    path = tmp_path / ".env.production"
    with pytest.raises(ValueError):
        initialize(path, domain, "ops@nawei.cn")
    assert not path.exists()


def test_local_environment_cannot_override_cloud_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, env = production_env(tmp_path)
    monkeypatch.setenv("LLM_API_KEY", "local-secret")
    monkeypatch.setenv("COMPOSE_PROFILES", "observability")
    monkeypatch.setenv("ASR_DEVICE", "cuda:0")
    child = compose_environment(path, env)
    assert child["LLM_API_KEY"] == "provider-test-key"
    assert child["ASR_DEVICE"] == "cpu"
    assert "COMPOSE_PROFILES" not in child
    assert child["NAWEI_ENV_FILE"] == str(path.resolve())


@pytest.mark.parametrize("changes,fragment", [
    ({"XIAOZHI_PUBLIC_WEBSOCKET_URL": "ws://voice.nawei.cn/xiaozhi/v1/"}, "XIAOZHI_PUBLIC"),
    ({"NAWEI_PUBLIC_ROBOT_URL": "https://another.nawei.cn"}, "NAWEI_PUBLIC"),
    ({"GATEWAY_VOICE_PREVIEW_ENABLED": "true", "LLM_WORKERS": "4"}, "LLM_WORKERS"),
    ({"EMBEDDING_DIMENSION": "384"}, "512"),
    ({"ASR_REALTIME_BACKEND": "paraformer_online"}, "ASR_REALTIME_BACKEND"),
    ({"ASR_DEVICE": "cuda:0"}, "ASR_DEVICE"),
])
def test_invalid_robot_and_model_configuration_is_rejected(tmp_path: Path, changes: dict, fragment: str) -> None:
    _, env = production_env(tmp_path)
    assert any(fragment in error for error in validate_environment({**env, **changes}))


@pytest.mark.parametrize("service,body", [
    ("tts", {"loaded": False}),
    ("gateway", {"status": "ready", "services": {}}),
    ("gateway", {"status": "ready", "services": None}),
    ("llm", {"status": "ok", "provider": "deepseek", "runtime": None}),
    ("llm", {"status": "ok", "provider": "deepseek", "runtime": {"knowledge": None}}),
    ("knowledge", {"status": "ok", "ready": True, "postgres": "ok"}),
    ("asr", None),
])
def test_http_success_alone_does_not_mean_ready(service: str, body: object) -> None:
    assert not is_ready(service, body)


def test_llm_readiness_requires_memory_and_knowledge() -> None:
    knowledge = {"status": "ok", "ready": True, **dict.fromkeys(("postgres", "redis", "qdrant", "neo4j"), "ok")}
    body = {"status": "ok", "provider": "deepseek", "runtime": {
        "memory": {"postgres": "ok", "redis": "ok"}, "knowledge": {"detail": knowledge},
    }}
    assert is_ready("llm", body)
    knowledge["qdrant"] = "unavailable"
    assert not is_ready("llm", body)
