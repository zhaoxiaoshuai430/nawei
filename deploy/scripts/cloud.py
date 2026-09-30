"""Prepare and operate the single Linux/NVIDIA cloud deployment."""

from __future__ import annotations

import argparse
import os
import platform
import secrets
import shutil
import subprocess
from pathlib import Path

from dotenv import dotenv_values, set_key

from deploy.scripts.validate_production_env import SECRET_KEYS, valid_domain, validate_environment

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "deploy/.env.production.example"
COMPOSE = ROOT / "deploy/docker-compose.production.yml"
EMBEDDINGS = ROOT / "models/embeddings"
CACHE_NAME = "models--Qdrant--bge-small-zh-v1.5"


def read_env(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"Environment file not found: {path}")
    return {key: value for key, value in dotenv_values(path, interpolate=False).items() if value is not None}


def initialize(path: Path, domain: str, email: str) -> None:
    if not valid_domain(domain):
        raise ValueError("Use the actual DNS hostname, without scheme, port or path.")
    values = {key: secrets.token_hex(32) for key in SECRET_KEYS if key != "LLM_API_KEY"}
    values.update({
        "NAWEI_DOMAIN": domain,
        "TLS_EMAIL": email,
        "NAWEI_PUBLIC_ROBOT_URL": f"https://{domain}",
        "XIAOZHI_PUBLIC_WEBSOCKET_URL": f"wss://{domain}/xiaozhi/v1/",
        "NAWEI_ENV_FILE": path.name,
        "DATABASE_URL": f"postgresql://nawei:{values['POSTGRES_PASSWORD']}@postgres:5432/nawei",
    })
    errors = validate_environment({**read_env(TEMPLATE), **values, "LLM_API_KEY": "validation-only"})
    if errors:
        raise ValueError("\n".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects an existing deployment's credentials.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as output:
        output.write(TEMPLATE.read_text(encoding="utf-8"))
    for key, value in values.items():
        set_key(str(path), key, value, quote_mode="always")
    print(f"Created {path}. Set LLM_API_KEY in this file before check/up.")


def verify_embeddings(cache: Path) -> None:
    from fastembed import TextEmbedding

    model = TextEmbedding(
        model_name="BAAI/bge-small-zh-v1.5", cache_dir=str(cache),
        local_files_only=True, threads=2, providers=["CPUExecutionProvider"],
    )
    vector = next(iter(model.embed(["cloud deployment readiness"])))
    if len(vector) != 512:
        raise ValueError("Embedding model must produce 512 dimensions.")
    print("Offline embedding inference: 512 dimensions, ready")


def prepare_models(source: Path) -> None:
    destination = EMBEDDINGS / CACHE_NAME
    if not destination.exists():
        # Dereference Windows cache links so the upload is portable to Linux.
        shutil.copytree(source / CACHE_NAME, destination, symlinks=False)
    verify_embeddings(EMBEDDINGS)


def check_assets(env: dict[str, str]) -> None:
    required = [
        ROOT / "models/asr/SenseVoiceSmall/model.pt",
        ROOT / "models/asr/fsmn-vad/model.pt",
        ROOT / "models/manifest.json",
        ROOT / "models/tts/liliya/checkpoint-epoch-1/config.json",
        ROOT / "models/tts/liliya/checkpoint-epoch-1/speech_tokenizer/model.safetensors",
    ]
    if env.get("ASR_REALTIME_BACKEND") == "paraformer":
        required.extend(ROOT / "models/asr/paraformer-online" / name for name in (
            "config.yaml", "configuration.json", "model.pt", "tokens.json", "am.mvn",
        ))
    # Docker restores the excluded merged weight from these uploaded parts.
    required.extend(ROOT / f"models/tts/liliya/checkpoint-epoch-1/model.safetensors.part{i:03}" for i in range(3))
    if env.get("EMBEDDING_PROVIDER") == "fastembed":
        if not any((EMBEDDINGS / CACHE_NAME).rglob("*.onnx")):
            raise ValueError("Run prepare-models before uploading models/embeddings.")
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise ValueError("Missing model assets: " + ", ".join(missing))


def compose_environment(path: Path, values: dict[str, str]) -> dict[str, str]:
    # Compose shell variables take precedence over --env-file. Keep only host
    # runtime/connection settings so local development secrets cannot override it.
    allowed = {
        "PATH", "HOME", "USERPROFILE", "SYSTEMROOT", "WINDIR", "TEMP", "TMP",
        "PROGRAMDATA", "PROGRAMFILES", "PROGRAMFILES(X86)", "APPDATA", "LOCALAPPDATA",
        "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG", "DOCKER_CERT_PATH", "DOCKER_TLS_VERIFY",
        "SSH_AUTH_SOCK", "XDG_RUNTIME_DIR",
    }
    result = {key: value for key, value in os.environ.items() if key.upper() in allowed}
    result.update(values)
    result["NAWEI_ENV_FILE"] = str(path.resolve())
    return result


def compose(path: Path, values: dict[str, str], *arguments: str) -> None:
    subprocess.run(
        ["docker", "compose", "--env-file", str(path.resolve()), "-f", str(COMPOSE), *arguments],
        cwd=ROOT, env=compose_environment(path, values), check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=ROOT / "deploy/.env.production")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--domain", required=True)
    init.add_argument("--tls-email", required=True)
    models = commands.add_parser("prepare-models")
    models.add_argument("--source", type=Path, default=ROOT / ".runtime/embedding-cache")
    commands.add_parser("check")
    commands.add_parser("up")
    commands.add_parser("status")
    args = parser.parse_args()
    try:
        if args.command == "init":
            initialize(args.env_file.resolve(), args.domain, args.tls_email)
            return
        if args.command == "prepare-models":
            prepare_models(args.source.resolve())
            return
        values = read_env(args.env_file)
        errors = validate_environment(values)
        if errors:
            raise ValueError("Invalid production configuration:\n- " + "\n- ".join(errors))
        if args.command == "status":
            compose(args.env_file, values, "ps", "--all")
            return
        check_assets(values)
        compose(args.env_file, values, "config", "--quiet")
        print("Configuration and model asset presence: ready (cloud runtime not yet verified)")
        if args.command == "up":
            if platform.system() != "Linux":
                raise ValueError("Run up on the Linux cloud host with NVIDIA Container Toolkit installed.")
            subprocess.run(["nvidia-smi", "-L"], check=True)
            compose(args.env_file, values, "up", "--build", "--detach", "--wait", "--wait-timeout", "1200")
            compose(args.env_file, values, "ps", "--all")
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
