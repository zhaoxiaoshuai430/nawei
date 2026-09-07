"""Regression tests for the bundled Liliya TTS checkpoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.restore_models import restore_models


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_manifest_bundles_only_latest_liliya_checkpoint() -> None:
    manifest = json.loads((PROJECT_ROOT / "models" / "manifest.json").read_text(encoding="utf-8"))
    tts_paths = [entry["path"] for entry in manifest["files"] if entry["path"].startswith("tts/")]

    assert tts_paths
    assert all(path.startswith("tts/liliya/checkpoint-epoch-1/") for path in tts_paths)


def test_docker_bundle_restores_model_without_shipping_parts() -> None:
    dockerignore = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    dockerfile = (PROJECT_ROOT / "deploy" / "Dockerfile.tts").read_text(encoding="utf-8")

    assert "/models/tts/liliya/checkpoint-epoch-0" in dockerignore
    assert "/models/tts/liliya/checkpoint-epoch-1/model.safetensors" in dockerignore
    assert "/models/tts/liliya/checkpoint-epoch-1/model.safetensors" in gitignore
    assert "/models/tts/nana/checkpoint-epoch-7/model.safetensors" not in gitignore
    assert "FROM ${TTS_BASE_IMAGE} AS tts-models" in dockerfile
    assert "--remove-parts" in dockerfile
    assert "COPY --from=tts-models" in dockerfile


def test_restore_models_can_remove_validated_parts(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    target = model_root / "tts" / "speaker" / "checkpoint-epoch-1" / "model.safetensors"
    parts = [target.with_name(f"model.safetensors.part{index:03d}") for index in range(2)]
    content = b"validated model content"
    midpoint = len(content) // 2
    parts[0].parent.mkdir(parents=True)
    parts[0].write_bytes(content[:midpoint])
    parts[1].write_bytes(content[midpoint:])
    manifest = {
        "version": 1,
        "files": [
            {
                "path": target.relative_to(model_root).as_posix(),
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "parts": [part.relative_to(model_root).as_posix() for part in parts],
            }
        ],
    }
    (model_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    restore_models(model_root, remove_parts=True)

    assert target.read_bytes() == content
    assert all(not part.exists() for part in parts)


def test_liliya_examples_match_checkpoint_speaker() -> None:
    model_config = json.loads(
        (PROJECT_ROOT / "models" / "tts" / "liliya" / "checkpoint-epoch-1" / "config.json").read_text(
            encoding="utf-8"
        )
    )
    speaker_ids = model_config["talker_config"]["spk_id"]
    documentation = "\n".join(
        (PROJECT_ROOT / path).read_text(encoding="utf-8")
        for path in ("README.md", "services/README.md", "services/tts/README.md", "app_protocols/README.md")
    )

    assert set(speaker_ids) == {"liliya"}
    assert "voice_lab/nana" not in documentation
    assert '"speaker_id": "nana"' not in documentation


def test_tts_status_does_not_expose_an_absolute_checkpoint_path() -> None:
    source = (
        PROJECT_ROOT / "services" / "tts" / "app" / "services" / "voice_lab" / "checkpoint.py"
    ).read_text(encoding="utf-8")

    assert '"checkpoint": ckpt.name if ckpt else None' in source
    assert '"checkpoint": str(ckpt) if ckpt else None' not in source


def test_asr_audio_conversion_avoids_insecure_mktemp() -> None:
    source = (PROJECT_ROOT / "services" / "asr" / "src" / "main.py").read_text(encoding="utf-8")

    assert "tempfile.mktemp" not in source


def test_tts_audio_conversion_avoids_insecure_mktemp() -> None:
    source = (PROJECT_ROOT / "services" / "tts" / "app" / "utils" / "audio.py").read_text(encoding="utf-8")

    assert "tempfile.mktemp" not in source
