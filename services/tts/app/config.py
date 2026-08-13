"""Application configuration and runtime paths."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = REPO_ROOT.parent.parent
MODEL_ROOT = Path(os.environ.get("NAWEI_MODEL_ROOT", str(PROJECT_ROOT / "models"))).expanduser()

try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:  # pragma: no cover - declared runtime dependency
    _load_dotenv = None

if _load_dotenv is not None:
    _load_dotenv(REPO_ROOT / ".env", override=False)
    _load_dotenv(REPO_ROOT / "cuda_env" / ".env", override=False)

# Runtime data directories
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
SESSIONS_DIR = DATA_DIR / "sessions"
REFERENCES_DIR = DATA_DIR / "references"
LOGS_DIR = DATA_DIR / "logs"

# Model caches: prefer legacy hf_download if present
_LEGACY_HF = REPO_ROOT / "hf_download"
HF_HOME = _LEGACY_HF if _LEGACY_HF.exists() else CACHE_DIR / "hf"
HF_HUB_CACHE = HF_HOME / "hub"
MODELSCOPE_CACHE = (
    _LEGACY_HF / "modelscope" if (_LEGACY_HF / "modelscope").exists() else CACHE_DIR / "modelscope"
)

ASSET_DIR = Path(os.environ.get("ASSET_DIR", str(REPO_ROOT / "assets")))
SENSEVOICE_DIR = REPO_ROOT / "SenseVoice"

# voice_lab: fine-tuned CustomVoice (speaker "nana" by default)
VOICE_LAB_DIR = REPO_ROOT / "voice_lab"
VOICE_LAB_SPEAKER = os.environ.get("VOICE_LAB_SPEAKER", "nana").strip() or "nana"

# Gateway
ASR_BASE_URL = os.environ.get("ASR_BASE_URL", "http://127.0.0.1:8001").rstrip("/")
MAX_TEXT_CHARS = 1000 / 2
MAX_TEXT_NEW_TOKENS = 640
# Chat clone: cap sampling so one reply stays one consistent voice.
CLONE_CHAT_TEMPERATURE = float(os.environ.get("CLONE_CHAT_TEMPERATURE", "0.65"))
MAX_AUDIO_BYTES = 10 * 1024 * 1024
AUDIO_TOO_LARGE_MSG = (
    "Audio file too large ({size_mb:.1f} MB). "
    "Voice cloning works best with short clips under 1 minute — please upload a shorter recording."
)

ALL_TTS_MODELS = [
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
]

_active_models_env = os.environ.get("ACTIVE_MODELS", "")
if _active_models_env:
    _allowed = {m.strip() for m in _active_models_env.split(",") if m.strip()}
    AVAILABLE_MODELS = [m for m in ALL_TTS_MODELS if m in _allowed]
else:
    AVAILABLE_MODELS = list(ALL_TTS_MODELS)


def list_available_models() -> list[str]:
    """HF presets plus every voice_lab/{speaker} with a local checkpoint."""
    models = list(AVAILABLE_MODELS)
    try:
        from app.services.voice_lab.checkpoint import list_trained_speakers, voice_lab_model_id

        for spk in list_trained_speakers():
            vid = voice_lab_model_id(spk)
            if vid not in models:
                models.append(vid)
    except Exception:
        pass
    return models


def setup_environment() -> None:
    """Set process env vars for HF / ModelScope before model imports."""
    for d in (DATA_DIR, CACHE_DIR, SESSIONS_DIR, REFERENCES_DIR, LOGS_DIR, HF_HOME, HF_HUB_CACHE, MODELSCOPE_CACHE):
        d.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE))
    os.environ.setdefault("MODELSCOPE_CACHE", str(MODELSCOPE_CACHE))
    os.environ.setdefault("ASSET_DIR", str(ASSET_DIR))

    if _load_dotenv is not None:
        _load_dotenv(REPO_ROOT / ".env", override=False)
        _load_dotenv(REPO_ROOT / "cuda_env" / ".env", override=False)
