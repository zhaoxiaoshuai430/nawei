"""Preset reference audio for TTS demo."""
from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from app.config import ASSET_DIR
from app.utils.cache import get_cached_ref_path

DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_AUDIO_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_TRANSCRIPT_BYTES = 1024 * 1024
DOWNLOAD_USER_AGENT = "NaweiTTS/1.0"

PRESET_TRANSCRIPTS = ASSET_DIR / "samples" / "parity" / "icl_transcripts.txt"
PRESET_REFS = [
    ("ref_audio_1", ASSET_DIR / "ref_audio_11.wav", "女主持人"),
    ("ref_audio_5", ASSET_DIR / "ref_audio_15.wav", "男主持人"),
    ("ref_audio_2", ASSET_DIR / "ref_audio_12.wav", "女老者"),
    ("ref_audio_3", ASSET_DIR / "ref_audio_13.wav", "男老者"),
    ("ref_audio_4", ASSET_DIR / "ref_audio_14.wav", "儿童"),
]

_GITHUB_RAW = "https://raw.githubusercontent.com/andimarafioti/faster-qwen3-tts/main"
_PRESET_REMOTE = {
    "ref_audio_1": f"{_GITHUB_RAW}/ref_audio.wav",
    "ref_audio_2": f"{_GITHUB_RAW}/ref_audio_2.wav",
    "ref_audio_3": f"{_GITHUB_RAW}/ref_audio_3.wav",
    "ref_audio_4": f"{_GITHUB_RAW}/ref_audio.wav",
    "ref_audio_5": f"{_GITHUB_RAW}/ref_audio_2.wav",
}
_TRANSCRIPT_REMOTE = f"{_GITHUB_RAW}/samples/parity/icl_transcripts.txt"

preset_refs: dict[str, dict] = {}


def _read_positive_float_env(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _read_positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _valid_wav(path: Path) -> bool:
    try:
        if path.stat().st_size < 12:
            return False
        with path.open("rb") as handle:
            header = handle.read(12)
        return header[:4] == b"RIFF" and header[8:12] == b"WAVE"
    except OSError:
        return False


def _valid_transcript(path: Path) -> bool:
    try:
        return bool(path.read_text(encoding="utf-8").strip())
    except (OSError, UnicodeError):
        return False


def _download_asset(url: str, destination: Path, *, max_bytes: int) -> None:
    if urlsplit(url).scheme.lower() != "https":
        raise ValueError("Preset assets must use HTTPS")
    destination.parent.mkdir(parents=True, exist_ok=True)
    timeout = _read_positive_float_env("TTS_PRESET_DOWNLOAD_TIMEOUT_SECONDS", DEFAULT_DOWNLOAD_TIMEOUT_SECONDS)
    request = Request(url, headers={"User-Agent": DOWNLOAD_USER_AGENT, "Accept": "*/*"})
    temporary_path: Path | None = None
    try:
        with urlopen(request, timeout=timeout) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > max_bytes:
                raise ValueError(f"Preset asset exceeds {max_bytes} bytes")
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                total = 0
                while chunk := response.read(64 * 1024):
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError(f"Preset asset exceeds {max_bytes} bytes")
                    temporary.write(chunk)
                if total == 0:
                    raise ValueError("Preset asset is empty")
                temporary.flush()
                os.fsync(temporary.fileno())
        if destination.suffix.lower() == ".wav":
            valid = _valid_wav(temporary_path)
        else:
            valid = _valid_transcript(temporary_path)
        if not valid:
            raise ValueError("Preset asset has an invalid format")
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def fetch_preset_assets() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    PRESET_TRANSCRIPTS.parent.mkdir(parents=True, exist_ok=True)
    max_transcript_bytes = _read_positive_int_env(
        "TTS_PRESET_MAX_TRANSCRIPT_BYTES", DEFAULT_MAX_TRANSCRIPT_BYTES
    )
    max_audio_bytes = _read_positive_int_env("TTS_PRESET_MAX_AUDIO_BYTES", DEFAULT_MAX_AUDIO_BYTES)
    if not _valid_transcript(PRESET_TRANSCRIPTS):
        try:
            _download_asset(_TRANSCRIPT_REMOTE, PRESET_TRANSCRIPTS, max_bytes=max_transcript_bytes)
        except Exception as exc:
            print(f"Warning: could not fetch transcripts: {exc}")
    for key, path, _ in PRESET_REFS:
        if not _valid_wav(path) and key in _PRESET_REMOTE:
            try:
                _download_asset(_PRESET_REMOTE[key], path, max_bytes=max_audio_bytes)
                print(f"Downloaded {path.name}")
            except Exception as exc:
                print(f"Warning: could not fetch {key}: {exc}")


def _load_preset_transcripts() -> dict[str, str]:
    if not _valid_transcript(PRESET_TRANSCRIPTS):
        return {}
    transcripts = {}
    for line in PRESET_TRANSCRIPTS.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key_part, text = line.split(":", 1)
        key = key_part.split("(")[0].strip()
        transcripts[key] = text.strip()
    return transcripts


def load_preset_refs() -> None:
    global preset_refs
    if os.environ.get("TTS_SKIP_PRESET_FETCH", "").strip().lower() not in {"1", "true", "yes"}:
        fetch_preset_assets()
    transcripts = _load_preset_transcripts()
    preset_refs.clear()
    for key, path, label in PRESET_REFS:
        if not _valid_wav(path):
            continue
        content = path.read_bytes()
        cached_path = get_cached_ref_path(content)
        preset_refs[key] = {
            "id": key,
            "label": label,
            "filename": path.name,
            "path": cached_path,
            "ref_text": transcripts.get(key, ""),
            "audio_b64": base64.b64encode(content).decode(),
        }


def prime_preset_voice_cache(model) -> None:
    from app.services.tts.warmup import warm_voice_clone_cache

    if not preset_refs:
        return
    for preset in preset_refs.values():
        ref_path = preset["path"]
        ref_text = preset["ref_text"]
        for xvec_only in (True, False):
            try:
                warm_voice_clone_cache(
                    model,
                    ref_audio=ref_path,
                    ref_text=ref_text,
                    xvec_only=xvec_only,
                )
            except Exception:
                continue
