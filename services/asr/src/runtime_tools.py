"""Runtime discovery for native audio tools used by the ASR service."""

from __future__ import annotations

import os
import io
import wave
import shutil
from pathlib import Path


def is_native_pcm_wav(content: bytes) -> bool:
    """Accept complete PCM16 mono 16 kHz WAV without transcoding."""
    try:
        with wave.open(io.BytesIO(content), "rb") as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype()) != (1, 2, 16000, "NONE"):
                return False
            frames = audio.getnframes()
            return frames > 0 and len(audio.readframes(frames)) == frames * 2
    except (wave.Error, EOFError, OSError):
        return False


def ensure_ffmpeg_on_path(runtime_dir: Path | None = None) -> Path | None:
    """Expose an available FFmpeg executable on PATH before FunASR imports."""

    existing = shutil.which("ffmpeg")
    if existing:
        return Path(existing)

    source = _find_ffmpeg_source()
    if source is None:
        return None

    tools_dir = runtime_dir or Path(__file__).resolve().parents[1] / ".runtime-tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    alias = tools_dir / "ffmpeg.exe"
    if not alias.exists() or alias.stat().st_size != source.stat().st_size:
        alias.unlink(missing_ok=True)
        try:
            os.link(source, alias)
        except OSError:
            shutil.copy2(source, alias)

    _prepend_to_path(tools_dir)
    return alias


def resolve_model_source(
    model_id: str,
    cache_root: Path,
    *,
    explicit_path: str = "",
) -> str:
    """Prefer a complete local ModelScope snapshot over a remote model ID."""

    if explicit_path.strip():
        explicit = Path(explicit_path).expanduser()
        if not explicit.is_dir():
            raise FileNotFoundError(f"Configured model directory does not exist: {explicit}")
        return str(explicit.resolve())

    snapshot = cache_root / "modelscope" / "models" / model_id.replace("/", "--") / "snapshots" / "master"
    if (snapshot / "config.yaml").is_file() and (snapshot / "model.pt").is_file():
        return str(snapshot.resolve())
    return model_id


def _find_ffmpeg_source() -> Path | None:
    explicit = os.getenv("NAWEI_FFMPEG_EXECUTABLE", "").strip()
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.is_file():
            return candidate.resolve()

    try:
        import imageio_ffmpeg
    except ImportError:
        return None

    candidate = Path(imageio_ffmpeg.get_ffmpeg_exe())
    return candidate.resolve() if candidate.is_file() else None


def _prepend_to_path(directory: Path) -> None:
    current = os.environ.get("PATH", "")
    entries = current.split(os.pathsep) if current else []
    directory_text = str(directory)
    if directory_text not in entries:
        os.environ["PATH"] = os.pathsep.join([directory_text, *entries])
