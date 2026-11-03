"""Audio encoding and format conversion helpers."""
from __future__ import annotations

import base64
import io

import numpy as np
import soundfile as sf


class AudioFormatError(Exception):
    """Raised when generated audio cannot satisfy the output format contract."""

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


def encode_audio_b64(
    audio: np.ndarray,
    source_sr: int,
    *,
    output_format: str,
    target_sr: int,
    channels: int,
) -> str:
    """Encode model audio according to the public AudioConfig contract."""

    normalized_format = output_format.strip().lower()
    if normalized_format not in {"wav", "pcm16", "mp3"}:
        raise AudioFormatError(f"Unsupported output audio format: {output_format}")
    if source_sr <= 0 or target_sr <= 0:
        raise AudioFormatError("Audio sample rates must be positive.")
    if channels < 1:
        raise AudioFormatError("Audio channels must be positive.")

    samples = np.asarray(audio, dtype=np.float32).squeeze()
    if samples.ndim != 1:
        raise AudioFormatError("TTS model output must be mono audio.")
    samples = _resample_audio(samples, source_sr, target_sr)
    samples = np.clip(samples, -1.0, 1.0)
    if channels > 1:
        samples = np.repeat(samples[:, np.newaxis], channels, axis=1)

    if normalized_format == "pcm16":
        encoded = (samples * 32767.0).astype("<i2", copy=False).tobytes()
    else:
        buffer = io.BytesIO()
        kwargs = {"format": normalized_format.upper()}
        if normalized_format == "wav":
            kwargs["subtype"] = "PCM_16"
        elif normalized_format == "mp3":
            kwargs["subtype"] = "MPEG_LAYER_III"
        try:
            sf.write(buffer, samples, target_sr, **kwargs)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise AudioFormatError(f"Failed to encode {normalized_format} audio: {exc}") from exc
        encoded = buffer.getvalue()
    return base64.b64encode(encoded).decode("ascii")


def _resample_audio(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr or audio.size == 0:
        return audio
    target_size = max(1, round(audio.size * target_sr / source_sr))
    if audio.size == 1:
        return np.full(target_size, audio[0], dtype=np.float32)
    source_positions = np.arange(audio.size, dtype=np.float64) / source_sr
    target_positions = np.arange(target_size, dtype=np.float64) / target_sr
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def concat_audio(audio_list) -> np.ndarray:
    if isinstance(audio_list, np.ndarray):
        return audio_list.astype(np.float32).squeeze()
    parts = [np.array(a, dtype=np.float32).squeeze() for a in audio_list if len(a) > 0]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
