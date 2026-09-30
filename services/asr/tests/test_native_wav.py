import io
import wave
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from runtime_tools import is_native_pcm_wav


def make_wav(rate=16000, channels=1, width=2):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(width)
        audio.setframerate(rate)
        audio.writeframes(bytes(320 * channels * width))
    return buffer.getvalue()


def test_native_wav_skips_unnecessary_transcoding():
    assert is_native_pcm_wav(make_wav())


def test_other_formats_still_require_decoder():
    for content in [make_wav(rate=24000), make_wav(channels=2), make_wav(width=1), b"invalid", make_wav()[:-8]]:
        assert not is_native_pcm_wav(content)
