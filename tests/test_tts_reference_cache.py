"""Reference cache publication must never expose partially written audio."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def cache(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[1] / "services/tts/app/utils/cache.py"
    spec = importlib.util.spec_from_file_location("reference_cache_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.tempfile, "gettempdir", lambda: str(tmp_path))
    return module


@pytest.mark.parametrize("existing", [b"short", b"wrong-audio"])
def test_existing_corrupt_cache_is_repaired(cache, tmp_path, existing):
    content = b"valid-audio"
    path = tmp_path / f"faster_qwen3_tts_ref_{hashlib.sha1(content).hexdigest()}.wav"
    path.write_bytes(existing)
    assert Path(cache.get_cached_ref_path(content)) == path
    assert path.read_bytes() == content


def test_failed_publication_does_not_leave_partial_final_file(cache, tmp_path, monkeypatch):
    def failed_replace(source, target):
        assert Path(source).read_bytes() == b"audio"
        assert not Path(target).exists()
        raise OSError("disk unavailable")

    monkeypatch.setattr(cache.os, "replace", failed_replace)
    with pytest.raises(OSError, match="disk unavailable"):
        cache.get_cached_ref_path(b"audio")
    assert list(tmp_path.iterdir()) == []


def test_concurrent_callers_share_complete_cache_and_recover_deleted_file(cache):
    content = b"sample" * 10000
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(cache.get_cached_ref_path, [content] * 24))
    assert len(set(results)) == 1
    path = Path(results[0])
    assert path.read_bytes() == content
    path.unlink()
    assert cache.get_cached_ref_path(content) == str(path)
    assert path.read_bytes() == content
    assert list(path.parent.iterdir()) == [path]


def test_valid_cache_is_reused_without_replacement(cache, monkeypatch):
    path = cache.get_cached_ref_path(b"audio")

    def unexpected_replace(*args):
        pytest.fail("Valid cached reference must not be rewritten")

    monkeypatch.setattr(cache.os, "replace", unexpected_replace)
    assert cache.get_cached_ref_path(b"audio") == path


def test_interrupted_write_leaves_existing_target_intact(cache, tmp_path, monkeypatch):
    content = b"new-reference"
    path = tmp_path / f"faster_qwen3_tts_ref_{hashlib.sha1(content).hexdigest()}.wav"
    path.write_bytes(b"old-incomplete-reference")
    create_file = cache.tempfile.NamedTemporaryFile

    class InterruptedFile:
        def __init__(self, **kwargs):
            self.file = create_file(**kwargs)
            self.name = self.file.name

        def __enter__(self):
            return self

        def write(self, value):
            self.file.write(value[:3])
            raise OSError("write interrupted")

        def __exit__(self, *args):
            self.file.close()

    monkeypatch.setattr(cache.tempfile, "NamedTemporaryFile", InterruptedFile)
    with pytest.raises(OSError, match="write interrupted"):
        cache.get_cached_ref_path(content)
    assert path.read_bytes() == b"old-incomplete-reference"
    assert list(tmp_path.iterdir()) == [path]
