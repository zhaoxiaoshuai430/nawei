"""Keep local credentials from enabling paid API calls in regression tests."""

import pytest


@pytest.fixture(autouse=True)
def disable_external_moderation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALIYUN_CONTENT_SAFETY_ENABLED", "false")
