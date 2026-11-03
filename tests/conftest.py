"""Keep local deployment settings out of regression tests."""

import os

import pytest

from services.llm.summary.policy import SummarySettings


# Set defaults before collection imports main.py, which loads the local .env
# and constructs a global runtime. Summary tests opt in explicitly themselves.
for name, value in SummarySettings().model_dump().items():
    os.environ[f"SUMMARY_{name.upper()}"] = str(value).lower()
os.environ["ALIYUN_CONTENT_SAFETY_ENABLED"] = "false"


@pytest.fixture(autouse=True)
def disable_external_moderation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALIYUN_CONTENT_SAFETY_ENABLED", "false")
