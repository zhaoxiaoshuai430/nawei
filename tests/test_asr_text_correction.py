"""Contextual ASR transcript correction regressions."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def correction_module(monkeypatch: pytest.MonkeyPatch):
    source = Path(__file__).resolve().parents[1] / "services" / "asr" / "src"
    monkeypatch.syspath_prepend(str(source))
    return importlib.import_module("text_correction")


def test_default_correction_only_changes_observed_command_phrase(correction_module, monkeypatch) -> None:
    monkeypatch.delenv("ASR_TEXT_CORRECTIONS_JSON", raising=False)
    result = correction_module.correct_transcript("一加一等于几，请指回答结果")
    assert result.text == "一加一等于几，请只回答结果"
    assert result.applied[0]["source"] == "请指回答结果"
    assert correction_module.correct_transcript("请指出这道题的错误").text == "请指出这道题的错误"


def test_configured_corrections_extend_defaults(correction_module, monkeypatch) -> None:
    monkeypatch.setenv("ASR_TEXT_CORRECTIONS_JSON", '{"小为同学":"小维同学"}')
    result = correction_module.correct_transcript("小为同学，请知回答结果")
    assert result.text == "小维同学，请只回答结果"
    assert len(result.applied) == 2


def test_corrections_can_be_disabled(correction_module, monkeypatch) -> None:
    monkeypatch.setenv("ASR_TEXT_CORRECTIONS_ENABLED", "false")
    assert correction_module.correct_transcript("请指回答结果").text == "请指回答结果"
