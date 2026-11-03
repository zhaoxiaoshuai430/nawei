from copy import deepcopy

import pytest

from scripts.backfill_verified_summary_report import verify_report


def fixture():
    report = {"user_id": "owner", "session_id": "session", "requested_rounds": 1, "completed_rounds": 1,
              "results": [{"round": 1, "prompt": "你好", "answer": "你好！",
                           "metadata": {"persistence": {"saved_turn": True}}}]}
    messages = [{"user_id": "owner", "session_id": "session", "role": role, "content": content}
                for role, content in (("user", "你好"), ("assistant", "你好！"))]
    return report, messages


def test_exact_report_required():
    report, messages = fixture()
    assert len(verify_report(report, messages)) == 1
    for key, value in (("content", "changed"), ("role", "assistant"), ("user_id", "other"), ("session_id", "other")):
        altered = deepcopy(messages)
        altered[0][key] = value
        with pytest.raises(ValueError):
            verify_report(report, altered)
    with pytest.raises(ValueError):
        verify_report(report, messages[:1])
    report["results"][0]["metadata"]["fallback"] = True
    with pytest.raises(ValueError):
        verify_report(report, messages)
