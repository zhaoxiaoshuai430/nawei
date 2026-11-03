"""Real-model evaluation over labeled synthetic conversations; writes no user memory.

Run after loading model configuration: python -m scripts.check_summary_understanding.
This complements HTTP/DB tests; synthetic dialogue is never presented as real user history.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from dotenv import load_dotenv

from services.llm.agents.summary_agent import ConversationSummaryAgent, prompt_overhead
from services.llm.summary.context import read_context
from services.llm.summary.policy import SummarySettings
from services.llm.summary.repository import normalize
from services.llm.summary.snapshot import pack


async def run() -> bool:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    settings = SummarySettings.from_env().model_copy(update={"enabled": True, "apply_to_prompt": True,
        "profile_promotion_enabled": True, "input_max_tokens": 110000, "output_max_tokens": 4000,
        "timeout_seconds": 300, "prompt_max_tokens": 2000})
    dialogue = [
        ("我这周六前要完成数学异分母加法三道练习。我喜欢先举例再解释，每次只给一个提示。", "先看一道分数加法。"),
        ("我算1/2+1/3=2/5，我不会通分，这道题还没完成。", "分母不能直接相加，先找公倍数6。"),
        ("隔天再练，我还是把1/3+1/4算成2/7了，通分又错了。", "画一块分成12份的饼，分别取4份和3份。"),
        ("用饼的例子我明白了是7/12，但还需要你提示；我仍然喜欢先举例再解释，每次一个提示。", "可以继续练习。"),
        ("今天我独立算了1/2+1/3，公倍数是6，变成3/6+2/6=5/6，这次没有看提示。", "计算正确。"),
        ("三道数学练习已全部完成，我也能独立通分了。周六练习的目标完成，下周改为练分数减法。", "可以从分数减法开始。"),
        ("这几次先画饼再说通分对我很有效，我依然喜欢先举例后解释，一次一个提示。", "会参考你的反馈。"),
        ("今天我有点累，下周考试也有些压力。这会儿讲慢一点，每次少讲一点就好。", "可以先做少量练习。"),
    ]
    now = datetime.now(timezone.utc)
    rows = [{"turn_id": str(uuid4()), "user_message_id": str(uuid4()), "assistant_message_id": str(uuid4()),
             "session_id": f"synthetic-{index // 2}", "completed_seq": index+1,
             "user_text": user, "assistant_text": assistant, "completed_at": now-timedelta(minutes=20-index)}
            for index, (user, assistant) in enumerate(dialogue)]
    report: dict = {"fixture": "synthetic_dialogue_real_model_no_user_memory_writes", "passed": False}
    agent = ConversationSummaryAgent(settings)
    try:
        report["stage"] = "first_generation"
        print("Generating first summary from four synthetic turns", flush=True)
        # Keep each real provider request small enough for the local safety-review
        # deadline; the second request still has four later turns plus references.
        first = pack(list(reversed(rows[:2])), None, [], {}, settings.input_max_tokens-prompt_overhead())
        async with asyncio.timeout(settings.timeout_seconds):
            candidate, model = await agent.generate(first, "understanding-first-" + uuid4().hex)
        previous, _ = normalize(candidate, first)
        previous["summary_revision"] = 1
        report.update(model=model, first=previous)
        print("PASS: first real-model summary", flush=True)
        second = pack(list(reversed(rows[2:])), previous, rows[:2], first, settings.input_max_tokens-prompt_overhead())
        report["stage"] = "comparison_generation"
        print("Comparing new turns with the evidenced previous summary", flush=True)
        async with asyncio.timeout(settings.timeout_seconds):
            candidate, _ = await agent.generate(second, "understanding-second-" + uuid4().hex)
        current, _ = normalize(candidate, second)
        current["summary_revision"] = 2
        report.update(second=current, previous_summary_revision=second["previous_summary_revision"])
        checks = {
            "compared_previous_summary": second["previous_summary_revision"] == 1,
            "resolved_problem_or_goal": any(c["progress"] == "resolved" for c in current["claims"]),
            "longitudinal_trend": any(c["trend"] for c in current["learning_observations"]),
            "teaching_feedback": bool(current["teaching_strategy_observations"]),
            "temporary_state": bool(current["temporary_states"]),
            # A model may conservatively omit a profile; the server gate is
            # still tested here by requiring every emitted item to be eligible.
            "stable_profile_gate": all(item.get("corroborated") and item.get("evidence_turn_count", 0) >= 3
                                        and item.get("evidence_session_count", 0) >= 2
                                        for item in current["stable_profile"]),
        }

        async def read(_):
            return {"summary": current, "enabled": True, "apply_to_prompt": True, "memory_epoch": 0}

        context = await read_context(SimpleNamespace(settings=settings, read=read), "synthetic", "数学分数复习怎么安排", "full")
        report["selected_context"] = list(context.entries) if context else []
        checks["online_context_selected"] = bool(context)
        report["checks"] = checks
        report["passed"] = all(checks.values())
        report["stage"] = "complete"
        for name, passed in checks.items():
            print(("PASS: " if passed else "FAIL: ") + name, flush=True)
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        print("FAIL: " + type(exc).__name__, flush=True)
    finally:
        target = Path(".runtime/summary-v2-understanding.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Report: {target}", flush=True)
    return bool(report["passed"])


if __name__ == "__main__":
    raise SystemExit(0 if asyncio.run(run()) else 1)
