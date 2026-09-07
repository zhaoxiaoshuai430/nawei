from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


GATEWAY_URL = "http://127.0.0.1:8000/api/v1/text/completions"
OUTPUT_PATH = Path("logs/atlas-llm-auto-retrieval-answers.json")
QUESTIONS = [
    "数学史属于哪个学科？",
    "自然科学在图谱中的完整路径和分类来源是什么？",
    "这个知识图谱共有多少节点，教育课程映射有多少条？",
    "中国3到5岁儿童的课程主要覆盖哪些教育知识领域？",
    "中国人民大学附属中学的课程结构和主要教学方式是什么？",
    "上海中学国际部提供哪些课程或资格路径？",
]


def call_llm(question: str, index: int) -> dict[str, Any]:
    suffix = f"auto-20260825-q{index:02d}"
    user_id = f"atlas-sample-user-{suffix}"
    payload = {
        "request_id": f"atlas-sample-{suffix}",
        "trace_id": f"atlas-sample-{suffix}",
        "session_id": f"atlas-answer-{suffix}",
        "user_id": user_id,
        "message": {"role": "user", "content": question, "metadata": {}},
        "stream": False,
        "output_audio": False,
        "metadata": {},
    }
    request = urllib.request.Request(
        GATEWAY_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "X-User-ID": user_id,
        },
        method="POST",
    )

    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            if attempt < 3:
                time.sleep(attempt * 2)

    raise RuntimeError(f"LLM request failed after 3 attempts: {last_error}")


def validate_answer(answer: str) -> dict[str, Any]:
    source_names = [
        "人类知识图谱总表0824(1).xlsx",
        "Human_Knowledge_Education_Atlas_完整统一版_v2.html",
    ]
    return {
        "has_inline_citation": "[资料" in answer,
        "has_reference_section": "参考来源" in answer,
        "has_known_source": any(source_name in answer for source_name in source_names),
        "has_locator": any(
            locator in answer
            for locator in ("第", "行", "Sheet", "定位：", "HTML")
        ),
    }


def main() -> None:
    results: list[dict[str, Any]] = []
    for index, question in enumerate(QUESTIONS, start=1):
        print(f"[{index}/{len(QUESTIONS)}] {question}", flush=True)
        response = call_llm(question, index)
        answer = str(response.get("text", ""))
        result = {
            "index": index,
            "question": question,
            "answer": answer,
            "validation": validate_answer(answer),
            "usage": response.get("usage", {}),
            "provider": response.get("metadata", {}).get("provider"),
            "model": response.get("metadata", {}).get("model"),
            "knowledge": response.get("metadata", {})
            .get("runtime", {})
            .get("knowledge", {}),
            "message_id": response.get("message_id"),
            "session_id": response.get("session_id"),
        }
        results.append(result)
        print(answer, flush=True)
        print("-" * 80, flush=True)

    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "gateway_url": GATEWAY_URL,
        "question_count": len(results),
        "all_citations_valid": all(
            all(item["validation"].values()) for item in results
        ),
        "results": results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved report: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
