"""Regenerate a revised question set with the exact original generation policy."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import shutil
import sys

from docx import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_light_health_comparison as original
from services.knowledge.schemas import KnowledgeContext


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_inputs(evidence, question_set):
    cases, questions = evidence["cases"], question_set["questions"]
    if len(cases) != 30 or len(questions) != 30:
        raise ValueError("Exactly 30 questions required")
    if len({item["question"] for item in questions}) != 30:
        raise ValueError("Duplicate questions")
    for index, (case, question) in enumerate(zip(cases, questions), 1):
        if case["index"] != index or question["index"] != index:
            raise ValueError("Question order changed")
        if any(case[field] != question[field] for field in ("category", "question")):
            raise ValueError("Evidence does not match selected questions")
        vector = KnowledgeContext.model_validate(case["vector_context"])
        graph = KnowledgeContext.model_validate(case["graph_context"])
        if vector.relations or vector.resolved_entities or vector.relation_groups or graph.chunks:
            raise ValueError("Mixed retrieval branches")
        vector_prompt = vector.render_prompt(max_chars=7000)
        graph_prompt = graph.render_prompt(max_chars=5000)
        expected = {"neither": "", "both": "\n\n".join(part for part in (vector_prompt, graph_prompt) if part),
                    "graph_only": graph_prompt, "vector_only": vector_prompt}
        if case["prompts"] != expected:
            raise ValueError("Evidence prompts do not match the first generation method")


def validate_record(record, case, mode, manifest):
    response, request = record["response"], record["request"]
    context = case["prompts"][mode]
    expected_prompt = original.SYSTEM_PROMPT + ("\n\n" + context if context else "")
    if (record["index"], record["mode"], record["question"]) != (case["index"], mode, case["question"]):
        raise ValueError("Question/group mismatch")
    if request["messages"] != [{"role": "system", "content": expected_prompt},
                               {"role": "user", "content": case["question"]}]:
        raise ValueError("Unexpected question or prompt")
    if request["generation_config"] != original.GENERATION or request.get("tools") or request.get("stream"):
        raise ValueError("Not the first generation configuration")
    metadata = response.get("metadata", {})
    if metadata.get("provider") != "deepseek" or metadata.get("model") != manifest["model"]:
        raise ValueError("Unexpected model/provider")
    if metadata.get("fallback") or metadata.get("tool_calls") or response.get("finish_reason") != "stop":
        raise ValueError("Fallback or incomplete inference")
    if not response.get("text", "").strip():
        raise ValueError("Empty response")
    if metadata.get("runtime", {}).get("agent", {}).get("enabled") is not False:
        raise ValueError("Agent was not explicitly bypassed")
    if response["request_id"] != request["request_id"]:
        raise ValueError("Response identity mismatch")


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        raise ValueError("Use a new output directory")
    evidence = read_json(args.source / "evidence.json")
    question_set = read_json(args.source / "question_set.json")
    validate_inputs(evidence, question_set)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "answers").mkdir(exist_ok=True)
    for name in ("evidence.json", "question_set.json", "graph_recall_audit.json"):
        source, target = args.source / name, args.output / name
        if not source.exists():
            if name != "graph_recall_audit.json":
                raise ValueError(f"Missing {name}")
            continue
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError(f"Cannot change saved input: {name}")
        if not target.exists():
            shutil.copyfile(source, target)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["system_prompt"] != original.SYSTEM_PROMPT or manifest["generation_config"] != original.GENERATION:
            raise ValueError("Saved generation policy differs from the original")
        if manifest["evidence_sha256"] != hashlib.sha256((args.output / "evidence.json").read_bytes()).hexdigest():
            raise ValueError("Saved evidence fingerprint mismatch")
        for case in evidence["cases"]:
            for mode in original.MODES:
                path = args.output / "answers" / f"q{case['index']:02d}_{mode}.json"
                if path.exists():
                    validate_record(read_json(path), case, mode, manifest)
    if not args.render_only:
        await original.generate_answers(args.output, evidence, max(1, args.concurrency))
    manifest = read_json(manifest_path)
    identities, audits = set(), []
    for case in evidence["cases"]:
        for mode in original.MODES:
            record = read_json(args.output / "answers" / f"q{case['index']:02d}_{mode}.json")
            validate_record(record, case, mode, manifest)
            identities.add(record["request"]["session_id"])
            audits.append({"index": case["index"], "mode": mode,
                           **original.answer_format_audit(record["response"]["text"])})
    if len(identities) != 120:
        raise ValueError("Sessions were reused")
    original.build_report(args.output, evidence)
    document = Document(args.output / "光与健康_30题四组对比报告.docx")
    question_headings = sum(paragraph.style.name == "Heading 3" and paragraph.text.startswith("Q")
                            for paragraph in document.paragraphs)
    group_headings = sum(paragraph.style.name == "Heading 4" and paragraph.text in original.MODES.values()
                         for paragraph in document.paragraphs)
    if question_headings != 30 or group_headings != 120:
        raise ValueError("Incomplete DOCX report")
    original.save_json(args.output / "validation.json", {
        "valid": True, "questions": 30, "groups": 4, "answers": 120,
        "question_set_matches_latest_revision": True, "replaced_questions_retained": question_set["replaced_count"],
        "first_generation_prompt_and_config": True, "all_agent_disabled": True,
        "source_isolation_verified": True, "all_finish_reason_stop": True,
        "all_real_model": manifest["model"], "unique_sessions": len(identities),
        "evidence_sha256": manifest["evidence_sha256"],
        "docx_question_headings": question_headings, "docx_group_headings": group_headings,
        "format_audits": audits,
    })
    print("Validated: 30 questions, 4 groups, 120 real answers, original generation policy", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
