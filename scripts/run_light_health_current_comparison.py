"""Replay the original retrieval ablation through the current LLM application."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import sys
import time
import uuid
from datetime import datetime

import httpx
from docx import Document
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_light_health_comparison import MODES, QUESTIONS
from app_protocols import LLMRequest
from services.knowledge.schemas import KnowledgeContext
from services.llm import main as llm_main
from services.llm.agent_runtime import LLMAgentRuntime
from services.llm.memory import MemoryStore

VERSION = "agent02-12-age-persona-fusion"
GENERATION = {"agent_enabled": True, "answer_mode": "standard", "memory_mode": "off"}
TOOL_FLAGS = (
    "TIME_TOOL_ENABLED", "CALCULATOR_TOOL_ENABLED", "UNIT_TOOL_ENABLED",
    "DICTIONARY_TOOL_ENABLED", "WEATHER_TOOL_ENABLED", "BOCHA_WEB_SEARCH_ENABLED",
    "MEMORY_TOOLS_ENABLED",
)
ACTIVE_CASE: ContextVar[dict] = ContextVar("comparison_case")


def save_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def select_context(case: dict, mode: str) -> KnowledgeContext:
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    vector = KnowledgeContext.model_validate(case["vector_context"])
    graph = KnowledgeContext.model_validate(case["graph_context"])
    return KnowledgeContext(
        query=case["question"],
        chunks=vector.chunks if mode in {"both", "vector_only"} else [],
        resolved_entities=graph.resolved_entities if mode in {"both", "graph_only"} else [],
        relations=graph.relations if mode in {"both", "graph_only"} else [],
        relation_groups=graph.relation_groups if mode in {"both", "graph_only"} else {},
    )


class FrozenKnowledgeClient:
    async def retrieve(self, *, knowledge_base_id: str, query: str, **kwargs: object) -> KnowledgeContext:
        active = ACTIVE_CASE.get()
        if query != active["case"]["question"] or knowledge_base_id != "default":
            raise RuntimeError("Unexpected retrieval query or knowledge base")
        active["retrieval_calls"] += 1
        return select_context(active["case"], active["mode"])

    async def close(self) -> None:
        pass


class ComparisonRuntime(LLMAgentRuntime):
    def __init__(self) -> None:
        super().__init__(memory_store=MemoryStore(enabled=False), knowledge_client=FrozenKnowledgeClient())

    async def prepare(self, request: LLMRequest):
        turn = await super().prepare(request)
        if turn.context is None or turn.direct_answer is not None:
            raise RuntimeError("Current Agent must prepare a real model answer")
        if turn.request.tools:
            raise RuntimeError("External tools must be disabled in this ablation")
        if any((turn.context.memory.history, turn.context.memory.memories, turn.context.memory.profile,
                turn.context.concepts)):
            raise RuntimeError("Memory contamination")
        active = ACTIVE_CASE.get()
        context = turn.context.knowledge
        prompt = turn.request.messages[0].content
        turn.metadata["comparison_agent_audit"] = {
            "agent_enabled": turn.request.generation_config.get("agent_enabled") is True,
            "context_prepared": turn.context is not None,
            "strategy_prepared": bool(turn.context.tutor_strategy),
            "system_prompt_generated": (
                turn.request.messages[0].role == "system"
                and all(message.role != "system" for message in request.messages)
                and bool(prompt)
            ),
            "orchestrator": "LLMAgentRuntime.prepare",
            "strategy_agent": turn.context.skill_metadata.get("agent"),
            "knowledge_mode": active["mode"],
        }
        budget = int(os.getenv("KNOWLEDGE_MAX_CONTEXT_CHARS", "6000"))
        rendered = context.render_prompt(max_chars=budget)
        active["prepared_request"] = turn.request.model_dump(mode="json")
        active["runtime"] = turn.metadata
        active["knowledge_context"] = context.model_dump(mode="json")
        active["evidence_audit"] = {
            "retrieval_calls": active["retrieval_calls"],
            "skipped": bool(context.metadata.get("retrieval_skipped")),
            "context_budget": budget,
            "context_chars": len(rendered),
            "context_truncated": len(context.render_prompt(max_chars=10**7)) > budget,
            "visible_vector_sources": re.findall(r"\[资料(\d+)\]", rendered),
            "visible_graph_entities": [entity.name for entity in context.resolved_entities
                                       if f"- {entity.name} [" in rendered],
            "visible_graph_relations": [relation.model_dump(mode="json") for relation in context.relations
                                        if f"- {relation.source_name} --{relation.relation_type}--> {relation.target_name}" in rendered],
            "knowledge_prompt": rendered,
            "prompt_sha256": digest(prompt),
        }
        return turn

    async def after_turn(self, turn, assistant_text: str) -> dict:
        return {"enabled": False, "reason": "isolated_comparison_no_persistence"}


def request_for(case: dict, mode: str, run_id: str) -> LLMRequest:
    identity = f"lh-current-{run_id}-{case['index']:02d}-{mode}"
    return LLMRequest(
        request_id=identity, trace_id=identity, session_id=identity, user_id=identity,
        messages=[{"role": "user", "content": case["question"]}],
        generation_config=GENERATION, metadata={"knowledge_base_id": "default"}, stream=False,
    )


def validate_evidence(evidence: dict, question_set: dict | None = None) -> None:
    cases = evidence["cases"]
    if len(cases) != 30:
        raise ValueError("Expected exactly 30 original questions")
    expected = QUESTIONS
    if question_set is not None:
        items = question_set["questions"]
        if len(items) != 30 or [item["index"] for item in items] != list(range(1, 31)):
            raise ValueError("Question set must contain 30 ordered unique indices")
        if len({item["question"] for item in items}) != 30:
            raise ValueError("Question set contains duplicate questions")
        for item, (category, question) in zip(items, QUESTIONS):
            if item["original_question"] != question or item["category"] != category:
                raise ValueError("Question set original provenance changed")
            if item["replaced"] != (item["question"] != question):
                raise ValueError("Incorrect replacement marker")
        if sum(item["replaced"] for item in items) != question_set["replaced_count"]:
            raise ValueError("Incorrect replacement count")
        expected = [(item["category"], item["question"]) for item in items]
    for index, ((category, question), case) in enumerate(zip(expected, cases), 1):
        if (case["index"], case["category"], case["question"]) != (index, category, question):
            raise ValueError(f"Original question changed: {index}")


def validate_record(record: dict, manifest: dict, plan: dict) -> None:
    response = record["response"]
    metadata = response.get("metadata", {})
    if not response.get("text", "").strip() or response.get("finish_reason") != "stop":
        raise ValueError("Empty or incomplete answer")
    if metadata.get("provider") != "deepseek" or metadata.get("model") != manifest["model"]:
        raise ValueError("Unexpected model/provider")
    if metadata.get("fallback") or metadata.get("tool_calls"):
        raise ValueError("Fallback or unexecuted tools")
    runtime = metadata.get("runtime", {})
    if runtime.get("conversation_version") != VERSION or runtime.get("answer_mode") != "standard":
        raise ValueError("Not the current dialogue logic")
    agent_audit = runtime.get("comparison_agent_audit", {})
    if any(agent_audit.get(field) is not True for field in (
        "agent_enabled", "context_prepared", "strategy_prepared", "system_prompt_generated",
    )):
        raise ValueError("Agent execution was not verified")
    if agent_audit != plan["runtime"]["comparison_agent_audit"]:
        raise ValueError("Agent execution differs from the prepared plan")
    if record["request"]["generation_config"].get("agent_enabled") is not True:
        raise ValueError("Original request bypassed the Agent")
    if record["prepared_request"]["generation_config"].get("agent_enabled") is not True:
        raise ValueError("Prepared request bypassed the Agent")
    if runtime.get("agent", {}).get("enabled") is False:
        raise ValueError("Runtime reported a disabled Agent")
    if record["index"] != plan["index"] or record["mode"] != plan["mode"] or record["question"] != plan["question"]:
        raise ValueError("Mismatched question/group")
    if record["evidence_audit"] != plan["evidence_audit"]:
        raise ValueError("Evidence or prepared prompt changed")
    if record["prepared_request"] != plan["prepared_request"]:
        raise ValueError("Prepared request changed")
    context = record["knowledge_context"]
    if record["mode"] in {"neither", "graph_only"} and context["chunks"]:
        raise ValueError("Vector evidence leaked")
    if record["mode"] in {"neither", "vector_only"} and any(
        context[name] for name in ("relations", "resolved_entities", "relation_groups")
    ):
        raise ValueError("Graph evidence leaked")
    if record["prepared_request"]["tools"] or metadata.get("persistence", {}).get("enabled") is not False:
        raise ValueError("Tools or persistence enabled")
    if any(runtime.get("memory", {}).get(name) for name in (
        "history_count", "memory_count", "profile_fields", "concept_count",
    )):
        raise ValueError("Memory leaked")


async def prepare_plans(runtime: ComparisonRuntime, evidence: dict, run_id: str) -> dict:
    plans = {}
    for case in evidence["cases"]:
        for mode in MODES:
            active = {"case": case, "mode": mode, "retrieval_calls": 0}
            token = ACTIVE_CASE.set(active)
            try:
                request = request_for(case, mode, run_id)
                await runtime.prepare(request)
                plans[f"q{case['index']:02d}_{mode}"] = {
                    "index": case["index"], "mode": mode, "question": case["question"],
                    "request": request.model_dump(mode="json"),
                    **{name: active[name] for name in (
                        "prepared_request", "runtime", "knowledge_context", "evidence_audit",
                    )},
                }
            finally:
                ACTIVE_CASE.reset(token)
    return plans


def plan_fingerprint(plans: dict) -> str:
    return digest({key: {name: plan[name] for name in (
        "prepared_request", "knowledge_context", "evidence_audit",
    )} for key, plan in plans.items()})


async def generate(output: Path, evidence: dict, manifest: dict, plans: dict, concurrency: int) -> None:
    semaphore = asyncio.Semaphore(concurrency)
    completed = 0

    async def one(case: dict, mode: str) -> None:
        nonlocal completed
        name = f"q{case['index']:02d}_{mode}"
        path = output / "answers" / f"{name}.json"
        plan = plans[name]
        if path.exists():
            validate_record(read_json(path), manifest, plan)
            completed += 1
            return
        async with semaphore:
            errors = []
            for attempt in range(1, 4):
                active = {"case": case, "mode": mode, "retrieval_calls": 0}
                token = ACTIVE_CASE.set(active)
                started = time.perf_counter()
                try:
                    response = await llm_main.complete_llm(LLMRequest.model_validate(plan["request"]))
                    record = {
                        **plan, "attempt": attempt, "previous_errors": errors,
                        "generation_ms": round((time.perf_counter() - started) * 1000),
                        "response": response.model_dump(mode="json"),
                        **{key: active[key] for key in (
                            "prepared_request", "runtime", "knowledge_context", "evidence_audit",
                        )},
                    }
                    validate_record(record, manifest, plan)
                    save_json(path, record)
                    completed += 1
                    print(f"answer {completed}/120 {name}: {record['generation_ms']} ms", flush=True)
                    return
                except Exception as error:
                    detail = f"{type(error).__name__}: {error}"[:500]
                    secret = os.getenv("LLM_API_KEY", "")
                    if secret:
                        detail = detail.replace(secret, "[redacted]")
                    errors.append(detail)
                    print(f"retry {name} attempt={attempt}: {detail}", flush=True)
                    if attempt < 3:
                        await asyncio.sleep(attempt * 3)
                finally:
                    ACTIVE_CASE.reset(token)
            save_json(output / f"error_{name}.json", {"errors": errors})
            raise RuntimeError(f"Failed {name}")

    jobs = []
    for case in evidence["cases"]:
        order = list(MODES)
        offset = (case["index"] - 1) % len(order)
        jobs.extend(one(case, mode) for mode in order[offset:] + order[:offset])
    outcomes = await asyncio.gather(*jobs, return_exceptions=True)
    failures = [str(outcome) for outcome in outcomes if isinstance(outcome, BaseException)]
    if failures:
        raise RuntimeError("; ".join(failures))


def citation_flags(text: str) -> dict:
    links = re.findall(r"链接[：:]\s*([^\s\n]+)", text)
    return {
        "has_reference_section": "参考来源" in text,
        "non_url_links": [link for link in links if not re.match(r"https?://", link)],
        "inline_source_markers": bool(re.search(r"[\[【]资料\d+[\]】]", text.split("参考来源")[0])),
    }


def render_report(output: Path, evidence: dict, manifest: dict, plans: dict) -> None:
    records = {}
    for name, plan in plans.items():
        record = read_json(output / "answers" / f"{name}.json")
        validate_record(record, manifest, plan)
        record["citation_flags"] = citation_flags(record["response"]["text"])
        records[name] = record
    summary = {}
    for mode in MODES:
        group = [record for record in records.values() if record["mode"] == mode]
        summary[mode] = {
            "answers": len(group),
            "agent_verified_answers": sum(record["response"]["metadata"]["runtime"]["comparison_agent_audit"]["agent_enabled"] for record in group),
            "mean_generation_seconds": round(statistics.mean(record["generation_ms"] for record in group) / 1000, 2),
            "mean_answer_chars": round(statistics.mean(len(record["response"]["text"]) for record in group)),
            "visible_vector_questions": sum(bool(record["evidence_audit"]["visible_vector_sources"]) for record in group),
            "visible_graph_questions": sum(bool(record["evidence_audit"]["visible_graph_entities"] or
                                                record["evidence_audit"]["visible_graph_relations"]) for record in group),
            "visible_graph_relation_questions": sum(bool(record["evidence_audit"]["visible_graph_relations"]) for record in group),
            "truncated_context_questions": sum(record["evidence_audit"]["context_truncated"] for record in group),
            "retrieval_skipped_questions": sum(record["evidence_audit"]["skipped"] for record in group),
            "reference_section_answers": sum(record["citation_flags"]["has_reference_section"] for record in group),
            "non_url_link_answers": sum(bool(record["citation_flags"]["non_url_links"]) for record in group),
            "inline_source_marker_answers": sum(record["citation_flags"]["inline_source_markers"] for record in group),
            "input_tokens": sum(record["response"]["usage"]["input_tokens"] for record in group),
            "output_tokens": sum(record["response"]["usage"]["output_tokens"] for record in group),
            "retries": sum(record["attempt"] - 1 for record in group),
        }
    same_inputs = {}
    for first, second in (("neither", "graph_only"), ("both", "vector_only")):
        same_inputs[f"{first}/{second}"] = [case["index"] for case in evidence["cases"] if
            plans[f"q{case['index']:02d}_{first}"]["evidence_audit"]["prompt_sha256"] ==
            plans[f"q{case['index']:02d}_{second}"]["evidence_audit"]["prompt_sha256"]]
    save_json(output / "results.json", {"manifest": manifest, "summary": summary,
                                      "same_input_questions": same_inputs, "answers": records})
    save_json(output / "agent_execution_audit.json", {
        "entrypoint": "services.llm.main.complete_llm",
        "verified_by_group": {mode: row["agent_verified_answers"] for mode, row in summary.items()},
        "turns": [{"index": record["index"], "mode": record["mode"],
                   "request_id": record["request"]["request_id"],
                   "prompt_sha256": record["evidence_audit"]["prompt_sha256"],
                   **record["response"]["metadata"]["runtime"]["comparison_agent_audit"]}
                  for record in records.values()],
    })
    snapshot = evidence["snapshot"]
    budget = next(iter(plans.values()))["evidence_audit"]["context_budget"]
    revision = manifest.get("question_set")
    replaced_count = revision["replaced_count"] if revision else 0
    design = (f"保留原题{30-replaced_count}道，替换{replaced_count}道为图谱关系类问题；四组定义不变，共120份本次重新生成的真实模型回答。旧版报告未覆盖。"
              if revision else "沿用原报告全部30道原题、分类和四组定义，共120份本次真实模型回答。旧报告未覆盖。")
    evidence_description = "本轮沿用已采集的30题真实检索快照，不重新查询数据库；四组中的‘调用’指Agent允许使用对应分支证据，禁用分支不提供证据。全部120份回答重新运行Agent与模型生成，不复用旧答案。"
    comparability = ("本轮沿用上一版含10道图谱替换题的题组与检索快照，只恢复四组全部开启Agent。相较最早的原始题组仍有10题不同；这些题按图谱实际覆盖范围选取，不代表自然用户问题的随机样本。"
                     if revision else "题目、组别、模型及原始检索证据保持一致；变化包括启用当前Agent、教学策略、标准模式温度、输出长度和项目原有参考来源格式。因此新旧答案差异不能全部归因于检索。")
    lines = [
        "# 《光与健康》30题四组对比报告（四组均开启Agent）", "",
        f"生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}", "",
        "## 1. 对照设计与执行路径", "",
        design, evidence_description, "",
        "| 组别 | 项目Agent | 向量检索证据 | 图谱检索证据 |", "|---|---|---|---|",
        "| A 两者都不调用 | 开启 | 无 | 无 |", "| B 向量库与知识图谱都调用 | 开启 | 有 | 有，未匹配时为空 |",
        "| C 只调用知识图谱 | 开启 | 无 | 有，未匹配时为空 |", "| D 只调用向量库 | 开启 | 有 | 无 |", "",
        f"模型：{manifest['model']}；当前逻辑版本：{VERSION}；标准模式；temperature=0.32；Agent启用；不额外设置top_p和max_tokens。",
        "执行路径：当前 services.llm.main.complete_llm → 输入安全检查 → LLMAgentRuntime.prepare → 当前意图路由、TutorGraph、教学策略、系统提示词和知识库引用规则 → 真实DeepSeek推理 → 当前输出安全检查与收尾逻辑。",
        "不是旧脚本的agent_enabled=false，也没有沿用旧的200至350字、最多450字及‘依据：’结尾要求；回答长度、组织方式与参考来源由当前逻辑控制。",
        "为隔离变量，每题每组独立身份与会话，关闭记忆读写、联网搜索及其他外部工具；KnowledgeClient在独立评测进程中替换为按组返回固定证据的适配器。没有修改或重启线上服务。",
        "本报告是当前对话生成逻辑下的受控消融，不是网关、语音、持久记忆和实时检索全流程性能测试。意图跳过检索保留原逻辑，实际跳过与可见证据另有统计。", "",
        "### 与原报告的可比性", "",
        comparability,
        "向量快照来自项目BGE嵌入及Qdrant现有向量与关键词混合排序，top_k=8。图谱快照只按问题中实体名称/别名访问Neo4j，最多5个入口、2跳扩展，不通过向量结果定位实体。",
        f"本次使用项目当前KnowledgeContext.render_prompt，统一上下文预算{budget}字符，而非旧脚本向量7000与图谱5000分开拼接。图谱关系排在片段之后，可能被截断；下表区分图谱实体可见和关系可见，不能把数据库返回条数等同于模型实际看到的条数。", "",
        "### 沿用的数据快照（不是本次重新盘点）", "",
        f"证据采集时间：{evidence['created_at']}；集合：{snapshot['collection']}；总点数：{snapshot['collection_points']}；本书点数：{snapshot['book_points']}；嵌入模型：{snapshot['embedding_model']}，{snapshot['dimension']}维。",
        f"该书图谱Chunk节点：{snapshot['book_graph']['chunks']}；实体提及连接：{snapshot['book_graph']['mentions']}。本书尚未形成完整语义图谱；图谱组使用原有学科关系。未为本实验补造实体关系。", "",
        "## 2. 本次运行统计", "",
        "| 组别 | 回答数 | 文本可见题数 | 图谱可见题数 | 图谱关系可见题数 | 上下文截断题数 | 平均耗时/秒 | 平均字符数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, label in MODES.items():
        row = summary[mode]
        lines.append(f"| {label} | {row['answers']} | {row['visible_vector_questions']} | {row['visible_graph_questions']} | {row['visible_graph_relation_questions']} | {row['truncated_context_questions']} | {row['mean_generation_seconds']} | {row['mean_answer_chars']} |")
    lines += ["", "### Agent执行核验", "",
              "核验并非只检查agent_enabled参数：每份回答均检查Agent实际生成上下文、教学策略与系统提示词，并核对响应中的运行记录。缺少任意一项均视为失败，不计入报告。", "",
              "| 组别 | Agent执行核验通过 |", "|---|---:|"]
    for mode, label in MODES.items():
        lines.append(f"| {label} | {summary[mode]['agent_verified_answers']}/30 |")
    lines += ["", "耗时包括当前Agent准备、安全处理及真实推理，不含并发排队和原始检索；不是严格性能基准。每组每题保留一次完整成功回答，失败重试记录保留。", "",
              "### 相同输入与引用格式检查", ""]
    for pair, indices in same_inputs.items():
        lines.append(f"{pair}系统提示词完全相同的题号：{', '.join(map(str, indices)) or '无'}。这些题即使输出不同，也不能认定为增加检索的收益。")
    lines += ["", "| 组别 | 有参考来源段 | 链接字段不是HTTP URL | 正文资料编号 | 意图跳过检索 |",
              "|---|---:|---:|---:|---:|"]
    for mode, label in MODES.items():
        row = summary[mode]
        lines.append(f"| {label} | {row['reference_section_answers']} | {row['non_url_link_answers']} | {row['inline_source_marker_answers']} | {row['retrieval_skipped_questions']} |")
    lines += ["", "以上是可复核的格式统计，不是事实正确率。‘链接：无’、书名代替URL等会原样保留并计数，未替模型修补引用。链接存在也不意味着真实可访问或支持结论。",
              "没有人工标准答案、专家盲评或重复采样，不能宣称某组一定最好。重点复核Q26出版信息、Q27病例范围、Q29临床方案证据边界，以及图谱关系是否真的相关。旧报告的典型差异结论未照搬。", ""]
    if revision:
        lines += ["### 替换题目与目的", ""]
        for item in revision["questions"]:
            if item["replaced"]:
                lines += [f"Q{item['index']:02d} 原题：{item['original_question']}",
                          f"新题：{item['question']}", f"图谱考察点：{item['graph_task']}。", ""]
        lines += ["### 图谱目标关系核验", "",
                  "下表沿用证据采集时的只读图谱核验，比较数据库中的目标直接上位关系与快照中的实际召回结果；核验结果只用于报告分析，没有额外注入模型。关系存在但未召回，属于检索覆盖问题；召回也不等于模型正确使用。", "",
                  "| 题号 | 图谱中存在的上位关系 | 实际召回的目标关系数 | 返回关系总数 |",
                  "|---|---|---:|---:|"]
        for row in manifest["graph_recall_audit"]:
            parents = "；".join(f"{pair['parent']} → {pair['target']}" for pair in row["target_parents_in_graph"])
            lines.append(f"| Q{row['index']:02d} | {parents} | {len(row['target_parents_recalled'])} | {row['relation_count']} |")
        lines += ["", "这些层级是本项目图谱的分类记录，不应视为唯一公认的学科分类，更不能从CONTAINS关系推出光照具有某种临床疗效。", ""]
    lines += ["## 3. 30题逐题原始回答", ""]
    for case in evidence["cases"]:
        lines += [f"### Q{case['index']:02d} [{case['category']}] {case['question']}", "",
                  f"原始证据：向量片段{len(case['vector_context']['chunks'])}个，其中本书{case['book_hits']}个；图谱实体{len(case['graph_context']['resolved_entities'])}个、关系{len(case['graph_context']['relations'])}条。", ""]
        for mode, label in MODES.items():
            record = records[f"q{case['index']:02d}_{mode}"]
            audit = record["evidence_audit"]
            lines += [f"#### {label}", "", record["response"]["text"], "",
                      f"运行记录：Agent开启且执行核验通过；{record['generation_ms']/1000:.2f}秒；可见资料{len(audit['visible_vector_sources'])}条、图谱实体{len(audit['visible_graph_entities'])}个、关系{len(audit['visible_graph_relations'])}条；finish_reason=stop。", ""]
    lines += ["## 4. 复核文件", "", "evidence.json：与来源报告逐字节一致的30题证据快照。",
              "agent_execution_audit.json：120份Agent实际执行核验及分组计数。",
              "manifest.json：模型、版本、配置、代码哈希与快照哈希。",
              "prepared.json：120份当前Agent实际构建的请求、意图策略和可见证据。",
              "answers/：120份独立原始请求、准备后请求、真实响应与重试记录。",
              "results.json、validation.json：汇总数据、格式检查与完整性验证。", ""]
    if revision:
        lines += ["question_set.json：完整30题、替换标识、原题与图谱考察点。",
                  "graph_recall_audit.json：目标关系存在性与实际检索召回核验。", ""]
    title = "光与健康_30题四组对比报告_Agent开启"
    (output / f"{title}.md").write_text("\n".join(lines), encoding="utf-8")
    document = Document()
    section = document.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.top_margin = section.bottom_margin = Cm(2)
    section.left_margin = section.right_margin = Cm(2)
    for style_name in ("Normal", "Title", "Heading 1", "Heading 2", "Heading 3", "Heading 4"):
        style = document.styles[style_name]
        style.font.name = "Microsoft YaHei"
        style.element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    document.styles["Normal"].font.size = Pt(10)
    table = None
    for line in lines:
        if not line.strip():
            table = None
        elif line.startswith("|---"):
            continue
        elif line.startswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if table is None or len(table.columns) != len(cells):
                table = document.add_table(rows=0, cols=len(cells))
                table.style = "Table Grid"
            for cell, value in zip(table.add_row().cells, cells):
                cell.text = value
        elif line.startswith("#"):
            table = None
            level = len(line) - len(line.lstrip("#"))
            heading = document.add_heading(line[level:].strip(), level=min(level, 4))
            if line.startswith("### Q"):
                heading.paragraph_format.page_break_before = True
        else:
            table = None
            document.add_paragraph(line)
    document.save(output / f"{title}.docx")
    reopened = Document(output / f"{title}.docx")
    question_headings = sum(paragraph.text.startswith("Q") and paragraph.style.name == "Heading 3"
                            for paragraph in reopened.paragraphs)
    group_headings = sum(paragraph.text in MODES.values() and paragraph.style.name == "Heading 4"
                         for paragraph in reopened.paragraphs)
    if question_headings != 30 or group_headings != 120:
        raise RuntimeError("DOCX missing question or group sections")
    save_json(output / "validation.json", {
        "valid": True, "answer_count": len(records), "question_count": 30, "group_count": 4,
        "original_questions_unchanged": not bool(revision), "replaced_question_count": replaced_count,
        "selected_questions_matched": True, "evidence_sha256": manifest["evidence_sha256"],
        "current_agent_all_answers": True, "finish_reason_all_stop": True,
        "agent_verified_by_group": {mode: row["agent_verified_answers"] for mode, row in summary.items()},
        "source_isolation_passed": True, "no_memory_or_external_tools": True,
        "docx_question_headings": question_headings, "docx_group_headings": group_headings,
        "summary": summary, "same_input_questions": same_inputs,
    })
    print(json.dumps({"completed": len(records), "summary": summary, "output": str(output)}, ensure_ascii=True), flush=True)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--question-set", type=Path)
    args = parser.parse_args()
    if args.output.resolve() == args.source.resolve():
        raise ValueError("Preserve the original report: use a new output directory")
    for flag in TOOL_FLAGS:
        os.environ[flag] = "false"
    os.environ["LLM_FALLBACK_ENABLED"] = "false"
    os.environ["LLM_TIMEOUT_SECONDS"] = "180"
    os.environ["LLM_TOOL_SELECTION_MODE"] = "native"
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "answers").mkdir(exist_ok=True)
    source = args.source / "evidence.json"
    target = args.output / "evidence.json"
    if target.exists() and target.read_bytes() != source.read_bytes():
        raise ValueError("Evidence snapshot changed")
    if not target.exists():
        shutil.copyfile(source, target)
    evidence = read_json(target)
    question_set = read_json(args.question_set) if args.question_set else None
    validate_evidence(evidence, question_set)
    graph_audit = None
    if question_set:
        if evidence.get("question_set_sha256") != hashlib.sha256(args.question_set.read_bytes()).hexdigest():
            raise ValueError("Evidence/question-set fingerprint mismatch")
        graph_audit = read_json(args.source / "graph_recall_audit.json")
        for name, value in (("question_set.json", question_set), ("graph_recall_audit.json", graph_audit)):
            destination = args.output / name
            if destination.exists() and read_json(destination) != value:
                raise ValueError(f"Cannot replace existing revision metadata: {name}")
            save_json(destination, value)
    manifest_path = args.output / "manifest.json"
    previous = read_json(manifest_path) if manifest_path.exists() else None
    settings = llm_main.load_settings()
    if settings.provider != "deepseek":
        raise RuntimeError("Real configured DeepSeek is required")
    key = os.getenv("LLM_INTERNAL_API_KEY", "")
    async with httpx.AsyncClient(timeout=15, trust_env=False) as client:
        response = await client.get("http://127.0.0.1:8002/health", headers={"X-Internal-API-Key": key} if key else {})
        response.raise_for_status()
        health = response.json()
    if health.get("model") != settings.model or VERSION not in json.dumps(health):
        raise RuntimeError("Local evaluation differs from the running LLM version/model")
    original_runtime = llm_main.agent_runtime
    runtime = ComparisonRuntime()
    llm_main.agent_runtime = runtime
    llm_main.safety_guard = runtime.safety_guard
    try:
        run_id = previous["run_id"] if previous else uuid.uuid4().hex[:10]
        plans = await prepare_plans(runtime, evidence, run_id)
        source_files = sorted((ROOT / "services" / "llm").rglob("*.py"))
        source_files += [ROOT / "app_protocols" / "persona.py", ROOT / "services" / "knowledge" / "schemas.py", Path(__file__)]
        manifest = {
            "created_at": previous["created_at"] if previous else datetime.now().astimezone().isoformat(),
            "run_id": run_id, "provider": settings.provider, "model": settings.model,
            "conversation_version": VERSION, "generation_config": GENERATION,
            "effective_generation_config": next(iter(plans.values()))["prepared_request"]["generation_config"],
            "concurrency": max(1, args.concurrency), "question_count": 30, "mode_count": 4,
            "evidence_source": str(source.resolve()), "evidence_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "prepared_sha256": plan_fingerprint(plans),
            "code_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
            "retrieval": "frozen_evidence_at_knowledge_client_boundary",
            "evidence_collected_at": evidence["created_at"],
            "question_set": question_set, "graph_recall_audit": graph_audit,
            "memory_read_write": False, "external_tools": False, "entrypoint": "services.llm.main.complete_llm",
        }
        if previous:
            for field in ("model", "conversation_version", "generation_config", "evidence_sha256", "prepared_sha256", "code_sha256", "question_set", "graph_recall_audit"):
                if previous.get(field) != manifest[field]:
                    raise ValueError(f"Cannot resume changed experiment: {field}")
        else:
            save_json(manifest_path, manifest)
        save_json(args.output / "prepared.json", plans)
        print(json.dumps({"prepared": len(plans), "model": settings.model,
                          "version": VERSION, "context_budget": next(iter(plans.values()))["evidence_audit"]["context_budget"],
                          "retrieval_skips": sum(plan["evidence_audit"]["skipped"] for plan in plans.values())}), flush=True)
        if args.prepare_only:
            return
        if not args.render_only:
            await generate(args.output, evidence, manifest, plans, max(1, args.concurrency))
        render_report(args.output, evidence, manifest, plans)
    finally:
        await runtime.close()
        await original_runtime.close()
        for client in llm_main._deepseek_clients.values():
            await client.aclose()
        llm_main._deepseek_clients.clear()


if __name__ == "__main__":
    asyncio.run(main())
