"""Import the Human Knowledge and Education Atlas workbook into GraphRAG."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import posixpath
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import httpx


MAIN_TABLES = (
    "KG_All",
    "EduCourse300",
    "EduMatrix",
    "EduEvolution",
    "SchoolCases",
    "SchoolEdges",
    "Reference64",
    "AnalysisFindings",
)
DOMAIN_COLUMNS = (
    "语言、读写与多语沟通",
    "数学、逻辑与统计",
    "自然科学",
    "计算、信息与媒体",
    "工程、技术与设计制作",
    "社会、公民、经济与法律",
    "历史、地理、哲学与伦理",
    "艺术与文化表达",
    "健康、体育与福祉",
    "环境、农业与可持续发展",
    "生活、家庭、职业与服务",
    "社会情感、能动性与学会学习",
)
XML_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "office_rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


@dataclass(frozen=True)
class TableRow:
    row_number: int
    values: dict[str, str]


@dataclass(frozen=True)
class WorkbookTable:
    name: str
    sheet_name: str
    rows: list[TableRow]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx", type=Path, required=True)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8003")
    parser.add_argument("--knowledge-base-id", default="default")
    parser.add_argument("--admin-api-key", default=os.getenv("KNOWLEDGE_ADMIN_API_KEY", ""))
    parser.add_argument("--limit-per-table", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    xlsx_path = args.xlsx.resolve()
    html_path = args.html.resolve()
    if not xlsx_path.is_file():
        raise SystemExit(f"Workbook was not found: {xlsx_path}")
    if not html_path.is_file():
        raise SystemExit(f"Atlas HTML was not found: {html_path}")

    tables = _read_workbook_tables(xlsx_path)
    missing = [name for name in MAIN_TABLES if name not in tables]
    if missing:
        raise SystemExit(f"Workbook is missing required tables: {', '.join(missing)}")
    html_summary = _read_html_summary(html_path)
    _validate_mirrored_counts(tables, html_summary)
    documents = _build_documents(
        tables=tables,
        xlsx_path=xlsx_path,
        html_path=html_path,
        html_summary=html_summary,
        knowledge_base_id=args.knowledge_base_id,
        limit_per_table=max(0, args.limit_per_table),
    )
    summary = {
        "knowledge_base_id": args.knowledge_base_id,
        "xlsx": xlsx_path.name,
        "xlsx_sha256": _sha256(xlsx_path),
        "html": html_path.name,
        "html_sha256": _sha256(html_path),
        "documents": len(documents),
        "chunks": sum(len(document["chunks"]) for document in documents),
        "document_chunks": {document["title"]: len(document["chunks"]) for document in documents},
        "citation_audit": _citation_audit(documents),
    }
    if args.dry_run:
        print(json.dumps({"status": "dry-run", **summary}, ensure_ascii=False, indent=2))
        return

    results = asyncio.run(
        _import_documents(
            api_url=args.api_url,
            admin_api_key=args.admin_api_key,
            documents=documents,
        )
    )
    report = {
        "status": "completed",
        "imported_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **summary,
        "results": results,
    }
    report_path = args.report or (
        Path(__file__).resolve().parents[1]
        / "logs"
        / f"knowledge-atlas-import-{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {report_path}")


async def _import_documents(
    *,
    api_url: str,
    admin_api_key: str,
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    headers = {"X-Admin-API-Key": admin_api_key} if admin_api_key else {}
    endpoint = api_url.rstrip("/") + "/api/v1/knowledge/documents/structured"
    results: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=900.0, trust_env=False) as client:
        for index, document in enumerate(documents, start=1):
            print(f"[{index}/{len(documents)}] Importing {document['title']} ({len(document['chunks'])} chunks)...")
            response = await client.post(endpoint, json=document, headers=headers)
            if response.is_error:
                raise RuntimeError(
                    f"Import failed for {document['title']}: HTTP {response.status_code} {response.text[:1000]}"
                )
            body = response.json()
            results.append(body)
            print(
                f"  ready document={body.get('document_id')} chunks={body.get('chunk_count')} "
                f"entities={body.get('entity_count')} relations={body.get('relation_count')} "
                f"duplicate={body.get('duplicate', False)}"
            )
    return results


def _build_documents(
    *,
    tables: dict[str, WorkbookTable],
    xlsx_path: Path,
    html_path: Path,
    html_summary: dict[str, Any],
    knowledge_base_id: str,
    limit_per_table: int,
) -> list[dict[str, Any]]:
    workbook_hash = _sha256(xlsx_path)
    html_hash = _sha256(html_path)
    kg_rows = tables["KG_All"].rows
    node_names = {
        row.values.get("Node_ID", "").strip(): row.values.get("当前节点", "").strip()
        for row in kg_rows
        if row.values.get("Node_ID", "").strip()
    }
    builders = {
        "KG_All": lambda row: _kg_chunk(row, tables["KG_All"].sheet_name, xlsx_path.name, node_names),
        "EduCourse300": lambda row: _education_course_chunk(
            row, tables["EduCourse300"].sheet_name, xlsx_path.name
        ),
        "EduMatrix": lambda row: _education_matrix_chunk(
            row, tables["EduMatrix"].sheet_name, xlsx_path.name
        ),
        "EduEvolution": lambda row: _education_evolution_chunk(
            row, tables["EduEvolution"].sheet_name, xlsx_path.name
        ),
        "SchoolCases": lambda row: _school_case_chunk(
            row, tables["SchoolCases"].sheet_name, xlsx_path.name
        ),
        "SchoolEdges": lambda row: _school_edge_chunk(
            row, tables["SchoolEdges"].sheet_name, xlsx_path.name
        ),
        "Reference64": lambda row: _reference_chunk(
            row, tables["Reference64"].sheet_name, xlsx_path.name
        ),
        "AnalysisFindings": lambda row: _analysis_chunk(
            row, tables["AnalysisFindings"].sheet_name, xlsx_path.name
        ),
    }
    titles = {
        "KG_All": "人类知识图谱节点总表",
        "EduCourse300": "五国教育课程映射",
        "EduMatrix": "五国教育年龄矩阵",
        "EduEvolution": "教育领域年龄演化",
        "SchoolCases": "学校教育案例",
        "SchoolEdges": "学校课程关系",
        "Reference64": "知识与教育图谱来源总表",
        "AnalysisFindings": "知识与教育图谱分析发现",
    }
    documents: list[dict[str, Any]] = []
    for table_name in MAIN_TABLES:
        table = tables[table_name]
        rows = table.rows[:limit_per_table] if limit_per_table else table.rows
        chunks = [builders[table_name](row) for row in rows]
        chunks = [chunk for chunk in chunks if chunk is not None]
        documents.append(
            {
                "knowledge_base_id": knowledge_base_id,
                "title": titles[table_name],
                "source_name": titles[table_name],
                "metadata": {
                    "source_file": xlsx_path.name,
                    "source_sha256": workbook_hash,
                    "source_table": table_name,
                    "source_sheet": table.sheet_name,
                    "derived_visualization_file": html_path.name,
                    "derived_visualization_sha256": html_hash,
                },
                "chunks": chunks,
            }
        )
    documents.append(
        {
            "knowledge_base_id": knowledge_base_id,
            "title": "Human Knowledge Atlas 与 Education Graph 总览",
            "source_name": "Human Knowledge Atlas × Education Graph（项目可视化总览）",
            "metadata": {
                "source_file": html_path.name,
                "source_sha256": html_hash,
                "source_role": "derived_visualization_overview",
                "authoritative_workbook": xlsx_path.name,
                "authoritative_workbook_sha256": workbook_hash,
            },
            "chunks": [_html_overview_chunk(html_path, html_summary)],
        }
    )
    return documents


def _kg_chunk(
    row: TableRow,
    sheet_name: str,
    workbook_name: str,
    node_names: dict[str, str],
) -> dict[str, Any] | None:
    values = row.values
    current = values.get("当前节点", "").strip()
    if not current:
        return None
    node_id = values.get("Node_ID", "").strip() or f"row-{row.row_number}"
    parent_id = values.get("Parent_ID", "").strip()
    parent = node_names.get(parent_id, "").strip()
    description = values.get("解释/原始说明", "").strip()
    entities = [_entity(current, "KnowledgeNode", description)]
    relations: list[dict[str, Any]] = []
    if parent:
        entities.append(_entity(parent, "KnowledgeNode"))
        relations.append(_relation(parent, current, "CONTAINS", "KnowledgeNode", "KnowledgeNode", values))
    for field in ("OECD FORD主映射", "OECD FORD次映射"):
        classification = values.get(field, "").strip()
        if classification:
            entities.append(_entity(classification, "Classification"))
            relations.append(
                _relation(current, classification, "MAPPED_TO", "KnowledgeNode", "Classification", values)
            )
    return _chunk(
        chunk_key=node_id,
        record_type="知识图谱节点",
        row=row,
        sheet_name=sheet_name,
        workbook_name=workbook_name,
        entities=entities,
        relations=relations,
    )


def _education_course_chunk(
    row: TableRow,
    sheet_name: str,
    workbook_name: str,
) -> dict[str, Any] | None:
    values = row.values
    mapping_id = values.get("映射ID", "").strip()
    country = values.get("国家", "").strip()
    age = values.get("统一年龄", "").strip()
    framework = values.get("课程框架", "").strip()
    domain = values.get("教育知识领域", "").strip()
    if not mapping_id or not domain:
        return None
    entities = _entities(
        (country, "Country"),
        (age, "AgeGroup"),
        (framework, "CurriculumFramework"),
        (domain, "EducationDomain"),
    )
    relations: list[dict[str, Any]] = []
    if country and framework:
        relations.append(_relation(country, framework, "USES_CURRICULUM", "Country", "CurriculumFramework", values))
    if framework and domain:
        relations.append(_relation(framework, domain, "COVERS", "CurriculumFramework", "EducationDomain", values))
    if framework and age:
        relations.append(_relation(framework, age, "TARGETS_AGE_GROUP", "CurriculumFramework", "AgeGroup", values))
    reference_id = values.get("Reference_ID", "").strip()
    if reference_id and framework:
        entities.append(_entity(reference_id, "ReferenceSource"))
        relations.append(
            _relation(framework, reference_id, "SUPPORTED_BY", "CurriculumFramework", "ReferenceSource", values)
        )
    return _chunk(
        chunk_key=mapping_id,
        record_type="教育课程映射",
        row=row,
        sheet_name=sheet_name,
        workbook_name=workbook_name,
        entities=entities,
        relations=relations,
    )


def _education_matrix_chunk(row: TableRow, sheet_name: str, workbook_name: str) -> dict[str, Any] | None:
    values = row.values
    country = values.get("国家", "").strip()
    age = values.get("年龄段", "").strip()
    if not country or not age:
        return None
    context = _name(f"{country} {age}")
    entities = _entities((context, "EducationContext"), (country, "Country"), (age, "AgeGroup"))
    relations = [
        _relation(context, country, "APPLIES_TO_COUNTRY", "EducationContext", "Country", values),
        _relation(context, age, "APPLIES_TO_AGE_GROUP", "EducationContext", "AgeGroup", values),
    ]
    for domain in DOMAIN_COLUMNS:
        status = values.get(domain, "").strip()
        if not status:
            continue
        entities.append(_entity(domain, "EducationDomain"))
        relations.append(
            _relation(context, domain, _status_relation(status), "EducationContext", "EducationDomain", values)
        )
    return _chunk(
        chunk_key=f"{country}-{age}",
        record_type="五国教育年龄矩阵",
        row=row,
        sheet_name=sheet_name,
        workbook_name=workbook_name,
        entities=entities,
        relations=relations,
    )


def _education_evolution_chunk(
    row: TableRow,
    sheet_name: str,
    workbook_name: str,
) -> dict[str, Any] | None:
    values = row.values
    domain = values.get("教育知识领域", "").strip()
    age = values.get("年龄段", "").strip()
    if not domain or not age:
        return None
    context = _name(f"{domain} {age}")
    entities = _entities((context, "EducationProgression"), (domain, "EducationDomain"), (age, "AgeGroup"))
    relations = [
        _relation(context, domain, "TRACKS_DOMAIN", "EducationProgression", "EducationDomain", values),
        _relation(context, age, "TRACKS_AGE_GROUP", "EducationProgression", "AgeGroup", values),
    ]
    for country in ("中国", "美国", "英国", "德国", "日本"):
        status = values.get(country, "").strip()
        if not status:
            continue
        entities.append(_entity(country, "Country"))
        relations.append(
            _relation(country, context, _status_relation(status), "Country", "EducationProgression", values)
        )
    return _chunk(
        chunk_key=f"{domain}-{age}",
        record_type="教育领域年龄演化",
        row=row,
        sheet_name=sheet_name,
        workbook_name=workbook_name,
        entities=entities,
        relations=relations,
    )


def _school_case_chunk(row: TableRow, sheet_name: str, workbook_name: str) -> dict[str, Any] | None:
    values = row.values
    school_id = values.get("学校ID", "").strip()
    school = values.get("学校中文名", "").strip() or values.get("学校英文/原文名", "").strip()
    if not school_id or not school:
        return None
    country = values.get("国家", "").strip()
    curriculum = values.get("课程/资格路径", "").strip()
    entities = _entities((school, "School"), (country, "Country"), (curriculum, "CurriculumFramework"))
    relations: list[dict[str, Any]] = []
    if country:
        relations.append(_relation(school, country, "LOCATED_IN", "School", "Country", values))
    if curriculum:
        relations.append(_relation(school, curriculum, "OFFERS", "School", "CurriculumFramework", values))
    return _chunk(
        chunk_key=school_id,
        record_type="学校教育案例",
        row=row,
        sheet_name=sheet_name,
        workbook_name=workbook_name,
        entities=entities,
        relations=relations,
    )


def _school_edge_chunk(row: TableRow, sheet_name: str, workbook_name: str) -> dict[str, Any] | None:
    values = row.values
    school_id = values.get("学校ID", "").strip()
    school = values.get("学校", "").strip()
    category = values.get("课程类别", "").strip()
    if not school_id or not school or not category:
        return None
    relation_value = values.get("关系", "").strip()
    entities = _entities((school, "School"), (category, "EducationDomain"))
    relations = [
        _relation(
            school,
            category,
            _school_relation(relation_value),
            "School",
            "EducationDomain",
            values,
        )
    ]
    return _chunk(
        chunk_key=f"{school_id}-{category}",
        record_type="学校课程关系",
        row=row,
        sheet_name=sheet_name,
        workbook_name=workbook_name,
        entities=entities,
        relations=relations,
    )


def _reference_chunk(row: TableRow, sheet_name: str, workbook_name: str) -> dict[str, Any] | None:
    values = row.values
    source_id = values.get("Source_ID", "").strip()
    title = values.get("文件/网页标题", "").strip()
    organization = values.get("发布机构", "").strip()
    if not source_id or not title:
        return None
    entities = _entities((source_id, "ReferenceSource"), (title, "Publication"), (organization, "Organization"))
    relations = [_relation(source_id, title, "IDENTIFIES", "ReferenceSource", "Publication", values)]
    if organization:
        relations.append(_relation(title, organization, "PUBLISHED_BY", "Publication", "Organization", values))
    return _chunk(
        chunk_key=source_id,
        record_type="Reference来源",
        row=row,
        sheet_name=sheet_name,
        workbook_name=workbook_name,
        entities=entities,
        relations=relations,
    )


def _analysis_chunk(row: TableRow, sheet_name: str, workbook_name: str) -> dict[str, Any] | None:
    values = row.values
    finding_id = values.get("Finding_ID", "").strip()
    finding = values.get("发现", "").strip() or values.get("解释", "").strip()
    layer = values.get("分析层", "").strip()
    if not finding_id or not finding:
        return None
    entities = _entities((finding, "Finding"), (layer, "AnalysisLayer"))
    relations = []
    if layer:
        relations.append(_relation(finding, layer, "ANALYZES", "Finding", "AnalysisLayer", values))
    return _chunk(
        chunk_key=finding_id,
        record_type="分析发现",
        row=row,
        sheet_name=sheet_name,
        workbook_name=workbook_name,
        entities=entities,
        relations=relations,
    )


def _html_overview_chunk(_html_path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    knowledge = summary["knowledge"]
    education = summary["education"]
    text = "\n".join(
        (
            "记录类型：知识与教育图谱可视化总览",
            f"网页标题：{summary['title']}",
            f"知识图谱节点数：{knowledge['nodes']}",
            f"教育课程映射数：{education['entries']}",
            f"国家数：{education['countries']}",
            f"年龄段数：{education['ages']}",
            f"教育领域数：{education['domains']}",
            f"年龄进阶关系数：{education['progression']}",
            f"学校案例数：{education['schools']}",
            f"学校课程关系数：{education['school_edges']}",
            "数据说明：该总览由权威工作簿数据派生，用于展示全局统计，不作为第二份事实数据重复导入。",
        )
    )
    return {
        "chunk_key": "atlas-overview",
        "text": text,
        "source_name": "Human Knowledge Atlas × Education Graph（项目派生可视化总览）",
        "source_url": None,
        "source_locator": "Human Knowledge Atlas → Knowledge Graph 与 Education Graph → 全局统计总览",
        "metadata": {"source_role": "derived_visualization_overview"},
        "entities": _entities(
            ("Human Knowledge Atlas", "KnowledgeAtlas"),
            ("Knowledge Graph", "GraphView"),
            ("Education Graph", "GraphView"),
        ),
        "relations": [
            _relation_simple("Human Knowledge Atlas", "Knowledge Graph", "CONTAINS", "KnowledgeAtlas", "GraphView"),
            _relation_simple("Human Knowledge Atlas", "Education Graph", "CONTAINS", "KnowledgeAtlas", "GraphView"),
        ],
    }


def _chunk(
    *,
    chunk_key: str,
    record_type: str,
    row: TableRow,
    sheet_name: str,
    workbook_name: str,
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
) -> dict[str, Any]:
    values = {name: value.strip() for name, value in row.values.items() if value.strip()}
    lines = [f"记录类型：{record_type}"]
    lines.extend(f"{name}：{value}" for name, value in values.items())
    source_name, source_url, source_locator = _citation_details(
        record_type=record_type,
        values=values,
        chunk_key=chunk_key,
    )
    return {
        "chunk_key": _name(chunk_key, 500),
        "text": "\n".join(lines)[:20_000],
        "source_name": source_name,
        "source_url": source_url,
        "source_locator": source_locator,
        "metadata": {
            "sheet": sheet_name,
            "row": row.row_number,
            "record_type": record_type,
            "record_id": chunk_key,
        },
        "entities": _dedupe_entities(entities),
        "relations": _dedupe_relations(relations),
    }


def _citation_details(
    *,
    record_type: str,
    values: dict[str, str],
    chunk_key: str,
) -> tuple[str, str | None, str]:
    source_url = _first_value(values, "Reference_URL", "原始URL", "证据URL") or None
    if record_type == "知识图谱节点":
        source_name = values.get("分类来源") or "Human Knowledge Atlas 项目知识本体"
        source_locator = _source_path(values.get("完整路径"), values.get("具体定位"))
    elif record_type == "教育课程映射":
        source_name = _source_label(
            values.get("Reference机构"),
            values.get("Reference标题"),
            fallback=values.get("课程框架") or "五国教育课程公开资料",
        )
        source_locator = _source_path(
            values.get("国家"),
            values.get("统一年龄"),
            values.get("课程框架"),
            values.get("教育知识领域"),
            values.get("页码/章节或定位"),
        )
    elif record_type == "五国教育年龄矩阵":
        source_name = "Human Knowledge Atlas 五国教育课程映射（项目派生矩阵）"
        source_locator = _source_path(values.get("国家"), values.get("年龄段"), "五国教育年龄矩阵")
    elif record_type == "教育领域年龄演化":
        source_name = "Human Knowledge Atlas 五国教育课程映射（项目派生分析）"
        source_locator = _source_path(values.get("教育知识领域"), values.get("年龄段"), "五国课程演化")
    elif record_type == "学校教育案例":
        school = values.get("学校中文名") or values.get("学校英文/原文名")
        source_name = values.get("Reference标题") or f"{school}课程与培养方案公开资料"
        source_locator = _source_path(
            values.get("国家"),
            school,
            values.get("课程/资格路径"),
            values.get("核心课程"),
        )
    elif record_type == "学校课程关系":
        source_name = f"{values.get('学校', '学校')}课程与培养方案公开资料"
        source_locator = _source_path(
            values.get("学校"),
            values.get("课程类别"),
            values.get("代表课程/元素"),
        )
    elif record_type == "Reference来源":
        source_name = _source_label(
            values.get("发布机构"),
            values.get("文件/网页标题"),
            fallback="知识与教育图谱权威来源",
        )
        source_locator = _source_path(
            values.get("国家/范围"),
            values.get("页码/章节"),
        )
    elif record_type == "分析发现":
        source_name = "Human Knowledge Atlas × Education Graph（项目派生分析）"
        source_locator = _source_path(values.get("分析层"), values.get("发现") or values.get("解释"))
    else:
        source_name = f"Human Knowledge Atlas｜{record_type}"
        source_locator = str(chunk_key)
    return _name(source_name, 500), source_url, _name(source_locator or str(chunk_key), 500)


def _source_label(organization: str | None, title: str | None, *, fallback: str) -> str:
    parts = [str(value).strip() for value in (organization, title) if str(value or "").strip()]
    return "｜".join(parts) or fallback


def _source_path(*parts: str | None) -> str:
    return " → ".join(str(value).strip() for value in parts if str(value or "").strip())


def _citation_audit(documents: list[dict[str, Any]]) -> dict[str, int]:
    chunks = [chunk for document in documents for chunk in document["chunks"]]
    return {
        "total_chunks": len(chunks),
        "public_url_chunks": sum(
            bool(re.match(r"https?://", str(chunk.get("source_url") or ""), re.I))
            for chunk in chunks
        ),
        "internal_source_urls": sum(
            str(chunk.get("source_url") or "").lower().startswith("file:") for chunk in chunks
        ),
        "internal_source_names": sum(
            bool(re.search(r"\.(?:json|xlsx?|csv|md|html)$", str(chunk.get("source_name") or ""), re.I))
            for chunk in chunks
        ),
        "internal_source_locators": sum(
            bool(
                re.search(
                    r"(?:section:|!第\d+行|\brow[-:]?\d+\b|HTML\s*总览页)",
                    str(chunk.get("source_locator") or ""),
                    re.I,
                )
            )
            for chunk in chunks
        ),
    }


def _entity(name: str, entity_type: str, description: str = "") -> dict[str, str]:
    return {
        "name": _name(name),
        "entity_type": _name(entity_type, 60),
        "description": description.strip()[:500],
    }


def _entities(*items: tuple[str, str]) -> list[dict[str, str]]:
    return [_entity(name, entity_type) for name, entity_type in items if name.strip()]


def _relation(
    source: str,
    target: str,
    relation_type: str,
    source_type: str,
    target_type: str,
    values: dict[str, str],
) -> dict[str, Any]:
    evidence = "；".join(f"{name}={value}" for name, value in values.items() if value.strip())
    return _relation_simple(source, target, relation_type, source_type, target_type, evidence[:1000])


def _relation_simple(
    source: str,
    target: str,
    relation_type: str,
    source_type: str,
    target_type: str,
    evidence: str = "",
) -> dict[str, Any]:
    return {
        "source": _name(source),
        "target": _name(target),
        "relation_type": _name(relation_type, 60),
        "source_type": _name(source_type, 60),
        "target_type": _name(target_type, 60),
        "confidence": 1.0,
        "evidence": evidence[:1000],
    }


def _status_relation(value: str) -> str:
    normalized = value.strip().removesuffix(".0")
    return {
        "0": "NOT_CONFIRMED",
        "1": "EMBEDDED_OR_AVAILABLE",
        "2": "CORE_OR_EXPLICIT",
    }.get(normalized, "HAS_REPORTED_STATUS")


def _school_relation(value: str) -> str:
    if "核心" in value:
        return "CORE_COURSE"
    if "明确" in value:
        return "EXPLICITLY_OFFERS"
    if "选修" in value:
        return "ELECTIVE_COURSE"
    if "嵌入" in value:
        return "EMBEDDED_COURSE"
    return "HAS_COURSE_RELATION"


def _dedupe_entities(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        if not item["name"]:
            continue
        key = (item["name"], item["entity_type"])
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _dedupe_relations(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for item in items:
        if not item["source"] or not item["target"] or item["source"] == item["target"]:
            continue
        key = (
            item["source"],
            item["target"],
            item["relation_type"],
            item["source_type"],
            item["target_type"],
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _name(value: str, limit: int = 120) -> str:
    return " ".join(str(value).strip().split())[:limit]


def _first_value(values: dict[str, str], *names: str) -> str:
    return next((values[name].strip() for name in names if values.get(name, "").strip()), "")


def _validate_mirrored_counts(
    tables: dict[str, WorkbookTable],
    summary: dict[str, Any],
) -> None:
    expected = {
        "knowledge nodes": (len(tables["KG_All"].rows), summary["knowledge"]["nodes"]),
        "education entries": (len(tables["EduCourse300"].rows), summary["education"]["entries"]),
        "education progression": (
            len(tables["EduEvolution"].rows),
            summary["education"]["progression"],
        ),
        "schools": (len(tables["SchoolCases"].rows), summary["education"]["schools"]),
        "school edges": (len(tables["SchoolEdges"].rows), summary["education"]["school_edges"]),
    }
    mismatches = [f"{name}: workbook={left}, html={right}" for name, (left, right) in expected.items() if left != right]
    if mismatches:
        raise ValueError("Workbook and HTML counts do not match: " + "; ".join(mismatches))


def _read_html_summary(path: Path) -> dict[str, Any]:
    outer = path.read_text(encoding="utf-8")
    title_match = re.search(r"<title>(.*?)</title>", outer, re.S)
    knowledge_data = _embedded_data(outer, "K64")
    education_data = _embedded_data(outer, "E64")
    return {
        "title": title_match.group(1).strip() if title_match else path.stem,
        "knowledge": {"nodes": len(knowledge_data.get("nodes") or [])},
        "education": {
            "entries": len(education_data.get("entries") or []),
            "countries": len(education_data.get("countries") or []),
            "ages": len(education_data.get("ages") or []),
            "domains": len(education_data.get("domains") or []),
            "progression": len(education_data.get("progression") or []),
            "schools": len(education_data.get("schools") or []),
            "school_edges": len(education_data.get("schoolEdges") or []),
        },
    }


def _embedded_data(outer: str, variable: str) -> dict[str, Any]:
    if variable == "K64":
        pattern = rf"const\s+{variable}\s*=\s*'([^']+)'"
    else:
        pattern = rf"(?:^|,){variable}\s*=\s*'([^']+)'"
    match = re.search(pattern, outer, re.S)
    if not match:
        raise ValueError(f"Embedded {variable} document was not found.")
    decoded = base64.b64decode(match.group(1)).decode("utf-8")
    marker = re.search(r"const\s+DATA\s*=\s*", decoded)
    if not marker:
        raise ValueError(f"Embedded {variable} DATA object was not found.")
    body, _ = json.JSONDecoder().raw_decode(decoded[marker.end() :])
    if not isinstance(body, dict):
        raise ValueError(f"Embedded {variable} DATA must be an object.")
    return body


def _read_workbook_tables(path: Path) -> dict[str, WorkbookTable]:
    with zipfile.ZipFile(path) as archive:
        shared = _shared_strings(archive)
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        workbook_rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_map = {item.attrib["Id"]: item.attrib["Target"] for item in workbook_rels}
        tables: dict[str, WorkbookTable] = {}
        sheets = workbook.find("main:sheets", XML_NS)
        if sheets is None:
            return tables
        for sheet in sheets:
            sheet_name = sheet.attrib["name"]
            rel_id = sheet.attrib[f"{{{XML_NS['office_rel']}}}id"]
            sheet_path = "xl/" + rel_map[rel_id].lstrip("/")
            worksheet_rows = _worksheet_rows(archive, sheet_path, shared)
            rel_path = posixpath.join(
                posixpath.dirname(sheet_path),
                "_rels",
                posixpath.basename(sheet_path) + ".rels",
            )
            if rel_path not in archive.namelist():
                continue
            sheet_rels = ET.fromstring(archive.read(rel_path))
            for relation in sheet_rels:
                target = relation.attrib.get("Target", "")
                if "tables/" not in target:
                    continue
                table_path = posixpath.normpath(posixpath.join(posixpath.dirname(sheet_path), target))
                table_root = ET.fromstring(archive.read(table_path))
                namespace = {"main": table_root.tag.split("}")[0].lstrip("{")}
                table_name = table_root.attrib["displayName"]
                columns = [
                    item.attrib["name"]
                    for item in table_root.findall("main:tableColumns/main:tableColumn", namespace)
                ]
                _, start_row, _, end_row = _parse_range(table_root.attrib["ref"])
                rows: list[TableRow] = []
                for row_number in range(start_row + 1, end_row + 1):
                    values = worksheet_rows.get(row_number, {})
                    row_values = {
                        column: str(values.get(index, "")).strip()
                        for index, column in enumerate(columns)
                    }
                    if any(row_values.values()):
                        rows.append(TableRow(row_number=row_number, values=row_values))
                tables[table_name] = WorkbookTable(name=table_name, sheet_name=sheet_name, rows=rows)
        return tables


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return [_xml_text(item) for item in root.findall("main:si", XML_NS)]


def _worksheet_rows(
    archive: zipfile.ZipFile,
    sheet_path: str,
    shared: list[str],
) -> dict[int, dict[int, str]]:
    root = ET.fromstring(archive.read(sheet_path))
    rows: dict[int, dict[int, str]] = {}
    for row in root.findall(".//main:sheetData/main:row", XML_NS):
        row_number = int(row.attrib["r"])
        cells: dict[int, str] = {}
        for cell in row.findall("main:c", XML_NS):
            reference = cell.attrib.get("r", "A1")
            column_index = _column_index(reference)
            kind = cell.attrib.get("t")
            inline = cell.find("main:is", XML_NS)
            value_node = cell.find("main:v", XML_NS)
            value = ""
            if inline is not None:
                value = _xml_text(inline)
            elif value_node is not None:
                raw = value_node.text or ""
                value = shared[int(raw)] if kind == "s" and raw else raw
            cells[column_index] = value
        rows[row_number] = cells
    return rows


def _xml_text(node: ET.Element) -> str:
    return "".join(child.text or "" for child in node.iter() if child.tag.endswith("}t"))


def _parse_range(reference: str) -> tuple[int, int, int, int]:
    start, end = reference.split(":", 1)
    start_column, start_row = _cell_parts(start)
    end_column, end_row = _cell_parts(end)
    return start_column, start_row, end_column, end_row


def _cell_parts(reference: str) -> tuple[int, int]:
    match = re.fullmatch(r"([A-Z]+)(\d+)", reference)
    if not match:
        raise ValueError(f"Invalid cell reference: {reference}")
    return _column_index(match.group(1)), int(match.group(2))


def _column_index(reference: str) -> int:
    match = re.match(r"[A-Z]+", reference)
    if not match:
        raise ValueError(f"Invalid column reference: {reference}")
    value = 0
    for character in match.group(0):
        value = value * 26 + ord(character) - 64
    return value - 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
