"""Prepare a revised 30-question set against read-only live graph evidence."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_light_health_comparison import QUESTIONS, collect_evidence, save_json
from services.knowledge.config import load_settings
from services.knowledge.neo4j_store import Neo4jKnowledgeStore

REPLACEMENTS = {
    3: ("在学科分类图谱中，光谱学与生物光学与光生物物理学分别属于哪个直接上位学科？请比较两条归属关系，并说明研究光谱对健康的影响为什么需要跨学科合作。",
        ["光谱学", "生物光学与光生物物理学"], "跨学科归属比较"),
    4: ("在学科分类图谱中，感官生理学与内分泌生理学有哪些共同的上位学科？请列出两级归属链，并区分这些分类关系与光影响视觉、激素的生理机制证据。",
        ["感官生理学", "内分泌生理学"], "共同祖先与证据边界"),
    7: ("建筑光学在学科分类图谱中向上两级分别属于什么学科？请列出完整关系链，并说明它与健康照明设计的联系，区分图谱事实和应用推论。",
        ["建筑光学"], "两跳上位链"),
    8: ("神经生理学在学科分类图谱中的直接上位学科是什么？它和生理学是否具有相同的直接上位学科？请说明两条归属路径，以及研究光的非视觉效应时各自能提供什么视角。",
        ["神经生理学", "生理学"], "不同归属路径比较"),
    10: ("色彩心理学与认知神经科学在学科分类图谱中是否具有共同的直接上位学科？请给出关系链，并说明颜色体验与光的神经效应为什么不能仅凭分类关系画等号。",
         ["色彩心理学", "认知神经科学"], "共同上位与非因果边界"),
    15: ("医学统计学与医学心理学在学科分类图谱中分别属于什么上位学科？在评价健康照明干预时，两者分别可以帮助回答哪些问题，分类关系又不能证明哪些临床结论？",
         ["医学统计学", "医学心理学"], "医学与心理学交叉关系"),
    18: ("年龄生理学与老年心理学（包括长寿心理学）在学科分类图谱中分别归属哪条学科路径？针对养老空间照明，如何区分年龄相关的生理需求与心理需求，哪些建议不能单靠分类图谱得出？",
         ["年龄生理学", "老年心理学（包括长寿心理学）"], "养老应用的跨分支比较"),
    19: ("学校心理学与学习心理学在学科分类图谱中有什么共同上位学科？请向上追溯两级，并说明评价教室照明对学习表现的影响时，这条关系链能帮助组织哪些研究问题、不能替代什么实验证据。",
         ["学校心理学", "学习心理学"], "教育应用的共同祖先"),
    23: ("环境质量监测与评价在学科分类图谱中属于哪个直接上位学科？请列出该上位学科下至少两个并列分支，并说明这种结构对评估城市夜景照明干扰有什么帮助及局限。",
         ["环境质量监测与评价"], "环境评价的父节点与兄弟节点"),
    24: ("心理统计原理与心理统计方法在学科分类图谱中如何通过共同上位节点连接？请列出到心理学的关系链，并说明健康照明效果评估为什么既需要测量设计，也需要统计方法。",
         ["心理统计原理", "心理统计方法"], "方法学的关系路径"),
}


def build_question_set():
    questions = []
    for index, (category, original) in enumerate(QUESTIONS, 1):
        replacement = REPLACEMENTS.get(index)
        questions.append({
            "index": index, "category": category,
            "question": replacement[0] if replacement else original,
            "original_question": original, "replaced": replacement is not None,
            "graph_targets": replacement[1] if replacement else [],
            "graph_task": replacement[2] if replacement else "",
        })
    return {"name": "光与健康_知识图谱增强题组", "revision": "graph-revision-1",
            "replaced_count": len(REPLACEMENTS), "evidence_policy": "fresh_retrieval",
            "questions": questions}


async def verify_graph_targets(question_set):
    settings = load_settings()
    graph = Neo4jKnowledgeStore(uri=settings.neo4j_uri, user=settings.neo4j_user,
                               password=settings.neo4j_password, database=settings.neo4j_database)
    verified = {}
    try:
        for item in question_set["questions"]:
            if not item["replaced"]:
                continue
            records, _, _ = await graph.driver.execute_query(
                """
                MATCH (target:Entity)
                WHERE target.knowledge_base_id = $base AND target.name IN $names
                OPTIONAL MATCH (parent:Entity)-[relation:RELATED]->(target)
                WHERE relation.relation_type = 'CONTAINS' AND parent.knowledge_base_id = $base
                RETURN DISTINCT target.name AS target, parent.name AS parent
                ORDER BY target, parent
                """, base="default", names=item["graph_targets"], database_=graph.database, routing_="r",
            )
            parents = [record.data() for record in records]
            if {row["target"] for row in parents if row["parent"]} != set(item["graph_targets"]):
                raise ValueError(f"Question {item['index']} has no verified graph parent: {parents}")
            verified[str(item["index"])] = parents
        return verified
    finally:
        await graph.close()


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    question_set = build_question_set()
    path = args.output / "question_set.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != question_set:
        raise ValueError("Question set changed: use a new directory")
    verified = await verify_graph_targets(question_set)
    save_json(path, question_set)
    save_json(args.output / "graph_target_verification.json", verified)
    print(json.dumps({"verified": verified}, ensure_ascii=True), flush=True)
    if args.verify_only:
        return
    evidence_path = args.output / "evidence.json"
    if evidence_path.exists():
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if [(item["category"], item["question"]) for item in evidence["cases"]] != [
            (item["category"], item["question"]) for item in question_set["questions"]
        ]:
            raise ValueError("Existing evidence does not match revised questions")
    else:
        evidence = await collect_evidence(args.output, questions=[
            (item["category"], item["question"]) for item in question_set["questions"]
        ])
        evidence["question_set_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        evidence["evidence_policy"] = "fresh_retrieval"
        save_json(evidence_path, evidence)
    audit = []
    for case in evidence["cases"]:
        if case["index"] not in REPLACEMENTS:
            continue
        parents = verified[str(case["index"])]
        relations = case["graph_context"]["relations"]
        recalled = [pair for pair in parents if any(
            relation["source_name"] == pair["parent"] and relation["target_name"] == pair["target"]
            and relation["relation_type"] == "CONTAINS" for relation in relations
        )]
        audit.append({"index": case["index"], "target_parents_in_graph": parents,
                      "target_parents_recalled": recalled, "relation_count": len(relations)})
    save_json(args.output / "graph_recall_audit.json", audit)
    print(json.dumps({"graph_nonempty_questions": sum(bool(case["graph_context"]["relations"]) for case in evidence["cases"]),
                      "replaced": len(audit), "target_parent_recall_questions": sum(bool(row["target_parents_recalled"]) for row in audit)},
                     ensure_ascii=True), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
