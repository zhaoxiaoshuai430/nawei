"""Render the captured daily test as a readable Word/Markdown document."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


def decode(value):
    return json.loads(value) if isinstance(value, str) else value


def export(out):
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    db = report["database"]
    before_path = out / "database-before-51.json"
    before = json.loads(before_path.read_text(encoding="utf-8")) if before_path.exists() else db
    summary = report["summary"]["summary"]
    coverage = summary["coverage"]
    last = report["round_51"]
    runtime = last["response"]["metadata"]["runtime"]
    prep = next(item for item in report["prepared"] if item["round"] == 51)
    calls = [item for item in report["outbound_model_requests"] if item["round"] == 51]
    rows = sorted(db["navi_messages"], key=lambda item: (item["created_at"], item["id"]))
    expected = [(role, text) for t in report["rounds"] + [last]
                for role, text in (("user", t["prompt"]), ("assistant", t["answer"]))]
    actual = [(row["role"], row["content"]) for row in rows]
    snapshot = json.loads((out / "summary-input-snapshot.json").read_text(encoding="utf-8"))
    assert actual == expected, "Database messages differ from recorded responses"
    assert len(rows) == 102 and len(report["rounds"]) == 50
    assert [t["completed_seq"] for t in snapshot["turns"]] == list(range(1, 51))
    assert all((t["user_text"], t["assistant_text"]) == (original["prompt"], original["answer"])
               for t, original in zip(snapshot["turns"], report["rounds"]))
    assert coverage["actual_input_turn_count"] == 50 and not coverage["truncated"]
    assert runtime["summary"]["status"] == "applied"
    assert runtime["memory"]["history_count"] == 0
    systems = [m["content"] for call in calls for m in call["body"]["messages"] if m["role"] == "system"]
    injected = all(any(entry in system for system in systems) for entry in prep["summary_entries"])
    assert injected
    snapshot_ids = {t[key] for t in snapshot["turns"] for key in ("user_message_id", "assistant_message_id")}
    assert all(e["message_id"] in snapshot_ids for e in db["navi_summary_evidence"])
    verification = {"all_102_messages_match": True, "summary_all_50_turns_match": True,
        "coverage": {k: v for k, v in coverage.items() if k != "actual_input_turn_ids"},
        "evidence_ids_in_original_input": True, "round_51_summary_in_actual_model_request": injected,
        "round_51_model_calls": len(calls), "round_51_history_count": runtime["memory"]["history_count"],
        "reading_frequency": "通过：每月一本薄书", "communication_preferences": "通过：温和、幽默、不说教、先结论后步骤",
        "cat_name": "未通过：原始消息和长期记忆存在团子，但第51轮没有检索，回答没记住",
        "persistence_failures": report["health_after"]["runtime"]["persistence"]["failed"]}
    (out / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    doc = Document()
    section = doc.sections[0]
    section.top_margin = section.bottom_margin = Cm(2)
    section.left_margin = section.right_margin = Cm(2.1)
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
    normal.paragraph_format.space_after = Pt(6)
    for name in ("Title", "Heading 1", "Heading 2", "Heading 3"):
        style = doc.styles[name]
        style._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "微软雅黑")
        style.font.color.rgb = RGBColor.from_string("174A65")
    footer = section.footer.paragraphs[0]
    footer.alignment = 2
    footer.add_run("日常对话与长期记忆实测 · ")
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)
    md = []

    def heading(text, level=1):
        doc.add_heading(text, level=level)
        md.extend(["", "#" * (level + 1) + " " + text, ""])

    def para(text=""):
        for line in str(text).split("\n"):
            doc.add_paragraph(line)
        md.extend([str(text), ""])

    def table(headers, values):
        values = list(values)
        t = doc.add_table(rows=1, cols=len(headers))
        t.style = "Light Shading Accent 1"
        for cell, text in zip(t.rows[0].cells, headers):
            cell.text = str(text)
        for value in values:
            for cell, text in zip(t.add_row().cells, value):
                cell.text = str(text)
        md.append("| " + " | ".join(headers) + " |")
        md.append("| " + " | ".join("---" for _ in headers) + " |")
        for value in values:
            md.append("| " + " | ".join(str(v).replace("|", "／").replace("\n", "<br>") for v in value) + " |")
        md.append("")

    def raw(value):
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
        for line in text.splitlines():
            p = doc.add_paragraph(line)
            p.paragraph_format.space_after = Pt(0)
            for run in p.runs:
                run.font.size = Pt(9)
        md.extend(["```text", text, "```", ""])

    doc.add_heading("50 轮日常对话、阶段总结与第 51 轮记忆验证", 0)
    md.extend(["# 50 轮日常对话、阶段总结与第 51 轮记忆验证", ""])
    para("虚构人物：林舟 ｜ 真实模型与数据库实测 ｜ 2026 年 10 月 14 日")
    para("结论：50 轮完整入库并成功总结，第 51 轮确实使用了摘要；阅读目标和交流偏好回忆成功，猫名回忆失败。本文保留实际模型输出，没有把失败回答改成成功回答。")
    heading("1. 测试范围与结果")
    table(["项目", "实际结果"], [
        ("用户 ID", report["user_id"]), ("前 50 轮会话 ID", report["session_id"]),
        ("第 51 轮会话 ID", last["request"]["session_id"]),
        ("模型", last["response"]["metadata"]["model"]),
        ("对话存储", "51 个完整轮次，102 条消息；逐条核对 API 回答与 PostgreSQL 一致"),
        ("总结覆盖", "第 1—50 轮，实际输入 50，遗漏 0，未截断"),
        ("总结任务", report["summary_job"]["job_id"]),
        ("总结尝试", "第一次 invalid_schema；第二次成功，未手工改写摘要"),
        ("长期存储", f"{len(before['navi_memories'])} 条记忆、{len(before['navi_user_profiles'])} 份画像、{len(before['navi_learning_concepts'])} 条学习概念（第51轮前快照）"),
        ("摘要证据", f"{len(db['navi_summary_evidence'])} 条关联，全部指向真实输入消息"),
        ("第51轮", f"摘要 applied；选中 {runtime['summary']['selected_count']} 条；新会话历史 0；真实模型调用 {len(calls)} 次"),
        ("读取预算", "保留近期窗口；本次摘要输入上限 128000 字节保守估计，超时 300 秒"),
    ])
    para("使用临时本机 LLM HTTP 服务 8015，复用项目真实模型、内容审核、PostgreSQL 和 Redis。未测试麦克风、ASR、TTS 或 Gateway；不代表主服务 8002 已切换配置。此次临时服务在测试后关闭。")
    para("前 50 轮使用 fast 回答模式，但显式设置 LLM_FAST_POST_TURN_MODE=full，确保后台抽取长期记忆；每轮等待抽取完成。总结触发设为第 50 轮，前 50 轮不注入摘要；摘要生成后核对证据，再开启该测试用户的注入开关。未直接向数据库写入预设人物信息。")
    heading("2. 我生成的测试人物设定")
    table(["维度", "虚构设定"], report["synthetic_persona"].items())
    para("这些是编写合成用户消息时使用的测试设定，不等于项目实际提取结果。性格只记录人物自述与交流偏好，不进行心理诊断或人格评分。所有对话发生于本次运行，消息中的“明天”“这些天”等属于合成叙事，不是跨多天测试。")
    heading("3. 总结 Agent 的真实输出")
    para("以下逐项呈现数据库里已发布的摘要内容；每项同时列出它引用的原始轮次。覆盖 50 轮表示模型读入了全部 50 轮，不表示摘要逐字保留每一条事实。")
    groups = [("topics", "主题"), ("claims", "未完成事项、目标与交流偏好"),
              ("learning_observations", "学习观察"), ("teaching_strategy_observations", "教学策略观察"),
              ("profile_change_proposals", "画像修改提议"), ("temporary_states", "临时状态"), ("limitations", "局限说明")]
    evidence = db["navi_summary_evidence"]
    for key, name in groups:
        heading(f"{name}（{len(summary[key])} 项）", 2)
        if not summary[key]:
            para("空：本次没有生成该板块内容。")
        elif key == "limitations":
            for i, value in enumerate(summary[key], 1):
                para(f"{i}. {value}")
        else:
            for i, item in enumerate(summary[key], 1):
                refs = [e for e in evidence if e["claim_id"] == item["claim_id"]]
                seqs = sorted({e["completed_seq"] for e in refs})
                para(f"{i}. {item.get('topic', item.get('content'))}")
                para(f"来源轮次：{', '.join(map(str, seqs))}" +
                     (f"；类型：{item['kind']}；来源：{item['origin']}；状态：{item['status']}" if "kind" in item else ""))
    para("质量核查：摘要里的“仅提及杭州但未提供具体城市”自相矛盾；“未明确采纳建议”忽略了第43轮提前准备钥匙和电脑包的反馈。教学策略观察把助手在第26轮之后给的比喻当作用户正面反馈的例子，证据时序不严谨。交流偏好中的“结尾不追问”也比原文“不用每次都追问”更绝对。这些原始输出被保留，不作为已证实事实。")
    heading("4. 人物画像与长期记忆：系统实际保存了什么")
    para("下面是第51轮之前的数据库快照，便于区分此前形成的记忆和第51轮之后的新内容。数据库标签是项目原始分类，保留误分类以便评估。")
    heading("实际用户画像", 2)
    for profile in before["navi_user_profiles"]:
        raw(decode(profile["profile_data"]))
    heading("长期记忆完整清单", 2)
    memories = sorted(before["navi_memories"], key=lambda m: (m["created_at"], m["id"]))
    table(["编号", "数据库分类", "实际内容", "重要度 / 置信度", "已删除"],
          [(i, m["memory_type"], m["content"], f"{m['importance']} / {m['confidence']}", m["is_deleted"])
           for i, m in enumerate(memories, 1)])
    heading("实际学习概念", 2)
    for concept in before["navi_learning_concepts"]:
        raw(concept)
    para("值得注意：猫名“团子”和“性格慢热，早上喜欢安静一点”确实存在长期记忆。画像没有单独保存完整职业、年龄、猫名字段，不等于全部信息丢失，它们也可能存在 profile_fact 类长期记忆中。")
    para("抽取噪声包括：把手机影响工作写成学习弱点；把午休、睡眠、记账写成 learning_state；把“今天先到这里”“今天怎么选”写成计划；把“不会太粗糙”写成 weakness。用户记录了某个状态，也不意味着它应该永久作为人物特征。")
    heading("5. 第 51 轮：新会话真实验证")
    heading("用户输入原文", 2)
    para(last["prompt"])
    heading("AI 回答原文", 2)
    para(last["answer"])
    table(["检查项", "结果"], [("阅读频率", verification["reading_frequency"]),
        ("交流偏好", verification["communication_preferences"]), ("猫名", verification["cat_name"]),
        ("回复风格", "没有结尾追问，但仍较长，未严格达到一两个步骤的偏好"),
        ("完成与存储", f"finish_reason={last['response']['finish_reason']}；saved_turn=true；completed_seq=51"),
        ("响应耗时", f"{last['elapsed_ms']/1000:.2f} 秒（不含后台记忆抽取等待）"),
        ("模型报告用量", json.dumps(last["response"]["usage"], ensure_ascii=False)),
    ])
    heading("实际使用的上下文与工具", 2)
    table(["来源", "实际情况"], [
        ("近期历史", runtime["memory"]["history_count"]), ("画像字段", runtime["memory"]["profile_fields"]),
        ("长期记忆上下文", runtime["memory"]["memory_count"]), ("学习概念", runtime["memory"]["concept_count"]),
        ("知识库资料 / 关系", f"{runtime['knowledge']['chunk_count']} / {runtime['knowledge']['relation_count']}"),
        ("摘要", f"revision={runtime['summary']['summary_revision']}，{runtime['summary']['selected_count']} 条，读取 {runtime['summary']['read_ms']}ms"),
        ("实际工具调用", json.dumps(runtime["tools"]["selected_tools"], ensure_ascii=False)),
        ("可用工具", "、".join(runtime["tools"]["registered_tools"])),
    ])
    for i, entry in enumerate(prep["summary_entries"], 1):
        para(f"摘要输入 {i}：{entry}")
    para("全部选中摘要文字都在捕获到的实际 HTTP 模型请求中出现。猫名不在这7条摘要、当前画像和近期历史里；工具定义虽可用，模型没有调用 memory_search 或 history_search，因此它回答“没记住”。这是检索/上下文装配漏用已存信息，不能归结为数据库未保存。")
    heading("第51轮完整系统提示词和模型消息", 2)
    for i, call in enumerate(calls, 1):
        para(f"模型调用 {i}；模型 {call['body'].get('model')}；temperature={call['body'].get('temperature')}；stream={call['body'].get('stream')}。完整工具 schema 见 ROUND-51-REQUEST.json。")
        for message in call["body"]["messages"]:
            heading(f"角色：{message['role']}", 3)
            raw(message.get("content") or "")
    heading("第51轮持久化记录", 2)
    raw(last["response"]["metadata"]["persistence"])
    heading("6. 完整 50 轮日常对话")
    para("下面为模型实际回答，未修饰、压缩或重写。每轮均保存了用户消息和助手消息；小结仅描述运行记录，不替代原文。")
    for turn in report["rounds"]:
        heading(f"第 {turn['round']} 轮", 2)
        para("用户：\n" + turn["prompt"])
        para("AI：\n" + turn["answer"])
        meta = turn["response"]["metadata"]
        rt = meta.get("runtime", {})
        tools = meta.get("tool_agent", {}).get("selected_tools", rt.get("tools", {}).get("selected_tools", []))
        para(f"记录：{turn['elapsed_ms']/1000:.2f}秒；完成原因 {turn['response']['finish_reason']}；轮次已保存；"
             f"近期历史 {rt.get('memory', {}).get('history_count', '未报告')} 条；实际工具 {', '.join(tools) or '无'}。")
    heading("7. 总体评估与后续改进方向")
    para("已通过：真实51轮、102条消息逐条匹配；原始50轮完整输入总结；摘要成功发布、52条证据关联有效；第51轮新会话使用7条摘要；阅读目标与明确交流偏好回忆正确；全部后台持久化失败计数为0。")
    para("未通过：第51轮漏查已保存的猫名。第49轮在摘要尚未注入时漏掉早期偏好；部分日常问题误入教学讲解、回答过长并强行追问；摘要存在少量不严谨的概括；长期记忆有误分类、碎片和重复。人物记忆的存储、检索和实际使用是三个不同环节，本次不能宣称整体完全通过。")
    para("建议下一步优先补齐：对名字、人物关系等事实回忆，在摘要不足时检索长期记忆；避免把日常聊天路由成教学任务；去除无语义的计划和弱点条目；人物沟通偏好与本轮表达约束保持一致。本文是实测报告，没有为改善结果偷偷改写模型回答或测试数据。")
    heading("8. 原文证据与可复核附件")
    table(["摘要条目", "原始轮次", "原始用户/助手消息"], [
        (e["claim_id"], e["completed_seq"], f"[{e['role']}] {e['content']}") for e in evidence])
    para("report.json：全部请求响应、抽取完成状态、摘要、数据库快照；summary-input-snapshot.json：真实50轮模型输入；database-before-51.json：第51轮前数据库快照；ROUND-51-REQUEST.json：捕获的完整模型请求体（无授权头）；verification.json：逐条核对结果。")
    raw(verification)
    (out / "ROUND-51-REQUEST.json").write_text(json.dumps(calls, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "ROUND-51-SYSTEM-PROMPT.txt").write_text("\n\n".join(systems), encoding="utf-8")
    (out / "完整测试文档.md").write_text("\n".join(md), encoding="utf-8")
    doc.save(out / "日常对话50轮与第51轮记忆验证.docx")
    print(json.dumps({"docx": str(out / '日常对话50轮与第51轮记忆验证.docx'), "verification": verification}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.output)
