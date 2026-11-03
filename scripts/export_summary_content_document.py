"""Export captured content tests, coverage and review without inventing outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt


LABELS = {
    'topics': '主题', 'claims': '任务、目标与沟通偏好', 'preference_observations': '结构化偏好',
    'learning_observations': '学习观察', 'teaching_strategy_observations': '教学策略反馈',
    'profile_change_proposals': '画像候选', 'temporary_states': '临时状态',
    'limitations': '总结限制', 'stable_profile': '稳定画像（服务端派生）',
    'progress': '任务/目标进度', 'trend': '学习趋势', 'preference_type': '偏好类型',
    'polarity': '偏好方向', 'status': '结论状态', 'origin': '来源类型', 'scope': '适用范围',
    'kinds': '内容类型',
}


def export(source: Path, destination: Path) -> None:
    report = json.loads((source / 'report.json').read_text(encoding='utf-8'))
    review_path = source / 'semantic-review.json'
    review = json.loads(review_path.read_text(encoding='utf-8')) if review_path.exists() else {}
    inventory = report['inventory']
    observed: dict[str, set] = {k: set() for k in inventory}
    structural = []
    rounds = []
    probes = []
    expected_rounds = sum(len(c['scenario']['prompts']) for c in report['cases'].values())
    expected_probes = sum(len(c['scenario']['probes']) for c in report['cases'].values())
    expected_summaries = 2 * len(report['cases'])
    succeeded_summaries = sum(bool(s.get('document')) for c in report['cases'].values() for s in c['summaries'])
    for case_id, case in report['cases'].items():
        rounds.extend(case['rounds'])
        probes.extend(case['probes'])
        for stage in case['summaries']:
            snap = stage['snapshot']
            inputs = snap['turns'] + snap.get('reference_turns', [])
            ids = {t[key] for t in inputs for key in ('user_message_id', 'assistant_message_id')}
            by_round = {r['round']: r for r in case['rounds']}
            source_matches = all(t['user_text'] == by_round[t['completed_seq']]['prompt']
                                 and t['assistant_text'] == by_round[t['completed_seq']]['answer'] for t in inputs)
            evidence_ok = bool(stage.get('document'))
            for group, items in stage.get('document', {}).items():
                if not isinstance(items, list):
                    continue
                if items:
                    for inv in ('sections', 'derived_sections'):
                        if group in inventory[inv]:
                            observed[inv].add(group)
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    evidence_ok &= set(item.get('evidence_message_ids', []) + item.get('counterevidence_message_ids', [])) <= ids
                    for key in inventory:
                        value = item.get('kind' if key == 'kinds' else key)
                        if value is not None:
                            observed[key].add(value)
            structural.append({'case': case_id, 'phase': stage['phase'], 'source_matches': source_matches,
                               'evidence_valid': evidence_ok, 'input_turns': len(snap['turns']),
                               'reference_turns': len(snap.get('reference_turns', [])),
                               'omitted': snap['omitted_turn_count'],
                               'previous_revision': snap.get('previous_summary_revision')})
    checks = {
        f'{expected_rounds}轮真实模型对话已完整记录': len(rounds) == expected_rounds,
        f'{expected_summaries}次总结全部成功': len(structural) == expected_summaries and all(x['evidence_valid'] for x in structural),
        '总结输入逐条匹配真实对话': all(x['source_matches'] for x in structural),
        '近期输入无预算遗漏': all(x['omitted'] == 0 for x in structural),
        '各组第二阶段输入均引用第一阶段总结': sum(x['previous_revision'] == 1 for x in structural) == len(report['cases']),
        f'{expected_probes}次新问题均完成且不带原始聊天历史': len(probes) == expected_probes and all(p['history_count'] == 0 for p in probes),
        f'{expected_probes}次均有总结进入实际模型请求': len(probes) == expected_probes and all(p['entries'] and all(
            e in p['request']['messages'][0]['content'] for e in p['entries']) for p in probes),
        f'所有{expected_rounds + expected_probes}次对话请求均正常结束': len(rounds) + len(probes) == expected_rounds + expected_probes and all(
            p['response']['choices'][0]['finish_reason'] == 'stop' for p in rounds + probes),
    }
    coverage = {k: {'expected': values, 'observed': sorted(observed[k] & set(values)),
                    'not_emitted': sorted(set(values) - observed[k])} for k, values in inventory.items()}
    regressions = ET.parse(source / 'regression.xml').getroot()
    suites = list(regressions.iter('testsuite'))
    regression = {k: sum(int(s.get(k, '0')) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}

    doc = Document()
    section = doc.sections[0]
    section.top_margin = section.bottom_margin = Cm(1.8)
    section.left_margin = section.right_margin = Cm(2)
    for name in ('Normal', 'Title', 'Subtitle', 'Heading 1', 'Heading 2', 'Heading 3'):
        style = doc.styles[name]
        style.font.name = '微软雅黑'
        style._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), '微软雅黑')
    doc.styles['Normal'].font.size = Pt(10)
    doc.styles['Normal'].paragraph_format.space_after = Pt(4)
    footer = section.footer.paragraphs[0]
    footer.add_run('总结 Agent 内容覆盖测试 · ')
    field = OxmlElement('w:fldSimple')
    field.set(qn('w:instr'), 'PAGE')
    footer._p.append(field)
    doc.add_heading('总结 Agent 全内容类型补充对话与测试', 0)
    doc.add_paragraph(f'2026-10-30 · {len(rounds)} 轮新增对话 + {len(probes)} 轮总结引用追问 · 合成测试人物')
    doc.add_heading('1. 结论与范围', 1)
    doc.add_paragraph(review.get('conclusion', '语义复核尚未完成，不能把结构校验成功等同于所有内容正确。'))
    doc.add_paragraph(f"本次新增对话与上一份50轮报告相互独立，保留原报告。用户消息为预先设计的合成测试情境；{len(rounds)}条助手回复、{expected_summaries}个阶段总结任务、{len(probes)}条最终追问回复均调用真实配置模型，未用预期答案替代。其中成功总结{succeeded_summaries}份。生成模型：" + report['cases']['life']['summaries'][0]['model'])
    doc.add_paragraph('运行边界：本机 PostgreSQL 连接被拒绝，Docker 引擎访问被系统权限拒绝，常规五个服务端口未响应；启动脚本未完成。本次直接调用真实模型，并使用项目 ConversationSummaryAgent、pack、normalize、read_context 和 render_context 原有实现。总结读取使用内存适配器，无数据库写入，未验证 HTTP 路由、持久化、调度器、长期记忆抽取、音频或实体设备。对话采集使用本测试的简短 system 提示词，不是完整主 Agent 编排。')
    doc.add_paragraph(f'每组对话按顺序生成，标记为两个测试会话；第二段采集仍把前段对话作为历史提供给模型。跨会话证据门槛由生产normalize函数检验，但这不等于真实跨会话数据库链路验收。最终{len(probes)}个追问只带当前问题及所选摘要，不带原始对话。')
    doc.add_paragraph('原始60轮之外，增加6轮“稳定画像”对照及2个追问；该组输出被模型归为偏好，再增加8轮以实际行为为主的独立对照及1个追问。原画像组第二阶段连续三次被隐私检查拦截，未改写原始失败，也未降低保护规则；该组3个追问使用最后可用的第一阶段总结，以实测旧总结导致的回忆缺口。新对照组独立生成，不能算作原失败已修复。')
    doc.add_paragraph('状态、趋势等字段是可选输出；未输出某枚举不一定是错误，也不代表实测覆盖成功。expired 过期通常由服务端时间判断，不能靠数分钟实验声称验证真实两小时到期。')
    for label, ok in checks.items():
        doc.add_paragraph(('通过：' if ok else '未通过：') + label)
    doc.add_paragraph(f"专项回归：{regression['tests']} 项；失败 {regression['failures']}；错误 {regression['errors']}；跳过 {regression['skipped']}。这是独立的自动化回归，不等同于真实数据库测试。")
    for finding in review.get('findings', []):
        doc.add_paragraph(finding, style='List Bullet')

    doc.add_heading('2. 能力清单与实际输出覆盖', 1)
    doc.add_paragraph('依据 services/llm/summary/schema.py、agents/summary_agent.py、summary/repository.py 与 summary/context.py 核对；“出现”仅说明模型在至少一份总结中输出该字段，语义正确性另见复核。')
    table = doc.add_table(rows=1, cols=3)
    table.style = 'Light Shading Accent 1'
    for cell, text in zip(table.rows[0].cells, ['维度', '真实总结中已出现', '本次未输出']):
        cell.text = text
    for key, value in coverage.items():
        cells = table.add_row().cells
        cells[0].text = LABELS.get(key, {'sections': '总结主栏目', 'derived_sections': '派生栏目'}.get(key, key))
        cells[1].text = '、'.join(value['observed']) or '无'
        cells[2].text = '、'.join(value['not_emitted']) or '无'
    doc.add_paragraph('关联回归：证据门槛和画像晋升；未知消息ID/助手消息不能证明用户事实；教学反馈必须晚于讲解；趋势至少两次用户表现；临时状态过期过滤；tentative/contradicted/inferred/反证过滤；当前纠正优先；任务结束后不再当待办；隐私及诊断标签阻断。精确用例名见证据目录 regression.xml。')
    doc.add_heading('场景与原始对话轮次对应', 2)
    for item in review.get('scenario_coverage', []):
        doc.add_paragraph(item['dimension'] + '：' + item['source'])
        doc.add_paragraph('核对点：' + item['expected'])
    doc.add_paragraph('各阶段总结输入核验：' + json.dumps(structural, ensure_ascii=False))
    if report['errors']:
        doc.add_paragraph('运行中失败及重试原始记录：' + json.dumps(report['errors'], ensure_ascii=False))
    for filename in ('profile-phase-2-privacy-diagnostic.json', 'profile-label-diagnostic.json'):
        diagnostic_path = source / filename
        if diagnostic_path.exists():
            doc.add_paragraph('额外诊断调用（不计作成功总结）：' + diagnostic_path.read_text(encoding='utf-8'))

    doc.add_heading(f'3. {len(probes)}轮总结引用与最终结果', 1)
    probe_review = review.get('probes', {})
    for case_id, case in report['cases'].items():
        for p in case['probes']:
            label = f"{case_id}-{p['number']}"
            doc.add_heading(f"{label}：{case['title']}", 2)
            doc.add_paragraph('用户：' + p['query'])
            doc.add_paragraph('预期核对点：' + p['expected'])
            doc.add_paragraph('语义复核：' + probe_review.get(label, '待复核'))
            if p.get('latest_summary_failed'):
                doc.add_paragraph(f"注意：最新阶段失败，本轮仅使用阶段{p['source_summary_phase']}的旧总结。")
            doc.add_paragraph('实际最终回答：' + p['answer'])
            doc.add_paragraph('注入诊断：' + json.dumps(p['diagnostics'], ensure_ascii=False))
            doc.add_paragraph('以下为实际请求中的全部总结条目：')
            for entry in p['entries']:
                doc.add_paragraph(entry, style='Quote')

    doc.add_heading(f'4. {expected_summaries}个总结任务及{succeeded_summaries}份成功总结原文', 1)
    for case_id, case in report['cases'].items():
        for stage in case['summaries']:
            doc.add_heading(f"{case['title']} · 阶段{stage['phase']}", 2)
            doc.add_paragraph('生成尝试：' + json.dumps(stage['attempts'], ensure_ascii=False))
            if not stage.get('document'):
                doc.add_paragraph('本阶段未通过输出审核，没有可用总结。失败记录已保留，拦截的内容未送入后续对话。')
                continue
            sources = {t[key]: f"{case_id}第{t['completed_seq']}轮{'用户' if key == 'user_message_id' else '助手'}"
                       for t in stage['snapshot']['turns'] + stage['snapshot'].get('reference_turns', [])
                       for key in ('user_message_id', 'assistant_message_id')}
            for group, items in stage['document'].items():
                if not isinstance(items, list):
                    continue
                doc.add_heading(f"{LABELS.get(group, group)}（{len(items)}项）", 3)
                if not items:
                    doc.add_paragraph('未生成。')
                for item in items:
                    if isinstance(item, str):
                        doc.add_paragraph(item)
                        continue
                    doc.add_paragraph(item.get('topic') or item['content'])
                    doc.add_paragraph('证据：' + '、'.join(sources[mid] for mid in item['evidence_message_ids']))
                    if item.get('counterevidence_message_ids'):
                        doc.add_paragraph('反证：' + '、'.join(sources[mid] for mid in item['counterevidence_message_ids']))
                    p = doc.add_paragraph(json.dumps({k: v for k, v in item.items() if k not in ('content', 'topic')}, ensure_ascii=False))
                    for run in p.runs:
                        run.font.size = Pt(8)

    doc.add_heading(f'5. 新增{len(rounds)}轮完整对话', 1)
    global_number = 0
    for case_id, case in report['cases'].items():
        doc.add_heading(case['title'], 2)
        for r in case['rounds']:
            global_number += 1
            doc.add_heading(f"新增第{global_number}轮（{case_id}-{r['round']}，会话{r['phase']}）", 3)
            doc.add_paragraph('用户：' + r['prompt'])
            doc.add_paragraph('助手：' + r['answer'])

    doc.add_heading('6. 证据、限制与复现', 1)
    doc.add_paragraph('证据目录：' + str(source.resolve()))
    doc.add_paragraph('report.json保存真实对话请求与响应、通过审核的模型候选与规范化总结、消息ID、证据来源、注入上下文与最终输出；各阶段snapshot/candidate文件保存中间证据；regression.xml为自动化回归记录；semantic-review.json为逐项语义复核；verification.json为文档及结构核验。被审核阻止的完整输出不保存在报告中，诊断文件仅记录非敏感匹配片段。')
    doc.add_paragraph('复现入口：python -m scripts.run_summary_content_test --output <新的独立证据目录>。已有目录会从已完成轮次继续，不应作为独立重复实验使用。')
    doc.add_paragraph('测试数据只写本地文件，没有创建或删除真实数据库用户。没有改写生产总结逻辑。结论只适用于本次合成情境与模型输出，不能声称覆盖所有可能的自然语言输入。')
    doc.add_paragraph(f'Word文件经过ZIP完整性、全部XML解析、正文重开及{len(rounds) + len(probes)}轮对话逐条回读校验。当前无Word/LibreOffice排版渲染器，未做逐页截图或打印分页验收。')
    destination.parent.mkdir(parents=True, exist_ok=True)
    doc.save(destination)
    reopened = Document(destination)
    full_text = '\n'.join(p.text for p in reopened.paragraphs)
    for r in rounds:
        assert r['prompt'] in full_text and r['answer'] in full_text
    for p in probes:
        assert p['query'] in full_text and p['answer'] in full_text
        assert all(e in full_text for e in p['entries'])
    with ZipFile(destination) as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            if name.endswith('.xml'):
                ET.fromstring(archive.read(name))
    result = {'checks': checks, 'coverage': coverage, 'regression': regression,
              'docx': str(destination.resolve()), 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
              'dialogue_count': len(rounds), 'probe_count': len(probes), 'summary_tasks': len(structural),
              'successful_summary_count': succeeded_summaries,
              'roundtrip_all_dialogue': True, 'zip_xml_valid': True}
    (source / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.destination)
