"""Export the exact single-user run, summaries, model prompts and outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt

from scripts.teen_summary_scenario import FINAL_PROBES, PROMPTS

GROUPS = [('topics', '主题'), ('claims', '任务、目标与沟通偏好'),
          ('preference_observations', '结构化偏好'), ('learning_observations', '学习表现与趋势'),
          ('teaching_strategy_observations', '教学方法与反馈'), ('profile_change_proposals', '画像候选'),
          ('stable_profile', '稳定画像'), ('temporary_states', '临时状态'), ('limitations', '总结的限制'),
          ('teaching_preferences', '总结衍生的持久教学偏好')]
UUID_RE = re.compile(r'\b[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\b', re.I)


def load(folder, name):
    return json.loads((folder / name).read_text(encoding='utf-8'))


def summary_items(checkpoint, key):
    response = checkpoint['response']
    return response.get(key, []) if key == 'teaching_preferences' else response['summary'].get(key, [])


def export(source: Path, destination: Path):
    report = load(source, 'report.json')
    cleanup = load(source, 'isolation.json')
    review = load(source, 'semantic-review.json')
    rounds, probes = report['rounds'], report['probes']
    assert report.get('stage') == review.get('stage') == 'completed'
    assert len(rounds) == 150 and len(probes) == 8
    records = rounds + probes
    db = report['database']
    db_messages = {m['id']: m for m in db['navi_messages']}
    db_turns = {t['request_id']: t for t in db['navi_conversation_turns']}
    source_matches = len(db_messages) == 2 * len(records) and len(db_turns) == len(records)
    for r in records:
        row = db_turns.get(r['request']['request_id'], {})
        source_matches &= db_messages.get(row.get('user_message_id'), {}).get('content') == r['prompt']
        source_matches &= db_messages.get(row.get('assistant_message_id'), {}).get('content') == r['answer']
    prepared = {p['request_id']: p for p in report['prepared']}
    all_requests = {}
    for entry in report['model_requests']:
        all_requests.setdefault(entry['request_id'], []).append(entry['body'])
    chosen = [r for r in rounds if r['number'] in (51, 101)] + probes
    injection = []
    for r in chosen:
        rid = r['request']['request_id']
        p = prepared.get(rid, {})
        entries = p.get('summary_entries', [])
        actual = all_requests.get(rid, [])
        injection.append({'label': r['label'], 'selected_count': len(entries),
                          'in_actual_request': bool(entries) and any(
                              json.dumps(entries, ensure_ascii=False) in (m.get('content') or '')
                              for b in actual for m in b.get('messages', [])),
                          'history_count': p.get('runtime', {}).get('memory', {}).get('history_count'),
                          'status': p.get('runtime', {}).get('summary', {}).get('status'),
                          'actual_request_count': len(actual)})
    coverage = []
    source_ids = set()
    for s in report['summaries']:
        snapshot = load(source, f"summary-input-{s['target']}.json")
        references = snapshot['turns'] + snapshot.get('reference_turns', [])
        source_ids.update(t['completed_seq'] for t in references)
        coverage.append({'target': s['target'], 'input_turns': len(snapshot['turns']),
                         'reference_turns': len(snapshot.get('reference_turns', [])),
                         'omitted': snapshot['omitted_turn_count'],
                         'unique_source_turns': len({t['completed_seq'] for t in references}),
                         'missing_source_rounds': sorted(set(range(1, s['target'] + 1))
                                                        - {t['completed_seq'] for t in references}),
                         'input_start': min(t['completed_seq'] for t in snapshot['turns']),
                         'input_end': max(t['completed_seq'] for t in snapshot['turns']),
                         'all_150_in_final_input_and_refs': set(t['completed_seq'] for t in references) == set(range(1, 151)),
                         'job_status': s['job']['status'],
                         'attempt_count': s['job']['attempt_count'],
                         'sections': {key: len(summary_items(s, key)) for key, _ in GROUPS}})
    stage_checks = []
    age_limit_checks = []
    for r in records:
        p = prepared.get(r['request']['request_id'], {})
        audience = p.get('response_policy', {}).get('audience_profile', {})
        age = r['response'].get('metadata', {}).get('age_constraints', {})
        stage_checks.append(audience.get('age_years') == 14 and audience.get('age_stage') == 'S3'
                            and age.get('stage') == 'S3' and age.get('applied') is True)
        age_limit_checks.append(
            isinstance(age.get('final_body_chinese_chars'), int)
            and age['final_body_chinese_chars'] <= age.get('maximum_body_chars', 0)
            and isinstance(age.get('final_question_count'), int)
            and age['final_question_count'] <= age.get('maximum_questions', 0))
    checks = {
        '150轮用户输入逐条匹配预定脚本': [r['prompt'] for r in rounds] == PROMPTS,
        '所有基础对话均属于同一测试用户': len({r['request']['user_id'] for r in rounds}) == 1,
        '同一用户跨三个会话各50轮': sorted(
            sum(r['request']['session_id'] == sid for r in rounds)
            for sid in {r['request']['session_id'] for r in rounds}) == [50, 50, 50],
        '8轮独立验证均完成': [r['prompt'] for r in probes] == [p[0] for p in FINAL_PROBES],
        '附加验证属于同一用户且各用独立新会话':
            len({r['request']['session_id'] for r in probes}) == 8
            and {r['request']['user_id'] for r in records} == {report['user_id']}
            and not ({r['request']['session_id'] for r in probes}
                     & {r['request']['session_id'] for r in rounds}),
        '10个重点验证准备提示时近期历史均为空':
            len(injection) == 10 and all(x['history_count'] == 0 for x in injection),
        '数据库消息与全部158轮接口原文一致': source_matches,
        '三次自动总结均发布成功': len(coverage) == 3 and all(c['job_status'] == 'succeeded' for c in coverage),
        '三阶段总结输入的并集覆盖全部150轮': source_ids == set(range(1, 151)),
        '最终总结输入及参考原文覆盖全部150轮': bool(coverage) and coverage[-1]['all_150_in_final_input_and_refs'],
        '所有回答均应用14岁S3策略': all(stage_checks) and len(stage_checks) == 158,
        '正文汉字及问题数均在年龄控制上限内': all(age_limit_checks) and len(age_limit_checks) == 158,
        '10个重点验证均将所选总结送入实际模型请求': len(injection) == 10 and all(x['in_actual_request'] for x in injection),
        '测试数据库清理且schema确认不存在': cleanup.get('cleaned') and cleanup.get('schema_absent_verified'),
    }
    result = {'checks': checks, 'coverage': coverage, 'injection': injection,
              'rounds': len(rounds), 'probes': len(probes), 'semantic_review': review,
              'model_request_capture_caveat': report.get('instrumentation_note'),
              'age_policy_all_passed': all(stage_checks)}
    enum_inventory = {}
    for key in ('kind', 'progress', 'trend', 'preference_type', 'polarity', 'status'):
        enum_inventory[key] = sorted({item[key] for s in report['summaries'] for group, _ in GROUPS
                                      for item in summary_items(s, group)
                                      if isinstance(item, dict) and item.get(key)})
    result['actual_output_enum_inventory'] = enum_inventory

    template = Path('D:/daima/wendang/总结Agent_50轮对话与下一轮验证_2026-10-29.docx')
    doc = Document(template)
    for element in list(doc._element.body):
        if element.tag != qn('w:sectPr'):
            doc._element.body.remove(element)
    footer = doc.sections[0].footer.paragraphs[0]
    footer.clear()
    footer.add_run('同一位14岁学生 · 150轮对话与总结验证 · ')
    page = OxmlElement('w:fldSimple')
    page.set(qn('w:instr'), 'PAGE')
    footer._p.append(page)

    def para(text, style=None):
        return doc.add_paragraph(UUID_RE.sub('[内部标识已省略]', str(text)), style)

    def table(headers, rows):
        t = doc.add_table(rows=1, cols=len(headers))
        t.style = 'Light Shading Accent 1'
        for cell, value in zip(t.rows[0].cells, headers):
            cell.text = str(value)
        for row in rows:
            for cell, value in zip(t.add_row().cells, row):
                cell.text = str(value)
        repeat = OxmlElement('w:tblHeader')
        t.rows[0]._tr.get_or_add_trPr().append(repeat)
        for row in t.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    for run in p.runs:
                        run.font.size = Pt(9)

    doc.add_heading('总结 Agent：同一位14岁学生的150轮对话测试', 0)
    para('虚构人物：林晓，14岁，初二学生；测试日期：2026年10月30日。')
    para('本报告含150轮基础对话（每轮为一条用户消息及一条实际助手回答）、第50/100/150轮后的三份总结、以及8轮附加总结引用验证。第51轮和第101轮属于150轮内的跨会话验证，不额外计数。全部回答均来自真实项目服务。')
    doc.add_heading('1. 结果概览', 1)
    para(review['conclusion'])
    table(['核验项', '结果'], [(key, '通过' if value else '未通过') for key, value in checks.items()])
    doc.add_heading('首次失败与修复补测', 2)
    para(report.get('repair_note', '未发生生产代码修复。'))
    para('第50轮总结首次连续3次发生DataError，教学策略的observed_at以字符串传给PostgreSQL时间字段，发布事务回滚。修复为datetime参数后，使用同一50轮原文、同一固定快照重试。原始失败报告、诊断和重试审计均已保存。')
    para('先前91项专项回归通过；新增真实数据库教学偏好入库回归后，PostgreSQL套件16项通过。两次用例存在重叠，不能简单相加为独立用例总数。')
    for error in report.get('summary_errors', []):
        para(f"第{error['target']}轮总结生成异常：{error['type']}；{error['reason']}")
    doc.add_heading('内容复核', 2)
    for item in review.get('findings', []):
        para(item, 'List Bullet')
    if review.get('recommendations'):
        doc.add_heading('本轮发现的后续改进项', 2)
        for item in review['recommendations']:
            para(item, 'List Bullet')

    doc.add_heading('2. 测试设置', 1)
    para('所有基础对话使用同一个测试用户。第1—50轮为会话一，第51—100轮为会话二，第101—150轮为会话三；后续验证每题另开新会话，避免直接带入近期聊天原文。虚构初始生日和年级仅用于启用14岁年龄策略，姓名、猫名、偏好、任务及学习表现不预填。')
    para('使用临时LLM服务8017、真实模型、真实内容审核和PostgreSQL隔离schema。摘要阈值50轮，窗口150轮，输入预算128000（UTF-8字节保守上界），输出上限8000，超时300秒；摘要注入预算2000。测试参数仅在临时进程内设置。')
    para('每轮采用history_only保存真实消息，不运行独立长期记忆抽取器；总结发布仍可写入摘要衍生的偏好或计划。最终回答可能同时参考摘要及其衍生记忆，不能将每条回忆都归因于摘要单一路径。知识库服务未连接，其失败和旁路行为保留在结果里。')
    para('实际年龄规则为S3（12—14岁）：平等直接、解释必要术语、允许适度抽象和因果链；程序约束最多320个正文汉字及2个问题。汉字计数不同于包含标点、拉丁字母和换行的总字符数。规则已传入不等于每条回答都满足用户“两三句话”的偏好。')
    para('运行期间修正了一处测试记录器重复挂载观察回调的问题；前21轮的请求观察记录有重复，不能按该记录条数推算真实模型调用次数。原始对话与数据库保持不变，后续记录不再重复。')
    table(['总结节点', '近期轮数', '参考轮数', '近期遗漏', '去重覆盖', '状态/尝试'],
          [(c['target'], c['input_turns'], c['reference_turns'], c['omitted'], c['unique_source_turns'],
            f"{c['job_status']}/{c['attempt_count']}") for c in coverage])
    para('近期遗漏是窗口预算排除的轮数；其中部分原文可能重新作为旧总结的参考证据进入输入，以去重覆盖列为准。第50轮的尝试次数只计修复后同一任务重试，修复前3次另行保留。')
    for c in coverage:
        para(f"第{c['target']}轮总结未直接重读的基础轮次：" +
             ('、'.join(map(str, c['missing_source_rounds'])) or '无'))
    para('覆盖150轮表示实际输入或参考原文覆盖，不等于每一个事实都被保留。文档中的证据以轮次替代消息ID。')
    doc.add_heading('内容类型覆盖', 2)
    table(['输出类型', '第50轮后', '第100轮后', '第150轮后'],
          [(title, *(c['sections'][key] for c in coverage)) for key, title in GROUPS])
    table(['实际产出字段', '三份总结中出现的值'],
          [(key, '、'.join(values) or '无') for key, values in enum_inventory.items()])
    para('上表是实际产出清单，不把脚本涉及但模型未生成的字段算作已覆盖。权限开关、删除并发、过期时间推进、所有安全输入分支等并不由这150轮内容测试穷尽；本次相关代码回归与内容测试分开记录。')
    for item in review.get('coverage_notes', []):
        para(item, 'List Bullet')

    doc.add_heading('3. 总结 Agent 实际生成的内容', 1)
    for s in report['summaries']:
        doc.add_heading(f"第{s['target']}轮后的总结", 2)
        snapshot = load(source, f"summary-input-{s['target']}.json")
        mapping = {t[k]: (t['completed_seq'], '用户' if k == 'user_message_id' else '助手')
                   for t in snapshot['turns'] + snapshot.get('reference_turns', [])
                   for k in ('user_message_id', 'assistant_message_id')}
        for group, title in GROUPS:
            items = summary_items(s, group)
            doc.add_heading(f'{title}（{len(items)}项）', 3)
            if not items:
                para('未生成。')
            for item in items:
                if isinstance(item, str):
                    para(item)
                    continue
                para(item.get('topic') or item['content'])
                evidence = sorted({mapping[m] for m in item.get('evidence_message_ids', [])})
                if evidence:
                    para('证据轮次：' + '、'.join(f'第{n}轮{role}' for n, role in evidence))
                values = [f'{label}：{item[k]}' for k, label in [('status', '状态'), ('origin', '来源'),
                    ('progress', '进展'), ('trend', '趋势'), ('preference_type', '偏好类型'), ('polarity', '方向')]
                    if item.get(k)]
                if values:
                    para('；'.join(values))
                if item.get('counterevidence_message_ids'):
                    para('反证轮次：' + '、'.join(f'第{mapping[m][0]}轮' for m in item['counterevidence_message_ids']))
                if group in ('profile_change_proposals', 'stable_profile'):
                    para(f"证据轮数{item.get('evidence_turn_count', 0)}，会话数{item.get('evidence_session_count', 0)}；稳定证据门槛：{item.get('corroborated', False)}。")
                # Preserve all remaining semantic fields, not just the prose content.
                excluded = {'content', 'topic', 'claim_id', 'preference_id', 'source_summary_id',
                            'user_id', 'evidence_message_ids', 'counterevidence_message_ids', 'evidence'}
                fields = {k: v for k, v in item.items() if k not in excluded}
                if fields:
                    para('完整属性：' + json.dumps(fields, ensure_ascii=False))
                if item.get('evidence'):
                    def convert_evidence(value):
                        if isinstance(value, dict):
                            return {k: convert_evidence(v) for k, v in value.items()}
                        if isinstance(value, list):
                            return [convert_evidence(v) for v in value]
                        if isinstance(value, str) and value in mapping:
                            n, role = mapping[value]
                            return f'第{n}轮{role}'
                        return value
                    raw_evidence = item['evidence']
                    if isinstance(raw_evidence, str):
                        raw_evidence = json.loads(raw_evidence)
                    para('教学偏好证据：' + json.dumps(convert_evidence(raw_evidence), ensure_ascii=False))

    doc.add_heading('4. 引用总结后的实际问题与生成结果', 1)
    for r, diagnostic in zip(chosen, injection):
        label = f"基础对话第{r['number']}轮" if r['label'].startswith('round') else f"附加验证{r['number']}"
        doc.add_heading(label, 2)
        para('用户：' + r['prompt'])
        if r.get('expected'):
            para('核对目标：' + r['expected'])
        para('复核：' + review.get('probes', {}).get(r['label'], '见总体复核。'))
        para('实际回答：' + r['answer'])
        para('注入核验：' + json.dumps(diagnostic, ensure_ascii=False))
        p = prepared[r['request']['request_id']]
        doc.add_heading('实际选中的总结内容', 3)
        for entry in p['summary_entries']:
            para(entry, 'Quote')
        if not p['summary_entries']:
            para('本轮没有选中总结；按实际结果保留。')

    doc.add_heading('5. 全部150轮对话原文', 1)
    para('用户消息为预先设计的同一虚构人物脚本，助手消息为项目实际响应，没有将预期答案代替模型回答。每50轮等待总结生成后再继续下一会话。')
    for r in rounds:
        doc.add_heading(f"第{r['number']}轮", 2)
        para('用户：' + r['prompt'])
        para('助手：' + r['answer'])
        p = prepared.get(r['request']['request_id'], {})
        summary = p.get('runtime', {}).get('summary', {})
        para(f"会话{(r['number']-1)//50+1}；总结使用状态：{summary.get('status')}；实际选中{len(p.get('summary_entries', []))}条；响应耗时{r['elapsed_ms']}毫秒。")

    doc.add_heading('6. 引用总结后的完整模型提示词', 1)
    para('以下逐条列出第51、101轮及8个附加验证的实际出站messages，包含system、用户问题以及可能的工具调用和工具结果。保留原文，只将UUID形式内部标识替换为“内部标识已省略”；Authorization请求头从未写入记录。其余所有轮次的完整请求保存在report.json。')
    for r in chosen:
        doc.add_heading(r['label'], 2)
        requests = all_requests.get(r['request']['request_id'], [])
        if not requests:
            para('没有捕获到对外模型请求。若本轮走确定性直接回答，则不能声称调用了模型。')
        for i, body in enumerate(requests, 1):
            para(f"实际模型请求{i}；模型{body.get('model')}；生成上限{body.get('max_tokens')}。")
            for j, message in enumerate(body.get('messages', []), 1):
                doc.add_heading(f"消息{j} · {message.get('role')}", 3)
                para(message.get('content') or '')
                for k in ('name', 'tool_calls', 'tool_call_id'):
                    if message.get(k):
                        para(k + '：' + json.dumps(message[k], ensure_ascii=False))
            if body.get('tools'):
                para('同时发送的工具定义：' + json.dumps(body['tools'], ensure_ascii=False))

    doc.add_heading('7. 测试边界与证据', 1)
    para('本次测试一个14岁虚构人物的文本对话。没有验证实体设备、语音识别、语音合成、真实学校数据或长时间日常使用。用户脚本包含模拟的前后变化，实际运行是短时间内完成，不冒充跨数周观察。')
    para('生产代码仅修复已复现的教学反馈时间字段入库错误；没有改写模型输出。失败样本与修复前后记录均保留。年龄和年级是初始画像提供的信息，不将回忆它们计为总结独立记住的证明。')
    para('证据目录：' + str(source.resolve()))
    para('report.json含全部请求响应、完整提示词、已发布总结和数据库快照；summary-input-50/100/150.json保留固定输入；isolation.json记录清理；semantic-review.json是逐项内容复核；verification.json保存文档核验。')
    para('Word经ZIP、全部XML解析、正文回读、150轮逐条原文和总结全文一致性校验。当前没有Word/LibreOffice渲染器，未做逐页打印效果验收。')
    destination.parent.mkdir(parents=True, exist_ok=True)
    doc.save(destination)
    reopened = Document(destination)
    texts = [p.text for p in reopened.paragraphs]
    full = '\n'.join(texts)
    start = texts.index('5. 全部150轮对话原文')
    end = texts.index('6. 引用总结后的完整模型提示词')
    assert [t[3:] for t in texts[start:end] if t.startswith('用户：')] == [r['prompt'] for r in rounds]
    assert [t[3:] for t in texts[start:end] if t.startswith('助手：')] == [r['answer'] for r in rounds]
    for r in probes:
        assert r['prompt'] in full and r['answer'] in full
    for s in report['summaries']:
        for key, _ in GROUPS:
            for item in summary_items(s, key):
                assert (item if isinstance(item, str) else item.get('topic') or item['content']) in full
    for r in chosen:
        for request in all_requests.get(r['request']['request_id'], []):
            for message in request.get('messages', []):
                assert UUID_RE.sub('[内部标识已省略]', message.get('content') or '') in full
    assert not UUID_RE.search(full) and '\ufffd' not in full
    with ZipFile(destination) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if name.endswith('.xml'):
                ET.fromstring(z.read(name))
    result.update(docx=str(destination.resolve()), bytes=destination.stat().st_size,
                  sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                  docx_full_transcript_and_selected_prompts_verified=True, uuid_identifiers_removed=True)
    (source / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('semantic_review',)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.destination)
