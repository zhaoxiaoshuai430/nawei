"""Real-model content coverage without DB writes; not an HTTP/persistence test.

python -m scripts.run_summary_content_test --output .runtime/summary-content-YYYYMMDD
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from dotenv import load_dotenv
import httpx

from scripts.summary_content_scenarios import CONTENT_INVENTORY, SCENARIOS
from services.llm.agents.summary_agent import ConversationSummaryAgent, prompt_overhead
from services.llm.summary.context import read_context, render_context
from services.llm.summary.policy import SummarySettings
from services.llm.summary.repository import normalize
from services.llm.summary.snapshot import pack

SYSTEM = "你是中文对话助手。根据用户当前消息及有证据的历史回答，先结论后简短步骤，通常三四句话。不要编造用户事实，不将自己的建议当作用户承诺，不作人格或心理诊断。"


def save(path: Path, value: object) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    temporary.replace(path)


async def run(output: Path) -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / '.env', override=False)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'report.json'
    settings = SummarySettings(enabled=True, apply_to_prompt=True, input_max_tokens=128000,
                               output_max_tokens=8000, timeout_seconds=300, prompt_max_tokens=2000)
    report = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {
        'started_at': datetime.now(timezone.utc).isoformat(),
        'mode': 'real_model_dialogue_and_production_summary_with_in_memory_repository_adapter',
        'scope': '真实模型对话、生产总结Agent、normalize证据校验、read_context选择、实际模型请求注入；无HTTP服务/数据库/定时任务/长期记忆写入。',
        'infrastructure': {'postgres': 'connection_refused', 'docker': 'access_denied',
                           'regular_services': 'unreachable', 'data_writes': 'local_evidence_only'},
        'synthetic_users': True, 'settings': settings.model_dump(), 'inventory': CONTENT_INVENTORY,
        'cases': {}, 'errors': [],
    }
    agent = ConversationSummaryAgent(settings)
    base = os.environ['LLM_API_BASE_URL'].rstrip('/')
    url = base if base.endswith('/chat/completions') else base + '/chat/completions'
    headers = {'Authorization': 'Bearer ' + os.environ['LLM_API_KEY']}

    async def chat(messages: list[dict], label: str) -> dict:
        payload = {'model': os.environ['LLM_MODEL'], 'messages': messages, 'temperature': 0,
                   'max_tokens': 900, 'stream': False, 'thinking': {'type': 'disabled'}}
        for attempt in range(1, 4):
            try:
                async with httpx.AsyncClient(timeout=150) as client:
                    response = await client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
                choice = data['choices'][0]
                if choice.get('finish_reason') != 'stop' or not isinstance(choice['message'].get('content'), str):
                    raise ValueError('incomplete_model_output')
                return {'request': payload, 'response': data, 'answer': choice['message']['content'],
                        'attempt': attempt, 'at': datetime.now(timezone.utc).isoformat()}
            except (httpx.HTTPError, ValueError) as exc:
                report['errors'].append({'stage': label, 'attempt': attempt, 'type': type(exc).__name__})
                save(path, report)
                if attempt == 3:
                    raise
                await asyncio.sleep(2)
        raise AssertionError('unreachable')

    try:
        for scenario in SCENARIOS:
            case = report['cases'].setdefault(scenario['id'], {
                'title': scenario['title'], 'rounds': [], 'summaries': [], 'probes': [],
                'scenario': scenario,
            })
            total = len(scenario['prompts'])
            for number, prompt in enumerate(scenario['prompts'], 1):
                phase = 1 if number <= scenario['split'] else 2
                if number > len(case['rounds']):
                    history = [{'role': 'system', 'content': SYSTEM}]
                    # Across the two labeled sessions the real model receives the preceding
                    # dialogue as context; summary-only follow-up is isolated in probes.
                    for old in case['rounds']:
                        history.extend([{'role': 'user', 'content': old['prompt']},
                                        {'role': 'assistant', 'content': old['answer']}])
                    history.append({'role': 'user', 'content': prompt})
                    result = await chat(history, f"{scenario['id']}:round:{number}")
                    case['rounds'].append({**result, 'round': number, 'prompt': prompt, 'phase': phase,
                        'turn_id': str(uuid4()), 'user_message_id': str(uuid4()),
                        'assistant_message_id': str(uuid4()),
                        'session_id': f"synthetic-{scenario['id']}-session-{phase}"})
                    save(path, report)
                    print(f"DIALOGUE {scenario['id']} {number}/{total}", flush=True)
                if number not in (scenario['split'], total) or len(case['summaries']) >= phase:
                    continue
                rows = [{**r, 'completed_seq': r['round'], 'user_text': r['prompt'], 'assistant_text': r['answer'],
                         'completed_at': datetime.fromisoformat(r['at'])} for r in case['rounds'][:number]]
                prior = next((s for s in reversed(case['summaries']) if s.get('document')), None)
                snapshot = pack(list(reversed(rows if not prior else rows[scenario['split']:])),
                                prior['document'] if prior else None,
                                rows[:scenario['split']] if prior else [],
                                prior['snapshot'] if prior else {}, settings.input_max_tokens-prompt_overhead())
                attempts_path = output / f"{scenario['id']}-phase-{phase}-attempts.json"
                previous_attempts = json.loads(attempts_path.read_text(encoding='utf-8')) if attempts_path.exists() else []
                entry = {'phase': phase, 'snapshot': snapshot, 'attempts': previous_attempts, 'status': 'failed'}
                save(output / f"{scenario['id']}-phase-{phase}-snapshot.json", snapshot)
                for attempt in range(len(previous_attempts) + 1, 4):
                    request_id = f"content-{scenario['id']}-{phase}-{uuid4().hex}"
                    try:
                        print(f"SUMMARY {scenario['id']} phase={phase} attempt={attempt}", flush=True)
                        candidate, model = await agent.generate(snapshot, request_id)
                        raw = candidate.model_dump(mode='json')
                        save(output / f"{scenario['id']}-phase-{phase}-candidate-{attempt}.json", raw)
                        document, evidence = normalize(candidate, snapshot)
                        document['summary_revision'] = phase
                        entry.update(candidate=raw, document=document, model=model, evidence=evidence, status='passed')
                        entry['attempts'].append({'attempt': attempt, 'request_id': request_id, 'status': 'passed'})
                        break
                    except Exception as exc:
                        error = {'attempt': attempt, 'request_id': request_id, 'status': 'failed',
                                 'type': type(exc).__name__}
                        if isinstance(exc, ValueError):
                            # Fixed local error strings only, never emit HTTP request headers.
                            error['reason'] = str(exc)[:180]
                        entry['attempts'].append(error)
                        report['errors'].append({'stage': f"{scenario['id']}:summary:{phase}", **error})
                        save(output / f"{scenario['id']}-phase-{phase}-attempts.json", entry['attempts'])
                        save(path, report)
                case['summaries'].append(entry)
                save(path, report)
            accepted = next((s for s in reversed(case['summaries']) if s.get('document')), None)
            document = accepted['document'] if accepted else {}

            async def read(_: str) -> dict:
                return {'summary': document, 'enabled': True, 'apply_to_prompt': True, 'memory_epoch': 0}

            repository = SimpleNamespace(settings=settings, read=read)
            for number, probe in enumerate(scenario['probes'], 1):
                if number <= len(case['probes']):
                    continue
                diagnostics = {}
                context = await read_context(repository, 'synthetic-' + scenario['id'], probe['query'],
                                             'full', history=[], diagnostics=diagnostics)
                messages = [{'role': 'system', 'content': SYSTEM + render_context(context)},
                            {'role': 'user', 'content': probe['query']}]
                result = await chat(messages, f"{scenario['id']}:probe:{number}")
                case['probes'].append({**probe, **result, 'number': number, 'diagnostics': diagnostics,
                                      'entries': list(context.entries) if context else [], 'history_count': 0,
                                      'source_summary_phase': accepted['phase'] if accepted else None,
                                      'latest_summary_failed': not case['summaries'][-1].get('document')})
                save(path, report)
                print(f"PROBE {scenario['id']} {number}/{len(scenario['probes'])} {diagnostics['status']}", flush=True)
        report['completed_at'] = datetime.now(timezone.utc).isoformat()
        report['stage'] = 'complete'
    finally:
        save(path, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.output))
