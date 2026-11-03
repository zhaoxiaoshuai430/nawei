"""Real HTTP/DB/model test for one synthetic 14-year-old, in a private schema."""
from __future__ import annotations

import argparse
import asyncio
from contextvars import ContextVar
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import asyncpg
from dotenv import load_dotenv
import httpx
import uvicorn

from scripts.teen_summary_scenario import FINAL_PROBES, PERSONA, PROMPTS

ROOT = Path(__file__).resolve().parents[1]
PORT = 8017
ACTIVE_REQUEST = ContextVar('teen_test_active_request', default='')


def dump(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    tmp.replace(path)


async def run(out: Path):
    load_dotenv(ROOT / '.env', override=False)
    out.mkdir(parents=True, exist_ok=True)
    base = os.getenv('DATABASE_URL', 'postgresql://nawei:nawei@127.0.0.1:5432/nawei')
    iso_path = out / 'isolation.json'
    iso = json.loads(iso_path.read_text(encoding='utf-8')) if iso_path.exists() else {
        'schema': 'teen_summary_' + uuid4().hex, 'created': False, 'cleaned': False,
        'user_id': out.name + '-' + uuid4().hex[:8],
    }
    assert not iso['cleaned'], 'Completed evidence must not be overwritten; use a new directory'
    schema, uid = iso['schema'], iso['user_id']
    assert schema.startswith('teen_summary_') and schema.replace('_', '').isalnum()
    parts = urlsplit(base)
    query = [(k, v) for k, v in parse_qsl(parts.query) if k != 'search_path'] + [('search_path', schema)]
    isolated = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))
    os.environ.update(DATABASE_URL=isolated, REDIS_URL=os.getenv('REDIS_URL', 'redis://127.0.0.1:6379/0'),
        LLM_PORT=str(PORT), LLM_DRAIN_DIRECTORY=str(out / 'drain'), LLM_WORKERS='1',
        SUMMARY_ENABLED='true', SUMMARY_APPLY_TO_PROMPT='true', SUMMARY_TRIGGER_TURNS='50',
        SUMMARY_WINDOW_TURNS='150', SUMMARY_INPUT_MAX_TOKENS='128000', SUMMARY_OUTPUT_MAX_TOKENS='8000',
        SUMMARY_TIMEOUT_SECONDS='300', SUMMARY_PROMPT_MAX_TOKENS='2000', SUMMARY_READ_TIMEOUT_SECONDS='0.1',
        LLM_FAST_POST_TURN_MODE='history_only', MEMORY_EXTRACTOR_URL=f'http://127.0.0.1:{PORT}/api/v1/llm')
    admin = await asyncpg.connect(base)
    try:
        if not iso['created']:
            await admin.execute(f'CREATE SCHEMA "{schema}"')
            iso['created'] = True
            dump(iso_path, iso)
            from services.llm.memory import _SCHEMA_SQL
            conn = await asyncpg.connect(isolated)
            try:
                assert await conn.fetchval("SELECT to_regclass('navi_messages')") is None
                crypto = await admin.fetchval("SELECT n.nspname FROM pg_extension e JOIN pg_namespace n ON n.oid=e.extnamespace WHERE e.extname='pgcrypto'")
                if crypto:
                    quoted = '"' + crypto.replace('"', '""') + '"'
                    await conn.execute(f"CREATE FUNCTION digest(text,text) RETURNS bytea LANGUAGE SQL IMMUTABLE STRICT AS 'SELECT {quoted}.digest($1,$2)'")
                await conn.execute(_SCHEMA_SQL)
                for name in ('001_conversation_summary.sql', '002_summary_understanding.sql', '003_teaching_preferences.sql'):
                    await conn.execute((ROOT / 'deploy/migrations' / name).read_text(encoding='utf-8'))
                # Only age/grade bootstrapping is seeded; other facts arrive in dialogue.
                await conn.execute("INSERT INTO navi_user_profiles(user_id,profile_data) VALUES($1,$2::jsonb)",
                                   uid, json.dumps({'birth_date': '2012-06-15', 'grade': '初二'}))
                iso['seeded_fields'] = {'birth_date': '2012-06-15', 'grade': '初二', 'synthetic': True}
                dump(iso_path, iso)
            finally:
                await conn.close()
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', PORT))
        from services.llm import main as app_main
        from services.llm.summary.repository import decode
        runtime = app_main.agent_runtime
        path = out / 'report.json'
        report = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {
            'user_id': uid, 'persona': PERSONA, 'started_at': datetime.now(timezone.utc).isoformat(),
            'rounds': [], 'probes': [], 'summaries': [], 'attempts': [], 'prepared': [],
            'model_requests': [], 'summary_generations': [], 'summary_errors': [],
            'settings': runtime.summary_settings.model_dump(),
            'scope': '真实临时LLM HTTP服务、模型、审核、PostgreSQL隔离schema；同一用户三个连续会话。',
            'limitations': ['每轮采用history_only保存对话，不额外调用长期记忆抽取器；摘要发布可照常晋升记忆。',
                           '初始仅在隔离schema设置虚构生日和年级，不预填姓名、猫名、偏好或学习答案。',
                           '最终验证会独立保存为附加对话，不计入150轮基础对话。'],
        }
        assert report['user_id'] == uid

        def save():
            dump(path, report)

        original_prepare = runtime.prepare
        original_factory = app_main._deepseek_client
        original_generate = runtime.summary_worker.agent.generate
        labels = {}

        async def observe(request):
            request_id = ACTIVE_REQUEST.get()
            if not request_id or request.method != 'POST':
                return
            if request.extensions.get('teen_test_observed'):
                return
            request.extensions['teen_test_observed'] = True
            report['model_requests'].append({'request_id': request_id, 'label': labels.get(request_id),
                                             'body': json.loads(request.content)})

        def factory(timeout):
            client = original_factory(timeout)
            if observe not in client.event_hooks['request']:
                client.event_hooks['request'].append(observe)
            return client

        async def prepare(request):
            if request.user_id == uid:
                ACTIVE_REQUEST.set(request.request_id)
            turn = await original_prepare(request)
            if request.user_id == uid:
                context = turn.context
                report['prepared'].append({'request_id': request.request_id, 'label': labels.get(request.request_id),
                    'runtime': turn.metadata, 'messages': [m.model_dump(mode='json') for m in turn.request.messages],
                    'response_policy': context.tutor_strategy.get('response_policy', {}) if context else {},
                    'summary_entries': list(context.summary_context.entries) if context and context.summary_context else []})
            return turn

        async def generate(snapshot, request_id):
            target = max(t['completed_seq'] for t in snapshot['turns'])
            dump(out / f'summary-input-{target}.json', snapshot)
            try:
                candidate, model = await original_generate(snapshot, request_id)
                report['summary_generations'].append({'target': target, 'job_id': request_id, 'model': model,
                    'candidate': candidate.model_dump(mode='json')})
                save()
                return candidate, model
            except Exception as exc:
                report['summary_errors'].append({'target': target, 'type': type(exc).__name__,
                    'reason': str(exc)[:140] if isinstance(exc, ValueError) else type(exc).__name__})
                save()
                raise

        runtime.prepare, app_main._deepseek_client = prepare, factory
        runtime.summary_worker.agent.generate = generate
        server = uvicorn.Server(uvicorn.Config(app_main.app, host='127.0.0.1', port=PORT,
            log_level='error', access_log=False, timeout_graceful_shutdown=330))
        server_task = asyncio.create_task(server.serve())
        headers = {'X-Internal-API-Key': os.getenv('LLM_INTERNAL_API_KEY', '')}

        async def conversation(client, number, prompt, session, *, is_probe=False, expected=None):
            label = f'probe-{number}' if is_probe else f'round-{number}'
            previous = next((a for a in report['attempts'] if a['label'] == label), None)
            request_id = previous['request_id'] if previous else uuid4().hex
            labels[request_id] = label
            payload = {'request_id': request_id, 'trace_id': uuid4().hex, 'user_id': uid, 'session_id': session,
                'messages': [{'role': 'user', 'content': prompt}],
                'generation_config': {'agent_enabled': True, 'memory_mode': 'lightweight', 'answer_mode': 'fast'}}
            for attempt in range(1, 4):
                started = time.perf_counter()
                try:
                    response = await client.post('/api/v1/llm', json=payload)
                    response.raise_for_status()
                    body = response.json()
                    report['attempts'].append({'label': label, 'request_id': request_id, 'attempt': attempt,
                                              'response': body})
                    save()
                    if body.get('finish_reason') != 'stop' or not body.get('metadata', {}).get('persistence', {}).get('saved_turn'):
                        raise ValueError('turn_not_durably_completed')
                    health = (await client.get('/health')).json()
                    return {'number': number, 'label': label, 'request': payload, 'prompt': prompt,
                            'answer': body['text'], 'response': body, 'expected': expected,
                            'elapsed_ms': round(1000 * (time.perf_counter()-started), 2),
                            'persistence': health.get('runtime', {}).get('persistence', {})}
                except (httpx.HTTPError, ValueError) as exc:
                    report['attempts'].append({'label': label, 'request_id': request_id, 'attempt': attempt,
                                              'error_type': type(exc).__name__})
                    save()
                    if attempt == 3:
                        raise
                    await asyncio.sleep(2)
            raise AssertionError('unreachable')

        async def checkpoint(client, target):
            if any(s['target'] == target for s in report['summaries']):
                return
            async with asyncio.timeout(60):
                while True:
                    async with runtime.summary_repository.pool.acquire() as conn:
                        job_id = await conn.fetchval('SELECT job_id FROM navi_summary_jobs WHERE user_id=$1 AND target_completed_seq=$2 ORDER BY created_at DESC LIMIT 1', uid, target)
                    if job_id:
                        break
                    await asyncio.sleep(1)
            last = None
            async with asyncio.timeout(1150):
                while True:
                    response = await client.get(f'/api/v1/llm/memory/{uid}/summary/jobs/{job_id}')
                    response.raise_for_status()
                    job = response.json()
                    status = (job['status'], job['attempt_count'])
                    if status != last:
                        print('SUMMARY', target, *status, flush=True)
                        last = status
                    if job['status'] in ('succeeded', 'failed', 'obsolete', 'cancelled'):
                        break
                    await asyncio.sleep(2)
            response = await client.get(f'/api/v1/llm/memory/{uid}/summary')
            response.raise_for_status()
            published = response.json()
            report['summaries'].append({'target': target, 'job': job, 'response': published,
                                        'automatic_trigger': True})
            save()
            if job['status'] != 'succeeded':
                # Prevent each subsequent turn from starting a new failing job.
                raise RuntimeError('summary_checkpoint_failed')
            print('COVERAGE', target, published['summary']['coverage'], flush=True)

        try:
            async with asyncio.timeout(90):
                while not server.started:
                    if server_task.done():
                        await server_task
                        raise RuntimeError('temporary_service_exited')
                    await asyncio.sleep(.2)
            async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{PORT}', headers=headers,
                                         timeout=180, trust_env=False) as client:
                report['health_before'] = (await client.get('/health')).json()
                profile = await runtime.memory_store.get_profile(uid)
                from services.llm.age_policy import audience_profile_from_profile
                report['initial_audience'] = audience_profile_from_profile(profile)
                assert report['initial_audience']['age_years'] == 14
                assert report['initial_audience']['age_stage'] == 'S3'
                save()
                for number, prompt in enumerate(PROMPTS, 1):
                    if number > len(report['rounds']):
                        session = f'{uid}-session-{(number-1)//50+1}'
                        result = await conversation(client, number, prompt, session)
                        report['rounds'].append(result)
                        save()
                        print(f"ROUND {number}/150 chars={len(result['answer'])}", flush=True)
                    if number % 50 == 0:
                        await checkpoint(client, number)
                for number, (prompt, expected) in enumerate(FINAL_PROBES, 1):
                    if number <= len(report['probes']):
                        continue
                    result = await conversation(client, number, prompt, f'{uid}-validation-{number}',
                                                is_probe=True, expected=expected)
                    report['probes'].append(result)
                    save()
                    print(f"PROBE {number}/{len(FINAL_PROBES)} chars={len(result['answer'])}", flush=True)
                report['stage'] = 'completed'
                report['completed_at'] = datetime.now(timezone.utc).isoformat()
                report['health_after'] = (await client.get('/health')).json()
        finally:
            save()
            server.should_exit = True
            await server_task
            runtime.prepare, app_main._deepseek_client = original_prepare, original_factory
            runtime.summary_worker.agent.generate = original_generate
            conn = await asyncpg.connect(isolated)
            try:
                database = {}
                for name in ('navi_messages', 'navi_memories', 'navi_user_profiles', 'navi_learning_concepts',
                             'navi_user_memory_state', 'navi_conversation_turns', 'navi_conversation_summaries',
                             'navi_summary_jobs', 'navi_summary_evidence'):
                    database[name] = [dict(r) for r in await conn.fetch(f'SELECT * FROM {name} WHERE user_id=$1', uid)]
                    for row in database[name]:
                        for key in ('document', 'profile_data', 'input_snapshot'):
                            if row.get(key) is not None:
                                row[key] = decode(row[key])
                report['database'] = database
                save()
            finally:
                await conn.close()
        if report.get('stage') == 'completed':
            from services.llm.memory import MemoryStore
            memory = MemoryStore(database_url=isolated)
            try:
                await memory.initialize()
                await memory.clear_user(user_id=uid)
            finally:
                await memory.close()
            await admin.execute(f'DROP SCHEMA "{schema}" CASCADE')
            iso['cleaned'] = True
            iso['schema_absent_verified'] = not bool(await admin.fetchval('SELECT 1 FROM pg_namespace WHERE nspname=$1', schema))
            dump(iso_path, iso)
            print('COMPLETE 150 rounds, 8 probes, isolated data cleaned', flush=True)
    finally:
        await admin.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.output.resolve()))
