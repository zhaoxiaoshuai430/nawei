"""Fresh synthetic daily-life test using real HTTP, model, memory and summary."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import asyncpg
import httpx
import uvicorn
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
PERSONA = {
    "声明": "完全虚构的测试人物，不代表真实用户；通过用户消息逐步输入，没有直接伪造数据库记忆或模型回答。",
    "称呼": "林舟", "年龄": 29, "职业及城市": "杭州的设计师", "家庭": "独居，养猫团子",
    "自述性格": "慢热，偏爱安静，有时计划过满；不是心理诊断或量表结论",
    "沟通偏好": "温和、稍带幽默、不说教，先结论后步骤，日常建议简短",
    "生活偏好": "热豆浆、鸡蛋；清淡少辣；轻音乐；纸质书；周末喜欢安静公园",
    "近期目标": "每月一本薄书；睡觉时间从零点半逐步提前到十二点；散步每周三次每次二十分钟；三行日记；简单记账",
    "预算及边界": "本周零食预算六十元；下班后划清工作边界；少做临时决定",
}
PROMPTS = [
    "早呀，我叫林舟，29岁，在杭州做设计，平时一个人住，家里有只猫叫团子。刚起来有点迷糊，今天先陪我轻松聊聊吧。日常建议别太长，三四句话就好。",
    "团子刚才扒拉我的拖鞋，算是催我起床了。我性格有点慢热，早上喜欢安静一点。先洗漱还是先把早餐弄上？",
    "我早餐一般喝热豆浆配鸡蛋，不太爱甜面包。冰箱里还有一根玉米，今天这样凑一顿行吗？",
    "今天想把三份设计稿都弄完，但一看就有点满。我老有把计划排太紧的毛病，帮我收一收。",
    "那我先处理最急的封面，另外两份留个余地。桌上手机一直亮，我手又忍不住去拿。",
    "我把手机放远了，准备听点轻音乐开工。我不太喜欢歌词很多的歌，会跟着唱跑神。",
    "刚进入状态，同事就发来一个小修改。我有点烦，不过他也不是故意的，怎么把心思收回来？",
    "第一份稿子总算发出去了，有点小开心。别讲大道理，给我一句像朋友一样的鼓励就行。",
    "午饭想省事点。我口味偏清淡，不太能吃辣，附近有面馆和盖饭店，今天怎么选？",
    "吃完有点困，我午休只有二十分钟，不想一睡醒来更懵。现在做点什么好？",
    "下午那份稿子还差一截，但今天没全做完也不是世界末日吧。我想练习别总责怪自己。",
    "下班了，团子在门边等我，突然觉得累也值得了。先喂猫再吃自己的饭，还是先坐五分钟？",
    "我最近总是零点半才睡，想先慢慢挪到十二点，不要求一下变成早睡模范。第一周怎么来？",
    "最难的是睡前短视频，一刷就停不住。我不想靠骂自己解决，有没有温和一点的替代？",
    "明早九点半要开个小会，我今晚把衣服和电脑包准备好应该能省点心。还要留意什么？",
    "我喜欢房间清清爽爽，但一次收拾一小时会烦。今晚只整理餐桌，你觉得怎么开始最轻松？",
    "周末想去附近公园晃一圈。我不太爱拥挤的打卡点，就想安静走走，看看树和水。",
    "要是周末下雨，我就不勉强出门了。你帮我想一个在家也有休息感的小安排吧。",
    "运动这事我先不定太狠：每周散步三次，每次二十分钟，能做到就很好。怎么让它更容易开始？",
    "今天没走成二十分钟，只在楼下转了十分钟。我是不是也可以算自己动过了？",
    "我总等到口渴才喝水，杯子经常被稿纸挡住。给我一个能马上做的小办法。",
    "下午想吃零食，这周我给自己留了六十块零食预算，不想天天奶茶。买什么比较踏实？",
    "我想开始记账，但一想到分类就头大。如果先只记每天花了多少，会不会太粗糙？",
    "刚看见一个好看的杯子差点下单，家里明明有三个了。你温柔地拦我一下。",
    "到家脑子还在改稿，身体已经坐沙发了，灵魂还在公司。我想有个下班的切换动作。",
    "我挺喜欢你刚才这种轻松的说法。以后提醒我时温和一点、带一点幽默就好，不要说教，也别动不动给我打鸡血。",
    "妈妈打电话希望我周末多出去认识人，可我更想休息。我怎么说既让她放心，也保留自己的安排？",
    "朋友今晚临时约饭，我其实想在家陪团子。拒绝时总怕他觉得我不重视他，帮我想句话。",
    "拿起书十分钟就走神了。我想从每天十分钟开始读，不追求一口气读很多。走神时怎么办？",
    "我给自己定的阅读目标是每月读完一本薄书，先求稳定，不要每周一本那么赶。怎么安排更轻松？",
    "书里遇到看不懂的词我就想查，结果查着查着忘了在读什么。要不要先做个记号？",
    "我还是喜欢纸质书，翻页和摸纸的感觉很舒服。不过通勤包有点重，带书还是用手机读好呢？",
    "今天情绪一般，不是什么大事，就是雨天有点没精神。不想被安排一堆任务，你陪我缓一缓。",
    "明天的事情还没发生，我已经开始担心了。今晚我只想做一件能让心里松一点的事。",
    "我发现连休息都容易被我变成待办清单。能不能给我一个不需要打卡的休息办法？",
    "下周可能比较忙，我想提前准备，但不想把这个周末也搭进去。只准备哪一件最值？",
    "我想试试每天写三行日记，不写长篇。今天有点不知道从哪里写起，给我一个小框架。",
    "今天主动关掉工作聊天去吃饭了，虽然只是小事，但我挺满意。这个可以写进日记吗？",
    "如果某天计划全乱了，我想只保留喂团子和按时上床，其他事第二天再说。这样会不会太随意？",
    "再告诉你个聊天偏好：我喜欢先听结论，再看一两个步骤。日常问题尽量简短，不用每次都在结尾追问。",
    "画稿坐久了肩颈有点紧，没有受伤，就是僵。现在离开椅子活动一下，有什么轻一点的动作？",
    "周末早餐想换点花样，但我不想刷好多锅。鸡蛋、玉米、豆浆还有生菜，十分钟能弄什么？",
    "昨晚提前放好钥匙和电脑包，今天早上果然轻松了。还有什么小东西适合固定放一个地方？",
    "翻抽屉看到一堆用不上的数据线，又觉得扔了可惜。我只想整理这一个抽屉，怎么判断留什么？",
    "下个月可能出去住一晚，我最容易忘充电器。帮我想个出门前一分钟的检查顺序，不要太长。",
    "如果以后我说计划被打乱，你先帮我找最重要的一件事，别马上给我补一张更满的表，好吗？",
    "晚上又收到工作消息，但不是急事。我想慢慢立个下班边界，怎么回复比较自然？",
    "我想把散步、读几页纸质书、早点上床连成一个舒服的晚上，别像军训那样卡点。帮我试着串一下。",
    "聊了这些天，你觉得我自己说过的性格和交流偏好是什么？只说有根据的，不要给我贴心理标签。",
    "今天先到这里。请帮我把明天最值得保留的一件小事挑出来，其他不加码，我想轻松一点收尾。",
]
QUERY_51 = "周五下班了，按你记得的我，给我一个不赶的晚间安排。顺便说说我给自己定的阅读频率、家里要照顾的小伙伴叫什么，以及我喜欢你怎样跟我说话；没记住的就直说，不要猜。"


def dump(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


async def run(out: Path):
    load_dotenv(ROOT / ".env")
    os.environ.setdefault("DATABASE_URL", "postgresql://nawei:nawei@127.0.0.1:5432/nawei")
    os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6379/0")
    # Process-local settings only; .env and the regular service are untouched.
    os.environ.update(SUMMARY_ENABLED="true", SUMMARY_APPLY_TO_PROMPT="false", SUMMARY_TRIGGER_TURNS="50",
        SUMMARY_INPUT_MAX_TOKENS="128000", SUMMARY_OUTPUT_MAX_TOKENS="8000", SUMMARY_TIMEOUT_SECONDS="300",
        LLM_FAST_POST_TURN_MODE="full", LLM_WORKERS="1")
    uid = out.name
    session = uid + "-main"
    out.mkdir(parents=True, exist_ok=True)
    print("OUTPUT", str(out), flush=True)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 8015))
    from services.llm import main as app_main
    from services.llm.summary.repository import decode
    runtime = app_main.agent_runtime
    original_prepare = runtime.prepare
    original_factory = app_main._deepseek_client
    original_generate = runtime.summary_worker.agent.generate
    report_path = out / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {
        "user_id": uid, "session_id": session, "synthetic_persona": PERSONA, "prompts": PROMPTS,
        "rounds": [], "attempts": [], "prepared": [], "outbound_model_requests": [], "summary_generations": [],
        "scope": "临时 LLM HTTP 服务 8015；真实模型/数据库/内容审核；合成日常用户；非麦克风或 Gateway 测试",
        "test_config": {"summary_trigger_turns": 50, "summary_input_max_tokens": 128000,
                        "summary_timeout_seconds": 300, "fast_post_turn_mode": "full"}}
    current_round = 0

    def save():
        dump(report_path, report)

    async def observe_request(request):
        if request.method != "POST":
            return
        body = json.loads(request.content)
        if current_round == 51:
            report["outbound_model_requests"].append({"round": 51, "body": body})

    def client_factory(timeout):
        client = original_factory(timeout)
        if observe_request not in client.event_hooks["request"]:
            client.event_hooks["request"].append(observe_request)
        return client

    async def prepare(request):
        turn = await original_prepare(request)
        if request.user_id == uid:
            entry = {"round": current_round, "request_id": request.request_id, "runtime": turn.metadata,
                     "summary_entries": list(turn.context.summary_context.entries)
                     if turn.context and turn.context.summary_context else []}
            if current_round == 51:
                entry["prepared_messages"] = [m.model_dump(mode="json") for m in turn.request.messages]
            report["prepared"].append(entry)
        return turn

    async def generate(snapshot, request_id):
        if snapshot["turns"] and snapshot["turns"][0]["session_id"].startswith(uid):
            dump(out / "summary-input-snapshot.json", snapshot)
        candidate, model = await original_generate(snapshot, request_id)
        if snapshot["turns"] and snapshot["turns"][0]["session_id"].startswith(uid):
            report["summary_generations"].append({"job_id": request_id, "model": model,
                "candidate": candidate.model_dump(mode="json")})
            save()
        return candidate, model

    runtime.prepare = prepare
    app_main._deepseek_client = client_factory
    runtime.summary_worker.agent.generate = generate
    server = uvicorn.Server(uvicorn.Config(app_main.app, host="127.0.0.1", port=8015,
        log_level="warning", access_log=False, timeout_graceful_shutdown=330))
    server_task = asyncio.create_task(server.serve())
    headers = {"X-Internal-API-Key": os.getenv("LLM_INTERNAL_API_KEY", "")}

    async def wait_persistence(client):
        async with asyncio.timeout(180):
            while True:
                health = (await client.get("/health")).json()
                state = health["runtime"]["persistence"]
                if not state["pending"]:
                    return state
                await asyncio.sleep(1)

    async def conversation(client, number, prompt, session_id):
        payload = {"request_id": uuid4().hex, "trace_id": uuid4().hex, "user_id": uid,
            "session_id": session_id, "messages": [{"role": "user", "content": prompt}],
            "generation_config": {"agent_enabled": True, "memory_mode": "lightweight", "answer_mode": "fast"}}
        for attempt in range(1, 4):
            start = time.perf_counter()
            try:
                response = await client.post("/api/v1/llm", json=payload)
                response.raise_for_status()
                body = response.json()
            except (httpx.HTTPError, ValueError) as exc:
                report["attempts"].append({"round": number, "attempt": attempt, "error_type": type(exc).__name__})
                save()
                if attempt == 3:
                    raise
                continue
            elapsed = round((time.perf_counter() - start) * 1000, 2)
            report["attempts"].append({"round": number, "attempt": attempt, "response": body})
            save()
            if body.get("finish_reason") == "stop" and body.get("metadata", {}).get("persistence", {}).get("saved_turn"):
                result = {"round": number, "request": payload, "prompt": prompt, "answer": body["text"],
                          "response": body, "elapsed_ms": elapsed}
                result["persistence_after_extraction"] = await wait_persistence(client)
                return result
            if attempt == 3:
                raise RuntimeError(f"Round {number} did not durably complete")
        raise AssertionError("unreachable")

    try:
        async with asyncio.timeout(90):
            while not server.started:
                if server_task.done():
                    await server_task
                    raise RuntimeError("temporary service exited")
                await asyncio.sleep(.2)
        async with httpx.AsyncClient(base_url="http://127.0.0.1:8015", headers=headers,
                                     timeout=150, trust_env=False) as client:
            report["health_before"] = (await client.get("/health")).json()
            for number, prompt in enumerate(PROMPTS, 1):
                if number <= len(report["rounds"]):
                    continue
                current_round = number
                result = await conversation(client, number, prompt, session)
                report["rounds"].append(result)
                save()
                print(f"ROUND {number}/50 saved; chars={len(result['answer'])}; extraction={result['persistence_after_extraction']}", flush=True)

            refresh = await client.post(f"/api/v1/llm/memory/{uid}/summary/refresh")
            refresh.raise_for_status()
            job_id = refresh.json()["job_id"]
            report["summary_job_id"] = job_id
            last_state = None
            async with asyncio.timeout(1100):
                while True:
                    response = await client.get(f"/api/v1/llm/memory/{uid}/summary/jobs/{job_id}")
                    response.raise_for_status()
                    job = response.json()
                    report["summary_job"] = job
                    save()
                    if (job["status"], job["attempt_count"]) != last_state:
                        print("SUMMARY", job, flush=True)
                        last_state = job["status"], job["attempt_count"]
                    if job["status"] == "succeeded":
                        break
                    if job["status"] in {"failed", "obsolete", "cancelled"}:
                        raise RuntimeError(f"Summary failed: {job}")
                    await asyncio.sleep(3)
            response = await client.get(f"/api/v1/llm/memory/{uid}/summary")
            response.raise_for_status()
            report["summary"] = response.json()
            coverage = report["summary"]["summary"]["coverage"]
            assert coverage["actual_input_turn_count"] == 50 and coverage["omitted_turn_count"] == 0
            # Stop here for an explicit quality review by the operator. Rerun with
            # --reviewed after inspecting the persisted candidate and evidence.
            if not args.reviewed:
                report["stage"] = "summary_ready_for_review"
                print("READY_FOR_REVIEW", out, flush=True)
                save()
            else:
                runtime.summary_settings.apply_to_prompt = True
                runtime.summary_repository.settings.apply_to_prompt = True
                response = await client.patch(f"/api/v1/llm/memory/{uid}/summary/settings", json={"apply_to_prompt": True})
                response.raise_for_status()
                current_round = 51
                if not report.get("round_51"):
                    report["round_51"] = await conversation(client, 51, QUERY_51, uid + "-new-session-51")
                report["stage"] = "completed"
                save()
                print("ROUND 51", report["round_51"]["answer"], flush=True)
            report["health_after"] = (await client.get("/health")).json()
    finally:
        save()
        server.should_exit = True
        await server_task
        runtime.prepare = original_prepare
        app_main._deepseek_client = original_factory
        runtime.summary_worker.agent.generate = original_generate
        conn = await asyncpg.connect(os.environ["DATABASE_URL"])
        try:
            tables = {}
            for name in ("navi_messages", "navi_memories", "navi_user_profiles", "navi_learning_concepts",
                         "navi_user_memory_state", "navi_conversation_turns", "navi_conversation_summaries", "navi_summary_jobs"):
                rows = await conn.fetch(f"SELECT * FROM {name} WHERE user_id=$1", uid)
                tables[name] = [dict(row) for row in rows]
                for row in tables[name]:
                    for key in ("document", "profile_data", "input_snapshot"):
                        if row.get(key) is not None:
                            row[key] = decode(row[key])
            evidence = await conn.fetch("""SELECT e.*,t.completed_seq,m.role,m.content FROM navi_summary_evidence e
                JOIN navi_conversation_turns t ON t.turn_id=e.turn_id AND t.user_id=e.user_id
                JOIN navi_messages m ON m.id=e.message_id AND m.user_id=e.user_id
                WHERE e.user_id=$1 ORDER BY t.completed_seq,e.claim_id""", uid)
            tables["navi_summary_evidence"] = [dict(row) for row in evidence]
            report["database"] = tables
            if len(report["rounds"]) == 50 and not report.get("round_51"):
                report["database_before_51"] = tables
                dump(out / "database-before-51.json", tables)
            save()
        finally:
            await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reviewed", action="store_true")
    args = parser.parse_args()
    output = args.output or ROOT / ".runtime" / ("daily-natural-50-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    asyncio.run(run(output))
