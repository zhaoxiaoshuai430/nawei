"""Compare ASR languages against explicit transcripts without rewriting recognition output."""

import argparse
import asyncio
import base64
import io
import json
import time
import unicodedata
import uuid
import wave
from pathlib import Path

import httpx


def normalized_text(text):
    return "".join(character.lower() for character in unicodedata.normalize("NFKC", text) if character.isalnum())


def character_errors(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for position, expected in enumerate(reference, 1):
        current = [position]
        for index, actual in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[index] + 1, previous[index - 1] + (expected != actual)))
        previous = current
    return previous[-1]


async def evaluate(args):
    args.output.mkdir(parents=True, exist_ok=True)
    user = "asr-evaluation-" + uuid.uuid4().hex[:12]
    results = {"synthetic": args.manifest is None, "phase": args.phase, "cases": []}
    if args.manifest:
        cases = json.loads(args.manifest.read_text(encoding="utf-8"))
        for case in cases:
            case["path"] = str((args.manifest.parent / case["path"]).resolve())
    else:
        cases = [
            {"id": "brief", "text": "一加一等于几，请只回答结果。", "reference": "1加1等于几，请只回答结果。"},
            {"id": "name", "text": "我叫小明，我喜欢蓝色。", "reference": "我叫小明，我喜欢蓝色。"},
            {"id": "negation", "text": "不要开灯，请关闭音乐。", "reference": "不要开灯，请关闭音乐。"},
            {"id": "concept", "text": "三角函数与惯性有什么区别？", "reference": "三角函数与惯性有什么区别？"},
        ]
        args.audio_dir.mkdir(parents=True, exist_ok=True)
        for case in cases:
            case["path"] = str(args.audio_dir / (case["id"] + ".wav"))
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000", headers={"X-User-ID": user}, timeout=180, trust_env=False) as client:
        for case in cases:
            path = Path(case["path"])
            if not path.exists() and not args.manifest:
                response = await client.post("/api/v1/audio/speech", json={
                    "request_id": uuid.uuid4().hex, "trace_id": user, "session_id": user,
                    "text": case["text"], "voice": {"speaker_id": "liliya", "language": "Chinese"},
                    "audio_config": {"format": "pcm16", "sample_rate": 16000, "channels": 1},
                })
                response.raise_for_status()
                audio = response.json()
                assert audio["format"] == "pcm16" and audio["sample_rate"] == 16000
                with wave.open(str(path), "wb") as output:
                    output.setnchannels(1)
                    output.setsampwidth(2)
                    output.setframerate(16000)
                    output.writeframes(base64.b64decode(audio["audio_base64"], validate=True))
            content = path.read_bytes()
            with wave.open(io.BytesIO(content), "rb") as audio:
                rate, channels = audio.getframerate(), audio.getnchannels()
            for language in ["auto", "zh"]:
                started = time.perf_counter()
                response = await client.post("/api/v1/audio/transcriptions/base64", json={
                    "request_id": uuid.uuid4().hex, "trace_id": user, "session_id": user,
                    "audio_base64": base64.b64encode(content).decode(), "format": "wav", "sample_rate": rate,
                    "channels": channels, "language": language, "metadata": {"use_itn": True},
                })
                response.raise_for_status()
                hypothesis = response.json()["text"]
                reference = normalized_text(case["reference"])
                assert reference, "Reference transcript must not be empty"
                record = {"id": case["id"], "language": language, "reference": case["reference"],
                          "hypothesis": hypothesis, "errors": character_errors(reference, normalized_text(hypothesis)),
                          "reference_characters": len(reference), "elapsed_ms": round((time.perf_counter() - started) * 1000)}
                results["cases"].append(record)
                (args.output / (args.phase + ".json")).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
                print(json.dumps(record, ensure_ascii=True), flush=True)
    results["summary"] = {}
    for language in ["auto", "zh"]:
        records = [record for record in results["cases"] if record["language"] == language]
        results["summary"][language] = {
            "cer": sum(record["errors"] for record in records) / sum(record["reference_characters"] for record in records),
            "mean_ms": sum(record["elapsed_ms"] for record in records) / len(records),
        }
    (args.output / (args.phase + ".json")).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results["summary"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--audio-dir", type=Path, default=Path(".runtime/asr-evaluation/audio"))
    parser.add_argument("--output", type=Path, default=Path(".runtime/asr-evaluation"))
    parser.add_argument("--phase", choices=["baseline", "optimized"], default="baseline")
    asyncio.run(evaluate(parser.parse_args()))
