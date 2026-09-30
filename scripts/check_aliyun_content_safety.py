"""Probe Aliyun text moderation without printing credentials."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".runtime" / "aliyun-safety-probe-deps"))

from dotenv import dotenv_values
from alibabacloud_green20220302.client import Client
from alibabacloud_green20220302 import models
from alibabacloud_tea_openapi.models import Config
from alibabacloud_tea_util.models import RuntimeOptions


def main() -> int:
    settings = dotenv_values(ROOT / ".env")
    credentials = [settings.get(name) or "" for name in (
        "ALIBABA_CLOUD_ACCESS_KEY_ID", "ALIBABA_CLOUD_ACCESS_KEY_SECRET"
    )]
    if not all(credentials):
        print(json.dumps({"error": "Missing credentials"}))
        return 1

    def emit(value: dict) -> None:
        output = json.dumps(value, ensure_ascii=True, default=str)
        for credential in credentials:
            output = output.replace(credential, "[REDACTED]")
        print(output, flush=True)

    config = Config(
        access_key_id=credentials[0], access_key_secret=credentials[1],
        region_id=settings.get("ALIYUN_GREEN_REGION_ID") or "cn-shanghai",
        endpoint=settings.get("ALIYUN_GREEN_ENDPOINT") or "green-cip.cn-shanghai.aliyuncs.com",
        connect_timeout=10000, read_timeout=15000,
    )
    client = Client(config)
    runtime = RuntimeOptions(autoretry=False, connect_timeout=10000, read_timeout=15000)
    samples = [
        ("benign", "\u4f60\u597d\uff0c\u8bf7\u4ecb\u7ecd\u4e00\u4e0b\u4e09\u89d2\u51fd\u6570\u3002"),
        ("insult", "\u4f60\u662f\u4e2a\u50bb\u903c\uff0c\u6eda\u86cb\u3002"),
    ]
    for name, content in samples:
        request = models.TextModerationRequest(
            service="chat_detection",
            service_parameters=json.dumps({"content": content}, ensure_ascii=True),
        )
        try:
            response = client.text_moderation_with_options(request, runtime)
            body = response.body.to_map()
            emit({"sample": name, "service": "chat_detection", "http_status": response.status_code, "body": body})
            if str(body.get("Code")) != "200":
                return 1
        except Exception as exc:
            data = getattr(exc, "data", None)
            emit({
                "sample": name, "error_type": type(exc).__name__,
                "code": getattr(exc, "code", None),
                "message": getattr(exc, "message", None) or str(exc),
                "request_id": data.get("RequestId") if isinstance(data, dict) else None,
            })
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
