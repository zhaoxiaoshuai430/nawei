"""Run production Gateway health and authenticated text-chain checks."""

from __future__ import annotations

import argparse
import json
import ssl
import time
import urllib.error
import urllib.request
import uuid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--token")
    parser.add_argument("--user-id")
    parser.add_argument("--insecure", action="store_true")
    args = parser.parse_args()

    context = ssl._create_unverified_context() if args.insecure else ssl.create_default_context()
    base_url = args.base_url.rstrip("/")
    health = _request_json(f"{base_url}/health", context=context)
    if health.get("status") != "ok":
        raise SystemExit(f"Gateway health check failed: {health}")
    print("Gateway health: ok")

    status = _request_json(f"{base_url}/api/v1/status", context=context)
    services = status.get("services")
    if not isinstance(services, dict):
        raise SystemExit(f"Gateway status response has no services object: {status}")
    unavailable = [
        str(name)
        for name, detail in services.items()
        if not isinstance(detail, dict) or not detail.get("ready")
    ]
    if unavailable:
        raise SystemExit(f"Dependencies are not ready: {', '.join(unavailable)}")
    print("Dependency status: all ready")

    if not args.token and not args.user_id:
        print("Authenticated text-chain test skipped; pass --token and --user-id to enable it.")
        return
    if not args.token or not args.user_id:
        raise SystemExit("--token and --user-id must be provided together")

    request_id = f"smoke-{uuid.uuid4()}"
    payload = {
        "request_id": request_id,
        "trace_id": request_id,
        "session_id": f"smoke-session-{int(time.time())}",
        "user_id": args.user_id,
        "message": {"role": "user", "content": "只回复：生产链路正常"},
        "stream": False,
    }
    body = _request_json(
        f"{base_url}/api/v1/text/completions",
        method="POST",
        payload=payload,
        headers={"Authorization": f"Bearer {args.token}"},
        context=context,
        timeout=180,
    )
    if not str(body.get("text", "")).strip():
        raise SystemExit(f"Text-chain response is empty: {body}")
    print("Authenticated Gateway/LLM text chain: ok")


def _request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, object] | None = None,
    headers: dict[str, str] | None = None,
    context: ssl.SSLContext,
    timeout: int = 30,
) -> dict[str, object]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request_headers = {"Accept": "application/json", **(headers or {})}
    if body is not None:
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=request_headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
            decoded = json.loads(response.read().decode("utf-8"))
            if not isinstance(decoded, dict):
                raise SystemExit(f"{method} {url} returned a non-object JSON response")
            return decoded
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"{method} {url} failed with HTTP {exc.code}: {detail}") from exc


if __name__ == "__main__":
    main()
