import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .config import Settings


def ota_response(result: dict[str, Any], settings: Settings, *, request_url: str = "") -> dict[str, Any]:
    body: dict[str, Any] = {"server_time": {"timestamp": int(time.time() * 1000), "timezone_offset": 0}}
    if result["bound"]:
        url = settings.websocket_url
        if url == "auto":
            origin = urlsplit(request_url)
            if origin.scheme not in {"http", "https"} or not origin.hostname or origin.username or origin.password:
                raise ValueError("Auto robot discovery requires a valid HTTP request origin")
            url = urlunsplit(("wss" if origin.scheme == "https" else "ws", origin.netloc, "/xiaozhi/v1/", "", ""))
        body["websocket"] = {"url": url, "token": result["token"], "version": 1}
    else:
        body["activation"] = {
            "code": result["code"], "message": "Nawei device activation",
            "expires_at": result["expires_at"], "timeout_ms": 3000,
        }
    return body
