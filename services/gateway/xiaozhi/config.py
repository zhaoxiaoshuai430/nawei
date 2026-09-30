from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Settings:
    enabled: bool
    database: Path
    websocket_url: str
    legacy_device_secret: str
    activation_ttl: int
    hello_timeout: float
    heartbeat_interval: float

    @classmethod
    def load(cls) -> Settings:
        enabled = os.getenv("XIAOZHI_ENABLED", "false").lower() in {"1", "true", "yes"}
        public = os.getenv("NAWEI_PUBLIC_ROBOT_URL", "http://127.0.0.1:8000").rstrip("/")
        default_url = "auto" if public == "auto" else public.replace("http", "ws", 1) + "/xiaozhi/v1/"
        websocket_url = os.getenv("XIAOZHI_PUBLIC_WEBSOCKET_URL", default_url)
        parsed = urlsplit(websocket_url)
        if websocket_url != "auto" and (parsed.scheme not in {"ws", "wss"} or not parsed.hostname or parsed.username or parsed.query):
            raise ValueError("Invalid XIAOZHI_PUBLIC_WEBSOCKET_URL")
        if enabled and os.getenv("NAWEI_ENV", "development") in {"prod", "production"}:
            if parsed.scheme != "wss" or os.getenv("GATEWAY_AUTH_MODE") != "jwt":
                raise ValueError("Public robot access requires WSS and Gateway JWT authentication")
        allow_legacy_devices = os.getenv("XIAOZHI_ALLOW_LEGACY_DEVICES", "false").lower() in {"1", "true", "yes"}
        legacy_device_secret = os.getenv("XIAOZHI_LEGACY_DEVICE_SECRET", "") if allow_legacy_devices else ""
        if allow_legacy_devices and len(legacy_device_secret) < 32:
            raise ValueError("XIAOZHI_LEGACY_DEVICE_SECRET must contain at least 32 characters")
        return cls(
            enabled=enabled,
            database=Path(os.getenv("XIAOZHI_DATABASE_PATH", ".runtime/devices.sqlite3")),
            websocket_url=websocket_url,
            legacy_device_secret=legacy_device_secret,
            activation_ttl=max(60, min(600, int(os.getenv("XIAOZHI_ACTIVATION_TTL_SECONDS", "600")))),
            hello_timeout=float(os.getenv("XIAOZHI_HELLO_TIMEOUT_SECONDS", "10")),
            heartbeat_interval=max(1, float(os.getenv("XIAOZHI_HEARTBEAT_INTERVAL_SECONDS", "25"))),
        )
