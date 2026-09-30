import hashlib
import hmac
import re

from fastapi import HTTPException


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def device_identity(headers: object) -> tuple[str, str]:
    device = headers.get("device-id", "").lower()  # type: ignore[attr-defined]
    client = headers.get("client-id", "")  # type: ignore[attr-defined]
    if not re.fullmatch(r"[0-9a-f]{2}(?::[0-9a-f]{2}){5}", device):
        raise HTTPException(400, "Invalid Device-Id")
    if not re.fullmatch(r"[A-Za-z0-9_-]{16,80}", client):
        raise HTTPException(400, "Invalid Client-Id")
    return device, client


def provisioning_key(headers: object, device: str = "", client: str = "", legacy_secret: str = "") -> str:
    key = headers.get("x-device-key", "")  # type: ignore[attr-defined]
    if not key and legacy_secret:
        key = hmac.new(legacy_secret.encode(), f"{device}:{client}".encode(), "sha256").hexdigest()
    if not re.fullmatch(r"[A-Za-z0-9_-]{32,128}", key):
        raise HTTPException(401, "Device initialization key required")
    return key
