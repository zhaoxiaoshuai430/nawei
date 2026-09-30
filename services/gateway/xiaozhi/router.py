import asyncio
import time
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse, Response

from services.gateway.auth import authenticate_user

from .auth import device_identity, provisioning_key
from .config import Settings
from .device_registry import DeviceRegistry
from .provisioning import ota_response
from .schemas import BindRequest, DeviceUpdate
from .session import SessionManager
from .websocket import serve

router = APIRouter()
sessions = SessionManager()
User = Annotated[str, Depends(authenticate_user)]


class RateLimiter:
    def __init__(self) -> None:
        self.buckets: dict[str, deque[float]] = defaultdict(deque)
        self.lock = Lock()

    def check(self, key: str, limit: int) -> None:
        now = time.monotonic()
        with self.lock:
            for stale in [name for name, values in self.buckets.items() if not values or values[-1] <= now - 60]:
                del self.buckets[stale]
            if key not in self.buckets and len(self.buckets) >= 10000:
                raise HTTPException(429, "Too many clients")
            bucket = self.buckets[key]
            while bucket and bucket[0] <= now - 60:
                bucket.popleft()
            if len(bucket) >= limit:
                raise HTTPException(429, "Rate limit exceeded", headers={"Retry-After": "60"})
            bucket.append(now)


limiter = RateLimiter()


@lru_cache(maxsize=4)
def _registry(path: Path) -> DeviceRegistry:
    return DeviceRegistry(path)


def settings() -> Settings:
    config = Settings.load()
    if not config.enabled:
        raise HTTPException(404, "Robot access disabled")
    return config


def registry() -> DeviceRegistry:
    return _registry(settings().database)


def _rate(request: Any, action: str, limit: int) -> None:
    host = request.client.host if request.client else "unknown"
    limiter.check(f"{action}:{host}", limit)


async def _provision(request: Request) -> dict[str, Any]:
    config = settings()
    _rate(request, "ota", 120)
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > 16384:
            raise HTTPException(413, "OTA payload too large")
    device, client = device_identity(request.headers)
    key = provisioning_key(request.headers, device, client, config.legacy_device_secret)
    return await asyncio.to_thread(registry().provision, device, client, key, config.activation_ttl)


@router.post("/xiaozhi/ota/")
async def ota(request: Request) -> JSONResponse:
    result = await _provision(request)
    return JSONResponse(ota_response(result, settings(), request_url=str(request.url)), headers={"Cache-Control": "no-store"})


@router.post("/xiaozhi/ota/activate")
async def activate(request: Request) -> Response:
    result = await _provision(request)
    return Response(status_code=200 if result["bound"] else 202, headers={"Cache-Control": "no-store"})


@router.post("/api/v1/devices/bind")
def bind(payload: BindRequest, request: Request, user: User) -> dict[str, Any]:
    _rate(request, "bind", 10)
    limiter.check(f"bind-user:{user}", 10)
    return registry().bind(payload.code, user, payload.name)


def _with_status(device: dict[str, Any]) -> dict[str, Any]:
    return {**device, "online": device["device_id"] in sessions.active}


@router.get("/api/v1/devices")
def list_devices(user: User) -> list[dict[str, Any]]:
    return [_with_status(device) for device in registry().list_devices(user)]


@router.get("/api/v1/devices/{device_id}")
def get_device(device_id: str, user: User) -> dict[str, Any]:
    return _with_status(registry().get(device_id, user))


@router.patch("/api/v1/devices/{device_id}")
async def update_device(device_id: str, payload: DeviceUpdate, user: User) -> dict[str, Any]:
    device = await asyncio.to_thread(registry().update, device_id, user, payload.model_dump(exclude_none=True))
    if not device["enabled"]:
        sessions.disconnect(device_id)
    return _with_status(device)


@router.delete("/api/v1/devices/{device_id}/binding", status_code=204)
async def unbind_device(device_id: str, user: User) -> Response:
    await asyncio.to_thread(registry().unbind, device_id, user)
    sessions.disconnect(device_id)
    return Response(status_code=204)


@router.websocket("/xiaozhi/v1/")
async def websocket_endpoint(websocket: WebSocket) -> None:
    try:
        config = settings()
        _rate(websocket, "ws", 30)
        await serve(websocket, registry(), sessions, config)
    except HTTPException:
        await websocket.close(code=4403)
