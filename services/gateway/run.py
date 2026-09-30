"""Run the Gateway service on its documented port."""

from __future__ import annotations

import os

import uvicorn
from dotenv import load_dotenv
from pathlib import Path


HOST = "0.0.0.0"
PORT = 8000


def main() -> None:
    """Start the Gateway FastAPI application."""

    load_dotenv(Path(__file__).resolve().parents[2] / ".env.robot", override=False)
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    uvicorn.run("services.gateway.main:app", host=os.getenv("GATEWAY_HOST", HOST),
                port=int(os.getenv("GATEWAY_PORT", str(PORT))), reload=False,
                ws_max_size=1024 * 1024, ws_max_queue=32, ws_ping_interval=25, ws_ping_timeout=25,
                forwarded_allow_ips=os.getenv("GATEWAY_FORWARDED_ALLOW_IPS", "127.0.0.1"))


if __name__ == "__main__":
    main()
