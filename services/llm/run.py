"""Run the LLM service on its documented port."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import uvicorn


HOST = os.getenv("LLM_HOST", "127.0.0.1")
PORT = int(os.getenv("LLM_PORT", "8002"))
WORKERS = int(os.getenv("LLM_WORKERS", "4"))
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    """Start the LLM FastAPI application."""

    uvicorn.run(
        "services.llm.main:app",
        host=HOST,
        port=PORT,
        reload=False,
        workers=WORKERS,
        loop="services.llm.event_loop:SharedSocketSelectorEventLoop" if sys.platform == "win32" and WORKERS > 1 else "auto",
        app_dir=str(PROJECT_ROOT),
        timeout_graceful_shutdown=45,
    )


if __name__ == "__main__":
    main()
