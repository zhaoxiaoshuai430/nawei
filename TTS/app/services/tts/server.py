#!/usr/bin/env python3
"""Standalone TTS API service (no ASR / LLM). Default port 19000."""
from __future__ import annotations

import argparse
import os
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.config import WEB_STATIC, WEB_TEMPLATES, setup_environment
from app.services.tts import state
from app.services.tts.engine import load_model_into_cache
from app.services.tts.emotion_refs import load_emotion_refs
from app.services.tts.presets import load_preset_refs
from app.web.tts_routes import router as tts_router

try:
    from faster_qwen3_tts import FasterQwen3TTS  # noqa: F401
except ImportError:
    print("Error: faster_qwen3_tts not found.")
    print("Install with:  pip install faster-qwen3-tts")
    sys.exit(1)

_app: FastAPI | None = None
_initialized = False

DEFAULT_MODEL = os.environ.get("TTS_DEFAULT_MODEL", "voice_lab/nana").strip() or "voice_lab/nana"
DEFAULT_PORT = int(os.environ.get("TTS_PORT", "19000"))


def create_app() -> FastAPI:
    application = FastAPI(
        title="Qwen3 TTS API",
        description="Standalone text-to-speech service (CustomVoice, voice clone, voice design).",
        version="1.0.0",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(tts_router)
    if WEB_STATIC.is_dir():
        application.mount("/static", StaticFiles(directory=str(WEB_STATIC)), name="static")

    demo_html = WEB_TEMPLATES / "tts_demo.html"

    @application.get("/demo", response_class=HTMLResponse)
    async def tts_demo_page():
        if demo_html.is_file():
            return demo_html.read_text(encoding="utf-8")
        return HTMLResponse("<p>tts_demo.html not found</p>", status_code=404)

    @application.get("/")
    async def root():
        return {
            "service": "tts",
            "demo": "/demo",
            "docs": "/docs",
            "health": "/api/v1/health",
            "status": "/api/v1/status",
        }

    return application


def get_app() -> FastAPI:
    global _app
    if _app is None:
        _app = create_app()
    return _app


def initialize_tts_service() -> None:
    global _initialized
    if _initialized:
        return
    setup_environment()
    print(f"HF_HOME: {os.environ.get('HF_HOME')}")
    load_preset_refs()
    load_emotion_refs()
    _initialized = True


def run_tts_service(argv: list[str] | None = None) -> None:
    initialize_tts_service()
    app = get_app()

    parser = argparse.ArgumentParser(description="Standalone Qwen3 TTS API")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model to preload at startup (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="Skip model loading at startup (load via POST /api/v1/models/load)",
    )
    args = parser.parse_args(argv)

    if not args.no_preload:
        print(f"Loading model: {args.model}")
        load_model_into_cache(args.model)
        print("TTS model ready.")
    print(f"Ready. Demo: http://localhost:{args.port}/demo  API: http://localhost:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def main() -> None:
    run_tts_service()


if __name__ == "__main__":
    main()
