"""Run the TTS service on its documented port."""

from __future__ import annotations

import uvicorn


HOST = "0.0.0.0"
PORT = 19000


def main() -> None:
    """Start the TTS FastAPI application."""

    uvicorn.run("services.tts_service.main:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()
