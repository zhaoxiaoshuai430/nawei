"""Run the ASR service on its documented port."""

from __future__ import annotations

import uvicorn


HOST = "0.0.0.0"
PORT = 8001


def main() -> None:
    """Start the ASR FastAPI application."""

    uvicorn.run("services.asr_service.main:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()
