"""Run the LLM service on its documented port."""

from __future__ import annotations

import uvicorn


HOST = "0.0.0.0"
PORT = 8002


def main() -> None:
    """Start the LLM FastAPI application."""

    uvicorn.run("services.llm_service.main:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()
