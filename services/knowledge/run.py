"""Run the knowledge service on its configured port."""

from __future__ import annotations

import uvicorn

from services.knowledge.config import load_settings


def main() -> None:
    settings = load_settings()
    uvicorn.run(
        "services.knowledge.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        workers=1,
    )


if __name__ == "__main__":
    main()
