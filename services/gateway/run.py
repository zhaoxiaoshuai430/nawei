"""Run the Gateway service on its documented port."""

from __future__ import annotations

import uvicorn


HOST = "0.0.0.0"
PORT = 8000


def main() -> None:
    """Start the Gateway FastAPI application."""

    uvicorn.run("services.gateway.main:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()
