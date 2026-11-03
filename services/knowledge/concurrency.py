"""Run dependent retrieval branches without leaving work behind on failure."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any


async def gather_cancel_on_error(*operations: Awaitable[Any]) -> list[Any]:
    """Preserve result order and the original error after all siblings settle."""
    tasks = [asyncio.ensure_future(operation) for operation in operations]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
