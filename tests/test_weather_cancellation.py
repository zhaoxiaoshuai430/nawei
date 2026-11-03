"""Provider races must finish losing requests before returning or closing HTTP."""

import asyncio

import pytest

from services.llm.tools.weather import OpenMeteoWeatherTool


@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
async def test_weather_race_awaits_provider_cleanup(monkeypatch, outcome):
    monkeypatch.setenv("WEATHER_PERSISTENT_CACHE_ENABLED", "false")
    monkeypatch.setenv("WEATHER_OPEN_METEO_RACE_ENABLED", "true")
    monkeypatch.setattr("services.llm.tools.weather._amap_api_key", lambda: "")
    monkeypatch.setattr("services.llm.tools.weather._cma_station_id", lambda city: None)
    started, cleanup_started, release_cleanup = asyncio.Event(), asyncio.Event(), asyncio.Event()
    finished = asyncio.Event()
    tool = OpenMeteoWeatherTool()
    result = object()

    async def slow(**kwargs):
        try:
            started.set()
            if outcome == "failure":
                raise RuntimeError("provider failed")
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await release_cleanup.wait()
            finished.set()

    async def fast(**kwargs):
        await started.wait()
        if outcome == "failure":
            raise RuntimeError("provider failed")
        if outcome == "cancel":
            await asyncio.Event().wait()
        return result

    monkeypatch.setattr(tool, "_lookup_wttr", slow)
    monkeypatch.setattr(tool, "_lookup_open_meteo", fast)
    task = asyncio.create_task(tool._lookup_first_available(city="test", day="today"))
    try:
        async with asyncio.timeout(2):
            await started.wait()
            if outcome == "cancel":
                task.cancel()
            await cleanup_started.wait()
            assert not task.done()
            release_cleanup.set()
            if outcome == "cancel":
                with pytest.raises(asyncio.CancelledError):
                    await task
            elif outcome == "failure":
                with pytest.raises(RuntimeError, match="All weather providers failed"):
                    await task
            else:
                assert await task is result
            assert finished.is_set()
    finally:
        release_cleanup.set()
        await asyncio.gather(task, return_exceptions=True)
        await tool.aclose()
