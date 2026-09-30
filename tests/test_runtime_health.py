import asyncio
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from services.llm.memory import MemoryStore
from services.llm.agent_runtime import LLMAgentRuntime


POWERSHELL = shutil.which("powershell.exe")


@pytest.mark.skipif(POWERSHELL is None, reason="Windows process records require PowerShell")
@pytest.mark.parametrize("locked", [False, True])
def test_process_records_survive_updates_and_locked_reads(tmp_path, locked):
    command = r"""
    $ErrorActionPreference = 'Stop'
    . ./scripts/project_runtime.ps1
    $path = Join-Path $env:NAWEI_TEST_RECORD_DIRECTORY 'processes.json'
    Save-NaweiProcessRecords -Records @([pscustomobject]@{name='llm'; pid=123}) -Path $path
    $records = Get-Content $path -Raw | ConvertFrom-Json
    $records = @($records | Where-Object name -ne 'gateway') + @([pscustomobject]@{name='gateway'; pid=456})
    Save-NaweiProcessRecords -Records $records -Path $path
    $records = Get-Content $path -Raw | ConvertFrom-Json
    $records = @($records | Where-Object name -ne 'llm') + @([pscustomobject]@{name='llm'; pid=789})
    if ($env:NAWEI_TEST_LOCKED -eq 'true') {
        $reader = [IO.File]::Open($path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
        try {
            try { Save-NaweiProcessRecords -Records $records -Path $path; throw 'Expected locked write failure' }
            catch [IO.IOException] { }
        } finally { $reader.Dispose() }
    } else { Save-NaweiProcessRecords -Records $records -Path $path }
    Get-Content $path -Raw
    """
    result = subprocess.run(
        [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "NAWEI_TEST_RECORD_DIRECTORY": str(tmp_path), "NAWEI_TEST_LOCKED": str(locked).lower()},
        capture_output=True, text=True, timeout=15, check=True,
    )
    records = json.loads(result.stdout)
    assert len(records) == 2
    assert {record["name"]: record["pid"] for record in records} == {"llm": 123 if locked else 789, "gateway": 456}
    assert not list(tmp_path.glob("*.tmp"))


async def test_drain_monitor_recovers_after_status_file_is_locked(tmp_path, monkeypatch):
    from services.llm.drain import LocalDrainState

    state = LocalDrainState(str(tmp_path))
    monkeypatch.setattr(state, "snapshot", lambda runtime: {"drained": False})
    publish = AsyncMock(side_effect=[PermissionError("file is locked"), None])
    monkeypatch.setattr("services.llm.drain.asyncio.to_thread", publish)
    task = asyncio.create_task(state.monitor(None))
    try:
        async with asyncio.timeout(2):
            while publish.await_count < 2:
                await asyncio.sleep(0.01)
        assert not task.done()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def test_drain_status_write_retries_transient_windows_lock(tmp_path, monkeypatch):
    from services.llm.drain import LocalDrainState

    state = LocalDrainState(str(tmp_path))
    original_replace = Path.replace
    replace_calls = 0
    delays = []

    def flaky_replace(path, target):
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls < 3:
            raise PermissionError("file is temporarily locked")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    monkeypatch.setattr("services.llm.drain.time.sleep", delays.append)

    state._write({"drained": False})

    target = tmp_path / f"{os.getpid()}.json"
    assert json.loads(target.read_text(encoding="utf-8")) == {"drained": False}
    assert replace_calls == 3
    assert delays == [0.01, 0.02]


async def test_required_memory_startup_waits_for_dependencies(monkeypatch):
    memory_store = SimpleNamespace(
        initialize=AsyncMock(),
        health=AsyncMock(side_effect=[
            {"postgres": "fallback", "redis": "ok"},
            {"postgres": "ok", "redis": "ok"},
        ]),
    )
    knowledge_client = SimpleNamespace(health=AsyncMock(return_value={"status": "ok"}))
    runtime = LLMAgentRuntime(memory_store=memory_store, knowledge_client=knowledge_client)
    sleep = AsyncMock()
    monkeypatch.setattr("services.llm.agent_runtime.asyncio.sleep", sleep)
    monkeypatch.setenv("LLM_MEMORY_REQUIRED", "true")
    monkeypatch.setenv("LLM_MEMORY_STARTUP_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("KNOWLEDGE_REQUIRED", "false")

    await runtime.initialize()

    memory_store.initialize.assert_awaited_once()
    assert memory_store.health.await_count == 2
    sleep.assert_awaited_once_with(0.5)
    knowledge_client.health.assert_awaited_once()


async def test_required_memory_startup_respects_timeout(monkeypatch):
    memory_store = SimpleNamespace(
        initialize=AsyncMock(),
        health=AsyncMock(return_value={"postgres": "fallback", "redis": "fallback"}),
    )
    knowledge_client = SimpleNamespace(health=AsyncMock(return_value={"status": "ok"}))
    runtime = LLMAgentRuntime(memory_store=memory_store, knowledge_client=knowledge_client)
    monkeypatch.setenv("LLM_MEMORY_REQUIRED", "true")
    monkeypatch.setenv("LLM_MEMORY_STARTUP_TIMEOUT_SECONDS", "0")

    with pytest.raises(RuntimeError, match="postgres, redis"):
        await runtime.initialize()

    memory_store.health.assert_awaited_once()
    knowledge_client.health.assert_not_awaited()


@pytest.mark.skipif(POWERSHELL is None, reason="Windows startup scripts require PowerShell")
@pytest.mark.parametrize(("body", "expected"), [
    ({"status": "ok"}, True),
    ({"status": "ready"}, True),
    ({"status": "degraded"}, False),
    ({"status": "ok", "ready": False}, False),
    ({"status": "ok", "ready": True}, True),
    ({"loaded": False}, False),
    ({"loaded": True}, True),
    ({}, False),
])
def test_windows_health_checks_readiness_body(body, expected):
    command = """
    . ./scripts/project_runtime.ps1
    function Invoke-WebRequest {
        [pscustomobject]@{StatusCode=200; Content=$env:NAWEI_TEST_HEALTH_BODY}
    }
    Test-NaweiHealth 'http://test/health'
    """
    result = subprocess.run(
        [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "NAWEI_TEST_HEALTH_BODY": json.dumps(body)},
        capture_output=True, text=True, timeout=15, check=True,
    )
    assert result.stdout.strip().lower() == str(expected).lower()


@pytest.mark.parametrize("available", [True, False])
async def test_memory_health_checks_live_postgres_connection(monkeypatch, available):
    monkeypatch.setenv("MEMORY_ENABLED", "true")
    store = MemoryStore()
    store.initialize = AsyncMock()
    store._pool = AsyncMock()
    store._pool.fetchval.return_value = 1
    if not available:
        store._pool.fetchval.side_effect = ConnectionError("database is offline")
    store._redis = AsyncMock()
    store._redis.ping.return_value = True
    result = await store.health()
    assert result["postgres"] == ("ok" if available else "fallback")
    assert result["redis"] == "ok"
    store._pool.fetchval.assert_awaited_once()


async def test_memory_health_bounds_postgres_probe(monkeypatch):
    monkeypatch.setenv("MEMORY_ENABLED", "true")
    monkeypatch.setenv("MEMORY_CONNECT_TIMEOUT_SECONDS", "0.01")
    store = MemoryStore()
    store.initialize = AsyncMock()
    store._pool = AsyncMock()
    store._redis = AsyncMock()

    async def stalled(*args, **kwargs):
        await asyncio.Event().wait()

    store._pool.fetchval.side_effect = stalled
    result = await asyncio.wait_for(store.health(), timeout=1)
    assert result["postgres"] == "fallback"
    assert "TimeoutError" in result["last_error"]


@pytest.mark.skipif(POWERSHELL is None, reason="Windows process ownership checks require PowerShell")
@pytest.mark.parametrize(("project", "module", "child", "expected"), [
    (True, "services.llm.run", False, "accepted"),
    (False, "services.llm.run", False, "rejected"),
    (True, "services.gateway.run", False, "rejected"),
    (True, "services.llm.run", True, "accepted"),
    (True, "services.llm.run-other", False, "rejected"),
])
def test_stop_checks_project_and_service_ownership(project, module, child, expected):
    command = r"""
    . ./scripts/project_runtime.ps1
    function Get-CimInstance {
        param($Filter)
        if ($env:NAWEI_TEST_CHILD -eq 'true' -and $Filter -eq 'ProcessId=123') {
            return [pscustomobject]@{Name='python.exe'; ExecutablePath='C:\base\python.exe';
                CommandLine='C:\base\python.exe -m services.llm.run'; ParentProcessId=456}
        }
        [pscustomobject]@{Name='python.exe'; CommandLine=$env:NAWEI_TEST_PROCESS_COMMAND;
            ExecutablePath=$env:NAWEI_TEST_PROCESS_EXE; ParentProcessId=0}
    }
    $service = Get-NaweiServiceDefinitions | Where-Object Name -eq 'llm'
    try { Assert-NaweiProcessOwner -Service $service -ProcessId 123; 'accepted' } catch { 'rejected' }
    """
    root = Path(__file__).resolve().parents[1]
    directory = str(root) if project else r"C:\unrelated"
    result = subprocess.run(
        [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        cwd=root, env={**os.environ, "NAWEI_TEST_PROCESS_COMMAND": f'"{directory}\\.venv\\Scripts\\python.exe" -m {module}',
                       "NAWEI_TEST_PROCESS_EXE": f"{directory}\\.venv\\Scripts\\python.exe", "NAWEI_TEST_CHILD": str(child).lower()},
        capture_output=True, text=True, timeout=15, check=True,
    )
    assert result.stdout.strip() == expected


@pytest.mark.skipif(POWERSHELL is None, reason="Windows drain coordination requires PowerShell")
@pytest.mark.parametrize(("workers", "age", "drained", "expected"), [
    (1, 0, True, "accepted"), (2, 0, True, "rejected"),
    (1, 10, True, "rejected"), (1, 0, False, "rejected"),
])
def test_stop_waits_for_every_fresh_drained_worker(tmp_path, workers, age, drained, expected):
    command = """
    . ./scripts/project_runtime.ps1
    $state = @{pid=$PID; updated_at=([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()/1000 - [int]$env:NAWEI_TEST_AGE);
               drained=($env:NAWEI_TEST_DRAINED -eq 'true')}
    $state | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $env:NAWEI_TEST_DRAIN_PATH 'worker.json')
    $record = [pscustomobject]@{drain_directory=$env:NAWEI_TEST_DRAIN_PATH; workers=[int]$env:NAWEI_TEST_WORKERS}
    try { Wait-NaweiLlmDrain -Record $record -TimeoutSeconds 1; 'accepted' } catch { 'rejected' }
    """
    result = subprocess.run(
        [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "NAWEI_TEST_AGE": str(age), "NAWEI_TEST_DRAINED": str(drained).lower(),
             "NAWEI_TEST_DRAIN_PATH": str(tmp_path), "NAWEI_TEST_WORKERS": str(workers)},
        capture_output=True, text=True, timeout=15, check=True,
    )
    assert result.stdout.strip().splitlines()[-1] == expected
    assert (tmp_path / "drain.request").exists() == (expected == "accepted")
