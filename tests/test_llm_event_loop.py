import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest

from services.llm import run


@pytest.mark.parametrize(("platform", "workers", "expected"), [
    ("win32", 4, "services.llm.event_loop:SharedSocketSelectorEventLoop"),
    ("win32", 1, "auto"),
    ("linux", 4, "auto"),
])
def test_llm_configures_shared_socket_loop_only_for_windows_workers(monkeypatch, platform, workers, expected):
    observed = {}
    monkeypatch.setattr(run.sys, "platform", platform)
    monkeypatch.setattr(run, "WORKERS", workers)
    monkeypatch.setattr(run.uvicorn, "run", lambda app, **kwargs: observed.update(kwargs))
    run.main()
    assert observed["loop"] == expected
    assert observed["workers"] == workers


@pytest.mark.skipif(sys.platform != "win32", reason="Windows shared socket regression")
def test_shared_listener_mode_change_does_not_block_worker_timers():
    command = r'''
import asyncio
import json
import os
import socket
import threading
from uvicorn import Config

config = Config("services.llm.main:app", workers=4,
                loop="services.llm.event_loop:SharedSocketSelectorEventLoop")
factory = config.get_loop_factory()
loop = factory()
asyncio.set_event_loop(loop)
listener = socket.socket()
listener.bind(("127.0.0.1", 0))
listener.listen()
listener.setblocking(False)
address = listener.getsockname()
connections = []

class Protocol(asyncio.Protocol):
    def connection_made(self, transport):
        connections.append(transport)

async def main():
    server = await loop.create_server(Protocol, sock=listener)
    duplicate = socket.fromshare(listener.share(os.getpid()))
    duplicate.close()
    listener.setblocking(True)
    client = socket.create_connection(address, timeout=2)
    try:
        await asyncio.sleep(0.1)
        assert len(connections) == 1
        listener.setblocking(True)
        loop._accept_connection(Protocol, listener)
        print(json.dumps({"timer_ran": True, "accepted": len(connections)}), flush=True)
    finally:
        client.close()
        for transport in connections:
            transport.close()
        server.close()
        await server.wait_closed()

watchdog = threading.Timer(4, lambda: os._exit(9))
watchdog.start()
try:
    loop.run_until_complete(main())
finally:
    watchdog.cancel()
    loop.close()
'''
    result = subprocess.run(
        [sys.executable, "-c", command], cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        capture_output=True, text=True, timeout=8,
    )
    assert result.returncode == 0, result.stderr or "Worker event loop stalled while accepting a connection"
    assert json.loads(result.stdout) == {"timer_ran": True, "accepted": 1}


@pytest.mark.skipif(sys.platform != "win32", reason="Windows cross-process accept mutex")
def test_accept_mutex_does_not_wait_for_another_process():
    from services.llm.event_loop import _AcceptMutex

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mutex = _AcceptMutex(port)
    command = r'''
import sys
from services.llm.event_loop import _AcceptMutex
mutex = _AcceptMutex(int(sys.argv[1]))
try:
    acquired = mutex.acquire()
    print(acquired, flush=True)
    if acquired:
        mutex.release()
finally:
    mutex.close()
'''

    def acquire_in_child():
        result = subprocess.run(
            [sys.executable, "-c", command, str(port)],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=5, check=True,
        )
        return result.stdout.strip()

    try:
        assert mutex.acquire()
        try:
            assert acquire_in_child() == "False"
        finally:
            mutex.release()
        assert acquire_in_child() == "True"
    finally:
        mutex.close()
