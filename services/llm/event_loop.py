"""Serialize acceptance from a shared Windows listening socket without blocking the loop."""

from __future__ import annotations

import asyncio
import ctypes
from ctypes import wintypes
import select
import socket
import sys
from typing import Any


def _last_windows_error() -> OSError:
    if sys.platform == "win32":
        return ctypes.WinError(ctypes.get_last_error())
    return OSError("Windows accept mutex is unavailable on this platform")


class _AcceptMutex:
    kernel: Any
    handle: int | None

    def __init__(self, port: int) -> None:
        if sys.platform != "win32":
            raise _last_windows_error()
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
        self.kernel.CreateMutexW.restype = wintypes.HANDLE
        self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.kernel.WaitForSingleObject.restype = wintypes.DWORD
        self.kernel.ReleaseMutex.argtypes = [wintypes.HANDLE]
        self.kernel.ReleaseMutex.restype = wintypes.BOOL
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL
        self.handle = self.kernel.CreateMutexW(None, False, f"Local\\NaweiLLMAccept-{port}")
        if not self.handle:
            raise _last_windows_error()

    def acquire(self) -> bool:
        result = self.kernel.WaitForSingleObject(self.handle, 0)
        if result in (0, 0x80):
            return True
        if result == 0x102:
            return False
        raise _last_windows_error()

    def release(self) -> None:
        if not self.kernel.ReleaseMutex(self.handle):
            raise _last_windows_error()

    def close(self) -> None:
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


class SharedSocketSelectorEventLoop(asyncio.SelectorEventLoop):
    def __init__(self) -> None:
        super().__init__()
        self._accept_mutexes: dict[int, _AcceptMutex] = {}

    def _accept_connection(
        self, protocol_factory: Any, sock: socket.socket, sslcontext: Any = None,
        server: Any = None, backlog: int = 100, ssl_handshake_timeout: float | None = None,
        ssl_shutdown_timeout: float | None = None,
    ) -> None:
        port = sock.getsockname()[1]
        mutex = self._accept_mutexes.get(port)
        if mutex is None:
            mutex = self._accept_mutexes[port] = _AcceptMutex(port)
        if not mutex.acquire():
            return
        try:
            sock.setblocking(False)
            if select.select([sock], [], [], 0)[0]:
                getattr(super(), "_accept_connection")(
                    protocol_factory, sock, sslcontext, server, 1,
                    ssl_handshake_timeout, ssl_shutdown_timeout,
                )
        finally:
            mutex.release()

    def close(self) -> None:
        super().close()
        for mutex in self._accept_mutexes.values():
            mutex.close()
        self._accept_mutexes.clear()
