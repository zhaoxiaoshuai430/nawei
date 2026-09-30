from __future__ import annotations

import hmac
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from fastapi import HTTPException

from .auth import digest


class DeviceRegistry:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.transaction() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS devices (
                    device_id TEXT PRIMARY KEY, client_id TEXT NOT NULL, key_hash TEXT NOT NULL,
                    user_id TEXT, name TEXT NOT NULL DEFAULT 'Nawei Robot', voice TEXT NOT NULL DEFAULT 'liliya',
                    role TEXT NOT NULL DEFAULT 'companion', enabled INTEGER NOT NULL DEFAULT 1,
                    nonce TEXT NOT NULL, token_hash TEXT, last_seen REAL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS activations (
                    code TEXT PRIMARY KEY, device_id TEXT NOT NULL,
                    expires_at REAL NOT NULL, used INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS activation_device ON activations(device_id);
            """)

    @contextmanager
    def transaction(self, *, write: bool = True) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
                yield connection
        finally:
            connection.close()

    def provision(self, device: str, client: str, key: str, ttl: int) -> dict[str, Any]:
        now = time.time()
        with self.transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO devices(device_id,client_id,key_hash,nonce,created_at) VALUES(?,?,?,?,?)",
                (device, client, digest(key), secrets.token_hex(32), now),
            )
            row = connection.execute("SELECT * FROM devices WHERE device_id=?", (device,)).fetchone()
            if not row["user_id"] and not row["key_hash"]:
                connection.execute("UPDATE devices SET client_id=?,key_hash=? WHERE device_id=?",
                                   (client, digest(key), device))
                row = connection.execute("SELECT * FROM devices WHERE device_id=?", (device,)).fetchone()
            self._check_identity(row, client, key)
            if not row["enabled"]:
                raise HTTPException(403, "Device disabled")
            if row["user_id"]:
                token = "nwdev_" + hmac.new(key.encode(), f"{device}:{client}:{row['nonce']}".encode(), "sha256").hexdigest()
                connection.execute("UPDATE devices SET token_hash=? WHERE device_id=?", (digest(token), device))
                return {"bound": True, "token": token}
            activation = connection.execute(
                "SELECT * FROM activations WHERE device_id=? AND used=0 AND expires_at>?", (device, now)
            ).fetchone()
            if activation is None:
                connection.execute("DELETE FROM activations WHERE expires_at<?", (now - 86400,))
                for attempt in range(20):
                    code = f"{secrets.randbelow(1000000):06d}"
                    inserted = connection.execute(
                        "INSERT OR IGNORE INTO activations VALUES(?,?,?,0)", (code, device, now + ttl)
                    )
                    if inserted.rowcount:
                        activation = {"code": code, "expires_at": now + ttl}
                        break
                if activation is None:
                    raise HTTPException(503, "Activation capacity exceeded")
            return {"bound": False, "code": activation["code"], "expires_at": activation["expires_at"]}

    @staticmethod
    def _check_identity(row: Any, client: str, key: str) -> None:
        if row is None or row["client_id"] != client or not hmac.compare_digest(row["key_hash"], digest(key)):
            raise HTTPException(401, "Invalid device identity")

    def bind(self, code: str, user: str, name: str) -> dict[str, Any]:
        with self.transaction() as connection:
            activation = connection.execute("SELECT * FROM activations WHERE code=?", (code,)).fetchone()
            if activation is None:
                raise HTTPException(404, "activation_invalid")
            if activation["used"]:
                raise HTTPException(409, "activation_used")
            if activation["expires_at"] <= time.time():
                raise HTTPException(410, "activation_expired")
            device = activation["device_id"]
            row = connection.execute("SELECT * FROM devices WHERE device_id=?", (device,)).fetchone()
            if row["user_id"]:
                raise HTTPException(409, "device_already_bound")
            if not row["enabled"]:
                raise HTTPException(403, "device_disabled")
            connection.execute("UPDATE devices SET user_id=?,name=?,nonce=?,token_hash=NULL WHERE device_id=?",
                               (user, name, secrets.token_hex(32), device))
            connection.execute("UPDATE activations SET used=1 WHERE device_id=?", (device,))
            return self._public(connection.execute("SELECT * FROM devices WHERE device_id=?", (device,)).fetchone())

    def authenticate(self, device: str, client: str, token: str) -> dict[str, Any]:
        with self.transaction(write=False) as connection:
            row = connection.execute("SELECT * FROM devices WHERE device_id=?", (device,)).fetchone()
            if (row is None or not row["user_id"] or not row["enabled"] or row["client_id"] != client
                    or not row["token_hash"] or not hmac.compare_digest(row["token_hash"], digest(token))):
                raise HTTPException(401, "Invalid device credential")
            return dict(row)

    def list_devices(self, user: str) -> list[dict[str, Any]]:
        with self.transaction(write=False) as connection:
            return [self._public(row) for row in connection.execute(
                "SELECT * FROM devices WHERE user_id=? ORDER BY created_at", (user,))]

    def get(self, device: str, user: str) -> dict[str, Any]:
        with self.transaction(write=False) as connection:
            return self._public(self._owned(connection, device, user))

    @staticmethod
    def _owned(connection: sqlite3.Connection, device: str, user: str) -> sqlite3.Row:
        row = connection.execute("SELECT * FROM devices WHERE device_id=? AND user_id=?", (device, user)).fetchone()
        if row is None:
            raise HTTPException(404, "Device not found")
        return row

    def update(self, device: str, user: str, values: dict[str, Any]) -> dict[str, Any]:
        with self.transaction() as connection:
            self._owned(connection, device, user)
            for field in ("name", "voice", "role", "enabled"):
                if field in values:
                    connection.execute(f"UPDATE devices SET {field}=? WHERE device_id=?", (values[field], device))
            return self._public(self._owned(connection, device, user))

    def unbind(self, device: str, user: str) -> None:
        with self.transaction() as connection:
            self._owned(connection, device, user)
            connection.execute(
                "UPDATE devices SET user_id=NULL,token_hash=NULL,key_hash='',client_id='',enabled=1,nonce=?,name='Nawei Robot',"
                "voice='liliya',role='companion' WHERE device_id=?", (secrets.token_hex(32), device))
            connection.execute("UPDATE activations SET used=1 WHERE device_id=?", (device,))

    def seen(self, device: str) -> None:
        with self.transaction() as connection:
            connection.execute("UPDATE devices SET last_seen=? WHERE device_id=?", (time.time(), device))

    @staticmethod
    def _public(row: sqlite3.Row) -> dict[str, Any]:
        return {field: row[field] for field in (
            "device_id", "client_id", "name", "voice", "role", "enabled", "last_seen", "created_at")}
