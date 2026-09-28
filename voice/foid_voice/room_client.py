from __future__ import annotations

import asyncio
import logging
import socket
import threading
from typing import Any

from foid_protocol.messages import Ack
from foid_protocol.tcp import LineClient

log = logging.getLogger("foid.room")


class RoomLink:
    """Thread-safe wrapper around the async JSON/TCP client."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._client = LineClient(host, port)
        self._ready = threading.Event()
        self._link_lock = threading.Lock()
        self._thread.start()
        self._ready.wait(timeout=2)
        self.connect()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    def configure(self, host: str, port: int) -> None:
        self.host = host
        self.port = port

        async def _reset() -> None:
            await self._client.close()
            self._client.host = host
            self._client.port = port

        self._submit(_reset())
        self.connect()

    @property
    def connected(self) -> bool:
        return self._client.connected

    def reachable(self) -> bool:
        """True when the configured host:port accepts a TCP connection."""
        if self.connected:
            return True
        try:
            with socket.create_connection((self.host, self.port), timeout=0.4):
                return True
        except OSError:
            return False

    def connect(self) -> bool:
        """Open the protocol connection if the room box is up."""
        with self._link_lock:
            if self.connected:
                return True
            try:
                self._submit(self._client.connect(), timeout=2.0)
                return self.connected
            except Exception as exc:
                log.debug("room connect failed: %s", exc)
                return False

    def probe(self) -> bool:
        """Mark the room as up if TCP is reachable, and connect when possible."""
        if self.connected:
            return True
        if not self.reachable():
            return False
        self.connect()
        return True

    def command(self, device: str, action: str) -> Ack:
        return self._submit(self._ensure_command(device, action))

    async def _ensure_command(self, device: str, action: str) -> Ack:
        try:
            return await self._client.command(device, action)
        except Exception:
            await self._client.close()
            return await self._client.command(device, action)

    def close(self) -> None:
        self._submit(self._client.close())
        self._loop.call_soon_threadsafe(self._loop.stop)

    def _submit(self, coro, timeout: float = 8) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)
