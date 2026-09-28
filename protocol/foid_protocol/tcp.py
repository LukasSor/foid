"""Async TCP line server/client for the room protocol (default port 9500)."""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from foid_protocol.codec import decode_line, encode
from foid_protocol.messages import Ack, Command, Hello, Snapshot, parse_message

DEFAULT_PORT = 9500
MAX_LINE = 1_000_000

Handler = Callable[[dict[str, Any], asyncio.StreamWriter], Awaitable[None]]


class LineServer:
    def __init__(self, host: str, port: int, handler: Handler):
        self.host = host
        self.port = port
        self.handler = handler
        self._server: asyncio.AbstractServer | None = None
        self.writers: set[asyncio.StreamWriter] = set()

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._client, self.host, self.port)

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        for writer in list(self.writers):
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
        self.writers.clear()

    async def broadcast(self, message: Any) -> None:
        data = encode(message)
        dead: list[asyncio.StreamWriter] = []
        for writer in self.writers:
            try:
                writer.write(data)
                await writer.drain()
            except (ConnectionError, RuntimeError):
                dead.append(writer)
        for writer in dead:
            self.writers.discard(writer)
            writer.close()

    async def _client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.writers.add(writer)
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                if len(line) > MAX_LINE:
                    break
                try:
                    payload = decode_line(line)
                except (ValueError, UnicodeDecodeError):
                    continue
                await self.handler(payload, writer)
        finally:
            self.writers.discard(writer)
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()


class LineClient:
    def __init__(self, host: str, port: int, timeout: float = 4.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._lock = asyncio.Lock()

    @property
    def connected(self) -> bool:
        return self._writer is not None and not self._writer.is_closing()

    async def connect(self) -> None:
        if self.connected:
            return
        self._reader, self._writer = await asyncio.wait_for(
            asyncio.open_connection(self.host, self.port),
            timeout=self.timeout,
        )
        await self.send(Hello(role="voice"))

    async def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            with contextlib.suppress(Exception):
                await self._writer.wait_closed()
        self._reader = None
        self._writer = None

    async def send(self, message: Any) -> None:
        await self.connect()
        assert self._writer is not None
        self._writer.write(encode(message))
        await self._writer.drain()

    async def command(self, device: str, action: str, extra: dict[str, Any] | None = None) -> Ack:
        msg = Command(id=str(uuid.uuid4()), device=device, action=action, extra=extra or {})
        async with self._lock:
            await self.send(msg)
            ack = await self._read_ack(msg.id)
        return ack

    async def _read_ack(self, command_id: str) -> Ack:
        assert self._reader is not None
        deadline = asyncio.get_event_loop().time() + self.timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise TimeoutError("room box did not acknowledge command")
            line = await asyncio.wait_for(self._reader.readline(), timeout=remaining)
            if not line:
                await self.close()
                raise ConnectionError("room box closed the connection")
            parsed = parse_message(decode_line(line))
            if isinstance(parsed, Ack) and parsed.id == command_id:
                return parsed
            if isinstance(parsed, (Snapshot, Hello)):
                continue
