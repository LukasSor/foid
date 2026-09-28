from __future__ import annotations

import argparse
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from foid_protocol.catalog import load_catalog
from foid_protocol.i18n import javascript_bundle
from foid_protocol.codec import encode
from foid_protocol.messages import Ack, Command, Hello, Snapshot, StateEvent, parse_message
from foid_protocol.tcp import DEFAULT_PORT, LineServer
from foid_sim.room import Room

log = logging.getLogger("foid.sim")
HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"


class Hub:
    def __init__(self, tcp_host: str, tcp_port: int):
        self.catalog = load_catalog()
        self.room = Room(self.catalog)
        self.sockets: set[WebSocket] = set()
        self.tcp = LineServer(tcp_host, tcp_port, self._on_tcp)

    async def start(self) -> None:
        await self.tcp.start()
        log.info("room TCP on %s:%s", self.tcp.host, self.tcp.port)

    async def close(self) -> None:
        await self.tcp.close()

    async def _on_tcp(self, payload: dict[str, Any], writer) -> None:
        try:
            message = parse_message(payload)
        except Exception:
            return
        if isinstance(message, Hello):
            writer.write(encode(Snapshot(devices=self.room.snapshot())))
            await writer.drain()
            return
        if isinstance(message, Command):
            await self._run_command(message.device, message.action, writer, message.id)

    async def _run_command(self, device: str, action: str, writer=None, command_id: str = "") -> dict[str, Any]:
        try:
            state = self.room.apply(device, action)
            ack = Ack(id=command_id or "ui", ok=True, state=state)
        except Exception as exc:
            ack = Ack(id=command_id or "ui", ok=False, error=str(exc))
            if writer is not None:
                writer.write(encode(ack))
                await writer.drain()
            return ack.model_dump()
        if writer is not None:
            writer.write(encode(ack))
            await writer.drain()
        event = StateEvent(device=device, state=state, all=self.room.snapshot())
        await self.tcp.broadcast(event)
        await self.broadcast_ui()
        return {"ok": True, **self.room.public(), "changed": {"device": device, "state": state}}

    async def broadcast_ui(self) -> None:
        payload = self.room.public()
        dead: list[WebSocket] = []
        for socket in self.sockets:
            try:
                await socket.send_json(payload)
            except Exception:
                dead.append(socket)
        for socket in dead:
            self.sockets.discard(socket)


def create_app(tcp_host: str, tcp_port: int) -> FastAPI:
    hub = Hub(tcp_host, tcp_port)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await hub.start()
        app.state.hub = hub
        yield
        await hub.close()

    app = FastAPI(title="Foid Room", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/i18n.js")
    def i18n_js():
        return Response(javascript_bundle(), media_type="application/javascript")

    @app.get("/api/state")
    def api_state():
        return hub.room.public()

    @app.post("/api/command")
    async def api_command(payload: dict):
        device = str(payload.get("device") or "")
        action = str(payload.get("action") or "")
        result = await hub._run_command(device, action)
        if not result.get("ok"):
            return JSONResponse(result, status_code=400)
        return result

    @app.websocket("/ws")
    async def ws_room(socket: WebSocket):
        await socket.accept()
        hub.sockets.add(socket)
        await socket.send_json(hub.room.public())
        try:
            while True:
                await socket.receive_text()
        except WebSocketDisconnect:
            hub.sockets.discard(socket)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Foid room simulator")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--tcp-host", default="0.0.0.0")
    parser.add_argument("--tcp-port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = create_app(args.tcp_host, args.tcp_port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
