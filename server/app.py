# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Long Horizon Memory Environment.

This module creates an HTTP server that exposes the LongHorizonMemoryEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from models import LongHorizonMemoryAction, LongHorizonMemoryObservation
    from server.long_horizon_memory_environment import LongHorizonMemoryEnvironment
except (ImportError, ModuleNotFoundError):
    try:
        from ..models import LongHorizonMemoryAction, LongHorizonMemoryObservation
        from .long_horizon_memory_environment import LongHorizonMemoryEnvironment
    except (ImportError, ModuleNotFoundError):
        from long_horizon_memory.models import LongHorizonMemoryAction, LongHorizonMemoryObservation
        from long_horizon_memory.server.long_horizon_memory_environment import LongHorizonMemoryEnvironment


from datetime import datetime
import json
import asyncio
import queue          # <-- thread-safe queue (this is what V2 uses!)
from typing import List
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
import os


# ---------------------------------------------------------------------------
# Connection Manager  (mirrors the WORKING V2 pattern exactly)
# ---------------------------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        # CRITICAL: Use stdlib queue.Queue, NOT asyncio.Queue.
        # queue.Queue.put() is thread-safe from ANY thread.
        # asyncio.Queue is NOT safe from non-event-loop threads.
        self.broadcast_queue = queue.Queue()
        self.worker_task = None

    async def broadcast_worker(self):
        """Continuously drain the thread-safe queue and broadcast to WS clients."""
        print("[SERVER] broadcast_worker started")
        while True:
            try:
                # run_in_executor blocks on queue.get() in a thread,
                # yielding control to the event loop while waiting
                data = await asyncio.get_event_loop().run_in_executor(
                    None, self.broadcast_queue.get
                )
                await self._send_to_all(data)
            except Exception as e:
                print(f"[SERVER] broadcast_worker error: {e}")
                await asyncio.sleep(0.1)

    async def _send_to_all(self, data: dict):
        if "timestamp" not in data:
            data["timestamp"] = datetime.now().isoformat()
        message = json.dumps(data)
        n = len(self.active_connections)
        if n > 0:
            print(f"[SERVER] Broadcasting to {n} client(s)")
        dead = []
        for ws in self.active_connections:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active_connections.remove(ws)

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        # Start worker on first connection
        if not self.worker_task or self.worker_task.done():
            self.worker_task = asyncio.create_task(self.broadcast_worker())
        print(f"[SERVER] WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"[SERVER] WebSocket disconnected. Remaining: {len(self.active_connections)}")


manager = ConnectionManager()


# ---------------------------------------------------------------------------
# Monitored Environment  (mirrors V2 exactly)
# ---------------------------------------------------------------------------
def get_monitored_env_class(mgr):
    class MonitoredEnv(LongHorizonMemoryEnvironment):
        def _broadcast(self, data: dict):
            """Put data into the thread-safe queue. Safe from any thread."""
            print(f"[SERVER] _broadcast: putting data in queue (op={data.get('operation')})")
            mgr.broadcast_queue.put(data)

        def step(self, action: LongHorizonMemoryAction) -> LongHorizonMemoryObservation:
            obs = super().step(action)
            try:
                data = obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()
                data["operation"] = action.operation
                self._broadcast(data)
            except Exception as e:
                print(f"[SERVER] step broadcast error: {e}")
            return obs

        def reset(self) -> LongHorizonMemoryObservation:
            obs = super().reset()
            try:
                data = obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()
                data["operation"] = "reset"
                self._broadcast(data)
            except Exception as e:
                print(f"[SERVER] reset broadcast error: {e}")
            return obs

    return MonitoredEnv


# ---------------------------------------------------------------------------
# Create the OpenEnv FastAPI app
# ---------------------------------------------------------------------------
app = create_app(
    get_monitored_env_class(manager),
    LongHorizonMemoryAction,
    LongHorizonMemoryObservation,
    env_name="long_horizon_memory",
    max_concurrent_envs=1,
)


# ---------------------------------------------------------------------------
# Serve custom dashboard UI
# ---------------------------------------------------------------------------
ui_dist_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard_dist"
)
if not os.path.exists(ui_dist_path):
    ui_dist_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "dist")

if os.path.exists(ui_dist_path):
    print(f"[SERVER] Mounting custom UI from {ui_dist_path}")
    # Remove existing default /web route so ours takes priority
    new_routes = [r for r in app.router.routes if getattr(r, "path", None) != "/web"]
    app.router.routes.clear()
    app.router.routes.extend(new_routes)
    app.mount("/web", StaticFiles(directory=ui_dist_path, html=True), name="custom_web")

    @app.get("/web")
    async def web_redirect():
        return RedirectResponse(url="/web/")
else:
    print(f"[SERVER] No custom UI found at {ui_dist_path}, using default.")


# ---------------------------------------------------------------------------
# WebSocket endpoint for dashboard
# ---------------------------------------------------------------------------
@app.websocket("/ws/monitor")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ---------------------------------------------------------------------------
# Utility routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/")
async def root_redirect():
    return RedirectResponse(url="/web/")


@app.get("/routes")
async def list_routes():
    return [{"path": getattr(r, "path", str(r)), "name": getattr(r, "name", "")} for r in app.routes]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(host: str = "0.0.0.0", port: int = 7860):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m long_horizon_memory.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn long_horizon_memory.server.app:app --workers 4
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
