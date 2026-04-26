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
import queue
from typing import List
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
import os

import httpx
import websockets

# --- Monitor Logic ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.broadcast_queue = asyncio.Queue()
        self.worker_task = None

    async def broadcast_worker(self):
        print("[SERVER] ConnectionManager v2.0 initialized.")
        print("[SERVER] Broadcast worker started.")
        while True:
            try:
                data = await self.broadcast_queue.get()
                await self.enrichment_broadcast(data)
                self.broadcast_queue.task_done()
            except Exception as e:
                print(f"[SERVER] Broadcast error: {e}")
                await asyncio.sleep(0.1)

    async def enrichment_broadcast(self, data: dict):
        if "timestamp" not in data:
            data["timestamp"] = datetime.now().isoformat()
        
        message = json.dumps(data)
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        if not self.worker_task or self.worker_task.done():
            self.worker_task = asyncio.create_task(self.broadcast_worker())
        print(f"[SERVER] WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"[SERVER] WebSocket disconnected. Remaining: {len(self.active_connections)}")

manager = ConnectionManager()

def get_monitored_env_class(manager):
    class MonitoredEnv(LongHorizonMemoryEnvironment):
        def _broadcast(self, obs, action=None):
            try:
                data = obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()
                if action:
                    data["operation"] = action.operation
                else:
                    data["operation"] = "reset"
                
                # Non-blocking put into the async queue
                manager.broadcast_queue.put_nowait(data)
            except Exception as e:
                print(f"[SERVER] Broadcast error: {e}")

        def step(self, action: LongHorizonMemoryAction) -> LongHorizonMemoryObservation:
            obs = super().step(action)
            self._broadcast(obs, action)
            return obs

        def reset(self) -> LongHorizonMemoryObservation:
            obs = super().reset()
            self._broadcast(obs)
            return obs
    return MonitoredEnv

app = create_app(
    get_monitored_env_class(manager),
    LongHorizonMemoryAction,
    LongHorizonMemoryObservation,
    env_name="long_horizon_memory",
    max_concurrent_envs=1,
)

# --- Serve custom UI ---
ui_dist_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard_dist")
if not os.path.exists(ui_dist_path):
    ui_dist_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "dist")

if os.path.exists(ui_dist_path):
    print(f"[SERVER] Mounting custom UI from {ui_dist_path}")
    # Clear default /web route by mutating the router's routes list
    new_routes = [r for r in app.router.routes if getattr(r, "path", None) != "/web"]
    app.router.routes.clear()
    app.router.routes.extend(new_routes)
    app.mount("/web", StaticFiles(directory=ui_dist_path, html=True), name="custom_web")
else:
    print(f"[SERVER] No custom UI found at {ui_dist_path}, using default.")


@app.websocket("/ws/monitor")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Just keep connection alive, we primarily push
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Middleware to intercept environment calls and broadcast updates
@app.post("/step")
async def monitored_step(action_req: dict):
    # This is a bit tricky because create_app hides the original route
    # We'll use a wrapper or just rely on the environment class broadcasting
    pass # See next step for better integration

# --- Existing routes ---


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/")
async def root_redirect():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/web")


@app.get("/routes")
async def list_routes():
    return [{"path": route.path, "name": route.name} for route in app.routes]


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
