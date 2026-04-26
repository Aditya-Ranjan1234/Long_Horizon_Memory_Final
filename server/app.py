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
        self.hf_task = None
        self.broadcast_queue = queue.Queue()
        self.worker_task = None

    async def broadcast_worker(self):
        print("[DEBUG] broadcast_worker started")
        while True:
            try:
                # Use run_in_executor to avoid blocking the loop while waiting for the queue
                data = await asyncio.get_event_loop().run_in_executor(None, self.broadcast_queue.get)
                await self.enrichment_broadcast(data)
            except Exception as e:
                print(f"[DEBUG] broadcast_worker error: {e}")
                await asyncio.sleep(0.1)

    async def enrichment_broadcast(self, data: dict):
        print(f"[DEBUG] enrichment_broadcast entered with {len(self.active_connections)} connections")
        if "timestamp" not in data:
            data["timestamp"] = datetime.now().isoformat()
        
        message = json.dumps(data)
        client_count = len(self.active_connections)
        if client_count > 0:
            print(f"[BROADCAST] Sending update to {client_count} clients")
            
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
                print(f"[DEBUG] Sent message to client {id(connection)}")
            except Exception as e:
                print(f"[BROADCAST ERROR] {e}")
                pass

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Start worker task if not running
        if not self.worker_task or self.worker_task.done():
            self.worker_task = asyncio.create_task(self.broadcast_worker())
            
        print(f"[DEBUG] Connection accepted. Total connections: {len(self.active_connections)}")
        is_hf = os.environ.get("SPACE_ID") is not None
        if not is_hf:
            if not self.hf_task or self.hf_task.done():
                self.hf_task = asyncio.create_task(self.proxy_hf_updates())
        else:
            print("[SERVER] Running on HF Space, skipping self-proxy.")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def proxy_hf_updates(self):
        """Proxy updates from the HF Space WebSocket to our local clients."""
        # Using the base URL provided in test scripts but with wss protocol
        hf_ws_url = "wss://aditya-ranjan1234-long-horizon-memory-env-final.hf.space/ws/monitor"
        print(f"[PROXY] Connecting to HF Space: {hf_ws_url}")
        
        while True:
            try:
                import websockets
                async with websockets.connect(hf_ws_url) as hf_ws:
                    print("[PROXY] Connected to HF Space WebSocket")
                    while True:
                        msg = await hf_ws.recv()
                        data = json.loads(msg)
                        await self.enrichment_broadcast(data)
            except Exception as e:
                print(f"[PROXY] HF Space Connection Error: {e}. Retrying in 5s...")
                await asyncio.sleep(5)

manager = ConnectionManager()

# Create the app with web interface and README integration
def get_monitored_env_class(manager):
    class MonitoredEnv(LongHorizonMemoryEnvironment):
        def _broadcast(self, data: dict):
            print(f"[DEBUG] _broadcast bridge triggered. Putting data in queue.")
            manager.broadcast_queue.put(data)

        def step(self, action: LongHorizonMemoryAction) -> LongHorizonMemoryObservation:
            obs = super().step(action)
            try:
                data = obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()
                data["operation"] = action.operation
                self._broadcast(data)
            except Exception as e:
                print(f"[BROADCAST ERROR] {e}")
            return obs

        def reset(self) -> LongHorizonMemoryObservation:
            obs = super().reset()
            try:
                data = obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()
                data["operation"] = "reset"
                self._broadcast(data)
            except Exception as e:
                print(f"[BROADCAST ERROR] {e}")
            return obs
    return MonitoredEnv

app = create_app(
    get_monitored_env_class(manager),
    LongHorizonMemoryAction,
    LongHorizonMemoryObservation,
    env_name="long_horizon_memory",
    max_concurrent_envs=1,
)

# --- Serve custom UI if available ---
def mount_custom_ui(app, dist_path):
    # Remove existing /web routes to override default UI (in-place mutation)
    new_routes = [r for r in app.routes if getattr(r, "path", None) != "/web"]
    app.routes.clear()
    app.routes.extend(new_routes)
    print(f"[SERVER] Mounting custom UI from {dist_path}")
    app.mount("/web", StaticFiles(directory=dist_path, html=True), name="custom_web")

# Primary path (Project Root)
ui_dist_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard_dist")
if os.path.exists(ui_dist_path):
    mount_custom_ui(app, ui_dist_path)
else:
    # Fallback 1: Nested in server/dist
    ui_dist_path_alt1 = os.path.join(os.path.dirname(__file__), "dist")
    # Fallback 2: Local dev path
    ui_dist_path_alt2 = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui", "dist")
    
    selected_path = None
    if os.path.exists(ui_dist_path_alt1):
        selected_path = ui_dist_path_alt1
    elif os.path.exists(ui_dist_path_alt2):
        selected_path = ui_dist_path_alt2
        
    if selected_path:
        mount_custom_ui(app, selected_path)
    else:
        print(f"[SERVER] Custom UI dist not found, using default OpenEnv UI")


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
