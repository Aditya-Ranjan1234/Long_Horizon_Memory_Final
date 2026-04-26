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
from typing import List
from fastapi import WebSocket, WebSocketDisconnect, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import iterate_in_threadpool
import os


# ---------------------------------------------------------------------------
# Connection Manager (async-safe)
# ---------------------------------------------------------------------------
class ConnectionManager:
    """
    Manages WebSocket connections and a broadcast queue.

    KEY DESIGN: OpenEnv calls step/reset on the environment from a *sync*
    worker thread (thread-pool executor), NOT the asyncio event loop.
    We therefore cannot call asyncio.Queue.put_nowait() directly from those
    methods — it would enqueue onto whichever loop happens to be running in
    that thread (usually none), silently dropping the message.

    The fix: store a reference to the *main* event loop at startup and use
    loop.call_soon_threadsafe() to safely post items from any thread.
    """

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue | None = None
        self.worker_task = None

    # Called once the lifespan event fires (event loop is running)
    def _init_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._queue = asyncio.Queue()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def thread_safe_put(self, data: dict):
        """Enqueue data from *any* thread into the async broadcast queue."""
        if self._loop is None or self._queue is None:
            print("[SERVER] thread_safe_put: event loop not ready, message dropped")
            return
        self._loop.call_soon_threadsafe(self._queue.put_nowait, data)

    async def broadcast_worker(self):
        print("[SERVER] ConnectionManager v2.0 initialized.")
        print("[SERVER] Broadcast worker started.")
        while True:
            try:
                data = await self._queue.get()
                if "timestamp" not in data:
                    data["timestamp"] = datetime.now().isoformat()
                message = json.dumps(data)
                dead = []
                for ws in list(self.active_connections):
                    try:
                        await ws.send_text(message)
                    except Exception:
                        dead.append(ws)
                for ws in dead:
                    if ws in self.active_connections:
                        self.active_connections.remove(ws)
                self._queue.task_done()
            except Exception as e:
                print(f"[SERVER] Broadcast worker error: {e}")
                await asyncio.sleep(0.1)

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        # Lazily start the broadcast worker
        if not self.worker_task or self.worker_task.done():
            self.worker_task = asyncio.create_task(self.broadcast_worker())
        print(f"[SERVER] WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"[SERVER] WebSocket disconnected. Remaining: {len(self.active_connections)}")


manager = ConnectionManager()


# ---------------------------------------------------------------------------
# Monitored Environment (wraps the real env and broadcasts every step/reset)
# ---------------------------------------------------------------------------
def get_monitored_env_class(mgr: ConnectionManager):
    class MonitoredEnv(LongHorizonMemoryEnvironment):
        def _broadcast(self, obs, action=None):
            try:
                data = obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()
                data["operation"] = action.operation if action else "reset"
                print(f"[SERVER] Broadcasting: {data['operation']}")
                mgr.thread_safe_put(data)
            except Exception as e:
                print(f"[SERVER] _broadcast error: {e}")

        def step(self, action: LongHorizonMemoryAction) -> LongHorizonMemoryObservation:
            print(f"[SERVER] Step: {action.operation}")
            obs = super().step(action)
            self._broadcast(obs, action)
            return obs

        def reset(self) -> LongHorizonMemoryObservation:
            print("[SERVER] Reset")
            obs = super().reset()
            self._broadcast(obs)
            return obs

    return MonitoredEnv


# ---------------------------------------------------------------------------
# Create the OpenEnv FastAPI app
# ---------------------------------------------------------------------------
env_cls = get_monitored_env_class(manager)
print(f"[SERVER] Initializing OpenEnv with monitored class: {env_cls.__name__}")
app = create_app(
    env_cls,
    LongHorizonMemoryAction,
    LongHorizonMemoryObservation,
    env_name="long_horizon_memory",
    max_concurrent_envs=1,
)

# ---------------------------------------------------------------------------
# Initialise the event loop reference on startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def _startup():
    loop = asyncio.get_running_loop()
    manager._init_loop(loop)
    print(f"[SERVER] Event loop captured. Broadcast system ready.")


# ---------------------------------------------------------------------------
# Serve custom dashboard UI
# ---------------------------------------------------------------------------
ui_dist_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard_dist")
if not os.path.exists(ui_dist_path):
    ui_dist_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "dist")

if os.path.exists(ui_dist_path):
    print(f"[SERVER] Mounting custom UI from {ui_dist_path}")
    # Remove any existing /web route so we can replace it
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
# Middleware: secondary broadcast path (intercepts /step and /reset responses)
# ---------------------------------------------------------------------------
@app.middleware("http")
async def broadcast_env_middleware(request: Request, call_next):
    path = request.url.path
    # Match any path ending in /step or /reset (handles OpenEnv prefixes like /env/{id}/step)
    is_step = path.endswith("/step") or path == "/step"
    is_reset = path.endswith("/reset") or path == "/reset"
    if not (is_step or is_reset):
        return await call_next(request)

    operation = "reset" if is_reset else "step"
    print(f"[SERVER] Middleware intercepted: {request.method} {path}")
    response = await call_next(request)

    if response.status_code == 200:
        try:
            response_body = [chunk async for chunk in response.body_iterator]
            response.body_iterator = iterate_in_threadpool(iter(response_body))
            full_body = b"".join(response_body).decode()
            data = json.loads(full_body)
            obs = data.get("observation", data.get("payload", data))
            if isinstance(obs, dict):
                obs["operation"] = operation
                obs["timestamp"] = datetime.now().isoformat()
                print(f"[SERVER] Middleware broadcast: {operation} step={obs.get('step', '?')}")
                manager.thread_safe_put(obs)
        except Exception as e:
            print(f"[SERVER] Middleware broadcast error: {e}")

    return response


# ---------------------------------------------------------------------------
# Utility routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/")
async def root_redirect():
    return RedirectResponse(url="/web/")


@app.post("/telemetry")
async def receive_telemetry(payload: dict):
    """Receive telemetry from external agents running the environment locally."""
    try:
        manager.thread_safe_put(payload)
        return {"status": "ok"}
    except Exception as e:
        print(f"[SERVER] Telemetry Error: {e}")
        return {"status": "error", "message": str(e)}


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
