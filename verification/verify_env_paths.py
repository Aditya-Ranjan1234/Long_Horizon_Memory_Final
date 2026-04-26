import asyncio
import json
import httpx
import websockets
import subprocess
import time
import os
import signal

async def test_env_paths():
    base_url = "http://localhost:8000"
    ws_url = "ws://localhost:8000/ws/monitor"
    
    print("--- 1. Testing Health Check ---")
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{base_url}/health")
            print(f"Health: {resp.status_code} - {resp.json()}")
        except Exception as e:
            print(f"Health check failed: {e}")
            return

    print("\n--- 2. Testing Reset ---")
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{base_url}/reset")
        print(f"Reset: {resp.status_code}")
        obs = resp.json()
        print(f"Initial Observation: {obs.get('domain')} - {obs.get('new_message')[:50]}...")

    print("\n--- 3. Testing WebSocket Connection ---")
    try:
        async with websockets.connect(ws_url) as ws:
            print("WebSocket: Connected")
            
            print("\n--- 4. Testing Step with Broadcast ---")
            async with httpx.AsyncClient() as client:
                action = {"operation": "add"}
                resp = await client.post(f"{base_url}/step", json=action)
                print(f"Step: {resp.status_code}")
                
            # Wait for broadcast
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(msg)
                print(f"Broadcast Received: {data.get('operation')} - Step {data.get('step')}")
                if data.get('operation') == 'add':
                    print("SUCCESS: Broadcast verified")
            except asyncio.TimeoutError:
                print("FAILURE: No broadcast received via WebSocket")
                
    except Exception as e:
        print(f"WebSocket test failed: {e}")

if __name__ == "__main__":
    # Start server
    print("Starting local server for testing...")
    # Make sure we are in the right directory so imports work
    cwd = "d:/6th Sem/scaler/Long_Horizon_Memory_Final"
    env = os.environ.copy()
    env["PYTHONPATH"] = cwd
    
    uvicorn_path = "d:/6th Sem/scaler/Long_Horizon_Memory_V2/venv/Scripts/uvicorn.exe"
    server_proc = subprocess.Popen(
        [uvicorn_path, "server.app:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
    )
    
    time.sleep(5) # Wait for server to start
    
    try:
        asyncio.run(test_env_paths())
    finally:
        print("\nShutting down server...")
        if os.name == 'nt':
            subprocess.run(['taskkill', '/F', '/T', '/PID', str(server_proc.pid)])
        else:
            os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
