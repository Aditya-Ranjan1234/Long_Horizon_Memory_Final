import requests
import json
import time

URL = "https://aditya-ranjan1234-long-horizon-memory-env-final.hf.space"

def test_api():
    print("Testing /reset...")
    resp = requests.post(f"{URL}/reset", json={}, timeout=10)
    print("Reset response:", resp.status_code)
    print(resp.json())
    
    print("\nTesting /step...")
    action = {"operation": "noop"}
    resp = requests.post(f"{URL}/step", json=action, timeout=10)
    print("Step response:", resp.status_code)
    print(resp.json())

if __name__ == "__main__":
    test_api()
