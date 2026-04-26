import os
import sys

# Add V2 to path to use its openenv client if needed
sys.path.insert(0, r"d:\6th Sem\scaler\Long_Horizon_Memory_V2")

import requests
import time

URL = "https://aditya-ranjan1234-long-horizon-memory-env-final.hf.space"

def test_env():
    print(f"Testing {URL}/health")
    for _ in range(30):
        try:
            resp = requests.get(f"{URL}/health", timeout=10)
            if resp.status_code == 200:
                print("✅ Health check passed:", resp.json())
                break
            else:
                print(f"Waiting for HF space... status {resp.status_code}")
        except Exception as e:
            print(f"Waiting for HF space... error {e}")
        time.sleep(5)
    
    print(f"\nTesting {URL}/routes")
    try:
        resp = requests.get(f"{URL}/routes", timeout=10)
        print("Routes:", resp.json())
    except Exception as e:
        print(f"Error getting routes: {e}")

if __name__ == "__main__":
    test_env()
