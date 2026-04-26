import os
import time

# Set env var for testing the final env
os.environ["API_BASE_URL"] = "https://aditya-ranjan1234-long-horizon-memory-env-final.hf.space"

from client import LongHorizonMemoryEnv
from models import LongHorizonMemoryAction

def main():
    print(f"Testing environment at {os.environ['API_BASE_URL']}")
    # Create the client pointing to HF Space
    env = LongHorizonMemoryEnv(
        api_base_url=os.environ["API_BASE_URL"],
        # dummy token just to satisfy if openenv asks
        api_key="dummy_key", 
    )
    
    print("Resetting env...")
    obs = env.reset()
    print("Initial observation:", obs)
    
    print("Taking step (add)...")
    action = LongHorizonMemoryAction(operation="add")
    obs = env.step(action)
    print("Observation after step:", obs)
    
    print("Testing complete!")

if __name__ == "__main__":
    main()
