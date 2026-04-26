#!/usr/bin/env python3
"""
Hugging Face Spaces entry point for Long Horizon Memory Environment.

This file serves as the main entry point for Hugging Face Spaces deployment.
It directly runs the FastAPI server with Gradio integration.
"""

import os
import sys

# Add the current directory to Python path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # Import the server app directly
    from server.app import app
    import uvicorn
    
    # Get port from environment variable (HF Spaces typically uses 7860)
    port = int(os.environ.get("PORT", 7860))
    
    # Run the FastAPI server directly
    # This will automatically serve the OpenEnv web interface
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )
