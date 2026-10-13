#!/usr/bin/env python
"""
run_server.py -- SpeechMirror Unified Server Launcher.

Starts the FastAPI backend and serves the compiled React dashboard
on a single port (http://localhost:8000).

Usage:
    python run_server.py [--host 0.0.0.0] [--port 8000] [--open-browser]
"""

import os
import sys
import argparse
import webbrowser
import subprocess
from pathlib import Path

# Ensure project root is on PYTHONPATH and local ffmpeg is on PATH
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

local_ffmpeg = PROJECT_ROOT / "ffmpeg.exe"
if local_ffmpeg.exists():
    os.environ["PATH"] = f"{str(PROJECT_ROOT)}{os.pathsep}{os.environ.get('PATH', '')}"


def check_prerequisites():
    """Verify configs, processed data, and build status."""
    config_file = PROJECT_ROOT / "configs" / "pipeline.json"
    demo_file = PROJECT_ROOT / "data" / "demo" / "demo_results.json"
    build_dir = PROJECT_ROOT / "app" / "dashboard" / "build"

    print("=" * 60)
    print(" SpeechMirror Contrastive Speech Analytics Server")
    print("=" * 60)
    print(f"[*] Project root: {PROJECT_ROOT}")
    print(f"[*] Pipeline config: {'FOUND' if config_file.exists() else 'MISSING'}")
    print(f"[*] Demo assets: {'FOUND' if demo_file.exists() else 'MISSING'}")
    print(f"[*] Dashboard build: {'READY' if build_dir.exists() else 'NOT BUILT (Run npm run build)'}")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run SpeechMirror Unified Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--open-browser", action="store_true", help="Open default browser on start")
    args = parser.parse_args()

    check_prerequisites()

    url = f"http://localhost:{args.port}"
    print(f"[+] Access Dashboard at: {url}")
    print(f"[+] API Documentation:   {url}/docs")
    print(f"[+] API Health check:    {url}/health")
    print("=" * 60)

    if args.open_browser:
        webbrowser.open(url)

    import uvicorn
    uvicorn.run(
        "app.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
