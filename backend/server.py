"""
EmbedEye Backend Server Launcher.
Picks a free local port, initializes auth token, outputs the ready JSON line to stdout,
and binds uvicorn strictly to 127.0.0.1.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import socket
import sys
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("embedeye.server")


def find_free_port() -> int:
    """Find and return an available ephemeral TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def main():
    """Main entrypoint for backend server."""
    # Ensure stdout uses UTF-8 and auto-flushes
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

    # Ensure EMBEDEYE_TOKEN exists in environment
    if "EMBEDEYE_TOKEN" not in os.environ or not os.environ["EMBEDEYE_TOKEN"].strip():
        token = secrets.token_hex(16)
        os.environ["EMBEDEYE_TOKEN"] = token
        logger.info("Generated ephemeral EMBEDEYE_TOKEN for session.")
    else:
        token = os.environ["EMBEDEYE_TOKEN"].strip()

    # Find a free port
    port = find_free_port()

    # Print the ready handshake line expected by the Avalonia GUI process
    # Format: {"event":"ready","port":NNNN}
    ready_payload = {"event": "ready", "port": port}
    print(json.dumps(ready_payload), flush=True)

    # Start Uvicorn bound strictly to 127.0.0.1
    # Do not bind to 0.0.0.0
    uvicorn.run(
        "backend.api:app",
        host="127.0.0.1",
        port=port,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    main()
