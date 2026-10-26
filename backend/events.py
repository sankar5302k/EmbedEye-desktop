"""
Thread-safe event bus and WebSocket broadcaster for EmbedEye backend.
Allows worker threads to stream progress, logs, VRAM status, and stage transitions
to connected Avalonia desktop clients in real time.
"""

import asyncio
import json
import logging
import threading
from typing import Any, Dict, List, Optional, Set
from fastapi import WebSocket

logger = logging.getLogger("embedeye.events")


class EventBus:
    """
    Thread-safe event bus distributing events to active WebSockets.
    """

    def __init__(self):
        self._clients: Set[WebSocket] = set()
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the active asyncio event loop for threadsafe dispatch."""
        self._loop = loop

    def register_client(self, websocket: WebSocket):
        """Register a connected WebSocket client."""
        with self._lock:
            self._clients.add(websocket)
            logger.info("Client connected. Total active clients: %d", len(self._clients))

    def unregister_client(self, websocket: WebSocket):
        """Unregister a disconnected WebSocket client."""
        with self._lock:
            self._clients.discard(websocket)
            logger.info("Client disconnected. Remaining active clients: %d", len(self._clients))

    def broadcast_sync(self, event_type: str, data: Optional[Dict[str, Any]] = None):
        """
        Thread-safe broadcast called from worker threads.
        Schedules async transmission onto the FastAPI event loop.
        """
        payload = {"event": event_type, "data": data or {}}
        json_str = json.dumps(payload)

        if not self._loop or self._loop.is_closed():
            return

        asyncio.run_coroutine_threadsafe(self._broadcast_async(json_str), self._loop)

    async def _broadcast_async(self, message: str):
        with self._lock:
            clients_snapshot = list(self._clients)

        dead_clients = []
        for ws in clients_snapshot:
            try:
                await ws.send_text(message)
            except Exception:
                dead_clients.append(ws)

        if dead_clients:
            with self._lock:
                for ws in dead_clients:
                    self._clients.discard(ws)

    # Convenience helper methods
    def emit_job_started(self, job_type: str, folder: str, total_files: int):
        self.broadcast_sync(
            "job_started",
            {"job_type": job_type, "folder": folder, "total_files": total_files},
        )

    def emit_stage_changed(self, name: str, index: int, total: int):
        self.broadcast_sync(
            "stage_changed",
            {"name": name, "index": index, "total": total},
        )

    def emit_progress(
        self,
        done: int,
        total: int,
        current_file: str = "",
        files_per_sec: float = 0.0,
        eta_seconds: float = 0.0,
    ):
        self.broadcast_sync(
            "progress",
            {
                "done": done,
                "total": total,
                "current_file": current_file,
                "files_per_sec": round(files_per_sec, 2),
                "eta_seconds": round(eta_seconds, 1),
            },
        )

    def emit_log(self, level: str, message: str):
        self.broadcast_sync("log", {"level": level.upper(), "message": message})

    def emit_vram(self, allocated_mb: float, reserved_mb: float, total_mb: float, resident: List[str]):
        self.broadcast_sync(
            "vram",
            {
                "allocated": round(allocated_mb, 1),
                "reserved": round(reserved_mb, 1),
                "total": round(total_mb, 1),
                "resident": resident,
            },
        )

    def emit_job_finished(self, summary: Dict[str, Any]):
        self.broadcast_sync("job_finished", {"summary": summary})

    def emit_job_failed(self, message: str):
        self.broadcast_sync("job_failed", {"message": message})

    def emit_job_cancelled(self):
        self.broadcast_sync("job_cancelled", {})


# Global event bus singleton
event_bus = EventBus()
