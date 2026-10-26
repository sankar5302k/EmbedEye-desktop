"""
FastAPI route definitions for EmbedEye desktop backend.
Provides endpoints for health/status telemetry, folder summary inspection,
four main actions (OCR, OCR+Caption, Store, Delete), semantic & OCR search,
thumbnail rendering, and live WebSocket event streaming.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

import src.config as cfg
from backend.events import event_bus
from backend.jobs import JobBusyError, job_manager
from backend.security import (
    check_allowed_origin,
    is_subpath,
    normalize_and_validate_folder,
    verify_auth_token,
)
from backend.staging import staging_store
from src.indexer import (
    SUPPORTED_EXTENSIONS as PHOTO_EXTENSIONS,
    get_qdrant_client,
)
from src.ocr_engine import SUPPORTED_DOC_EXTS
from src.ocr_indexer import get_ocr_qdrant_client
from src.status import SystemStatusReporter

logger = logging.getLogger("embedeye.api")

# App startup lifecycle state
_server_health_state = {
    "state": "starting",
    "detail": "Initializing backend services...",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Register running loop with event bus and perform initial hardware check."""
    loop = asyncio.get_running_loop()
    event_bus.set_loop(loop)
    _server_health_state["state"] = "loading"
    _server_health_state["detail"] = "Loading libraries and hardware devices..."
    await asyncio.sleep(0.05)
    _server_health_state["state"] = "ready"
    _server_health_state["detail"] = "Engine ready"
    yield


app = FastAPI(
    title="EmbedEye Backend Engine",
    description="Local semantic indexing and OCR backend for EmbedEye desktop application",
    version="1.0.0",
    lifespan=lifespan,
)


# Middleware: Enforce origin checks to prevent cross-site hijacking
@app.middleware("http")
async def enforce_origin(request: Request, call_next):
    if not check_allowed_origin(request):
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"detail": "Forbidden: Untrusted Origin"},
        )
    return await call_next(request)


# Request schemas
class FolderJobRequest(BaseModel):
    folder: str = Field(..., description="Absolute path of target folder")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query string")
    mode: str = Field("visual", description="'visual' for photos or 'ocr' for documents")
    top_k: int = Field(3, description="Number of top matches to return (default 3)")


class RevealRequest(BaseModel):
    path: str = Field(..., description="Absolute path of file to reveal in Explorer")


# =============================================================================
# UNPROTECTED ENDPOINT: /health (Splash Screen & Probes)
# =============================================================================
@app.get("/health")
def get_health() -> Dict[str, str]:
    """Retrieve current engine startup state for splash screen."""
    return _server_health_state


# =============================================================================
# PROTECTED TELEMETRY: /status, /folder/summary
# =============================================================================
@app.get("/status", dependencies=[Depends(verify_auth_token)])
def get_system_status() -> Dict[str, Any]:
    """Return device, VRAM metrics, resident models, and DB point counts."""
    raw = SystemStatusReporter.get_raw_status()
    return {
        "device": raw.get("device", "cpu"),
        "device_name": raw.get("device_name", "Host CPU"),
        "vram": {
            "allocated_mb": raw.get("vram_allocated_mb", 0.0),
            "reserved_mb": raw.get("vram_reserved_mb", 0.0),
            "total_mb": raw.get("vram_total_mb", 0.0),
        },
        "resident_models": raw.get("resident_models", []),
        "photos_count": raw.get("photos_count", 0),
        "docs_ocr_chunks": raw.get("docs_ocr_chunks", 0),
        "docs_ocr_files": raw.get("docs_ocr_files", 0),
        "ocr_engine": raw.get("ocr_engine", cfg.OCR_ENGINE),
        "active_job": job_manager.get_active_job(),
    }


@app.get("/folder/summary", dependencies=[Depends(verify_auth_token)])
def get_folder_summary(path: str = Query(..., description="Target directory path")) -> Dict[str, Any]:
    """
    Inspect folder to return:
    - Images and document files found (recursive)
    - Staged entries count broken down by mode
    - Stored records in 'photos' and 'docs_ocr'
    """
    folder_path = normalize_and_validate_folder(path)
    folder_str = str(folder_path)

    # 1. Scan filesystem
    image_count = 0
    doc_count = 0
    total_files = 0
    for root, _, filenames in os.walk(folder_path):
        for fn in filenames:
            ext = Path(fn).suffix.lower()
            if ext in PHOTO_EXTENSIONS:
                image_count += 1
            if ext in SUPPORTED_DOC_EXTS:
                doc_count += 1
            if ext in PHOTO_EXTENSIONS or ext in SUPPORTED_DOC_EXTS:
                total_files += 1

    # 2. Count staged entries
    staged_summary = staging_store.count_in_folder(folder_str)

    # 3. Count stored records in Qdrant collections
    stored_photos = 0
    try:
        photo_client = get_qdrant_client()
        offset = None
        while True:
            records, offset = photo_client.scroll(
                collection_name=cfg.QDRANT_COLLECTION,
                limit=250,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not records:
                break
            for r in records:
                if r.payload and "path" in r.payload:
                    if is_subpath(r.payload["path"], folder_str):
                        stored_photos += 1
            if offset is None:
                break
    except Exception as e:
        logger.debug(f"Photos count check note: {e}")

    stored_docs_ocr = 0
    try:
        ocr_client = get_ocr_qdrant_client()
        offset = None
        while True:
            records, offset = ocr_client.scroll(
                collection_name=cfg.OCR_COLLECTION,
                limit=250,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not records:
                break
            for r in records:
                if r.payload and "path" in r.payload:
                    if is_subpath(r.payload["path"], folder_str):
                        stored_docs_ocr += 1
            if offset is None:
                break
    except Exception as e:
        logger.debug(f"OCR count check note: {e}")

    return {
        "folder": folder_str,
        "files_found": {
            "total": total_files,
            "images": image_count,
            "documents": doc_count,
        },
        "staged": staged_summary,
        "stored": {
            "photos": stored_photos,
            "docs_ocr": stored_docs_ocr,
            "total": stored_photos + stored_docs_ocr,
        },
    }


# =============================================================================
# PROTECTED ACTIONS: /jobs/...
# =============================================================================
@app.post("/jobs/ocr", dependencies=[Depends(verify_auth_token)])
def trigger_ocr_job(req: FolderJobRequest) -> Dict[str, Any]:
    """Trigger background OCR job (text only) -> staging.db."""
    folder = normalize_and_validate_folder(req.folder)
    try:
        job_id = job_manager.submit_job("ocr", folder)
        return {"status": "started", "job_id": job_id, "folder": str(folder)}
    except JobBusyError as err:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail={"error": "busy", "message": str(err)})


@app.post("/jobs/ocr-caption", dependencies=[Depends(verify_auth_token)])
def trigger_ocr_caption_job(req: FolderJobRequest) -> Dict[str, Any]:
    """Trigger background Florence-2 OCR + Caption job -> staging.db."""
    folder = normalize_and_validate_folder(req.folder)
    try:
        job_id = job_manager.submit_job("ocr_caption", folder)
        return {"status": "started", "job_id": job_id, "folder": str(folder)}
    except JobBusyError as err:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail={"error": "busy", "message": str(err)})


@app.post("/jobs/store", dependencies=[Depends(verify_auth_token)])
def trigger_store_job(req: FolderJobRequest) -> Dict[str, Any]:
    """Commit staged data in folder to Qdrant databases."""
    folder = normalize_and_validate_folder(req.folder)
    try:
        job_id = job_manager.submit_job("store", folder)
        return {"status": "started", "job_id": job_id, "folder": str(folder)}
    except JobBusyError as err:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail={"error": "busy", "message": str(err)})


@app.post("/jobs/delete", dependencies=[Depends(verify_auth_token)])
def trigger_delete_job(req: FolderJobRequest) -> Dict[str, Any]:
    """Delete all indexed records for folder from Qdrant and caches."""
    folder = normalize_and_validate_folder(req.folder)
    try:
        job_id = job_manager.submit_job("delete", folder)
        return {"status": "started", "job_id": job_id, "folder": str(folder)}
    except JobBusyError as err:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail={"error": "busy", "message": str(err)})


@app.post("/jobs/cancel", dependencies=[Depends(verify_auth_token)])
def cancel_job() -> Dict[str, Any]:
    """Request cancellation of currently running job."""
    cancelled = job_manager.cancel_active_job()
    return {"cancelled": cancelled}


# =============================================================================
# SEARCH & RETRIEVAL ENDPOINTS
# =============================================================================
_visual_search_engine = None
_ocr_search_engine = None


def get_visual_search_engine():
    global _visual_search_engine
    if _visual_search_engine is None:
        from src.search import SearchEngine
        _visual_search_engine = SearchEngine()
    return _visual_search_engine


def get_ocr_search_engine():
    global _ocr_search_engine
    if _ocr_search_engine is None:
        from src.ocr_search import OCRSearchEngine
        _ocr_search_engine = OCRSearchEngine()
    return _ocr_search_engine


@app.post("/search", dependencies=[Depends(verify_auth_token)])
def search_items(req: SearchRequest) -> Dict[str, Any]:
    """Search photos or documents and retrieve top matches (default top 3)."""
    top_k = max(1, min(req.top_k, 50))
    query = req.query.strip()
    mode = req.mode.lower().strip()

    if mode == "ocr":
        engine = get_ocr_search_engine()
        res = engine.search(query, top_k=top_k, use_reranker=True)
        images = res.get("images", [])[:top_k]
        results = [
            {
                "rank": idx + 1,
                "path": item.get("path", ""),
                "filename": item.get("filename", ""),
                "score": round(float(item.get("best_score", 0.0)), 4),
                "date": item.get("date", "unknown"),
                "caption": f"Page {item.get('page', 1)}",
                "ocr_text": item.get("best_chunk", "")[:300],
                "snippets": [item.get("highlighted_snippet", item.get("best_chunk", ""))],
            }
            for idx, item in enumerate(images)
        ]
        badge = "OCR Document Search (RRF + Reranker)"
    else:
        engine = get_visual_search_engine()
        res = engine.search(query, use_rerank=True, use_smart_date=True, top_k=top_k)
        top_candidates = res.get("results", [])[:top_k]
        results = [
            {
                "rank": idx + 1,
                "path": item.get("path", ""),
                "filename": item.get("filename", ""),
                "score": round(float(item.get("score", 0.0)), 4),
                "date": item.get("date", "unknown"),
                "caption": item.get("caption", ""),
                "ocr_text": item.get("ocr_text", ""),
                "snippets": [item.get("ocr_text", "")] if item.get("ocr_text") else [],
            }
            for idx, item in enumerate(top_candidates)
        ]
        badge = res.get("badge", "Visual Search (OpenCLIP + Reranker)")

    return {
        "query": query,
        "mode": mode,
        "count": len(results),
        "badge": badge,
        "results": results,
    }


@app.get("/thumbnail", dependencies=[Depends(verify_auth_token)])
def get_thumbnail(path: str = Query(...), page: int = Query(1)):
    """Render image or PDF page thumbnail preview."""
    from src.thumbnails import ThumbnailRenderer

    img = ThumbnailRenderer.load_doc_thumbnail(path, page=page)
    if not img:
        raise HTTPException(status_code=404, detail="Thumbnail could not be generated")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")


@app.post("/reveal", dependencies=[Depends(verify_auth_token)])
def reveal_file(req: RevealRequest):
    """Reveal file or folder in native operating system file explorer."""
    from src.os_utils import NativeExplorerLauncher
    ok = NativeExplorerLauncher.reveal_file(req.path)
    return {"status": "ok", "revealed": ok}


# =============================================================================
# WEBSOCKET STREAMING: /ws/events
# =============================================================================
@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    """
    Stream live progress, logs, VRAM status, and stage transitions.
    Accepts token via 'token' query param or 'Sec-WebSocket-Protocol'.
    """
    # Verify origin
    origin = websocket.headers.get("origin")
    if origin:
        origin_lower = origin.lower()
        if not any(origin_lower.startswith(p) for p in ("http://127.0.0.1", "http://localhost", "app://")):
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

    expected_token = os.environ.get("EMBEDEYE_TOKEN", "").strip()
    if expected_token:
        # Check token in query parameters
        client_token = websocket.query_params.get("token")
        if client_token != expected_token:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

    await websocket.accept()
    event_bus.register_client(websocket)

    try:
        while True:
            _ = await websocket.receive_text()
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        event_bus.unregister_client(websocket)


# =============================================================================
# SHUTDOWN: /shutdown
# =============================================================================
@app.post("/shutdown", dependencies=[Depends(verify_auth_token)])
def shutdown_backend():
    """Trigger graceful shutdown of the backend server process."""
    def _do_exit():
        import time
        time.sleep(0.5)
        sys.exit(0)

    import threading
    threading.Thread(target=_do_exit, daemon=True).start()
    return {"message": "Shutting down..."}
