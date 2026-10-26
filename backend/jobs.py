"""
Single-job runner and background worker thread for EmbedEye.
Guarantees ONE job at a time, strict one-model-on-GPU isolation,
cancellation support via threading.Event, and detailed event streaming.
"""

from __future__ import annotations

import datetime
import logging
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

import src.config as cfg
from backend.events import event_bus
from backend.security import is_subpath, paths_match
from backend.staging import staging_store
from src.gpu_manager import get_model_manager
from src.indexer import (
    CACHE_DB_PATH as PHOTO_CACHE_DB,
    PhotoCacheManager,
    PhotoCollectionManager,
    PhotoMetadataExtractor,
    SUPPORTED_EXTENSIONS as PHOTO_EXTENSIONS,
    get_qdrant_client,
)
from src.models import (
    caption_and_ocr_florence,
    clean_ocr_text,
    clear_gpu_memory,
    embed_images_clip,
    is_document_or_screenshot,
    run_rapidocr_on_image,
    safe_load_image,
)
from src.ocr_engine import (
    SUPPORTED_DOC_EXTS,
    chunk_text,
    extract_pages_from_file,
    extract_text_from_image,
    get_minilm_embedder,
    is_supported_ocr_file,
)
from src.ocr_indexer import (
    CACHE_DB_PATH as OCR_CACHE_DB,
    OCRCacheManager,
    OCRCollectionManager,
    OCRDocumentChunk,
    get_ocr_qdrant_client,
)

logger = logging.getLogger("embedeye.jobs")


class JobBusyError(Exception):
    """Raised when an operation is attempted while another job is actively running."""
    pass


class JobCancelledException(Exception):
    """Raised internally when cancel_event is signaled."""
    pass


@dataclass
class JobState:
    job_id: str
    job_type: str
    folder: str
    started_at: float
    is_running: bool


class JobManager:
    """
    Manages job lifecycle on a dedicated worker thread.
    Enforces concurrency limit = 1 and handles cancellation.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._active_state: Optional[JobState] = None
        self._cancel_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

    def get_active_job(self) -> Optional[Dict[str, Any]]:
        """Return dict describing currently active job, if any."""
        with self._lock:
            if self._active_state and self._active_state.is_running:
                return {
                    "job_id": self._active_state.job_id,
                    "job_type": self._active_state.job_type,
                    "folder": self._active_state.folder,
                    "is_running": True,
                    "elapsed_seconds": round(time.time() - self._active_state.started_at, 1),
                }
            return None

    def cancel_active_job(self) -> bool:
        """Signal cancellation for currently running job."""
        with self._lock:
            if not self._active_state or not self._active_state.is_running:
                return False
            logger.info("Cancellation requested for job: %s", self._active_state.job_id)
            self._cancel_event.set()
            event_bus.emit_log("WARNING", "Cancellation requested by user...")
            return True

    def submit_job(self, job_type: str, folder_path: Path) -> str:
        """
        Submit a new job. Raises JobBusyError if a job is already executing.
        """
        with self._lock:
            if self._active_state and self._active_state.is_running:
                raise JobBusyError(
                    f"A job ({self._active_state.job_type}) is already running on {self._active_state.folder}"
                )

            job_id = str(uuid.uuid4())[:8]
            self._cancel_event.clear()
            self._active_state = JobState(
                job_id=job_id,
                job_type=job_type,
                folder=str(folder_path),
                started_at=time.time(),
                is_running=True,
            )

        thread = threading.Thread(
            target=self._run_job_wrapper,
            args=(job_id, job_type, folder_path),
            daemon=True,
            name=f"JobWorker-{job_id}",
        )
        self._worker_thread = thread
        thread.start()
        return job_id

    def _run_job_wrapper(self, job_id: str, job_type: str, folder_path: Path):
        """Worker thread entrypoint that executes the job with recovery and telemetry."""
        logger.info("Starting job [%s] of type '%s' on %s", job_id, job_type, folder_path)
        manager = get_model_manager()
        self._emit_vram_telemetry()

        try:
            if job_type == "ocr":
                summary = self._execute_ocr_job(folder_path)
            elif job_type == "ocr_caption":
                summary = self._execute_ocr_caption_job(folder_path)
            elif job_type == "store":
                summary = self._execute_store_job(folder_path)
            elif job_type == "delete":
                summary = self._execute_delete_job(folder_path)
            else:
                raise ValueError(f"Unknown job type: {job_type}")

            event_bus.emit_job_finished(summary)
            event_bus.emit_log("INFO", f"Job '{job_type}' finished successfully.")
        except JobCancelledException:
            logger.info("Job [%s] cancelled.", job_id)
            event_bus.emit_job_cancelled()
            event_bus.emit_log("WARNING", f"Job '{job_type}' was cancelled.")
        except Exception as exc:
            msg = str(exc)
            logger.error("Job [%s] failed: %s", job_id, msg, exc_info=True)
            if "database is locked" in msg.lower() or "storage lock" in msg.lower():
                msg = "Another EmbedEye process or CLI command is using the database. Please close other instances."
            event_bus.emit_job_failed(msg)
            event_bus.emit_log("ERROR", f"Job failed: {msg}")
        finally:
            # Guarantee GPU memory release
            clear_gpu_memory()
            self._emit_vram_telemetry()
            with self._lock:
                if self._active_state and self._active_state.job_id == job_id:
                    self._active_state.is_running = False

    def _emit_vram_telemetry(self):
        """Emit current VRAM metrics to clients."""
        try:
            manager = get_model_manager()
            rep = manager.vram_report()
            event_bus.emit_vram(
                allocated_mb=rep.get("allocated_mb", 0.0),
                reserved_mb=rep.get("reserved_mb", 0.0),
                total_mb=rep.get("total_mb", 0.0),
                resident=rep.get("resident_models", []),
            )
        except Exception:
            pass

    def _check_cancelled(self):
        if self._cancel_event.is_set():
            raise JobCancelledException("Job cancelled by user request.")

    # =========================================================================
    # ACTION 1: OCR (Text Only)
    # =========================================================================
    def _execute_ocr_job(self, folder_path: Path) -> Dict[str, Any]:
        """
        Action: OCR (text only).
        Extracts text from images and PDF pages using configured OCR engine.
        Saves ONLY to staging.db (no Qdrant writes). Skips unchanged files.
        """
        event_bus.emit_log("INFO", f"Scanning files for OCR in {folder_path}...")
        files: List[Path] = []
        for root, _, filenames in os.walk(folder_path):
            for fn in filenames:
                ext = Path(fn).suffix.lower()
                if ext in SUPPORTED_DOC_EXTS:
                    files.append(Path(root) / fn)

        files.sort()
        total_files = len(files)
        event_bus.emit_job_started("ocr", str(folder_path), total_files)
        event_bus.emit_stage_changed("Document Text Extraction (CPU)", 1, 1)

        engine = cfg.OCR_ENGINE
        if engine not in ("rapidocr", "tesseract"):
            engine = "rapidocr"

        staged_count = 0
        skipped_count = 0
        failed_count = 0
        start_time = time.time()

        for idx, file_path in enumerate(files, start=1):
            self._check_cancelled()
            norm_path = str(file_path.resolve())
            current_file_name = file_path.name

            try:
                stat = file_path.stat()
                mtime = stat.st_mtime
                size = stat.st_size

                # Skip if already staged and mtime unchanged
                if staging_store.is_staged_and_unchanged(norm_path, mtime):
                    skipped_count += 1
                else:
                    pages_text = []
                    for page_num, img in extract_pages_from_file(file_path):
                        self._check_cancelled()
                        t = extract_text_from_image(img, engine=engine)
                        if t and t.strip():
                            pages_text.append(f"[Page {page_num}]\n{t.strip()}")

                    full_ocr = "\n\n".join(pages_text).strip()
                    staging_store.upsert(
                        path=norm_path,
                        mtime=mtime,
                        size=size,
                        mode="ocr",
                        ocr_text=full_ocr,
                        caption=None,
                        extra={"pages": len(pages_text)},
                    )
                    staged_count += 1
            except JobCancelledException:
                raise
            except Exception as e:
                failed_count += 1
                event_bus.emit_log("WARNING", f"Failed OCR for {file_path.name}: {e}")

            # Telemetry update
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            eta = (total_files - idx) / rate if rate > 0 else 0
            event_bus.emit_progress(
                done=idx,
                total=total_files,
                current_file=current_file_name,
                files_per_sec=rate,
                eta_seconds=eta,
            )

        return {
            "action": "ocr",
            "total_files": total_files,
            "staged": staged_count,
            "skipped": skipped_count,
            "failed": failed_count,
        }

    # =========================================================================
    # ACTION 2: OCR + Caption (Florence-2 + RapidOCR)
    # =========================================================================
    def _execute_ocr_caption_job(self, folder_path: Path) -> Dict[str, Any]:
        """
        Action: OCR + Caption.
        Florence-2 caption + OCR under strict one-model-on-GPU isolation.
        Optional RapidOCR verification for documents/screenshots.
        Saves caption + OCR to staging.db with mode='ocr_caption'.
        """
        event_bus.emit_log("INFO", f"Scanning images for OCR + Caption in {folder_path}...")
        files: List[Path] = []
        for root, _, filenames in os.walk(folder_path):
            for fn in filenames:
                ext = Path(fn).suffix.lower()
                if ext in PHOTO_EXTENSIONS:
                    files.append(Path(root) / fn)

        files.sort()
        total_files = len(files)
        event_bus.emit_job_started("ocr_caption", str(folder_path), total_files)

        staged_count = 0
        skipped_count = 0
        failed_count = 0
        start_time = time.time()

        manager = get_model_manager()

        # Filter out files that are already staged and unchanged
        files_to_process: List[Path] = []
        for f in files:
            norm_p = str(f.resolve())
            try:
                mtime = f.stat().st_mtime
                if staging_store.is_staged_and_unchanged(norm_p, mtime):
                    skipped_count += 1
                else:
                    files_to_process.append(f)
            except Exception:
                failed_count += 1

        total_to_process = len(files_to_process)
        event_bus.emit_stage_changed("Florence-2 Caption & OCR (GPU)", 1, 2)
        event_bus.emit_log("INFO", f"Processing {total_to_process} files with Florence-2 (skipped {skipped_count})...")

        # Process in batches of 4 with Florence-2
        batch_size = cfg.FLORENCE_BATCH_SIZE
        processed_entries = []

        if files_to_process:
            self._emit_vram_telemetry()
            with manager.use("florence") as (florence_proc, florence_model):
                self._emit_vram_telemetry()
                for i in range(0, total_to_process, batch_size):
                    self._check_cancelled()
                    batch_files = files_to_process[i : i + batch_size]
                    batch_images = []
                    valid_files = []

                    for bf in batch_files:
                        img = safe_load_image(bf)
                        if img:
                            batch_images.append(img)
                            valid_files.append(bf)
                        else:
                            failed_count += 1

                    if batch_images:
                        outputs = caption_and_ocr_florence(
                            batch_images,
                            florence_proc,
                            florence_model,
                            device=manager.device,
                            batch_size=len(batch_images),
                        )
                        for bf, img, item in zip(valid_files, batch_images, outputs):
                            stat = bf.stat()
                            processed_entries.append({
                                "file_path": bf,
                                "norm_path": str(bf.resolve()),
                                "mtime": stat.st_mtime,
                                "size": stat.st_size,
                                "image": img,
                                "caption": item.get("caption", ""),
                                "florence_ocr": item.get("ocr_text", ""),
                            })

                    done_so_far = skipped_count + min(i + batch_size, total_to_process)
                    elapsed = time.time() - start_time
                    rate = done_so_far / elapsed if elapsed > 0 else 0
                    eta = (total_files - done_so_far) / rate if rate > 0 else 0
                    event_bus.emit_progress(
                        done=done_so_far,
                        total=total_files,
                        current_file=batch_files[-1].name if batch_files else "",
                        files_per_sec=rate,
                        eta_seconds=eta,
                    )

            # Florence unloaded automatically by context manager
            self._emit_vram_telemetry()

        # Stage 2: RapidOCR CPU Verification for documents/screenshots if enabled
        event_bus.emit_stage_changed("RapidOCR Verification (CPU)", 2, 2)
        for item in processed_entries:
            self._check_cancelled()
            bf = item["file_path"]
            img = item["image"]
            florence_ocr = item["florence_ocr"]
            caption = item["caption"]

            final_ocr = florence_ocr
            if getattr(cfg, "USE_RAPIDOCR", True):
                try:
                    rapid_text, _ = run_rapidocr_on_image(img)
                    if len(rapid_text) > len(florence_ocr):
                        final_ocr = rapid_text
                except Exception:
                    pass

            cleaned_ocr = clean_ocr_text(final_ocr)
            staging_store.upsert(
                path=item["norm_path"],
                mtime=item["mtime"],
                size=item["size"],
                mode="ocr_caption",
                ocr_text=cleaned_ocr,
                caption=caption,
            )
            staged_count += 1

        return {
            "action": "ocr_caption",
            "total_files": total_files,
            "staged": staged_count,
            "skipped": skipped_count,
            "failed": failed_count,
        }

    # =========================================================================
    # ACTION 3: Store in DB
    # =========================================================================
    def _execute_store_job(self, folder_path: Path) -> Dict[str, Any]:
        """
        Action: Store in DB.
        Commits staged rows inside folder to Qdrant vector databases:
        - OCR-only rows -> chunk + MiniLM embed (384-d) -> docs_ocr
        - Caption rows -> compute CLIP vectors (512-d) -> photos
        """
        event_bus.emit_log("INFO", f"Querying staged entries for {folder_path}...")
        staged_entries = staging_store.get_entries_in_folder(str(folder_path))
        if not staged_entries:
            raise ValueError(
                "Run OCR or OCR + Caption first. No staged data found for this folder."
            )

        ocr_rows = [e for e in staged_entries if e["mode"] == "ocr"]
        caption_rows = [e for e in staged_entries if e["mode"] == "ocr_caption"]

        total_stages = (1 if ocr_rows else 0) + (1 if caption_rows else 0)
        current_stage = 1

        total_items = len(staged_entries)
        event_bus.emit_job_started("store", str(folder_path), total_items)

        docs_ocr_upserted = 0
        photos_upserted = 0
        start_time = time.time()
        processed_count = 0

        # Sub-process 1: Store OCR-only items into docs_ocr
        if ocr_rows:
            event_bus.emit_stage_changed("Storing Document Text Chunks (docs_ocr)", current_stage, total_stages)
            current_stage += 1
            ocr_client = get_ocr_qdrant_client()
            ocr_coll_mgr = OCRCollectionManager(ocr_client)
            embedder = get_minilm_embedder()

            all_chunks: List[OCRDocumentChunk] = []
            for row in ocr_rows:
                self._check_cancelled()
                p = Path(row["path"])
                text = row["ocr_text"] or ""
                mtime = float(row["file_mtime"])
                date_iso = datetime.datetime.fromtimestamp(mtime, tz=datetime.timezone.utc).isoformat()

                # Chunk full document text
                chunks = chunk_text(text, chunk_words=cfg.CHUNK_WORDS, chunk_overlap=cfg.CHUNK_OVERLAP) if text else []
                for idx, ch_text in enumerate(chunks):
                    chunk_obj = OCRDocumentChunk.create(
                        file_path=str(p.resolve()),
                        filename=p.name,
                        page=1,
                        chunk_index=idx,
                        text=ch_text,
                        full_text_len=len(text),
                        date_iso=date_iso,
                        mtime=mtime,
                    )
                    all_chunks.append(chunk_obj)

                processed_count += 1
                elapsed = time.time() - start_time
                event_bus.emit_progress(
                    done=processed_count,
                    total=total_items,
                    current_file=p.name,
                    files_per_sec=processed_count / elapsed if elapsed > 0 else 0,
                    eta_seconds=(total_items - processed_count) / (processed_count / elapsed) if elapsed > 0 and processed_count > 0 else 0,
                )

            # Embed and upsert chunks in batches
            if all_chunks:
                event_bus.emit_log("INFO", f"Embedding {len(all_chunks)} OCR chunks with MiniLM (CPU)...")
                batch_size = 64
                for i in range(0, len(all_chunks), batch_size):
                    self._check_cancelled()
                    batch = all_chunks[i : i + batch_size]
                    texts = [c.text for c in batch]
                    embeddings = embedder.embed_texts(texts)

                    points = [
                        qmodels.PointStruct(
                            id=c.point_id,
                            vector={"text": emb.tolist()},
                            payload={
                                "path": c.file_path,
                                "filename": c.filename,
                                "page": c.page,
                                "chunk_index": c.chunk_index,
                                "text": c.text,
                                "full_text_len": c.full_text_len,
                                "date": c.date_iso,
                                "mtime": c.mtime,
                            },
                        )
                        for c, emb in zip(batch, embeddings)
                    ]

                    ocr_client.upsert(collection_name=cfg.OCR_COLLECTION, points=points)
                    docs_ocr_upserted += len(points)

        # Sub-process 2: Store Caption items into photos
        if caption_rows:
            event_bus.emit_stage_changed("Computing CLIP Vectors & Storing (photos)", current_stage, total_stages)
            photo_client = get_qdrant_client()
            photo_coll_mgr = PhotoCollectionManager(photo_client)
            manager = get_model_manager()

            batch_size = cfg.CLIP_BATCH_SIZE
            self._emit_vram_telemetry()
            with manager.use("clip") as (clip_model, clip_preprocess, clip_tok):
                self._emit_vram_telemetry()
                for i in range(0, len(caption_rows), batch_size):
                    self._check_cancelled()
                    batch = caption_rows[i : i + batch_size]
                    images = []
                    valid_items = []

                    for r in batch:
                        p = Path(r["path"])
                        img = safe_load_image(p)
                        if img:
                            images.append(img)
                            valid_items.append((r, p, img))

                    if images:
                        embeddings = embed_images_clip(
                            images,
                            clip_model,
                            clip_preprocess,
                            device=manager.device,
                            batch_size=len(images),
                        )
                        points = []
                        for (r, p, img), emb in zip(valid_items, embeddings):
                            det_id = PhotoMetadataExtractor.get_deterministic_id(p)
                            date_iso = PhotoMetadataExtractor.extract_image_date(p)
                            points.append(
                                qmodels.PointStruct(
                                    id=det_id,
                                    vector={"clip": emb.tolist()},
                                    payload={
                                        "path": str(p.resolve().as_posix()),
                                        "filename": p.name,
                                        "caption": r["caption"] or "",
                                        "ocr_text": r["ocr_text"] or "",
                                        "date": date_iso,
                                        "mtime": float(r["file_mtime"]),
                                    },
                                )
                            )

                        photo_client.upsert(
                            collection_name=cfg.QDRANT_COLLECTION,
                            points=points,
                        )
                        photos_upserted += len(points)

                    processed_count += len(batch)
                    elapsed = time.time() - start_time
                    event_bus.emit_progress(
                        done=processed_count,
                        total=total_items,
                        current_file=batch[-1]["path"] if batch else "",
                        files_per_sec=processed_count / elapsed if elapsed > 0 else 0,
                        eta_seconds=(total_items - processed_count) / (processed_count / elapsed) if elapsed > 0 and processed_count > 0 else 0,
                    )

            # CLIP model unloaded cleanly
            self._emit_vram_telemetry()

        return {
            "action": "store",
            "docs_ocr_points": docs_ocr_upserted,
            "photos_points": photos_upserted,
            "total_committed": docs_ocr_upserted + photos_upserted,
        }

    # =========================================================================
    # ACTION 4: Delete from DB
    # =========================================================================
    def _execute_delete_job(self, folder_path: Path) -> Dict[str, Any]:
        """
        Action: Delete from DB.
        Removes every point whose path is inside folder (recursive) from BOTH collections
        ('photos' and 'docs_ocr'), plus its staging and cache rows.
        NEVER deletes image files.
        """
        folder_str = str(folder_path)
        event_bus.emit_log("INFO", f"Searching for points inside {folder_str} to remove...")
        event_bus.emit_job_started("delete", folder_str, 1)
        event_bus.emit_stage_changed("Purging Database Records", 1, 1)
        event_bus.emit_progress(done=0, total=1, current_file="Scanning database records...", files_per_sec=0.0, eta_seconds=0.0)

        # 1. Delete from photos collection
        photo_client = get_qdrant_client()
        photos_deleted = 0
        try:
            offset = None
            ids_to_delete = []
            while True:
                self._check_cancelled()
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
                            ids_to_delete.append(r.id)

                if offset is None:
                    break

            if ids_to_delete:
                for i in range(0, len(ids_to_delete), 100):
                    batch = ids_to_delete[i : i + 100]
                    photo_client.delete(
                        collection_name=cfg.QDRANT_COLLECTION,
                        points_selector=qmodels.PointIdsList(points=batch),
                    )
                photos_deleted = len(ids_to_delete)
                event_bus.emit_log("INFO", f"Deleted {photos_deleted} points from 'photos'.")
        except JobCancelledException:
            raise
        except Exception as e:
            event_bus.emit_log("WARNING", f"Error scrolling/deleting photos collection: {e}")

        # 2. Delete from docs_ocr collection
        ocr_client = get_ocr_qdrant_client()
        ocr_deleted = 0
        try:
            offset = None
            ocr_ids_to_delete = []
            while True:
                self._check_cancelled()
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
                            ocr_ids_to_delete.append(r.id)

                if offset is None:
                    break

            if ocr_ids_to_delete:
                for i in range(0, len(ocr_ids_to_delete), 100):
                    batch = ocr_ids_to_delete[i : i + 100]
                    ocr_client.delete(
                        collection_name=cfg.OCR_COLLECTION,
                        points_selector=qmodels.PointIdsList(points=batch),
                    )
                ocr_deleted = len(ocr_ids_to_delete)
                event_bus.emit_log("INFO", f"Deleted {ocr_deleted} points from 'docs_ocr'.")
        except JobCancelledException:
            raise
        except Exception as e:
            event_bus.emit_log("WARNING", f"Error scrolling/deleting docs_ocr collection: {e}")

        # 3. Delete from staging.db
        staged_deleted = staging_store.delete_in_folder(folder_str)

        # 4. Clean up SQLite caches if files are inside folder
        cache_cleaned = 0
        try:
            if PHOTO_CACHE_DB.exists():
                with sqlite3.connect(str(PHOTO_CACHE_DB)) as conn:
                    cur = conn.cursor()
                    cur.execute("SELECT path FROM intermediate_cache")
                    rows = cur.fetchall()
                    paths_to_del = [r[0] for r in rows if is_subpath(r[0], folder_str)]
                    if paths_to_del:
                        cur.executemany("DELETE FROM intermediate_cache WHERE path = ?", [(p,) for p in paths_to_del])
                        conn.commit()
                        cache_cleaned += len(paths_to_del)
        except Exception as e:
            logger.debug(f"Cache cleanup note: {e}")

        try:
            if OCR_CACHE_DB.exists():
                with sqlite3.connect(str(OCR_CACHE_DB)) as conn:
                    cur = conn.cursor()
                    cur.execute("SELECT file_path FROM ocr_files")
                    rows = cur.fetchall()
                    paths_to_del = [r[0] for r in rows if is_subpath(r[0], folder_str)]
                    if paths_to_del:
                        cur.executemany("DELETE FROM ocr_files WHERE file_path = ?", [(p,) for p in paths_to_del])
                        cur.executemany("DELETE FROM ocr_chunks WHERE file_path = ?", [(p,) for p in paths_to_del])
                        conn.commit()
                        cache_cleaned += len(paths_to_del)
        except Exception as e:
            logger.debug(f"OCR cache cleanup note: {e}")

        event_bus.emit_progress(done=1, total=1, current_file="Purge complete", files_per_sec=0.0, eta_seconds=0.0)

        return {
            "action": "delete",
            "photos_deleted": photos_deleted,
            "docs_ocr_deleted": ocr_deleted,
            "staged_deleted": staged_deleted,
            "cache_entries_removed": cache_cleaned,
        }


# Global job manager singleton
job_manager = JobManager()
