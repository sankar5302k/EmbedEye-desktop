"""Incremental Image Indexer Module with Strict Staged GPU Memory Management.

This module provides an Object-Oriented pipeline for indexing images into Qdrant:
1. Stage A (OpenCLIP): Visual embeddings (fp16 on CUDA) -> system RAM -> unload.
2. Stage B (Florence-2): Captioning and OCR (fp16 on CUDA) -> system RAM -> unload.
3. Stage C (RapidOCR): CPU-only verification on documents/screenshots (0 VRAM).
4. Stage D (Qdrant): Batch upsert (64 points) + crash recovery via SQLite cache.

OOP Architecture:
- `PhotoMetadataExtractor`: Extracts EXIF timestamps and deterministic UUIDs.
- `PhotoCacheManager`: Data Access Object (DAO) managing SQLite intermediate indexing cache.
- `PhotoCollectionManager`: Manages Qdrant 'photos' collection schema, points, and clearing.
- `Indexer`: Main orchestrator coordinating scanning, staged GPU isolation, and upserts.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import ExifTags, Image
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

import src.config as cfg
from src.gpu_manager import ModelManager, get_model_manager
from src.models import (
    caption_and_ocr_florence,
    clean_ocr_text,
    clear_gpu_memory,
    embed_images_clip,
    is_document_or_screenshot,
    run_rapidocr_on_image,
    safe_load_image,
)

logger = logging.getLogger("photo_search.indexer")

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
    ".heic",
    ".heif",
}

CACHE_DB_PATH = cfg.DATA_DIR / "indexer_cache.db"


# ============================================================================
# 1. PHOTO METADATA EXTRACTOR
# ============================================================================

class PhotoMetadataExtractor:
    """Extracts metadata such as EXIF capture dates and deterministic identifiers."""

    @staticmethod
    def get_deterministic_id(file_path: Path | str) -> str:
        """Generate a deterministic UUID5 based on absolute normalized file path.

        Args:
            file_path: Path to target image.

        Returns:
            Deterministic string UUID5.
        """
        resolved = str(Path(file_path).resolve().as_posix())
        return str(uuid.uuid5(uuid.NAMESPACE_URL, resolved))

    @staticmethod
    def extract_image_date(file_path: Path) -> str:
        """Extract EXIF DateTimeOriginal if available; otherwise fallback to file mtime.

        Args:
            file_path: Path to target image.

        Returns:
            ISO 8601 formatted date string.
        """
        try:
            with Image.open(file_path) as img:
                exif = img.getexif()
                if exif:
                    dt_str = None
                    for tag_id, val in exif.items():
                        tag_name = ExifTags.TAGS.get(tag_id, "")
                        if tag_name in ("DateTimeOriginal", "DateTimeDigitized", "DateTime") and val:
                            dt_str = str(val).strip()
                            break

                    if dt_str:
                        try:
                            clean_dt = dt_str.replace("/", ":")
                            dt = datetime.datetime.strptime(clean_dt[:19], "%Y:%m:%d %H:%M:%S")
                            return dt.isoformat()
                        except Exception:
                            pass
        except Exception:
            pass

        try:
            mtime = file_path.stat().st_mtime
            return datetime.datetime.fromtimestamp(mtime).isoformat()
        except Exception:
            return datetime.datetime.now().isoformat()


# Backwards compatibility facades
def get_deterministic_id(file_path: Path | str) -> str:
    """Facade for PhotoMetadataExtractor.get_deterministic_id."""
    return PhotoMetadataExtractor.get_deterministic_id(file_path)


def extract_image_date(file_path: Path) -> str:
    """Facade for PhotoMetadataExtractor.extract_image_date."""
    return PhotoMetadataExtractor.extract_image_date(file_path)


# ============================================================================
# 2. QDRANT CLIENT SINGLETON & PHOTO COLLECTION MANAGER
# ============================================================================

_CLIENT_CACHE: Dict[str, QdrantClient] = {}


def get_qdrant_client(storage_path: Optional[Path] = None) -> QdrantClient:
    """Initialize local Qdrant client as a cached singleton per directory path.

    Args:
        storage_path: Optional custom path for local Qdrant storage.

    Returns:
        QdrantClient instance.
    """
    path = (storage_path or cfg.QDRANT_DIR).resolve()
    path.mkdir(parents=True, exist_ok=True)
    key = str(path)
    if key in _CLIENT_CACHE:
        cl = _CLIENT_CACHE[key]
        if getattr(getattr(cl, "_client", None), "closed", False):
            _CLIENT_CACHE.pop(key, None)
    if key not in _CLIENT_CACHE:
        _CLIENT_CACHE[key] = QdrantClient(path=key)
    return _CLIENT_CACHE[key]


def close_qdrant_client(storage_path: Optional[Path] = None) -> None:
    """Close and remove cached Qdrant client instance.

    Args:
        storage_path: Optional storage path to close.
    """
    path = (storage_path or cfg.QDRANT_DIR).resolve()
    key = str(path)
    client = _CLIENT_CACHE.pop(key, None)
    if client is not None:
        try:
            client.close()
        except Exception:
            pass


class PhotoCollectionManager:
    """Manages Qdrant 'photos' collection schema, retrieval, and point deletions."""

    def __init__(self, client: QdrantClient, collection_name: str = cfg.QDRANT_COLLECTION):
        """Initialize collection manager.

        Args:
            client: QdrantClient instance.
            collection_name: Qdrant collection name.
        """
        self.client = client
        self.collection_name = collection_name
        self.ensure_collection()

    def ensure_collection(self) -> None:
        """Ensure Qdrant collection exists with named vector 'clip' and payload index on 'ocr_text'."""
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            logger.info(f"Creating Qdrant collection '{self.collection_name}' with 'clip' vector...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "clip": qmodels.VectorParams(
                        size=cfg.CLIP_DIM,
                        distance=qmodels.Distance.COSINE,
                    ),
                },
            )

        # Ensure text payload index on ocr_text exists for fast keyword filtering
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="ocr_text",
                field_schema=qmodels.TextIndexParams(
                    type=qmodels.TextIndexType.TEXT,
                    tokenizer=qmodels.TokenizerType.WORD,
                    lowercase=True,
                ),
            )
        except Exception as e:
            logger.debug(f"Payload index creation note: {e}")

    def get_existing_indexed_files(self) -> Dict[str, Tuple[str, float]]:
        """Retrieve mapping of {normalized_abs_path: (point_id, mtime)} from existing collection.

        Returns:
            Dictionary mapping normalized file paths to (point_id, mtime).
        """
        self.ensure_collection()
        existing_map: Dict[str, Tuple[str, float]] = {}

        offset = None
        while True:
            records, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=250,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for r in records:
                if r.payload and "path" in r.payload:
                    p = Path(r.payload["path"]).resolve().as_posix()
                    mtime = float(r.payload.get("mtime", 0.0))
                    existing_map[p] = (str(r.id), mtime)

            if next_offset is None:
                break
            offset = next_offset

        return existing_map

    def clear(self) -> int:
        """Delete all points from the photo collection.

        Returns:
            Number of points removed.
        """
        points_removed = 0
        try:
            existing = [c.name for c in self.client.get_collections().collections]
            if self.collection_name in existing:
                records, _ = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=10000,
                    with_payload=False,
                    with_vectors=False,
                )
                points_removed = len(records)
                point_ids = [r.id for r in records]
                if point_ids:
                    self.client.delete(
                        collection_name=self.collection_name,
                        points_selector=qmodels.PointIdsList(points=point_ids),
                    )
        except Exception as e:
            logger.warning(f"Error checking/deleting points from Qdrant collection: {e}")

        self.ensure_collection()
        return points_removed


def ensure_collection(client: QdrantClient, collection_name: str = cfg.QDRANT_COLLECTION) -> None:
    """Facade for ensuring photo collection exists."""
    PhotoCollectionManager(client, collection_name)


def get_existing_indexed_files(
    client: QdrantClient, collection_name: str = cfg.QDRANT_COLLECTION
) -> Dict[str, Tuple[str, float]]:
    """Facade for retrieving existing indexed files."""
    return PhotoCollectionManager(client, collection_name).get_existing_indexed_files()


# ============================================================================
# 3. PHOTO CACHE MANAGER (DAO)
# ============================================================================

class PhotoCacheManager:
    """Data Access Object managing SQLite intermediate indexing checkpoint state."""

    def __init__(self, db_path: Path = CACHE_DB_PATH):
        """Initialize cache manager and create tables.

        Args:
            db_path: Path to SQLite cache file.
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database for persisting intermediate indexing stages."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS indexing_cache (
                    path TEXT PRIMARY KEY,
                    mtime REAL,
                    clip_vector BLOB,
                    caption TEXT,
                    ocr_text TEXT,
                    has_text INTEGER,
                    stage_a_done INTEGER DEFAULT 0,
                    stage_b_done INTEGER DEFAULT 0,
                    stage_c_done INTEGER DEFAULT 0,
                    stage_d_done INTEGER DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def get_connection(self) -> sqlite3.Connection:
        """Obtain a new SQLite connection.

        Returns:
            sqlite3.Connection instance.
        """
        return sqlite3.connect(str(self.db_path))

    def clear(self) -> int:
        """Wipe all cached indexing records.

        Returns:
            Count of deleted cache records.
        """
        cache_entries_removed = 0
        if self.db_path.exists():
            try:
                with self.get_connection() as conn:
                    cur = conn.execute("SELECT COUNT(*) FROM indexing_cache")
                    cache_entries_removed = cur.fetchone()[0]
                    conn.execute("DELETE FROM indexing_cache")
            except Exception as e:
                logger.warning(f"Error resetting SQLite cache: {e}")
        return cache_entries_removed


def init_cache_db() -> sqlite3.Connection:
    """Facade for obtaining cache database connection."""
    return PhotoCacheManager(CACHE_DB_PATH).get_connection()


def clear_database(
    client: Optional[QdrantClient] = None, storage_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Clear all records from Qdrant vector collection and reset intermediate SQLite cache.

    Args:
        client: Optional QdrantClient instance.
        storage_path: Optional custom storage path for Qdrant.

    Returns:
        Status dictionary with deletion counts.
    """
    q_client = client or get_qdrant_client(storage_path)
    coll_mgr = PhotoCollectionManager(q_client, cfg.QDRANT_COLLECTION)
    points_removed = coll_mgr.clear()

    cache_mgr = PhotoCacheManager(CACHE_DB_PATH)
    cache_entries_removed = cache_mgr.clear()

    return {
        "success": True,
        "points_removed": points_removed,
        "cache_entries_removed": cache_entries_removed,
        "message": f"Successfully cleared database: {points_removed} vector points and {cache_entries_removed} cache records removed.",
    }


# ============================================================================
# 4. MAIN IMAGE INDEXER PIPELINE
# ============================================================================

class Indexer:
    """Orchestrates staged image indexing with strict GPU isolation and crash recovery."""

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        """Initialize Indexer with storage path and hardware device.

        Args:
            storage_path: Optional storage directory path for local Qdrant.
            device: Hardware device identifier ('cuda' or 'cpu').
        """
        self.device = device or cfg.get_device()
        self.client = get_qdrant_client(storage_path)
        self.coll_mgr = PhotoCollectionManager(self.client, cfg.QDRANT_COLLECTION)
        self.cache_mgr = PhotoCacheManager(CACHE_DB_PATH)
        self.manager = get_model_manager(device=self.device)

    def scan_directory(self, folder_path: Path | str) -> List[Path]:
        """Scan folder recursively for all valid image files.

        Args:
            folder_path: Directory path to scan.

        Returns:
            Sorted list of Path objects for supported image files.
        """
        root = Path(folder_path).resolve()
        if not root.is_dir():
            raise ValueError(f"Directory not found: {folder_path}")

        image_files: List[Path] = []
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    image_files.append(Path(dirpath) / filename)
        return sorted(image_files)

    def index_directory(
        self,
        folder_path: Path | str,
        recaption_only: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """Run 4-stage incremental indexing with strict GPU memory isolation.

        Args:
            folder_path: Directory containing images to index.
            recaption_only: If True, re-run caption and OCR without re-embedding CLIP vectors.
            progress_callback: Optional callback reporting (current_step, total_steps, message).

        Returns:
            Dictionary containing indexing metrics, processed counts, and peak VRAM per stage.
        """
        root = Path(folder_path).resolve()
        logger.info(
            f"Starting {'recaptioning' if recaption_only else 'indexing'} of folder: {root} "
            f"on device: {self.device}"
        )

        existing_map = self.coll_mgr.get_existing_indexed_files()
        all_disk_files = self.scan_directory(root)
        current_disk_paths: Set[str] = {f.resolve().as_posix() for f in all_disk_files}

        to_process: List[Path] = []
        skipped_count = 0

        for f in all_disk_files:
            posix_path = f.resolve().as_posix()
            try:
                mtime = f.stat().st_mtime
            except OSError:
                continue

            if recaption_only:
                if posix_path in existing_map:
                    to_process.append(f)
                continue

            if posix_path in existing_map:
                existing_id, existing_mtime = existing_map[posix_path]
                if abs(existing_mtime - mtime) < 0.001:
                    skipped_count += 1
                    continue
            to_process.append(f)

        # Handle file deletions if not in recaption mode
        points_to_remove: List[str] = []
        if not recaption_only:
            root_posix = root.as_posix()
            for indexed_path, (point_id, _) in existing_map.items():
                if indexed_path.startswith(root_posix) and indexed_path not in current_disk_paths:
                    points_to_remove.append(point_id)

            if points_to_remove:
                logger.info(f"Removing {len(points_to_remove)} deleted image points from Qdrant...")
                self.client.delete(
                    collection_name=cfg.QDRANT_COLLECTION,
                    points_selector=qmodels.PointIdsList(points=points_to_remove),
                )

        total_to_process = len(to_process)
        logger.info(
            f"Found {len(all_disk_files)} images: {total_to_process} to process, "
            f"{skipped_count} skipped, {len(points_to_remove)} removed."
        )

        if total_to_process == 0:
            if progress_callback:
                progress_callback(1, 1, "Index up-to-date. No new images to process.")
            dev_stats = self.manager.get_vram_stats()
            return {
                "total_scanned": len(all_disk_files),
                "new_or_updated": 0,
                "skipped": skipped_count,
                "failed": 0,
                "removed": len(points_to_remove),
                "vram_allocated_mb": dev_stats["allocated_mb"],
                "peak_vram_stages": {},
                "device": self.device,
            }

        # Filter out corrupt/unreadable files before deep model stages
        valid_files: List[Path] = []
        failed_count = 0
        for f in to_process:
            img = safe_load_image(f)
            if img is not None:
                valid_files.append(f)
            else:
                logger.warning(f"Skipping corrupt image: {f}")
                failed_count += 1

        total_valid = len(valid_files)
        stage_peak_vram: Dict[str, float] = {}
        db_conn = self.cache_mgr.get_connection()

        # Check existing cached progress from SQLite
        cached_records: Dict[str, Dict[str, Any]] = {}
        with db_conn:
            cursor = db_conn.cursor()
            for f in valid_files:
                posix_p = f.resolve().as_posix()
                mtime = f.stat().st_mtime
                cursor.execute(
                    "SELECT mtime, clip_vector, caption, ocr_text, has_text, stage_a_done, stage_b_done, stage_c_done "
                    "FROM indexing_cache WHERE path = ?",
                    (posix_p,),
                )
                row = cursor.fetchone()
                if row and abs(row[0] - mtime) < 0.001:
                    cached_records[posix_p] = {
                        "clip_vector": np.frombuffer(row[1], dtype=np.float32) if row[1] else None,
                        "caption": row[2] or "",
                        "ocr_text": row[3] or "",
                        "has_text": bool(row[4]),
                        "stage_a_done": bool(row[5]),
                        "stage_b_done": bool(row[6]),
                        "stage_c_done": bool(row[7]),
                    }

        clip_vectors: Dict[str, np.ndarray] = {}
        florence_results: Dict[str, Dict[str, Any]] = {}

        # -------------------------------------------------------------
        # STAGE A: OpenCLIP Image Embedding
        # -------------------------------------------------------------
        files_needing_clip = []
        if recaption_only:
            for f in valid_files:
                posix_p = f.resolve().as_posix()
                pt_id = PhotoMetadataExtractor.get_deterministic_id(f)
                try:
                    pts = self.client.retrieve(
                        collection_name=cfg.QDRANT_COLLECTION,
                        ids=[pt_id],
                        with_vectors=True,
                    )
                    if pts and pts[0].vector:
                        vec = pts[0].vector
                        if isinstance(vec, dict) and "clip" in vec:
                            clip_vectors[posix_p] = np.array(vec["clip"], dtype=np.float32)
                        elif isinstance(vec, list):
                            clip_vectors[posix_p] = np.array(vec, dtype=np.float32)
                except Exception:
                    pass
                if posix_p not in clip_vectors:
                    files_needing_clip.append(f)
        else:
            for f in valid_files:
                posix_p = f.resolve().as_posix()
                if cached_records.get(posix_p, {}).get("stage_a_done") and cached_records[posix_p]["clip_vector"] is not None:
                    clip_vectors[posix_p] = cached_records[posix_p]["clip_vector"]
                else:
                    files_needing_clip.append(f)

        if files_needing_clip:
            logger.info(f"Stage A (CLIP): Embedding {len(files_needing_clip)} images...")
            self.manager.reset_peak_vram()

            with self.manager.use("clip") as (clip_model, clip_preprocess, _):
                for i in range(0, len(files_needing_clip), cfg.CLIP_BATCH_SIZE):
                    batch_paths = files_needing_clip[i : i + cfg.CLIP_BATCH_SIZE]
                    batch_imgs = [safe_load_image(p) for p in batch_paths]
                    active = [(p, img) for p, img in zip(batch_paths, batch_imgs) if img]

                    if active:
                        imgs = [pair[1] for pair in active]
                        vecs = embed_images_clip(
                            imgs,
                            clip_model,
                            clip_preprocess,
                            device=self.device,
                            batch_size=len(imgs),
                        )
                        with db_conn:
                            for (p, _), v in zip(active, vecs):
                                posix_p = p.resolve().as_posix()
                                clip_vectors[posix_p] = v
                                db_conn.execute(
                                    """
                                    INSERT INTO indexing_cache (path, mtime, clip_vector, stage_a_done)
                                    VALUES (?, ?, ?, 1)
                                    ON CONFLICT(path) DO UPDATE SET
                                        mtime=excluded.mtime,
                                        clip_vector=excluded.clip_vector,
                                        stage_a_done=1
                                    """,
                                    (posix_p, p.stat().st_mtime, v.tobytes()),
                                )

                    if progress_callback:
                        progress_callback(
                            min(i + cfg.CLIP_BATCH_SIZE, len(files_needing_clip)),
                            total_valid,
                            f"Stage A/D: CLIP Visual Embeddings ({min(i + cfg.CLIP_BATCH_SIZE, len(files_needing_clip))}/{len(files_needing_clip)})",
                        )
                stage_peak_vram["Stage A (CLIP)"] = self.manager.vram_report()["peak_mb"]

            logger.info(f"Stage A complete. OpenCLIP unloaded. Peak VRAM: {stage_peak_vram['Stage A (CLIP)']} MB")
        else:
            logger.info("Stage A (CLIP): All vectors retrieved from cache/database.")
            stage_peak_vram["Stage A (CLIP)"] = 0.0

        # -------------------------------------------------------------
        # STAGE B: Florence-2 (Caption + OCR)
        # -------------------------------------------------------------
        files_needing_florence = []
        for f in valid_files:
            posix_p = f.resolve().as_posix()
            if not recaption_only and cached_records.get(posix_p, {}).get("stage_b_done"):
                florence_results[posix_p] = {
                    "caption": cached_records[posix_p]["caption"],
                    "ocr_text": cached_records[posix_p]["ocr_text"],
                    "has_text": cached_records[posix_p]["has_text"],
                }
            else:
                files_needing_florence.append(f)

        if files_needing_florence:
            logger.info(f"Stage B (Florence-2): Captioning & OCR on {len(files_needing_florence)} images...")
            self.manager.reset_peak_vram()

            with self.manager.use("florence") as (florence_proc, florence_model):
                for i in range(0, len(files_needing_florence), cfg.FLORENCE_BATCH_SIZE):
                    batch_paths = files_needing_florence[i : i + cfg.FLORENCE_BATCH_SIZE]
                    batch_imgs = [safe_load_image(p) for p in batch_paths]
                    active = [(p, img) for p, img in zip(batch_paths, batch_imgs) if img]

                    if active:
                        imgs = [pair[1] for pair in active]
                        outputs = caption_and_ocr_florence(
                            imgs,
                            florence_proc,
                            florence_model,
                            device=self.device,
                            batch_size=len(imgs),
                        )
                        with db_conn:
                            for (p, _), item in zip(active, outputs):
                                posix_p = p.resolve().as_posix()
                                florence_results[posix_p] = item
                                db_conn.execute(
                                    """
                                    INSERT INTO indexing_cache (path, mtime, caption, ocr_text, has_text, stage_b_done)
                                    VALUES (?, ?, ?, ?, ?, 1)
                                    ON CONFLICT(path) DO UPDATE SET
                                        mtime=excluded.mtime,
                                        caption=excluded.caption,
                                        ocr_text=excluded.ocr_text,
                                        has_text=excluded.has_text,
                                        stage_b_done=1
                                    """,
                                    (
                                        posix_p,
                                        p.stat().st_mtime,
                                        item["caption"],
                                        item["ocr_text"],
                                        1 if item["has_text"] else 0,
                                    ),
                                )

                    if progress_callback:
                        progress_callback(
                            min(i + cfg.FLORENCE_BATCH_SIZE, len(files_needing_florence)),
                            total_valid,
                            f"Stage B/D: Florence-2 Caption & OCR ({min(i + cfg.FLORENCE_BATCH_SIZE, len(files_needing_florence))}/{len(files_needing_florence)})",
                        )
                stage_peak_vram["Stage B (Florence-2)"] = self.manager.vram_report()["peak_mb"]

            logger.info(f"Stage B complete. Florence-2 unloaded. Peak VRAM: {stage_peak_vram['Stage B (Florence-2)']} MB")
        else:
            logger.info("Stage B (Florence-2): All captions and OCR loaded from cache.")
            stage_peak_vram["Stage B (Florence-2)"] = 0.0

        # -------------------------------------------------------------
        # STAGE C: RapidOCR Verification (CPU Only, 0 VRAM)
        # -------------------------------------------------------------
        logger.info(f"Stage C (RapidOCR): Validating text extraction (USE_RAPIDOCR={cfg.USE_RAPIDOCR})...")
        rapid_updated = 0
        if cfg.USE_RAPIDOCR:
            for idx, f in enumerate(valid_files):
                posix_p = f.resolve().as_posix()
                item = florence_results.get(posix_p, {})
                florence_ocr = item.get("ocr_text", "")

                img = safe_load_image(f)
                if img is None:
                    continue

                if bool(florence_ocr) or is_document_or_screenshot(img):
                    rapid_text, conf = run_rapidocr_on_image(img)
                    if rapid_text and (not florence_ocr or (len(rapid_text) > len(florence_ocr) and conf >= 0.40)):
                        item["ocr_text"] = rapid_text
                        item["has_text"] = True
                        florence_results[posix_p] = item
                        rapid_updated += 1
                        with db_conn:
                            db_conn.execute(
                                "UPDATE indexing_cache SET ocr_text = ?, has_text = 1, stage_c_done = 1 WHERE path = ?",
                                (rapid_text, posix_p),
                            )

                if progress_callback and (idx % 10 == 0 or idx == total_valid - 1):
                    progress_callback(
                        idx + 1,
                        total_valid,
                        f"Stage C/D: RapidOCR CPU Verification ({idx + 1}/{total_valid})",
                    )

        stage_peak_vram["Stage C (RapidOCR)"] = 0.0
        logger.info(f"Stage C complete. {rapid_updated} items updated with RapidOCR.")

        # -------------------------------------------------------------
        # STAGE D: Qdrant Batch Upsert (Batches of 64)
        # -------------------------------------------------------------
        logger.info("Stage D (Qdrant): Preparing payload and vectors for upsert...")
        info = self.client.get_collection(cfg.QDRANT_COLLECTION)
        has_text_dim = False
        if isinstance(info.config.params.vectors, dict) and "text" in info.config.params.vectors:
            has_text_dim = True

        points: List[qmodels.PointStruct] = []
        for f in valid_files:
            posix_p = f.resolve().as_posix()
            clip_vec = clip_vectors.get(posix_p)
            f_data = florence_results.get(posix_p, {})

            if clip_vec is None:
                failed_count += 1
                continue

            stat = f.stat()
            date_str = PhotoMetadataExtractor.extract_image_date(f)
            point_id = PhotoMetadataExtractor.get_deterministic_id(f)

            payload = {
                "path": str(f.resolve()),
                "filename": f.name,
                "caption": f_data.get("caption", ""),
                "ocr_text": f_data.get("ocr_text", ""),
                "has_text": bool(f_data.get("has_text", False)),
                "date": date_str,
                "mtime": float(stat.st_mtime),
                "size": int(stat.st_size),
            }

            vec_dict: Dict[str, Any] = {"clip": clip_vec.tolist()}
            if has_text_dim:
                vec_dict["text"] = [0.0] * 384

            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector=vec_dict,
                    payload=payload,
                )
            )

        batch_upsert_size = 64
        for i in range(0, len(points), batch_upsert_size):
            chunk = points[i : i + batch_upsert_size]
            self.client.upsert(
                collection_name=cfg.QDRANT_COLLECTION,
                points=chunk,
            )
            with db_conn:
                for pt in chunk:
                    p_path = pt.payload.get("path")
                    if p_path:
                        db_conn.execute(
                            "UPDATE indexing_cache SET stage_d_done = 1 WHERE path = ?",
                            (Path(p_path).resolve().as_posix(),),
                        )

            if progress_callback:
                progress_callback(
                    min(i + batch_upsert_size, len(points)),
                    len(points),
                    f"Stage D/D: Upserting to Database ({min(i + batch_upsert_size, len(points))}/{len(points)})",
                )

        stage_peak_vram["Stage D (Qdrant)"] = 0.0
        db_conn.close()
        clear_gpu_memory()

        device_info = cfg.get_device_status_info()
        logger.info(
            f"Indexing completed successfully! Scanned: {len(all_disk_files)}, "
            f"Indexed: {len(points)}, Skipped: {skipped_count}, Failed: {failed_count}."
        )

        return {
            "total_scanned": len(all_disk_files),
            "new_or_updated": len(points),
            "skipped": skipped_count,
            "failed": failed_count,
            "removed": len(points_to_remove),
            "vram_allocated_mb": device_info["vram_allocated_mb"],
            "peak_vram_stages": stage_peak_vram,
            "device": self.device,
        }


# ============================================================================
# 5. CLI ENTRYPOINT
# ============================================================================

def main() -> None:
    """CLI for staged incremental image indexing."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Staged incremental image indexer with strict GPU management")
    parser.add_argument("folder", type=str, nargs="?", default="", help="Path to directory containing images to index")
    parser.add_argument(
        "--recaption",
        action="store_true",
        help="Re-run Florence-2 caption + OCR for already-indexed images without recomputing CLIP vectors",
    )
    parser.add_argument(
        "--clear",
        "--clear-db",
        action="store_true",
        dest="clear_db",
        help="Clear all records from Qdrant vector database and reset SQLite cache",
    )
    args = parser.parse_args()

    if args.clear_db:
        try:
            res = clear_database()
            print("\n" + "=" * 60)
            print("DATABASE CLEARED:")
            print("=" * 60)
            print(f"Points removed from Qdrant : {res['points_removed']}")
            print(f"Cache checkpoints cleared   : {res['cache_entries_removed']}")
            print("Status                     : Success")
            print("=" * 60)
        except Exception as e:
            if "already accessed by another instance" in str(e).lower() or "permission denied" in str(e).lower():
                print("\n" + "=" * 60)
                print("[NOTE] Local database is currently open by a running app.py instance.")
                print("To clear the database while app.py is running, click the 'Clear Database'")
                print("button in the Gradio web UI under the Indexing tab, or stop app.py first.")
                print("=" * 60)
            else:
                raise
        return

    if not args.folder:
        parser.print_help()
        return

    indexer = Indexer()
    res = indexer.index_directory(args.folder, recaption_only=args.recaption)

    print("\n" + "=" * 60)
    print("INDEXING SUMMARY:")
    print("=" * 60)
    print(f"Device used         : {res['device']}")
    print(f"Total scanned       : {res['total_scanned']}")
    print(f"New / Updated       : {res['new_or_updated']}")
    print(f"Skipped (unchanged) : {res['skipped']}")
    print(f"Failed / Unreadable : {res['failed']}")
    print(f"Removed (deleted)   : {res['removed']}")
    print(f"Final VRAM Allocated: {res['vram_allocated_mb']:.1f} MB")
    print("\nPeak VRAM per stage:")
    for stg, vram in res.get("peak_vram_stages", {}).items():
        print(f"  - {stg:22}: {vram:.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
