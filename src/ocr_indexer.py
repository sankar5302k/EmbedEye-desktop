"""OCR-Only Indexer Module for Documents and Images.

This module provides an Object-Oriented pipeline for extracting text from images
and multi-page PDF documents, dividing text into overlapping chunks, generating
384-dimensional dense semantic vectors using a CPU-based MiniLM model, and storing
them into a local Qdrant collection ('docs_ocr') with full-text payload indexing.

OOP Architecture:
- `OCRDocumentChunk`: Data transfer object representing an extracted text chunk.
- `OCRCacheManager`: Data Access Object (DAO) managing SQLite checkpoint persistence.
- `OCRCollectionManager`: Manages Qdrant collection lifecycle and schema.
- `OCRWorker`: Encapsulates document parsing and OCR execution across worker processes.
- `OCRIndexer`: Main orchestrator coordinating scanning, worker dispatch, batching,
  dense embedding generation, and Qdrant upserts.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import logging
import os
import sqlite3
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union

import numpy as np
from PIL import ExifTags, Image
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

import src.config as cfg
from src.ocr_engine import (
    SUPPORTED_DOC_EXTS,
    chunk_text,
    extract_pages_from_file,
    extract_text_from_image,
    get_minilm_embedder,
    is_supported_ocr_file,
)

logger = logging.getLogger("photo_search.ocr_indexer")

CACHE_DB_PATH = cfg.DATA_DIR / "ocr_indexer_cache.db"


# ============================================================================
# 1. DATA TRANSFER OBJECTS (DTOs)
# ============================================================================

@dataclass
class OCRDocumentChunk:
    """Represents a discrete text chunk extracted from a document page.

    Attributes:
        file_path: Absolute normalized path of the source document or image.
        filename: Name of the source file.
        page: 1-indexed page number (page 1 for standard images).
        chunk_index: 0-indexed chunk sequence number on this page.
        text: Extracted text content of this chunk.
        full_text_len: Total character count of full text extracted from this page.
        date_iso: ISO 8601 creation or modification timestamp.
        mtime: Unix timestamp of file last modified time.
        point_id: Deterministic UUID5 string identifying this chunk in Qdrant.
    """
    file_path: str
    filename: str
    page: int
    chunk_index: int
    text: str
    full_text_len: int
    date_iso: str
    mtime: float
    point_id: str

    @classmethod
    def create(
        cls,
        file_path: str,
        filename: str,
        page: int,
        chunk_index: int,
        text: str,
        full_text_len: int,
        date_iso: str,
        mtime: float,
    ) -> OCRDocumentChunk:
        """Factory method to construct an OCRDocumentChunk with deterministic point_id.

        Args:
            file_path: Normalized path to file.
            filename: File basename.
            page: Page number.
            chunk_index: Chunk index within page.
            text: Chunk text.
            full_text_len: Length of full extracted text.
            date_iso: ISO date string.
            mtime: File modification timestamp.

        Returns:
            Instantiated OCRDocumentChunk with UUID5 point_id.
        """
        resolved = f"{Path(file_path).resolve().as_posix()}:{page}:{chunk_index}"
        deterministic_id = str(uuid.uuid5(uuid.NAMESPACE_URL, resolved))
        return cls(
            file_path=file_path,
            filename=filename,
            page=page,
            chunk_index=chunk_index,
            text=text,
            full_text_len=full_text_len,
            date_iso=date_iso,
            mtime=mtime,
            point_id=deterministic_id,
        )

    def to_qdrant_payload(self) -> Dict[str, Any]:
        """Convert chunk metadata to Qdrant storage payload dictionary.

        Returns:
            Dictionary matching Qdrant payload schema for OCR documents.
        """
        return {
            "path": self.file_path,
            "filename": self.filename,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "full_text_len": self.full_text_len,
            "date": self.date_iso,
            "mtime": self.mtime,
        }


# ============================================================================
# 2. UTILITY / HELPER FUNCTIONS & FACADES
# ============================================================================

def get_deterministic_ocr_id(file_path: Path | str, page: int, chunk_index: int) -> str:
    """Generate a deterministic UUID5 based on path + page + chunk_index.

    Args:
        file_path: Path to target image or document.
        page: 1-indexed page number.
        chunk_index: 0-indexed chunk index.

    Returns:
        Deterministic string UUID5.
    """
    resolved = f"{Path(file_path).resolve().as_posix()}:{page}:{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, resolved))


def extract_file_date(file_path: Path) -> str:
    """Extract EXIF DateTimeOriginal if available; otherwise fallback to file mtime.

    Args:
        file_path: Target path to inspect.

    Returns:
        ISO 8601 formatted date string.
    """
    if file_path.suffix.lower() != ".pdf":
        try:
            with Image.open(file_path) as img:
                exif = img.getexif()
                if exif:
                    for tag_id, val in exif.items():
                        tag_name = ExifTags.TAGS.get(tag_id, "")
                        if tag_name in ("DateTimeOriginal", "DateTimeDigitized", "DateTime") and val:
                            dt_str = str(val).strip().replace("/", ":")
                            dt = datetime.datetime.strptime(dt_str[:19], "%Y:%m:%d %H:%M:%S")
                            return dt.isoformat()
        except Exception:
            pass

    try:
        mtime = file_path.stat().st_mtime
        return datetime.datetime.fromtimestamp(mtime).isoformat()
    except Exception:
        return datetime.datetime.now().isoformat()


def get_ocr_qdrant_client(storage_path: Optional[Path] = None) -> QdrantClient:
    """Initialize or reuse local Qdrant client singleton.

    Args:
        storage_path: Optional custom storage path for Qdrant local files.

    Returns:
        Shared QdrantClient instance.
    """
    from src.indexer import get_qdrant_client

    return get_qdrant_client(storage_path)


# ============================================================================
# 3. OCR CACHE MANAGER (DAO)
# ============================================================================

class OCRCacheManager:
    """Data Access Object managing SQLite checkpoint persistence for OCR files.

    Tracks indexed file modification timestamps, chunk counts, and text presence
    to support incremental indexing, skip unchanged files, and detect deletions.
    """

    def __init__(self, db_path: Path = CACHE_DB_PATH):
        """Initialize cache manager and establish database schema.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Create database tables and indices if they do not exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ocr_files (
                    path TEXT PRIMARY KEY,
                    mtime REAL NOT NULL,
                    has_text INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    indexed_at TEXT NOT NULL
                )
                """
            )

    def get_connection(self) -> sqlite3.Connection:
        """Obtain a new SQLite connection with reasonable timeout.

        Returns:
            sqlite3.Connection instance.
        """
        return sqlite3.connect(str(self.db_path), timeout=30.0)

    def get_cached_mtimes(self) -> Dict[str, float]:
        """Fetch all recorded file paths and modification times.

        Returns:
            Dictionary mapping normalized file paths to modification timestamps.
        """
        cached: Dict[str, float] = {}
        with self.get_connection() as conn:
            cur = conn.execute("SELECT path, mtime FROM ocr_files")
            for row in cur.fetchall():
                cached[row[0]] = row[1]
        return cached

    def get_stats(self) -> Tuple[int, int]:
        """Return total tracked files and count of files that contained text.

        Returns:
            Tuple of (total_files, files_with_text).
        """
        with self.get_connection() as conn:
            cur = conn.execute("SELECT COUNT(*), SUM(has_text) FROM ocr_files")
            row = cur.fetchone()
            if row:
                return (row[0] or 0, row[1] or 0)
        return (0, 0)

    def record_files(self, records: List[Tuple[str, float, int, int, str]]) -> None:
        """Insert or replace indexing status records for processed files.

        Args:
            records: List of tuples (path, mtime, has_text, chunk_count, indexed_at).
        """
        if not records:
            return
        with self.get_connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO ocr_files (path, mtime, has_text, chunk_count, indexed_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                records,
            )

    def delete_files(self, paths: List[str]) -> int:
        """Remove deleted file paths from checkpoint database.

        Args:
            paths: List of file paths to remove.

        Returns:
            Number of rows deleted.
        """
        if not paths:
            return 0
        with self.get_connection() as conn:
            cur = conn.executemany("DELETE FROM ocr_files WHERE path = ?", [(p,) for p in paths])
            return cur.rowcount

    def clear(self) -> int:
        """Wipe all checkpoint records from database.

        Returns:
            Number of deleted records.
        """
        with self.get_connection() as conn:
            cur = conn.execute("DELETE FROM ocr_files")
            return cur.rowcount


# ============================================================================
# 4. OCR COLLECTION MANAGER
# ============================================================================

class OCRCollectionManager:
    """Encapsulates Qdrant collection setup, schema validation, and deletion for OCR."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = cfg.OCR_COLLECTION,
    ):
        """Initialize collection manager.

        Args:
            client: QdrantClient instance.
            collection_name: Target Qdrant collection name.
        """
        self.client = client
        self.collection_name = collection_name
        self.ensure_collection()

    def ensure_collection(self) -> None:
        """Ensure collection exists with 'text' vector (384-d, cosine) and text payload index."""
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            logger.info(f"Creating Qdrant collection '{self.collection_name}' with 'text' vector (384-d)...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "text": qmodels.VectorParams(
                        size=cfg.MINILM_DIM,
                        distance=qmodels.Distance.COSINE,
                    ),
                },
            )

        # Ensure full-text payload index on 'text'
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="text",
                field_schema=qmodels.TextIndexParams(
                    type=qmodels.TextIndexType.TEXT,
                    tokenizer=qmodels.TokenizerType.WORD,
                    lowercase=True,
                ),
            )
        except Exception as e:
            logger.debug(f"OCR text payload index note: {e}")

    def count_chunks(self) -> int:
        """Return total number of OCR chunk points in collection.

        Returns:
            Total point count.
        """
        try:
            return self.client.count(collection_name=self.collection_name).count
        except Exception:
            return 0

    def delete_by_path(self, path: str) -> None:
        """Delete all points corresponding to a specific source file path.

        Args:
            path: Normalized path of file to remove.
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=qmodels.Filter(
                    must=[qmodels.FieldCondition(key="path", match=qmodels.MatchValue(value=path))]
                ),
            )
        except Exception as e:
            logger.debug(f"Error removing points for path '{path}': {e}")

    def clear(self) -> int:
        """Clear all points from the OCR collection.

        Returns:
            Number of points removed.
        """
        points_removed = 0
        try:
            existing = [c.name for c in self.client.get_collections().collections]
            if self.collection_name in existing:
                count_resp = self.client.count(collection_name=self.collection_name)
                points_removed = count_resp.count
                offset = None
                all_ids = []
                while True:
                    scroll_res, offset = self.client.scroll(
                        collection_name=self.collection_name,
                        limit=200,
                        offset=offset,
                        with_payload=False,
                        with_vectors=False,
                    )
                    if not scroll_res:
                        break
                    all_ids.extend([p.id for p in scroll_res])
                    if offset is None:
                        break

                if all_ids:
                    self.client.delete(
                        collection_name=self.collection_name,
                        points_selector=qmodels.PointIdsList(points=all_ids),
                    )
        except Exception as e:
            logger.error(f"Error clearing Qdrant OCR collection: {e}")
        return points_removed


def ensure_ocr_collection(
    client: QdrantClient, collection_name: str = cfg.OCR_COLLECTION
) -> None:
    """Facade for ensuring OCR collection exists."""
    OCRCollectionManager(client, collection_name)


def init_cache_db(db_path: Path = CACHE_DB_PATH) -> sqlite3.Connection:
    """Facade for obtaining cache database connection."""
    return OCRCacheManager(db_path).get_connection()


def clear_ocr_database(
    client: Optional[QdrantClient] = None, storage_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Clear all records from Qdrant OCR collection and reset checkpoint cache.

    Args:
        client: Optional QdrantClient instance.
        storage_path: Optional custom storage path for Qdrant.

    Returns:
        Status dictionary with deletion metrics.
    """
    q_client = client or get_ocr_qdrant_client(storage_path)
    coll_mgr = OCRCollectionManager(q_client, cfg.OCR_COLLECTION)
    points_removed = coll_mgr.clear()

    cache_mgr = OCRCacheManager(CACHE_DB_PATH)
    db_removed = cache_mgr.clear()

    return {
        "status": "success",
        "points_removed": points_removed,
        "db_records_removed": db_removed,
        "message": f"Successfully cleared {points_removed} OCR chunks from collection '{cfg.OCR_COLLECTION}'.",
    }


# ============================================================================
# 5. WORKER IMPLEMENTATION FOR PARALLEL OCR EXTRACTION
# ============================================================================

class OCRWorker:
    """Encapsulates worker execution to decode pages, extract OCR text, and generate chunks."""

    @staticmethod
    def process_file(
        args: Tuple[str, str, float, int, int]
    ) -> Dict[str, Any]:
        """Worker task executing in isolated process or thread.

        Args:
            args: Tuple containing:
                - file_path_str: Path string of target document.
                - engine_name: 'rapidocr', 'tesseract', or 'florence'.
                - min_conf: Minimum confidence threshold.
                - chunk_words: Words per chunk.
                - chunk_overlap: Overlapping words.

        Returns:
            Dictionary containing extraction results, success flag, and chunks.
        """
        file_path_str, engine_name, min_conf, chunk_words, chunk_overlap = args
        file_path = Path(file_path_str)
        result: Dict[str, Any] = {
            "path": file_path_str,
            "filename": file_path.name,
            "success": False,
            "has_text": False,
            "chunks": [],
            "error": None,
        }

        try:
            pages = list(extract_pages_from_file(file_path))
            if not pages:
                result["error"] = "No readable pages found or file corrupt"
                return result

            all_chunks: List[Dict[str, Any]] = []

            for page_num, img in pages:
                try:
                    full_text = extract_text_from_image(
                        img, engine=engine_name, min_conf=min_conf
                    )
                    if not full_text:
                        continue

                    text_chunks = chunk_text(
                        full_text, chunk_words=chunk_words, chunk_overlap=chunk_overlap
                    )
                    full_text_len = len(full_text)

                    for chunk_idx, chunk in enumerate(text_chunks):
                        all_chunks.append({
                            "page": page_num,
                            "chunk_index": chunk_idx,
                            "text": chunk,
                            "full_text_len": full_text_len,
                        })
                except Exception as pe:
                    logger.debug(f"Error extracting page {page_num} of {file_path_str}: {pe}")

            result["success"] = True
            result["has_text"] = len(all_chunks) > 0
            result["chunks"] = all_chunks

        except Exception as e:
            result["error"] = str(e)

        return result


def _process_file_worker(args: Tuple[str, str, float, int, int]) -> Dict[str, Any]:
    """Top-level worker function compatible with ProcessPoolExecutor serialization."""
    return OCRWorker.process_file(args)


# ============================================================================
# 6. MAIN OCR INDEXER PIPELINE
# ============================================================================

class OCRIndexer:
    """Manages full OCR indexing lifecycle, multiprocessing pool, and Qdrant storage."""

    def __init__(
        self,
        engine: str = cfg.OCR_ENGINE,
        num_workers: int = cfg.OCR_WORKERS,
        min_conf: float = cfg.OCR_MIN_CONF,
        collection_name: str = cfg.OCR_COLLECTION,
    ):
        """Initialize OCRIndexer with engine preferences and resource configurations.

        Args:
            engine: OCR engine choice ('rapidocr', 'tesseract', 'florence').
            num_workers: Number of parallel CPU worker processes.
            min_conf: Minimum OCR line confidence score (0.0 to 1.0).
            collection_name: Name of Qdrant collection for storage.
        """
        self.engine = engine
        self.num_workers = num_workers
        self.min_conf = min_conf
        self.collection_name = collection_name
        self.client = get_ocr_qdrant_client()
        self.collection_mgr = OCRCollectionManager(self.client, self.collection_name)
        self.cache_mgr = OCRCacheManager(CACHE_DB_PATH)
        self.embedder = get_minilm_embedder()

    def get_stats(self) -> Dict[str, Any]:
        """Return counts of indexed files and chunks.

        Returns:
            Dictionary with statistics: collection, engine, total_files, files_with_text, total_chunks.
        """
        total_chunks = self.collection_mgr.count_chunks()
        total_files, files_with_text = self.cache_mgr.get_stats()

        return {
            "collection": self.collection_name,
            "engine": self.engine,
            "total_files": total_files,
            "files_with_text": files_with_text,
            "total_chunks": total_chunks,
        }

    def index_folder(
        self,
        folder_path: Union[str, Path],
        force_reindex: bool = False,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Scan folder, run OCR on images/PDFs, embed chunks via MiniLM, and upsert to Qdrant.

        Args:
            folder_path: Target directory to index.
            force_reindex: If True, bypass modification time cache and reprocess all files.
            progress_callback: Optional callback receiving metrics dictionary.

        Returns:
            Metrics dictionary summarizing results (total_files, with_text, failed, chunks_upserted).
        """
        folder = Path(folder_path).resolve()
        if not folder.exists() or not folder.is_dir():
            raise ValueError(f"Folder does not exist or is not a directory: {folder}")

        # 1. Discover all candidate files
        all_files: List[Path] = []
        for root, _, files in os.walk(folder):
            for f in files:
                p = Path(root) / f
                if is_supported_ocr_file(p):
                    all_files.append(p)

        total_files = len(all_files)
        logger.info(f"Discovered {total_files} candidate documents/images in {folder}")

        # 2. Query cache to partition new vs unchanged files
        cached_mtimes = self.cache_mgr.get_cached_mtimes()

        files_to_process: List[Path] = []
        files_unchanged: List[Path] = []
        current_paths_set: Set[str] = set()

        for p in all_files:
            norm_path = p.as_posix()
            current_paths_set.add(norm_path)
            try:
                curr_mtime = p.stat().st_mtime
            except Exception:
                curr_mtime = 0.0

            if not force_reindex and norm_path in cached_mtimes:
                if abs(cached_mtimes[norm_path] - curr_mtime) < 1e-4:
                    files_unchanged.append(p)
                    continue

            files_to_process.append(p)

        # 3. Purge deleted files from Qdrant and SQLite
        deleted_paths = [
            p for p in cached_mtimes
            if p not in current_paths_set and p.startswith(folder.as_posix())
        ]
        if deleted_paths:
            logger.info(f"Removing {len(deleted_paths)} deleted files from OCR index...")
            for del_p in deleted_paths:
                self.collection_mgr.delete_by_path(del_p)
            self.cache_mgr.delete_files(deleted_paths)

        metrics = {
            "total_files": total_files,
            "unchanged_files": len(files_unchanged),
            "files_done": len(files_unchanged),
            "with_text": 0,
            "without_text": 0,
            "failed": 0,
            "chunks_upserted": 0,
        }

        if not files_to_process:
            logger.info("All files are up to date. Nothing to index.")
            if progress_callback:
                progress_callback(metrics)
            return metrics

        logger.info(
            f"Processing {len(files_to_process)} files using engine '{self.engine}' (workers={self.num_workers})..."
        )

        batch_points: List[qmodels.PointStruct] = []
        batch_records_to_cache: List[Tuple[str, float, int, int, str]] = []

        def flush_batch() -> None:
            """Batch embed text chunks and upsert to Qdrant collection."""
            nonlocal batch_points, batch_records_to_cache
            if not batch_points:
                return

            # Batch encode chunks using CPU MiniLM
            texts = [p.payload["text"] for p in batch_points]
            embeddings = self.embedder.embed_texts(texts, batch_size=32)

            for point, emb in zip(batch_points, embeddings):
                point.vector = {"text": emb.tolist()}

            # Upsert points to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch_points,
            )
            metrics["chunks_upserted"] += len(batch_points)

            # Update SQLite checkpoint
            self.cache_mgr.record_files(batch_records_to_cache)

            batch_points.clear()
            batch_records_to_cache.clear()

        # 4. Dispatch processing: Florence-2 runs sequentially to protect GPU;
        # RapidOCR / Tesseract run via ProcessPoolExecutor
        if self.engine == "florence":
            for idx, p in enumerate(files_to_process):
                res = _process_file_worker((
                    p.as_posix(),
                    self.engine,
                    self.min_conf,
                    cfg.CHUNK_WORDS,
                    cfg.CHUNK_OVERLAP,
                ))
                self._handle_processed_result(p, res, batch_points, batch_records_to_cache, metrics)
                if len(batch_points) >= 64:
                    flush_batch()
                if progress_callback:
                    progress_callback(metrics)
        else:
            tasks = [
                (
                    p.as_posix(),
                    self.engine,
                    self.min_conf,
                    cfg.CHUNK_WORDS,
                    cfg.CHUNK_OVERLAP,
                )
                for p in files_to_process
            ]

            max_workers = max(1, self.num_workers)
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                for idx, res in enumerate(executor.map(_process_file_worker, tasks)):
                    orig_path = files_to_process[idx]
                    self._handle_processed_result(orig_path, res, batch_points, batch_records_to_cache, metrics)
                    if len(batch_points) >= 64:
                        flush_batch()
                    if progress_callback:
                        progress_callback(metrics)

        # Final batch flush
        flush_batch()

        if progress_callback:
            progress_callback(metrics)

        return metrics

    def _handle_processed_result(
        self,
        file_path: Path,
        result: Dict[str, Any],
        batch_points: List[qmodels.PointStruct],
        batch_records_to_cache: List[Tuple[str, float, int, int, str]],
        metrics: Dict[str, Any],
    ) -> None:
        """Process output from worker, create Qdrant PointStructs, and update counters.

        Args:
            file_path: Source Path object.
            result: Worker output dictionary.
            batch_points: Accumulator for Qdrant PointStructs.
            batch_records_to_cache: Accumulator for SQLite checkpoint rows.
            metrics: Progress counters dictionary.
        """
        norm_path = file_path.as_posix()
        try:
            mtime = file_path.stat().st_mtime
        except Exception:
            mtime = 0.0

        now_iso = datetime.datetime.now().isoformat()
        date_iso = extract_file_date(file_path)

        metrics["files_done"] += 1

        if not result["success"]:
            metrics["failed"] += 1
            logger.warning(f"Failed processing {norm_path}: {result.get('error')}")
            # Cache failure with 0 chunks to prevent retry loop
            batch_records_to_cache.append((norm_path, mtime, 0, 0, now_iso))
            return

        chunks = result.get("chunks", [])
        if not chunks:
            metrics["without_text"] += 1
            batch_records_to_cache.append((norm_path, mtime, 0, 0, now_iso))
            return

        metrics["with_text"] += 1
        batch_records_to_cache.append((norm_path, mtime, 1, len(chunks), now_iso))

        # First, remove any older points for this file path (in case of reindex)
        self.collection_mgr.delete_by_path(norm_path)

        for c in chunks:
            chunk_obj = OCRDocumentChunk.create(
                file_path=norm_path,
                filename=file_path.name,
                page=c["page"],
                chunk_index=c["chunk_index"],
                text=c["text"],
                full_text_len=c["full_text_len"],
                date_iso=date_iso,
                mtime=mtime,
            )

            batch_points.append(
                qmodels.PointStruct(
                    id=chunk_obj.point_id,
                    vector={},  # Will be populated in flush_batch with MiniLM embedding
                    payload=chunk_obj.to_qdrant_payload(),
                )
            )


# ============================================================================
# 7. CLI ENTRYPOINT
# ============================================================================

def main() -> None:
    """CLI for indexing a folder in OCR-only mode."""
    parser = argparse.ArgumentParser(description="OCR-Only Folder Indexer")
    parser.add_argument("folder", type=str, help="Path to folder containing documents/images")
    parser.add_argument(
        "--engine",
        type=str,
        default=cfg.OCR_ENGINE,
        choices=["tesseract", "rapidocr", "florence"],
        help="OCR engine to use (default: config OCR_ENGINE)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=cfg.OCR_WORKERS,
        help="Number of parallel OCR worker processes (default: config OCR_WORKERS)",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force re-indexing of all files even if unchanged",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing OCR collection and checkpoint database before indexing",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.clear:
        res = clear_ocr_database()
        print(f"[CLEAR] {res['message']}")

    indexer = OCRIndexer(engine=args.engine, num_workers=args.workers)
    print(f"Starting OCR indexing on: {args.folder} (engine: {args.engine}, workers: {args.workers})")

    def print_progress(m: Dict[str, Any]) -> None:
        total = m["total_files"]
        done = m["files_done"]
        pct = (done / total * 100) if total > 0 else 100.0
        sys.stdout.write(
            f"\rProgress: {done}/{total} ({pct:.1f}%) | Text: {m['with_text']} | No text: {m['without_text']} | Failed: {m['failed']} | Chunks: {m['chunks_upserted']}"
        )
        sys.stdout.flush()

    metrics = indexer.index_folder(args.folder, force_reindex=args.reindex, progress_callback=print_progress)
    print()
    print("=" * 60)
    print("[PASS] OCR Indexing Complete!")
    print(f"Total files:      {metrics['total_files']}")
    print(f"Files with text:  {metrics['with_text']}")
    print(f"Files discarded:  {metrics['without_text']} (under 4 alnum chars)")
    print(f"Failed files:     {metrics['failed']}")
    print(f"Chunks indexed:   {metrics['chunks_upserted']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
