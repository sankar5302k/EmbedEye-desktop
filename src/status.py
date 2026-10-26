"""System Status and Hardware Telemetry Module for EmbedEye.

Provides real-time reporting of GPU VRAM metrics, resident models, active devices,
and vector/checkpoint database record counts for both visual photo search and
OCR document collections.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import src.config as cfg
from src.gpu_manager import ModelManager, get_model_manager
from src.indexer import ensure_collection, get_qdrant_client
from src.ocr_indexer import (
    CACHE_DB_PATH as OCR_CACHE_DB_PATH,
    ensure_ocr_collection,
    get_ocr_qdrant_client,
)


class SystemStatusReporter:
    """Gathers real-time metrics on hardware, VRAM, collections, and model stacks."""

    @staticmethod
    def get_status_summary() -> Tuple[str, str, str]:
        """Retrieve formatted status strings: Hardware & Live VRAM, DB counts, and Model Stack.

        Returns:
            Tuple of (hw_status_markdown, db_status_markdown, models_info_markdown).
        """
        manager = get_model_manager()
        report = manager.vram_report()
        dev_name = report["device_name"]
        res_models = report["resident_models"]
        res_str = ", ".join(res_models) if res_models else "None (idle)"

        if report["device"] == "cuda":
            hw_status = (
                f"**[CUDA Active] {dev_name}** | GPU Resident: `[{res_str}]`\n\n"
                f"VRAM: **{report['allocated_mb']} MB** allocated / **{report['reserved_mb']} MB** reserved "
                f"(Total: {report['total_mb']:.0f} MB)"
            )
        else:
            hw_status = "**[CPU Active]**: Host Processor (CUDA unavailable or disabled)"

        # Photo collection stats
        photo_count = 0
        try:
            client = get_qdrant_client()
            ensure_collection(client)
            info = client.get_collection(cfg.QDRANT_COLLECTION)
            photo_count = info.points_count or 0
        except Exception:
            pass

        # OCR collection stats
        ocr_chunks = 0
        ocr_files = 0
        try:
            ocr_cl = get_ocr_qdrant_client()
            ensure_ocr_collection(ocr_cl)
            ocr_info = ocr_cl.get_collection(cfg.OCR_COLLECTION)
            ocr_chunks = ocr_info.points_count or 0

            if OCR_CACHE_DB_PATH.exists():
                conn = sqlite3.connect(str(OCR_CACHE_DB_PATH))
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM ocr_files")
                ocr_files = cur.fetchone()[0] or 0
                conn.close()
        except Exception:
            pass

        db_status = (
            f"**Photos Database**: `{photo_count}` indexed (`{cfg.QDRANT_COLLECTION}`)\n\n"
            f"**OCR Database**: `{ocr_files}` files / `{ocr_chunks}` chunks (`{cfg.OCR_COLLECTION}`)"
        )

        models_info = (
            f"**Visual Engine**: `{cfg.CLIP_MODEL_NAME}` | `{cfg.FLORENCE_MODEL_NAME.split('/')[-1]}`\n\n"
            f"**OCR Engine**: `{cfg.OCR_ENGINE}` | MiniLM (`384-d`) | **Reranker**: `{cfg.RERANKER_MODEL_NAME.split('/')[-1]}`"
        )

        return hw_status, db_status, models_info

    @staticmethod
    def get_raw_status() -> Dict[str, Any]:
        """Query structured dictionary of system status metrics.

        Returns:
            Dictionary with device, vram, resident_models, and collection counts.
        """
        manager = get_model_manager()
        report = manager.vram_report()

        photo_count = 0
        try:
            client = get_qdrant_client()
            ensure_collection(client)
            info = client.get_collection(cfg.QDRANT_COLLECTION)
            photo_count = info.points_count or 0
        except Exception:
            pass

        ocr_chunks = 0
        ocr_files = 0
        try:
            ocr_cl = get_ocr_qdrant_client()
            ensure_ocr_collection(ocr_cl)
            ocr_info = ocr_cl.get_collection(cfg.OCR_COLLECTION)
            ocr_chunks = ocr_info.points_count or 0

            if OCR_CACHE_DB_PATH.exists():
                conn = sqlite3.connect(str(OCR_CACHE_DB_PATH))
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM ocr_files")
                ocr_files = cur.fetchone()[0] or 0
                conn.close()
        except Exception:
            pass

        return {
            "device": report["device"],
            "device_name": report["device_name"],
            "vram_allocated_mb": report["allocated_mb"],
            "vram_reserved_mb": report["reserved_mb"],
            "vram_total_mb": report["total_mb"],
            "resident_models": report["resident_models"],
            "photos_count": photo_count,
            "ocr_files_count": ocr_files,
            "ocr_chunks_count": ocr_chunks,
        }


def get_status_summary() -> Tuple[str, str, str]:
    """Facade for SystemStatusReporter.get_status_summary."""
    return SystemStatusReporter.get_status_summary()
