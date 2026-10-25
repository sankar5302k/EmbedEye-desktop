"""Gradio Web Application for EmbedEye (Local-First Semantic Photo & OCR Search).

This module provides an Object-Oriented web interface built with Gradio:
1. Visual Photo Search: OpenCLIP + Florence-2 (caption + OCR) + Cross-Encoder Reranker.
2. OCR-Only Search: RapidOCR/Tesseract + MiniLM (384-d, CPU) + Cross-Encoder Reranker + RRF.
3. Hardware & Memory Monitoring: Live VRAM allocation, GPU-resident models, and database statistics.

OOP Architecture:
- `ThumbnailRenderer`: Utility class for rendering image/PDF thumbnails.
- `NativeExplorerLauncher`: Utility class for revealing files in native OS explorers.
- `SystemStatusReporter`: Queries hardware stats, VRAM, and Qdrant collection metrics.
- `PhotoSearchApp`: Main application class managing state, layouts, callbacks, and event bindings.
"""

from __future__ import annotations

import os
import platform
import string
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

# --- HARD REQUIREMENT: SET MODEL ENV VARS BEFORE ANY MODEL IMPORTS ---
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
HF_CACHE = MODELS_DIR / "huggingface"
TORCH_CACHE = MODELS_DIR / "torch"
TRANSFORMERS_CACHE = MODELS_DIR / "transformers"
SENTENCE_TRANSFORMERS_CACHE = MODELS_DIR / "sentence_transformers"
OPENCLIP_CACHE = MODELS_DIR / "open_clip"
RERANKER_CACHE = MODELS_DIR / "reranker"

for path in [
    MODELS_DIR,
    HF_CACHE,
    TORCH_CACHE,
    TRANSFORMERS_CACHE,
    SENTENCE_TRANSFORMERS_CACHE,
    OPENCLIP_CACHE,
    RERANKER_CACHE,
]:
    path.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(HF_CACHE.resolve())
os.environ["HF_HUB_CACHE"] = str(HF_CACHE.resolve())
os.environ["TRANSFORMERS_CACHE"] = str(TRANSFORMERS_CACHE.resolve())
os.environ["TORCH_HOME"] = str(TORCH_CACHE.resolve())
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(SENTENCE_TRANSFORMERS_CACHE.resolve())
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import gradio as gr
from PIL import Image

import src.config as cfg
from src.gpu_manager import ModelManager, get_model_manager
from src.indexer import Indexer, PhotoCollectionManager, clear_database, ensure_collection, get_qdrant_client
from src.ocr_indexer import (
    CACHE_DB_PATH as OCR_CACHE_DB_PATH,
    OCRCollectionManager,
    OCRIndexer,
    clear_ocr_database,
    ensure_ocr_collection,
    get_ocr_qdrant_client,
)
from src.ocr_search import OCRSearchEngine, OCRSearcher, highlight_snippet
from src.search import SearchEngine


# ============================================================================
# 1. HELPER CLASSES FOR UI UTILITIES
# ============================================================================

class ThumbnailRenderer:
    """Renders preview thumbnails for local images and PDF document pages."""

    @staticmethod
    def load_doc_thumbnail(file_path: str, page: int = 1) -> Optional[Image.Image]:
        """Render thumbnail for image or specific page of PDF.

        Args:
            file_path: Absolute or relative path to file.
            page: 1-indexed page number (default 1).

        Returns:
            PIL Image object in RGB mode, or None if reading fails.
        """
        try:
            p = Path(file_path)
            if not p.exists():
                return None
            if p.suffix.lower() == ".pdf":
                import pypdfium2 as pdfium

                pdf = pdfium.PdfDocument(str(p))
                page_idx = max(0, page - 1)
                if page_idx < len(pdf):
                    bitmap = pdf[page_idx].render(scale=1.5)
                    img = bitmap.to_pil().convert("RGB")
                    pdf.close()
                    return img
                pdf.close()
            else:
                with Image.open(p) as raw_img:
                    img = raw_img.convert("RGB")
                    img.thumbnail((800, 800))
                    return img
        except Exception:
            return None


class NativeExplorerLauncher:
    """Opens local directories or reveals files in the OS native file explorer."""

    @staticmethod
    def open_folder(path_str: str) -> str:
        """Open containing directory in OS native file explorer.

        Args:
            path_str: Path string of file or folder.

        Returns:
            Status message string.
        """
        if not path_str or not path_str.strip():
            return "No path provided."

        target = Path(path_str.strip())
        folder = target.parent if target.is_file() else target

        if not folder.exists():
            return f"Directory does not exist: {folder}"

        system = platform.system()
        try:
            if system == "Windows":
                if target.is_file():
                    subprocess.Popen(["explorer", f"/select,{str(target.resolve())}"])
                else:
                    os.startfile(str(folder.resolve()))
            elif system == "Darwin":
                subprocess.Popen(["open", str(folder.resolve())])
            else:
                subprocess.Popen(["xdg-open", str(folder.resolve())])
            return f"Opened folder: {folder}"
        except Exception as e:
            return f"Failed to open folder: {e}"


class SystemStatusReporter:
    """Gathers real-time metrics on hardware, VRAM, collections, and models."""

    @staticmethod
    def get_status_summary() -> Tuple[str, str, str]:
        """Retrieve system status metrics: Hardware & Live VRAM, DB counts, and Model Stack.

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
                f"🟢 **{dev_name}** | GPU Resident: `[{res_str}]`\n\n"
                f"VRAM: **{report['allocated_mb']} MB** allocated / **{report['reserved_mb']} MB** reserved "
                f"(Total: {report['total_mb']:.0f} MB)"
            )
        else:
            hw_status = "🟡 **CPU Active**: Host Processor (CUDA unavailable or disabled)"

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

            import sqlite3
            if OCR_CACHE_DB_PATH.exists():
                conn = sqlite3.connect(str(OCR_CACHE_DB_PATH))
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM ocr_files")
                ocr_files = cur.fetchone()[0] or 0
                conn.close()
        except Exception:
            pass

        db_status = (
            f"📁 **Photos DB**: `{photo_count}` indexed (`{cfg.QDRANT_COLLECTION}`)\n\n"
            f"📄 **OCR DB**: `{ocr_files}` files / `{ocr_chunks}` chunks (`{cfg.OCR_COLLECTION}`)"
        )

        models_info = (
            f"🤖 **Visual**: `{cfg.CLIP_MODEL_NAME}` | `{cfg.FLORENCE_MODEL_NAME.split('/')[-1]}`\n\n"
            f"📝 **OCR**: `{cfg.OCR_ENGINE}` | MiniLM (`384-d`) | **Reranker**: `{cfg.RERANKER_MODEL_NAME.split('/')[-1]}`"
        )

        return hw_status, db_status, models_info


# Facades for backwards compatibility
def load_doc_thumbnail(file_path: str, page: int = 1) -> Optional[Image.Image]:
    """Facade for ThumbnailRenderer.load_doc_thumbnail."""
    return ThumbnailRenderer.load_doc_thumbnail(file_path, page=page)


def open_folder(path_str: str) -> str:
    """Facade for NativeExplorerLauncher.open_folder."""
    return NativeExplorerLauncher.open_folder(path_str)


def get_status_summary() -> Tuple[str, str, str]:
    """Facade for SystemStatusReporter.get_status_summary."""
    return SystemStatusReporter.get_status_summary()


# ============================================================================
# 2. CUSTOM CSS STYLING
# ============================================================================

CUSTOM_CSS = """
body, .gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background-color: #0f172a;
    color: #f8fafc;
}

.status-card {
    background: rgba(30, 41, 59, 0.7);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 14px;
}

.header-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}

.header-sub {
    color: #94a3b8;
    font-size: 1.05rem;
    margin-bottom: 16px;
}

mark {
    background-color: #fde047;
    color: #0f172a;
    padding: 2px 4px;
    border-radius: 4px;
    font-weight: 700;
}
"""


# ============================================================================
# 3. PHOTO SEARCH APP (MAIN OOP CONTROLLER)
# ============================================================================

class PhotoSearchApp:
    """Encapsulates the state, layout builders, callbacks, and event bindings for Gradio UI."""

    def __init__(self):
        """Initialize PhotoSearchApp with lazy engine instances."""
        self._search_engine: Optional[SearchEngine] = None
        self._ocr_searcher: Optional[OCRSearchEngine] = None

    def get_search_engine(self) -> SearchEngine:
        """Obtain or lazily instantiate the Visual SearchEngine."""
        if self._search_engine is None:
            self._search_engine = SearchEngine()
        return self._search_engine

    def get_ocr_engine(self) -> OCRSearchEngine:
        """Obtain or lazily instantiate the OCRSearchEngine."""
        if self._ocr_searcher is None:
            self._ocr_searcher = OCRSearchEngine()
        return self._ocr_searcher

    # ------------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------------

    def handle_clear_photo_db(self) -> Tuple[str, str, str, str]:
        """Clear all records from Qdrant photo collection and reset SQLite cache.

        Returns:
            Tuple of (status_markdown, hw_status, db_status, models_info).
        """
        self._search_engine = None
        res = clear_database()
        summary = (
            f"🗑️ **Photo Database Cleared Successfully!**\n\n"
            f"- **Qdrant Vector Points Removed:** {res['points_removed']}\n"
            f"- **SQLite Cache Checkpoints Cleared:** {res['cache_entries_removed']}\n"
            f"- **Status:** Database is empty and ready for fresh indexing."
        )
        hw, db, api = SystemStatusReporter.get_status_summary()
        return summary, hw, db, api

    def handle_clear_ocr_db(self) -> Tuple[str, str, str, str]:
        """Clear all records from Qdrant OCR collection and checkpoint DB.

        Returns:
            Tuple of (status_markdown, hw_status, db_status, models_info).
        """
        self._ocr_searcher = None
        res = clear_ocr_database()
        summary = (
            f"🗑️ **OCR Database Cleared!**\n\n"
            f"- **Chunks Removed from Qdrant:** {res['points_removed']}\n"
            f"- **SQLite Records Removed:** {res['db_records_removed']}\n"
            f"- **Collection:** `{cfg.OCR_COLLECTION}`"
        )
        hw, db, api = SystemStatusReporter.get_status_summary()
        return summary, hw, db, api

    def run_photo_indexing(
        self,
        folder_path: str,
        recaption_only: bool = False,
        progress=gr.Progress(track_tqdm=True),
    ) -> Generator[Tuple[str, str, str, str], None, None]:
        """Index or update photo folder with live progress callback.

        Args:
            folder_path: Path string of folder to index.
            recaption_only: Whether to skip CLIP and re-run captioning only.
            progress: Gradio Progress tracker.

        Yields:
            Tuple of (summary_markdown, hw_status, db_status, models_info).
        """
        if not folder_path or not folder_path.strip():
            hw, db, api = SystemStatusReporter.get_status_summary()
            yield "❌ Error: Please specify a valid folder path.", hw, db, api
            return

        folder = Path(folder_path.strip())
        if not folder.is_dir():
            hw, db, api = SystemStatusReporter.get_status_summary()
            yield f"❌ Error: Directory '{folder}' does not exist.", hw, db, api
            return

        def cb(current: int, total: int, msg: str):
            progress((current, total), desc=msg)

        indexer = Indexer()
        try:
            res = indexer.index_directory(folder, recaption_only=recaption_only, progress_callback=cb)
            stage_lines = []
            for stg, vram in res.get("peak_vram_stages", {}).items():
                stage_lines.append(f"  - **{stg}**: {vram:.1f} MB peak")

            stages_str = "\n".join(stage_lines) if stage_lines else "None"

            summary = (
                f"✅ **{'Recaptioning' if recaption_only else 'Indexing'} Completed!**\n\n"
                f"- **Device Used:** `{res['device'].upper()}`\n"
                f"- **Total Scanned:** {res['total_scanned']}\n"
                f"- **New / Updated:** {res['new_or_updated']}\n"
                f"- **Skipped (Unchanged):** {res['skipped']}\n"
                f"- **Failed / Unreadable:** {res['failed']}\n"
                f"- **Removed (Deleted):** {res['removed']}\n"
                f"- **VRAM Allocated:** {res['vram_allocated_mb']:.1f} MB\n\n"
                f"**Peak VRAM per Stage:**\n{stages_str}"
            )
        except Exception as e:
            summary = f"❌ **Indexing Failed:** {e}"

        hw, db, api = SystemStatusReporter.get_status_summary()
        yield summary, hw, db, api

    def run_ocr_indexing(
        self,
        folder_path: str,
        engine: str,
        force_reindex: bool,
        progress=gr.Progress(track_tqdm=True),
    ) -> Generator[Tuple[str, str, str, str], None, None]:
        """Index or update documents/images in OCR-Only Mode.

        Args:
            folder_path: Path string of folder.
            engine: Selected OCR engine.
            force_reindex: Whether to ignore cached modification timestamps.
            progress: Gradio Progress tracker.

        Yields:
            Tuple of (summary_markdown, hw_status, db_status, models_info).
        """
        if not folder_path or not folder_path.strip():
            hw, db, api = SystemStatusReporter.get_status_summary()
            yield "❌ Error: Please specify a valid folder path.", hw, db, api
            return

        folder = Path(folder_path.strip())
        if not folder.is_dir():
            hw, db, api = SystemStatusReporter.get_status_summary()
            yield f"❌ Error: Directory '{folder}' does not exist.", hw, db, api
            return

        indexer = OCRIndexer(engine=engine, num_workers=cfg.OCR_WORKERS)

        def p_cb(m: Dict[str, Any]):
            total = m["total_files"]
            done = m["files_done"]
            pct = (done / total) if total > 0 else 1.0
            progress(
                pct,
                desc=f"OCR Indexing ({done}/{total}) | Text: {m['with_text']} | No text: {m['without_text']} | Failed: {m['failed']}",
            )

        try:
            metrics = indexer.index_folder(folder, force_reindex=force_reindex, progress_callback=p_cb)
            summary = (
                f"✅ **OCR Indexing Complete!**\n\n"
                f"- **OCR Engine:** `{engine.upper()}`\n"
                f"- **Total Candidate Files:** {metrics['total_files']}\n"
                f"- **Files with OCR Text:** {metrics['with_text']}\n"
                f"- **Discarded (<4 alphanumeric):** {metrics['without_text']}\n"
                f"- **Failed / Unreadable:** {metrics['failed']}\n"
                f"- **Unchanged (Skipped):** {metrics['unchanged_files']}\n"
                f"- **Text Chunks Upserted (Qdrant):** {metrics['chunks_upserted']}\n\n"
                f"All text chunks stored in collection `{cfg.OCR_COLLECTION}` with 384-d MiniLM vectors."
            )
        except Exception as e:
            summary = f"❌ **OCR Indexing Failed:** {e}"

        hw, db, api = SystemStatusReporter.get_status_summary()
        yield summary, hw, db, api

    def execute_photo_search(
        self,
        query: str,
        use_rerank: bool,
        use_smart_date: bool,
    ) -> Tuple[
        List[Tuple[Any, str]],
        str,
        List[Dict[str, Any]],
        str,
        str,
        str,
        str,
        str,
        str,
        str,
        str,
    ]:
        """Execute visual photo search and cross-encoder reranking.

        Args:
            query: Search query string.
            use_rerank: Cross-encoder rerank flag.
            use_smart_date: Date extraction flag.

        Returns:
            Tuple containing gallery items, formatted details, candidate table, badge, and paths.
        """
        engine = self.get_search_engine()
        output = engine.search(
            query=query,
            use_rerank=use_rerank,
            use_smart_date=use_smart_date,
        )
        results = output.get("results", [])
        all_cands = output.get("all_candidates", [])
        badge = output.get("badge", "")

        gallery_items: List[Tuple[Any, str]] = []
        details_md_lines = [f"### Results Badge: `{badge}`\n"]
        paths = ["", "", "", ""]

        for i, item in enumerate(results[:4]):
            p = item.get("path", "")
            cap = item.get("caption", "")
            ocr_t = item.get("ocr_text", "")
            dt = item.get("date", "unknown")
            fn = item.get("filename", "")
            paths[i] = p

            thumb = ThumbnailRenderer.load_doc_thumbnail(p)
            if thumb is not None:
                gallery_items.append((thumb, f"#{i+1}: {fn}"))
            else:
                gallery_items.append((p, f"#{i+1}: {fn}"))

            details_md_lines.append(
                f"**#{i+1}: {fn}**\n"
                f"- **Caption:** {cap}\n"
                f"- **OCR Text:** {ocr_t if ocr_t else '*(No text detected)*'}\n"
                f"- **Date:** {dt}\n"
                f"- **Path:** `{p}`\n"
            )

        candidates_table = [
            {
                "Rank": c.get("rank"),
                "Filename": c.get("filename"),
                "Caption": c.get("caption"),
                "OCR Text": c.get("ocr_text", ""),
                "Date": c.get("date"),
                "Score": round(float(c.get("score", 0.0)), 4),
                "Path": c.get("path"),
            }
            for c in all_cands
        ]

        hw, db, api = SystemStatusReporter.get_status_summary()
        return (
            gallery_items,
            "\n".join(details_md_lines) if gallery_items else "No results found.",
            candidates_table,
            badge,
            paths[0],
            paths[1],
            paths[2],
            paths[3],
            hw,
            db,
            api,
        )

    def execute_ocr_search(
        self,
        query: str,
        use_rerank: bool,
    ) -> Tuple[
        List[Tuple[Any, str]],
        str,
        List[Dict[str, Any]],
        str,
        str,
        str,
        str,
        str,
        str,
        str,
        str,
    ]:
        """Execute hybrid OCR text search, RRF merging, cross-encoder reranking, and document grouping.

        Args:
            query: Document search query.
            use_rerank: Neural reranker flag.

        Returns:
            Tuple containing gallery items, formatted snippet details, candidate table, badge, and paths.
        """
        searcher = self.get_ocr_engine()
        res = searcher.search(query=query, top_k=4, use_reranker=use_rerank)

        images = res.get("images", [])
        candidates = res.get("candidates", [])

        gallery_items: List[Tuple[Any, str]] = []
        details_md_lines = [f"### Query: `{query}` | Reranker: `{'mxbai-rerank-large-v1' if use_rerank else 'RRF Only'}`\n"]
        paths = ["", "", "", ""]

        for i, img_item in enumerate(images[:4]):
            p = img_item.get("path", "")
            fn = img_item.get("filename", "")
            page = img_item.get("page", 1)
            score = img_item.get("best_score", 0.0)
            rrf_s = img_item.get("rrf_score", 0.0)
            snippet = img_item.get("highlighted_snippet", "")
            paths[i] = p

            thumb = ThumbnailRenderer.load_doc_thumbnail(p, page=page)
            label = f"#{i+1}: {fn} (Page {page})" if page > 1 else f"#{i+1}: {fn}"
            if thumb is not None:
                gallery_items.append((thumb, label))
            else:
                gallery_items.append((p, label))

            details_md_lines.append(
                f"**#{i+1}: {fn}**" + (f" *(Page {page})*" if page > 1 else "") + "\n"
                f"- **Score:** `{score:.4f}` | **RRF Score:** `{rrf_s:.4f}`\n"
                f"- **Matched Snippet:** {snippet}\n"
                f"- **File Path:** `{p}`\n"
            )

        candidates_table = [
            {
                "Rank": idx + 1,
                "Filename": c.get("filename"),
                "Page": c.get("page", 1),
                "Chunk": c.get("chunk_index", 0),
                "Score": round(float(c.get("score", 0.0)), 4),
                "RRF Score": round(float(c.get("rrf_score", 0.0)), 4),
                "Sources": ", ".join(c.get("sources", [])),
                "Text Snippet": c.get("text", "")[:140] + ("..." if len(c.get("text", "")) > 140 else ""),
                "Path": c.get("path"),
            }
            for idx, c in enumerate(candidates)
        ]

        status_badge = f"### OCR Results: `{len(images)} documents matched ({len(candidates)} candidate chunks)`"
        hw, db, api = SystemStatusReporter.get_status_summary()

        return (
            gallery_items,
            "\n".join(details_md_lines) if gallery_items else "No matching OCR text found for this query.",
            candidates_table,
            status_badge,
            paths[0],
            paths[1],
            paths[2],
            paths[3],
            hw,
            db,
            api,
        )

    # ------------------------------------------------------------------------
    # UI Layout Construction
    # ------------------------------------------------------------------------

    def build_ui(self) -> gr.Blocks:
        """Construct the complete Gradio Blocks application layout.

        Returns:
            Configured gr.Blocks application instance.
        """
        hw_init, db_init, api_init = SystemStatusReporter.get_status_summary()

        with gr.Blocks(title="EmbedEye — Local Photo & OCR Search") as demo:
            with gr.Row():
                with gr.Column(scale=12):
                    gr.Markdown(
                        """
                        <div class="header-title">⚡ EmbedEye</div>
                        <div class="header-sub">
                            100% Local Intelligent Search: Dual Modes (Visual Photo Search & OCR-Only Document Search).
                            Strict Single-Model GPU Isolation for NVIDIA RTX 3050 4GB VRAM with CPU Fallback.
                        </div>
                        """
                    )

            # Status panel
            with gr.Row(elem_classes=["status-card"]):
                with gr.Column(scale=5):
                    hw_display = gr.Markdown(hw_init, elem_id="hw_status_display")
                with gr.Column(scale=3):
                    db_display = gr.Markdown(db_init, elem_id="db_status_display")
                with gr.Column(scale=4):
                    api_display = gr.Markdown(api_init, elem_id="api_status_display")

            with gr.Tabs():
                # ===========================================================
                # TAB 1: OCR SEARCH
                # ===========================================================
                with gr.Tab("📄 OCR Search (Documents & Bills)"):
                    with gr.Row():
                        with gr.Column(scale=9):
                            ocr_folder_input = gr.Textbox(
                                label="Document / Image Folder Path",
                                placeholder="e.g. e:\\documents or sample_ocr_docs",
                                lines=1,
                                elem_id="ocr_folder_input",
                            )
                        with gr.Column(scale=3):
                            ocr_engine_select = gr.Dropdown(
                                label="OCR Engine",
                                choices=["tesseract", "rapidocr", "florence"],
                                value=cfg.OCR_ENGINE,
                                elem_id="ocr_engine_select",
                            )

                    with gr.Row():
                        ocr_reindex_cb = gr.Checkbox(
                            label="🔄 Force Re-index all files",
                            value=False,
                            scale=5,
                        )
                        ocr_index_btn = gr.Button(
                            "⚡ Index OCR Folder",
                            variant="primary",
                            scale=4,
                            elem_id="ocr_index_button",
                        )
                        ocr_clear_db_btn = gr.Button(
                            "🗑️ Clear OCR Database",
                            variant="stop",
                            size="sm",
                            scale=3,
                            elem_id="ocr_clear_db_button",
                        )

                    ocr_index_status = gr.Markdown(
                        "Enter a directory with images or PDFs (jpg, png, webp, pdf, etc.) and click 'Index OCR Folder'.\n"
                        "Extracts OCR text, filters confidence, discards <4 alphanumeric chars, chunks text, and embeds via MiniLM.",
                        elem_id="ocr_index_status_box",
                    )

                    gr.Markdown("---")

                    # OCR Search controls
                    with gr.Row():
                        with gr.Column(scale=9):
                            ocr_search_box = gr.Textbox(
                                label="OCR Query (Natural Language or \"Exact Quoted Terms\")",
                                placeholder="e.g. 'electricity bill', 'invoice total $450', or exact phrase: \"Account Number\"",
                                lines=1,
                                elem_id="ocr_search_query_box",
                            )
                        with gr.Column(scale=3):
                            ocr_search_btn = gr.Button(
                                "🔎 Search OCR",
                                variant="primary",
                                elem_id="ocr_search_button",
                            )

                    with gr.Row():
                        ocr_use_rerank_cb = gr.Checkbox(
                            label="⚡ Local Cross-Encoder Reranker (mxbai-rerank-large-v1 on GPU, batch size 8)",
                            value=True,
                            elem_id="ocr_use_rerank_cb",
                        )
                        ocr_refresh_btn = gr.Button("🔄 Refresh Status", size="sm")

                    with gr.Row():
                        ocr_badge_display = gr.Markdown("### Mode: `Ready for OCR search`", elem_id="ocr_badge")

                    # OCR Gallery
                    gr.Markdown("### 🌟 Top 4 Matched Documents")
                    ocr_gallery = gr.Gallery(
                        label="Top 4 Document Matches",
                        show_label=False,
                        elem_id="ocr_result_gallery",
                        columns=4,
                        rows=1,
                        height=340,
                        object_fit="contain",
                    )

                    ocr_p1 = gr.State("")
                    ocr_p2 = gr.State("")
                    ocr_p3 = gr.State("")
                    ocr_p4 = gr.State("")

                    with gr.Row():
                        ocr_open_btn_1 = gr.Button("📂 Open #1 Folder", size="sm")
                        ocr_open_btn_2 = gr.Button("📂 Open #2 Folder", size="sm")
                        ocr_open_btn_3 = gr.Button("📂 Open #3 Folder", size="sm")
                        ocr_open_btn_4 = gr.Button("📂 Open #4 Folder", size="sm")

                    ocr_folder_msg = gr.Markdown("")
                    ocr_results_details = gr.Markdown("Matching text snippet and file details will appear here.", elem_id="ocr_results_details")

                    with gr.Accordion("📋 Show All Candidates with Scores (RRF & Cross-Encoder Pool)", open=False):
                        ocr_candidates_df = gr.Dataframe(
                            headers=["Rank", "Filename", "Page", "Chunk", "Score", "RRF Score", "Sources", "Text Snippet", "Path"],
                            datatype=["number", "str", "number", "number", "number", "number", "str", "str", "str"],
                            interactive=False,
                            elem_id="ocr_candidates_table",
                        )

                # ===========================================================
                # TAB 2: VISUAL PHOTO SEARCH
                # ===========================================================
                with gr.Tab("🔍 Visual Photo Search"):
                    with gr.Row():
                        with gr.Column(scale=9):
                            search_box = gr.Textbox(
                                label="Natural Language or Keyword Query",
                                placeholder="e.g. 'sunset over snowy mountains', 'red car on beach', or keyword: \"bill from 'Reliance'\"",
                                lines=1,
                                elem_id="search_query_box",
                            )
                        with gr.Column(scale=3):
                            search_btn = gr.Button("🔎 Search", variant="primary", elem_id="search_button")

                    with gr.Row():
                        use_rerank_cb = gr.Checkbox(
                            label="⚡ Local Neural Reranker (mxbai-rerank-large-v1)",
                            value=True,
                            elem_id="use_rerank_checkbox",
                        )
                        use_date_cb = gr.Checkbox(
                            label="📅 Offline Date Filter (e.g. '2024')",
                            value=False,
                            elem_id="use_date_checkbox",
                        )
                        refresh_status_btn = gr.Button("🔄 Refresh Status", size="sm")

                    with gr.Row():
                        badge_display = gr.Markdown("### Mode: `Ready for search`", elem_id="search_badge")

                    # Top 4 Result Gallery
                    gr.Markdown("### 🌟 Top Matching Photos (Ranked)")
                    gallery = gr.Gallery(
                        label="Top 4 Matches",
                        show_label=False,
                        elem_id="result_gallery",
                        columns=4,
                        rows=1,
                        height=340,
                        object_fit="contain",
                    )

                    path1 = gr.State("")
                    path2 = gr.State("")
                    path3 = gr.State("")
                    path4 = gr.State("")

                    with gr.Row():
                        open_btn_1 = gr.Button("📂 Open #1 Folder", size="sm")
                        open_btn_2 = gr.Button("📂 Open #2 Folder", size="sm")
                        open_btn_3 = gr.Button("📂 Open #3 Folder", size="sm")
                        open_btn_4 = gr.Button("📂 Open #4 Folder", size="sm")

                    folder_action_msg = gr.Markdown("")
                    results_details = gr.Markdown("Search details will appear here.", elem_id="results_details")

                    with gr.Accordion("📋 Show All Ranked Candidates (Cross-Encoder & Vector Pool)", open=False):
                        candidates_df = gr.Dataframe(
                            headers=["Rank", "Filename", "Caption", "OCR Text", "Date", "Score", "Path"],
                            datatype=["number", "str", "str", "str", "str", "number", "str"],
                            interactive=False,
                            elem_id="candidates_table",
                        )

                # ===========================================================
                # TAB 3: PHOTO LIBRARY INDEXING
                # ===========================================================
                with gr.Tab("📁 Index Photo Library"):
                    with gr.Row():
                        with gr.Column(scale=9):
                            folder_input = gr.Textbox(
                                label="Photo Folder Path",
                                placeholder="e.g. e:\\photos or test_images",
                                lines=1,
                                elem_id="folder_input_path",
                            )
                        with gr.Column(scale=3):
                            index_btn = gr.Button(
                                "⚡ Index Photo Folder",
                                variant="primary",
                                elem_id="index_action_button",
                            )

                    with gr.Row():
                        recaption_cb = gr.Checkbox(
                            label="🔄 Recaption Only (Re-run Florence-2 without re-embedding CLIP)",
                            value=False,
                            elem_id="recaption_checkbox",
                            scale=8,
                        )
                        clear_db_btn = gr.Button(
                            "🗑️ Clear Photo Database",
                            variant="stop",
                            size="sm",
                            scale=4,
                            elem_id="clear_database_button",
                        )

                    index_status = gr.Markdown(
                        "Select a folder containing photos and click 'Index Photo Folder' to index.\n"
                        "Pipeline executes staged phases (CLIP -> Florence-2 -> Qdrant) with GPU memory isolation.",
                        elem_id="index_status_box",
                    )

            # ---------------------------------------------------------------
            # EVENT BINDINGS
            # ---------------------------------------------------------------
            ocr_index_btn.click(
                fn=self.run_ocr_indexing,
                inputs=[ocr_folder_input, ocr_engine_select, ocr_reindex_cb],
                outputs=[ocr_index_status, hw_display, db_display, api_display],
            )

            ocr_clear_db_btn.click(
                fn=self.handle_clear_ocr_db,
                inputs=[],
                outputs=[ocr_index_status, hw_display, db_display, api_display],
            )

            ocr_search_btn.click(
                fn=self.execute_ocr_search,
                inputs=[ocr_search_box, ocr_use_rerank_cb],
                outputs=[
                    ocr_gallery,
                    ocr_results_details,
                    ocr_candidates_df,
                    ocr_badge_display,
                    ocr_p1,
                    ocr_p2,
                    ocr_p3,
                    ocr_p4,
                    hw_display,
                    db_display,
                    api_display,
                ],
            )

            ocr_search_box.submit(
                fn=self.execute_ocr_search,
                inputs=[ocr_search_box, ocr_use_rerank_cb],
                outputs=[
                    ocr_gallery,
                    ocr_results_details,
                    ocr_candidates_df,
                    ocr_badge_display,
                    ocr_p1,
                    ocr_p2,
                    ocr_p3,
                    ocr_p4,
                    hw_display,
                    db_display,
                    api_display,
                ],
            )

            ocr_open_btn_1.click(fn=NativeExplorerLauncher.open_folder, inputs=[ocr_p1], outputs=[ocr_folder_msg])
            ocr_open_btn_2.click(fn=NativeExplorerLauncher.open_folder, inputs=[ocr_p2], outputs=[ocr_folder_msg])
            ocr_open_btn_3.click(fn=NativeExplorerLauncher.open_folder, inputs=[ocr_p3], outputs=[ocr_folder_msg])
            ocr_open_btn_4.click(fn=NativeExplorerLauncher.open_folder, inputs=[ocr_p4], outputs=[ocr_folder_msg])

            ocr_refresh_btn.click(
                fn=SystemStatusReporter.get_status_summary,
                inputs=[],
                outputs=[hw_display, db_display, api_display],
            )

            search_btn.click(
                fn=self.execute_photo_search,
                inputs=[search_box, use_rerank_cb, use_date_cb],
                outputs=[
                    gallery,
                    results_details,
                    candidates_df,
                    badge_display,
                    path1,
                    path2,
                    path3,
                    path4,
                    hw_display,
                    db_display,
                    api_display,
                ],
            )
            search_box.submit(
                fn=self.execute_photo_search,
                inputs=[search_box, use_rerank_cb, use_date_cb],
                outputs=[
                    gallery,
                    results_details,
                    candidates_df,
                    badge_display,
                    path1,
                    path2,
                    path3,
                    path4,
                    hw_display,
                    db_display,
                    api_display,
                ],
            )

            index_btn.click(
                fn=self.run_photo_indexing,
                inputs=[folder_input, recaption_cb],
                outputs=[index_status, hw_display, db_display, api_display],
            )

            clear_db_btn.click(
                fn=self.handle_clear_photo_db,
                inputs=[],
                outputs=[index_status, hw_display, db_display, api_display],
            )

            refresh_status_btn.click(
                fn=SystemStatusReporter.get_status_summary,
                inputs=[],
                outputs=[hw_display, db_display, api_display],
            )

            open_btn_1.click(fn=NativeExplorerLauncher.open_folder, inputs=[path1], outputs=[folder_action_msg])
            open_btn_2.click(fn=NativeExplorerLauncher.open_folder, inputs=[path2], outputs=[folder_action_msg])
            open_btn_3.click(fn=NativeExplorerLauncher.open_folder, inputs=[path3], outputs=[folder_action_msg])
            open_btn_4.click(fn=NativeExplorerLauncher.open_folder, inputs=[path4], outputs=[folder_action_msg])

        return demo


def build_app() -> gr.Blocks:
    """Facade for instantiating PhotoSearchApp and generating the UI."""
    app_instance = PhotoSearchApp()
    return app_instance.build_ui()


# ============================================================================
# 4. APPLICATION LAUNCH ENTRYPOINT
# ============================================================================

def main() -> None:
    """Main application launcher."""
    allowed_drives = [
        f"{d}:\\" for d in string.ascii_uppercase if Path(f"{d}:/").exists()
    ]
    allowed_drives.append(str(PROJECT_ROOT.resolve()))

    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        allowed_paths=allowed_drives,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
