"""Unified Command-Line Interface for EmbedEye.

Provides command-line utilities for:
- Visual photo indexing and semantic search
- OCR-only document indexing and hybrid search
- Database clearing and hardware telemetry status reporting

Usage examples:
    python -m src.cli index "C:\\Photos"
    python -m src.cli search "red car on a sunny beach"
    python -m src.cli ocr-index "C:\\Documents" --engine tesseract
    python -m src.cli ocr-search "\"invoice\" total $500"
    python -m src.cli clear --all
    python -m src.cli status
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import src.config as cfg
from src.indexer import Indexer, clear_database
from src.ocr_indexer import OCRIndexer, clear_ocr_database
from src.ocr_search import OCRSearchEngine
from src.search import SearchEngine
from src.status import SystemStatusReporter


def cmd_index(args: argparse.Namespace) -> None:
    """Index a folder of photos using the visual OpenCLIP + Florence-2 pipeline."""
    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        print(f"Error: Directory not found: {folder}", file=sys.stderr)
        sys.exit(1)

    print(f"Starting visual photo indexing on: {folder}")
    if args.recaption:
        print("Mode: Recaption only (Florence-2 caption + OCR without re-embedding CLIP)")

    indexer = Indexer()
    res = indexer.index_directory(folder, recaption_only=args.recaption)

    print("\n" + "=" * 65)
    print("VISUAL INDEXING SUMMARY:")
    print("=" * 65)
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
    print("=" * 65)


def cmd_search(args: argparse.Namespace) -> None:
    """Execute visual semantic search with optional cross-encoder reranking."""
    engine = SearchEngine()
    output = engine.search(
        query=args.query,
        use_rerank=not args.no_rerank,
        use_smart_date=args.smart_date,
    )

    results = output.get("results", [])
    print("\n" + "=" * 65)
    print(f"SEARCH RESULTS FOR: \"{args.query}\"")
    print(f"Retrieval Mode    : {output.get('badge', 'Visual Search')}")
    if output.get("date_range"):
        print(f"Date Filter Applied: {output['date_range']}")
    print(f"Found {len(results)} top matches")
    print("=" * 65)

    if not results:
        print("No matching photos found in database.")
        return

    for idx, res in enumerate(results, start=1):
        print(f"\n[{idx}] {res.get('filename')}")
        print(f"    Path    : {res.get('path')}")
        print(f"    Date    : {res.get('date', 'unknown')}")
        print(f"    Score   : {res.get('score', 0.0):.4f}")
        print(f"    Caption : {res.get('caption', '')}")
        ocr_text = res.get("ocr_text", "")
        if ocr_text:
            print(f"    OCR Text: {ocr_text[:200]}...")
        print("-" * 65)


def cmd_ocr_index(args: argparse.Namespace) -> None:
    """Index a folder of documents or images in OCR-only mode."""
    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        print(f"Error: Directory not found: {folder}", file=sys.stderr)
        sys.exit(1)

    print(f"Starting OCR-only indexing on: {folder}")
    print(f"Engine: {args.engine} | Workers: {args.workers} | Min Confidence: {args.min_conf}")

    indexer = OCRIndexer(
        engine=args.engine,
        num_workers=args.workers,
        min_conf=args.min_conf,
    )

    def print_progress(m: dict) -> None:
        total = m["total_files"]
        done = m["files_done"]
        pct = (done / total * 100) if total > 0 else 100.0
        sys.stdout.write(
            f"\rProgress: {done}/{total} ({pct:.1f}%) | Text: {m['with_text']} | Discarded: {m['without_text']} | Failed: {m['failed']} | Chunks: {m['chunks_upserted']}"
        )
        sys.stdout.flush()

    metrics = indexer.index_folder(
        folder,
        force_reindex=args.reindex,
        progress_callback=print_progress,
    )
    print()
    print("\n" + "=" * 65)
    print("OCR INDEXING COMPLETE!")
    print("=" * 65)
    print(f"Total files scanned : {metrics['total_files']}")
    print(f"Files with text     : {metrics['with_text']}")
    print(f"Discarded (<4 alnum): {metrics['without_text']}")
    print(f"Unchanged (skipped) : {metrics['unchanged_files']}")
    print(f"Failed files        : {metrics['failed']}")
    print(f"Text chunks upserted: {metrics['chunks_upserted']}")
    print("=" * 65)


def cmd_ocr_search(args: argparse.Namespace) -> None:
    """Execute hybrid dense vector + full-text search on OCR documents."""
    searcher = OCRSearchEngine()
    res = searcher.search(
        query=args.query,
        top_k=args.top,
        use_reranker=not args.no_rerank,
    )

    images = res.get("images", [])
    candidates = res.get("candidates", [])

    print("\n" + "=" * 65)
    print(f"OCR SEARCH RESULTS FOR: \"{args.query}\"")
    print(f"Reranker: {'Disabled (RRF Only)' if args.no_rerank else 'Active (mxbai-rerank-large-v1)'}")
    print(f"Matched {len(candidates)} candidate chunks across {len(images)} documents")
    print("=" * 65)

    if not images:
        print("No matching documents found.")
        return

    for idx, img in enumerate(images, start=1):
        page_str = f" (Page {img['page']})" if img['page'] > 1 else ""
        print(f"\n[{idx}] {img['filename']}{page_str}")
        print(f"    Path      : {img['path']}")
        print(f"    Score     : {img['best_score']:.4f} | RRF Score: {img['rrf_score']:.4f}")
        print(f"    Date      : {img.get('date', 'unknown')}")
        snippet = img.get("highlighted_snippet", img["best_chunk"])
        # Strip HTML tags for clean console display
        clean_snippet = snippet.replace("<mark>", "**").replace("</mark>", "**")
        print(f"    Snippet   : {clean_snippet}")
        print("-" * 65)

    if args.show_candidates:
        print("\n" + "=" * 65)
        print("ALL RETRIEVED CANDIDATE CHUNKS:")
        print("=" * 65)
        for idx, cand in enumerate(candidates, start=1):
            print(f"[{idx:02d}] Score: {cand['score']:.4f} | File: {cand['filename']} (p.{cand['page']} c.{cand['chunk_index']})")
            print(f"     Sources: {', '.join(cand['sources'])}")
            print(f"     Text   : {cand['text'][:140]}...\n")


def cmd_clear(args: argparse.Namespace) -> None:
    """Clear visual photo database, OCR database, or both."""
    if not (args.photos or args.ocr or args.all):
        print("Please specify what to clear: --photos, --ocr, or --all")
        return

    try:
        if args.photos or args.all:
            res_p = clear_database()
            print("\n" + "=" * 65)
            print("PHOTOS DATABASE CLEARED:")
            print("=" * 65)
            print(f"Points removed from Qdrant ('{cfg.QDRANT_COLLECTION}'): {res_p['points_removed']}")
            print(f"Cache checkpoints cleared                       : {res_p['cache_entries_removed']}")

        if args.ocr or args.all:
            res_o = clear_ocr_database()
            print("\n" + "=" * 65)
            print("OCR DATABASE CLEARED:")
            print("=" * 65)
            print(f"Points removed from Qdrant ('{cfg.OCR_COLLECTION}'): {res_o['points_removed']}")
            print(f"Cache checkpoints cleared                       : {res_o['db_records_removed']}")

        print("=" * 65)
        print("Clear operation completed successfully.")
    except Exception as e:
        if "already accessed by another instance" in str(e).lower() or "permission denied" in str(e).lower():
            print("\n[NOTE] Database is currently locked by another EmbedEye process.", file=sys.stderr)
            print("Only one process may access the local Qdrant database at a time.", file=sys.stderr)
            print("Please close any active desktop app or CLI process before running this command.", file=sys.stderr)
        else:
            raise


def cmd_status(args: argparse.Namespace) -> None:
    """Print hardware telemetry, VRAM allocation, and database record counts."""
    hw, db, models = SystemStatusReporter.get_status_summary()
    print("\n" + "=" * 65)
    print("EMBEDEYE SYSTEM STATUS & TELEMETRY")
    print("=" * 65)
    # Strip basic markdown for console readability
    for block in [hw, db, models]:
        clean_block = block.replace("**", "").replace("`", "")
        print(clean_block)
        print("-" * 65)


def main(argv: Optional[List[str]] = None) -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="EmbedEye: Local-First Semantic Photo Search & OCR Intelligence CLI",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # 1. index
    p_index = subparsers.add_parser("index", help="Index photos with visual embeddings (OpenCLIP) and captioning (Florence-2)")
    p_index.add_argument("folder", type=str, help="Directory containing images to index")
    p_index.add_argument("--recaption", action="store_true", help="Re-run Florence-2 captioning without re-embedding CLIP vectors")
    p_index.set_defaults(func=cmd_index)

    # 2. search
    p_search = subparsers.add_parser("search", help="Search photos using natural language semantic queries or quotes for OCR text")
    p_search.add_argument("query", type=str, help="Search query (e.g., 'sunset at beach' or \"receipt from 'Store'\")")
    p_search.add_argument("--no-rerank", action="store_true", help="Disable neural cross-encoder reranking")
    p_search.add_argument("--smart-date", action="store_true", help="Enable offline 4-digit year date filter")
    p_search.set_defaults(func=cmd_search)

    # 3. ocr-index
    p_ocr_index = subparsers.add_parser("ocr-index", help="Index documents or images in OCR-only mode (MiniLM dense vectors + full-text)")
    p_ocr_index.add_argument("folder", type=str, help="Directory containing documents/images/PDFs to index")
    p_ocr_index.add_argument("--engine", type=str, default=cfg.OCR_ENGINE, choices=["tesseract", "rapidocr", "florence"], help="OCR engine to use")
    p_ocr_index.add_argument("--workers", type=int, default=cfg.OCR_WORKERS, help="Parallel worker processes for CPU OCR")
    p_ocr_index.add_argument("--min-conf", type=float, default=cfg.OCR_MIN_CONF, help="Minimum line confidence threshold (0.0 to 1.0)")
    p_ocr_index.add_argument("--reindex", action="store_true", help="Force reindexing all files even if unchanged")
    p_ocr_index.set_defaults(func=cmd_ocr_index)

    # 4. ocr-search
    p_ocr_search = subparsers.add_parser("ocr-search", help="Search OCR-indexed documents with hybrid vector + keyword matching")
    p_ocr_search.add_argument("query", type=str, help="Search query (e.g., 'invoice total $500' or exact quote: \"Account Number\")")
    p_ocr_search.add_argument("--top", type=int, default=4, help="Number of top document matches to return")
    p_ocr_search.add_argument("--no-rerank", action="store_true", help="Disable neural cross-encoder reranking (pure RRF order)")
    p_ocr_search.add_argument("--show-candidates", action="store_true", help="Show all retrieved candidate text chunks")
    p_ocr_search.set_defaults(func=cmd_ocr_search)

    # 5. clear
    p_clear = subparsers.add_parser("clear", help="Clear vector collections and SQLite cache checkpoints")
    p_clear.add_argument("--photos", action="store_true", help="Clear 'photos' visual vector collection and checkpoint cache")
    p_clear.add_argument("--ocr", action="store_true", help="Clear 'docs_ocr' document collection and checkpoint cache")
    p_clear.add_argument("--all", action="store_true", help="Clear both 'photos' and 'docs_ocr' collections")
    p_clear.set_defaults(func=cmd_clear)

    # 6. status
    p_status = subparsers.add_parser("status", help="Display hardware status, VRAM telemetry, and collection record counts")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
