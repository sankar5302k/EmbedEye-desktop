"""Search pipeline combining OpenCLIP and local mixedbread-ai neural cross-encoder.

Strict GPU Memory Management:
- Only OpenCLIP + Cross-Encoder Reranker may be loaded on GPU at search time (~1.5GB total).
- Florence-2 is NEVER loaded at search time.
- Supports keyword OCR full-text search when queries contain quotes (e.g. 'Reliance').
- Guarantees CLIP rank #1 is always preserved in the final top 4.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

import src.config as cfg
from src.gpu_manager import get_model_manager
from src.indexer import ensure_collection, get_qdrant_client
from src.models import embed_text_clip, rerank_pairs

logger = logging.getLogger("photo_search.search")


class SearchEngine:
    """Manages query embedding, Qdrant vector retrieval, OCR text search, and neural reranking."""

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        self.device = device or cfg.get_device()
        self.client = get_qdrant_client(storage_path)
        ensure_collection(self.client)
        self.manager = get_model_manager(device=self.device)

    def parse_offline_date(self, query: str) -> Tuple[str, Optional[str], Optional[str]]:
        """Extract 4-digit years from query locally without online LLM.

        Example: 'family vacation 2024' -> ('family vacation', '2024-01-01', '2024-12-31')
        """
        match = re.search(r"\b(19\d\d|20\d\d)\b", query)
        if match:
            year = match.group(1)
            clean_q = re.sub(rf"\b{year}\b", "", query).strip()
            clean_q = re.sub(r"\s+", " ", clean_q).strip()
            date_from = f"{year}-01-01"
            date_to = f"{year}-12-31"
            return clean_q or query, date_from, date_to
        return query, None, None

    def build_date_filter(
        self, date_from: Optional[str], date_to: Optional[str]
    ) -> Optional[qmodels.Filter]:
        """Build Qdrant range filter on the 'date' payload field."""
        if not date_from and not date_to:
            return None

        range_cond = qmodels.DatetimeRange(
            gte=f"{date_from}T00:00:00" if date_from else None,
            lte=f"{date_to}T23:59:59" if date_to else None,
        )

        return qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="date",
                    range=range_cond,
                )
            ]
        )

    def extract_quoted_terms(self, query: str) -> List[str]:
        """Extract quoted words or phrases from query (e.g. \"bill from 'Reliance'\" -> ['Reliance'])."""
        matches = re.findall(r"['\"]([^'\"]+)['\"]", query)
        terms = [m.strip() for m in matches if m.strip()]
        return terms

    def search(
        self,
        query: str,
        use_rerank: bool = True,
        use_smart_date: bool = False,
        top_k: int = 4,
    ) -> Dict[str, Any]:
        """Execute visual search with optional OCR keyword boost and neural cross-encoder reranking."""
        clean_query = query.strip()
        date_from = None
        date_to = None
        date_parsing_applied = False

        if not clean_query:
            return {
                "results": [],
                "all_candidates": [],
                "rerank_used": False,
                "badge": "No query",
                "clean_query": "",
                "date_range": None,
            }

        # Step 1: Parse quoted terms for OCR text match
        quoted_terms = self.extract_quoted_terms(clean_query)

        # Step 2: Local offline date extraction if enabled
        if use_smart_date:
            clean_query, date_from, date_to = self.parse_offline_date(clean_query)
            if date_from or date_to:
                date_parsing_applied = True
                logger.info(f"Offline date parsed: from {date_from} to {date_to}")

        date_filter = self.build_date_filter(date_from, date_to) if date_parsing_applied else None

        # Step 3: Run with GPU Manager (CLIP + Reranker coexisting; Florence NEVER loaded)
        with self.manager.use_search() as (clip_bundle, reranker_bundle):
            clip_model, _, clip_tokenizer = clip_bundle

            # Embed query text
            clip_vector = embed_text_clip(
                text=clean_query,
                clip_model=clip_model,
                tokenizer=clip_tokenizer,
                device=self.device,
            ).tolist()

            # Execute Qdrant vector search on 'clip' named vector
            def execute_qdrant_query(filter_obj: Optional[qmodels.Filter]) -> List[Any]:
                res = self.client.query_points(
                    collection_name=cfg.QDRANT_COLLECTION,
                    query=clip_vector,
                    using="clip",
                    limit=16,
                    query_filter=filter_obj,
                    with_payload=True,
                )
                return res.points

            try:
                points = execute_qdrant_query(date_filter)
                if not points and date_filter is not None:
                    logger.info("Date filter yielded 0 results; retrying search without date filter.")
                    points = execute_qdrant_query(None)
                    date_parsing_applied = False
            except Exception as e:
                logger.error(f"Qdrant query failed: {e}")
                points = []

            # Step 4: If quoted words exist, run Qdrant full-text match on ocr_text
            ocr_text_points: List[Any] = []
            if quoted_terms:
                try:
                    logger.info(f"Executing Qdrant full-text match on 'ocr_text' for quoted terms: {quoted_terms}")
                    text_filter = qmodels.Filter(
                        should=[
                            qmodels.FieldCondition(
                                key="ocr_text",
                                match=qmodels.MatchText(text=term),
                            )
                            for term in quoted_terms
                        ]
                    )
                    records, _ = self.client.scroll(
                        collection_name=cfg.QDRANT_COLLECTION,
                        scroll_filter=text_filter,
                        limit=16,
                        with_payload=True,
                        with_vectors=False,
                    )
                    ocr_text_points = records
                except Exception as e:
                    logger.warning(f"Qdrant OCR text search failed: {e}")

            # Merge points: visual hits + OCR text hits
            seen_ids = set()
            candidates: List[Dict[str, Any]] = []

            for rank, p in enumerate(points, 1):
                p_id = str(p.id)
                seen_ids.add(p_id)
                payload = p.payload or {}
                candidates.append(
                    {
                        "rank": rank,
                        "id": p_id,
                        "score": getattr(p, "score", 0.0),
                        "path": payload.get("path", ""),
                        "filename": payload.get("filename", ""),
                        "caption": payload.get("caption", ""),
                        "ocr_text": payload.get("ocr_text", ""),
                        "has_text": payload.get("has_text", False),
                        "date": payload.get("date", "unknown"),
                        "mtime": payload.get("mtime", 0.0),
                        "size": payload.get("size", 0),
                        "is_clip_rank_1": (rank == 1),
                    }
                )

            # Append OCR text matches that weren't already included
            for otp in ocr_text_points:
                otp_id = str(otp.id)
                if otp_id not in seen_ids:
                    seen_ids.add(otp_id)
                    payload = otp.payload or {}
                    candidates.append(
                        {
                            "rank": len(candidates) + 1,
                            "id": otp_id,
                            "score": 0.50,  # Baseline score for text match
                            "path": payload.get("path", ""),
                            "filename": payload.get("filename", ""),
                            "caption": payload.get("caption", ""),
                            "ocr_text": payload.get("ocr_text", ""),
                            "has_text": payload.get("has_text", False),
                            "date": payload.get("date", "unknown"),
                            "mtime": payload.get("mtime", 0.0),
                            "size": payload.get("size", 0),
                            "is_clip_rank_1": False,
                        }
                    )

            if not candidates:
                return {
                    "results": [],
                    "all_candidates": [],
                    "rerank_used": False,
                    "badge": "No matches found in database",
                    "clean_query": clean_query,
                    "date_range": None,
                }

            # Remember CLIP rank #1
            clip_rank_1 = candidates[0] if candidates else None

            # Step 5: Local Neural Reranking (mixedbread-ai/mxbai-rerank-large-v1)
            rerank_used = False
            device_label = "CUDA" if cfg.is_cuda_available() else "CPU"
            badge = f"Visual Search (OpenCLIP ViT-B-32) [{device_label}]"
            final_top4: List[Dict[str, Any]] = []

            if use_rerank and candidates:
                try:
                    # Strict format required:
                    # "Caption: {caption} | Text in image: {ocr_text[:200]} | Date: {date}"
                    docs = [
                        f"Caption: {c.get('caption', '')} | "
                        f"Text in image: {c.get('ocr_text', '')[:200]} | "
                        f"Date: {c.get('date', 'unknown')}"
                        for c in candidates
                    ]

                    scores = rerank_pairs(
                        clean_query,
                        docs,
                        reranker_bundle=reranker_bundle,
                    )

                    if scores and len(scores) == len(candidates):
                        rerank_used = True
                        badge = f"Neural Reranker (mxbai-rerank-large-v1) [{device_label}]"

                        scored_candidates = []
                        for cand, sc in zip(candidates, scores):
                            cand_copy = dict(cand)
                            cand_copy["rerank_score"] = sc
                            cand_copy["score"] = sc
                            scored_candidates.append(cand_copy)

                        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
                        candidates = scored_candidates

                        top_subset = candidates[:top_k]

                        # Rule: Always keep CLIP rank #1 in the final top results even if the reranker drops it
                        if clip_rank_1 is not None and top_k > 0:
                            top_ids = {c["id"] for c in top_subset}
                            if clip_rank_1["id"] not in top_ids:
                                if len(top_subset) >= top_k:
                                    top_subset = top_subset[: top_k - 1] + [clip_rank_1]
                                else:
                                    top_subset.append(clip_rank_1)

                        # Reassign 1-based ranks
                        for r, c in enumerate(top_subset, 1):
                            c["rank"] = r

                        for r, c in enumerate(candidates, 1):
                            c["rank"] = r

                        final_top4 = top_subset
                except Exception as e:
                    logger.warning(
                        f"Local neural reranking failed ({e}). Falling back to visual CLIP ordering."
                    )
                    final_top4 = []

            if not final_top4:
                final_top4 = candidates[:top_k]
                for r, c in enumerate(final_top4, 1):
                    c["rank"] = r

            return {
                "results": final_top4,
                "all_candidates": candidates,
                "rerank_used": rerank_used,
                "badge": badge,
                "clean_query": clean_query,
                "date_range": (
                    f"{date_from} to {date_to}"
                    if (date_parsing_applied and (date_from or date_to))
                    else None
                ),
            }


def main() -> None:
    parser = argparse.ArgumentParser(description="Search indexed photos by semantic prompt")
    parser.add_argument("query", type=str, help="Search query (e.g. 'red car near a beach' or \"bill from 'Reliance'\")")
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable local neural cross-encoder reranking",
    )
    parser.add_argument(
        "--smart-date",
        action="store_true",
        help="Enable offline smart date parsing",
    )
    args = parser.parse_args()

    engine = SearchEngine()
    output = engine.search(
        args.query,
        use_rerank=not args.no_rerank,
        use_smart_date=args.smart_date,
    )

    print("\n" + "=" * 60)
    print(f"SEARCH RESULTS for: '{args.query}'")
    print(f"Mode / Badge      : {output['badge']}")
    if output["date_range"]:
        print(f"Date Filter       : {output['date_range']}")
    print("=" * 60)

    for i, res in enumerate(output["results"], 1):
        print(f"[{i}] {res['filename']}")
        print(f"    Path    : {res['path']}")
        print(f"    Date    : {res['date']}")
        print(f"    Score   : {res.get('score', 0.0):.4f}")
        print(f"    Caption : {res['caption']}")
        print(f"    OCR Text: {res.get('ocr_text', '')}")
        print("-" * 60)


if __name__ == "__main__":
    main()
