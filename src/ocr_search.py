"""OCR-Only Hybrid Search & Cross-Encoder Reranker Module.

This module provides an Object-Oriented pipeline for searching OCR-indexed documents:
1. Dense semantic search on MiniLM 384-d text embeddings.
2. Sparse keyword full-text matching using Qdrant text indexes and exact phrase filters.
3. Hybrid rank fusion using Reciprocal Rank Fusion (RRF, k=60).
4. GPU-isolated neural cross-encoder reranking via 'mixedbread-ai/mxbai-rerank-large-v1'.
5. Multi-page document aggregation and dynamic HTML snippet highlighting.

OOP Architecture:
- `TextHighlighter`: Extracts query terms and wraps matched tokens in `<mark>` tags.
- `RRFMerger`: Combines disparate rank lists into unified candidate scores.
- `CrossEncoderReranker`: Executes GPU-isolated neural reranking using `ModelManager`.
- `DocumentAggregator`: Aggregates chunk candidates by document (file path, page number).
- `OCRSearchEngine` / `OCRSearcher`: Facade and main coordinator for the hybrid search workflow.
"""

from __future__ import annotations

import argparse
import html
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

import src.config as cfg
from src.gpu_manager import ModelManager, get_model_manager
from src.ocr_engine import MiniLMEmbedder, get_minilm_embedder
from src.ocr_indexer import OCRCollectionManager, ensure_ocr_collection, get_ocr_qdrant_client

logger = logging.getLogger("photo_search.ocr_search")


# ============================================================================
# 1. TEXT HIGHLIGHTER COMPONENT
# ============================================================================

class TextHighlighter:
    """Extracts quoted and individual keywords from search queries and formats snippet highlights."""

    @staticmethod
    def extract_quoted_terms(query: str) -> List[str]:
        """Extract phrases enclosed in double quotes.

        Args:
            query: Raw user query string.

        Returns:
            List of quoted phrases with whitespace stripped.
        """
        return [term.strip() for term in re.findall(r'"([^"]+)"', query) if term.strip()]

    @classmethod
    def highlight_snippet(
        cls,
        text: str,
        query: str,
        max_words: int = 80,
    ) -> str:
        """Highlight matched query words and quoted phrases in text using HTML <mark> tags.

        Args:
            text: Full text chunk extracted from document.
            query: Search query used for retrieval.
            max_words: Fallback length limit if no mark tags are generated.

        Returns:
            HTML string with highlighted terms and contextual snippet ellipsis.
        """
        if not text:
            return ""

        quoted = cls.extract_quoted_terms(query)
        clean_q = re.sub(r'"[^"]+"', " ", query)
        words = [w.strip() for w in clean_q.split() if len(w.strip()) >= 2]

        patterns = set(quoted + words)
        if not patterns:
            words_list = text.split()
            return " ".join(words_list[:max_words])

        escaped_text = html.escape(text)
        for p in sorted(patterns, key=len, reverse=True):
            pattern = re.compile(rf"\b({re.escape(p)})\b", re.IGNORECASE)
            escaped_text = pattern.sub(r"<mark>\1</mark>", escaped_text)

        # Center window around the first highlight if present
        if "<mark>" in escaped_text:
            idx = escaped_text.find("<mark>")
            start_char = max(0, idx - 150)
            end_char = min(len(escaped_text), idx + 350)
            prefix = "..." if start_char > 0 else ""
            suffix = "..." if end_char < len(escaped_text) else ""
            return prefix + escaped_text[start_char:end_char] + suffix

        words_list = escaped_text.split()
        return " ".join(words_list[:max_words]) + ("..." if len(words_list) > max_words else "")


# Facades for backwards compatibility
def extract_quoted_terms(query: str) -> List[str]:
    """Facade for TextHighlighter.extract_quoted_terms."""
    return TextHighlighter.extract_quoted_terms(query)


def highlight_snippet(text: str, query: str, max_words: int = 80) -> str:
    """Facade for TextHighlighter.highlight_snippet."""
    return TextHighlighter.highlight_snippet(text, query, max_words=max_words)


# ============================================================================
# 2. RECIPROCAL RANK FUSION (RRF) MERGER
# ============================================================================

class RRFMerger:
    """Merges heterogeneous rank lists using Reciprocal Rank Fusion (RRF).

    Formula: RRF_Score(d) = SUM( 1 / (k + rank_i(d)) ) across all rank lists i.
    """

    def __init__(self, k: int = 60):
        """Initialize merger with smoothing constant k.

        Args:
            k: Smoothing constant balancing top ranks vs lower ranks (default 60).
        """
        self.k = k

    def merge(
        self,
        vector_results: List[Any],
        keyword_results: List[Any],
        top_n: int = 20,
    ) -> List[Dict[str, Any]]:
        """Combine vector hits and keyword hits into sorted candidate list.

        Args:
            vector_results: List of Qdrant points returned by dense vector search.
            keyword_results: List of Qdrant points returned by keyword text search.
            top_n: Number of fused candidates to return.

        Returns:
            List of candidate dictionaries sorted descending by RRF score.
        """
        scores: Dict[str, float] = defaultdict(float)
        point_map: Dict[str, Any] = {}
        sources_map: Dict[str, List[str]] = defaultdict(list)

        # 1. Rank vector hits
        for rank, hit in enumerate(vector_results, start=1):
            pid = str(hit.id)
            scores[pid] += 1.0 / (self.k + rank)
            point_map[pid] = hit
            sources_map[pid].append(f"vector (rank #{rank})")

        # 2. Rank keyword hits
        for rank, hit in enumerate(keyword_results, start=1):
            pid = str(hit.id)
            scores[pid] += 1.0 / (self.k + rank)
            point_map[pid] = hit
            sources_map[pid].append(f"keyword (rank #{rank})")

        # Sort descending by RRF score
        sorted_ids = sorted(scores.keys(), key=lambda pid: scores[pid], reverse=True)

        candidates: List[Dict[str, Any]] = []
        for pid in sorted_ids[:top_n]:
            hit = point_map[pid]
            payload = hit.payload or {}
            candidates.append({
                "id": pid,
                "path": payload.get("path", ""),
                "filename": payload.get("filename", ""),
                "page": payload.get("page", 1),
                "chunk_index": payload.get("chunk_index", 0),
                "text": payload.get("text", ""),
                "full_text_len": payload.get("full_text_len", len(payload.get("text", ""))),
                "date": payload.get("date", ""),
                "mtime": payload.get("mtime", 0.0),
                "rrf_score": scores[pid],
                "sources": sources_map[pid],
                "score": scores[pid],  # Initial score is RRF score
            })

        return candidates


def rrf_merge(
    vector_results: List[Any],
    keyword_results: List[Any],
    k: int = 60,
    top_n: int = 20,
) -> List[Dict[str, Any]]:
    """Facade for RRFMerger.merge."""
    return RRFMerger(k=k).merge(vector_results, keyword_results, top_n=top_n)


# ============================================================================
# 3. NEURAL CROSS-ENCODER RERANKER
# ============================================================================

class CrossEncoderReranker:
    """Manages neural cross-encoder reranking via ModelManager GPU memory isolation."""

    def __init__(self, gpu_manager: Optional[ModelManager] = None):
        """Initialize reranker with GPU ModelManager instance.

        Args:
            gpu_manager: Optional ModelManager instance (defaults to global singleton).
        """
        self.gpu_manager = gpu_manager or get_model_manager()

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        batch_size: int = 8,
    ) -> List[Dict[str, Any]]:
        """Score (query, chunk_text) pairs using mixedbread-ai cross-encoder.

        Args:
            query: User search query.
            candidates: List of candidate dictionaries containing 'text'.
            batch_size: Forward pass batch size.

        Returns:
            Updated candidate list sorted descending by cross-encoder score.
        """
        if not candidates:
            return []

        from src.models import rerank_pairs

        # Strictly isolate reranker: Florence-2 or CLIP are never loaded alongside it
        with self.gpu_manager.use("reranker") as bundle:
            tokenizer, model = bundle
            texts = [c["text"] for c in candidates]
            scores = rerank_pairs(
                query,
                texts,
                batch_size=batch_size,
                reranker_bundle=(tokenizer, model),
            )

            for cand, score in zip(candidates, scores):
                cand["score"] = float(score)
                cand["reranker_score"] = float(score)

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates


# ============================================================================
# 4. DOCUMENT AGGREGATOR
# ============================================================================

class DocumentAggregator:
    """Groups text chunk candidates by document (file path, page number) and selects the best chunk."""

    @staticmethod
    def group_by_document(
        candidates: List[Dict[str, Any]],
        query: str,
        top_k: int = 4,
    ) -> List[Dict[str, Any]]:
        """Group candidates by (path, page) and return top-k documents sorted by score.

        Args:
            candidates: List of candidate chunk dictionaries.
            query: Search query for highlighting snippets.
            top_k: Number of unique documents/pages to return.

        Returns:
            List of aggregated document results with best score, chunk, and snippet.
        """
        grouped: Dict[str, Dict[str, Any]] = {}
        for cand in candidates:
            img_key = f"{cand['path']}__p{cand['page']}"
            if img_key not in grouped or cand["score"] > grouped[img_key]["best_score"]:
                grouped[img_key] = {
                    "path": cand["path"],
                    "filename": cand["filename"],
                    "page": cand["page"],
                    "best_score": cand["score"],
                    "best_chunk": cand["text"],
                    "best_chunk_index": cand["chunk_index"],
                    "full_text_len": cand["full_text_len"],
                    "date": cand["date"],
                    "highlighted_snippet": TextHighlighter.highlight_snippet(cand["text"], query),
                    "sources": cand["sources"],
                    "rrf_score": cand["rrf_score"],
                }

        sorted_images = sorted(
            grouped.values(), key=lambda x: x["best_score"], reverse=True
        )
        return sorted_images[:top_k]


# ============================================================================
# 5. MAIN OCR SEARCH ENGINE
# ============================================================================

class OCRSearchEngine:
    """Hybrid vector + keyword retriever and neural cross-encoder reranker for OCR text."""

    def __init__(self, collection_name: str = cfg.OCR_COLLECTION):
        """Initialize OCRSearchEngine and underlying components.

        Args:
            collection_name: Qdrant collection name to search.
        """
        self.collection_name = collection_name
        self.client = get_ocr_qdrant_client()
        self.collection_mgr = OCRCollectionManager(self.client, self.collection_name)
        self.embedder = get_minilm_embedder()
        self.gpu_manager = get_model_manager()
        self.merger = RRFMerger(k=60)
        self.reranker = CrossEncoderReranker(self.gpu_manager)

    def search(
        self,
        query: str,
        top_k: int = 4,
        use_reranker: bool = True,
        rerank_batch_size: int = 8,
    ) -> Dict[str, Any]:
        """Perform hybrid retrieval, RRF merging, cross-encoder reranking, and document grouping.

        Args:
            query: User search query.
            top_k: Number of top document results to return.
            use_reranker: If True, invoke neural cross-encoder reranker.
            rerank_batch_size: Batch size for cross-encoder inference.

        Returns:
            Dictionary containing:
                - query: Original query string.
                - images: Top-k grouped document dictionaries.
                - candidates: All fused candidate chunk dictionaries.
        """
        clean_query = query.strip()
        if not clean_query:
            return {"query": query, "images": [], "candidates": []}

        # 1. Vector Search on 'text'
        query_vector = self.embedder.embed_query(clean_query)
        try:
            res = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                using="text",
                limit=20,
                with_payload=True,
            )
            vector_hits = res.points
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            vector_hits = []

        # 2. Keyword Full-Text Search on 'text'
        quoted_terms = TextHighlighter.extract_quoted_terms(clean_query)
        unquoted = re.sub(r'"[^"]+"', " ", clean_query).strip()
        kw_terms = [t for t in (quoted_terms + unquoted.split()) if len(t) >= 2]
        kw_query = " ".join(kw_terms) if kw_terms else clean_query

        keyword_hits = []
        try:
            scroll_res, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="text",
                            match=qmodels.MatchText(text=kw_query),
                        )
                    ]
                ),
                limit=40,
                with_payload=True,
                with_vectors=False,
            )

            # Filter exact quoted phrase matches if present
            if quoted_terms:
                for hit in scroll_res:
                    text_lower = (hit.payload.get("text") or "").lower()
                    if all(q.lower() in text_lower for q in quoted_terms):
                        keyword_hits.append(hit)
            else:
                keyword_hits = scroll_res[:20]
        except Exception as e:
            logger.warning(f"Keyword search failed: {e}")
            keyword_hits = []

        # 3. Merge candidates with RRF into top-20
        candidates = self.merger.merge(vector_hits, keyword_hits, top_n=20)
        if not candidates:
            return {"query": query, "images": [], "candidates": []}

        # 4. Cross-Encoder Reranking
        if use_reranker and candidates:
            try:
                candidates = self.reranker.rerank(
                    clean_query, candidates, batch_size=rerank_batch_size
                )
            except Exception as e:
                logger.warning(f"Reranker failed; falling back to RRF order: {e}")

        # 5. Group by Document (path, page)
        top_images = DocumentAggregator.group_by_document(
            candidates, clean_query, top_k=top_k
        )

        return {
            "query": query,
            "images": top_images,
            "candidates": candidates,
        }


# Alias for backwards compatibility
OCRSearcher = OCRSearchEngine


# ============================================================================
# 6. CLI ENTRYPOINT
# ============================================================================

def main() -> None:
    """CLI for querying OCR-only indexed documents."""
    parser = argparse.ArgumentParser(description="OCR-Only Document Search CLI")
    parser.add_argument("query", type=str, help="Search query (use quotes for exact term match)")
    parser.add_argument("--top", type=int, default=4, help="Number of top images to return (default 4)")
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable neural cross-encoder reranker and use pure RRF order",
    )
    parser.add_argument(
        "--show-candidates",
        action="store_true",
        help="Print all candidate chunks and scores",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    searcher = OCRSearchEngine()
    res = searcher.search(args.query, top_k=args.top, use_reranker=not args.no_rerank)

    images = res.get("images", [])
    candidates = res.get("candidates", [])

    print(f"\nSearch Query: \"{args.query}\"")
    print(f"Reranker: {'Disabled (RRF Only)' if args.no_rerank else 'Active (mxbai-rerank-large-v1)'}")
    print(f"Found {len(candidates)} matching candidate chunks across {len(images)} images/documents.\n")

    if not images:
        print("No matching text found.")
        return

    print("=" * 70)
    print(f"TOP {len(images)} MATCHED DOCUMENTS:")
    print("=" * 70)
    for idx, img in enumerate(images, start=1):
        page_str = f" (Page {img['page']})" if img['page'] > 1 else ""
        print(f"\n[{idx}] {img['filename']}{page_str}")
        print(f"    Path:  {img['path']}")
        print(f"    Score: {img['best_score']:.4f} | RRF Score: {img['rrf_score']:.4f}")
        print(f"    Text:  {img['best_chunk']}")

    if args.show_candidates:
        print("\n" + "=" * 70)
        print("ALL RETRIEVED CANDIDATES:")
        print("=" * 70)
        for idx, cand in enumerate(candidates, start=1):
            print(f"[{idx:02d}] Score: {cand['score']:.4f} | File: {cand['filename']} (p.{cand['page']} c.{cand['chunk_index']})")
            print(f"     Sources: {', '.join(cand['sources'])}")
            print(f"     Snippet: {cand['text'][:120]}...\n")


if __name__ == "__main__":
    main()
