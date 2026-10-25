#!/usr/bin/env python3
"""Script to pre-download all model weights into the local ./models/ directory.

HARD REQUIREMENT: All models live inside ./models/ and nowhere else.
Environment variables are set BEFORE any deep learning library imports.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# --- HARD REQUIREMENT: SET ENV VARS BEFORE ANY AI LIBRARIES ARE IMPORTED ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # Suppress Windows symlink warning

# Now safe to import from src.config
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import (
    BLIP_MODEL_NAME,
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    FLORENCE_MODEL_NAME,
    FLORENCE_REVISIONS,
    OPENCLIP_CACHE_DIR,
    RERANKER_CACHE_DIR,
    RERANKER_MODEL_NAME,
    TRANSFORMERS_CACHE_DIR,
    MINILM_MODEL_NAME,
    SENTENCE_TRANSFORMERS_CACHE_DIR,
)


def get_dir_size_str(path: Path) -> str:
    """Calculate total size of directory and return human-readable string."""
    if not path.exists():
        return "0 B"
    total_bytes = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total_bytes += os.path.getsize(fp)
            except OSError:
                pass
    if total_bytes < 1024:
        return f"{total_bytes} B"
    elif total_bytes < 1024 * 1024:
        return f"{total_bytes / 1024:.1f} KB"
    elif total_bytes < 1024 * 1024 * 1024:
        return f"{total_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{total_bytes / (1024 * 1024 * 1024):.2f} GB"


def download_openclip() -> None:
    """Download OpenCLIP ViT-B-32 weights and tokenizer into ./models/open_clip/."""
    print(f"\n[1/3] Downloading OpenCLIP ({CLIP_MODEL_NAME}, pretrained={CLIP_PRETRAINED})...")
    print(f"      Target cache: {OPENCLIP_CACHE_DIR}")
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME,
        pretrained=CLIP_PRETRAINED,
        cache_dir=str(OPENCLIP_CACHE_DIR),
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    del model, preprocess, tokenizer
    print("      OpenCLIP downloaded and verified successfully!")


def download_florence(model_name: str = FLORENCE_MODEL_NAME) -> None:
    """Download Florence-2 (captioning + OCR) model into ./models/transformers/."""
    rev = FLORENCE_REVISIONS.get(model_name, "21a599d414c4d928c9032694c424fb94458e3594")
    print(f"\n[2/3] Downloading Florence-2 ({model_name}, rev={rev[:8]}...)...")
    print(f"      Target cache: {TRANSFORMERS_CACHE_DIR}")
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_name,
        revision=rev,
        trust_remote_code=True,
        cache_dir=str(TRANSFORMERS_CACHE_DIR),
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=rev,
        trust_remote_code=True,
        cache_dir=str(TRANSFORMERS_CACHE_DIR),
    )
    del processor, model
    print(f"      Florence-2 ({model_name}) downloaded and verified successfully!")


def download_reranker(model_name: str = RERANKER_MODEL_NAME) -> None:
    """Download mixedbread-ai neural cross-encoder into ./models/reranker/."""
    print(f"\n[3/3] Downloading Neural Reranker ({model_name})...")
    print(f"      Target cache: {RERANKER_CACHE}")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(RERANKER_CACHE),
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        cache_dir=str(RERANKER_CACHE),
    )
    del tokenizer, model
    print(f"      Neural Reranker ({model_name}) downloaded and verified successfully!")


def download_blip(model_name: str = BLIP_MODEL_NAME) -> None:
    """Download optional legacy BLIP captioning model into ./models/transformers/."""
    print(f"\n[Optional] Downloading legacy BLIP Captioner ({model_name})...")
    print(f"           Target cache: {TRANSFORMERS_CACHE_DIR}")
    from transformers import BlipForConditionalGeneration, BlipProcessor

    processor = BlipProcessor.from_pretrained(
        model_name,
        cache_dir=str(TRANSFORMERS_CACHE_DIR),
    )
    model = BlipForConditionalGeneration.from_pretrained(
        model_name,
        cache_dir=str(TRANSFORMERS_CACHE_DIR),
    )
    del processor, model
    print(f"           BLIP ({model_name}) downloaded and verified successfully!")


def download_minilm(model_name: str = MINILM_MODEL_NAME) -> None:
    """Download sentence-transformers all-MiniLM-L6-v2 into ./models/sentence_transformers/."""
    print(f"\n[OCR] Downloading MiniLM text embedder ({model_name})...")
    print(f"      Target cache: {SENTENCE_TRANSFORMERS_CACHE}")
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(SENTENCE_TRANSFORMERS_CACHE),
    )
    model = AutoModel.from_pretrained(
        model_name,
        cache_dir=str(SENTENCE_TRANSFORMERS_CACHE),
    )
    del tokenizer, model
    print(f"      MiniLM ({model_name}) downloaded and verified successfully!")


def download_rapidocr() -> None:
    """Initialize RapidOCR engine to verify and fetch ONNX models."""
    print("\n[OCR] Verifying RapidOCR engine and ONNX models...")
    import numpy as np
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR()
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    engine(dummy)
    print("      RapidOCR verified successfully!")


def print_size_summary() -> None:
    """Print size of each cache directory and the overall models/ directory."""
    print("\n" + "=" * 60)
    print("LOCAL MODEL DIRECTORY SUMMARY (./models/):")
    print("=" * 60)
    for name, p in [
        ("OpenCLIP", OPENCLIP_CACHE),
        ("Transformers (Florence-2)", TRANSFORMERS_CACHE),
        ("Neural Reranker", RERANKER_CACHE),
        ("MiniLM (Sentence-Transformers)", SENTENCE_TRANSFORMERS_CACHE),
        ("Torch Core Cache", TORCH_CACHE),
        ("HuggingFace Hub Cache", HF_CACHE),
    ]:
        print(f"  {name:30}: {get_dir_size_str(p):>10}")
    print("-" * 60)
    print(f"  TOTAL ./models/ SIZE          : {get_dir_size_str(MODELS_DIR):>10}")
    print("=" * 60)
    print("All models are self-contained in ./models/ for 100% local and offline use.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download all models into ./models/")
    parser.add_argument(
        "--ocr-mode",
        action="store_true",
        help="Download only models required for OCR-Only mode (RapidOCR, MiniLM, Reranker)",
    )
    parser.add_argument(
        "--reranker-only",
        action="store_true",
        help="Only download the mixedbread-ai/mxbai-rerank-large-v1 model",
    )
    parser.add_argument(
        "--florence-model",
        type=str,
        default=FLORENCE_MODEL_NAME,
        help="Florence-2 model name (default: microsoft/Florence-2-large or from .env)",
    )
    parser.add_argument(
        "--also-florence-base",
        action="store_true",
        help="Also pre-download microsoft/Florence-2-base for fast/lightweight testing",
    )
    parser.add_argument(
        "--also-blip",
        action="store_true",
        help="Also pre-download legacy Salesforce/blip-image-captioning-large",
    )
    args = parser.parse_args()

    print("============================================================")
    print("Local Semantic Photo Search - Model Pre-downloader")
    print("Target: NVIDIA RTX 3050 4GB VRAM (Single-Model GPU Isolation)")
    print(f"Root Models Directory: {MODELS_DIR.resolve()}")
    print("============================================================")

    if args.ocr_mode:
        print("\n--- Running in --ocr-mode: Fetching RapidOCR, MiniLM, and Reranker ---")
        download_rapidocr()
        download_minilm()
        download_reranker()
        print_size_summary()
        return

    if args.reranker_only:
        download_reranker()
        print_size_summary()
        return

    download_openclip()
    download_florence(args.florence_model)
    if args.also_florence_base and args.florence_model != "microsoft/Florence-2-base":
        download_florence("microsoft/Florence-2-base")
    if args.also_blip:
        download_blip()
    download_minilm()
    download_reranker()

    print_size_summary()


if __name__ == "__main__":
    main()

