"""Configuration, environment setup, and hardware utilities for EmbedEye.

HARD REQUIREMENT: All models live inside ./models/ in this project and nowhere else.
This module sets the necessary cache environment variables BEFORE any deep learning
libraries are initialized.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

# Define root paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
QDRANT_DIR = DATA_DIR / "qdrant"

# Specific sub-cache directories under ./models/
OPENCLIP_CACHE_DIR = MODELS_DIR / "open_clip"
TRANSFORMERS_CACHE_DIR = MODELS_DIR / "transformers"
RERANKER_CACHE_DIR = MODELS_DIR / "reranker"
SENTENCE_TRANSFORMERS_CACHE_DIR = MODELS_DIR / "sentence_transformers"
TORCH_CACHE_DIR = MODELS_DIR / "torch"
HF_CACHE_DIR = MODELS_DIR / "huggingface"

# Ensure models and data directories exist
for d in [
    MODELS_DIR,
    DATA_DIR,
    QDRANT_DIR,
    OPENCLIP_CACHE_DIR,
    TRANSFORMERS_CACHE_DIR,
    RERANKER_CACHE_DIR,
    SENTENCE_TRANSFORMERS_CACHE_DIR,
    TORCH_CACHE_DIR,
    HF_CACHE_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)


def setup_environment() -> None:
    """Configure environment variables so all models and caches strictly live in ./models/."""
    # Absolute paths
    models_path_str = str(MODELS_DIR.resolve())
    hf_path_str = str(HF_CACHE_DIR.resolve())
    torch_path_str = str(TORCH_CACHE_DIR.resolve())
    st_path_str = str(SENTENCE_TRANSFORMERS_CACHE_DIR.resolve())
    tf_path_str = str(TRANSFORMERS_CACHE_DIR.resolve())

    os.environ["HF_HOME"] = hf_path_str
    os.environ["HF_HUB_CACHE"] = hf_path_str
    os.environ["TRANSFORMERS_CACHE"] = tf_path_str
    os.environ["TORCH_HOME"] = torch_path_str
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = st_path_str
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    # Load project .env if present
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        load_dotenv(dotenv_path=env_file, override=False)

    # Check offline flag from environment
    if os.getenv("HF_HUB_OFFLINE", "0").lower() in ("1", "true", "yes"):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


# Initialize environment immediately upon import
setup_environment()

# Model constants
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
CLIP_DIM = 512

# Florence-2 Model Configuration & Pinned Commit Revisions (for reproducible, secure loading)
FLORENCE_REVISIONS: Dict[str, str] = {
    "microsoft/Florence-2-large": "21a599d414c4d928c9032694c424fb94458e3594",
    "microsoft/Florence-2-base": "5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac",
}

FLORENCE_MODEL_NAME = os.getenv("FLORENCE_MODEL_NAME", "microsoft/Florence-2-large").strip()
FLORENCE_REVISION = os.getenv(
    "FLORENCE_REVISION",
    FLORENCE_REVISIONS.get(FLORENCE_MODEL_NAME, "21a599d414c4d928c9032694c424fb94458e3594"),
).strip()
FLORENCE_MAX_NEW_TOKENS = int(os.getenv("FLORENCE_MAX_NEW_TOKENS", "256"))
FLORENCE_NUM_BEAMS = int(os.getenv("FLORENCE_NUM_BEAMS", "3"))

# Optional / legacy BLIP model configuration
BLIP_MODEL_NAME = os.getenv("BLIP_MODEL_NAME", "Salesforce/blip-image-captioning-large")
BLIP_MAX_NEW_TOKENS = 30
BLIP_NUM_BEAMS = 3

# Local Neural Reranker Configuration (mixedbread-ai/mxbai-rerank-large-v1)
RERANKER_MODEL_NAME = os.getenv(
    "RERANKER_MODEL_NAME", "mixedbread-ai/mxbai-rerank-large-v1"
).strip()
RERANKER_BATCH_SIZE = int(os.getenv("RERANKER_BATCH_SIZE", "16"))

QDRANT_COLLECTION = "photos"

# Batch sizes for RTX 3050 4GB VRAM
CLIP_BATCH_SIZE = int(os.getenv("CLIP_BATCH_SIZE", "16"))
FLORENCE_BATCH_SIZE = int(os.getenv("FLORENCE_BATCH_SIZE", "2"))
BLIP_BATCH_SIZE = 4

# OCR Configuration
USE_RAPIDOCR = os.getenv("USE_RAPIDOCR", "true").lower() in ("1", "true", "yes")
MAX_OCR_CHARS = int(os.getenv("MAX_OCR_CHARS", "1000"))

# OCR-Only Mode Configuration
OCR_ENGINE = os.getenv("OCR_ENGINE", "tesseract").strip().lower()
OCR_WORKERS = int(os.getenv("OCR_WORKERS", "2"))
OCR_MIN_CONF = float(os.getenv("OCR_MIN_CONF", "0.5"))
OCR_MAX_SIDE = int(os.getenv("OCR_MAX_SIDE", "2000"))
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "250"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "30"))
OCR_COLLECTION = os.getenv("OCR_COLLECTION", "docs_ocr").strip()
OCR_LANGS = os.getenv("OCR_LANGS", "en").strip()
TESSERACT_CMD = os.getenv(
    "TESSERACT_CMD",
    str((MODELS_DIR / "tesseract" / "tesseract.exe").resolve()),
).strip()
MINILM_MODEL_NAME = os.getenv("MINILM_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2").strip()
MINILM_DIM = 384



def is_offline_mode() -> bool:
    """Return True if HF_HUB_OFFLINE is set to true/1."""
    return os.getenv("HF_HUB_OFFLINE", "0").lower() in ("1", "true", "yes")


def set_offline_mode(enabled: bool) -> None:
    """Toggle offline mode dynamically."""
    if enabled:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)


def get_device() -> str:
    """Detect CUDA availability and return 'cuda' or 'cpu'.

    Supports DEVICE_OVERRIDE env var for testing.
    """
    override = os.getenv("DEVICE_OVERRIDE", "auto").strip().lower()
    if override in ("cpu", "cuda"):
        return override

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def is_cuda_available() -> bool:
    """Return True if active device is CUDA."""
    return get_device() == "cuda"


def get_device_status_info() -> Dict[str, Any]:
    """Return detailed human-readable and structured status of device and VRAM."""
    device = get_device()
    info: Dict[str, Any] = {
        "device": device,
        "is_cuda": device == "cuda",
        "device_name": "CPU",
        "vram_allocated_mb": 0.0,
        "vram_reserved_mb": 0.0,
        "vram_total_mb": 0.0,
        "vram_free_mb": 0.0,
        "status_text": "Running on CPU (CUDA unavailable or disabled)",
    }

    if device == "cuda":
        try:
            import torch

            gpu_name = torch.cuda.get_device_name(0)
            allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
            reserved = torch.cuda.memory_reserved(0) / (1024 * 1024)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            free = total - allocated

            info["device_name"] = gpu_name
            info["vram_allocated_mb"] = round(allocated, 1)
            info["vram_reserved_mb"] = round(reserved, 1)
            info["vram_total_mb"] = round(total, 1)
            info["vram_free_mb"] = round(free, 1)
            info["status_text"] = (
                f"GPU Active: {gpu_name} | VRAM: {allocated:.1f} MB allocated / {total:.1f} MB total"
            )
        except Exception as e:
            info["status_text"] = f"GPU detected but query failed: {e}"

    return info
