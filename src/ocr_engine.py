"""OCR Engine & Text Embedder for OCR-Only Mode.

Supports Tesseract, RapidOCR (CPU onnxruntime, 0 VRAM), and Florence-2 (GPU isolated).
Includes EXIF normalization, resizing, PDF page extraction, confidence filtering,
text chunking, and CPU MiniLM (384-d) dense embedding.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageOps

import src.config as cfg

logger = logging.getLogger("photo_search.ocr_engine")

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
SUPPORTED_DOC_EXTS = SUPPORTED_IMAGE_EXTS | {".pdf"}


def is_supported_ocr_file(path: Union[str, Path]) -> bool:
    """Return True if path has an image or PDF extension supported by OCR mode."""
    p = Path(path)
    return p.suffix.lower() in SUPPORTED_DOC_EXTS


def preprocess_image(img: Image.Image, max_side: int = cfg.OCR_MAX_SIDE) -> Image.Image:
    """Normalize EXIF orientation, convert to RGB, and cap longest side."""
    try:
        img = ImageOps.exif_transpose(img)
    except Exception as e:
        logger.debug(f"EXIF transpose failed: {e}")

    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    return img


def clean_and_validate_text(lines: List[str], min_alnum: int = 4) -> str:
    """Normalize whitespace and discard images where total alphanumeric count is under threshold."""
    cleaned_lines = []
    for line in lines:
        cleaned = re.sub(r"[ \t]+", " ", line).strip()
        if cleaned:
            cleaned_lines.append(cleaned)

    full_text = "\n".join(cleaned_lines).strip()
    alnum_count = sum(1 for c in full_text if c.isalnum())
    if alnum_count < min_alnum:
        return ""
    return full_text


def chunk_text(
    text: str,
    chunk_words: int = cfg.CHUNK_WORDS,
    chunk_overlap: int = cfg.CHUNK_OVERLAP,
) -> List[str]:
    """Chunk OCR text into ~200-300 word chunks with ~30 words of overlap."""
    words = text.split()
    if not words:
        return []

    if len(words) <= chunk_words:
        return [text]

    step = max(1, chunk_words - chunk_overlap)
    chunks = []
    for i in range(0, len(words), step):
        chunk_words_slice = words[i : i + chunk_words]
        if not chunk_words_slice:
            break
        chunk = " ".join(chunk_words_slice)
        chunks.append(chunk)
        if i + chunk_words >= len(words):
            break

    return chunks


# ---------------------------------------------------------------------------
# OCR Runners: Tesseract, RapidOCR, Florence-2
# ---------------------------------------------------------------------------

_RAPIDOCR_INSTANCE: Optional[Any] = None


def get_rapidocr_engine() -> Any:
    """Get or create singleton RapidOCR instance."""
    global _RAPIDOCR_INSTANCE
    if _RAPIDOCR_INSTANCE is None:
        from rapidocr_onnxruntime import RapidOCR

        _RAPIDOCR_INSTANCE = RapidOCR()
    return _RAPIDOCR_INSTANCE


def run_rapidocr(img: Image.Image, min_conf: float = cfg.OCR_MIN_CONF) -> str:
    """Extract text using RapidOCR (CPU, ONNX, zero VRAM)."""
    engine = get_rapidocr_engine()
    np_img = np.array(img)
    result, _ = engine(np_img)
    if not result:
        return ""

    valid_lines = []
    for item in result:
        # RapidOCR format: [box, text, score]
        if len(item) >= 3:
            text = str(item[1]).strip()
            score = float(item[2])
            if score >= min_conf and text:
                valid_lines.append(text)

    return clean_and_validate_text(valid_lines)


_LOGGED_TESS_WARNING = False


def _find_tesseract_binary() -> Optional[str]:
    """Locate tesseract executable from ./models/tesseract, config, or system paths."""
    candidates = [
        str((cfg.MODELS_DIR / "tesseract" / "tesseract.exe").resolve()),
        cfg.TESSERACT_CMD,
        r"C:\Users\Admin\Tesseract-OCR\tesseract.exe",
        shutil.which("tesseract"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for c in candidates:
        if c and os.path.exists(c) and os.path.isfile(c):
            tessdata_dir = Path(c).parent / "tessdata"
            if tessdata_dir.exists() and "TESSDATA_PREFIX" not in os.environ:
                os.environ["TESSDATA_PREFIX"] = str(tessdata_dir.resolve())
            return c
    return None


def run_tesseract(
    img: Image.Image,
    min_conf: float = cfg.OCR_MIN_CONF,
    lang: str = cfg.OCR_LANGS,
) -> str:
    """Extract text using Tesseract via pytesseract with line/word confidence filtering."""
    global _LOGGED_TESS_WARNING

    try:
        import pytesseract
    except ImportError:
        if not _LOGGED_TESS_WARNING:
            logger.warning("pytesseract is not installed; falling back to RapidOCR.")
            _LOGGED_TESS_WARNING = True
        return run_rapidocr(img, min_conf)

    tess_bin = _find_tesseract_binary()
    if not tess_bin:
        if not _LOGGED_TESS_WARNING:
            logger.warning(
                "Tesseract executable not found. Configure TESSERACT_CMD or install Tesseract. "
                "Falling back to RapidOCR."
            )
            _LOGGED_TESS_WARNING = True
        return run_rapidocr(img, min_conf)

    pytesseract.pytesseract.tesseract_cmd = tess_bin

    # Map language codes (e.g., 'en' -> 'eng')
    lang_map = {"en": "eng", "fr": "fra", "de": "deu", "es": "spa", "it": "ita"}
    tess_lang = lang_map.get(lang.lower(), lang)

    try:
        data = pytesseract.image_to_data(
            img, lang=tess_lang, output_type=pytesseract.Output.DICT
        )
    except Exception as e:
        logger.warning(f"Tesseract execution error: {e}. Falling back to RapidOCR.")
        return run_rapidocr(img, min_conf)

    # Group words into lines by (block_num, par_num, line_num)
    lines_map: Dict[Tuple[int, int, int], List[Tuple[str, float]]] = {}
    n_boxes = len(data.get("text", []))

    for i in range(n_boxes):
        text = data["text"][i].strip()
        conf_val = data["conf"][i]
        try:
            conf = float(conf_val)
        except (ValueError, TypeError):
            conf = -1.0

        if not text or conf < 0:
            continue

        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        if key not in lines_map:
            lines_map[key] = []
        lines_map[key].append((text, conf))

    # Filter lines where average confidence is >= min_conf * 100
    target_conf = min_conf * 100.0 if min_conf <= 1.0 else min_conf
    accepted_lines = []
    for key, words in lines_map.items():
        if not words:
            continue
        avg_conf = sum(c for _, c in words) / len(words)
        if avg_conf >= target_conf:
            line_str = " ".join(t for t, _ in words)
            accepted_lines.append(line_str)

    return clean_and_validate_text(accepted_lines)


def run_florence_ocr(img: Image.Image) -> str:
    """Extract text using Florence-2 <OCR> task via ModelManager (isolated on GPU)."""
    from src.gpu_manager import get_model_manager

    manager = get_model_manager()
    with manager.use("florence") as bundle:
        model, processor = bundle
        device = manager.device
        dtype = model.dtype if hasattr(model, "dtype") else None

        prompt = "<OCR>"
        inputs = processor(text=prompt, images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if dtype and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)

        import torch

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs.get("pixel_values"),
                max_new_tokens=cfg.FLORENCE_MAX_NEW_TOKENS,
                num_beams=cfg.FLORENCE_NUM_BEAMS,
                do_sample=False,
            )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            generated_text, task=prompt, image_size=(img.width, img.height)
        )
        raw_ocr = parsed.get("<OCR>", "")
        return clean_and_validate_text(raw_ocr.splitlines())


def extract_text_from_image(
    img: Image.Image,
    engine: str = cfg.OCR_ENGINE,
    min_conf: float = cfg.OCR_MIN_CONF,
) -> str:
    """Run OCR on a PIL image using the specified engine with preprocessing."""
    preprocessed = preprocess_image(img)
    engine_name = engine.lower().strip()

    if engine_name == "florence":
        return run_florence_ocr(preprocessed)
    elif engine_name == "tesseract":
        return run_tesseract(preprocessed, min_conf=min_conf)
    else:
        # Default: rapidocr
        return run_rapidocr(preprocessed, min_conf=min_conf)


def extract_pages_from_file(
    file_path: Union[str, Path]
) -> Generator[Tuple[int, Image.Image], None, None]:
    """Yield (page_number_1_based, PIL_Image) for an image or PDF file.

    Never crashes the batch on corrupt or unreadable files.
    """
    path = Path(file_path)
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return

    ext = path.suffix.lower()

    if ext == ".pdf":
        try:
            import pypdfium2 as pdfium

            pdf = pdfium.PdfDocument(str(path))
            n_pages = len(pdf)
            for page_idx in range(n_pages):
                page_num = page_idx + 1
                try:
                    page = pdf[page_idx]
                    # Render at scale 2.0 (~144 dpi) for crisp OCR
                    bitmap = page.render(scale=2.0)
                    pil_img = bitmap.to_pil()
                    yield page_num, pil_img
                except Exception as pe:
                    logger.warning(f"Error rendering page {page_num} of {path}: {pe}")
            pdf.close()
        except Exception as e:
            logger.warning(f"Error opening PDF {path}: {e}")
    elif ext in SUPPORTED_IMAGE_EXTS:
        try:
            with Image.open(path) as img:
                # Load image data into memory before returning
                img.load()
                yield 1, img.copy()
        except Exception as e:
            logger.warning(f"Corrupt or unreadable image {path}: {e}")
    else:
        logger.debug(f"Skipping unsupported file extension: {path}")


# ---------------------------------------------------------------------------
# Sentence-Transformers all-MiniLM-L6-v2 CPU Text Embedder (384-d)
# ---------------------------------------------------------------------------

class MiniLMEmbedder:
    """Direct AutoModel + AutoTokenizer CPU embedder for all-MiniLM-L6-v2 (384-d, 0 VRAM)."""

    def __init__(self, model_name: str = cfg.MINILM_MODEL_NAME):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.device = "cpu"
        logger.info(f"Loading {model_name} on CPU for OCR embedding...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(cfg.SENTENCE_TRANSFORMERS_CACHE_DIR),
            local_files_only=cfg.is_offline_mode(),
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            cache_dir=str(cfg.SENTENCE_TRANSFORMERS_CACHE_DIR),
            local_files_only=cfg.is_offline_mode(),
        )
        self.model.to("cpu")
        self.model.eval()

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode a list of text strings into normalized 384-d vectors on CPU."""
        if not texts:
            return np.empty((0, cfg.MINILM_DIM), dtype=np.float32)

        import torch
        import torch.nn.functional as F

        all_embeddings: List[np.ndarray] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {k: v.to("cpu") for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                token_embeddings = outputs[0]
                mask = (
                    inputs["attention_mask"]
                    .unsqueeze(-1)
                    .expand(token_embeddings.size())
                    .float()
                )
                pooled = torch.sum(token_embeddings * mask, dim=1) / torch.clamp(
                    mask.sum(dim=1), min=1e-9
                )
                normed = F.normalize(pooled, p=2, dim=1)
                all_embeddings.append(normed.cpu().numpy().astype(np.float32))

        return np.vstack(all_embeddings)

    def embed_query(self, query: str) -> List[float]:
        """Encode a single search query string into a 384-d float list."""
        emb = self.embed_texts([query], batch_size=1)
        return emb[0].tolist()


_GLOBAL_MINILM: Optional[MiniLMEmbedder] = None


def get_minilm_embedder() -> MiniLMEmbedder:
    """Retrieve or initialize global CPU MiniLMEmbedder singleton."""
    global _GLOBAL_MINILM
    if _GLOBAL_MINILM is None:
        _GLOBAL_MINILM = MiniLMEmbedder()
    return _GLOBAL_MINILM
