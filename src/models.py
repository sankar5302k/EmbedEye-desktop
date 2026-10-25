"""Model loaders, inference helpers, and VRAM management for RTX 3050 4GB.

HARD REQUIREMENT: All models load strictly from ./models/ cache directories.
- OpenCLIP: fp16 on CUDA (unloaded during captioning).
- Florence-2: fp16 on CUDA (caption + OCR, strictly isolated on GPU).
- RapidOCR: CPU-only fallback for document/screenshot validation (0 VRAM).
- Cross-Encoder Reranker: fp16 on CUDA (coexists with OpenCLIP during search).
"""

from __future__ import annotations

import gc
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

# Ensure env vars are set before imports
import src.config as cfg

logger = logging.getLogger("photo_search.models")

# Optional HEIC support
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass


def get_vram_mb() -> float:
    """Return currently allocated CUDA VRAM in megabytes."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def clear_gpu_memory() -> None:
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def clean_ocr_text(raw_text: str, max_chars: int = cfg.MAX_OCR_CHARS) -> str:
    """Clean and validate raw OCR output.
    
    Rules:
    - Strip whitespace
    - Discard if fewer than 4 alphanumeric characters
    - Discard if mostly repeated or garbage characters
    - Cap stored ocr_text at max_chars (default: 1000)
    """
    if not raw_text:
        return ""
    text = raw_text.strip()
    if not text:
        return ""

    # Discard if fewer than 4 alphanumeric characters
    alnum_chars = [c for c in text if c.isalnum()]
    if len(alnum_chars) < 4:
        return ""

    # Discard if mostly repeated or garbage characters
    # e.g. "!!!!!!!", "------", "aaaaaa"
    clean_no_ws = re.sub(r"\s+", "", text)
    if clean_no_ws:
        unique_chars = set(clean_no_ws)
        # If very few unique characters in a medium/long string
        if len(clean_no_ws) >= 6 and len(unique_chars) <= 2:
            return ""
        if len(clean_no_ws) > 15 and (len(unique_chars) / len(clean_no_ws)) < 0.15:
            return ""

    # Check for excessive unprintable / garbage sequence
    if sum(1 for c in text if not c.isprintable()) > len(text) * 0.1:
        return ""

    # Cap at max_chars
    return text[:max_chars].strip()


def is_document_or_screenshot(img: Image.Image) -> bool:
    """Check if an image likely represents a screenshot, document, receipt, or graphic."""
    try:
        w, h = img.size
        # Wide aspect ratio (e.g. desktop screenshot, banner) or tall (receipt, mobile screenshot)
        aspect = max(w / max(1, h), h / max(1, w))
        if aspect >= 1.6:
            return True

        # Check grayscale luminance: mostly white page (> 215) or dark mode (< 40)
        gray = img.convert("L").resize((64, 64))
        arr = np.asarray(gray, dtype=np.float32)
        mean_lum = float(np.mean(arr))
        std_lum = float(np.std(arr))

        # Documents/receipts have high mean (white paper) or very low (dark mode screenshot)
        if mean_lum > 215 or (mean_lum < 40 and std_lum < 35):
            return True

        return False
    except Exception:
        return False


_RAPIDOCR_INSTANCE = None


def get_rapidocr():
    """Lazily load RapidOCR on CPU (ONNX Runtime, 0 VRAM)."""
    global _RAPIDOCR_INSTANCE
    if _RAPIDOCR_INSTANCE is None:
        try:
            from rapidocr_onnxruntime import RapidOCR

            _RAPIDOCR_INSTANCE = RapidOCR()
        except Exception as e:
            logger.warning(f"RapidOCR initialization failed: {e}")
            _RAPIDOCR_INSTANCE = False
    return _RAPIDOCR_INSTANCE if _RAPIDOCR_INSTANCE is not False else None


def run_rapidocr_on_image(img: Image.Image) -> Tuple[str, float]:
    """Run RapidOCR on CPU and return (extracted_text, average_confidence)."""
    ocr_engine = get_rapidocr()
    if ocr_engine is None:
        return "", 0.0

    try:
        img_np = np.asarray(img.convert("RGB"))
        result, _ = ocr_engine(img_np)
        if not result:
            return "", 0.0

        texts = []
        confidences = []
        for item in result:
            if len(item) >= 3:
                txt = str(item[1]).strip()
                try:
                    conf = float(item[2])
                except (ValueError, TypeError):
                    conf = 0.5
                if txt:
                    texts.append(txt)
                    confidences.append(conf)

        combined_text = "\n".join(texts)
        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        return clean_ocr_text(combined_text), avg_conf
    except Exception as e:
        logger.debug(f"RapidOCR execution failed on image: {e}")
        return "", 0.0


def safe_load_image(image_path: Path | str, max_dimension: int = 1024) -> Optional[Image.Image]:
    """Safely open, EXIF-orient, convert to RGB, and thumbnail an image.

    Returns None if the image is unreadable or corrupt. Never crashes.
    """
    try:
        path = Path(image_path)
        if not path.is_file():
            logger.warning(f"File not found: {path}")
            return None

        with Image.open(path) as img:
            # Handle EXIF orientation
            try:
                img = ImageOps.exif_transpose(img)
            except Exception:
                pass

            # Convert mode to RGB (handles RGBA, P, CMYK, etc.)
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Resize thumbnail if exceeds max_dimension to preserve VRAM
            if max(img.size) > max_dimension:
                img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

            # Copy image so file descriptor can be closed
            return img.copy()
    except Exception as e:
        logger.warning(f"Skipping corrupt or unreadable image {image_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# 1. OpenCLIP (Visual Embeddings)
# ---------------------------------------------------------------------------


def load_clip(
    model_name: str = cfg.CLIP_MODEL_NAME,
    pretrained: str = cfg.CLIP_PRETRAINED,
    device: Optional[str] = None,
) -> Tuple[Any, Any, Any]:
    """Load OpenCLIP model, preprocess transform, and tokenizer into local cache."""
    import open_clip

    target_device = device or cfg.get_device()
    logger.info(
        f"Loading OpenCLIP {model_name} ({pretrained}) on {target_device} "
        f"from cache: {cfg.OPENCLIP_CACHE_DIR}"
    )

    precision = "fp16" if target_device == "cuda" else "fp32"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        precision=precision,
        device=target_device,
        cache_dir=str(cfg.OPENCLIP_CACHE_DIR),
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    logger.info(f"OpenCLIP loaded. GPU VRAM allocated: {get_vram_mb():.1f} MB")
    return model, preprocess, tokenizer


def unload_clip(clip_components: Optional[Tuple[Any, Any, Any]]) -> None:
    """Unload OpenCLIP model and release CUDA memory."""
    if clip_components is not None:
        model, preprocess, tokenizer = clip_components
        if hasattr(model, "to"):
            try:
                model.to("cpu")
            except Exception:
                pass
        del model, preprocess, tokenizer
    clear_gpu_memory()
    logger.info(f"OpenCLIP unloaded. GPU VRAM allocated: {get_vram_mb():.1f} MB")


def embed_text_clip(
    text: str,
    clip_model: Any,
    tokenizer: Any,
    device: Optional[str] = None,
) -> np.ndarray:
    """Encode query text into normalized 512-d CLIP vector."""
    import torch

    target_device = device or cfg.get_device()
    tokens = tokenizer([text]).to(target_device)
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy().flatten().astype(np.float32)


def embed_images_clip(
    images: List[Image.Image],
    clip_model: Any,
    preprocess: Any,
    device: Optional[str] = None,
    batch_size: int = cfg.CLIP_BATCH_SIZE,
) -> List[np.ndarray]:
    """Embed list of PIL images into normalized 512-d vectors with automatic OOM recovery."""
    import torch

    target_device = device or cfg.get_device()
    results: List[np.ndarray] = []
    current_bs = batch_size

    i = 0
    while i < len(images):
        chunk = images[i : i + current_bs]
        try:
            tensor_list = [preprocess(img) for img in chunk]
            dtype = torch.float16 if target_device == "cuda" else torch.float32
            batch_tensor = torch.stack(tensor_list).to(device=target_device, dtype=dtype)
            with torch.no_grad():
                feats = clip_model.encode_image(batch_tensor)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            batch_vectors = feats.cpu().numpy().astype(np.float32)
            for vec in batch_vectors:
                results.append(vec)
            i += len(chunk)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() and target_device == "cuda":
                logger.warning(
                    f"CUDA OOM during CLIP embedding with batch size {current_bs}. "
                    "Clearing cache and adjusting..."
                )
                clear_gpu_memory()
                if current_bs > 1:
                    current_bs = max(1, current_bs // 2)
                    continue
                else:
                    # Batch size is already 1; fall back to CPU for this item
                    logger.warning(f"CUDA OOM with batch size 1. Falling back to CPU for image {i}...")
                    try:
                        tensor = preprocess(chunk[0]).unsqueeze(0).to(device="cpu", dtype=torch.float32)
                        cpu_model = clip_model.cpu()
                        with torch.no_grad():
                            feats = cpu_model.encode_image(tensor)
                            feats = feats / feats.norm(dim=-1, keepdim=True)
                        results.append(feats.cpu().numpy().astype(np.float32)[0])
                        # Put model back to cuda if possible
                        clip_model.to(target_device)
                        i += 1
                        continue
                    except Exception as cpu_e:
                        logger.error(f"CPU fallback for image {i} failed: {cpu_e}")
                        raise
            else:
                logger.error(f"Error during CLIP embedding: {e}")
                raise

    return results


# ---------------------------------------------------------------------------
# 2. Florence-2 (Captioning + OCR in One Model)
# ---------------------------------------------------------------------------


def load_florence(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Load Microsoft Florence-2 model and processor with pinned commit revision."""
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    name = model_name or cfg.FLORENCE_MODEL_NAME
    rev = cfg.FLORENCE_REVISIONS.get(name, cfg.FLORENCE_REVISION)
    target_device = device or cfg.get_device()
    is_cuda = target_device == "cuda"
    dtype = torch.float16 if is_cuda else torch.float32

    logger.info(
        f"Loading Florence-2 ({name}, rev={rev[:8]}...) on {target_device} ({dtype}) "
        f"from cache: {cfg.TRANSFORMERS_CACHE_DIR}"
    )

    processor = AutoProcessor.from_pretrained(
        name,
        revision=rev,
        trust_remote_code=True,
        cache_dir=str(cfg.TRANSFORMERS_CACHE_DIR),
    )

    model = AutoModelForCausalLM.from_pretrained(
        name,
        revision=rev,
        trust_remote_code=True,
        torch_dtype=dtype,
        cache_dir=str(cfg.TRANSFORMERS_CACHE_DIR),
    )
    model.to(target_device)
    model.eval()

    logger.info(f"Florence-2 loaded. GPU VRAM allocated: {get_vram_mb():.1f} MB")
    return processor, model


def unload_florence(florence_components: Optional[Tuple[Any, Any]]) -> None:
    """Unload Florence-2 model and processor from GPU memory."""
    if florence_components is not None:
        processor, model = florence_components
        if hasattr(model, "to"):
            try:
                model.to("cpu")
            except Exception:
                pass
        del processor, model
    clear_gpu_memory()
    logger.info(f"Florence-2 unloaded. GPU VRAM allocated: {get_vram_mb():.1f} MB")


def _run_florence_task_batch(
    images: List[Image.Image],
    task: str,
    processor: Any,
    model: Any,
    target_device: str,
    dtype: Any,
    max_new_tokens: int,
    num_beams: int,
) -> List[str]:
    """Execute a single Florence-2 task on a batch of images and return raw generation strings."""
    import torch

    prompts = [task] * len(images)
    inputs = processor(text=prompts, images=images, return_tensors="pt")
    if target_device == "cuda":
        inputs = {
            k: v.to(device=target_device, dtype=dtype if v.dtype == torch.float32 else v.dtype)
            for k, v in inputs.items()
        }

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )

    raw_outputs = processor.batch_decode(generated_ids, skip_special_tokens=False)
    return raw_outputs


def caption_and_ocr_florence(
    images: List[Image.Image],
    processor: Any,
    model: Any,
    device: Optional[str] = None,
    batch_size: int = cfg.FLORENCE_BATCH_SIZE,
    max_new_tokens: int = cfg.FLORENCE_MAX_NEW_TOKENS,
    num_beams: int = cfg.FLORENCE_NUM_BEAMS,
) -> List[Dict[str, Any]]:
    """Generate detailed captions and extracted OCR text using Florence-2.
    
    Includes automatic OOM halving and single-item CPU fallback.
    Returns: List of {"caption": str, "ocr_text": str, "has_text": bool}
    """
    import torch

    target_device = device or cfg.get_device()
    dtype = torch.float16 if target_device == "cuda" else torch.float32
    results: List[Dict[str, Any]] = []

    current_bs = batch_size
    i = 0

    while i < len(images):
        chunk = images[i : i + current_bs]
        try:
            # Task A: More Detailed Caption
            raw_caps = _run_florence_task_batch(
                chunk,
                "<MORE_DETAILED_CAPTION>",
                processor,
                model,
                target_device,
                dtype,
                max_new_tokens,
                num_beams,
            )

            # Task B: OCR Text Extraction
            raw_ocrs = _run_florence_task_batch(
                chunk,
                "<OCR>",
                processor,
                model,
                target_device,
                dtype,
                max_new_tokens,
                num_beams,
            )

            for img, raw_c, raw_o in zip(chunk, raw_caps, raw_ocrs):
                # Post-process generation with image size
                parsed_c = processor.post_process_generation(
                    raw_c, task="<MORE_DETAILED_CAPTION>", image_size=img.size
                )
                parsed_o = processor.post_process_generation(
                    raw_o, task="<OCR>", image_size=img.size
                )

                cap_str = ""
                if isinstance(parsed_c, dict):
                    cap_str = str(parsed_c.get("<MORE_DETAILED_CAPTION>", "")).strip()
                elif isinstance(parsed_c, str):
                    cap_str = parsed_c.strip()

                # Clean special tokens like <pad>, </s>, <s> from captions
                cap_str = re.sub(r"(?:<pad>|</s>|<s>|<unk>)", "", cap_str).strip()

                ocr_str = ""
                if isinstance(parsed_o, dict):
                    ocr_str = str(parsed_o.get("<OCR>", "")).strip()
                elif isinstance(parsed_o, str):
                    ocr_str = parsed_o.strip()

                ocr_str = re.sub(r"(?:<pad>|</s>|<s>|<unk>)", "", ocr_str).strip()


                cleaned_ocr = clean_ocr_text(ocr_str)
                results.append(
                    {
                        "caption": cap_str,
                        "ocr_text": cleaned_ocr,
                        "has_text": bool(cleaned_ocr),
                    }
                )

            i += len(chunk)

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() and target_device == "cuda":
                logger.warning(
                    f"CUDA OOM during Florence-2 batch (size={current_bs}). "
                    "Clearing cache and retrying with reduced batch size..."
                )
                clear_gpu_memory()
                if current_bs > 1:
                    current_bs = max(1, current_bs // 2)
                    continue
                else:
                    # Batch size is 1 and still failed: fall back to CPU for this image
                    logger.warning(
                        f"Florence-2 failed on GPU for image {i}. Falling back to CPU for this item..."
                    )
                    try:
                        model.to("cpu")
                        single_img = [chunk[0]]
                        raw_c = _run_florence_task_batch(
                            single_img,
                            "<MORE_DETAILED_CAPTION>",
                            processor,
                            model,
                            "cpu",
                            torch.float32,
                            max_new_tokens,
                            num_beams,
                        )
                        raw_o = _run_florence_task_batch(
                            single_img,
                            "<OCR>",
                            processor,
                            model,
                            "cpu",
                            torch.float32,
                            max_new_tokens,
                            num_beams,
                        )
                        parsed_c = processor.post_process_generation(

                            raw_c[0], task="<MORE_DETAILED_CAPTION>", image_size=chunk[0].size
                        )
                        parsed_o = processor.post_process_generation(
                            raw_o[0], task="<OCR>", image_size=chunk[0].size
                        )


                        cap_str = str(parsed_c.get("<MORE_DETAILED_CAPTION>", "")).strip() if isinstance(parsed_c, dict) else str(parsed_c).strip()
                        cap_str = re.sub(r"(?:<pad>|</s>|<s>|<unk>)", "", cap_str).strip()
                        ocr_str = str(parsed_o.get("<OCR>", "")).strip() if isinstance(parsed_o, dict) else str(parsed_o).strip()
                        ocr_str = re.sub(r"(?:<pad>|</s>|<s>|<unk>)", "", ocr_str).strip()
                        cleaned_ocr = clean_ocr_text(ocr_str)

                        results.append(
                            {
                                "caption": cap_str,
                                "ocr_text": cleaned_ocr,
                                "has_text": bool(cleaned_ocr),
                            }
                        )
                        # Move model back to cuda
                        model.to(target_device)
                        i += 1
                        continue
                    except Exception as cpu_e:
                        logger.error(f"CPU fallback for Florence-2 item {i} failed: {cpu_e}")
                        results.append(
                            {
                                "caption": "Uncaptioned image (inference failed)",
                                "ocr_text": "",
                                "has_text": False,
                            }
                        )
                        i += 1
                        continue
            else:
                logger.error(f"Error during Florence-2 execution: {e}")
                raise

    return results


# ---------------------------------------------------------------------------
# 3. Neural Cross-Encoder Reranker (mixedbread-ai/mxbai-rerank-large-v1)
# ---------------------------------------------------------------------------

_reranker_tokenizer = None
_reranker_model = None


def load_reranker(
    model_name: str = cfg.RERANKER_MODEL_NAME,
    device: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Load neural cross-encoder tokenizer and model."""
    global _reranker_tokenizer, _reranker_model
    if _reranker_tokenizer is None or _reranker_model is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        target_device = device or cfg.get_device()
        is_cuda = target_device == "cuda"
        dtype = torch.float16 if is_cuda else torch.float32
        logger.info(
            f"Loading neural reranker ({model_name}) on {target_device.upper()} ({dtype}) "
            f"from cache: {cfg.RERANKER_CACHE_DIR}"
        )
        _reranker_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(cfg.RERANKER_CACHE_DIR),
        )
        _reranker_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=str(cfg.RERANKER_CACHE_DIR),
            torch_dtype=dtype,
        )
        _reranker_model.to(target_device)
        _reranker_model.eval()

    return _reranker_tokenizer, _reranker_model


def unload_reranker(bundle: Optional[Tuple[Any, Any]] = None) -> None:
    """Unload neural reranker and clear GPU memory."""
    global _reranker_tokenizer, _reranker_model
    if _reranker_model is not None:
        if hasattr(_reranker_model, "to"):
            try:
                _reranker_model.to("cpu")
            except Exception:
                pass
        del _reranker_model
        _reranker_model = None
    if _reranker_tokenizer is not None:
        del _reranker_tokenizer
        _reranker_tokenizer = None
    clear_gpu_memory()
    logger.info(f"Reranker unloaded. GPU VRAM allocated: {get_vram_mb():.1f} MB")


def rerank_pairs(
    query: str,
    documents: List[str],
    batch_size: int = cfg.RERANKER_BATCH_SIZE,
    reranker_bundle: Optional[Tuple[Any, Any]] = None,
) -> List[float]:
    """Score (query, doc) candidate pairs using local neural cross-encoder."""
    if not documents:
        return []

    import torch

    if reranker_bundle is not None:
        tokenizer, model = reranker_bundle
    else:
        tokenizer, model = load_reranker()

    device = cfg.get_device()
    pairs = [[query, doc] for doc in documents]
    all_scores: List[float] = []

    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i : i + batch_size]
        inputs = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        if cfg.is_cuda_available():
            inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.squeeze(-1)
            if logits.ndim == 0:
                chunk_scores = [float(logits.item())]
            else:
                chunk_scores = [float(s) for s in logits.cpu().tolist()]
            all_scores.extend(chunk_scores)

    return all_scores


# ---------------------------------------------------------------------------
# 4. Legacy BLIP (Optional / Backward Compatibility)
# ---------------------------------------------------------------------------


def load_blip(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Load BLIP Captioner model and processor into local cache."""
    import torch
    from transformers import BlipForConditionalGeneration, BlipProcessor

    name = model_name or cfg.BLIP_MODEL_NAME
    target_device = device or cfg.get_device()
    logger.info(
        f"Loading BLIP Captioner ({name}) on {target_device} "
        f"from cache: {cfg.TRANSFORMERS_CACHE_DIR}"
    )

    processor = BlipProcessor.from_pretrained(
        name,
        cache_dir=str(cfg.TRANSFORMERS_CACHE_DIR),
    )

    dtype = torch.float16 if target_device == "cuda" else torch.float32
    model = BlipForConditionalGeneration.from_pretrained(
        name,
        torch_dtype=dtype,
        cache_dir=str(cfg.TRANSFORMERS_CACHE_DIR),
    )
    model.to(target_device)
    model.eval()
    logger.info(f"BLIP Captioner loaded. GPU VRAM allocated: {get_vram_mb():.1f} MB")
    return processor, model


def unload_blip(blip_components: Optional[Tuple[Any, Any]]) -> None:
    """Unload BLIP model and release CUDA memory."""
    if blip_components is not None:
        processor, model = blip_components
        if hasattr(model, "to"):
            try:
                model.to("cpu")
            except Exception:
                pass
        del processor, model
    clear_gpu_memory()
    logger.info(f"BLIP unloaded. GPU VRAM allocated: {get_vram_mb():.1f} MB")
