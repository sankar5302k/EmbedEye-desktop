"""Thumbnail & Document Preview Rendering Module for EmbedEye.

Provides rendering and scaling utilities for generating thumbnail previews
from raster images and multi-page PDF document pages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image


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


def load_doc_thumbnail(file_path: str, page: int = 1) -> Optional[Image.Image]:
    """Facade for ThumbnailRenderer.load_doc_thumbnail."""
    return ThumbnailRenderer.load_doc_thumbnail(file_path, page=page)
