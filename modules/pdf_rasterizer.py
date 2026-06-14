"""PDF → image rasterization.

Converts PDF pages to high-resolution PIL images so Tesseract can OCR them.
Requires the `poppler-utils` system package (provides `pdftoppm`).
"""

from __future__ import annotations

from pdf2image import convert_from_path
from PIL import Image


def rasterize_pdf(
    pdf_path: str,
    dpi: int = 300,
    pages: list[int] | None = None,
) -> list[Image.Image]:
    """Rasterize a PDF into a list of PIL Image objects (one per page).

    Args:
        pdf_path: Path to the input PDF.
        dpi: Render resolution. Higher = better OCR accuracy but slower.
        pages: Optional 1-based page numbers to render. ``None`` renders all.

    Returns:
        List of RGB PIL Images in page order.
    """
    if pages:
        images: list[Image.Image] = []
        for page_num in pages:
            images.extend(
                convert_from_path(
                    pdf_path, dpi=dpi, first_page=page_num, last_page=page_num
                )
            )
        return images

    return convert_from_path(pdf_path, dpi=dpi)


def parse_page_range(spec: str | None) -> list[int] | None:
    """Parse a CLI page spec like ``"1,2,5-8"`` into ``[1, 2, 5, 6, 7, 8]``.

    Returns ``None`` for an empty/falsy spec, meaning "all pages".
    """
    if not spec:
        return None

    pages: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            pages.extend(range(int(start), int(end) + 1))
        else:
            pages.append(int(chunk))
    return sorted(set(pages))
