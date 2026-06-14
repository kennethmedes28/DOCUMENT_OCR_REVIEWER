"""Tesseract OCR wrapper.

Extracts per-word text, bounding boxes, and confidence scores from a page image.
Requires the `tesseract-ocr` system package (Tesseract 5.x recommended).
"""

from __future__ import annotations

import pytesseract
from pytesseract import Output
from PIL import Image

# PSM 6 = "assume a single uniform block of text", which is a good default for
# scanned documents. OEM 1 = LSTM-only engine (better on handwriting/cursive).
DEFAULT_CONFIG = "--oem 1 --psm 6"


def extract_words(
    image: Image.Image,
    lang: str = "eng",
    config: str = DEFAULT_CONFIG,
) -> dict:
    """Run Tesseract and return its TSV data as a dict of parallel lists.

    The returned dict contains (among others) the keys:
        text, left, top, width, height, conf,
        block_num, par_num, line_num, word_num

    ``conf`` is a per-word confidence in ``0–100`` (``-1`` for non-text rows).
    """
    return pytesseract.image_to_data(
        image, lang=lang, config=config, output_type=Output.DICT
    )


def extract_hocr(
    image: Image.Image,
    lang: str = "eng",
    config: str = DEFAULT_CONFIG,
) -> bytes:
    """Return hOCR (HTML) output with full layout information."""
    return pytesseract.image_to_pdf_or_hocr(
        image, lang=lang, config=config, extension="hocr"
    )
