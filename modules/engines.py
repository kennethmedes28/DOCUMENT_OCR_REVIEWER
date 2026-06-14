"""Pluggable OCR engines.

Every engine exposes ``extract(image) -> tsv_dict`` returning the same
Tesseract-style TSV structure (parallel lists keyed by ``text, conf, left, top,
width, height, block_num, par_num, line_num, word_num``). That keeps
``sentence_assembler`` and the rest of the pipeline engine-agnostic.

Engines:
    tesseract  CPU, zero extra deps (default).
    easyocr    GPU/CPU, CRAFT detector + CRNN recognizer — gives boxes + conf.
    trocr      GPU transformer: EasyOCR/Tesseract detects lines, Microsoft
               TrOCR recognizes each crop (strong on handwriting).

Heavy deps (torch, easyocr, transformers) are imported lazily so the base
install stays light.
"""

from __future__ import annotations

from statistics import median
from typing import Protocol

from PIL import Image

from .device import device_info, resolve_device

# Keys every engine must populate (subset of Tesseract's TSV columns).
_TSV_KEYS = (
    "text", "conf", "left", "top", "width", "height",
    "block_num", "par_num", "line_num", "word_num",
)


class OCREngine(Protocol):
    """Structural type for an OCR engine."""

    def extract(self, image: Image.Image) -> dict: ...


# ----------------------------------------------------------------------------
# Geometry → TSV synthesis (pure; used by the box-based engines)
# ----------------------------------------------------------------------------
def detections_to_tsv(
    items: list[dict],
    line_gap_ratio: float = 0.6,
    para_gap_ratio: float = 1.6,
) -> dict:
    """Turn a flat list of axis-aligned detections into a Tesseract-style TSV.

    Each item is ``{"text", "conf"(0-100), "x", "y", "w", "h"}``. Lines and
    paragraphs are inferred geometrically (vertical clustering by height), so
    the result feeds straight into ``group_words_into_sentences``.
    """
    tsv = {k: [] for k in _TSV_KEYS}
    items = [it for it in items if (it.get("text") or "").strip()]
    if not items:
        return tsv

    med_h = median([it["h"] for it in items]) or 1.0

    # Cluster into visual lines by vertical center proximity.
    def center(it: dict) -> float:
        return it["y"] + it["h"] / 2.0

    lines: list[list[dict]] = []
    for it in sorted(items, key=center):
        if lines:
            cur = lines[-1]
            cur_center = sum(center(w) for w in cur) / len(cur)
            if abs(center(it) - cur_center) <= line_gap_ratio * med_h:
                cur.append(it)
                continue
        lines.append([it])

    lines.sort(key=lambda ln: min(w["y"] for w in ln))

    # Assign paragraph + line numbers using vertical gaps between lines.
    par, line_in_par, prev_bottom = 1, 1, None
    for ln in lines:
        top = min(w["y"] for w in ln)
        bottom = max(w["y"] + w["h"] for w in ln)
        if prev_bottom is not None and (top - prev_bottom) > para_gap_ratio * med_h:
            par += 1
            line_in_par = 1
        for word_num, w in enumerate(sorted(ln, key=lambda d: d["x"]), start=1):
            tsv["text"].append(w["text"])
            tsv["conf"].append(float(w["conf"]))
            tsv["left"].append(int(w["x"]))
            tsv["top"].append(int(w["y"]))
            tsv["width"].append(int(w["w"]))
            tsv["height"].append(int(w["h"]))
            tsv["block_num"].append(1)
            tsv["par_num"].append(par)
            tsv["line_num"].append(line_in_par)
            tsv["word_num"].append(word_num)
        line_in_par += 1
        prev_bottom = bottom

    return tsv


def _poly_to_box(poly) -> dict:
    """Convert a 4-point polygon ``[[x,y], ...]`` to axis-aligned x,y,w,h."""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x, y = min(xs), min(ys)
    return {"x": int(x), "y": int(y), "w": int(max(xs) - x), "h": int(max(ys) - y)}


# ----------------------------------------------------------------------------
# Tesseract (CPU, default)
# ----------------------------------------------------------------------------
class TesseractEngine:
    def __init__(self, lang: str = "eng", config: str | None = None):
        from .ocr_extractor import DEFAULT_CONFIG

        self.lang = lang
        self.config = config or DEFAULT_CONFIG

    def extract(self, image: Image.Image) -> dict:
        from .ocr_extractor import extract_words

        return extract_words(image, lang=self.lang, config=self.config)


# ----------------------------------------------------------------------------
# EasyOCR (GPU, CRAFT + CRNN)
# ----------------------------------------------------------------------------
class EasyOCREngine:
    def __init__(self, lang: str = "en", device: str = "auto"):
        import numpy  # noqa: F401  (ensures the dep is present early)
        import easyocr

        self.device = resolve_device(device)
        self._reader = easyocr.Reader([lang], gpu=(self.device == "cuda"))

    def extract(self, image: Image.Image) -> dict:
        import numpy as np

        results = self._reader.readtext(np.array(image.convert("RGB")))
        items = []
        for poly, text, conf in results:
            box = _poly_to_box(poly)
            box.update(text=text, conf=float(conf) * 100.0)
            items.append(box)
        return detections_to_tsv(items)


# ----------------------------------------------------------------------------
# TrOCR transformer (GPU) — detect lines, recognize each crop
# ----------------------------------------------------------------------------
class TrOCREngine:
    """Microsoft TrOCR recognition over detected line/word crops.

    TrOCR has no detector of its own, so an existing engine supplies the boxes
    (EasyOCR by default, Tesseract as fallback) and TrOCR re-recognizes each
    crop for higher accuracy on handwriting.
    """

    def __init__(
        self,
        model_name: str = "microsoft/trocr-base-handwritten",
        lang: str = "en",
        device: str = "auto",
        detector: OCREngine | None = None,
    ):
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self.device = resolve_device(device)
        self._torch = torch
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.detector = detector or _default_detector(lang, self.device)

    def extract(self, image: Image.Image) -> dict:
        torch = self._torch
        boxes_tsv = self.detector.extract(image)
        items: list[dict] = []
        rgb = image.convert("RGB")

        for i in range(len(boxes_tsv["text"])):
            if not (boxes_tsv["text"][i] or "").strip():
                continue
            x, y = boxes_tsv["left"][i], boxes_tsv["top"][i]
            w, h = boxes_tsv["width"][i], boxes_tsv["height"][i]
            crop = rgb.crop((x, y, x + w, y + h))
            text, conf = self._recognize(crop)
            if not text:
                continue
            items.append({"text": text, "conf": conf, "x": x, "y": y, "w": w, "h": h})

        return detections_to_tsv(items)

    def _recognize(self, crop: Image.Image) -> tuple[str, float]:
        torch = self._torch
        pixel_values = self.processor(crop, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                pixel_values, output_scores=True, return_dict_in_generate=True
            )
        text = self.processor.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()

        # Confidence = mean per-token probability of the chosen sequence × 100.
        try:
            scores = self.model.compute_transition_scores(
                out.sequences, out.scores, normalize_logits=True
            )
            probs = scores[0].exp()
            conf = float(probs.mean().item()) * 100.0
        except Exception:
            conf = 90.0  # neutral default if scores unavailable
        return text, conf


def _default_detector(lang: str, device: str) -> OCREngine:
    """Pick a line/word detector for TrOCR: EasyOCR if available, else Tesseract."""
    try:
        return EasyOCREngine(lang=lang, device=device)
    except Exception:
        tess_lang = {"en": "eng", "de": "deu", "fr": "fra", "es": "spa"}.get(lang, lang)
        return TesseractEngine(lang=tess_lang)


# ----------------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------------
def get_engine(name: str, lang: str = "en", device: str = "auto") -> OCREngine:
    """Build an engine by name. ``name`` ∈ {tesseract, easyocr, trocr}."""
    name = (name or "tesseract").lower()
    if name == "tesseract":
        tess_lang = {"en": "eng", "de": "deu", "fr": "fra", "es": "spa"}.get(lang, lang)
        return TesseractEngine(lang=tess_lang)
    if name == "easyocr":
        return EasyOCREngine(lang=lang, device=device)
    if name == "trocr":
        return TrOCREngine(lang=lang, device=device)
    raise ValueError(f"Unknown engine '{name}'. Choose tesseract, easyocr, or trocr.")


def describe_device(device: str = "auto") -> str:
    """Resolve and describe a device preference (for CLI logging)."""
    return device_info(resolve_device(device))
