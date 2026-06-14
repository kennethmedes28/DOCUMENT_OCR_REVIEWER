"""Group Tesseract words into sentences with merged bounding boxes.

Words are grouped by their layout identity (block → paragraph → line). Adjacent
lines within ``line_tolerance`` pixels are merged so a sentence that wraps
across a line break stays as a single unit.
"""

from __future__ import annotations

from .bbox_utils import merge_bboxes


def _word_rows(tsv: dict) -> list[dict]:
    """Yield clean per-word rows from Tesseract TSV, dropping empty/low rows."""
    rows: list[dict] = []
    n = len(tsv["text"])
    for i in range(n):
        text = (tsv["text"][i] or "").strip()
        conf = float(tsv["conf"][i])
        if not text or conf < 0:
            continue
        rows.append(
            {
                "text": text,
                "conf": conf,
                "x": int(tsv["left"][i]),
                "y": int(tsv["top"][i]),
                "w": int(tsv["width"][i]),
                "h": int(tsv["height"][i]),
                "block": int(tsv["block_num"][i]),
                "par": int(tsv["par_num"][i]),
                "line": int(tsv["line_num"][i]),
            }
        )
    return rows


def group_words_into_sentences(
    tsv_data: dict,
    line_tolerance: int = 10,
) -> list[dict]:
    """Group words into sentence dicts.

    Returns a list of dicts, each with:
        text         joined words for the sentence
        bbox         merged {x, y, w, h} pixel box
        confidences  list of per-word Tesseract confidences
    """
    rows = _word_rows(tsv_data)

    # First group by (block, par, line) — a single visual line of text.
    lines: dict[tuple[int, int, int], list[dict]] = {}
    for row in rows:
        key = (row["block"], row["par"], row["line"])
        lines.setdefault(key, []).append(row)

    # Order lines top-to-bottom, then merge vertically-adjacent lines in the
    # same paragraph into one sentence block.
    ordered = sorted(lines.values(), key=lambda ws: min(w["y"] for w in ws))

    sentences: list[dict] = []
    current: list[dict] | None = None
    current_par: tuple[int, int] | None = None
    last_bottom: int | None = None

    for words in ordered:
        par_key = (words[0]["block"], words[0]["par"])
        top = min(w["y"] for w in words)

        same_par = current_par == par_key
        close = last_bottom is not None and (top - last_bottom) <= line_tolerance

        if current is not None and same_par and close:
            current.extend(words)
        else:
            if current is not None:
                sentences.append(_finalize(current))
            current = list(words)
            current_par = par_key

        last_bottom = max(w["y"] + w["h"] for w in words)

    if current is not None:
        sentences.append(_finalize(current))

    return sentences


def _finalize(words: list[dict]) -> dict:
    """Build a sentence dict from its constituent word rows."""
    words_sorted = sorted(words, key=lambda w: (w["y"], w["x"]))
    boxes = [{"x": w["x"], "y": w["y"], "w": w["w"], "h": w["h"]} for w in words_sorted]
    return {
        "text": " ".join(w["text"] for w in words_sorted),
        "bbox": merge_bboxes(boxes),
        "confidences": [w["conf"] for w in words_sorted],
    }
