"""Bounding-box math and pixel → percentage conversion."""

from __future__ import annotations


def merge_bboxes(boxes: list[dict]) -> dict:
    """Merge a list of ``{x, y, w, h}`` word boxes into one enclosing box.

    Returns ``{x, y, width, height}`` in pixels.
    """
    if not boxes:
        return {"x": 0, "y": 0, "width": 0, "height": 0}

    xs = [b["x"] for b in boxes]
    ys = [b["y"] for b in boxes]
    x2s = [b["x"] + b["w"] for b in boxes]
    y2s = [b["y"] + b["h"] for b in boxes]

    x, y = min(xs), min(ys)
    return {
        "x": x,
        "y": y,
        "width": max(x2s) - x,
        "height": max(y2s) - y,
    }


def to_percentage(bbox: dict, page_w: int, page_h: int) -> dict:
    """Convert a pixel ``{x, y, width, height}`` box to page-relative percents."""
    page_w = max(page_w, 1)
    page_h = max(page_h, 1)
    return {
        "x_pct": round(bbox["x"] / page_w * 100, 2),
        "y_pct": round(bbox["y"] / page_h * 100, 2),
        "w_pct": round(bbox["width"] / page_w * 100, 2),
        "h_pct": round(bbox["height"] / page_h * 100, 2),
    }
