"""Output generation: image overlays + self-contained HTML report."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from PIL import Image, ImageDraw, ImageFont

# RGBA fills keyed by severity.
SEVERITY_COLORS = {
    "clean": (34, 197, 94, 120),  # green
    "minor": (234, 179, 8, 140),  # yellow
    "moderate": (249, 115, 22, 150),  # orange
    "critical": (239, 68, 68, 170),  # red
}

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"


def draw_bounding_boxes(image: Image.Image, sentences: list[dict]) -> Image.Image:
    """Overlay colored, semi-transparent boxes + error% labels on a page image."""
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    font = _load_font()

    for s in sentences:
        color = SEVERITY_COLORS.get(s["severity"], SEVERITY_COLORS["moderate"])
        bb = s["bounding_box"]
        x1, y1 = bb["x"], bb["y"]
        x2, y2 = x1 + bb["width"], y1 + bb["height"]
        draw.rectangle([x1, y1, x2, y2], outline=color[:3] + (255,), fill=color, width=2)
        draw.text(
            (x1, max(y1 - 14, 0)),
            f"{s['scores']['composite_error_pct']:.0f}%",
            fill=color[:3] + (255,),
            font=font,
        )

    return Image.alpha_composite(base, overlay)


def render_html_report(
    pages: list[dict],
    output_path: str,
    summary: dict | None = None,
) -> str:
    """Render the self-contained HTML report and write it to ``output_path``.

    Args:
        pages: list of ``{"page": int, "image_b64": str, "sentences": [...]}``.
        output_path: where to write the .html file.
        summary: optional aggregate stats for the top bar.

    Returns the path written.
    """
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("report.html.jinja")
    html = template.render(pages=pages, summary=summary or {})
    Path(output_path).write_text(html, encoding="utf-8")
    return output_path


def image_to_base64(image: Image.Image, fmt: str = "JPEG") -> str:
    """Encode a PIL image as a base64 data-URI payload (no prefix)."""
    if fmt.upper() in ("JPEG", "JPG") and image.mode == "RGBA":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _load_font() -> ImageFont.ImageFont:
    """Best-effort TrueType font, falling back to PIL's bitmap default."""
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, 14)
        except OSError:
            continue
    return ImageFont.load_default()
