"""OCR Error Detection Tool — CLI entry point / orchestrator.

Pipeline (see CLAUDE.md §3):
    PDF → rasterize → OCR → assemble sentences → score errors → report

Usage:
    python ocr_checker.py --input document.pdf --output report.html \
        --json results.json --dpi 300 --lang en --pages 1,2,5-8
"""

from __future__ import annotations

import argparse
import json
import sys

from modules.bbox_utils import to_percentage
from modules.engines import get_engine
from modules.error_analyzer import ErrorAnalyzer
from modules.pdf_rasterizer import parse_page_range, rasterize_pdf
from modules.report_generator import (
    draw_bounding_boxes,
    image_to_base64,
    render_html_report,
)
from modules.sentence_assembler import group_words_into_sentences


def process_pdf(
    pdf_path: str,
    dpi: int = 300,
    lang: str = "en",
    pages: list[int] | None = None,
    engine: str = "tesseract",
    device: str = "auto",
) -> dict:
    """Run the full pipeline and return structured results for all pages."""
    images = rasterize_pdf(pdf_path, dpi=dpi, pages=pages)
    analyzer = ErrorAnalyzer(lang=lang, spell_lang=lang)
    ocr = get_engine(engine, lang=lang, device=device)

    page_results: list[dict] = []
    page_numbers = pages if pages else list(range(1, len(images) + 1))

    for page_no, image in zip(page_numbers, images):
        page_w, page_h = image.size
        tsv = ocr.extract(image)
        raw_sentences = group_words_into_sentences(tsv)

        sentences: list[dict] = []
        for idx, sent in enumerate(raw_sentences, start=1):
            analysis = analyzer.analyze_sentence(sent["text"], sent["confidences"])
            bbox = dict(sent["bbox"])
            bbox.update(to_percentage(bbox, page_w, page_h))
            sentences.append(
                {
                    "page": page_no,
                    "sentence_id": f"p{page_no}_s{idx}",
                    "text": sent["text"],
                    "bounding_box": bbox,
                    **analysis,
                }
            )

        annotated = draw_bounding_boxes(image, sentences)
        page_results.append(
            {
                "page": page_no,
                "width": page_w,
                "height": page_h,
                "image_b64": image_to_base64(annotated, fmt="JPEG"),
                "sentences": sentences,
            }
        )

    return {"source": pdf_path, "pages": page_results, "summary": _summarize(page_results)}


def _summarize(page_results: list[dict]) -> dict:
    """Aggregate counts and average error across all sentences."""
    all_sents = [s for p in page_results for s in p["sentences"]]
    total = len(all_sents)
    dist = {"clean": 0, "minor": 0, "moderate": 0, "critical": 0}
    for s in all_sents:
        dist[s["severity"]] = dist.get(s["severity"], 0) + 1
    avg = (
        sum(s["scores"]["composite_error_pct"] for s in all_sents) / total
        if total
        else 0.0
    )
    return {
        "total_sentences": total,
        "distribution": dist,
        "avg_error_pct": round(avg, 2),
        "pages": len(page_results),
    }


def _write_json(results: dict, path: str) -> None:
    """Write JSON results without the heavy base64 image payloads."""
    slim = {
        "source": results["source"],
        "summary": results["summary"],
        "pages": [
            {
                "page": p["page"],
                "width": p["width"],
                "height": p["height"],
                "sentences": p["sentences"],
            }
            for p in results["pages"]
        ],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(slim, fh, indent=2, ensure_ascii=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detect typos, inconsistencies, and hard-to-read text in PDFs."
    )
    parser.add_argument("--input", required=True, help="Path to PDF file")
    parser.add_argument("--output", default="report.html", help="Output HTML report path")
    parser.add_argument("--json", default="results.json", help="Output JSON data path")
    parser.add_argument("--dpi", type=int, default=300, help="Rasterization DPI")
    parser.add_argument("--lang", default="en", help="Language code (en, de, fr, es)")
    parser.add_argument("--pages", default=None, help="Page range, e.g. 1,2,5-8")
    parser.add_argument(
        "--engine",
        default="tesseract",
        choices=["tesseract", "easyocr", "trocr"],
        help="OCR engine (trocr/easyocr are GPU-accelerated)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device for GPU engines",
    )
    args = parser.parse_args(argv)

    if args.engine != "tesseract":
        from modules.engines import describe_device

        print(f"Engine: {args.engine} · device: {describe_device(args.device)}")

    pages = parse_page_range(args.pages)
    results = process_pdf(
        args.input,
        dpi=args.dpi,
        lang=args.lang,
        pages=pages,
        engine=args.engine,
        device=args.device,
    )

    _write_json(results, args.json)
    render_html_report(results["pages"], args.output, summary=results["summary"])

    s = results["summary"]
    print(
        f"Processed {s['pages']} page(s), {s['total_sentences']} sentence(s). "
        f"Avg error {s['avg_error_pct']}%. "
        f"Wrote {args.output} and {args.json}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
