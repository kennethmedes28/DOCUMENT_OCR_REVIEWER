"""Unit tests for scoring + bbox math (no Tesseract/Java required)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.bbox_utils import merge_bboxes, to_percentage  # noqa: E402
from modules.error_analyzer import (  # noqa: E402
    ErrorAnalyzer,
    compute_composite,
    severity_for,
)


# -- composite + severity ------------------------------------------------
def test_compute_composite_weights():
    # 100/0/0 with 0.40 weight → 40
    assert compute_composite(100, 0, 0) == 40.0
    assert compute_composite(0, 100, 0) == 35.0
    assert compute_composite(0, 0, 100) == 25.0
    assert compute_composite(100, 100, 100) == 100.0


@pytest.mark.parametrize(
    "score,label",
    [(0, "clean"), (15, "clean"), (16, "minor"), (40, "minor"),
     (41, "moderate"), (70, "moderate"), (71, "critical"), (100, "critical")],
)
def test_severity_bands(score, label):
    assert severity_for(score) == label


# -- readability ---------------------------------------------------------
def test_readability_inverts_confidence():
    assert ErrorAnalyzer.analyze_readability([90, 90, 90]) == 10.0
    assert ErrorAnalyzer.analyze_readability([]) == 0.0
    assert ErrorAnalyzer.analyze_readability([100]) == 0.0


# -- typo detection ------------------------------------------------------
def test_typo_detection_flags_misspellings():
    analyzer = ErrorAnalyzer()
    score, issues = analyzer.analyze_typos("Ths is an exmaple sentance")
    assert score > 0
    flagged = {i["word"] for i in issues}
    assert "ths" in flagged or "exmaple" in flagged or "sentance" in flagged


def test_clean_text_scores_low():
    analyzer = ErrorAnalyzer()
    score, issues = analyzer.analyze_typos("this is a clean sentence")
    assert score == 0.0
    assert issues == []


# -- bbox ----------------------------------------------------------------
def test_merge_bboxes():
    boxes = [{"x": 10, "y": 20, "w": 30, "h": 10}, {"x": 50, "y": 18, "w": 20, "h": 12}]
    merged = merge_bboxes(boxes)
    assert merged == {"x": 10, "y": 18, "width": 60, "height": 12}


def test_merge_bboxes_empty():
    assert merge_bboxes([]) == {"x": 0, "y": 0, "width": 0, "height": 0}


def test_to_percentage():
    bbox = {"x": 50, "y": 100, "width": 200, "height": 50}
    pct = to_percentage(bbox, 1000, 2000)
    assert pct == {"x_pct": 5.0, "y_pct": 5.0, "w_pct": 20.0, "h_pct": 2.5}


# -- engine: geometry → TSV synthesis (no torch/Tesseract needed) --------
from modules.engines import detections_to_tsv, get_engine  # noqa: E402
from modules.device import resolve_device  # noqa: E402


def test_detections_to_tsv_groups_into_lines_and_paragraphs():
    # Two words on line 1, two on line 2 (small gap = same paragraph),
    # one far below (big gap = new paragraph).
    items = [
        {"text": "Hello", "conf": 95, "x": 10, "y": 10, "w": 40, "h": 20},
        {"text": "world", "conf": 90, "x": 60, "y": 12, "w": 40, "h": 20},
        {"text": "second", "conf": 88, "x": 10, "y": 38, "w": 50, "h": 20},
        {"text": "line", "conf": 80, "x": 70, "y": 38, "w": 30, "h": 20},
        {"text": "FARAWAY", "conf": 70, "x": 10, "y": 400, "w": 80, "h": 20},
    ]
    tsv = detections_to_tsv(items)
    assert len(tsv["text"]) == 5
    # Reading order preserved.
    assert tsv["text"][:2] == ["Hello", "world"]
    # First two lines share a paragraph; the far word starts a new one.
    assert tsv["par_num"][0] == tsv["par_num"][2] == 1
    assert tsv["par_num"][4] == 2
    # Line numbers increment within a paragraph.
    assert tsv["line_num"][0] == 1 and tsv["line_num"][2] == 2


def test_detections_to_tsv_empty():
    tsv = detections_to_tsv([])
    assert tsv["text"] == [] and tsv["conf"] == []


def test_resolve_device_cpu():
    assert resolve_device("cpu") == "cpu"


def test_get_engine_unknown_raises():
    with pytest.raises(ValueError):
        get_engine("bogus")
