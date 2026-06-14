# OCR Error Detection Tool

Detects inconsistent sentences, typographical errors, and hard-to-read
characters in handwritten or printed PDFs. Each sentence is rendered with a
bounding box and a **percentage-based error score**.

## Install

**System dependencies** (required by `pytesseract` and `pdf2image`):

```bash
sudo apt install tesseract-ocr poppler-utils
```

`language_tool_python` additionally needs a Java runtime (`default-jre`). If
Java is missing, the grammar/consistency signal degrades to 0 and the rest of
the pipeline still runs.

**Python dependencies** — with [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv sync                 # runtime deps
uv sync --extra dev     # + pytest for the test suite
```

Or with pip:

```bash
pip install -r requirements.txt
```

## Usage

### Shell script (recommended)

`run_ocr.sh` wraps the Python CLI, handles venv activation, and checks system
dependencies automatically.

```bash
# First-time setup (installs Python deps into .venv)
./run_ocr.sh --install

# Basic run
./run_ocr.sh --input invoice.pdf

# Full options
./run_ocr.sh -i scan.pdf -o out.html -j out.json --dpi 400 --pages 1-3

# GPU engine
./run_ocr.sh -i doc.pdf --engine easyocr --device cuda
```

### Python directly

```bash
python ocr_checker.py \
  --input document.pdf \
  --output report.html \
  --json results.json \
  --dpi 300 \
  --lang en \
  --pages 1,2,5-8
```

| Flag | Default | Description |
|---|---|---|
| `--input` | required | Path to PDF file |
| `--output` | `report.html` | Output HTML report path |
| `--json` | `results.json` | Output JSON data path |
| `--dpi` | `300` | Rasterization DPI (higher = better OCR, slower) |
| `--lang` | `en` | Language code (`en`, `de`, `fr`, `es`) |
| `--pages` | all | Page range, e.g. `1,2,5-8` |
| `--engine` | `tesseract` | OCR engine: `tesseract` (CPU), `easyocr` or `trocr` (GPU) |
| `--device` | `auto` | `auto` / `cuda` / `cpu` for GPU engines |

## OCR engines

The OCR stage is pluggable — all engines emit the same word/box/confidence
structure, so scoring and reporting are identical regardless of engine.

| Engine | Device | Best for | Notes |
|---|---|---|---|
| `tesseract` | CPU | Clean printed text | Default, no extra deps |
| `easyocr` | GPU/CPU | Printed + light handwriting | CRAFT detector + CRNN, gives boxes + confidence |
| `trocr` | GPU | **Handwriting / cursive** | Transformer; detects lines (EasyOCR/Tesseract) then recognizes with Microsoft TrOCR |

```bash
# GPU transformer OCR (auto-detects CUDA):
uv run ocr-checker --input scan.pdf --engine trocr
```

### GPU setup (CUDA)

GPU engines need a CUDA build of PyTorch. On this machine (RTX 4050, CUDA 12.x):

```bash
uv sync --extra transformer        # torch + easyocr + transformers
# If the default CPU torch is pulled in, install a CUDA wheel explicitly:
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

`--device auto` uses the GPU when available and silently falls back to CPU
otherwise. TrOCR-base needs ~1.5 GB VRAM, well within 6 GB.

The HTML report shows the page image with colored bounding-box overlays
alongside a scrollable, clickable sentence list (hover/click links the two).

## Project layout

```
OCR/
├── run_ocr.sh                ← Linux shell script (setup + run)
├── ocr_checker.py            ← CLI entry point / orchestrator
├── modules/
│   ├── pdf_rasterizer.py     ← PDF → images
│   ├── ocr_extractor.py      ← Tesseract OCR wrapper
│   ├── sentence_assembler.py ← Word → sentence grouping
│   ├── error_analyzer.py     ← Typo, grammar, readability scoring
│   ├── bbox_utils.py         ← Bounding-box math + % conversion
│   └── report_generator.py   ← HTML + image overlay output
├── templates/
│   └── report.html.jinja     ← Jinja2 HTML report template
├── tests/
│   └── test_analyzer.py      ← Unit tests for scoring functions
├── requirements.txt
└── README.md
```

## Tests

The scoring and bbox tests run without Tesseract or Java:

```bash
uv run pytest          # or: pip install pytest pyspellchecker && pytest tests/
```
