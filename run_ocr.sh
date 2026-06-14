#!/usr/bin/env bash
set -euo pipefail

# ── paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"

# ── defaults ───────────────────────────────────────────────────────────────
INPUT=""
OUTPUT="report.html"
JSON="results.json"
DPI=300
LANG="en"
PAGES=""
ENGINE="tesseract"
DEVICE="auto"
INSTALL=0
HELP=0

# ── helpers ────────────────────────────────────────────────────────────────
usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  -i, --input    <file>    PDF file to analyze (required)
  -o, --output   <file>    HTML report path           (default: report.html)
  -j, --json     <file>    JSON results path           (default: results.json)
  -d, --dpi      <int>     Rasterization DPI           (default: 300)
  -l, --lang     <code>    Language code               (default: en)
  -p, --pages    <range>   Pages to process, e.g. 1,2,5-8 (default: all)
  -e, --engine   <name>    OCR engine: tesseract|easyocr|trocr (default: tesseract)
      --device   <name>    Device: auto|cuda|cpu       (default: auto)
      --install            Install/sync Python deps then exit
  -h, --help               Show this help and exit

Examples:
  $(basename "$0") --input invoice.pdf
  $(basename "$0") -i scan.pdf -o out.html -j out.json --dpi 400 --pages 1-3
  $(basename "$0") -i doc.pdf --engine easyocr --device cuda
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }

info() { echo "[OCR] $*"; }

# ── arg parsing ────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--input)    INPUT="$2";   shift 2 ;;
    -o|--output)   OUTPUT="$2";  shift 2 ;;
    -j|--json)     JSON="$2";    shift 2 ;;
    -d|--dpi)      DPI="$2";     shift 2 ;;
    -l|--lang)     LANG="$2";    shift 2 ;;
    -p|--pages)    PAGES="$2";   shift 2 ;;
    -e|--engine)   ENGINE="$2";  shift 2 ;;
       --device)   DEVICE="$2";  shift 2 ;;
       --install)  INSTALL=1;    shift   ;;
    -h|--help)     HELP=1;       shift   ;;
    *) die "Unknown option: $1" ;;
  esac
done

[[ $HELP -eq 1 ]] && { usage; exit 0; }

# ── system deps check ──────────────────────────────────────────────────────
check_system_deps() {
  local missing=()
  command -v tesseract &>/dev/null || missing+=("tesseract-ocr")
  command -v pdftoppm  &>/dev/null || missing+=("poppler-utils")

  if [[ ${#missing[@]} -gt 0 ]]; then
    echo "Missing system packages: ${missing[*]}"
    echo "Install them with:"
    echo "  sudo apt install ${missing[*]}"
    exit 1
  fi
}

# ── venv / dep install ─────────────────────────────────────────────────────
setup_env() {
  cd "$SCRIPT_DIR"

  if command -v uv &>/dev/null; then
    info "Syncing dependencies with uv..."
    uv sync
  elif [[ -d "$VENV" ]]; then
    info "Using existing virtual environment."
  else
    info "Creating virtual environment and installing dependencies..."
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --quiet --upgrade pip
    "$VENV/bin/pip" install --quiet -r requirements.txt
  fi
}

# ── activate venv ──────────────────────────────────────────────────────────
activate_env() {
  if [[ -f "$VENV/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
  else
    die "Virtual environment not found at $VENV. Run with --install first."
  fi
}

# ── main ───────────────────────────────────────────────────────────────────
check_system_deps

if [[ $INSTALL -eq 1 ]]; then
  setup_env
  info "Dependencies installed. Re-run without --install to process a PDF."
  exit 0
fi

[[ -z "$INPUT" ]] && { usage; die "--input is required."; }
[[ -f "$INPUT" ]] || die "File not found: $INPUT"

if [[ ! -d "$VENV" ]]; then
  info "No virtual environment found — running --install first."
  setup_env
fi

activate_env

# ── build python command ───────────────────────────────────────────────────
CMD=(python "$SCRIPT_DIR/ocr_checker.py"
  --input  "$INPUT"
  --output "$OUTPUT"
  --json   "$JSON"
  --dpi    "$DPI"
  --lang   "$LANG"
  --engine "$ENGINE"
  --device "$DEVICE"
)
[[ -n "$PAGES" ]] && CMD+=(--pages "$PAGES")

info "Processing: $INPUT"
info "Engine: $ENGINE | DPI: $DPI | Lang: $LANG${PAGES:+ | Pages: $PAGES}"
info "Output: $OUTPUT | JSON: $JSON"
echo ""

"${CMD[@]}"

echo ""
info "Done."
info "  Report : $OUTPUT"
info "  Data   : $JSON"
