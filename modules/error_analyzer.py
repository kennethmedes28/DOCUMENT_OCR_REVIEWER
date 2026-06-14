"""Per-sentence error scoring.

Three independent signals, combined into a weighted composite:

    typo_score          (40%)  misspellings via pyspellchecker
    inconsistency_score (35%)  grammar/consistency via LanguageTool
    readability_score   (25%)  Tesseract per-word confidence (inverted)

LanguageTool requires Java; if it is unavailable the grammar signal degrades
gracefully to 0 rather than crashing the whole pipeline.
"""

from __future__ import annotations

from spellchecker import SpellChecker

# Composite weights — must sum to 1.0.
WEIGHT_TYPO = 0.40
WEIGHT_INCONSISTENCY = 0.35
WEIGHT_READABILITY = 0.25

# Score → severity bands.
SEVERITY_BANDS = (
    (15, "clean"),
    (40, "minor"),
    (70, "moderate"),
    (100, "critical"),
)


class ErrorAnalyzer:
    """Holds the (expensive) spellchecker + grammar tool instances.

    Instantiate once and reuse across pages/sentences.
    """

    def __init__(self, lang: str = "en", spell_lang: str = "en"):
        self.spell = SpellChecker(language=spell_lang)
        self._tool = None
        try:  # LanguageTool is optional (needs a JVM).
            import language_tool_python

            self._tool = language_tool_python.LanguageTool(_lt_lang(lang))
        except Exception:  # pragma: no cover - environment dependent
            self._tool = None

    # -- typo signal -----------------------------------------------------
    def analyze_typos(self, text: str) -> tuple[float, list[dict]]:
        """Return ``(score_0_100, issues)`` from spellcheck."""
        words = [w for w in _tokens(text) if w]
        if not words:
            return 0.0, []
        misspelled = self.spell.unknown(words)
        score = len(misspelled) / len(words) * 100
        issues = [
            {"type": "typo", "word": w, "suggestion": self.spell.correction(w)}
            for w in misspelled
        ]
        return round(score, 2), issues

    # -- grammar / consistency signal ------------------------------------
    def analyze_inconsistencies(self, text: str) -> tuple[float, list[dict]]:
        """Return ``(score_0_100, issues)`` from LanguageTool grammar checks."""
        if self._tool is None or not text.strip():
            return 0.0, []
        matches = self._tool.check(text)
        n_words = max(len(text.split()), 1)
        # Density of issues, amplified, then capped at 100.
        score = min(len(matches) / n_words * 100 * 5, 100)
        issues = [
            {"type": "grammar", "message": m.message, "context": m.context}
            for m in matches
        ]
        return round(score, 2), issues

    # -- readability signal ----------------------------------------------
    @staticmethod
    def analyze_readability(word_confidences: list[float]) -> float:
        """Invert mean Tesseract confidence → hard-to-read score (0–100)."""
        if not word_confidences:
            return 0.0
        avg_conf = sum(word_confidences) / len(word_confidences)
        return round(max(0.0, 100.0 - avg_conf), 2)

    # -- composite -------------------------------------------------------
    def analyze_sentence(self, text: str, confidences: list[float]) -> dict:
        """Run all three signals for one sentence and return scores + issues."""
        typo, typo_issues = self.analyze_typos(text)
        incon, incon_issues = self.analyze_inconsistencies(text)
        read = self.analyze_readability(confidences)
        composite = compute_composite(typo, incon, read)
        return {
            "scores": {
                "typo_score": typo,
                "inconsistency_score": incon,
                "readability_score": read,
                "composite_error_pct": composite,
            },
            "issues": typo_issues + incon_issues,
            "severity": severity_for(composite),
        }


def compute_composite(typo: float, inconsistency: float, readability: float) -> float:
    """Weighted blend of the three signals."""
    return round(
        typo * WEIGHT_TYPO
        + inconsistency * WEIGHT_INCONSISTENCY
        + readability * WEIGHT_READABILITY,
        2,
    )


def severity_for(score: float) -> str:
    """Map a composite score to a severity label."""
    for threshold, label in SEVERITY_BANDS:
        if score <= threshold:
            return label
    return "critical"


def _tokens(text: str) -> list[str]:
    """Lowercase word tokens, stripping surrounding punctuation."""
    return [t.strip(".,;:!?\"'()[]{}").lower() for t in text.split()]


def _lt_lang(lang: str) -> str:
    """Map a short language code to a LanguageTool locale."""
    mapping = {"en": "en-US", "de": "de-DE", "fr": "fr", "es": "es"}
    return mapping.get(lang, lang)
