"""Device selection + GPU detection for the transformer/EasyOCR engines.

Kept dependency-light: torch is imported lazily so the Tesseract-only install
never needs it.
"""

from __future__ import annotations


def cuda_available() -> bool:
    """True if torch is installed and a CUDA device is visible."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_device(pref: str = "auto") -> str:
    """Resolve a device preference to a concrete ``"cuda"`` or ``"cpu"``.

    Args:
        pref: ``"auto"`` (use CUDA if available), ``"cuda"``, or ``"cpu"``.

    Raises:
        RuntimeError: if ``"cuda"`` is requested but unavailable.
    """
    pref = (pref or "auto").lower()
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        if not cuda_available():
            raise RuntimeError(
                "CUDA requested but not available. Install a CUDA build of torch "
                "or use --device cpu."
            )
        return "cuda"
    # auto
    return "cuda" if cuda_available() else "cpu"


def device_info(device: str) -> str:
    """Human-readable one-liner describing the resolved device."""
    if device == "cuda":
        try:
            import torch

            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return f"CUDA · {name} ({total:.1f} GB)"
        except Exception:
            return "CUDA"
    return "CPU"
