"""atypical.py — stub (engine not available on this machine)."""
from __future__ import annotations

import pandas as pd

_DET_COLS = ["nivel", "id", "ym", "actual", "expected", "z", "key"]


def detect_all(monthly: pd.DataFrame) -> pd.DataFrame:
    """Return empty atypical-months DataFrame (stub)."""
    return pd.DataFrame(columns=_DET_COLS)


def apply_corrections(monthly: pd.DataFrame, detections: pd.DataFrame,
                      decisions: dict) -> pd.DataFrame:
    """Return monthly data with atypical months corrected (stub — no-op)."""
    return monthly.copy()


def product_id(monthly: pd.DataFrame) -> pd.Series:
    """Return a product-id Series (stub — uses prod column or index)."""
    try:
        return monthly["prod"].astype(str)
    except Exception:
        return pd.Series("?", index=monthly.index)


def cv_test(monthly: pd.DataFrame, windows: list) -> pd.DataFrame:
    """Cross-validate atypical correction (stub — returns empty table)."""
    return pd.DataFrame(columns=["Método", "Error propio %", "Error corregido %", "Mejora (pts)"])
