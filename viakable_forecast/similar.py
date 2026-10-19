"""similar.py — stub (engine not available on this machine)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _unavailable(*_a, **_kw):
    raise NotImplementedError("Motor no disponible en este equipo.")


group    = _unavailable
blend    = _unavailable
forecast = _unavailable


def product_table(monthly: pd.DataFrame):
    """Return (Y, meta, months) — empty stub."""
    months = sorted(monthly["ym"].unique()) if "ym" in monthly.columns else []
    meta = pd.DataFrame(columns=["prod", "cal", "g1", "metal", "fam"])
    Y = pd.DataFrame()
    return Y, meta, months


def span_info(row) -> dict:
    """Return span/zero_share info for a product row — stub."""
    return {"span": 0, "zero_share": 1.0}


def is_target(info: dict) -> bool:
    """Return True if this product should use the group pattern — stub."""
    return False


def evaluate_products(monthly: pd.DataFrame, windows: list) -> dict:
    """Evaluate similar-series blending — stub returns n=0 so the block shows the hint."""
    cal_df = pd.DataFrame(columns=["cal", "_1", "_2", "_3", "_4",
                                   "Error propio %", "Error mezcla %",
                                   "Mejora de la mezcla (pts)"])
    res_df = pd.DataFrame(columns=["Series y ventanas", "Error propio %", "Error mezcla %"])
    return {"n": 0, "calibre": cal_df, "resumen": res_df}
