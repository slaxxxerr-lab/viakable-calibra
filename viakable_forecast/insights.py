"""insights.py — minimal working implementation for Viakable Calibra.

Functions that require the full engine return empty/neutral DataFrames or dicts
so the app renders without crashing. All signatures match what app.py expects.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


# ── entity_dim ────────────────────────────────────────────────────────────────
def entity_dim(monthly: pd.DataFrame) -> str | None:
    """Return the best dimension column to use as 'entity' (cli > ch > seg)."""
    for col in ("cli", "ch", "seg"):
        if col in monthly.columns:
            vals = monthly[col].astype(str).str.strip()
            if (vals != "").any():
                return col
    return None


# ── gauge_trends ──────────────────────────────────────────────────────────────
def gauge_trends(monthly: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with columns [calibre, anterior, cambio] per gauge."""
    if monthly.empty or "ym" not in monthly.columns:
        return pd.DataFrame(columns=["calibre", "anterior", "cambio"])
    cutoff = monthly["ym"].max()
    rows = []
    for cal, g in monthly.groupby("cal"):
        t3  = g[g["ym"] >  cutoff - pd.DateOffset(months=3)]["qty"].sum()
        t3b = g[(g["ym"] <= cutoff - pd.DateOffset(months=3)) &
                (g["ym"] >  cutoff - pd.DateOffset(months=6))]["qty"].sum()
        cambio = (t3 / t3b - 1) * 100 if t3b > 0 else 0.0
        rows.append({"calibre": str(cal), "anterior": float(t3b), "cambio": float(cambio)})
    return pd.DataFrame(rows)


# ── export_trend ──────────────────────────────────────────────────────────────
def export_trend(monthly: pd.DataFrame) -> dict | None:
    """Return {cambio: float|None} for the US export share trend."""
    try:
        from viakable_forecast.config import MKT_USA
        us = monthly[monthly["mkt"] == MKT_USA] if "mkt" in monthly.columns else pd.DataFrame()
        if us.empty or "ym" not in us.columns:
            return None
        cutoff = monthly["ym"].max()
        t3  = us[us["ym"] >  cutoff - pd.DateOffset(months=3)]["qty"].sum()
        t3b = us[(us["ym"] <= cutoff - pd.DateOffset(months=3)) &
                 (us["ym"] >  cutoff - pd.DateOffset(months=6))]["qty"].sum()
        if t3b == 0:
            return None
        return {"cambio": float((t3 / t3b - 1) * 100)}
    except Exception:
        return None


# ── entity_trends ─────────────────────────────────────────────────────────────
def entity_trends(monthly: pd.DataFrame, dim: str | None) -> pd.DataFrame:
    """Return a DataFrame with [entidad, cambio, pct12, baja] per entity."""
    empty = pd.DataFrame(columns=["entidad", "cambio", "pct12", "baja"])
    if not dim or monthly.empty or dim not in monthly.columns:
        return empty
    cutoff = monthly["ym"].max()
    rows = []
    for val, g in monthly.groupby(dim):
        val = str(val).strip()
        if not val:
            continue
        t6  = g[g["ym"] >  cutoff - pd.DateOffset(months=6)]["qty"].sum()
        t6b = g[(g["ym"] <= cutoff - pd.DateOffset(months=6)) &
                (g["ym"] >  cutoff - pd.DateOffset(months=12))]["qty"].sum()
        t12 = g[g["ym"] >  cutoff - pd.DateOffset(months=12)]["qty"].sum()
        cambio = (t6 / t6b - 1) * 100 if t6b > 0 else 0.0
        pct12  = float(t12)
        rows.append({"entidad": val, "cambio": float(cambio), "pct12": pct12,
                     "baja": cambio < -20})
    return pd.DataFrame(rows) if rows else empty


# ── rhythm_anomalies ──────────────────────────────────────────────────────────
def rhythm_anomalies(monthly: pd.DataFrame, dim: str | None,
                     tolerance: dict, gauges_filter) -> pd.DataFrame:
    """Return anomaly DataFrame (empty when engine not available)."""
    return pd.DataFrame(columns=["key", "level", "entity", "cal", "score"])


# ── learned_tolerance ─────────────────────────────────────────────────────────
def learned_tolerance(decisions: dict) -> tuple[dict, dict]:
    """Return (tolerance_dict, gauge_filter_dict) from stored decisions."""
    return {}, {}


# ── frozen_compare ────────────────────────────────────────────────────────────
def frozen_compare(payload: dict, monthly: pd.DataFrame) -> dict:
    """Compare a frozen forecast payload against actual data."""
    return {"rows": [], "fva": None}


# ── low_volume ────────────────────────────────────────────────────────────────
def low_volume(monthly: pd.DataFrame, gauges: list) -> pd.DataFrame:
    """Return a DataFrame of low-volume products (empty stub)."""
    return pd.DataFrame(columns=["prod", "cal", "qty_mean", "demand_class", "forecast"])


# ── tracking_test ─────────────────────────────────────────────────────────────
def tracking_test(monthly: pd.DataFrame, gauges: list, horizon: int) -> pd.DataFrame:
    """Simulate a frozen forecast made `horizon` months ago vs actuals."""
    return pd.DataFrame(columns=["cal", "forecast", "actual", "error"])


# ── abc_xyz ───────────────────────────────────────────────────────────────────
def abc_xyz(monthly: pd.DataFrame, margin_t: dict) -> pd.DataFrame:
    """Return ABC-XYZ classification DataFrame."""
    return pd.DataFrame(columns=["prod", "cal", "abc", "xyz", "sugerencia"])


# ── price_sensitivity ─────────────────────────────────────────────────────────
def price_sensitivity(monthly: pd.DataFrame, hist: dict) -> pd.DataFrame:
    """Return price-elasticity DataFrame."""
    return pd.DataFrame(columns=["cal", "elasticity", "confidence"])
