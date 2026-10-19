"""engine.py — stub with working module-level constants and dummy return objects.

Module-level constants are real and importable. Computation functions return
dummy zero-filled objects so the UI renders without crashing on Windows.
Set _FULL_ENGINE = True only in the real Mac production environment.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pandas as pd

# ── Flag ─────────────────────────────────────────────────────────────────────
_FULL_ENGINE = False   # True only in the real production environment (Mac)

# ── Measure identifiers ───────────────────────────────────────────────────────
TON    = "ton"
MXN    = "mxn"

# ── Method identifiers ────────────────────────────────────────────────────────
AUTO   = "auto"
SIGNAL = "signal"
ML     = "ml"

ML_FAMILY: set[str] = {"ml", "xgboost", "rf", "mlp", "stacking"}

TYPE_OF: dict[str, str] = {
    "mismo_mes_año":  "Estadístico",
    "promedio_3m":    "Estadístico",
    "suavización":    "Estadístico",
    "holt":           "Estadístico",
    "holt_winters":   "Estadístico",
    "regresión":      "Estadístico",
    "croston":        "Estadístico",
    "tsb":            "Estadístico",
    "ets":            "Estadístico",
    "sarima":         "Estadístico",
    "ensemble":       "Ensamble",
    "ml":             "Machine learning",
    "xgboost":        "Machine learning",
    "rf":             "Machine learning",
    "ridge":          "Machine learning",
    "mlp":            "Red neuronal",
    "stacking":       "Ensamble",
    "similar":        "Estadístico",
    "signal":         "Señales",
    AUTO:             "Automático",
}

AUTO_EXCLUDE: set[str] = {"ml", "xgboost", "rf", "mlp", "stacking", "similar"}

INFLATE_BY_WINDOWS: dict[int, float] = {1: 3.0, 2: 1.5}


# ── Dummy result objects ──────────────────────────────────────────────────────
@dataclasses.dataclass
class Projection:
    fc:      np.ndarray
    hist:    pd.Series
    lo:      np.ndarray
    hi:      np.ndarray
    method:  str
    detail:  list
    table:   pd.DataFrame
    index:   list
    error:        float = float("nan")
    window_errors: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class PlanResult:
    capacity: pd.DataFrame
    detail:   list


@dataclasses.dataclass
class CVResult:
    table:  pd.DataFrame
    fcs:    dict
    actual: list


# ── Real helper functions ─────────────────────────────────────────────────────
def tariff_multiplier(scenario: float, current: float, elasticity: float) -> float:
    return ((1 + scenario) / (1 + current)) ** (-elasticity)


def windows_for(monthly: pd.DataFrame) -> list[tuple[int, int]]:
    """Return list of (n_training_months, forecast_horizon) for valid CV windows."""
    try:
        n = monthly["ym"].nunique()
    except Exception:
        n = 0
    # Windows cut at 6, 9, 12, 15 months before end; need ≥12 months training; forecast 6 months
    return [(n - cut, 6) for cut in (6, 9, 12, 15) if n - cut >= 12]


def best_method(backtest_or_table) -> str:
    try:
        if isinstance(backtest_or_table, pd.DataFrame):
            return backtest_or_table["Método"].iloc[0]
        return backtest_or_table.method
    except Exception:
        return AUTO


def consensus(fc: np.ndarray, adj: dict, why: dict) -> tuple:
    """Apply monthly % adjustments to forecast. Returns (adjusted_fc, bad_list)."""
    try:
        out = fc.copy().astype(float)
        bad: list[str] = []
        for i, v in adj.items():
            idx = int(i)
            if 0 <= idx < len(out):
                reason = (why or {}).get(i, "")
                if not reason:
                    bad.append(str(i))
                else:
                    out[idx] = out[idx] * (1 + float(v) / 100)
        return out, bad
    except Exception:
        return fc, []


# ── Stub functions (return zeroed objects) ────────────────────────────────────
def _dummy_hist(monthly: pd.DataFrame) -> pd.Series:
    """Build a minimal history Series from monthly data."""
    try:
        agg = monthly.groupby("ym")["qty"].sum().sort_index()
        return agg
    except Exception:
        return pd.Series(dtype=float)


def project(monthly, units, market, horizon, method, measure, mode,
            mult, sig, w, prices, crm=None, ml=None, apply_signals=False) -> Projection:
    """Return a dummy zero forecast (engine unavailable on this machine)."""
    h = int(horizon) if horizon else 6
    fc   = np.zeros(h)
    hist = _dummy_hist(monthly)
    lo   = fc * 0.85
    hi   = fc * 1.15

    # Build dummy forecast index (months after last data point)
    try:
        last = monthly["ym"].max()
        idx  = [last + pd.DateOffset(months=i + 1) for i in range(h)]
    except Exception:
        idx = list(range(h))

    detail: list[dict] = []
    try:
        for g in sorted(monthly["cal"].unique()):
            sub = monthly[monthly["cal"] == g]["qty"].values
            detail.append({"label": str(g), "y": sub, "fc": np.zeros(h),
                           "error": float("nan"), "method": AUTO, "demand": "Estable"})
    except Exception:
        pass

    try:
        gauges = sorted(monthly["cal"].unique())
    except Exception:
        gauges = []
    table_rows = []
    for g in gauges:
        table_rows.append({
            "Calibre": str(g),
            "Método": AUTO,
            "Error %": float("nan"),
            "Precisión %": float("nan"),
            "Evaluación": "—",
            "Demanda": "Estable",
            "Pronóstico": 0.0,
            "Mismos meses año anterior": 0.0,
            "Cambio %": float("nan"),
        })
    if not table_rows:
        table_rows.append({
            "Calibre": "—", "Método": AUTO, "Error %": float("nan"),
            "Precisión %": float("nan"), "Evaluación": "—", "Demanda": "Estable",
            "Pronóstico": 0.0, "Mismos meses año anterior": 0.0, "Cambio %": float("nan"),
        })
    table = pd.DataFrame(table_rows)
    return Projection(fc=fc, hist=hist, lo=lo, hi=hi,
                      method=AUTO, detail=detail, table=table, index=idx)


def plan(monthly, gauges, horizon, params, prices, mult, sig, w,
         overrides=None, ml=None, crm=None) -> PlanResult:
    """Return a dummy plan result (engine unavailable on this machine)."""
    cap_rows = [{"cal": str(g), "over": 0.0, "util": 0.0} for g in (gauges or [])]
    cap = pd.DataFrame(cap_rows) if cap_rows else pd.DataFrame(columns=["cal", "over", "util"])
    return PlanResult(capacity=cap, detail=[])


def cv_run(y: np.ndarray) -> CVResult:
    table = pd.DataFrame({"Método": [AUTO], "Error %": [float("nan")]})
    return CVResult(table=table, fcs={AUTO: [None]}, actual=[])


def forecast(y: np.ndarray, h: int, method: str, table) -> np.ndarray:
    mean = float(np.mean(y[-3:])) if len(y) >= 3 else 0.0
    return np.full(h, mean)


def interval(actual, fcs, fc, inflate=1.0):
    return fc * 0.85 * inflate, fc * 1.15 * inflate


def series(monthly, units=None, market=None, measure=TON, prices=None, **kwargs) -> np.ndarray:
    try:
        m = monthly.copy()
        for k, v in kwargs.items():
            if k in m.columns and v:
                m = m[m[k].astype(str).isin([str(x) for x in v])]
        return m.groupby("ym")["qty"].sum().values
    except Exception:
        return np.zeros(0)


def accuracy_matrix(monthly=None, units=None, *_a, **_kw) -> pd.DataFrame:
    try:
        gauges = [str(u[0]) if u else "?" for u in (units or [])]
    except Exception:
        gauges = []
    cols = gauges + ["Total"]
    # Non-NaN placeholder so idxmin() and loc[] work in the lab section
    return pd.DataFrame({c: [0.0] for c in cols}, index=[AUTO])


def selection_backtest(*_a, **_kw) -> pd.DataFrame:
    return pd.DataFrame(columns=["Método", "Error %", "Precisión %", "Sesgo %", "Elegible", "Tiempo s"])


def method_totals(*_a, **_kw) -> dict:
    return {}


def signal_row(hist_values, w) -> dict:
    return {}


def ml_vs_stat(mx) -> pd.DataFrame:
    """Return ML vs statistical comparison DataFrame (stub)."""
    cols = ["Calibre", "Error ML %", "Mejor estadístico", "Error estadístico %", "Diferencia (pts)", "Gana"]
    try:
        gauges = [c for c in mx.columns if c != "Total"]
    except Exception:
        gauges = []
    rows = [{"Calibre": str(g), "Error ML %": float("nan"), "Mejor estadístico": AUTO,
             "Error estadístico %": float("nan"), "Diferencia (pts)": 0.0, "Gana": AUTO}
            for g in gauges]
    rows.append({"Calibre": "Total", "Error ML %": float("nan"), "Mejor estadístico": AUTO,
                 "Error estadístico %": float("nan"), "Diferencia (pts)": 0.0, "Gana": AUTO})
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def ml_models(ml) -> dict:
    return {}


def similar_parts(*_a, **_kw):
    return None, None
