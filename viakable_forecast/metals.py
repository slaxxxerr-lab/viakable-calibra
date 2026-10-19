"""metals.py — working stub for Viakable Calibra (no SQLite / full engine).

make_path() and path_change() are real so the price-path machinery works.
Heavy computation functions raise NotImplementedError.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from viakable_forecast.config import PLAN_DEFAULTS


def make_path(
    kind: str = "flat",
    cu: float | None = None,
    al: float | None = None,
    fx: float | None = None,
    cu_chg: float = 0.0,
    al_chg: float = 0.0,
    fx_chg: float = 0.0,
    shape: str = "lineal",
    manual=None,
    hist=None,
    months: int = 12,
) -> dict:
    """Return 12-month price paths for copper, aluminum and FX.

    kind: "flat" | "pct" | "manual" | "tendencia"
    cu/al: LME price USD/t; fx: MXN/USD rate.
    cu_chg/al_chg/fx_chg: total % change over the horizon.
    """
    cu  = float(cu  or PLAN_DEFAULTS["cu_lme"])
    al  = float(al  or PLAN_DEFAULTS["al_lme"])
    fx  = float(fx  or PLAN_DEFAULTS["fx"])

    def _ramp(start: float, total_chg_pct: float) -> list[float]:
        if kind == "flat" or total_chg_pct == 0:
            return [start] * months
        end = start * (1 + total_chg_pct / 100)
        if shape == "inmediata":
            return [end] * months
        # lineal ramp
        return [round(start + (end - start) * i / (months - 1), 6)
                for i in range(months)]

    # Manual monthly table from uploaded history
    if kind == "manual" and isinstance(manual, dict):
        cu_path = [float(manual.get(f"cu_{i}", cu)) for i in range(months)]
        al_path = [float(manual.get(f"al_{i}", al)) for i in range(months)]
        fx_path = [float(manual.get(f"fx_{i}", fx)) for i in range(months)]
    else:
        cu_path = _ramp(cu, cu_chg)
        al_path = _ramp(al, al_chg)
        fx_path = _ramp(fx, fx_chg)

    return {"cu": cu_path, "al": al_path, "fx": fx_path}


def path_change(path: list[float], base: float | None = None) -> float:
    """Return % change: average of first 6 months vs base (or first element)."""
    if not path:
        return 0.0
    ref = float(base) if base is not None else path[0]
    if ref == 0:
        return 0.0
    avg6 = float(np.mean(path[:6]))
    return (avg6 / ref - 1) * 100


def next_months(last_ym, n: int = 12) -> list:
    """Return list of n timestamps after last_ym."""
    try:
        start = pd.Timestamp(last_ym) + pd.DateOffset(months=1)
        return [start + pd.DateOffset(months=i) for i in range(n)]
    except Exception:
        return []


def shock(paths: dict, metal: str, pct: float) -> dict:
    """Apply an immediate % shock to one metal's path."""
    out = {k: list(v) for k, v in paths.items()}
    if metal in out:
        factor = 1 + pct / 100
        out[metal] = [round(x * factor, 6) for x in out[metal]]
    return out


def parse_history(df: pd.DataFrame, generic: bool = False) -> dict:
    """Parse an uploaded price history table (stub: return empty dict)."""
    return {}


def purchase_plan(req, now, path, fx0, fx_path, hold_pct, ahead_months) -> list:
    """Optimal purchase timing (stub: recommend buying today)."""
    return []
