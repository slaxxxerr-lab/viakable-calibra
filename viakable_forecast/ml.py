"""ml.py — stub (engine not available on this machine)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def exog_for(monthly, price_history) -> dict:
    """Return exogenous feature dict (empty — no ML on this machine)."""
    return {}


def train(*_a, **_kw):
    return None


def predict(*_a, **_kw) -> np.ndarray:
    return np.zeros(6)


def explain(*_a, **_kw) -> list:
    return []


def importance(*_a, **_kw) -> list:
    return []


def cv_run(*_a, **_kw):
    return None


def fit(monthly, windows, spec=None, exog=None) -> dict | None:
    """Train stub — returns None (no model on this machine)."""
    return None


def refit(monthly, spec_str, exog, paths, _ml=None) -> dict | None:
    return _ml


class _FinalModel:
    n_series  = 0
    n_samples = 0
    importance = pd.Series(
        [0.25, 0.20, 0.15, 0.15, 0.10, 0.08, 0.07],
        index=["lag_1", "lag_2", "lag_3", "mismo_mes_año", "promedio_3m", "calibre", "familia"],
        dtype=float,
    )


def fit_everything(monthly, windows, specs=None, exog=None) -> dict | None:
    """Train all global models (stub — returns minimal dict on this machine)."""
    tuning = pd.DataFrame([
        {"spec": ("gbr", (150, 3, 0.08)),  "Error validación %": float("nan")},
    ])
    return {
        "time_all":   0.0,
        "specs":      specs or {},
        "spec":       ("gbr", (150, 3, 0.08)),   # (kind, params) tuple expected by spec_label()
        "honest_err": float("nan"),
        "time":       0.0,
        "extra":      {},
        "models":     {},
        "windows":    windows,
        "final":      _FinalModel(),
        "tuning":     tuning,
    }


def refit_final(monthly, ml, exog, future) -> dict | None:
    """Refit with updated metal prices (stub — returns ml unchanged)."""
    return ml
