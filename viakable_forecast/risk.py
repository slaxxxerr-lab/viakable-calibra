"""risk.py — stub (engine not available on this machine)."""
from __future__ import annotations

import numpy as np
import pandas as pd


class Setup:
    """Minimal risk setup container (stub)."""
    def __init__(self, rows, params, mult, tariff_current, elasticity, h):
        self.rows           = rows
        self.params         = params
        self.mult           = mult
        self.tariff_current = tariff_current
        self.elasticity     = elasticity
        self.h              = h


def price_vol(hist, fx_daily) -> dict:
    """Return price volatility parameters (stub)."""
    return {
        "cu_vol": 0.0, "al_vol": 0.0, "fx_vol": 0.0,
        "corr": np.zeros((3, 3)), "src": "sin datos", "inflate": 1.0,
    }


def calibrate(rows) -> dict:
    """Calibrate error distribution from plan rows (stub)."""
    return {"inflate": 1.0, "coverage": float("nan"), "n": 0}


def run(setup, n: int = 5000, tariff=None, vol=None, hedge=None, inflate: float = 1.0) -> dict:
    """Monte Carlo simulation (stub — returns zero arrays)."""
    h = getattr(setup, "h", 6)
    margin = np.zeros(max(n, 1))
    return {
        "margin":     margin,
        "metal_need": {"cu": np.zeros((max(n, 1), h)), "al": np.zeros((max(n, 1), h))},
        "rev":        np.zeros(max(n, 1)),
        "vol":        np.zeros(max(n, 1)),
    }


def summarize(margin: np.ndarray, threshold: float = 0.0) -> dict:
    """Summarize a margin distribution (stub)."""
    mean = float(np.mean(margin)) if len(margin) else 0.0
    p5   = float(np.percentile(margin, 5)) if len(margin) else 0.0
    prob = float(np.mean(margin < threshold)) if len(margin) else 0.0
    worst = margin[margin <= p5] if len(margin) else np.array([0.0])
    cvar = float(np.mean(worst)) if len(worst) else p5
    at_risk = mean - p5
    return {
        "mean": mean, "p5": p5, "p95": float(np.percentile(margin, 95)) if len(margin) else 0.0,
        "prob_below": prob, "mar": at_risk, "at_risk": at_risk, "cvar": cvar,
    }


def deterministic(setup) -> dict:
    """Compute deterministic plan margin (stub)."""
    return {"margin": 0.0, "rev": 0.0, "metal_cost": 0.0}


def newsvendor(need, price_fx: float, rush_pct: float, hold_pct: float, k: int) -> dict:
    """Newsvendor optimal purchase quantity (stub)."""
    median = float(np.median(need)) if hasattr(need, "__len__") and len(need) else 0.0
    return {
        "q_star": median, "p10": 0.0, "median": median, "p90": median * 1.2,
        "critical_ratio": 0.5, "under_cost": 0.0, "over_cost": 0.0,
    }


def coverage_sweep(setup, vol, tariff, inflate, fracs, months, fee=0.0, n=2000) -> pd.DataFrame:
    """Sweep hedge fractions (stub)."""
    return pd.DataFrame(columns=["frac", "mean", "p5", "mar"])
