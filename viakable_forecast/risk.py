"""risk.py — stub (full engine not available on this machine).

All functions return the correct keys/shapes so section_riesgo renders
without KeyError. Values are zeros / placeholders.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42


class Setup:
    """Risk setup container."""
    def __init__(self, rows, params, mult, tariff_current, elasticity, h):
        self.rows           = rows
        self.params         = params
        self.mult           = mult
        self.tariff_current = tariff_current
        self.elasticity     = elasticity
        self.h              = h          # lowercase internal
        # uppercase aliases used throughout app.py
        self.H    = h
        self.mult0 = mult


def price_vol(hist, fx_daily) -> dict:
    """Return price volatility parameters (stub)."""
    return {
        "sigma":  {"cu": 0.0, "al": 0.0, "fx": 0.0},
        "source": {"cu": "sin datos", "al": "sin datos", "fx": "sin datos"},
        "corr":   np.zeros((3, 3)),
        "inflate": 1.0,
        # legacy flat keys kept for safety
        "cu_vol": 0.0, "al_vol": 0.0, "fx_vol": 0.0, "src": "sin datos",
    }


def calibrate(rows) -> dict:
    """Calibrate error distribution from plan rows (stub)."""
    return {"inflate": 1.0, "coverage": 80.0, "coverage_raw": 50.0, "n": 0}


def pooled_residuals(rows) -> np.ndarray:
    """Return pooled cross-validation residuals (stub)."""
    return np.zeros(50)


def simulate_demand(fc: np.ndarray, pool: np.ndarray, n: int,
                    rng: np.random.Generator, inflate: float = 1.0) -> np.ndarray:
    """Simulate n demand paths around fc (stub — returns fc repeated)."""
    return np.tile(fc, (n, 1)).astype(float)


def bands(draws: np.ndarray) -> dict:
    """Return P10/P50/P90 bands from simulated demand paths (stub)."""
    h = draws.shape[1] if draws.ndim == 2 else 1
    totals = draws.sum(axis=1) if draws.ndim == 2 else draws
    return {
        "p10": np.percentile(draws, 10, axis=0) if draws.ndim == 2 else np.zeros(h),
        "p50": np.percentile(draws, 50, axis=0) if draws.ndim == 2 else np.zeros(h),
        "p90": np.percentile(draws, 90, axis=0) if draws.ndim == 2 else np.zeros(h),
        "tot_p10": float(np.percentile(totals, 10)),
        "tot_p50": float(np.percentile(totals, 50)),
        "tot_p90": float(np.percentile(totals, 90)),
    }


def run(setup, n: int = 5000, tariff=None, vol=None, hedge=None,
        inflate: float = 1.0) -> dict:
    """Monte Carlo simulation (stub — returns zero arrays)."""
    h = getattr(setup, "H", getattr(setup, "h", 6))
    margin = np.zeros(max(n, 1))
    rev    = np.zeros(max(n, 1))
    return {
        "margin":     margin,
        "revenue":    rev,
        "rev":        rev,          # legacy alias
        "metal_need": {"cu": np.zeros((max(n, 1), h)),
                       "al": np.zeros((max(n, 1), h))},
        "vol":        np.zeros(max(n, 1)),
    }


def summarize(margin: np.ndarray, threshold: float = 0.0) -> dict:
    """Summarize a margin distribution."""
    mean = float(np.mean(margin)) if len(margin) else 0.0
    p5   = float(np.percentile(margin, 5))  if len(margin) else 0.0
    p95  = float(np.percentile(margin, 95)) if len(margin) else 0.0
    prob = float(np.mean(margin < threshold)) if len(margin) else 0.0
    worst = margin[margin <= p5] if len(margin) else np.array([0.0])
    cvar  = float(np.mean(worst)) if len(worst) else p5
    at_risk = mean - p5
    return {
        "mean": mean, "p5": p5, "p95": p95,
        "prob_below": prob,
        "at_risk": at_risk,
        "mar":      at_risk,        # legacy alias
        "cvar":     cvar,
    }


def deterministic(setup) -> dict:
    """Compute deterministic plan margin (stub)."""
    return {"margin": 0.0, "rev": 0.0, "metal_cost": 0.0}


def newsvendor(need, price_fx: float, rush_pct: float, hold_pct: float,
               k: int) -> dict:
    """Newsvendor optimal purchase quantity (stub)."""
    median = float(np.median(need)) if hasattr(need, "__len__") and len(need) else 0.0
    q_star = median
    n_pts  = 20
    curve_q    = np.linspace(0, median * 2 + 1, n_pts)
    curve_cost = np.zeros(n_pts)
    return {
        "q_star":         q_star,
        "p10":            0.0,
        "median":         median,
        "p90":            median * 1.2,
        "critical_ratio": 0.5,
        "under_cost":     0.0,
        "over_cost":      0.0,
        "cost_opt":       0.0,
        "cost_median":    0.0,
        "curve_q":        curve_q,
        "curve_cost":     curve_cost,
    }


def hedge_sweep(setup, ratios, months: int, fixed: dict, fee: float = 0.0,
                n: int = 2000, tariff=None, vol=None,
                inflate: float = 1.0) -> pd.DataFrame:
    """Sweep hedge fractions and return margin-at-risk per ratio (stub)."""
    rows = [{"cobertura": r, "en_riesgo": 0.0, "media": 0.0} for r in ratios]
    return pd.DataFrame(rows)


def coverage_sweep(setup, vol, tariff, inflate, fracs, months,
                   fee: float = 0.0, n: int = 2000) -> pd.DataFrame:
    """Legacy alias kept for compatibility."""
    return pd.DataFrame(columns=["frac", "mean", "p5", "mar"])
