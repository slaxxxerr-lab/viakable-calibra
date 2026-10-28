"""planning.py — stub with working PlanParams and dummy plan() return (engine not available on this machine)."""
from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pandas as pd


# ── PlanParams ────────────────────────────────────────────────────────────────

class PlanParams:
    """Flexible plan parameters — accepts and stores any keyword arguments."""
    def __init__(self, **kwargs):
        self.service       = 95.0
        self.lead_time     = 1.5
        self.hold_pct      = 25.0
        self.metal_pct     = 70.0
        self.cu_lme        = 14770.0
        self.al_lme        = 3181.0
        self.metal_chg     = 0.0
        self.fx            = 18.43
        self.conv_cu       = 5000.0
        self.conv_al       = 4500.0
        self.exp_lt        = 3.0
        self.freight       = 200.0
        self.ddp           = False
        self.tariff_scen   = 0.50
        self.base_method   = "auto"
        self.cu_path: Any  = None
        self.al_path: Any  = None
        self.fx_path: Any  = None
        self.pass_through  = 50.0
        self.apply_signals = False
        self.cu_ref        = 9840.0
        self.al_ref        = 2600.0
        self.fx_ref        = 18.5
        for k, v in kwargs.items():
            setattr(self, k, v)


# ── PlanResult ────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class PlanResult:
    rows:     pd.DataFrame
    capacity: pd.DataFrame
    months:   list
    params:   Any      # PlanParams


# ── Column lists ──────────────────────────────────────────────────────────────

_ROW_COLS = [
    "cal", "mkt", "is_exp", "metal", "mean", "error",
    "ss", "rop", "inv", "prod_t", "price", "rev", "margin",
    "freed", "metal_req", "prod", "vol", "fc",
]
_CAP_COLS = ["cal", "over", "cap", "util", "deficit"]


# ── totals() ──────────────────────────────────────────────────────────────────

def totals(res, market=None) -> dict:
    """Return a zero-filled totals dict (engine unavailable on this machine)."""
    z = 0.0
    return {
        "vol":           z,
        "rev":           z,
        "margin":        z,
        "margin_pct":    z,
        "margin_lost_10": z,
        "cu":            z,
        "al":            z,
        "ss":            z,
        "ss_value":      z,
        "freed":         z,
        "saving":        z,
        "err":           z,
        "err_trad":      z,
        "exp":           z,
        "over_months":   0,
        "metal_cost":    z,
        "fc":            z,
        "Error ML %":          float("nan"),
        "Error estadístico %": float("nan"),
        "Mejor estadístico":   "—",
    }


# ── plan() ────────────────────────────────────────────────────────────────────

def plan(monthly, gauges, horizon, params, prices, mult, sig, w,
         ml=None, overrides=None, crm=None) -> PlanResult:
    """Return a dummy plan result (engine unavailable on this machine)."""
    h = int(horizon) if horizon else 6

    # One stub row per gauge
    rows_data = []
    for g in (gauges or []):
        rows_data.append({
            "cal":      str(g),
            "mkt":      "Nacional",
            "is_exp":   False,
            "metal":    "cu",
            "mean":     0.0,
            "error":    float("nan"),
            "ss":       0.0,
            "rop":      0.0,
            "inv":      0.0,
            "prod_t":   0.0,
            "price":    0.0,
            "rev":      0.0,
            "margin":   0.0,
            "freed":    0.0,
            "metal_req": np.zeros(h),
            "prod":     np.zeros(h),
            "vol":      0.0,
            "fc":       np.zeros(h),
        })
    rows = (pd.DataFrame(rows_data, columns=_ROW_COLS)
            if rows_data
            else pd.DataFrame(columns=_ROW_COLS))

    cap_data = [{"cal": str(g), "over": 0, "cap": 0.0,
                 "util": [0.0] * h, "deficit": 0.0} for g in (gauges or [])]
    cap = (pd.DataFrame(cap_data, columns=_CAP_COLS)
           if cap_data
           else pd.DataFrame(columns=_CAP_COLS))

    try:
        last = monthly["ym"].max()
        months = [last + pd.DateOffset(months=i + 1) for i in range(h)]
    except Exception:
        months = list(range(h))

    return PlanResult(rows=rows, capacity=cap, months=months, params=params)


# ── Other stubs ───────────────────────────────────────────────────────────────

def safety_stock(*_a, **_kw):
    raise NotImplementedError("Motor no disponible en este equipo.")


def scenarios(*_a, **_kw):
    raise NotImplementedError("Motor no disponible en este equipo.")


def purchase_rec(*_a, **_kw):
    raise NotImplementedError("Motor no disponible en este equipo.")
