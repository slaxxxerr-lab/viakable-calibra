"""engine.py — forecasting engine for Viakable Calibra.

All computation functions return real forecasts based on statistical methods.
"""
from __future__ import annotations

import dataclasses
import logging
import time

import numpy as np
import pandas as pd

from viakable_forecast import methods as M

logger = logging.getLogger(__name__)

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

# Accuracy label thresholds (error %)
_ACC = [("Muy certero", 5.0), ("Certero", 10.0), ("Aceptable", 20.0)]


def _accuracy_label(error_pct: float) -> str:
    if not np.isfinite(error_pct):
        return "Sin prueba"
    for label, threshold in _ACC:
        if error_pct < threshold:
            return label
    return "Poco confiable"


# ── Result dataclasses ────────────────────────────────────────────────────────

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
    error:        float = dataclasses.field(default_factory=lambda: float("nan"))
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


# ── Helper functions ──────────────────────────────────────────────────────────

def tariff_multiplier(scenario: float, current: float, elasticity: float) -> float:
    return ((1 + scenario) / (1 + current)) ** (-elasticity)


def windows_for(monthly: pd.DataFrame) -> list[tuple[int, int]]:
    """Return list of (n_training_months, forecast_horizon) for valid CV windows."""
    try:
        n = monthly["ym"].nunique()
    except Exception:
        n = 0
    return [(n - cut, 6) for cut in (6, 9, 12, 15) if n - cut >= 12]


def best_method(backtest_or_table) -> str:
    try:
        if isinstance(backtest_or_table, pd.DataFrame):
            elig = backtest_or_table[backtest_or_table.get("Elegible", True) == True]
            if not elig.empty:
                return str(elig["Método"].iloc[0])
            return str(backtest_or_table["Método"].iloc[0])
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


# ── Series extraction ─────────────────────────────────────────────────────────

def _get_series(
    monthly: pd.DataFrame,
    gauges: list[str],
    market: str | None = None,
    measure: str = TON,
    prices: dict | None = None,
) -> tuple[np.ndarray, list[pd.Timestamp]]:
    """Extract a continuous monthly series for the given gauge(s)."""
    m = monthly[monthly["cal"].isin(gauges)].copy()
    if market:
        m = m[m["mkt"] == market]
    if m.empty:
        return np.array([]), []
    if measure == MXN and prices:
        m = m.copy()
        for cal, price in prices.items():
            mask = m["cal"] == str(cal)
            m.loc[mask, "qty"] = m.loc[mask, "qty"] * float(price)
    agg = m.groupby("ym")["qty"].sum().sort_index()
    if agg.empty:
        return np.array([]), []
    all_months = pd.date_range(agg.index.min(), agg.index.max(), freq="MS")
    agg = agg.reindex(all_months, fill_value=0.0)
    return agg.values.astype(float), list(agg.index)


def series(monthly, units=None, market=None, measure=TON, prices=None, **kwargs) -> np.ndarray:
    """Extract total series for given calibres (flat list or list of tuples)."""
    try:
        m = monthly.copy()
        if units:
            flat = [str(g) for u in units for g in (u if hasattr(u, "__iter__") and not isinstance(u, str) else [u])]
            m = m[m["cal"].isin(flat)]
        if market:
            m = m[m["mkt"] == market]
        for k, v in kwargs.items():
            if k in m.columns and v:
                m = m[m[k].astype(str).isin([str(x) for x in v])]
        y, _ = _get_series(m, m["cal"].unique().tolist() if not units else
                           [str(g) for u in units for g in (u if hasattr(u, "__iter__") and not isinstance(u, str) else [u])],
                           None, measure, prices)
        return y
    except Exception:
        return np.zeros(0)


# ── Cross-validation ──────────────────────────────────────────────────────────

def cv_run(y: np.ndarray, windows: list[tuple[int, int]] | None = None) -> CVResult:
    """Run rolling-origin CV for all methods on series y."""
    n = len(y)
    if windows is None:
        windows = [(n - cut, 6) for cut in (6, 9, 12, 15) if n - cut >= 12]

    if not windows or n < 6:
        table = pd.DataFrame({
            "Método": [AUTO],
            "Error %": [float("nan")],
            "Precisión %": [float("nan")],
            "Sesgo %": [float("nan")],
            "Elegible": [False],
            "Tiempo s": [0.0],
        })
        return CVResult(table=table, fcs={AUTO: [None]}, actual=[])

    raw = M.backtest(y, windows)

    rows = []
    all_fcs: dict[str, list] = {}
    for name, res in raw.items():
        mean_err = float(np.mean(res["errors"])) if res["errors"] else float("nan")
        mean_bias = float(np.mean(res["bias"])) if res["bias"] else float("nan")
        n_win = res["n_windows"]
        eligible = (name not in AUTO_EXCLUDE) and (n_win >= 1) and np.isfinite(mean_err)
        # holt_winters and regresión need ≥2 windows to be eligible
        if name in ("holt_winters", "regresión") and n_win < 2:
            eligible = False
        rows.append({
            "Método": name,
            "Error %": mean_err,
            "Precisión %": max(0.0, 100.0 - mean_err) if np.isfinite(mean_err) else float("nan"),
            "Sesgo %": mean_bias,
            "Elegible": eligible,
            "n_windows": n_win,
            "Tiempo s": 0.0,
        })
        # Store the last-window forecasts for ensemble building
        last_t, last_h = windows[-1]
        try:
            fc_last = M.forecast(y[:last_t], last_h, name)
            all_fcs[name] = [fc_last]
        except Exception:
            all_fcs[name] = [None]

    if not rows:
        rows.append({"Método": AUTO, "Error %": float("nan"),
                     "Precisión %": float("nan"), "Sesgo %": float("nan"),
                     "Elegible": False, "n_windows": 0, "Tiempo s": 0.0})

    df = pd.DataFrame(rows).sort_values("Error %").reset_index(drop=True)
    # Add ensemble as the inverse-error weighted average of top-5 eligible methods
    elig = df[df["Elegible"] & df["Error %"].apply(np.isfinite)].head(5)
    if len(elig) >= 2:
        weights = 1.0 / (elig["Error %"].values + 1e-9)
        weights /= weights.sum()
        ens_err = float(np.dot(weights, elig["Error %"].values))
        df = pd.concat([
            pd.DataFrame([{
                "Método": "ensemble",
                "Error %": ens_err,
                "Precisión %": max(0.0, 100.0 - ens_err),
                "Sesgo %": 0.0,
                "Elegible": True,
                "n_windows": int(elig["n_windows"].max()),
                "Tiempo s": 0.0,
            }]),
            df,
        ], ignore_index=True).sort_values("Error %").reset_index(drop=True)

    actual_vals = list(y[windows[-1][0]:windows[-1][0] + windows[-1][1]])
    return CVResult(table=df, fcs=all_fcs, actual=actual_vals)


# ── Backtest for a selection (all gauges together) ────────────────────────────

def selection_backtest(
    monthly: pd.DataFrame,
    gauges: list | tuple,
    market: str | None,
    measure: str,
    prices: dict | None,
    ml=None,
) -> pd.DataFrame:
    """Run rolling-origin CV for the total series of the given gauges."""
    t0 = time.perf_counter()
    y, _ = _get_series(monthly, list(gauges), market, measure, prices)
    windows = windows_for(monthly)
    if len(y) < 6:
        return pd.DataFrame(columns=["Método", "Error %", "Precisión %", "Sesgo %", "Elegible", "Tiempo s"])

    result = cv_run(y, windows)
    elapsed = time.perf_counter() - t0
    df = result.table.copy()
    df["Tiempo s"] = round(elapsed / max(len(df), 1), 3)
    # Rename n_windows if present
    if "n_windows" in df.columns:
        df = df.drop(columns=["n_windows"])
    return df


# ── Confidence interval ───────────────────────────────────────────────────────

def interval(
    actual: list | np.ndarray,
    fcs: list | np.ndarray,
    fc: np.ndarray,
    inflate: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 80% confidence interval from cross-validation residuals."""
    try:
        a = np.asarray(actual, float)
        f = np.asarray(fcs, float)
        if len(a) > 0 and len(f) > 0 and len(f) == len(a):
            ratios = (a - f) / (np.abs(a) + 1e-9)
            lo_q, hi_q = float(np.percentile(ratios, 10)), float(np.percentile(ratios, 90))
        else:
            lo_q, hi_q = -0.15, 0.15
        lo_q *= inflate
        hi_q *= inflate
        lo = fc * (1 + lo_q)
        hi = fc * (1 + hi_q)
        return np.maximum(0.0, lo), np.maximum(0.0, hi)
    except Exception:
        return fc * 0.85 * inflate, fc * 1.15 * inflate


# ── Forecast for a single gauge ───────────────────────────────────────────────

def _fallback_chain(y: np.ndarray, h: int, preferred: str) -> tuple[np.ndarray, str]:
    """Try preferred method, then fallback chain, never return zeros unless data is empty."""
    candidates = [
        preferred,
        "mismo_mes_año",
        "holt",
        "suavización",
        "promedio_3m",
    ]
    seen = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        try:
            fc = M.forecast(y, h, name)
            if np.any(fc > 0) or np.sum(y) == 0:
                return fc, name
        except Exception:
            continue
    # Absolute last resort: 3-month average
    ma = float(np.mean(y[-3:])) if len(y) >= 3 else float(np.mean(y) if len(y) > 0 else 0)
    return np.full(h, ma), "promedio_3m"


def _gauge_forecast(
    monthly: pd.DataFrame,
    gauge: str,
    market: str | None,
    horizon: int,
    method: str,
    measure: str,
    prices: dict | None,
    cv_table: pd.DataFrame | None = None,
) -> dict:
    """Compute forecast for a single calibre. Returns detail dict."""
    y, dates = _get_series(monthly, [gauge], market, measure, prices)
    n = len(y)
    demand_cls, adi, cv2 = M.demand_class(y)

    # Determine best method for this gauge
    if method == AUTO or method == "ensemble":
        windows = [(n - c, 6) for c in (6, 9, 12, 15) if n - c >= 12]
        if windows:
            cv_res = cv_run(y, windows)
            elig = cv_res.table[cv_res.table["Elegible"] & cv_res.table["Error %"].apply(np.isfinite)]
            if not elig.empty:
                best = str(elig["Método"].iloc[0])
                # Use ensemble unless another method beats it by 2+ points
                if best == "ensemble" and len(elig) >= 2:
                    chosen = "ensemble"
                else:
                    chosen = best
            else:
                chosen = "promedio_3m"
            cv_table_g = cv_res.table
        else:
            chosen = "mismo_mes_año" if n >= 12 else "promedio_3m"
            cv_table_g = pd.DataFrame()
    else:
        chosen = method
        cv_table_g = pd.DataFrame()

    # Intermittent demand → prefer Croston/TSB
    if demand_cls in ("Intermitente", "Irregular") and method == AUTO:
        if chosen not in ("croston", "tsb"):
            # Only override if Croston/TSB has similar or better error
            if not cv_table_g.empty:
                elig_crst = cv_table_g[cv_table_g["Método"].isin(["croston", "tsb"]) & cv_table_g["Elegible"]]
                if not elig_crst.empty:
                    crst_err = float(elig_crst["Error %"].min())
                    best_err = cv_table_g.loc[cv_table_g["Método"] == chosen, "Error %"].squeeze()
                    best_err = float(best_err) if np.isfinite(float(best_err)) else float("inf")
                    if crst_err <= best_err + 5.0:
                        chosen = str(elig_crst.loc[elig_crst["Error %"].idxmin(), "Método"])

    # Build ensemble forecast
    if chosen == "ensemble" and not cv_table_g.empty:
        elig = cv_table_g[cv_table_g["Elegible"] & cv_table_g["Método"].ne("ensemble") &
                          cv_table_g["Error %"].apply(np.isfinite)].head(5)
        if not elig.empty:
            w = 1.0 / (elig["Error %"].values + 1e-9)
            w /= w.sum()
            fc_parts = np.array([M.forecast(y, horizon, str(m)) for m in elig["Método"]])
            fc = np.maximum(0.0, np.dot(w, fc_parts))
        else:
            fc, chosen = _fallback_chain(y, horizon, "holt")
    else:
        fc, chosen = _fallback_chain(y, horizon, chosen)

    # Error from CV table
    err = float("nan")
    if not cv_table_g.empty:
        row = cv_table_g[cv_table_g["Método"] == chosen]
        if not row.empty:
            err = float(row["Error %"].iloc[0])
        elif len(cv_table_g) > 0:
            best_row = cv_table_g[cv_table_g["Elegible"] & cv_table_g["Error %"].apply(np.isfinite)]
            if not best_row.empty:
                err = float(best_row["Error %"].iloc[0])

    return {
        "label":   gauge,
        "y":       y,
        "fc":      fc,
        "error":   err,
        "method":  chosen,
        "demand":  demand_cls,
        "dates":   dates,
    }


# ── Main projection function ──────────────────────────────────────────────────

def project(
    monthly: pd.DataFrame,
    units: list,
    market: str | None,
    horizon: int,
    method: str,
    measure: str,
    mode: str,
    mult: dict,
    sig: dict,
    w: int,
    prices: dict | None,
    crm=None,
    ml=None,
    apply_signals: bool = False,
) -> Projection:
    """Compute a real forecast for the selected calibre groups."""
    h = int(horizon) if horizon else 6

    # Flatten units: each group may contain multiple gauges (e.g., 16+18)
    gauge_groups: list[list[str]] = [[str(g) for g in u] for u in (units or [])]
    if not gauge_groups:
        gauge_groups = [[str(g)] for g in sorted(monthly["cal"].unique())]

    # Build historical series (total across all selected gauges)
    all_gauges = [g for grp in gauge_groups for g in grp]
    hist_y, hist_dates = _get_series(monthly, all_gauges, market, measure, prices)
    hist = pd.Series(hist_y, index=hist_dates) if hist_dates else pd.Series(dtype=float)

    # Forecast each gauge group
    detail: list[dict] = []
    for grp in gauge_groups:
        if len(grp) == 1:
            d = _gauge_forecast(monthly, grp[0], market, h, method, measure, prices)
            detail.append(d)
        else:
            # Multiple gauges in one group: forecast each, combine
            sub_details = [_gauge_forecast(monthly, g, market, h, method, measure, prices) for g in grp]
            combined_fc = np.sum([d["fc"] for d in sub_details], axis=0)
            combined_y = np.sum([d["y"] for d in sub_details], axis=0) if all(len(d["y"]) > 0 for d in sub_details) else np.array([])
            errs = [d["error"] for d in sub_details if np.isfinite(d["error"])]
            combined_err = float(np.mean(errs)) if errs else float("nan")
            detail.append({
                "label":  "+".join(grp),
                "y":      combined_y,
                "fc":     combined_fc,
                "error":  combined_err,
                "method": sub_details[0]["method"] if sub_details else AUTO,
                "demand": sub_details[0]["demand"] if sub_details else "Estable",
            })

    if not detail:
        fc = np.zeros(h)
        idx: list = []
    elif mode == "sum" or len(detail) == 1:
        fc = np.sum([d["fc"] for d in detail], axis=0)
    else:
        # Individual: pick first group's forecast
        fc = detail[0]["fc"]

    # Forecast index (months after last actual)
    try:
        last = monthly["ym"].max()
        idx = [last + pd.DateOffset(months=i + 1) for i in range(h)]
    except Exception:
        idx = list(range(h))

    # Apply tariff multiplier to exports if applicable
    if mult and market and measure == TON:
        tariff_m = mult.get("tariff", 1.0)
        if tariff_m != 1.0:
            fc = fc * tariff_m
            for d in detail:
                d["fc"] = d["fc"] * tariff_m

    # Confidence interval
    lo, hi = fc * 0.85, fc * 1.15  # fallback
    if len(hist_y) >= 6:
        try:
            n = len(hist_y)
            windows = [(n - c, 6) for c in (6, 9, 12, 15) if n - c >= 12]
            if windows:
                cv_res = cv_run(hist_y, windows)
                if cv_res.actual and cv_res.fcs:
                    last_win_fc = next(iter(cv_res.fcs.values()), [None])[0]
                    if last_win_fc is not None:
                        n_w = len(windows)
                        inflate = INFLATE_BY_WINDOWS.get(n_w, 1.0)
                        lo, hi = interval(cv_res.actual, last_win_fc, fc, inflate)
        except Exception:
            pass

    # Build table
    table_rows = []
    for d in detail:
        lbl = d["label"]
        fc_total = float(np.sum(d["fc"]))
        err = d["error"]
        # Same month last year for comparison
        y_arr = np.asarray(d.get("y", []), float)
        n = len(y_arr)
        if n >= 12:
            prev_12 = float(np.sum(y_arr[max(0, n - 12 - h):max(0, n - h)]))
        else:
            prev_12 = float(np.sum(y_arr))
        chg = float((fc_total - prev_12) / prev_12 * 100) if prev_12 > 0 else float("nan")
        table_rows.append({
            "Calibre":                    lbl,
            "Método":                     d.get("method", AUTO),
            "Error %":                    err,
            "Precisión %":                max(0.0, 100.0 - err) if np.isfinite(err) else float("nan"),
            "Evaluación":                 _accuracy_label(err),
            "Demanda":                    d.get("demand", "Estable"),
            "Pronóstico":                 fc_total,
            "Mismos meses año anterior":  prev_12,
            "Cambio %":                   chg,
        })

    if not table_rows:
        table_rows.append({
            "Calibre": "—", "Método": AUTO, "Error %": float("nan"),
            "Precisión %": float("nan"), "Evaluación": "—", "Demanda": "Estable",
            "Pronóstico": 0.0, "Mismos meses año anterior": 0.0, "Cambio %": float("nan"),
        })
    table = pd.DataFrame(table_rows)

    total_err = float(np.nanmean([d["error"] for d in detail])) if detail else float("nan")
    chosen_method = detail[0]["method"] if len(detail) == 1 else AUTO

    return Projection(
        fc=fc,
        hist=hist,
        lo=lo,
        hi=hi,
        method=chosen_method,
        detail=detail,
        table=table,
        index=idx,
        error=total_err,
    )


# ── Plan (minimal stub — planning.py owns this) ───────────────────────────────

def plan(monthly, gauges, horizon, params, prices, mult, sig, w,
         overrides=None, ml=None, crm=None) -> PlanResult:
    cap_rows = [{"cal": str(g), "over": 0.0, "util": 0.0} for g in (gauges or [])]
    cap = pd.DataFrame(cap_rows) if cap_rows else pd.DataFrame(columns=["cal", "over", "util"])
    return PlanResult(capacity=cap, detail=[])


# ── Additional engine helpers ─────────────────────────────────────────────────

def forecast(y: np.ndarray, h: int, method: str, table=None) -> np.ndarray:
    return M.forecast(y, h, method)


def accuracy_matrix(monthly=None, units=None, *_a, **_kw) -> pd.DataFrame:
    try:
        gauges = [str(u[0]) if u else "?" for u in (units or [])]
    except Exception:
        gauges = []
    cols = gauges + ["Total"]
    return pd.DataFrame({c: [0.0] for c in cols}, index=[AUTO])


def method_totals(*_a, **_kw) -> dict:
    return {}


def signal_row(hist_values, w) -> dict:
    return {}


def ml_vs_stat(mx) -> pd.DataFrame:
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
