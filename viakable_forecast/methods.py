"""methods.py — statistical forecasting methods for Viakable Calibra.

All methods take (y: np.ndarray, h: int) and return np.ndarray of length h.
y must be a complete monthly time series (fill missing months with 0 before calling).
"""
from __future__ import annotations

import logging
import time
import warnings
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

# ── Error metrics ─────────────────────────────────────────────────────────────

def mape(actual, forecast) -> float:
    try:
        a, f = np.asarray(actual, float), np.asarray(forecast, float)
        mask = a != 0
        return float(np.mean(np.abs((a[mask] - f[mask]) / a[mask])) * 100) if mask.any() else float("nan")
    except Exception:
        return float("nan")


def wape(actual, forecast) -> float:
    try:
        a, f = np.asarray(actual, float), np.asarray(forecast, float)
        denom = float(np.sum(np.abs(a)))
        return float(np.sum(np.abs(a - f)) / denom * 100) if denom > 0 else float("nan")
    except Exception:
        return float("nan")


def error_metric(actual, forecast) -> float:
    """MAPE when actuals are non-zero, WAPE otherwise."""
    a = np.asarray(actual, float)
    if np.any(a != 0):
        return mape(actual, forecast)
    return wape(actual, forecast)


def bias(actual, forecast) -> float:
    try:
        a, f = np.asarray(actual, float), np.asarray(forecast, float)
        denom = float(np.sum(np.abs(a)))
        return float(np.sum(f - a) / denom * 100) if denom > 0 else float("nan")
    except Exception:
        return float("nan")


# ── Demand classification (Syntetos-Boylan) ───────────────────────────────────

def demand_class(y) -> tuple[str, float, float]:
    """Return (class_name, ADI, CV²) using Syntetos-Boylan quadrant."""
    try:
        arr = np.asarray(y, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return "Estable", 0.0, 0.0
        nonzero = arr[arr > 0]
        if len(nonzero) == 0:
            return "Intermitente", float("inf"), 0.0
        nz_idx = np.where(arr > 0)[0]
        adi = float(np.mean(np.diff(nz_idx))) if len(nz_idx) > 1 else float(len(arr))
        cv2 = float((np.std(nonzero) / np.mean(nonzero)) ** 2) if np.mean(nonzero) > 0 else 0.0
        if adi >= 1.32 and cv2 < 0.49:
            cls = "Intermitente"
        elif adi >= 1.32 and cv2 >= 0.49:
            cls = "Irregular"
        elif adi < 1.32 and cv2 >= 0.49:
            cls = "Errática"
        else:
            cls = "Estable"
        return cls, round(adi, 2), round(cv2, 2)
    except Exception:
        return "Estable", 0.0, 0.0


# ── Individual forecasting methods ────────────────────────────────────────────

def same_month_last_year(y: np.ndarray, h: int, growth: float = 0.0) -> np.ndarray:
    """Seasonal naive: same calendar month last year × (1 + average_growth)."""
    n = len(y)
    if n < 12:
        return moving_average(y, h)
    # Estimate year-over-year growth from available data
    if growth == 0.0 and n >= 24:
        g_vals = []
        for i in range(12, min(n, 24)):
            if y[i - 12] > 0:
                g_vals.append(y[i] / y[i - 12] - 1)
        growth = float(np.median(g_vals)) if g_vals else 0.0
    out = []
    for k in range(h):
        src_idx = n - 12 + (k % 12)
        if 0 <= src_idx < n:
            out.append(max(0.0, y[src_idx] * (1.0 + growth)))
        else:
            out.append(max(0.0, float(np.mean(y[-12:]))))
    return np.array(out)


def moving_average(y: np.ndarray, h: int, window: int = 3) -> np.ndarray:
    """Simple moving average of the last `window` months."""
    if len(y) == 0:
        return np.zeros(h)
    w = min(window, len(y))
    ma = float(np.mean(y[-w:]))
    return np.full(h, max(0.0, ma))


def ses(y: np.ndarray, h: int, alpha: float | None = None) -> np.ndarray:
    """Simple exponential smoothing with optimal alpha via grid search."""
    n = len(y)
    if n == 0:
        return np.zeros(h)
    if alpha is None:
        best_sse, alpha = float("inf"), 0.3
        for a in np.arange(0.05, 0.96, 0.05):
            level = y[0]
            sse = 0.0
            for v in y[1:]:
                sse += (v - level) ** 2
                level = a * v + (1.0 - a) * level
            if sse < best_sse:
                best_sse, alpha = sse, float(a)
    level = y[0]
    for v in y[1:]:
        level = alpha * v + (1.0 - alpha) * level
    return np.full(h, max(0.0, level))


def holt(y: np.ndarray, h: int, phi: float = 0.9) -> np.ndarray:
    """Holt's method with damped trend (additive, optimized alpha/beta)."""
    n = len(y)
    if n < 3:
        return moving_average(y, h)

    best_sse, best_params = float("inf"), (0.3, 0.1)
    for a in (0.1, 0.2, 0.3, 0.5, 0.7):
        for b in (0.05, 0.1, 0.2):
            level, trend = y[0], y[1] - y[0] if n > 1 else 0.0
            sse = 0.0
            for v in y[1:]:
                prev = level
                level = a * v + (1.0 - a) * (level + phi * trend)
                trend = b * (level - prev) + (1.0 - b) * phi * trend
                sse += (v - (prev + phi * trend)) ** 2
            if sse < best_sse:
                best_sse, best_params = sse, (a, b)

    alpha, beta = best_params
    level = y[0]
    trend = y[1] - y[0] if n > 1 else 0.0
    for v in y[1:]:
        prev = level
        level = alpha * v + (1.0 - alpha) * (level + phi * trend)
        trend = beta * (level - prev) + (1.0 - beta) * phi * trend

    phi_sum = 0.0
    out = []
    for k in range(1, h + 1):
        phi_sum += phi ** k
        out.append(max(0.0, level + phi_sum * trend))
    return np.array(out)


def holt_winters(y: np.ndarray, h: int, period: int = 12) -> np.ndarray:
    """Additive Holt-Winters. Requires at least 2 full seasons."""
    n = len(y)
    if n < 2 * period:
        return holt(y, h)
    try:
        alpha, beta, gamma = 0.3, 0.1, 0.2
        # Initialize seasonal indices from first full year
        s = y[:period] - np.mean(y[:period])
        level = float(np.mean(y[:period]))
        trend = float((np.mean(y[period:2 * period]) - np.mean(y[:period])) / period)
        for i in range(period, n):
            si = i % period
            prev = level
            level = alpha * (y[i] - s[si]) + (1.0 - alpha) * (level + trend)
            trend = beta * (level - prev) + (1.0 - beta) * trend
            s[si] = gamma * (y[i] - level) + (1.0 - gamma) * s[si]
        out = []
        for k in range(h):
            si = (n + k) % period
            out.append(max(0.0, level + (k + 1) * trend + s[si]))
        return np.array(out)
    except Exception:
        return holt(y, h)


def regression_seasonal(y: np.ndarray, h: int, period: int = 12) -> np.ndarray:
    """Linear regression on deseasonalized series (seasonal index = ratio to 12m MA)."""
    n = len(y)
    if n < period + 1:
        return holt(y, h)
    try:
        # Centered 12-month moving average
        half = period // 2
        ma = np.array([np.mean(y[max(0, i - half):i + half + 1]) for i in range(n)])
        # Seasonal factors (ratio method)
        factors = np.zeros(period)
        counts = np.zeros(period)
        for i in range(half, n - half):
            if ma[i] > 0:
                factors[i % period] += y[i] / ma[i]
                counts[i % period] += 1
        counts = np.where(counts == 0, 1, counts)
        factors /= counts
        # Normalize so factors average to 1
        mean_f = np.mean(factors)
        if mean_f > 0:
            factors /= mean_f
        factors = np.where(factors == 0, 1.0, factors)
        # Deseasonalize
        si_arr = np.array([factors[i % period] for i in range(n)])
        deseaz = np.where(si_arr > 0, y / si_arr, y)
        # Linear regression
        x = np.arange(n, dtype=float)
        slope, intercept = float(np.polyfit(x, deseaz, 1))
        out = []
        for k in range(h):
            trend = intercept + slope * (n + k)
            seasonal = factors[(n + k) % period]
            out.append(max(0.0, trend * seasonal))
        return np.array(out)
    except Exception:
        return holt(y, h)


def croston_sba(y: np.ndarray, h: int, alpha: float = 0.1) -> np.ndarray:
    """Croston SBA (Syntetos-Boylan approximation) for intermittent demand."""
    n = len(y)
    if n == 0:
        return np.zeros(h)
    nz_idx = np.where(y > 0)[0]
    if len(nz_idx) == 0:
        return np.zeros(h)
    if len(nz_idx) == 1:
        return np.full(h, y[nz_idx[0]] / n)

    z_hat = float(y[nz_idx[0]])
    p_hat = 1.0
    for i in range(1, len(nz_idx)):
        gap = float(nz_idx[i] - nz_idx[i - 1])
        z_hat = alpha * y[nz_idx[i]] + (1.0 - alpha) * z_hat
        p_hat = alpha * gap + (1.0 - alpha) * p_hat

    # SBA correction factor (1 - alpha/2)
    rate = (z_hat * (1.0 - alpha / 2.0)) / max(p_hat, 1.0)
    return np.full(h, max(0.0, rate))


def tsb(y: np.ndarray, h: int, alpha: float = 0.1, beta: float = 0.1) -> np.ndarray:
    """Teunter-Syntetos-Babai (TSB) method for intermittent demand."""
    n = len(y)
    if n == 0:
        return np.zeros(h)
    nz_idx = np.where(y > 0)[0]
    if len(nz_idx) == 0:
        return np.zeros(h)

    # Initialize: probability and demand size
    p_hat = float(len(nz_idx)) / n
    z_hat = float(np.mean(y[nz_idx]))

    for v in y:
        if v > 0:
            p_hat = beta + (1.0 - beta) * p_hat
            z_hat = alpha * v + (1.0 - alpha) * z_hat
        else:
            p_hat = (1.0 - beta) * p_hat

    return np.full(h, max(0.0, p_hat * z_hat))


def _try_ets(y: np.ndarray, h: int) -> np.ndarray | None:
    """ETS (automatic Holt-Winters family) via statsmodels. Returns None on failure."""
    try:
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            from statsmodels.tsa.holtwinters import ExponentialSmoothing as ES
            n = len(y)
            trend = "add" if n >= 5 else None
            seasonal = "add" if n >= 24 else None
            sp = 12 if seasonal else None
            fit = ES(y, trend=trend, seasonal=seasonal, seasonal_periods=sp,
                     initialization_method="estimated").fit(optimized=True)
            fc = fit.forecast(h)
            return np.maximum(0.0, np.asarray(fc, dtype=float))
    except Exception:
        return None


def _try_sarima(y: np.ndarray, h: int) -> np.ndarray | None:
    """Automatic SARIMA via statsmodels. Returns None on failure."""
    try:
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            import pandas as _pd
            n = len(y)
            idx = _pd.date_range("2000-01", periods=n, freq="MS")
            ts = _pd.Series(y, index=idx)
            best_aic, best_fc = float("inf"), None
            for p, d, q in [(1, 1, 0), (0, 1, 1), (1, 1, 1), (0, 1, 0), (2, 1, 0)]:
                try:
                    P, D, Q, s = (1, 0, 0, 12) if n >= 24 else (0, 0, 0, 0)
                    fit = SARIMAX(ts, order=(p, d, q),
                                  seasonal_order=(P, D, Q, s),
                                  enforce_stationarity=False,
                                  enforce_invertibility=False).fit(disp=False)
                    if fit.aic < best_aic:
                        best_aic = fit.aic
                        best_fc = np.maximum(0.0, np.asarray(fit.forecast(h), float))
                except Exception:
                    continue
            return best_fc
    except Exception:
        return None


def ets_auto(y: np.ndarray, h: int) -> np.ndarray:
    fc = _try_ets(y, h)
    return fc if fc is not None else holt(y, h)


def sarima_auto(y: np.ndarray, h: int) -> np.ndarray:
    fc = _try_sarima(y, h)
    return fc if fc is not None else holt(y, h)


# ── Method dispatch table ─────────────────────────────────────────────────────

# (id → callable(y, h) → np.ndarray)
ALL_METHODS: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "mismo_mes_año":   lambda y, h: same_month_last_year(y, h),
    "promedio_3m":     lambda y, h: moving_average(y, h, 3),
    "suavización":     lambda y, h: ses(y, h),
    "holt":            lambda y, h: holt(y, h),
    "holt_winters":    lambda y, h: holt_winters(y, h),
    "regresión":       lambda y, h: regression_seasonal(y, h),
    "croston":         lambda y, h: croston_sba(y, h),
    "tsb":             lambda y, h: tsb(y, h),
    "ets":             lambda y, h: ets_auto(y, h),
    "sarima":          lambda y, h: sarima_auto(y, h),
}

# Minimum training months required for each method
MIN_TRAIN: dict[str, int] = {
    "mismo_mes_año":  12,
    "promedio_3m":     3,
    "suavización":     6,
    "holt":            6,
    "holt_winters":   24,
    "regresión":      13,
    "croston":         6,
    "tsb":             6,
    "ets":             6,
    "sarima":         12,
}


def forecast(y: np.ndarray, h: int, method: str) -> np.ndarray:
    """Call a single method by name. Falls back gracefully."""
    fn = ALL_METHODS.get(method)
    if fn is None:
        return moving_average(y, h)
    try:
        fc = fn(y, h)
        if not np.all(np.isfinite(fc)):
            fc = moving_average(y, h)
        return np.maximum(0.0, fc)
    except Exception as exc:
        logger.debug("method %s failed: %s", method, exc)
        return moving_average(y, h)


# ── Cross-validation for a single series ─────────────────────────────────────

def backtest(y: np.ndarray, windows: list[tuple[int, int]]) -> dict[str, dict]:
    """Run rolling-origin CV for all eligible methods.

    windows: list of (n_train, h) pairs, e.g. [(24,6),(21,6),(18,6),(15,6)]

    Returns {method_id: {"errors": [err_per_window], "bias": [bias_per_window]}}
    """
    results: dict[str, dict] = {}
    n = len(y)

    for name, fn in ALL_METHODS.items():
        min_t = MIN_TRAIN.get(name, 6)
        valid_windows = [(t, h) for t, h in windows if t >= min_t and t + h <= n]
        if not valid_windows:
            continue
        errors, biases = [], []
        for n_train, h in valid_windows:
            train = y[:n_train]
            actual = y[n_train:n_train + h]
            try:
                fc = fn(train, h)
                if not np.all(np.isfinite(fc)):
                    continue
                fc = np.maximum(0.0, fc)
                err = error_metric(actual, fc)
                b = bias(actual, fc)
                if np.isfinite(err):
                    errors.append(err)
                if np.isfinite(b):
                    biases.append(b)
            except Exception:
                continue
        if errors:
            results[name] = {"errors": errors, "bias": biases, "n_windows": len(errors)}

    return results


def all_methods() -> list[str]:
    return list(ALL_METHODS.keys())
