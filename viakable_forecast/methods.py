"""methods.py — stub (engine not available on this machine)."""
from __future__ import annotations

import numpy as np


def _unavailable(*_a, **_kw):
    raise NotImplementedError("Motor no disponible en este equipo.")


def demand_class(y) -> tuple[str, float, float]:
    """Return (class_name, ADI, CV²) using Syntetos-Boylan quadrant (stub)."""
    try:
        arr = np.asarray(y, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return "Estable", 0.0, 0.0
        nonzero = arr[arr > 0]
        if len(nonzero) == 0:
            return "Intermitente", float("inf"), 0.0
        # ADI: average inter-demand interval
        nz_idx = np.where(arr > 0)[0]
        if len(nz_idx) > 1:
            adi = float(np.mean(np.diff(nz_idx)))
        else:
            adi = float(len(arr))
        # CV²: squared coefficient of variation of nonzero demand
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
        denom = np.sum(np.abs(a))
        return float(np.sum(np.abs(a - f)) / denom * 100) if denom > 0 else float("nan")
    except Exception:
        return float("nan")


forecast    = _unavailable
backtest    = _unavailable
all_methods = _unavailable
