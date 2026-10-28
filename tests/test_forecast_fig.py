"""Tests for ui.forecast_fig — chart structure after the dark-mode regression fix.

Spec numbers (from the user's report):
  - 30 history months: Jan 2024 – Jun 2026
  - Total history: 18 210.4 t
  - Last 12 months (Jul 2025 – Jun 2026): 7 523.6 t  →  avg ≈ 627 t/m
  - Forecast starts Jul 2026 at 1 071 t/m
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

import ui
from ui import forecast_fig

# ── constants ─────────────────────────────────────────────────────────────────

N_HIST = 30   # Jan 2024 – Jun 2026
N_FC   = 12   # Jul 2026 – Jun 2027

HIST_DATES = pd.date_range("2024-01-01", periods=N_HIST, freq="MS")
FC_DATES   = pd.date_range("2026-07-01", periods=N_FC,   freq="MS")

_AVG_LAST12 = 7_523.6 / 12          # ≈ 626.97 t
_AVG_FIRST18 = (18_210.4 - 7_523.6) / 18

HIST_VALS = np.array([_AVG_FIRST18] * 18 + [_AVG_LAST12] * 12)
FC_VALS   = np.ones(N_FC) * 1_071.0
LO_VALS   = FC_VALS - 66.0   # 1 005 t lower bound
HI_VALS   = FC_VALS + 44.0   # 1 115 t upper bound

# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def light_theme():
    """Light theme → FG='#000000' (black on white); predictable stable colors."""
    ui.configure(
        {"theme": "light", "accent": "#ff0000", "font": "brand",
         "density": "comfy", "unit": "t", "lang": "es", "anim": "off"},
        system_dark=False,
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def _fig(trend=True, lo=LO_VALS, hi=HI_VALS):
    return forecast_fig(
        list(HIST_DATES), HIST_VALS,
        list(FC_DATES),   FC_VALS,
        lo, hi,
        unit="t", main_label="Pronóstico (Auto)",
        trend=trend,
    )


def _trace(fig, name):
    for tr in fig.data:
        if tr.name == name:
            return tr
    return None


def _named(fig):
    return [tr for tr in fig.data if tr.name and getattr(tr, "showlegend", True) is not False]


# ── existence ─────────────────────────────────────────────────────────────────

def test_demanda_real_trace_exists():
    assert _trace(_fig(), "Demanda real") is not None


def test_forecast_trace_exists():
    assert _trace(_fig(), "Pronóstico (Auto)") is not None


def test_trend_trace_exists_when_enabled():
    assert _trace(_fig(trend=True),  "Tendencia (promedio 12 meses)") is not None


def test_trend_trace_absent_when_disabled():
    assert _trace(_fig(trend=False), "Tendencia (promedio 12 meses)") is None


# ── history shape ─────────────────────────────────────────────────────────────

def test_demanda_real_has_30_points():
    tr = _trace(_fig(), "Demanda real")
    assert len(tr.x) == N_HIST
    assert len(tr.y) == N_HIST


def test_demanda_real_total_matches_spec():
    tr = _trace(_fig(), "Demanda real")
    total = float(np.asarray(tr.y, float).sum())
    assert abs(total - HIST_VALS.sum()) < 0.1, f"total {total:.1f} ≠ {HIST_VALS.sum():.1f}"


def test_last_actual_month_label_is_jun_2026():
    tr = _trace(_fig(), "Demanda real")
    last = tr.x[-1]
    assert "Jun" in last and "26" in last, f"last history label is '{last}', expected Jun 26"


# ── forecast shape ────────────────────────────────────────────────────────────

def test_forecast_first_x_bridges_history():
    """Forecast trace starts at the last history label so the line connects."""
    tr = _trace(_fig(), "Pronóstico (Auto)")
    assert "Jun" in tr.x[0] and "26" in tr.x[0], f"bridge point is '{tr.x[0]}'"


def test_forecast_first_full_month_is_jul_2026():
    tr = _trace(_fig(), "Pronóstico (Auto)")
    assert "Jul" in tr.x[1] and "26" in tr.x[1], f"first forecast month is '{tr.x[1]}'"


def test_forecast_bridge_y_equals_last_history():
    """First y value of the forecast trace equals the last history value."""
    tr_hist = _trace(_fig(), "Demanda real")
    tr_fc   = _trace(_fig(), "Pronóstico (Auto)")
    assert abs(float(tr_fc.y[0]) - float(tr_hist.y[-1])) < 1e-6


# ── trend ─────────────────────────────────────────────────────────────────────

def test_trend_confined_to_history_period():
    """Trend trace must not extend into forecast months."""
    fig = _fig()
    tr_trend = _trace(fig, "Tendencia (promedio 12 meses)")
    tr_hist  = _trace(fig, "Demanda real")
    assert len(tr_trend.x) == len(tr_hist.x), (
        f"Trend has {len(tr_trend.x)} points; history has {len(tr_hist.x)}"
    )


def test_trend_x_values_match_history_x_values():
    fig = _fig()
    tr_trend = _trace(fig, "Tendencia (promedio 12 meses)")
    tr_hist  = _trace(fig, "Demanda real")
    assert list(tr_trend.x) == list(tr_hist.x)


def test_trend_last_value_approx_627():
    """12-month MA at Jun 2026 ≈ mean of the last 12 history values ≈ 627 t."""
    tr = _trace(_fig(), "Tendencia (promedio 12 meses)")
    last = float(tr.y[-1])
    assert abs(last - _AVG_LAST12) < 1.0, f"trend at last history point = {last:.1f}, expected ≈ {_AVG_LAST12:.1f}"


# ── legend ────────────────────────────────────────────────────────────────────

def test_all_legend_traces_have_names():
    for tr in _named(_fig()):
        assert tr.name, f"A trace has an empty name"


def test_at_least_three_named_traces():
    assert len(_named(_fig())) >= 3


# ── axes ──────────────────────────────────────────────────────────────────────

def test_x_axis_is_category_type():
    assert _fig().layout.xaxis.type == "category"


def test_x_values_are_strings():
    """All trace x values must be category strings produced by mes()."""
    for tr in _fig().data:
        if tr.x is not None and len(tr.x) > 0:
            assert isinstance(tr.x[0], str), (
                f"Trace '{tr.name}' x[0] is {type(tr.x[0])}, expected str"
            )


def test_categoryarray_starts_ene_24():
    cats = list(_fig().layout.xaxis.categoryarray)
    assert cats[0] == "Ene 24", f"First category is '{cats[0]}', expected 'Ene 24'"


def test_categoryarray_ends_jun_27():
    cats = list(_fig().layout.xaxis.categoryarray)
    assert cats[-1] == "Jun 27", f"Last category is '{cats[-1]}', expected 'Jun 27'"


def test_y_axis_rangemode_tozero():
    """y-axis starts at zero so the area fill is grounded."""
    assert _fig().layout.yaxis.rangemode == "tozero"


def test_y_axis_no_explicit_cap_below_forecast():
    """If an explicit y range is set it must not cut off the forecast max."""
    yrange = _fig().layout.yaxis.range
    if yrange is not None:
        assert yrange[1] >= FC_VALS.max(), (
            f"y max {yrange[1]} would clip forecast peak {FC_VALS.max()}"
        )
