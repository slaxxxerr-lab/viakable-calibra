"""Unit and integration tests for viakable_forecast.color.

Run from project root:  pytest tests/test_color.py -v
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import pytest

from viakable_forecast.color import (
    COLOR_MAP,
    COLOR_MULTICOLOR,
    COLOR_NA,
    add_color_column,
    color_forecast_breakdown,
    color_mix_table,
    color_shares,
    parse_color,
    verify_sums,
)


# ── parse_color unit tests ────────────────────────────────────────────────────

@pytest.mark.parametrize("name,expected", [
    # Descriptive-name style (non-letter boundary)
    ("C TF 16 NE 100MIN TF 16 100MT",           "Negro"),
    ("C    THW  10  NE100CRMIN THW 10 100MT",    "Negro"),
    ("THHN 14 VE 500FT",                         "Verde"),
    ("CABLE AM THW",                             "Amarillo"),
    # Multicolor packs
    ("THW12NEBL50/50",                           COLOR_MULTICOLOR),
    ("THW12NEBLRO3X30",                          COLOR_MULTICOLOR),
    # Product-ID style (digit-preceded code)
    ("CNC000NEM",                                "Negro"),
    ("CNH410BLR",                                "Blanco"),
    ("CNH410NER",                                "Negro"),
    ("CNH410ROR",                                "Rojo"),
    ("CNH410VER",                                "Verde"),
    ("CNY905BLM",                                "Blanco"),
    # Sin color — no color code
    ("A347",                                     COLOR_NA),
    ("DQ29",                                     COLOR_NA),
    ("AZA8",                                     COLOR_NA),      # AZ followed by A — not a code
    ("PSD 6 AL URD",                             COLOR_NA),
    # Edge cases
    ("",                                         COLOR_NA),
    (None,                                       COLOR_NA),
])
def test_parse_color(name, expected):
    assert parse_color(name) == expected


def test_cable_not_multicolor():
    """The word 'CABLE' must NOT trigger multicolor (CA+BL+E: E breaks the sequence)."""
    assert parse_color("CABLE THW 12 NE") == "Negro"
    assert parse_color("CABLES THW 10 BL") == "Blanco"


def test_cable_am_is_amarillo_not_multicolor():
    """'CABLE AM THW' — AM = Amarillo, no multicolor pair here."""
    assert parse_color("CABLE AM THW") == "Amarillo"


def test_multicolor_at_end_of_string():
    """'NEBL' at end of string matches because $ is in the lookahead."""
    result = parse_color("THW10NEBL")
    assert result == COLOR_MULTICOLOR


def test_all_color_codes_recognized():
    """Every code in COLOR_MAP should be recognized in a standard name template."""
    for code, label in COLOR_MAP.items():
        name = f"THW 10 {code} 100MT"
        assert parse_color(name) == label, f"code={code} not parsed as {label}"


# ── add_color_column ──────────────────────────────────────────────────────────

def _make_df():
    return pd.DataFrame({
        "prod_name": ["C TF 16 NE 100MIN", "C TF 16 BL 100MIN", "THW12NEBL50/50",
                      "A347", "CNC000NEM"],
        "qty": [10.0, 20.0, 5.0, 3.0, 7.0],
        "cal": ["16", "16", "12", "10", "10"],
    })


def test_add_color_column_adds_column():
    df, unclassified = add_color_column(_make_df(), prod_col="prod_name")
    assert "color" in df.columns


def test_add_color_column_values():
    df, _ = add_color_column(_make_df(), prod_col="prod_name")
    assert df.iloc[0]["color"] == "Negro"
    assert df.iloc[1]["color"] == "Blanco"
    assert df.iloc[2]["color"] == COLOR_MULTICOLOR
    assert df.iloc[3]["color"] == COLOR_NA


def test_add_color_column_unclassified_list():
    df, unclassified = add_color_column(_make_df(), prod_col="prod_name")
    assert "A347" in unclassified


def test_add_color_column_missing_col():
    df = _make_df().drop(columns=["prod_name"])
    df_out, unclassified = add_color_column(df, prod_col="prod_name")
    assert (df_out["color"] == COLOR_NA).all()
    assert unclassified == []


# ── color_mix_table ───────────────────────────────────────────────────────────

def test_color_mix_table_shares_sum_to_100():
    df, _ = add_color_column(_make_df(), prod_col="prod_name")
    df["prod"] = df["prod_name"]
    tbl = color_mix_table(df, qty_col="qty")
    for cal, grp in tbl.groupby("cal"):
        total = grp["share_pct"].sum()
        assert abs(total - 100.0) < 0.1, f"Cal {cal}: shares sum to {total:.2f} (expected 100)"


def test_color_mix_table_empty_on_no_color_col():
    df = _make_df()
    tbl = color_mix_table(df, qty_col="qty")
    assert tbl.empty


# ── verify_sums ───────────────────────────────────────────────────────────────

def test_verify_sums_zero_diff():
    df, _ = add_color_column(_make_df(), prod_col="prod_name")
    diffs = verify_sums(df, qty_col="qty")
    for cal, diff in diffs.items():
        assert diff == 0.0, f"Cal {cal}: sum diff {diff}"


# ── color_shares ──────────────────────────────────────────────────────────────

def test_color_shares_sum_to_1():
    df, _ = add_color_column(_make_df(), prod_col="prod_name")
    shares = color_shares(df, qty_col="qty")
    totals = {}
    for (cal, color), s in shares.items():
        totals[cal] = totals.get(cal, 0.0) + s
    for cal, total in totals.items():
        assert abs(total - 1.0) < 1e-9, f"Cal {cal}: shares sum to {total}"


def test_color_shares_gauge_filter():
    df, _ = add_color_column(_make_df(), prod_col="prod_name")
    shares_all = color_shares(df, qty_col="qty")
    shares_16  = color_shares(df, gauges=["16"], qty_col="qty")
    # cal 10 and 12 keys should not appear when filtering to 16
    for (cal, _) in shares_16:
        assert cal == "16"
    # all calibres should appear with no filter
    cals_all = {cal for (cal, _) in shares_all}
    assert "10" in cals_all and "12" in cals_all and "16" in cals_all


# ── color_forecast_breakdown ──────────────────────────────────────────────────

def _make_detail(cal, fc_vals):
    return {"label": cal, "fc": fc_vals, "y": [], "error": 0.0, "method": "auto", "demand": "Estable"}


def test_breakdown_reconciliation():
    """Sum of per-color fc_monthly for a calibre == calibre forecast."""
    df, _ = add_color_column(_make_df(), prod_col="prod_name")
    # cal 16 has Negro (10t) and Blanco (20t) → 2/3 and 1/3 shares
    detail = [_make_detail("16", [90.0, 60.0, 30.0])]
    cdf = color_forecast_breakdown(df, detail, selected_gauges=["16"])

    assert not cdf.empty
    for month_idx in range(3):
        total = sum(cdf["fc_monthly"].apply(lambda lst: lst[month_idx]))
        expected = detail[0]["fc"][month_idx]
        assert abs(total - expected) < 1e-9, f"Month {month_idx}: {total} != {expected}"


def test_breakdown_shares_sum_to_100_per_calibre():
    df, _ = add_color_column(_make_df(), prod_col="prod_name")
    detail = [_make_detail("16", [100.0, 100.0, 100.0])]
    cdf = color_forecast_breakdown(df, detail, selected_gauges=["16"])
    total_share = cdf[cdf["cal"] == "16"]["share_pct"].sum()
    assert abs(total_share - 100.0) < 0.1


def test_breakdown_empty_on_zero_forecast():
    """If the calibre forecast is all zeros, no rows are produced."""
    df, _ = add_color_column(_make_df(), prod_col="prod_name")
    detail = [_make_detail("16", [0.0, 0.0, 0.0])]
    cdf = color_forecast_breakdown(df, detail, selected_gauges=["16"])
    assert cdf.empty


def test_breakdown_missing_detail():
    """If a calibre has no entry in detail_list, it is silently skipped."""
    df, _ = add_color_column(_make_df(), prod_col="prod_name")
    detail = [_make_detail("12", [50.0, 60.0])]
    cdf = color_forecast_breakdown(df, detail, selected_gauges=["16"])
    assert cdf.empty


def test_breakdown_multi_calibre():
    df, _ = add_color_column(_make_df(), prod_col="prod_name")
    detail = [
        _make_detail("16", [100.0]),
        _make_detail("12", [200.0]),
    ]
    cdf = color_forecast_breakdown(df, detail, selected_gauges=["16", "12"])
    # Both calibres should produce rows
    assert "16" in cdf["cal"].values
    assert "12" in cdf["cal"].values
    # Check reconciliation for each
    for cal, group in cdf.groupby("cal"):
        fc_sum = group["fc_monthly"].apply(lambda lst: lst[0]).sum()
        expected = next(d["fc"][0] for d in detail if str(d["label"]) == cal)
        assert abs(fc_sum - expected) < 1e-9


# ── integration: real data (skipped unless CSV exists) ───────────────────────

CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "Viakable_Tons_Sales.csv"
)


@pytest.mark.skipif(not os.path.exists(CSV_PATH), reason="Real data file not found")
def test_real_data_color_column():
    from viakable_forecast import data as D

    raw = D.load_file(CSV_PATH)
    prep = D.prepare(raw, D.detect_columns(raw))
    m = prep.monthly
    assert "color" in m.columns, "color column missing after prepare()"


@pytest.mark.skipif(not os.path.exists(CSV_PATH), reason="Real data file not found")
def test_real_data_verify_sums_all_ok():
    from viakable_forecast import data as D

    raw = D.load_file(CSV_PATH)
    prep = D.prepare(raw, D.detect_columns(raw))
    m = prep.monthly
    diffs = verify_sums(m, qty_col="qty")
    for cal, diff in diffs.items():
        assert diff == 0.0, f"Cal {cal}: color sum differs from calibre total by {diff}"


@pytest.mark.skipif(not os.path.exists(CSV_PATH), reason="Real data file not found")
def test_real_data_cal6_all_sin_color():
    """Calibre 6 (aluminum PSD/URD) must be 100% Sin color / N/A."""
    from viakable_forecast import data as D

    raw = D.load_file(CSV_PATH)
    prep = D.prepare(raw, D.detect_columns(raw))
    m = prep.monthly
    cal6 = m[m["cal"] == "6"]
    assert (cal6["color"] == COLOR_NA).all(), "Cal 6 has unexpected colored products"


@pytest.mark.skipif(not os.path.exists(CSV_PATH), reason="Real data file not found")
def test_real_data_major_colors_present():
    """Negro, Blanco, Rojo, Verde must appear in calibres 10–18."""
    from viakable_forecast import data as D

    raw = D.load_file(CSV_PATH)
    prep = D.prepare(raw, D.detect_columns(raw))
    m = prep.monthly
    high_cal = m[m["cal"].isin(["10", "12", "14", "16", "18"])]
    colors_found = set(high_cal["color"].unique())
    for expected in ("Negro", "Blanco", "Rojo", "Verde"):
        assert expected in colors_found, f"{expected} missing from high-calibre data"
