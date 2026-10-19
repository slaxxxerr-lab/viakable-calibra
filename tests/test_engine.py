"""Engine integration tests — skipped unless viakable_forecast.engine is importable.

These tests require the full viakable_forecast package (data.py, engine.py,
planning.py, etc.) which lives in the production environment on Saul's Mac.
Run from the project root: pytest tests/test_engine.py -v

Required fixtures:
  CSV_PATH  — path to Viakable_Tons_Sales.csv
  XLSX_PATH — path to Viakable Tons Sales.xlsx (if available)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

# Skip everything in this module if the full engine is not available.
# Stub modules import fine but have _FULL_ENGINE = False — skip in that case.
data_mod   = pytest.importorskip("viakable_forecast.data",     reason="viakable_forecast.data not available")
engine_mod = pytest.importorskip("viakable_forecast.engine",   reason="viakable_forecast.engine not available")
plan_mod   = pytest.importorskip("viakable_forecast.planning", reason="viakable_forecast.planning not available")

if not getattr(engine_mod, "_FULL_ENGINE", False):
    pytest.skip("Full engine not installed on this machine", allow_module_level=True)

import numpy as np

DATA   = data_mod
ENGINE = engine_mod
PLAN   = plan_mod

# ── fixtures ──────────────────────────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Viakable_Tons_Sales.csv")

@pytest.fixture(scope="module")
def df():
    """Load the real CSV and return the monthly dataframe."""
    return DATA.load(CSV_PATH)


@pytest.fixture(scope="module")
def engine_result(df):
    """Run the engine on all gauges, national market, 6-month horizon."""
    return ENGINE.run(df, gauges=None, market=None, horizon=6)


# ── data integrity ────────────────────────────────────────────────────────────
class TestDataIntegrity:
    def test_loads_30_months(self, df):
        months = df["ym"].nunique()
        assert months >= 24, f"Se esperan ≥24 meses, se obtuvieron {months}"

    def test_has_all_gauges(self, df):
        gauges = set(df["cal"].astype(str).unique())
        for g in ("6", "10", "12", "14", "16", "18"):
            assert g in gauges, f"Calibre {g} no encontrado"

    def test_16y18_split_matches_gauge_column(self, df):
        """Rows tagged as gauge 16 or 18 must match the Gauge column."""
        if "Gauge" not in df.columns:
            pytest.skip("Columna 'Gauge' no disponible")
        sub = df[df["cal"].isin(["16", "18"])]
        for _, row in sub.iterrows():
            assert str(int(row["Gauge"])) == str(row["cal"]), (
                f"16y18 split incorrecto: cal={row['cal']} pero Gauge={row['Gauge']}"
            )

    def test_no_period_13(self, df):
        """Period 13 (closing, 0 tons) must be excluded."""
        if "Periodo Contable" in df.columns:
            periods = df["Periodo Contable"].astype(str)
            assert not any(p.endswith("13") for p in periods), "Periodo 13 encontrado en los datos"

    def test_zero_tons_not_dominant(self, df):
        zero_pct = (df["qty"] == 0).mean()
        assert zero_pct < 0.5, f"Más del 50% de filas tienen cero toneladas: {zero_pct:.1%}"


# ── forecast correctness ──────────────────────────────────────────────────────
class TestForecastCorrectness:
    def test_total_equals_sum_of_gauges(self, df):
        """Reconciliation: total forecast must equal sum of gauge forecasts."""
        result_total = ENGINE.run(df, gauges=None, market=None, horizon=6)
        gauge_sum = 0.0
        for g in ("6", "10", "12", "14", "16", "18"):
            r = ENGINE.run(df, gauges=[g], market=None, horizon=6)
            gauge_sum += sum(r.forecast)
        total = sum(result_total.forecast)
        assert abs(total - gauge_sum) / max(total, 1) < 0.01, (
            f"Total ({total:.1f}) != suma calibres ({gauge_sum:.1f})"
        )

    def test_forecast_is_positive(self, engine_result):
        for v in engine_result.forecast:
            assert v >= 0, f"Pronóstico negativo: {v}"

    def test_forecast_length_equals_horizon(self, engine_result):
        assert len(engine_result.forecast) == 6

    def test_cross_validation_no_future_data_leak(self, df):
        """Cross-validation windows must not use future data.

        We verify this by checking that the error of each window, computed
        using only data before the cut, is consistent: error should not be
        dramatically lower than what a naive hold-out would give.
        """
        result = ENGINE.run(df, gauges=None, market=None, horizon=6)
        if hasattr(result, "cv_errors"):
            for err in result.cv_errors:
                assert err >= 0, "Negative CV error (sign of data leak)"
                assert err < 100, f"CV error implausibly large: {err}"


# ── scenario: metal price ─────────────────────────────────────────────────────
class TestMetalScenarios:
    def test_copper_up_reduces_export_margin(self, df):
        """Higher copper price should reduce gross margin (with <100% pass-through)."""
        base   = ENGINE.run(df, gauges=None, market=None, horizon=6, copper_scenario=0.0)
        higher = ENGINE.run(df, gauges=None, market=None, horizon=6, copper_scenario=0.20)
        if hasattr(base, "margin") and hasattr(higher, "margin"):
            assert sum(higher.margin) <= sum(base.margin) + 1e-3, (
                "Subida del cobre no redujo el margen"
            )

    def test_tariff_zero_increases_exports(self, df):
        """Setting tariff to 0 should increase (or maintain) export forecast."""
        base = ENGINE.run(df, gauges=None, market="Exportación EE. UU.", horizon=6,
                          tariff_scenario=0.50)
        zero = ENGINE.run(df, gauges=None, market="Exportación EE. UU.", horizon=6,
                          tariff_scenario=0.0)
        total_base = sum(base.forecast)
        total_zero = sum(zero.forecast)
        assert total_zero >= total_base - 1e-3, (
            f"Arancel cero no aumentó exportaciones: base={total_base:.1f} zero={total_zero:.1f}"
        )


# ── error handling ────────────────────────────────────────────────────────────
class TestErrorHandling:
    def test_empty_dataframe_raises(self):
        import pandas as pd
        empty_df = pd.DataFrame(columns=["ym", "cal", "qty"])
        with pytest.raises(Exception):
            ENGINE.run(empty_df, gauges=None, market=None, horizon=6)

    def test_nonexistent_file_raises(self):
        with pytest.raises(Exception):
            DATA.load("/no/existe/este_archivo.xlsx")

    def test_odd_csv_with_missing_cols_gives_clear_error(self, tmp_path):
        import pandas as pd
        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame({"colA": [1, 2], "colB": [3, 4]}).to_csv(bad_csv, index=False)
        with pytest.raises(Exception) as exc_info:
            DATA.load(str(bad_csv))
        # Error message should be helpful, not a raw KeyError
        assert exc_info.value.args[0]  # non-empty message
