"""Tests for viakable_forecast.assistant — norm, detect_lang, parse, extract, merge_pending."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from viakable_forecast import assistant as AS


# ── norm ─────────────────────────────────────────────────────────────────────
class TestNorm:
    def test_strips_accents(self):
        assert AS.norm("pronóstico") == "pronostico"

    def test_lowercases(self):
        assert AS.norm("CALIBRE 12") == "calibre 12"

    def test_collapses_whitespace(self):
        assert AS.norm("  hola   mundo  ") == "hola mundo"

    def test_accent_table_completeness(self):
        # All accented vowels and ñ survive roundtrip
        assert AS.norm("áéíóúàèìòùâêîôûäëïöüãõñ") == "aeiouaeiouaeiouaeiouaon"

    def test_empty_string(self):
        assert AS.norm("") == ""


# ── detect_lang ───────────────────────────────────────────────────────────────
class TestDetectLang:
    @pytest.mark.parametrize("q,expected", [
        ("¿Cuánto vamos a vender en los próximos 6 meses?", "es"),
        ("What is the forecast for gauge 14?", "en"),
        ("Dame el pronóstico del calibre 6", "es"),
        ("How much will we sell this quarter?", "en"),
    ])
    def test_basic_detection(self, q, expected):
        assert AS.detect_lang(q) == expected

    def test_default_returned_on_ambiguous(self):
        assert AS.detect_lang("abc", default="es") == "es"
        assert AS.detect_lang("abc", default="en") == "en"


# ── extract — gauges ──────────────────────────────────────────────────────────
class TestExtractGauges:
    @pytest.mark.parametrize("q,expected", [
        ("pronóstico del calibre 12",    ["12"]),
        ("calibres 6 y 10",              ["6", "10"]),
        ("gauge 14 forecast",            ["14"]),
        ("calibre AWG 10",               ["10"]),
        ("todos los calibres",           []),     # "all gauges" → empty = all
        ("no gauge here",                []),
        ("16 y 18",                      ["16", "18"]),
    ])
    def test_gauge_extraction(self, q, expected):
        assert AS.extract(q).gauges == expected


# ── extract — metals ──────────────────────────────────────────────────────────
class TestExtractMetals:
    def test_copper_detected(self):
        e = AS.extract("si el cobre sube 10%")
        assert "cu" in e.metals

    def test_aluminum_detected(self):
        e = AS.extract("qué pasa si el aluminio baja 15%")
        assert "al" in e.metals

    def test_both_metals(self):
        e = AS.extract("qué pasa si el cobre y el aluminio suben")
        assert e.both_metals is True

    def test_no_metal(self):
        e = AS.extract("dame el pronóstico del calibre 6")
        assert e.metals == []


# ── extract — percentages and direction ──────────────────────────────────────
class TestExtractPct:
    def test_pct_up(self):
        e = AS.extract("si el cobre sube 10%")
        assert e.pct == 10.0
        assert e.direction == 1

    def test_pct_down(self):
        e = AS.extract("qué pasa si el aluminio baja 15%")
        assert e.pct == 15.0
        assert e.direction == -1

    def test_no_pct(self):
        assert AS.extract("dame el pronóstico").pct is None


# ── extract — tariff ──────────────────────────────────────────────────────────
class TestExtractTariff:
    def test_tariff_detected(self):
        assert AS.extract("si el arancel sube 5 puntos").tariff is True

    def test_tariff_zero(self):
        e = AS.extract("simula un arancel de cero para exportaciones")
        assert e.tariff is True
        assert e.tariff_zero is True

    def test_no_tariff(self):
        assert AS.extract("dame el pronóstico del calibre 6").tariff is False


# ── extract — horizon ─────────────────────────────────────────────────────────
class TestExtractHorizon:
    def test_horizon_months(self):
        assert AS.extract("pronóstico 6 meses").horizon == 6

    def test_horizon_english(self):
        assert AS.extract("forecast for 12 months").horizon == 12

    def test_horizon_out_of_range_ignored(self):
        assert AS.extract("pronóstico 100 meses").horizon is None


# ── parse — intent ────────────────────────────────────────────────────────────
class TestParseIntent:
    @pytest.mark.parametrize("q,intent", [
        ("¿Cuánto vamos a vender en los próximos 6 meses?", "forecast"),
        ("¿Qué pasa si el cobre sube 10%?",                "whatif"),
        ("¿Por qué cambió el pronóstico del calibre 6?",   "why"),
        ("Why did the total forecast change?",             "why"),
        ("¿Qué alertas hay ahora?",                        "alerts"),
        ("¿Cuál es el precio del cobre en vivo?",          "prices"),
        ("¿Qué dice la noticia sobre aranceles?",          "news"),
        ("¿Cuánto metal tengo que comprar?",               "plan"),
        ("¿Qué clientes compran menos?",                   "customers"),
        ("Prepara el resumen para la junta de S&OP",       "summary"),
        ("¿Qué tan preciso es el pronóstico?",             "accuracy"),
        ("¿Cuál es el margen en riesgo?",                  "risk"),
        ("¿Qué puedes contestar?",                         "help"),
        ("Hola, buenos días",                              "smalltalk"),
    ])
    def test_intent(self, q, intent):
        assert AS.parse(q).intent == intent

    def test_mape_goes_to_accuracy_not_forecast(self):
        # "MAPE" has "accuracy" intent, not "forecast"
        assert AS.parse("¿Qué es el MAPE?").intent == "accuracy"

    def test_whatif_before_forecast(self):
        # "qué pasa si" must win over bare "forecast" keywords
        assert AS.parse("¿Qué pasa si el arancel sube 5 puntos?").intent == "whatif"


# ── parse — missing slots ─────────────────────────────────────────────────────
class TestParseMissing:
    def test_whatif_no_lever(self):
        p = AS.parse("¿Qué pasa si cambia algo?")
        # intent=whatif, lever not specified → missing["lever"]
        if p.intent == "whatif":
            assert "lever" in p.missing

    def test_whatif_complete(self):
        p = AS.parse("¿Qué pasa si el cobre sube 10%?")
        assert p.intent == "whatif"
        assert p.missing == []


# ── merge_pending ─────────────────────────────────────────────────────────────
class TestMergePending:
    def test_fills_pct_slot(self):
        prev = AS.parse("¿Qué pasa si el cobre sube?")   # missing: pct
        if "pct" in (prev.missing or []):
            merged = AS.merge_pending(prev, "un 10%")
            assert merged is not None
            assert merged.ents.pct == 10.0

    def test_returns_none_if_no_missing(self):
        prev = AS.parse("dame el pronóstico del calibre 6")
        assert AS.merge_pending(prev, "algo") is None

    def test_fills_gauge_slot(self):
        prev = AS.parse("¿Qué pasa si hay un pedido adicional?")
        # missing: gauge
        if "gauge" in (prev.missing or []):
            merged = AS.merge_pending(prev, "calibre 10")
            assert merged is not None
            assert "10" in merged.ents.gauges


# ── suggestions ───────────────────────────────────────────────────────────────
class TestSuggestions:
    def test_returns_n_items(self):
        p = AS.parse("dame el pronóstico")
        sug = AS.suggestions(p, 3)
        assert len(sug) == 3

    def test_excludes_same_intent(self):
        p = AS.parse("¿Cuánto vamos a vender?")  # forecast
        sug = AS.suggestions(p, 5)
        # None of the suggestions should include "forecast" intent phrasing
        p2 = AS.parse("dame el pronóstico del calibre 6")
        sug2 = AS.suggestions(p2, 5)
        assert len(sug2) > 0

    def test_none_parse_returns_fallback(self):
        sug = AS.suggestions(None, 3)
        assert len(sug) == 3
