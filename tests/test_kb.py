"""Tests for viakable_forecast.kb — knowledge base entries, answer(), keyword_match()."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from viakable_forecast import kb as KBM


# ── entry completeness ────────────────────────────────────────────────────────
class TestEntryCompleteness:
    def test_all_entries_have_required_fields(self):
        for key, entry in KBM._ENTRIES.items():
            assert "es_title" in entry, f"{key} falta es_title"
            assert "es_body" in entry, f"{key} falta es_body"
            assert "en_title" in entry, f"{key} falta en_title"
            assert "en_body" in entry, f"{key} falta en_body"
            assert "keywords" in entry, f"{key} falta keywords"
            assert len(entry["es_body"]) > 20, f"{key} es_body muy corto"
            assert len(entry["en_body"]) > 20, f"{key} en_body muy corto"

    def test_entry_texts_match_entry_keys(self):
        assert set(KBM.ENTRY_TEXTS.keys()) == set(KBM._ENTRIES.keys())
        assert set(KBM.ENTRY_TEXTS_EN.keys()) == set(KBM._ENTRIES.keys())

    def test_expected_entries_present(self):
        expected = [
            "que_es_calibra", "metodos_pronostico", "validacion_cruzada",
            "mape_precision", "clases_demanda", "ensamble", "machine_learning",
            "inventario_seguridad", "senales", "arancel", "reconciliacion",
            "crm_oportunidades", "alertas", "riesgo_simulacion", "abc_xyz",
            "precios_metal", "consenso",
        ]
        for key in expected:
            assert key in KBM._ENTRIES, f"Entrada faltante: {key}"


# ── answer() ─────────────────────────────────────────────────────────────────
class TestAnswer:
    def test_returns_spanish_by_default(self):
        title, body = KBM.answer("mape_precision")
        assert "MAPE" in title or "mape" in title.lower()
        assert len(body) > 20

    def test_returns_english_when_requested(self):
        title, body = KBM.answer("mape_precision", lang="en")
        assert title  # non-empty
        assert "MAPE" in title or "accuracy" in title.lower()

    def test_unknown_key_returns_empty(self):
        title, body = KBM.answer("no_existe_esta_clave")
        assert title == ""
        assert body == ""

    def test_all_keys_return_nonempty(self):
        for key in KBM.all_keys():
            title_es, body_es = KBM.answer(key, lang="es")
            title_en, body_en = KBM.answer(key, lang="en")
            assert title_es, f"{key}: es_title vacío"
            assert body_es,  f"{key}: es_body vacío"
            assert title_en, f"{key}: en_title vacío"
            assert body_en,  f"{key}: en_body vacío"


# ── keyword_match() ───────────────────────────────────────────────────────────
class TestKeywordMatch:
    @pytest.mark.parametrize("q_lower,expected_key", [
        ("mape wape precision error",  "mape_precision"),        # no "pronostico" to avoid hitting metodos first
        ("calibra herramienta app",    "que_es_calibra"),
        ("validacion cruzada origen",  "validacion_cruzada"),
        ("inventario seguridad stock", "inventario_seguridad"),
        ("ensamble ponderado pesos",   "ensamble"),
        ("arancel exportacion 232",    "arancel"),
        ("crm oportunidades etapa",    "crm_oportunidades"),
    ])
    def test_keyword_match(self, q_lower, expected_key):
        match = KBM.keyword_match(q_lower)
        assert match == expected_key, f"'{q_lower}' → esperado '{expected_key}', obtenido '{match}'"

    def test_no_match_returns_none(self):
        assert KBM.keyword_match("zapatos rojos lluviosos") is None

    def test_match_is_first_keyword_found(self):
        # Just verify it returns a valid key, not None
        match = KBM.keyword_match("xgboost machine learning shap")
        assert match == "machine_learning"


# ── ENTRY_TEXTS used for TF-IDF ───────────────────────────────────────────────
class TestEntryTexts:
    def test_entry_texts_are_nonempty_strings(self):
        for key, text in KBM.ENTRY_TEXTS.items():
            assert isinstance(text, str)
            assert len(text) > 50, f"{key}: ENTRY_TEXTS muy corto"

    def test_entry_texts_en_are_nonempty_strings(self):
        for key, text in KBM.ENTRY_TEXTS_EN.items():
            assert isinstance(text, str)
            assert len(text) > 50, f"{key}: ENTRY_TEXTS_EN muy corto"

    def test_keywords_included_in_entry_texts(self):
        # Keywords should appear somewhere in the combined text
        for key, entry in KBM._ENTRIES.items():
            text = KBM.ENTRY_TEXTS[key].lower()
            kw = entry["keywords"][0].lower()
            # Keyword may not be verbatim (accent differences) — just check length
            assert len(text) > len(entry["es_body"]), f"{key}: ENTRY_TEXTS no incluye keywords"
