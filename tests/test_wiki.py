"""Tests for viakable_forecast.wiki — lookup() and short_answer() with mocked requests.

wiki.lookup() makes TWO requests per language:
  1. _search() — action=query&list=search  → returns {"query": {"search": [{"title": "..."}]}}
  2. _extract() — action=query&prop=extracts — returns {"query": {"pages": {...}}}
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from unittest.mock import patch, MagicMock, call
from viakable_forecast import wiki as WIKI


# ── mock builder helpers ──────────────────────────────────────────────────────
def _search_resp(title: str):
    """Mock response for the MediaWiki search API."""
    m = MagicMock()
    m.status_code = 200
    m.raise_for_status = MagicMock()
    m.json.return_value = {"query": {"search": [{"title": title}]}}
    return m


def _no_search_resp():
    """Mock response for search with zero hits."""
    m = MagicMock()
    m.status_code = 200
    m.raise_for_status = MagicMock()
    m.json.return_value = {"query": {"search": []}}
    return m


def _extract_resp(title: str, text: str, url: str = "https://es.wikipedia.org/wiki/Cobre"):
    """Mock response for the MediaWiki extract API."""
    m = MagicMock()
    m.status_code = 200
    m.raise_for_status = MagicMock()
    m.json.return_value = {"query": {"pages": {"12345": {
        "pageid": 12345,
        "title": title,
        "extract": text,
        "fullurl": url,
    }}}}
    return m


COPPER_EXTRACT = (
    "El cobre (símbolo Cu) es un elemento químico metálico de número atómico 29. "
    "Es un metal de transición blando, maleable y dúctil. "
    "Se usa ampliamente en cables eléctricos por su alta conductividad."
)

NEARSHORING_EXTRACT_EN = (
    "Nearshoring is the practice of transferring a business operation to a nearby country. "
    "It is often contrasted with offshoring, which involves more distant locations."
)


# ── lookup() — success path ───────────────────────────────────────────────────
class TestLookup:
    def test_returns_dict_on_success(self):
        # Two requests: search then extract
        responses = [
            _search_resp("Cobre"),
            _extract_resp("Cobre", COPPER_EXTRACT),
        ]
        with patch("viakable_forecast.wiki.requests.get", side_effect=responses):
            result = WIKI.lookup("cobre", "es")
        assert result is not None
        assert result["title"] == "Cobre"
        assert len(result["extract"]) > 20
        assert result["lang"] == "es"
        assert "url" in result

    def test_returns_none_when_search_empty(self):
        # Search returns no hits → no extract call needed
        with patch("viakable_forecast.wiki.requests.get", return_value=_no_search_resp()):
            result = WIKI.lookup("xyzzzzznonexistent", "es")
        assert result is None

    def test_english_fallback_when_spanish_empty(self):
        """Spanish search finds nothing; English finds a page."""
        responses = [
            _no_search_resp(),                                                    # es search → miss
            _search_resp("Nearshoring"),                                          # en search → hit
            _extract_resp("Nearshoring", NEARSHORING_EXTRACT_EN,
                          "https://en.wikipedia.org/wiki/Nearshoring"),           # en extract
        ]
        with patch("viakable_forecast.wiki.requests.get", side_effect=responses):
            result = WIKI.lookup("nearshoring", "es")
        assert result is not None
        assert result["lang"] == "en"

    def test_returns_none_on_network_error(self):
        with patch("viakable_forecast.wiki.requests.get", side_effect=Exception("timeout")):
            result = WIKI.lookup("cobre", "es")
        assert result is None

    def test_url_field_present(self):
        responses = [_search_resp("Cobre"), _extract_resp("Cobre", COPPER_EXTRACT)]
        with patch("viakable_forecast.wiki.requests.get", side_effect=responses):
            result = WIKI.lookup("cobre", "es")
        assert result["url"].startswith("http")


# ── short_answer() ────────────────────────────────────────────────────────────
class TestShortAnswer:
    COPPER_TEXT = (
        "El cobre es un elemento químico con símbolo Cu y número atómico 29. "
        "Es un metal de transición muy utilizado en la fabricación de cables eléctricos. "
        "La conductividad eléctrica del cobre es muy alta. "
        "El nombre proviene del latín cuprum."
    )

    def test_returns_nonempty_string(self):
        ans = WIKI.short_answer(self.COPPER_TEXT, "¿Qué es el cobre?")
        assert isinstance(ans, str)
        assert len(ans) > 10

    def test_returns_relevant_sentence(self):
        ans = WIKI.short_answer(self.COPPER_TEXT, "¿Qué es el cobre?")
        assert "cobre" in ans.lower() or "Cu" in ans

    def test_focus_term_boosts_relevance(self):
        ans = WIKI.short_answer(self.COPPER_TEXT, "¿Qué es?", focus="cables eléctricos")
        assert "cable" in ans.lower()

    def test_empty_text_returns_empty(self):
        assert WIKI.short_answer("", "pregunta") == ""

    def test_single_sentence_text(self):
        text = "El cobre es un metal conductor."
        ans = WIKI.short_answer(text, "¿Qué es el cobre?")
        assert "cobre" in ans.lower()

    def test_does_not_return_all_sentences(self):
        long_text = " ".join([f"Oración número {i} sobre el cobre." for i in range(50)])
        ans = WIKI.short_answer(long_text, "¿Qué es el cobre?")
        assert len(ans) < len(long_text)


# ── lookup() result structure ─────────────────────────────────────────────────
class TestLookupStructure:
    def test_result_has_all_expected_keys(self):
        responses = [_search_resp("Cobre"), _extract_resp("Cobre", COPPER_EXTRACT)]
        with patch("viakable_forecast.wiki.requests.get", side_effect=responses):
            result = WIKI.lookup("cobre", "es")
        for key in ("title", "extract", "lead", "url", "lang"):
            assert key in result, f"Falta clave '{key}' en resultado de lookup()"

    def test_lead_is_first_part_of_extract(self):
        responses = [_search_resp("Cobre"), _extract_resp("Cobre", COPPER_EXTRACT)]
        with patch("viakable_forecast.wiki.requests.get", side_effect=responses):
            result = WIKI.lookup("cobre", "es")
        assert result["extract"].startswith(result["lead"][:30])
