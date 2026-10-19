"""Tests for viakable_forecast.llm — providers(), chat(), grounding helpers, fallbacks."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from unittest.mock import patch, MagicMock
from viakable_forecast import llm as LLM


# ── fixtures ──────────────────────────────────────────────────────────────────
def _no_keys(section, key):
    return None

def _gemini_only(section, key):
    if section == "gemini" and key == "api_key":
        return "test-key-gemini"
    return None

def _groq_only(section, key):
    if section == "groq" and key == "api_key":
        return "test-key-groq"
    return None

def _all_keys(section, key):
    if key == "api_key":
        return f"test-key-{section}"
    return None


def _mock_gemini_ok(text="Hola"):
    """Return a mock that simulates a successful Gemini response."""
    mock = MagicMock()
    mock.ok = True
    mock.status_code = 200
    mock.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": text}]}}]
    }
    return mock


def _mock_gemini_rate_limit():
    mock = MagicMock()
    mock.ok = False
    mock.status_code = 429
    return mock


def _mock_gemini_unavailable():
    mock = MagicMock()
    mock.ok = False
    mock.status_code = 503
    return mock


def _mock_groq_ok(text="Hi there"):
    mock = MagicMock()
    mock.ok = True
    mock.status_code = 200
    mock.json.return_value = {
        "choices": [{"message": {"content": text}}]
    }
    return mock


# ── providers() ──────────────────────────────────────────────────────────────
class TestProviders:
    def test_returns_three_providers(self):
        with patch("viakable_forecast.llm.requests.get", side_effect=Exception("no Ollama")):
            pvs = LLM.providers(_no_keys)
        labels = [p.label for p in pvs]
        assert "Ollama" in labels
        assert "Gemini" in labels
        assert "Groq" in labels

    def test_gemini_ok_when_key_present(self):
        with patch("viakable_forecast.llm.requests.get", side_effect=Exception("no Ollama")):
            pvs = LLM.providers(_gemini_only)
        gem = next(p for p in pvs if p.label == "Gemini")
        assert gem.ok is True
        assert gem.model is not None

    def test_gemini_not_ok_without_key(self):
        with patch("viakable_forecast.llm.requests.get", side_effect=Exception("no Ollama")):
            pvs = LLM.providers(_no_keys)
        gem = next(p for p in pvs if p.label == "Gemini")
        assert gem.ok is False

    def test_groq_ok_when_key_present(self):
        with patch("viakable_forecast.llm.requests.get", side_effect=Exception("no Ollama")):
            pvs = LLM.providers(_groq_only)
        groq = next(p for p in pvs if p.label == "Groq")
        assert groq.ok is True

    def test_groq_not_ok_without_key(self):
        with patch("viakable_forecast.llm.requests.get", side_effect=Exception("no Ollama")):
            pvs = LLM.providers(_no_keys)
        groq = next(p for p in pvs if p.label == "Groq")
        assert groq.ok is False

    def test_ollama_not_ok_when_unreachable(self):
        with patch("viakable_forecast.llm.requests.get", side_effect=Exception("connection refused")):
            pvs = LLM.providers(_no_keys)
        ollama = next(p for p in pvs if p.label == "Ollama")
        assert ollama.ok is False

    def test_ollama_is_local(self):
        with patch("viakable_forecast.llm.requests.get", side_effect=Exception("no Ollama")):
            pvs = LLM.providers(_no_keys)
        ollama = next(p for p in pvs if p.label == "Ollama")
        assert ollama.local is True

    def test_cloud_providers_are_not_local(self):
        with patch("viakable_forecast.llm.requests.get", side_effect=Exception("no Ollama")):
            pvs = LLM.providers(_no_keys)
        for p in pvs:
            if p.label in ("Gemini", "Groq"):
                assert p.local is False


# ── any_available() ───────────────────────────────────────────────────────────
class TestAnyAvailable:
    def test_false_when_no_keys_no_ollama(self):
        with patch("viakable_forecast.llm.requests.get", side_effect=Exception("no Ollama")):
            assert LLM.any_available(_no_keys) is False

    def test_true_when_gemini_key_present(self):
        with patch("viakable_forecast.llm.requests.get", side_effect=Exception("no Ollama")):
            assert LLM.any_available(_gemini_only) is True


# ── chat() — LLMUnavailable when no provider ─────────────────────────────────
class TestChatUnavailable:
    def test_raises_when_no_provider(self):
        with patch("viakable_forecast.llm.requests.get", side_effect=Exception("no Ollama")):
            with pytest.raises(LLM.LLMUnavailable):
                LLM.chat("system", [{"role": "user", "content": "hola"}], _no_keys)

    def test_raises_when_all_denied(self):
        with patch("viakable_forecast.llm.requests.get", side_effect=Exception("no Ollama")):
            with pytest.raises(LLM.LLMUnavailable):
                LLM.chat("system", [{"role": "user", "content": "hola"}],
                          _gemini_only, allow=("groq",))  # gemini key present but groq not allowed


# ── chat() — Gemini success ───────────────────────────────────────────────────
class TestChatGeminiSuccess:
    def test_returns_dict_with_expected_keys(self):
        with patch("viakable_forecast.llm.requests.get", side_effect=Exception("no Ollama")):
            with patch("viakable_forecast.llm.requests.post", return_value=_mock_gemini_ok("Buenos días")):
                result = LLM.chat("system", [{"role": "user", "content": "hola"}], _gemini_only)
        assert "text" in result
        assert "provider" in result
        assert "model" in result
        assert "local" in result

    def test_gemini_result_has_correct_provider(self):
        with patch("viakable_forecast.llm.requests.get", side_effect=Exception("no Ollama")):
            with patch("viakable_forecast.llm.requests.post", return_value=_mock_gemini_ok("Hola")):
                result = LLM.chat("system", [{"role": "user", "content": "hola"}], _gemini_only)
        assert result["provider"] == "Gemini"
        assert result["local"] is False

    def test_gemini_result_text_correct(self):
        with patch("viakable_forecast.llm.requests.get", side_effect=Exception("no Ollama")):
            with patch("viakable_forecast.llm.requests.post", return_value=_mock_gemini_ok("Respuesta de prueba")):
                result = LLM.chat("system", [{"role": "user", "content": "hola"}], _gemini_only)
        assert result["text"] == "Respuesta de prueba"


# ── chat() — retry on 503/429 ─────────────────────────────────────────────────
class TestChatRetry:
    def test_retries_on_503_then_succeeds(self):
        responses = [_mock_gemini_unavailable(), _mock_gemini_unavailable(), _mock_gemini_ok("OK")]
        with patch("viakable_forecast.llm.requests.get", side_effect=Exception("no Ollama")):
            with patch("viakable_forecast.llm.requests.post", side_effect=responses):
                with patch("viakable_forecast.llm.time.sleep"):  # skip real sleep
                    result = LLM.chat("system", [{"role": "user", "content": "hola"}], _gemini_only)
        assert result["text"] == "OK"

    def test_raises_after_all_retries_fail(self):
        with patch("viakable_forecast.llm.requests.get", side_effect=Exception("no Ollama")):
            with patch("viakable_forecast.llm.requests.post", return_value=_mock_gemini_unavailable()):
                with patch("viakable_forecast.llm.time.sleep"):
                    with pytest.raises(LLM.LLMUnavailable):
                        LLM.chat("system", [{"role": "user", "content": "hola"}], _gemini_only)


# ── chat() — Groq fallback ────────────────────────────────────────────────────
class TestChatGroqFallback:
    def test_falls_back_to_groq_when_gemini_fails(self):
        """Gemini key set but keeps failing → should try Groq."""
        with patch("viakable_forecast.llm.requests.get", side_effect=Exception("no Ollama")):
            with patch("viakable_forecast.llm.requests.post") as mock_post:
                # All Gemini attempts fail, but Groq succeeds
                mock_post.side_effect = [
                    _mock_gemini_unavailable(),
                    _mock_gemini_unavailable(),
                    _mock_gemini_unavailable(),
                    _mock_groq_ok("Groq response"),
                ]
                with patch("viakable_forecast.llm.time.sleep"):
                    result = LLM.chat("system", [{"role": "user", "content": "hola"}], _all_keys)
        assert result["provider"] == "Groq"


# ── grounding: numbers_ok() ───────────────────────────────────────────────────
class TestNumbersOk:
    def test_ok_when_numbers_match(self):
        facts = "El pronóstico es 1234 toneladas con 95% de precisión."
        ai = "El pronóstico es de 1234 toneladas con precisión del 95%."
        assert LLM.numbers_ok(ai, facts) is True

    def test_fail_when_ai_invents_number(self):
        facts = "El pronóstico es 1234 toneladas."
        ai = "El pronóstico es 5678 toneladas."   # 5678 not in facts
        assert LLM.numbers_ok(ai, facts) is False

    def test_ok_with_no_numbers(self):
        assert LLM.numbers_ok("El mercado está activo.", "Análisis general.") is True

    def test_ok_when_ai_uses_subset_of_facts_numbers(self):
        facts = "Pronóstico: 1000 toneladas, margen: 12%, error: 8%."
        ai = "El margen es del 12%."   # only uses one number from facts
        assert LLM.numbers_ok(ai, facts) is True

    def test_fail_on_invented_percentage(self):
        facts = "Precisión del 92%."
        ai = "La precisión es del 75%."   # 75 not in facts
        assert LLM.numbers_ok(ai, facts) is False


# ── grounding: strip_business_numbers() ──────────────────────────────────────
class TestStripBusinessNumbers:
    def test_removes_sentence_with_mxn(self):
        text = "El análisis es útil. El costo es 5,000 MXN por tonelada. Considéralo en tu decisión."
        result = LLM.strip_business_numbers(text)
        assert "MXN" not in result
        assert "análisis" in result

    def test_removes_sentence_with_usd(self):
        text = "Good market conditions. Revenue was $2,500 USD last month. Focus on quality."
        result = LLM.strip_business_numbers(text)
        assert "USD" not in result or "$2,500" not in result
        assert "quality" in result

    def test_removes_sentence_with_toneladas(self):
        text = "La demanda creció. Se vendieron 320 toneladas en Q1. El mercado es sólido."
        result = LLM.strip_business_numbers(text)
        assert "toneladas" not in result
        assert "demanda" in result

    def test_removes_sentence_with_tons_abbreviation(self):
        text = "Strong demand expected. We sold 45t in the last period. Growth continues."
        result = LLM.strip_business_numbers(text)
        assert "45t" not in result
        assert "Growth" in result

    def test_clean_text_unchanged(self):
        text = "El mercado eléctrico es dinámico. La competencia es intensa. Vale la pena diversificar."
        result = LLM.strip_business_numbers(text)
        assert result == text

    def test_empty_input(self):
        assert LLM.strip_business_numbers("") == ""
