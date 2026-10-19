"""Tests for viakable_forecast.router — route(), redact(), content_tokens(), patterns."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from viakable_forecast import router as RT


# ── redact ───────────────────────────────────────────────────────────────────
class TestRedact:
    def test_removes_email(self):
        out = RT.redact("Escríbeme a saul@example.com mañana")
        assert "@" not in out
        assert "[email]" in out

    def test_removes_url(self):
        out = RT.redact("Visita https://viakable.com para más info")
        assert "viakable.com" not in out
        assert "[url]" in out

    def test_removes_phone(self):
        out = RT.redact("Llámame al +52 81 1234 5678 por favor")
        assert "1234" not in out
        assert "[phone]" in out

    def test_clean_text_unchanged(self):
        text = "dame el pronostico del calibre 10"
        assert RT.redact(text) == text

    def test_multiple_pii_in_one_string(self):
        out = RT.redact("Email: a@b.com tel: 5551234567 url: http://x.com")
        assert "[email]" in out
        assert "[phone]" in out
        assert "[url]" in out


# ── content_tokens ────────────────────────────────────────────────────────────
class TestContentTokens:
    def test_removes_stopwords(self):
        tokens = RT.content_tokens("dame el pronóstico del calibre 6")
        assert "el" not in tokens
        assert "del" not in tokens

    def test_keeps_content_words(self):
        tokens = RT.content_tokens("pronóstico cobre arancel")
        assert "pronostico" in tokens
        assert "cobre" in tokens

    def test_empty_returns_empty(self):
        assert RT.content_tokens("") == ""


# ── route — app ───────────────────────────────────────────────────────────────
class TestRouteApp:
    @pytest.mark.parametrize("q", [
        "¿Cuánto vamos a vender en los próximos 6 meses?",
        "¿Cuál es el pronóstico del calibre 12 para el mercado nacional?",
        "¿Qué pasa si el cobre sube 10%?",
        "¿Cómo cambia el pronóstico si el arancel sube 5 puntos?",
        "¿Por qué cambió el pronóstico del calibre 6 este mes?",
        "Why did the total forecast change?",
        "¿Qué alertas hay ahora?",
        "¿Cuál es el precio del cobre en vivo?",
        "¿Cuánto metal tengo que comprar en los próximos 6 meses?",
        "¿Qué clientes están comprando menos?",
        "Prepara el resumen para la junta de S&OP.",
        "¿Qué tan preciso es el pronóstico total?",
        "¿Cuál es el margen en riesgo a 6 meses?",
        "Hola, buenos días.",
    ])
    def test_app_route(self, q):
        r = RT.route(q)
        assert r.kind == "app", f"Esperado 'app', obtenido '{r.kind}' para: {q}"

    def test_app_route_has_parse(self):
        r = RT.route("dame el pronóstico del calibre 6")
        assert r.parse is not None
        assert r.parse.intent == "forecast"


# ── route — kb ────────────────────────────────────────────────────────────────
class TestRouteKb:
    @pytest.mark.parametrize("q", [
        "¿Cómo funciona el pronóstico?",
        "¿Qué es el MAPE y cómo se interpreta?",
        "¿Qué es la validación cruzada de origen móvil?",
        "Explica el inventario de seguridad y cómo se calcula.",
        "¿Qué es el ensamble ponderado?",
        "¿Cómo funciona el machine learning en la app?",
        "¿Qué son las señales de mercado en Calibra?",
        "How does cross-validation work in this app?",
    ])
    def test_kb_route(self, q):
        r = RT.route(q)
        assert r.kind == "kb", f"Esperado 'kb', obtenido '{r.kind}' para: {q}"

    def test_kb_route_has_entry(self):
        r = RT.route("¿Qué es el MAPE y cómo se interpreta?")
        assert r.entry is not None
        assert r.entry in ["mape_precision", "que_es_calibra", "validacion_cruzada",
                           "inventario_seguridad", "ensamble", "machine_learning",
                           "senales", "metodos_pronostico"]


# ── route — wiki ──────────────────────────────────────────────────────────────
class TestRouteWiki:
    @pytest.mark.parametrize("q", [
        "¿Qué es el cobre?",
        "¿Cuándo se fundó el TLCAN?",
        "¿Qué es la Reserva Federal de Estados Unidos?",
        "¿Dónde está Monterrey?",
        "What is nearshoring?",
        "What is the London Metal Exchange?",
        "¿Cuántos habitantes tiene Jalisco?",
        "What is GDP?",
    ])
    def test_wiki_route(self, q):
        r = RT.route(q)
        assert r.kind == "wiki", f"Esperado 'wiki', obtenido '{r.kind}' para: {q}"


# ── route — open / mixed ──────────────────────────────────────────────────────
class TestRouteOpenMixed:
    @pytest.mark.parametrize("q,kind", [
        ("¿Cómo debería negociar con un cliente que pide descuento?", "open"),
        ("¿Cómo afectaría una recesión en EE. UU. a nuestras exportaciones?", "mixed"),
        ("¿Qué impacto tiene el nearshoring en la demanda de cable eléctrico?", "mixed"),
        ("¿Cómo afecta la inflación a los costos de producción?", "mixed"),
        ("What are the best negotiation tactics for B2B sales?", "open"),
        ("¿Cuál es el futuro del cableado eléctrico con los autos eléctricos?", "open"),
    ])
    def test_open_mixed_route(self, q, kind):
        r = RT.route(q)
        assert r.kind == kind, f"Esperado '{kind}', obtenido '{r.kind}' para: {q}"


# ── route — forward-looking questions must NOT go to wiki ─────────────────────
class TestForwardLookingNotWiki:
    @pytest.mark.parametrize("q", [
        "¿Cuál es el futuro del cableado eléctrico con los autos eléctricos?",
        "What are the best negotiation tactics for B2B sales?",
        "¿Cuáles son las mejores prácticas de S&OP?",
    ])
    def test_not_wiki(self, q):
        r = RT.route(q)
        assert r.kind != "wiki", f"Pregunta prospectiva no debe ir a wiki: {q}"


# ── route — methodology trumps weak app intent ────────────────────────────────
class TestMethodologyOverride:
    def test_mape_goes_to_kb_not_app(self):
        r = RT.route("¿Qué es el MAPE?")
        assert r.kind == "kb"

    def test_como_funciona_goes_to_kb_not_app(self):
        r = RT.route("¿Cómo funciona el pronóstico?")
        assert r.kind == "kb"

    def test_inventario_explicar_goes_to_kb(self):
        r = RT.route("Explica el inventario de seguridad y cómo se calcula.")
        assert r.kind == "kb"


# ── route — confidence sanity ─────────────────────────────────────────────────
class TestRouteConfidence:
    def test_confidence_range(self):
        for q in [
            "dame el pronóstico",
            "¿qué es el mape?",
            "¿qué es el cobre?",
            "¿cómo negociar con clientes?",
        ]:
            r = RT.route(q)
            assert 0.0 <= r.conf <= 1.0, f"Confianza fuera de rango para: {q}"

    def test_strong_app_has_higher_conf_than_fallback(self):
        strong = RT.route("¿Qué pasa si el cobre sube 10%?")
        fallback = RT.route("cuéntame algo interesante")
        assert strong.conf >= fallback.conf


# ── route — learned intents ───────────────────────────────────────────────────
class TestLearnedIntents:
    def test_learned_phrase_overrides_default_route(self):
        learned = {"arancel cero": "whatif"}
        r = RT.route("simula arancel cero exportaciones", learned=learned)
        assert r.kind == "app"
        assert r.parse is not None
        assert r.parse.intent == "whatif"
