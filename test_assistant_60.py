"""60-question routing test for the hybrid assistant (no app engine needed).

Run with: python test_assistant_60.py
Tests routing only (not LLM or Wikipedia responses, which require network/keys).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from viakable_forecast import assistant as AS
from viakable_forecast import router as RT

QUESTIONS = [
    # ── (a) Business data / forecast – expect "app" ─────────────────────────
    ("¿Cuánto vamos a vender en los próximos 6 meses?",                 "app",  "forecast"),
    ("¿Cuál es el pronóstico del calibre 12 para el mercado nacional?", "app",  "forecast"),
    ("Dame el pronóstico de los calibres 6 y 10.",                      "app",  "forecast"),
    ("What is the 6-month forecast for gauge 14?",                      "app",  "forecast"),
    ("¿Qué pasa si el cobre sube 10%?",                                 "app",  "whatif"),
    ("¿Cómo cambia el pronóstico si el arancel sube 5 puntos?",         "app",  "whatif"),
    ("Simula un arancel de cero para las exportaciones.",               "app",  "whatif"),
    ("What happens if we get a 20t extra order in gauge 10 for 3 months?", "app", "whatif"),
    ("¿Por qué cambió el pronóstico del calibre 6 este mes?",           "app",  "why"),
    ("Why did the total forecast change?",                              "app",  "why"),
    ("¿Qué alertas hay ahora?",                                         "app",  "alerts"),
    ("¿Qué debo atender hoy?",                                          "app",  "alerts"),
    ("¿Cuál es el precio del cobre en vivo?",                           "app",  "prices"),
    ("¿A cuánto está el tipo de cambio USD/MXN ahora?",                 "app",  "prices"),
    ("¿Qué dice la noticia más relevante sobre aranceles?",             "app",  "news"),
    ("Muéstrame las noticias del Federal Register.",                    "app",  "news"),
    ("¿Cuánto metal tengo que comprar en los próximos 6 meses?",        "app",  "plan"),
    ("Dame el plan de producción y el inventario de seguridad.",        "app",  "plan"),
    ("¿Qué clientes están comprando menos?",                            "app",  "customers"),
    ("Which customers have reduced their purchases?",                   "app",  "customers"),
    ("Prepara el resumen para la junta de S&OP.",                       "app",  "summary"),
    ("¿Qué tan preciso es el pronóstico total?",                        "app",  "accuracy"),
    ("¿Cuál es el margen en riesgo a 6 meses?",                         "app",  "risk"),
    ("¿Cuánto puede caer el margen en el peor 5% de los escenarios?",   "app",  "risk"),
    ("Explica por qué el calibre 10 tiene ese pronóstico en enero.",    "app",  "explain"),
    ("¿Qué puedes contestar?",                                          "app",  "help"),
    ("Hola, buenos días.",                                              "app",  "smalltalk"),
    ("¿Cuál es el pronóstico de exportación a EE. UU. para el calibre 6?", "app", "forecast"),
    ("¿Qué pasa si el aluminio baja 15%?",                              "app",  "whatif"),
    ("Muéstrame la precisión de validación del calibre 14.",            "app",  "accuracy"),

    # ── (b) Methodology – expect "kb" ────────────────────────────────────────
    ("¿Cómo funciona el pronóstico?",                                   "kb",   None),
    ("¿Qué es el MAPE y cómo se interpreta?",                           "kb",   None),
    ("¿Qué diferencia hay entre Holt-Winters y ETS automático?",        "kb",   None),
    ("¿Qué es la validación cruzada de origen móvil?",                  "kb",   None),
    ("Explica el inventario de seguridad y cómo se calcula.",           "kb",   None),
    ("¿Qué significa una clase de demanda 'Intermitente'?",             "kb",   None),
    ("¿Qué es el ensamble ponderado?",                                  "kb",   None),
    ("¿Cómo funciona el machine learning en la app?",                   "kb",   None),
    ("¿Qué son las señales de mercado en Calibra?",                     "kb",   None),
    ("How does cross-validation work in this app?",                     "kb",   None),

    # ── (c) General facts – expect "wiki" ────────────────────────────────────
    ("¿Qué es el cobre?",                                               "wiki", None),
    ("¿Cuándo se fundó el TLCAN?",                                      "wiki", None),
    ("¿Qué es la Reserva Federal de Estados Unidos?",                   "wiki", None),
    ("¿Qué es el PMI manufacturero?",                                   "wiki", None),
    ("¿Dónde está Monterrey?",                                          "wiki", None),
    ("What is nearshoring?",                                            "wiki", None),
    ("¿Quién es el Secretario de Economía de México?",                  "wiki", None),
    ("What is the London Metal Exchange?",                              "wiki", None),
    ("¿Cuántos habitantes tiene Jalisco?",                              "wiki", None),
    ("¿Qué es el USMCA?",                                               "wiki", None),
    ("¿Cuánto mide la Torre Latinoamericana?",                          "wiki", None),
    ("What is Section 232 of the Trade Expansion Act?",                 "wiki", None),
    ("¿Qué es el nearshoring en México?",                               "wiki", None),
    ("¿Qué es la cadena de suministro?",                                "wiki", None),
    ("What is GDP?",                                                    "wiki", None),

    # ── (d) Open / general – expect "open" or "mixed" ────────────────────────
    ("¿Cómo debería negociar con un cliente que pide descuento?",       "open", None),
    ("¿Cómo afectaría una recesión en EE. UU. a nuestras exportaciones?", "mixed", None),
    ("¿Qué estrategia de inventario me recomiendas para temporada alta?", "open", None),
    ("¿Cómo se usa la teoría de restricciones en una planta de cable?", "open", None),
    ("¿Qué impacto tiene el nearshoring en la demanda de cable eléctrico?", "mixed", None),
    ("¿Cuáles son las mejores prácticas de S&OP?",                      "open", None),
    ("¿Qué tecnologías están transformando la cadena de suministro?",   "open", None),
    ("¿Cómo afecta la inflación a los costos de producción?",           "mixed", None),
    ("What are the best negotiation tactics for B2B sales?",            "open", None),
    ("¿Cuál es el futuro del cableado eléctrico con los autos eléctricos?", "open", None),
]

ROUTE_LABELS = {"app": "(a) Negocio", "kb": "(b) Metodología", "wiki": "(c) Hechos", "mixed": "(d) Mixta", "open": "(d) Abierta"}

def run():
    correct = 0
    wrong = []
    results = []

    for q, expected_route, expected_intent in QUESTIONS:
        r = RT.route(q, customers=[], lang="es")
        p = r.parse or AS.parse(q)

        route_ok = r.kind == expected_route
        intent_ok = (expected_intent is None) or (p.intent == expected_intent)
        ok = route_ok and intent_ok

        if ok:
            correct += 1
        else:
            note = ""
            if not route_ok:
                note += f" RUTA esperada={expected_route} obtenida={r.kind}"
            if not intent_ok:
                note += f" INTENTO esperado={expected_intent} obtenido={p.intent}"
            wrong.append((q, note.strip()))

        results.append({
            "q": q[:65],
            "expected": expected_route + (f"/{expected_intent}" if expected_intent else ""),
            "got": r.kind + (f"/{p.intent}" if p.intent else ""),
            "conf": f"{r.conf:.2f}",
            "ok": "✓" if ok else "✗",
        })

    print(f"\n{'-'*90}")
    print(f"{'#':>3}  {'OK':>2}  {'Esperado':<22}  {'Obtenido':<22}  {'Conf':>5}  Pregunta")
    print(f"{'-'*90}")
    for i, r in enumerate(results, 1):
        print(f"{i:>3}  {r['ok']:>2}  {r['expected']:<22}  {r['got']:<22}  {r['conf']:>5}  {r['q']}")

    print(f"{'-'*90}")
    pct = correct / len(QUESTIONS) * 100
    print(f"\nResultado: {correct}/{len(QUESTIONS)} correctas ({pct:.0f}%)\n")

    if wrong:
        print("Errores:")
        for q, note in wrong:
            print(f"  · {q[:70]} → {note}")
    else:
        print("Sin errores.")

if __name__ == "__main__":
    run()
