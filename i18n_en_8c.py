"""Traducciones al inglés de la fase 8C: riesgo y decisiones."""
EN = {
    "Riesgo": "Risk", "Distribución del margen": "Margin distribution", "Simulaciones": "Simulations", "Umbral de margen (millones de MXN)": "Margin threshold (millions of MXN)",
    "Exención (0%)": "Exemption (0%)", "Sigue el vigente": "Current stays", "Sube 10 puntos": "Rises 10 points",
    "Las tres se ajustan a 100%. Son supuestos tuyos; sin ventas de exportación definidas no mueven el margen.": "The three are rescaled to 100%. They are your assumptions; without defined export sales they do not move the margin.",
    "Meses que cubre la compra": "Months the purchase covers", "Recargo por comprar de urgencia": "Rush-purchase premium", "Compras fijadas al precio de hoy": "Purchases fixed at today's price",
    "Meses cubiertos": "Months covered", "Costo de la cobertura (% del precio fijado)": "Hedge cost (% of fixed price)", "Pronóstico probabilístico": "Probabilistic forecast",
    "P10, P50 y P90 por calibre: en 8 de cada 10 casos la demanda cae entre P10 y P90. Salen de simular la demanda con los errores de la validación cruzada.":
        "P10, P50 and P90 by gauge: in 8 of 10 cases demand falls between P10 and P90. They come from simulating demand with the cross-validation errors.",
    "Compra óptima de metal": "Optimal metal purchase", "Modelo del vendedor de periódicos: compra hoy hasta el punto donde un faltante y un sobrante cuestan lo mismo en promedio.":
        "Newsvendor model: buy today up to the point where a shortage and an excess cost the same on average.",
    "Cobertura del precio del metal": "Metal price hedge", "Cuánto riesgo de margen se quita si una parte de las compras de los próximos meses se fija hoy al precio actual.":
        "How much margin risk is removed if part of the next months' purchases is fixed today at the current price.",
    "Arancel (probabilidad de cada caso)": "Tariff (probability of each case)", "Cobertura": "Hedge",
    "En {n} simulaciones, el margen de {h} meses queda entre {a} y {b} (P5 a P95).": "In {n} simulations, the {h}-month margin falls between {a} and {b} (P5 to P95).",
    "Hay {p}% de probabilidad de que el margen quede bajo {u}. Combina la demanda (errores de la validación cruzada), el cobre, el aluminio y el tipo de cambio (volatilidad histórica) y el arancel. Los precios y costos son ilustrativos.":
        "There is a {p}% probability that the margin falls below {u}. It combines demand (cross-validation errors), copper, aluminum and the exchange rate (historical volatility) and the tariff. Prices and costs are illustrative.",
    "Supuestos de la simulación": "Simulation assumptions", "% de las compras fijado hoy": "% of purchases fixed today", "Margen en riesgo (M MXN)": "Margin at risk (M MXN)",
    "Millones de MXN. Rayas: P5, esperado y P95.": "Millions of MXN. Lines: P5, expected and P95.", "Distribución del ingreso": "Revenue distribution",
    "Millones de MXN. P5 {a}, esperado {b}, P95 {c}.": "Millions of MXN. P5 {a}, expected {b}, P95 {c}.",
    "P50 puede diferir del pronóstico porque en la validación los errores tuvieron un sesgo promedio. Con los errores sin ensanchar, {r}% de los meses reales caía dentro de P10 a P90; ensanchados, {v}%.":
        "P50 can differ from the forecast because the validation errors had an average bias. With the errors unwidened, {r}% of real months fell within P10 to P90; widened, {v}%.",
    "Faltante = recargo de comprar de urgencia (ajustable). Sobrante = costo anual de mantener inventario por los meses que cubre la compra. La necesidad es la producción del plan más la desviación de la demanda simulada. No incluye descuentos por volumen ni coberturas: es una guía de cantidad, no de momento (para el momento, ve Metales).":
        "Shortage = rush-purchase premium (adjustable). Excess = annual holding cost for the months the purchase covers. The need is the plan's production plus the simulated demand deviation. It excludes volume discounts and hedges: it guides quantity, not timing (for timing, see Metals).",
    "Fijar {x}% de las compras de los próximos {k} meses baja el margen en riesgo de {a} a {b} ({c}% menos).": "Fixing {x}% of the next {k} months' purchases lowers margin at risk from {a} to {b} ({c}% less).",
    "Margen en riesgo sin cobertura": "Margin at risk without hedge", "Margen en riesgo con {x}%": "Margin at risk with {x}%", "Cobertura con menos riesgo": "Hedge with the least risk", "riesgo {x}": "risk {x}",
    "Cambio del margen esperado": "Change in expected margin", "por el costo de la cobertura": "from the hedge cost", "Tu cobertura": "Your hedge", "Mínimo riesgo": "Minimum risk",
    "Cubrir todo no es lo mejor: con un traspaso de {pt}% del metal al precio de venta, ya trasladas esa parte del riesgo a tus clientes, y la cobertura que menos riesgo deja ronda {o}%. Se fija el precio de compra del metal (en USD); el tipo de cambio sigue flotando. Es una simulación, no una oferta de cobertura.":
        "Hedging everything is not best: with {pt}% of the metal passed through to the sale price, you already pass that part of the risk to your customers, and the hedge that leaves the least risk is around {o}%. The metal purchase price (in USD) is fixed; the exchange rate keeps floating. This is a simulation, not a hedging offer.",
    "Margen esperado": "Expected margin", "plan sin riesgo: {x}": "plan without risk: {x}", "Margen en el peor 5% (P5)": "Margin in the worst 5% (P5)", "Margen en riesgo (esperado menos P5)": "Margin at risk (expected minus P5)",
    "Probabilidad de margen bajo el umbral": "Probability of margin below the threshold",
    "Margen en riesgo = margen esperado menos el percentil 5: lo que se puede perder en un mal escenario de cada 20. En el peor 5% el margen promedia {x}.":
        "Margin at risk = expected margin minus the 5th percentile: what can be lost in a bad 1-in-20 scenario. In the worst 5%, margin averages {x}.",
    "Volatilidad mensual": "Monthly volatility", "Origen": "Source", "Costo esperado (M MXN)": "Expected cost (M MXN)", "No hay un plan que simular con esta selección.": "There is no plan to simulate for this selection.",
    "Los precios siguen una caminata aleatoria lognormal alrededor del camino de Metales (sin tendencia propia), con la correlación histórica entre ellos. La demanda es el pronóstico más errores tomados al azar de la validación cruzada de todos los calibres, ensanchados {c}x para que en validación {v}% de los meses reales caiga dentro de P10 a P90 (sin ensanchar: {r}%). Con tan pocas ventanas, las colas son una aproximación.":
        "Prices follow a lognormal random walk around the Metals path (no drift of their own), with the historical correlation between them. Demand is the forecast plus errors drawn at random from the cross-validation of all gauges, widened {c}x so that in validation {v}% of real months fall within P10 to P90 (unwidened: {r}%). With so few windows, the tails are an approximation.",
    "{m}: comprar hoy {q} t": "{m}: buy {q} t today",
    "La necesidad de los próximos {k} meses va de {a} a {b} t (mediana {c} t). Un faltante cuesta {u} por tonelada (compra urgente) y un sobrante {v} (mantener inventario {k} meses), así que conviene cubrir hasta el percentil {p}: {q} t. Comprar solo la mediana costaría en promedio {x} más.":
        "The need for the next {k} months ranges from {a} to {b} t (median {c} t). A shortage costs {u} per ton (rush purchase) and an excess {v} (holding inventory {k} months), so it pays to cover up to percentile {p}: {q} t. Buying only the median would cost {x} more on average.",
    "Óptimo": "Optimal", "Mediana": "Median", "Toneladas compradas hoy": "Tons bought today", "simulaciones": "simulations",
    "Tus datos no tienen ventas de exportación definidas, así que el arancel no afecta esta simulación.": "Your data has no export sales defined, so the tariff does not affect this simulation.",
    "{n} meses de historia": "{n} months of history", "supuesto (poca historia)": "assumption (little history)",
    "Riesgo y cobertura": "Risk and hedging",
    "El margen no es un solo número: se simulan demanda, metales y arancel para ver qué tan malo puede ser el peor 5% y cuánto lo reduce fijar parte de las compras.":
        "Margin is not a single number: demand, metals and tariff are simulated to see how bad the worst 5% can be and how much fixing part of the purchases reduces it.",
    "Margen esperado {m}; en el peor 5% queda en {p} y hay {q}% de probabilidad de perder dinero. Fijar {x}% de las compras lo deja en {y}.":
        "Expected margin {m}; in the worst 5% it is {p} and there is a {q}% probability of losing money. Fixing {x}% of purchases leaves it at {y}.",
}
EN.update({"Fijar {x}% de las compras de los próximos {k} meses sube el margen en riesgo de {a} a {b} ({c}% más): es más cobertura de la que conviene.": "Fixing {x}% of the next {k} months' purchases raises margin at risk from {a} to {b} ({c}% more): that is more hedging than is advisable."})
