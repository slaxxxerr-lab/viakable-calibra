"""Methodology knowledge base for Asistente Calibra.

Each entry has a Spanish title, Spanish body, and English equivalents.
The router uses TF-IDF similarity against these entries to detect methodology questions.
"""
from __future__ import annotations

# Each entry: {"es_title", "es_body", "en_title", "en_body", "keywords"}
# keywords are used for fast pre-filtering before TF-IDF.
_ENTRIES: dict[str, dict] = {
    "que_es_calibra": {
        "keywords": ["calibra", "app", "aplicación", "herramienta", "para qué sirve", "qué hace"],
        "es_title": "¿Qué es Viakable Calibra?",
        "es_body": (
            "Calibra es la app de pronóstico de ventas, planeación de inventarios y CRM de Viakable. "
            "Carga datos históricos de ventas (por calibre de cable y mercado), genera pronósticos a 1–36 meses "
            "con varios métodos estadísticos y de machine learning, y los convierte en planes de producción, "
            "compras de metal y gestión de clientes. Todo calcula en tiempo real con cada cambio."
        ),
        "en_title": "What is Viakable Calibra?",
        "en_body": (
            "Calibra is Viakable's sales forecasting, inventory planning and CRM app. "
            "It loads historical sales data (by cable gauge and market), generates 1–36-month forecasts "
            "with statistical and machine-learning methods, and converts them into production plans, "
            "metal purchasing plans and customer management. Everything recalculates in real time with each change."
        ),
    },
    "metodos_pronostico": {
        "keywords": ["método", "pronostico", "promedio", "suavizamiento", "holt", "winters", "sarima", "ets", "naïve", "naive", "croston", "tsb"],
        "es_title": "Métodos de pronóstico disponibles",
        "es_body": (
            "La app incluye: (1) Mismo mes del año anterior × crecimiento (naïve estacional). "
            "(2) Promedio móvil de 3 meses. "
            "(3) Suavizamiento exponencial simple (α óptimo). "
            "(4) Holt con tendencia amortiguada (φ=0.9). "
            "(5) Holt-Winters multiplicativo (necesita ≥24 meses). "
            "(6) Regresión sobre datos desestacionalizados. "
            "(7) Croston-SBA y TSB para demanda intermitente. "
            "(8) ETS automático y SARIMA automático (statsmodels, elegidos por AICc o error en muestra). "
            "Además hay modelos de machine learning: Gradient Boosting, XGBoost, Random Forest, Ridge y red neuronal. "
            "AUTO elige el método con menor error de validación cruzada."
        ),
        "en_title": "Available forecast methods",
        "en_body": (
            "The app includes: (1) Same month last year × growth (seasonal naïve). "
            "(2) 3-month moving average. "
            "(3) Simple exponential smoothing (optimal α). "
            "(4) Holt with damped trend (φ=0.9). "
            "(5) Multiplicative Holt-Winters (needs ≥24 months). "
            "(6) Regression on deseasonalized data. "
            "(7) Croston-SBA and TSB for intermittent demand. "
            "(8) Automatic ETS and automatic SARIMA (statsmodels, chosen by AICc or in-sample error). "
            "Machine-learning models: Gradient Boosting, XGBoost, Random Forest, Ridge and neural network. "
            "AUTO picks the method with the lowest cross-validation error."
        ),
    },
    "validacion_cruzada": {
        "keywords": ["validación", "validacion", "cross-validation", "cross validation", "ventanas", "origen", "backtesting", "backtest", "holdout"],
        "es_title": "Validación cruzada de origen móvil",
        "es_body": (
            "En lugar de reservar un solo periodo final (holdout), la app corta los datos en varias ventanas: "
            "6, 9, 12 y 15 meses antes del final. En cada ventana entrena solo con datos anteriores y pronostica "
            "los 6 meses siguientes. El error de un método es el promedio de sus errores en todas las ventanas, "
            "lo que evita que la selección dependa de un único periodo. "
            "Holt-Winters y la regresión desestacionalizada necesitan ≥24 meses de entrenamiento y solo entran al "
            "ranking si tienen al menos 3 ventanas completas."
        ),
        "en_title": "Rolling-origin cross-validation",
        "en_body": (
            "Instead of a single holdout period, the app cuts the data at several origins: 6, 9, 12 and 15 months "
            "before the end. At each origin it trains only on earlier data and forecasts the next 6 months. "
            "A method's error is the average across all windows, so selection does not depend on one period. "
            "Holt-Winters and deseasonalized regression need ≥24 training months and only rank when they have "
            "at least 3 complete windows."
        ),
    },
    "mape_precision": {
        "keywords": ["mape", "wape", "error", "precisión", "precision", "certero", "confiable", "porcentaje", "exactitud"],
        "es_title": "MAPE, WAPE y etiquetas de precisión",
        "es_body": (
            "El error principal es el MAPE (Mean Absolute Percentage Error): promedio del |real − pronóstico| / real. "
            "Si hay meses con cero en el holdout se usa el WAPE (error total / demanda total). "
            "Precisión = 100 − error. "
            "Etiquetas: error <5% → Muy certero; <10% → Certero; <20% → Aceptable; ≥20% → Poco confiable. "
            "La app nunca usa R² para evaluar pronósticos; solo MAPE/WAPE sobre meses futuros reales."
        ),
        "en_title": "MAPE, WAPE and accuracy labels",
        "en_body": (
            "The main error metric is MAPE (Mean Absolute Percentage Error): average of |actual − forecast| / actual. "
            "If the holdout contains zero months, WAPE is used (total error / total demand). "
            "Accuracy = 100 − error. "
            "Labels: error <5% → Very accurate; <10% → Accurate; <20% → Acceptable; ≥20% → Unreliable. "
            "The app never uses R² to evaluate forecasts; only MAPE/WAPE on real future months."
        ),
    },
    "clases_demanda": {
        "keywords": ["clase", "demanda", "estable", "erratica", "errática", "intermitente", "irregular", "syntetos", "boylan", "adi", "cv2", "cv²"],
        "es_title": "Clases de demanda (Syntetos-Boylan)",
        "es_body": (
            "Cada serie se clasifica con el método de Syntetos-Boylan usando dos indicadores: "
            "ADI (Average Demand Interval, intervalo promedio entre pedidos) y CV² (coeficiente de variación al cuadrado). "
            "Si ADI < 1.32 y CV² < 0.49 → Estable. "
            "ADI < 1.32 y CV² ≥ 0.49 → Errática. "
            "ADI ≥ 1.32 y CV² < 0.49 → Intermitente. "
            "ADI ≥ 1.32 y CV² ≥ 0.49 → Irregular. "
            "La clase guía qué métodos son más apropiados: Croston-SBA y TSB para intermitente e irregular."
        ),
        "en_title": "Demand classes (Syntetos-Boylan)",
        "en_body": (
            "Each series is classified with the Syntetos-Boylan method using two indicators: "
            "ADI (Average Demand Interval) and CV² (squared coefficient of variation). "
            "ADI < 1.32 and CV² < 0.49 → Smooth. "
            "ADI < 1.32 and CV² ≥ 0.49 → Erratic. "
            "ADI ≥ 1.32 and CV² < 0.49 → Intermittent. "
            "ADI ≥ 1.32 and CV² ≥ 0.49 → Lumpy. "
            "The class guides which methods are most appropriate: Croston-SBA and TSB for intermittent and lumpy."
        ),
    },
    "ensamble": {
        "keywords": ["ensamble", "ensemble", "combinación", "ponderado", "pesos", "stacking"],
        "es_title": "Ensamble ponderado por error inverso",
        "es_body": (
            "El ensamble combina los 5 métodos estadísticos más precisos asignando a cada uno un peso "
            "proporcional a 1 / error_de_validación. Los pesos de cada ventana se calculan dejando esa ventana fuera, "
            "así el ensamble no se puntúa sobre datos con los que se afinó. "
            "AUTO mantiene el ensamble a menos que otro método lo supere por más de 2 puntos porcentuales, "
            "porque combinar métodos es más robusto que elegir uno solo."
        ),
        "en_title": "Inverse-error weighted ensemble",
        "en_body": (
            "The ensemble combines the 5 most accurate statistical methods, giving each a weight proportional "
            "to 1 / validation_error. Weights for each window are computed leaving that window out, "
            "so the ensemble is not scored on data it was tuned with. "
            "AUTO keeps the ensemble unless another method beats it by more than 2 percentage points, "
            "because combining methods is more robust than picking a single one."
        ),
    },
    "machine_learning": {
        "keywords": ["machine learning", "ml", "gradient boosting", "xgboost", "bosque", "forest", "ridge", "red neuronal", "neural", "shap"],
        "es_title": "Modelos de machine learning",
        "es_body": (
            "La app entrena cuatro modelos globales con todas las series de productos a la vez: "
            "Gradient Boosting (scikit-learn, 150 árboles), XGBoost, Random Forest y Ridge. "
            "Las características incluyen: rezagos 1-3, mismo mes del año anterior, promedio de 3 meses, "
            "tendencia reciente, mes del año, calibre, familia, indicador de exportación y logaritmo del tamaño. "
            "El modelo XGBoost genera valores SHAP (contribuciones de características por predicción). "
            "AUTO excluye los modelos de ML porque en la validación honesta los métodos estadísticos simples "
            "tuvieron menor error en el total (7-7.5% vs 10-11%). Se pueden elegir manualmente."
        ),
        "en_title": "Machine learning models",
        "en_body": (
            "The app trains four global models on all product series at once: "
            "Gradient Boosting (scikit-learn, 150 trees), XGBoost, Random Forest and Ridge. "
            "Features include: lags 1-3, same month last year, 3-month average, recent trend, "
            "month of year, gauge, family, export flag and log of size. "
            "The XGBoost model generates SHAP values (feature contributions per prediction). "
            "AUTO excludes ML models because in honest validation simple statistical methods had lower error "
            "on the total (7-7.5% vs 10-11%). They can be selected manually."
        ),
    },
    "inventario_seguridad": {
        "keywords": ["inventario", "seguridad", "safety stock", "reorden", "stock", "nivel de servicio", "lead time"],
        "es_title": "Inventario de seguridad",
        "es_body": (
            "La fórmula es: SS = z × 1.25 × error × demanda_media × √lead_time. "
            "z depende del nivel de servicio: 90% → 1.28, 95% → 1.65, 98% → 2.05, 99% → 2.33. "
            "El factor 1.25 ajusta porque los errores de pronóstico tienden a subestimarse. "
            "El punto de reorden = demanda_media × lead_time + SS. "
            "La app calcula el capital liberado comparando el SS con el mejor método contra el método tradicional."
        ),
        "en_title": "Safety stock",
        "en_body": (
            "Formula: SS = z × 1.25 × error × mean_demand × √lead_time. "
            "z depends on service level: 90% → 1.28, 95% → 1.65, 98% → 2.05, 99% → 2.33. "
            "The 1.25 factor adjusts because forecast errors tend to be underestimated. "
            "Reorder point = mean_demand × lead_time + SS. "
            "The app computes capital freed by comparing SS with the best method versus the traditional method."
        ),
    },
    "senales": {
        "keywords": ["señal", "señales", "signal", "construcción", "cfe", "permisos", "pmi", "nearshoring", "cobre", "tipo de cambio", "indicador"],
        "es_title": "Señales de mercado",
        "es_body": (
            "Las señales son indicadores externos que ajustan el pronóstico antes de que aparezcan en las ventas. "
            "Mexicanas (mueven la demanda nacional): permisos de construcción, licitaciones de CFE, naves industriales, "
            "data centers, sell-out de distribuidores, PMI manufacturero, precio del cobre, tipo de cambio MXN/USD. "
            "EE. UU. (mueven exportaciones): inicios de vivienda, PMI manufacturero EE. UU., tipo de cambio. "
            "Efecto = sensibilidad × cambio_esperado% × min(1, mes / tiempo_de_respuesta). "
            "Las señales mexicanas no mueven el mercado de exportación y viceversa."
        ),
        "en_title": "Market signals",
        "en_body": (
            "Signals are external indicators that adjust the forecast before they appear in sales. "
            "Mexico (move domestic demand): construction permits, CFE tenders, industrial buildings, "
            "data centers, distributor sell-out, manufacturing PMI, copper price, MXN/USD exchange rate. "
            "US (move exports): housing starts, US manufacturing PMI, exchange rate. "
            "Effect = sensitivity × expected_change% × min(1, month / lead_time). "
            "Mexican signals do not move the export market and vice versa."
        ),
    },
    "arancel": {
        "keywords": ["arancel", "tarifa", "232", "sección 232", "usmca", "t-mec", "exportación", "eeuu", "elasticidad"],
        "es_title": "Arancel Sección 232 y elasticidad",
        "es_body": (
            "El arancel de la Sección 232 aplica al cable exportado a EE. UU. "
            "El multiplicador de demanda es ((1 + escenario) / (1 + actual))^(−elasticidad). "
            "Por defecto: arancel actual 50%, elasticidad 0.6. "
            "Solo afecta la parte de exportación del pronóstico. "
            "Si los datos no tienen ventas de exportación definidas, el arancel no mueve nada. "
            "Los escenarios de arancel se activan en el panel '¿Qué pasa si…?' o en la sección Señales."
        ),
        "en_title": "Section 232 tariff and elasticity",
        "en_body": (
            "The Section 232 tariff applies to cable exported to the US. "
            "The demand multiplier is ((1 + scenario) / (1 + current))^(−elasticity). "
            "Defaults: current tariff 50%, elasticity 0.6. "
            "It only affects the export share of the forecast. "
            "If the data has no export sales defined, the tariff moves nothing. "
            "Tariff scenarios are activated in the '¿Qué pasa si…?' panel or in the Signals section."
        ),
    },
    "reconciliacion": {
        "keywords": ["reconciliación", "reconciliacion", "total", "suma", "calibres", "coherencia"],
        "es_title": "Reconciliación de pronósticos",
        "es_body": (
            "Por defecto, el pronóstico total es la suma de los pronósticos de cada calibre (bottom-up), "
            "usando el mejor método de cada uno. Esto garantiza que los totales sean consistentes con la tabla por calibre. "
            "También hay una opción para un pronóstico total directo (top-down). "
            "El intervalo de confianza al 80% se calcula a partir de los errores reales de las ventanas de validación: "
            "ratios actual/pronóstico − 1 agrupados por meses 1-2, 3-4, 5-6."
        ),
        "en_title": "Forecast reconciliation",
        "en_body": (
            "By default the total forecast is the sum of each gauge's forecast (bottom-up), "
            "each using its best method. This ensures totals are consistent with the gauge table. "
            "There is also an option for a direct total forecast (top-down). "
            "The 80% confidence interval is derived from real errors in the validation windows: "
            "actual/forecast − 1 ratios grouped by months 1-2, 3-4, 5-6."
        ),
    },
    "crm_oportunidades": {
        "keywords": ["crm", "oportunidad", "oportunidades", "cliente", "pipeline", "etapa", "prospecto", "cotizacion", "negociacion", "ganada"],
        "es_title": "CRM y oportunidades",
        "es_body": (
            "El CRM almacena clientes, oportunidades por etapa, actividades, tareas y bitácora de cambios. "
            "Etapas y probabilidades: Prospecto 10%, Calificada 25%, Cotización 50%, Negociación 75%, Ganada 100%, Perdida 0%. "
            "Las oportunidades abiertas entran al pronóstico ponderadas por la probabilidad de su etapa. "
            "Roles disponibles: Administrador, Planeación, Comercial, Consulta (con permisos distintos). "
            "No son seguridad formal; solo controlan qué controles se muestran."
        ),
        "en_title": "CRM and opportunities",
        "en_body": (
            "The CRM stores customers, opportunities by stage, activities, tasks and a change log. "
            "Stages and probabilities: Prospect 10%, Qualified 25%, Quote 50%, Negotiation 75%, Won 100%, Lost 0%. "
            "Open opportunities feed the forecast weighted by their stage probability. "
            "Available roles: Administrator, Planning, Commercial, Viewer (with different permissions). "
            "These are not formal security; they only control which controls are shown."
        ),
    },
    "alertas": {
        "keywords": ["alerta", "alertas", "qué atender", "urgente", "caída", "capacidad", "anomalía", "anomalia"],
        "es_title": "Alertas de negocio",
        "es_body": (
            "Las alertas aparecen en 'Qué atender' (sección Inicio). Se generan cuando: "
            "un calibre cae >8% vs el año anterior, el pronóstico tiene error ≥20%, las exportaciones caen, "
            "la capacidad de producción se excede, un cliente compra >20% menos que en los 6 meses previos, "
            "hay tareas vencidas, o hay desvíos vs pronósticos congelados. "
            "También llegan alertas de precios del metal (±3% en el día o ±8% en 30 días) "
            "y noticias de alta relevancia regulatoria."
        ),
        "en_title": "Business alerts",
        "en_body": (
            "Alerts appear in 'What needs attention' (Home section). They trigger when: "
            "a gauge falls >8% vs the prior year, a forecast has ≥20% error, exports fall, "
            "production capacity is exceeded, a customer buys >20% less than in the previous 6 months, "
            "there are overdue tasks, or there are deviations vs frozen forecasts. "
            "Metal price alerts (±3% in a day or ±8% in 30 days) and high-relevance regulatory news also appear."
        ),
    },
    "riesgo_simulacion": {
        "keywords": ["riesgo", "simulación", "montecarlo", "monte carlo", "var", "margen en riesgo", "cobertura", "hedge", "probabilidad"],
        "es_title": "Simulación de riesgo (Monte Carlo)",
        "es_body": (
            "La sección Riesgo (modo avanzado) simula demanda, precio del cobre, aluminio, tipo de cambio "
            "y arancel con distribuciones calibradas en los últimos 60 meses y correlaciones históricas. "
            "Resultados: margen esperado, margen en riesgo (P5 del margen), probabilidad de quedar bajo un umbral. "
            "Compra óptima de metal: fraccionar las compras reduce el riesgo de cola. "
            "Con 3-4 ventanas de validación la distribución se calibra para cubrir ~80% de los meses reales."
        ),
        "en_title": "Risk simulation (Monte Carlo)",
        "en_body": (
            "The Risk section (advanced mode) simulates demand, copper price, aluminum, exchange rate "
            "and tariff with distributions calibrated on the last 60 months and historical correlations. "
            "Results: expected margin, margin at risk (P5 of margin), probability of falling below a threshold. "
            "Optimal metal purchase: fixing a fraction of purchases reduces tail risk. "
            "With 3-4 validation windows the distribution is calibrated to cover ~80% of real months."
        ),
    },
    "abc_xyz": {
        "keywords": ["abc", "xyz", "portafolio", "segmentación", "volumen", "variabilidad", "impulsar", "descontinuar", "discontinuar"],
        "es_title": "Análisis ABC-XYZ de portafolio",
        "es_body": (
            "ABC clasifica productos por volumen acumulado: A (top 70%), B (siguiente 20%), C (último 10%). "
            "XYZ clasifica por variabilidad del CV (coeficiente de variación): X (CV < 0.3, muy estable), "
            "Y (0.3 ≤ CV < 0.6), Z (CV ≥ 0.6, muy variable). "
            "La combinación guía decisiones: AX/BX → Impulsar, CZ/BZ con margen bajo → Revisar o descontinuar. "
            "El margen aproximado por producto es toneladas × margen por tonelada del calibre (ilustrativo)."
        ),
        "en_title": "ABC-XYZ portfolio analysis",
        "en_body": (
            "ABC classifies products by cumulative volume: A (top 70%), B (next 20%), C (last 10%). "
            "XYZ classifies by CV variability: X (CV < 0.3, very stable), Y (0.3 ≤ CV < 0.6), Z (CV ≥ 0.6, very variable). "
            "The combination guides decisions: AX/BX → Promote, CZ/BZ with low margin → Review or discontinue. "
            "Approximate product margin = tons × margin per ton of the gauge (illustrative)."
        ),
    },
    "precios_metal": {
        "keywords": ["lme", "precio del cobre", "precio del aluminio", "comex", "traspaso", "pass-through", "precio de venta", "precio de lista"],
        "es_title": "Precios de metal y traspaso al precio de venta",
        "es_body": (
            "Los precios de cobre y aluminio vienen del COMEX (referencia al LME) vía Yahoo Finance. "
            "El precio de venta mensual se calcula como: precio_lista + traspaso × (costo_metal_mes − costo_metal_ref). "
            "Traspaso por defecto: 50% (configurable). "
            "El camino de precios puede ser constante, con escenario lineal/inmediato, mes a mes o tendencia amortiguada. "
            "La señal de cobre recibe el cambio del precio en vivo vs su promedio de 90 días."
        ),
        "en_title": "Metal prices and pass-through to selling price",
        "en_body": (
            "Copper and aluminum prices come from COMEX (reference to LME) via Yahoo Finance. "
            "Monthly selling price = list_price + pass_through × (month_metal_cost − reference_metal_cost). "
            "Default pass-through: 50% (configurable). "
            "The price path can be constant, linear/immediate scenario, month-by-month or damped trend. "
            "The copper signal receives the live price change vs its 90-day average."
        ),
    },
    "consenso": {
        "keywords": ["consenso", "ajuste manual", "ajuste", "justificación", "override"],
        "es_title": "Consenso (ajuste manual del pronóstico)",
        "es_body": (
            "Para un calibre específico puedes ingresar ajustes porcentuales mensuales con una justificación obligatoria. "
            "El consenso se aplica encima del pronóstico estadístico o de ML seleccionado. "
            "Los ajustes quedan en la bitácora del CRM y se pueden comparar contra los resultados reales "
            "(valor añadido del pronóstico = error estadístico − error del consenso)."
        ),
        "en_title": "Consensus (manual forecast adjustment)",
        "en_body": (
            "For a specific gauge you can enter monthly percentage adjustments with a mandatory justification. "
            "The consensus is applied on top of the selected statistical or ML forecast. "
            "Adjustments are logged in the CRM and can be compared against actual results "
            "(forecast value added = statistical error − consensus error)."
        ),
    },
}

# Pre-compute searchable text for each entry (for TF-IDF in router)
ENTRY_TEXTS = {k: f"{v['es_title']} {v['es_body']} {' '.join(v['keywords'])}" for k, v in _ENTRIES.items()}
ENTRY_TEXTS_EN = {k: f"{v['en_title']} {v['en_body']}" for k, v in _ENTRIES.items()}


def answer(entry: str, lang: str = "es") -> tuple[str, str]:
    """Return (title, body) for a knowledge base entry key."""
    e = _ENTRIES.get(entry)
    if not e:
        return "", ""
    if lang == "en":
        return e["en_title"], e["en_body"]
    return e["es_title"], e["es_body"]


def all_keys() -> list[str]:
    return list(_ENTRIES.keys())


def keyword_match(q_lower: str) -> str | None:
    """Fast pre-filter: return the first entry that has a keyword match."""
    for key, entry in _ENTRIES.items():
        for kw in entry["keywords"]:
            if kw in q_lower:
                return key
    return None
