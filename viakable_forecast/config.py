"""Viakable Calibra — constants, signals, presets and default parameters."""
from __future__ import annotations

APP_NAME    = "Viakable Calibra"
APP_BRAND   = "Viakable"
APP_PRODUCT = "Calibra"
MKT_NAC = "Nacional"
MKT_USA = "Exportación EE. UU."

MESES_ES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
            "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
FOOTER = (
    "Viakable Calibra es un prototipo académico. Los precios, costos, "
    "capacidades y elasticidades son ilustrativos. No conecta con SAP. "
    "Los roles no son seguridad formal. Las señales son supuestos hasta "
    "que se calibren con datos reales."
)

TARIFF_CURRENT = 0.50
TARIFF_ELASTICITY = 0.6

# ── Demand-class descriptions ────────────────────────────────────────────────
DCLASS_NOTE: dict[str, str] = {
    "Estable":       "Demanda regular y predecible; los modelos estadísticos rinden bien.",
    "Errática":      "Alta variabilidad intermensual; se recomienda revisar el ensemble.",
    "Intermitente":  "Muchos meses en cero; usa Croston-SBA o TSB.",
    "Irregular":     "Alta variabilidad y muchos ceros; difícil de pronosticar.",
}

# ── Method descriptions (shown in the lab and method selector) ────────────────
METHOD_DESC: dict[str, str] = {
    "mismo_mes_año":    "Mismo mes del año anterior ajustado por crecimiento.",
    "promedio_3m":      "Promedio de los últimos 3 meses.",
    "suavización":      "Suavización exponencial simple (α óptimo).",
    "holt":             "Holt con tendencia amortiguada (φ=0.9).",
    "holt_winters":     "Holt-Winters multiplicativo (necesita 24+ meses).",
    "regresión":        "Regresión lineal sobre datos desestacionalizados.",
    "croston":          "Croston-SBA para demanda intermitente.",
    "tsb":              "Teunter-Syntetos-Babai para demanda intermitente.",
    "ets":              "ETS automático (Holt-Winters con AICc).",
    "sarima":           "SARIMA automático (orden elegido por error en muestra).",
    "ensemble":         "Ensemble ponderado por error inverso (5 mejores métodos).",
    "ml":               "Gradient Boosting global entrenado en todos los productos.",
    "xgboost":          "XGBoost global (50 árboles, profundidad 4).",
    "rf":               "Random Forest global.",
    "ridge":            "Ridge con variables de rezago y mes del año.",
    "mlp":              "Red neuronal pequeña (MLPRegressor).",
    "stacking":         "Apilamiento de los tres mejores modelos por error inverso.",
    "similar":          "Aprendizaje de series similares (mismo calibre y metal).",
    "signal":           "Modelo por señales de mercado.",
}

# ── Signals ───────────────────────────────────────────────────────────────────
# {name: {"sens": default_sensitivity, "lead": default_lead_months}}
SIGNALS: dict[str, dict] = {
    "Permisos de construcción":     {"sens":  0.30, "lead": 3, "mkt": "MX"},
    "Licitaciones CFE":             {"sens":  0.25, "lead": 2, "mkt": "MX"},
    "Naves industriales":           {"sens":  0.20, "lead": 2, "mkt": "MX"},
    "Data centers":                 {"sens":  0.15, "lead": 4, "mkt": "MX"},
    "PMI manufacturero México":     {"sens":  0.20, "lead": 1, "mkt": "MX"},
    "Precio del cobre":             {"sens": -0.10, "lead": 1, "mkt": "MX"},
    "MXN/USD":                      {"sens": -0.15, "lead": 1, "mkt": "MX"},
    "Inicios de vivienda EE. UU.":  {"sens":  0.25, "lead": 3, "mkt": "US"},
    "PMI manufacturero EE. UU.":    {"sens":  0.20, "lead": 1, "mkt": "US"},
}

# {name: (description, current_reading, source)}
SIGNAL_INFO: dict[str, tuple] = {
    "Permisos de construcción":  ("Nuevas viviendas y edificios autorizados.", "Estable", "INEGI"),
    "Licitaciones CFE":          ("Contratos de infraestructura eléctrica pública.", "Moderado", "CFE"),
    "Naves industriales":        ("Construcción de espacios industriales y logísticos.", "En alza", "CBRE/JLL"),
    "Data centers":              ("Inversión en centros de datos, nearshoring.", "En alza", "Bancomext"),
    "PMI manufacturero México":  ("Índice de actividad manufacturera.", "Estable", "S&P Global"),
    "Precio del cobre":          ("LME copper USD/t (COMEX como referencia).", "En alza", "COMEX"),
    "MXN/USD":                   ("Tipo de cambio peso-dólar.", "Estable", "Banxico"),
    "Inicios de vivienda EE. UU.":  ("Nuevas construcciones residenciales en EE. UU.", "Estable", "U.S. Census"),
    "PMI manufacturero EE. UU.":    ("ISM Manufacturing PMI de EE. UU.", "Estable", "ISM"),
}

# ── Presets ───────────────────────────────────────────────────────────────────
# {preset_name: {signal_name: expected_change_pct}}
SIGNAL_PRESETS: dict[str, dict] = {
    "Tendencia actual": {
        "Permisos de construcción":     0,
        "Licitaciones CFE":             0,
        "Naves industriales":           5,
        "Data centers":                10,
        "PMI manufacturero México":     0,
        "Precio del cobre":             0,
        "MXN/USD":                      0,
        "Inicios de vivienda EE. UU.":  0,
        "PMI manufacturero EE. UU.":    0,
    },
    "Auge nearshoring": {
        "Permisos de construcción":    10,
        "Licitaciones CFE":             5,
        "Naves industriales":          20,
        "Data centers":                25,
        "PMI manufacturero México":     5,
        "Precio del cobre":            -5,
        "MXN/USD":                     -5,
        "Inicios de vivienda EE. UU.": 10,
        "PMI manufacturero EE. UU.":    5,
    },
    "Desaceleración": {
        "Permisos de construcción":   -10,
        "Licitaciones CFE":           -10,
        "Naves industriales":          -5,
        "Data centers":                 0,
        "PMI manufacturero México":   -10,
        "Precio del cobre":           -10,
        "MXN/USD":                     10,
        "Inicios de vivienda EE. UU.":-15,
        "PMI manufacturero EE. UU.":  -10,
    },
    "Cobre al alza": {
        "Permisos de construcción":    0,
        "Licitaciones CFE":            0,
        "Naves industriales":          0,
        "Data centers":                0,
        "PMI manufacturero México":    0,
        "Precio del cobre":           20,
        "MXN/USD":                     0,
        "Inicios de vivienda EE. UU.": 0,
        "PMI manufacturero EE. UU.":   0,
    },
}

# ── Default prices ────────────────────────────────────────────────────────────
# cu/al: 12-month flat paths (USD/t); fx: MXN/USD; pass_through: fraction;
# metal_pct: fraction of cost that is metal; mxn_t/usd_t: list prices by gauge.
_CU = 14770.0
_AL = 3181.0
_FX = 18.43

DEFAULT_PRICES: dict = {
    "cu":  [_CU] * 12,
    "al":  [_AL] * 12,
    "fx":  _FX,
    "pass_through": 50.0,
    "metal_pct":    70.0,
    "cu_ref": 9840.0,
    "al_ref": 2600.0,
    "fx_ref": 18.5,
    "mxn_t": {"6": 85_000, "10": 78_000, "12": 72_000, "14": 68_000, "16": 65_000, "18": 60_000},
    "usd_t": {"6": 4_200,  "10": 3_900,  "12": 3_600,  "14": 3_400,  "16": 3_200,  "18": 3_000},
}

# ── Planning defaults ─────────────────────────────────────────────────────────
PLAN_DEFAULTS: dict = {
    "cu_lme":    _CU,
    "al_lme":    _AL,
    "fx":        _FX,
    "service":   95,
    "lead_time": 1.5,
    "exp_lt":    3.0,
    "freight":   200.0,
    "ddp":       False,
    "hold_pct":  25.0,
    "metal_pct": 70.0,
    "metal_chg": 0.0,
    "conv_cu":   5_000.0,
    "conv_al":   4_500.0,
    "pass_through": 50.0,
}
