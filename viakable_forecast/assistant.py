"""Intent parser and entity extractor for Asistente Calibra. No Streamlit dependency.

Parses free-text questions in Spanish or English and extracts:
- intent: what the user wants (forecast, whatif, alerts, prices, news, plan,
          customers, summary, accuracy, risk, explain, help, smalltalk)
- entities: gauges, market, metals, tariff, percentages, horizon, month, etc.
- missing: required slots not yet provided (for follow-up questions)
- lang: detected language
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Sequence

try:
    from viakable_forecast.config import MKT_NAC, MKT_USA
except ImportError:
    MKT_NAC = "Nacional"
    MKT_USA = "Exportación EE. UU."

# --------------------------------------------------------------------------- normalisation
_ACCENTS = str.maketrans(
    "áéíóúàèìòùâêîôûäëïöüãõñÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛÄËÏÖÜÃÕÑ",
    "aeiouaeiouaeiouaeiouaonAEIOUAEIOUAEIOUAEIOUAON",
)


def norm(q: str) -> str:
    """Lowercase, remove accents, collapse whitespace."""
    return re.sub(r"\s+", " ", q.lower().translate(_ACCENTS)).strip()


def detect_lang(q: str, default: str = "es") -> str:
    """Heuristic language detection (Spanish / English)."""
    low = norm(q)
    en_words = {"what", "how", "when", "where", "who", "why", "which", "does", "will",
                "would", "could", "should", "the", "is", "are", "forecast", "gauge",
                "tariff", "margin", "price", "customer", "happens", "affect"}
    es_words = {"qué", "que", "cómo", "como", "cuándo", "cuando", "dónde", "donde",
                "quién", "quien", "cuál", "cual", "pronóstico", "pronostico", "calibre",
                "toneladas", "margen", "arancel", "precio", "cliente", "pasa", "afecta"}
    words = set(re.findall(r"[a-z]{3,}", low))
    en_score = len(words & en_words)
    es_score = len(words & es_words)
    if en_score > es_score:
        return "en"
    if es_score > en_score:
        return "es"
    return default


# --------------------------------------------------------------------------- entity types
@dataclass
class Ents:
    gauges: list[str] = field(default_factory=list)
    market: str | None = None          # None, MKT_NAC, MKT_USA
    horizon: int | None = None         # forecast horizon in months
    month: int | None = None           # specific month number 1-12, or -1 for "next"
    metals: list[str] = field(default_factory=list)   # "cu", "al"
    both_metals: bool = False
    tariff: bool = False
    tariff_zero: bool = False          # "exenta", "cero arancel"
    pct: float | None = None           # percentage extracted (e.g. 10.0)
    pct_abs: bool = False              # True if pct is absolute (e.g. "50%"), False if delta
    direction: int = 1                 # +1 sube/rises, -1 baja/falls
    signals: list[str] = field(default_factory=list)
    extra_order: bool = False
    tons: float | None = None
    months_n: int | None = None        # number of months for extra order


@dataclass
class Parse:
    text: str
    lang: str
    intent: str | None
    ents: Ents
    missing: list[str] = field(default_factory=list)   # slot names needed


# --------------------------------------------------------------------------- gauge extraction
_GAUGE_NUMS = {"6", "10", "12", "14", "16", "18"}
_GAUGE_PAT = re.compile(
    r"\b(?:calibre?s?\s*)?(?:awg\s*)?"
    r"(6|10|12|14|16|18)\b"
    r"(?:\s*(?:y|and|\+|,)\s*(6|10|12|14|16|18)\b)*"
    , re.I
)
_ALL_GAUGES_PAT = re.compile(r"\b(todos(?:\s+los)?\s+calibres|all\s+gauges)\b", re.I)


def _extract_gauges(low: str) -> list[str]:
    if _ALL_GAUGES_PAT.search(low):
        return []
    found = []
    for m in _GAUGE_PAT.finditer(low):
        for g in m.groups():
            if g and g in _GAUGE_NUMS and g not in found:
                found.append(g)
    return found


# --------------------------------------------------------------------------- market extraction
_USA_PAT = re.compile(r"\b(eeuu|ee\.?\s?uu|usa|us\b|estados\s+unidos|united\s+states|exporta\w*|export\w*)\b", re.I)
_NAC_PAT = re.compile(r"\b(nacional|domestic|mexico|mexic\w+|local|domestico)\b", re.I)


def _extract_market(low: str) -> str | None:
    if _USA_PAT.search(low):
        return MKT_USA
    if _NAC_PAT.search(low):
        return MKT_NAC
    return None


# --------------------------------------------------------------------------- metal extraction
_CU_PAT = re.compile(r"\b(cobre|copper|cu\b|hg=f|comex\s+cobre)\b", re.I)
_AL_PAT = re.compile(r"\b(alumin\w+|aluminum|aluminium|al\b|ali=f)\b", re.I)


def _extract_metals(low: str) -> tuple[list[str], bool]:
    metals = []
    if _CU_PAT.search(low):
        metals.append("cu")
    if _AL_PAT.search(low):
        metals.append("al")
    both = bool(re.search(r"\b(ambos\s+metales?|both\s+metals?|metal(es)?)\b", low)) or len(metals) == 2
    return metals, both


# --------------------------------------------------------------------------- percentage extraction
_PCT_PAT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_DIR_UP = re.compile(r"\b(sube?|suba|subi\w+|alza|aument\w+|rises?|rises|increases?|up\b|higher)\b", re.I)
_DIR_DN = re.compile(r"\b(baja?|baje|cae|caig\w+|cay\w+|disminuy\w+|falls?|drops?|down\b|lower)\b", re.I)


def _extract_pct(low: str) -> tuple[float | None, int, bool]:
    m = _PCT_PAT.search(low)
    pct = float(m.group(1)) if m else None
    direction = -1 if _DIR_DN.search(low) else 1
    pct_abs = bool(m and re.search(r"(a|hasta|at|to)\s+\d+\s*%", low))
    return pct, direction, pct_abs


# --------------------------------------------------------------------------- horizon extraction
_HORIZON_PAT = re.compile(r"(\d+)\s*(?:meses?|months?)", re.I)
_NEXT_PAT = re.compile(r"\b(próxim\w+|siguiente|next)\b", re.I)


def _extract_horizon(low: str) -> int | None:
    m = _HORIZON_PAT.search(low)
    if m:
        h = int(m.group(1))
        return h if 1 <= h <= 36 else None
    return None


# --------------------------------------------------------------------------- month extraction
_MONTH_NAMES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _extract_month(low: str) -> int | None:
    if re.search(r"\b(próximo mes|next month|mes que viene)\b", low):
        return -1
    for name, num in _MONTH_NAMES.items():
        if re.search(r"\b" + name + r"\b", low):
            return num
    return None


# --------------------------------------------------------------------------- tariff extraction
_TARIFF_PAT = re.compile(r"\b(arancel\w*|tarif\w*|seccion\s+232|section\s+232|232)\b", re.I)
_TARIFF_ZERO = re.compile(r"\b(exen\w+|cero|zero|0\s*%|sin\s+arancel|tariff.?free|no\s+tariff|exempt\w*|t-mec|usmca)\b", re.I)


# --------------------------------------------------------------------------- signal extraction
_SIGNAL_MAP = {
    "permiso": "Permisos de construcción",
    "construccion": "Permisos de construcción",
    "cfe": "Licitaciones CFE",
    "licitacion": "Licitaciones CFE",
    "nave": "Naves industriales",
    "industrial": "Naves industriales",
    "data center": "Data centers",
    "nearshoring": "Naves industriales",
    "pmi": "PMI manufacturero México",
    "vivienda eeuu": "Inicios de vivienda EE. UU.",
    "housing": "Inicios de vivienda EE. UU.",
}


def _extract_signals(low: str) -> list[str]:
    found = []
    for kw, sig in _SIGNAL_MAP.items():
        if kw in low and sig not in found:
            found.append(sig)
    return found


# --------------------------------------------------------------------------- extra order extraction
_ORDER_PAT = re.compile(r"\b(pedido\s+adicional|pedido\s+extra|extra\s+order|orden\s+adicional)\b", re.I)
_TONS_PAT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:t\b|ton\w*)")


def _extract_order(low: str) -> tuple[bool, float | None]:
    is_order = bool(_ORDER_PAT.search(low))
    m = _TONS_PAT.search(low)
    tons = float(m.group(1)) if m else None
    return is_order, tons


# --------------------------------------------------------------------------- intent rules
# All patterns match against accent-stripped lowercase (norm'd) text.
# Order matters: more specific / higher-priority rules first.
_INTENT_RULES: list[tuple[re.Pattern, str]] = [
    # whatif — must come before forecast to catch "qué pasa si" and conditional "si X sube/baja"
    (re.compile(r"\b(que\s+pasa\s+si|what\s+(if|happens\s+if)|escenario\b|scenario\b|simul\w+|efecto\s+de\b|si\s+el\s+(cobre|aluminio|arancel|tipo\s+de\s+cambio|precio)\s+(sube|baja|cambia|aumenta|disminuye|llega))\b"), "whatif"),
    # why — before forecast; accent-free: cambio/subio/bajo
    (re.compile(r"\b(por\s+que\s+(cambio|subio|bajo|vario|crecio|cambiar)|why\s+(did|has|is)\s+(the\s+)?(total\s+)?forecast|cambio\s+el\s+pronostico|movio\s+el\s+pronostico)\b"), "why"),
    # alerts — specific phrases
    (re.compile(r"\b(alertas?\b|que\s+atender|que\s+debo\s+atender|what\s+(to\s+)?attend|urgent\w*)\b"), "alerts"),
    # prices — live prices
    (re.compile(r"\b(precio\w*\s+(del\s+)?(cobre|aluminio|metal|en\s+vivo)|live\s+price|tipo\s+de\s+cambio|exchange\s+rate|usd.?mxn|mxn.?usd)\b"), "prices"),
    # news
    (re.compile(r"\b(notic\w+|regulacion\w*|federal\s+register|dof\b)\b"), "news"),
    # plan — metal purchasing or production plan
    (re.compile(r"\b(plan\s+de\s+(produc|compra)|production\s+plan|metal\s+purchase|inventario\s+de\s+seguridad|safety\s+stock|compra\s+(de\s+metal|hoy)|cuanto\s+(metal|cobre|aluminio)\s+(tengo\s+que|hay\s+que|debo)\s+comprar|how\s+much\s+metal)\b"), "plan"),
    # customers — require explicit customer-buying signal, not just "cliente"
    (re.compile(r"\b(compra\w*\s+menos|comprando\s+menos|buying\s+less|clientes?\s+(que|esta\w+)\s+compra\w*|which\s+customers?\s+(are|have|reduced)|cartera\s+de\s+clientes|quienes\s+compran)\b"), "customers"),
    # summary
    (re.compile(r"\b(resumen\s+(del\s+mes|mensual|para\s+la\s+junta|para\s+la\s+reunion|s.?op)|monthly\s+summary|junta\s+de\s+s.?op|prepare\s+the\s+summary)\b"), "summary"),
    # explain — SHAP/model explanation; before accuracy/forecast
    (re.compile(r"\b(por\s+que\s+ese\s+numero|shap\b|contribuci\w+\s+del\s+modelo|explica\s+(la\s+)?(predicci\w+|el\s+pronostico\s+de|por\s+que)|feature\s+importance|por\s+que\s+el\s+calibre\s+\d+\s+tiene\s+ese|why\s+does\s+the\s+(model|xgboost))\b"), "explain"),
    # accuracy — before forecast to avoid forecast stealing "preciso"
    (re.compile(r"\b(precision\w*|exactitud|accuracy|mape\b|wape\b|que\s+tan\s+(preciso|exacto)|how\s+accurate|error\s+de\s+validaci\w+|muestra\s+(la\s+)?precision)\b"), "accuracy"),
    # risk
    (re.compile(r"\b(riesgo\b|risk\b|var\b|margen\s+en\s+riesgo|margin\s+at\s+risk|peor\s+(5\s*%|caso)|worst\s+(5\s*%|case)|cobertura\b|hedge\b|montecarlo|monte\s+carlo|cuanto\s+puede\s+(caer|bajar)\s+el\s+margen|probabilidad\s+de\s+que\s+el\s+margen)\b"), "risk"),
    # help
    (re.compile(r"\b(que\s+puedes\s+(hacer|contestar|responder|decir)|que\s+sabes\s+(hacer|responder)|ayuda\b|help\b|what\s+can\s+you|capabilities\b)\b"), "help"),
    # smalltalk
    (re.compile(r"\b(hola\b|hello\b|buenos\s+(dias|tardes|noches)|hi\b|hey\b|gracias\b|thanks\b|thank\s+you)\b"), "smalltalk"),
    # forecast — broad, comes last; "vender" and "ventas" added
    (re.compile(r"\b(pronostic\w+|forecast\w*|cuanto\s+(vamos\s+a\s+vender|se\s+espera|venderemos)|vamos\s+a\s+vender|cuanto\s+venderemos|how\s+much\s+(will|do)\s+we\s+sell|ventas\s+esperadas|expected\s+sales|venta\s+esperada)\b"), "forecast"),
]


def _match_intent(low: str) -> str | None:
    # low is already norm'd (accent-free, lowercase)
    for pat, intent in _INTENT_RULES:
        if pat.search(low):
            return intent
    return None


# --------------------------------------------------------------------------- missing slots
def _whatif_missing(ents: Ents) -> list[str]:
    missing = []
    no_lever = not ents.metals and not ents.both_metals and not ents.tariff and not ents.signals and not ents.extra_order
    if no_lever:
        missing.append("lever")
    elif (ents.metals or ents.tariff) and ents.pct is None and not ents.extra_order:
        missing.append("pct")
    if ents.extra_order:
        if not ents.gauges:
            missing.append("gauge")
        elif ents.tons is None:
            missing.append("tons")
        elif ents.months_n is None:
            missing.append("months_n")
    return missing


# --------------------------------------------------------------------------- public API
def extract(q: str) -> Ents:
    """Extract entities from a raw question string."""
    low = norm(q)
    gauges = _extract_gauges(low)
    market = _extract_market(low)
    metals, both = _extract_metals(low)
    pct, direction, pct_abs = _extract_pct(low)
    horizon = _extract_horizon(low)
    month = _extract_month(low)
    tariff = bool(_TARIFF_PAT.search(low))
    tariff_zero = bool(_TARIFF_ZERO.search(low)) and tariff
    signals = _extract_signals(low)
    is_order, tons = _extract_order(low)

    # try to extract months_n from context like "durante 3 meses"
    m_dur = re.search(r"durante\s+(\d+)\s*(?:meses?|months?)", low)
    months_n = int(m_dur.group(1)) if m_dur else None

    return Ents(
        gauges=gauges, market=market, horizon=horizon, month=month,
        metals=metals, both_metals=both, tariff=tariff, tariff_zero=tariff_zero,
        pct=pct, pct_abs=pct_abs, direction=direction,
        signals=signals, extra_order=is_order, tons=tons, months_n=months_n,
    )


def parse(q: str) -> Parse:
    """Parse a question into intent + entities."""
    lang = detect_lang(q)
    low = norm(q)
    ents = extract(q)
    intent = _match_intent(low)
    missing: list[str] = []
    if intent == "whatif":
        missing = _whatif_missing(ents)
    return Parse(text=q, lang=lang, intent=intent, ents=ents, missing=missing)


def merge_pending(prev: Parse, q: str, customers: list[str] | None = None) -> Parse | None:
    """Try to complete a pending parse (prev has missing slots) with new input.

    Returns a new Parse with the slot filled in, or None if no merge applies.
    """
    if not prev or not prev.missing:
        return None
    slot = prev.missing[0]
    low = norm(q)
    e = prev.ents

    if slot == "lever":
        if _CU_PAT.search(low):
            e.metals = ["cu"]
        elif _AL_PAT.search(low):
            e.metals = ["al"]
        elif _TARIFF_PAT.search(low):
            e.tariff = True
        elif _ORDER_PAT.search(low):
            e.extra_order = True
        elif re.search(r"\b(senal|señal|signal)\b", low):
            sigs = _extract_signals(low)
            if sigs:
                e.signals = sigs
        else:
            return None
        new_missing = _whatif_missing(e)
        return Parse(text=q, lang=prev.lang, intent=prev.intent, ents=e, missing=new_missing)

    if slot == "pct":
        pct, direction, pct_abs = _extract_pct(low)
        if pct is None:
            m = re.search(r"(\d+(?:\.\d+)?)", low)
            pct = float(m.group(1)) if m else None
        if pct is None:
            return None
        e.pct = pct
        e.direction = direction
        e.pct_abs = pct_abs
        new_missing = _whatif_missing(e)
        return Parse(text=q, lang=prev.lang, intent=prev.intent, ents=e, missing=new_missing)

    if slot == "gauge":
        gauges = _extract_gauges(low)
        if not gauges:
            m = re.search(r"\b(6|10|12|14|16|18)\b", low)
            if m:
                gauges = [m.group(1)]
        if not gauges:
            return None
        e.gauges = gauges
        new_missing = _whatif_missing(e)
        return Parse(text=q, lang=prev.lang, intent=prev.intent, ents=e, missing=new_missing)

    if slot == "tons":
        m = _TONS_PAT.search(low) or re.search(r"(\d+(?:\.\d+)?)", low)
        if not m:
            return None
        e.tons = float(m.group(1))
        new_missing = _whatif_missing(e)
        return Parse(text=q, lang=prev.lang, intent=prev.intent, ents=e, missing=new_missing)

    if slot == "months_n":
        m = re.search(r"(\d+)", low)
        if not m:
            return None
        e.months_n = int(m.group(1))
        new_missing = _whatif_missing(e)
        return Parse(text=q, lang=prev.lang, intent=prev.intent, ents=e, missing=new_missing)

    return None


# --------------------------------------------------------------------------- example questions and suggestions
EXAMPLES: dict[str, list[str]] = {
    "forecast": [
        "¿Cuánto vamos a vender en los próximos 6 meses?",
        "How much will we sell in the next 6 months?",
    ],
    "whatif": [
        "¿Qué pasa si el cobre sube 10%?",
        "What happens if copper rises 10%?",
    ],
    "why": [
        "¿Por qué cambió el pronóstico del calibre 6?",
        "Why did the forecast for gauge 6 change?",
    ],
    "alerts": [
        "¿Qué debo atender hoy?",
        "What should I pay attention to today?",
    ],
    "prices": [
        "¿Cuál es el precio del cobre ahora?",
        "What is the current copper price?",
    ],
    "news": [
        "¿Qué dice la noticia más relevante?",
        "What does the most relevant news item say?",
    ],
    "plan": [
        "¿Cuánto metal tengo que comprar?",
        "How much metal do I need to buy?",
    ],
    "customers": [
        "¿Qué clientes compran menos?",
        "Which customers are buying less?",
    ],
    "summary": [
        "Prepara el resumen para la junta de S&OP",
        "Prepare the summary for the S&OP meeting",
    ],
    "accuracy": [
        "¿Qué tan preciso es el pronóstico?",
        "How accurate is the forecast?",
    ],
    "risk": [
        "¿Cuál es el margen en riesgo?",
        "What is the margin at risk?",
    ],
    "explain": [
        "¿Por qué ese número en el pronóstico del calibre 10?",
        "Why does the XGBoost forecast gauge 10 that way?",
    ],
}

_FALLBACK_SUGGESTIONS = [
    "¿Cuánto vamos a vender en los próximos 6 meses?",
    "¿Qué pasa si el cobre sube 10%?",
    "¿Qué debo atender hoy?",
]


def suggestions(p: Parse | None, n: int = 3) -> list[str]:
    """Return up to n suggested follow-up questions based on a partial parse."""
    if p is None:
        return _FALLBACK_SUGGESTIONS[:n]
    lang = p.lang if p else "es"
    idx = 1 if lang == "en" else 0
    # Suggest different intents from what was just asked
    exclude = {p.intent} if p and p.intent else set()
    pool = [v[idx] for k, v in EXAMPLES.items() if k not in exclude]
    return pool[:n]
