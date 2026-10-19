"""Question router for Asistente Calibra. No Streamlit dependency.

Routes each question to one of five handlers:
  "app"   – business data / forecast / scenario  → app engine via assistant.py
  "kb"    – methodology / app concepts           → knowledge base (kb.py)
  "wiki"  – general factual questions            → Wikipedia (wiki.py)
  "mixed" – business entity + general context   → engine facts + LLM explanation
  "open"  – everything else                      → LLM (llm.py)

Routing uses:
1. Rule-based intent detection (fastest path, from assistant.py)
2. TF-IDF similarity against KB entries and example questions
3. Factual question patterns for "wiki" and "mixed" paths
4. Learned intents from the CRM (assistant learning loop)
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable

from viakable_forecast import assistant as AS
from viakable_forecast import kb as KBM

# --------------------------------------------------------------------------- data structures
@dataclass
class Route:
    kind: str            # "app", "kb", "wiki", "mixed", "open"
    conf: float          # 0.0 – 1.0 confidence
    meta: dict = field(default_factory=dict)
    parse: AS.Parse | None = None      # only for "app" route
    entry: str | None = None           # kb entry key for "kb" route
    reason: str = ""


# --------------------------------------------------------------------------- redaction
_EMAIL_PAT = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_PAT = re.compile(r"\+?[\d][\d\s\-\(\)]{7,}\d")
_URL_PAT = re.compile(r"https?://\S+|www\.\S+")


def redact(text: str) -> str:
    """Remove personal data (emails, phone numbers, URLs) from text."""
    text = _EMAIL_PAT.sub("[email]", text)
    text = _PHONE_PAT.sub("[phone]", text)
    text = _URL_PAT.sub("[url]", text)
    return text


# --------------------------------------------------------------------------- content tokens (for learning)
_STOPWORDS_ES = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "en", "con",
    "por", "para", "que", "qué", "se", "a", "al", "y", "o", "es", "son", "hay",
    "como", "cómo", "si", "no", "me", "mi", "tu", "su", "sus", "lo", "le", "les",
    "muy", "más", "mas", "pero", "este", "esta", "estos", "estas", "ese", "esa",
    "tengo", "tiene", "tienen", "hacer", "hago", "hace", "cuando", "cuándo",
    "donde", "dónde", "quien", "quién", "cual", "cuál", "esto", "eso", "aquí",
}
_STOPWORDS_EN = {
    "the", "a", "an", "of", "in", "for", "and", "or", "is", "are", "was", "were",
    "it", "its", "to", "be", "do", "does", "did", "have", "has", "had", "will",
    "would", "could", "should", "may", "might", "this", "that", "these", "those",
    "i", "you", "we", "he", "she", "they", "me", "him", "her", "us", "them",
    "my", "your", "our", "his", "her", "their", "what", "how", "when", "where",
    "which", "who", "why", "if", "not", "no", "yes", "so", "but",
}


def content_tokens(text: str) -> str:
    """Extract content words for the learning loop."""
    low = AS.norm(text)
    words = re.findall(r"[a-z]{3,}", low)
    stop = _STOPWORDS_ES | _STOPWORDS_EN
    return " ".join(w for w in words if w not in stop)


# --------------------------------------------------------------------------- TF-IDF helpers (lazy, no sklearn required)
def _tf(text: str) -> dict[str, float]:
    words = re.findall(r"[a-z]{3,}", AS.norm(text))
    if not words:
        return {}
    counts: dict[str, int] = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    total = len(words)
    return {w: c / total for w, c in counts.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    shared = set(a) & set(b)
    if not shared:
        return 0.0
    dot = sum(a[w] * b[w] for w in shared)
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


# Pre-compute KB TF vectors once
_KB_VECS: dict[str, dict[str, float]] = {k: _tf(t) for k, t in KBM.ENTRY_TEXTS.items()}
_KB_VECS_EN: dict[str, dict[str, float]] = {k: _tf(t) for k, t in KBM.ENTRY_TEXTS_EN.items()}


def _best_kb_match(q: str, lang: str) -> tuple[str | None, float]:
    vecs = _KB_VECS_EN if lang == "en" else _KB_VECS
    q_vec = _tf(q)
    best_key, best_score = None, 0.0
    for k, v in vecs.items():
        s = _cosine(q_vec, v)
        if s > best_score:
            best_score = s
            best_key = k
    # fast keyword pre-filter boost
    kw_match = KBM.keyword_match(AS.norm(q))
    if kw_match and kw_match != best_key:
        kw_score = _cosine(q_vec, vecs.get(kw_match, {}))
        if kw_score > best_score * 0.7:   # accept keyword match if close
            return kw_match, max(kw_score, best_score * 0.85)
    return best_key, best_score


# --------------------------------------------------------------------------- question-type patterns
# All patterns match against accent-stripped norm'd text. ¿/? are stripped before matching.
_FACTUAL_PAT = re.compile(
    r"(?:^|[.?!]\s*)"                                      # start or after sentence end
    r"(?:que\s+(?:es|son|fue|era|eran|fue\s+el)\b|"
    r"cual\s+(?:es|fue)\b|"
    r"cuando\s+(?:fue|es|se|nacio|inicio|empezo)\b|"
    r"quien\s+(?:es|fue|invento|fundo)\b|"
    r"donde\s+(?:esta|queda|se\s+ubica|se\s+encuentra)\b|"
    r"cuanto\s+(?:mide|vale|pesa|tiene|cuesta|tarda)\b|"
    r"cuantos?\s+(?:habitantes|anos|metros|km|veces)\b|"
    r"what\s+is\b|what\s+are\b|who\s+is\b|when\s+was\b|where\s+is\b|"
    r"how\s+(?:many|much|big|tall|old|far|long|deep)\b)",
)
# Topics that feel factual but are actually opinion/forward-looking → NOT wiki
_FORWARD_PAT = re.compile(r"\b(futuro\b|future\b|tendencia\w*\s+futura\w*|perspectiva\b|forecast\w+\s+for\s+\d{4}|best\s+\w*\s*(practice\w*|tactic\w*|strateg\w*|approach\w*|method\w*))\b")

_MIXED_PAT = re.compile(
    r"\b(recesion\b|recession\b|crisis\b|guerra\s+comercial|trade\s+war|"
    r"desaceleracion|slowdown|huelga\b|strike\b|"
    r"inversion\s+extranjera|foreign\s+investment|construction\s+boom|"
    r"mercado\s+inmobiliario|housing\s+market|economia\b|economy\b|tendencia\s+macro|"
    r"inflacion\b|inflation\b|nearshoring\b|"
    r"impacto\s+(de\s+la\s+)?(inflacion|nearshoring)|"
    r"afecta\w*\s+(la\s+)?(inflacion|nearshoring|recesion|crisis)|"
    r"(inflacion|nearshoring|recesion)\s+afect)\b",
)

_METHODOLOGY_PAT = re.compile(
    r"\b(metodo\w*\s+(de\s+pronostico|funciona)|como\s+(funciona|calcula|se\s+calcula)\s+(el\s+)?pronostic\w+|"
    r"como\s+funciona\s+(la\s+)?(app|calibra|aplicacion|validacion|ensamble|machine|ml)\b|"
    r"como\s+se\s+calcula\b|"
    r"que\s+es\s+(el\s+|la\s+|los\s+|las\s+|un\s+|una\s+)?(mape|wape|ensamble|croston|tsb|sarima|ets\b|holt|validacion\s+cruzada|inventario\s+de\s+seguridad)\b|"
    r"que\s+son\s+(las?\s+)?(senales\s+de\s+mercado|metodos\s+de\s+pronostico|clases\s+de\s+demanda)\b|"
    r"que\s+significa\s+(mape|wape|adi|cv2)\b|"
    r"explica\s+(el\s+|la\s+)?(inventario(\s+de\s+seguridad)?|ensamble|validacion(\s+cruzada)?|machine\s+learning|metodo\w*|pronostico)\b|"
    r"en\s+(calibra|la\s+app|la\s+aplicacion)\b|"
    r"how\s+does.{1,50}work\b|what\s+is\s+(mape|wape)\b|"
    r"explain\s+the\s+method|cross.?validat|validacion\s+cruzada\s+(funciona|trabaja))\b",
)

_OPEN_INDICATORS = re.compile(
    r"\b(recomienda\w*|aconseja\w*|opinas?\b|piensas?\b|crees?\b|cuéntame|"
    r"explica\w*\s+(?:la\s+)?(?:economía|economy|industria|sector)|"
    r"qué\s+(?:estrategia|táctica|accion)\w*|"
    r"recommend|advise|think|believe|opinion|strategy|tactic|best\s+practice)\b",
    re.I,
)

# --------------------------------------------------------------------------- main routing function
_APP_CONFIDENCE_THRESHOLD = 0.55
_KB_CONFIDENCE_THRESHOLD = 0.18
_WIKI_CONFIDENCE_THRESHOLD = 0.20


def route(
    q: str,
    customers: list[str] | None = None,
    lang: str = "es",
    learned: dict | None = None,
    last_intent: str | None = None,
) -> Route:
    """Route a question to the appropriate handler."""
    # Strip leading ¿ and ? for pattern matching
    low = AS.norm(q)
    low_clean = re.sub(r"^[¿¡\s]+", "", low)

    p = AS.parse(q)

    # 1. Learned intents take precedence
    if learned:
        toks = content_tokens(q)
        for phrase, intent in learned.items():
            if phrase and toks and phrase in toks:
                p.intent = intent
                return Route("app", 0.85, {}, p, reason=f"aprendido: {phrase}")

    # 2. Simple "¿Qué es X?" / "What is X?" factual definition → check for methodology first
    is_factual = bool(_FACTUAL_PAT.search(low_clean)) and not p.intent and not _FORWARD_PAT.search(low)
    if is_factual:
        # If it's asking about app methodology specifically, prefer KB
        if _METHODOLOGY_PAT.search(low):
            kb_key, kb_score = _best_kb_match(q, p.lang)
            if kb_key and kb_score >= _KB_CONFIDENCE_THRESHOLD:
                return Route("kb", min(0.9, kb_score + 0.35), {}, None, entry=kb_key, reason="metodología factual")
        return Route("wiki", 0.75, {}, None, reason="pregunta factual")

    # 3. Methodology pattern → KB (overrides weak app intents like forecast/accuracy
    #    triggered by incidental keyword overlap, e.g. "¿Qué es el MAPE?")
    if _METHODOLOGY_PAT.search(low):
        kb_key, kb_score = _best_kb_match(q, p.lang)
        # Lower threshold here: methodology signal is strong, trust KB at 0.10+
        if kb_key and kb_score >= 0.10:
            return Route("kb", min(0.9, kb_score + 0.35), {}, None, entry=kb_key, reason="metodología")

    # 4. Strong business intent
    if p.intent and p.intent not in ("help", "smalltalk"):
        conf = _app_confidence(p, low, customers)
        if conf >= _APP_CONFIDENCE_THRESHOLD:
            return Route("app", conf, {}, p, reason=f"intento: {p.intent}")

    # 5. KB match (without explicit methodology marker)
    kb_key, kb_score = _best_kb_match(q, p.lang)
    if kb_key and kb_score >= _KB_CONFIDENCE_THRESHOLD + 0.08:
        # Don't steal simple "qué es X" questions when confidence is marginal
        if not (_FACTUAL_PAT.search(low_clean) and kb_score < 0.35):
            return Route("kb", kb_score, {}, None, entry=kb_key, reason="base de conocimiento")

    # 6. Mixed: macro/market topic that also touches the business
    if _MIXED_PAT.search(low) and not is_factual:
        return Route("mixed", 0.75, {}, p, reason="tema mixto: macro + negocio")

    # 7. Help / smalltalk → app
    if p.intent in ("help", "smalltalk"):
        return Route("app", 0.7, {}, p, reason=f"intento: {p.intent}")

    # 8. Weak app intent
    if p.intent:
        return Route("app", 0.5, {}, p, reason=f"intento débil: {p.intent}")

    # 9. Factual question without a strong app intent → wiki
    if _FACTUAL_PAT.search(low_clean) and not _FORWARD_PAT.search(low):
        return Route("wiki", 0.70, {}, None, reason="pregunta factual")

    # 10. Fallback: open
    return Route("open", 0.6, {}, None, reason="pregunta abierta")


# --------------------------------------------------------------------------- helpers
def _app_confidence(p: AS.Parse, low: str, customers: list[str] | None) -> float:
    """Compute confidence for routing to the app engine."""
    base = 0.6
    e = p.ents

    # Boost for specific entities
    if e.gauges:
        base += 0.15
    if e.market:
        base += 0.08
    if e.metals or e.both_metals:
        base += 0.12
    if e.tariff:
        base += 0.10
    if e.pct is not None:
        base += 0.05
    if p.intent in ("forecast", "whatif", "plan", "risk", "accuracy"):
        base += 0.10
    if p.intent in ("prices", "alerts", "customers", "summary", "news"):
        base += 0.08

    # Customer name match
    if customers:
        for name in customers:
            if name and len(name) >= 3 and AS.norm(name) in low:
                base += 0.12
                break

    return min(base, 0.98)
