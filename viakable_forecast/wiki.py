"""Wikipedia lookup for the hybrid assistant. No Streamlit dependency.

Uses the official MediaWiki Action API (free, no key needed).
Searches Spanish Wikipedia first; falls back to English.
"""
from __future__ import annotations

import re
import unicodedata

import requests

_TIMEOUT = 6
_API = {"es": "https://es.wikipedia.org/w/api.php", "en": "https://en.wikipedia.org/w/api.php"}
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _strip(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _first_sentences(text: str, n: int = 3) -> str:
    parts = _SENTENCE.split(text.strip())
    return " ".join(parts[:n]).strip()


def _search(q: str, lang: str) -> str | None:
    """Return the best matching article title or None."""
    try:
        r = requests.get(_API[lang], params={
            "action": "query", "list": "search", "srsearch": q,
            "srnamespace": 0, "srlimit": 3, "format": "json",
        }, timeout=_TIMEOUT)
        r.raise_for_status()
        hits = r.json().get("query", {}).get("search", [])
        return hits[0]["title"] if hits else None
    except Exception:
        return None


def _extract(title: str, lang: str) -> dict | None:
    """Fetch the plain-text extract for an article title."""
    try:
        r = requests.get(_API[lang], params={
            "action": "query", "prop": "extracts|info",
            "titles": title, "exintro": True, "explaintext": True,
            "inprop": "url", "format": "json", "redirects": True,
        }, timeout=_TIMEOUT)
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        page = next(iter(pages.values()))
        if page.get("missing") is not None:
            return None
        extract = _strip(page.get("extract", ""))
        if not extract or len(extract) < 40:
            return None
        url = page.get("fullurl") or f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"
        return {"title": page.get("title", title), "extract": extract, "url": url}
    except Exception:
        return None


def _focus_terms(q: str) -> str:
    """Return content words from the question for relevance scoring."""
    STOP = {
        "qué", "que", "es", "el", "la", "los", "las", "un", "una", "de", "del", "en",
        "con", "por", "para", "como", "se", "cuándo", "cuando", "quién", "quien",
        "cuál", "cual", "dónde", "donde", "cómo", "como", "hay", "tiene", "son",
        "what", "is", "are", "the", "a", "an", "of", "in", "for", "how", "when",
        "who", "where", "which", "was", "were", "has", "have", "does", "do",
    }
    words = re.findall(r"[a-záéíóúñüà-ÿ]{3,}", q.lower())
    return " ".join(w for w in words if w not in STOP)


def lookup(q: str, lang: str = "es") -> dict | None:
    """Search Wikipedia for q.

    Returns dict with keys: title, extract, lead, url, lang, focus.
    Returns None if nothing useful is found.
    Tries the requested language first, then the other one.
    """
    focus = _focus_terms(q)
    search_q = focus if focus else q

    for try_lang in ([lang, "en"] if lang == "es" else ["en", "es"]):
        title = _search(search_q, try_lang)
        if not title:
            continue
        art = _extract(title, try_lang)
        if not art:
            continue
        extract = art["extract"]
        lead = _first_sentences(extract, 3)
        return {
            "title": art["title"],
            "extract": extract,
            "lead": lead,
            "url": art["url"],
            "lang": try_lang,
            "focus": focus,
        }
    return None


def short_answer(text: str, q: str, focus: str | None = None) -> str:
    """Extract the most relevant 1-2 sentences from a Wikipedia extract.

    Used when no LLM is available to rewrite the answer.
    """
    if not text:
        return text
    sentences = [s.strip() for s in _SENTENCE.split(text) if s.strip()]
    if not sentences:
        return text

    terms = set(re.findall(r"[a-záéíóúñüà-ÿ]{3,}", (focus or q).lower()))
    if not terms:
        return " ".join(sentences[:2])

    def score(s: str) -> int:
        low = s.lower()
        return sum(1 for term in terms if term in low)

    ranked = sorted(enumerate(sentences), key=lambda x: (-score(x[1]), x[0]))
    best_idx = ranked[0][0]
    chosen = [sentences[best_idx]]
    if best_idx + 1 < len(sentences) and len(chosen[0]) < 180:
        chosen.append(sentences[best_idx + 1])
    return " ".join(chosen)
