"""news.py — News & regulation monitor.

Sources:
  1. Google News RSS  (no key, rate-limited to 14 queries / 30 min)
  2. Federal Register (no key, 4 targeted queries)
  3. GDELT            (no key, optional, 1 query, 5-s timeout, 1-h back-off on 429)

Items are stored in a SQLite database (news.db, git-ignored).
Only title, source, date, link, summary and metadata are saved — no article text.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import sqlite3
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any

import feedparser
import requests

# ── constants ────────────────────────────────────────────────────────────────
TTL_MINUTES = 30
_DB_PATH = Path(__file__).parent.parent / "news.db"
_lock = threading.Lock()

TOPICS = [
    "Arancel / T-MEC",
    "Normativa México",
    "Precios de metales",
    "Construcción y vivienda",
    "Industria y nearshoring",
    "Competencia",
    "Otros",
]

STATUSES = ["nuevo", "relevante", "revisado", "ignorado"]

# short description per topic (used by app.py as WHY[topic])
WHY: dict[str, str] = {
    "Arancel / T-MEC": "Posible cambio en aranceles o T-MEC que podría afectar exportaciones.",
    "Normativa México": "Regulación mexicana que podría impactar la operación o demanda.",
    "Precios de metales": "Movimiento en precios de cobre o aluminio que afecta márgenes.",
    "Construcción y vivienda": "Cambio en permisos o actividad de construcción que influye en la demanda.",
    "Industria y nearshoring": "Inversión industrial o nearshoring que impulsa la demanda de cable.",
    "Competencia": "Movimiento de competidores que puede impactar participación de mercado.",
    "Otros": "Noticia con posible relevancia indirecta.",
}

# ── Google News RSS queries (Spanish + English, `when:30d`) ──────────────────
_GN_QUERIES = [
    # Tariffs / T-MEC
    ("arancel cobre aluminio México", "Arancel / T-MEC"),
    ("tariff copper aluminum Mexico", "Arancel / T-MEC"),
    ("T-MEC USMCA cable eléctrico", "Arancel / T-MEC"),
    ("Section 232 copper aluminum", "Arancel / T-MEC"),
    # Mexican regulation
    ("NOM cable eléctrico México ANCE", "Normativa México"),
    ("regulación eléctrica CFE México", "Normativa México"),
    # Metal prices
    ("precio cobre LME aluminio", "Precios de metales"),
    ("copper price LME aluminum", "Precios de metales"),
    # Construction
    ("permisos construcción vivienda México", "Construcción y vivienda"),
    ("CFE licitación proyecto eléctrico", "Construcción y vivienda"),
    # Industry / nearshoring
    ("nearshoring manufactura México inversión", "Industria y nearshoring"),
    ("data center México inversión", "Industria y nearshoring"),
    # Competition
    ("cable eléctrico fabricante México competencia", "Competencia"),
    # General cable/wire market
    ("cable conductor eléctrico México demanda", "Otros"),
]

# ── Federal Register queries ─────────────────────────────────────────────────
_FR_QUERIES = [
    # Section 232 notices from Commerce/BIS
    {"conditions[term]": "Section 232 copper", "conditions[agencies][]": "commerce-department", "per_page": "5"},
    {"conditions[term]": "Section 232 aluminum", "conditions[agencies][]": "commerce-department", "per_page": "5"},
    # USTR USMCA notices
    {"conditions[term]": "USMCA copper aluminum", "conditions[agencies][]": "trade-representative", "per_page": "5"},
    # Presidential documents on copper/aluminum
    {"conditions[term]": "copper aluminum tariff", "conditions[type][]": "PRESDOCU", "per_page": "5"},
]
_FR_BASE = "https://www.federalregister.gov/api/v1/documents.json"

# ── keyword scoring ──────────────────────────────────────────────────────────
_TOPIC_KEYWORDS: dict[str, list[tuple[str, int]]] = {
    "Arancel / T-MEC": [
        ("arancel", 4), ("tariff", 4), ("232", 3), ("t-mec", 3), ("usmca", 3),
        ("sanción", 2), ("sanction", 2), ("importaci", 2), ("export", 2),
    ],
    "Normativa México": [
        ("nom-", 4), ("ance", 3), ("cofece", 3), ("cre ", 3), ("sener", 3),
        ("regulaci", 2), ("norma", 2), ("dof", 2),
    ],
    "Precios de metales": [
        ("lme", 4), ("comex", 3), ("cobre", 3), ("copper", 3), ("alumini", 3),
        ("aluminum", 3), ("metal price", 3), ("precio metal", 2), ("commodity", 2),
    ],
    "Construcción y vivienda": [
        ("vivienda", 3), ("permiso", 3), ("housing", 3), ("construcci", 3),
        ("cfe", 2), ("licitaci", 2), ("infraestructura", 2),
    ],
    "Industria y nearshoring": [
        ("nearshoring", 4), ("manufactura", 3), ("data center", 3), ("centro de datos", 3),
        ("inversion", 2), ("planta industrial", 2), ("nave industrial", 2), ("parque industrial", 2),
    ],
    "Competencia": [
        ("competenci", 3), ("competitor", 3), ("market share", 2), ("participaci", 2),
        ("rival", 2), ("imported cable", 2), ("cable import", 2),
    ],
}

_HIGH_RELEVANCE_WORDS = {"arancel", "tariff", "232", "t-mec", "usmca", "lme", "comex",
                          "cobre", "copper", "alumini", "aluminum", "cfe", "nom-", "ance"}
_MEDIUM_WORDS = {"vivienda", "housing", "construcci", "nearshoring", "manufactura",
                 "data center", "permiso", "licitaci"}

_IMPACT_TAGS = [
    ("exportaciones", ["export", "tariff", "232", "usmca", "t-mec"]),
    ("calibres de edificación", ["vivienda", "housing", "construcci", "permiso"]),
    ("aluminio calibre 6", ["alumini", "aluminum"]),
    ("costos de metal", ["lme", "comex", "precio", "price", "cobre", "copper"]),
    ("demanda nacional", ["cfe", "construcci", "manufactura", "nearshoring"]),
]

_ACTION_MAP = {
    "Arancel / T-MEC": {
        "text": "Revisar el escenario de arancel y su impacto en las exportaciones.",
        "sim": {"tariff": "stress"},
    },
    "Precios de metales": {
        "text": "Revisar el escenario de precio del metal y la recomendación de compra en Metales.",
        "sim": {"metal": 10},
    },
    "Construcción y vivienda": {
        "text": "Ajustar la señal de permisos de construcción en el modelo de señales.",
        "sim": {"signal": ("Permisos de construcción", 5)},
    },
    "Industria y nearshoring": {
        "text": "Ajustar la señal de naves industriales o centros de datos.",
        "sim": {"signal": ("Naves industriales", 5)},
    },
    "Normativa México": {
        "text": "Verificar si la normativa afecta calibres específicos o requisitos de certificación.",
        "sim": None,
    },
    "Competencia": {
        "text": "Revisar precios y participación de mercado por calibre.",
        "sim": None,
    },
    "Otros": {
        "text": "Evaluar relevancia para el negocio.",
        "sim": None,
    },
}

# ── database ─────────────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def _init_db():
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS items (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                source TEXT,
                published TEXT,
                url TEXT,
                summary TEXT,
                topic TEXT,
                relevance TEXT,
                status TEXT DEFAULT 'nuevo',
                tags TEXT DEFAULT '[]',
                action TEXT,
                provider TEXT,
                also INTEGER DEFAULT 0,
                fetched TEXT
            );
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                val TEXT
            );
            CREATE TABLE IF NOT EXISTS changelog (
                ts TEXT,
                who TEXT,
                action TEXT,
                detail TEXT
            );
        """)


_init_db()

# ── helpers ───────────────────────────────────────────────────────────────────

def _item_id(url: str, title: str) -> str:
    return hashlib.sha1(f"{url}|{title[:80]}".encode()).hexdigest()[:16]


def _lc(s: str) -> str:
    return (s or "").lower()


def _classify_topic(title: str, summary: str) -> str:
    text = _lc(title) + " " + _lc(summary)
    scores: dict[str, int] = {}
    for topic, kws in _TOPIC_KEYWORDS.items():
        s = sum(w for kw, w in kws if kw in text)
        if s:
            scores[topic] = s
    if not scores:
        return "Otros"
    return max(scores, key=lambda t: scores[t])


def _score_relevance(title: str, summary: str, topic: str, provider: str, published: str) -> str:
    text = _lc(title) + " " + _lc(summary)
    score = 0
    for w in _HIGH_RELEVANCE_WORDS:
        if w in text:
            score += 2
    for w in _MEDIUM_WORDS:
        if w in text:
            score += 1
    if provider == "federal_register":
        score += 3
    if topic in ("Arancel / T-MEC", "Precios de metales"):
        score += 2
    # recency bonus
    try:
        age_days = (dt.datetime.now() - dt.datetime.fromisoformat(published)).days
        if age_days <= 3:
            score += 2
        elif age_days <= 7:
            score += 1
    except Exception:
        pass
    # penalty for clearly unrelated tariff noise
    if "tariff" in text and not any(w in text for w in ("copper", "aluminum", "cobre", "alumini", "cable", "metal")):
        score -= 3
    if score >= 6:
        return "Alta"
    if score >= 3:
        return "Media"
    return "Baja"


def _impact_tags(title: str, summary: str) -> list[str]:
    text = _lc(title) + " " + _lc(summary)
    return [tag for tag, kws in _IMPACT_TAGS if any(k in text for k in kws)]


def _parse_date(s: str | None) -> str:
    """Return ISO date string (YYYY-MM-DDTHH:MM:SS) or today."""
    if not s:
        return dt.datetime.now().isoformat(timespec="seconds")
    for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return dt.datetime.strptime(s, fmt).replace(tzinfo=None).isoformat(timespec="seconds")
        except ValueError:
            pass
    # feedparser gives struct_time
    try:
        import calendar
        return dt.datetime(*s[:6]).isoformat(timespec="seconds")
    except Exception:
        pass
    return dt.datetime.now().isoformat(timespec="seconds")


def _near_dup(title: str, existing_titles: list[str], window_days: int = 5) -> bool:
    """True if a very similar title is already in the last N days."""
    tl = re.sub(r"\W+", " ", _lc(title)).strip()
    words = set(tl.split())
    for et in existing_titles:
        el = re.sub(r"\W+", " ", _lc(et)).strip()
        ewords = set(el.split())
        if words and ewords:
            overlap = len(words & ewords) / max(len(words), len(ewords))
            if overlap > 0.7:
                return True
    return False


def _save_item(item: dict, existing_titles: list[str]) -> bool:
    """Insert item if not duplicate. Returns True if saved."""
    if _near_dup(item["title"], existing_titles):
        return False
    try:
        with _conn() as c:
            cur = c.execute(
                "INSERT OR IGNORE INTO items (id,title,source,published,url,summary,topic,relevance,status,tags,action,provider,also,fetched) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (item["id"], item["title"], item["source"], item["published"], item["url"],
                 item["summary"], item["topic"], item["relevance"], "nuevo",
                 json.dumps(item["tags"]), json.dumps(item["action"]), item["provider"], 0,
                 dt.datetime.now().isoformat(timespec="seconds")),
            )
            return cur.rowcount > 0
    except Exception:
        return False


# ── Google News RSS ───────────────────────────────────────────────────────────

def _fetch_google_news(query: str, default_topic: str, days: int = 30) -> list[dict]:
    items = []
    url = (
        "https://news.google.com/rss/search?"
        + urllib.parse.urlencode({"q": f"{query} when:{days}d", "hl": "es", "gl": "MX", "ceid": "MX:es"})
    )
    try:
        feed = feedparser.parse(url)
        for e in feed.entries[:8]:
            title = e.get("title", "").strip()
            if not title:
                continue
            link = e.get("link", "").strip()
            src = e.get("source", {}).get("title", "") or e.get("author", "") or "Google News"
            pub = _parse_date(e.get("published") or (e.get("published_parsed") and time.strftime("%a, %d %b %Y %H:%M:%S +0000", e.published_parsed)))
            summary = re.sub(r"<[^>]+>", "", e.get("summary", "")).strip()[:300]
            topic = _classify_topic(title, summary) if default_topic == "Otros" else default_topic
            relevance = _score_relevance(title, summary, topic, "google_news", pub)
            tags = _impact_tags(title, summary)
            action = _ACTION_MAP.get(topic)
            items.append({
                "id": _item_id(link, title),
                "title": title,
                "source": src,
                "published": pub,
                "url": link,
                "summary": summary,
                "topic": topic,
                "relevance": relevance,
                "tags": tags,
                "action": action,
                "provider": "google_news",
            })
    except Exception:
        pass
    return items


# ── Federal Register ──────────────────────────────────────────────────────────

def _fetch_federal_register() -> list[dict]:
    items = []
    for params in _FR_QUERIES:
        try:
            r = requests.get(_FR_BASE, params={**params, "fields[]": ["title", "abstract", "publication_date", "html_url", "agencies"]},
                             timeout=10)
            if r.status_code != 200:
                continue
            data = r.json()
            for doc in data.get("results", [])[:5]:
                title = doc.get("title", "").strip()
                if not title:
                    continue
                link = doc.get("html_url", "")
                pub_str = doc.get("publication_date", "")
                pub = _parse_date(pub_str + "T00:00:00" if pub_str else None)
                summary = (doc.get("abstract") or "")[:300].strip()
                # filter out Foreign-Trade Zone noise
                if "foreign-trade zone" in _lc(title) or "foreign trade zone" in _lc(title):
                    continue
                topic = _classify_topic(title, summary)
                relevance = _score_relevance(title, summary, topic, "federal_register", pub)
                tags = _impact_tags(title, summary)
                action = _ACTION_MAP.get(topic)
                items.append({
                    "id": _item_id(link, title),
                    "title": title,
                    "source": "Federal Register",
                    "published": pub,
                    "url": link,
                    "summary": summary,
                    "topic": topic,
                    "relevance": relevance,
                    "tags": tags,
                    "action": action,
                    "provider": "federal_register",
                })
            time.sleep(0.5)
        except Exception:
            pass
    return items


# ── GDELT ─────────────────────────────────────────────────────────────────────

_GDELT_BACKOFF_UNTIL: float = 0.0


def _fetch_gdelt() -> list[dict]:
    global _GDELT_BACKOFF_UNTIL
    if time.time() < _GDELT_BACKOFF_UNTIL:
        return []
    try:
        r = requests.get(
            "https://api.gdeltproject.org/api/v2/doc/doc",
            params={"query": "cable eléctrico cobre aluminio México arancel", "mode": "artlist",
                    "maxrecords": "10", "timespan": "30d", "format": "json"},
            timeout=5,
        )
        if r.status_code == 429:
            _GDELT_BACKOFF_UNTIL = time.time() + 3600
            return []
        data = r.json()
        items = []
        for art in data.get("articles", [])[:8]:
            title = art.get("title", "").strip()
            link = art.get("url", "")
            if not title or not link:
                continue
            pub = _parse_date(art.get("seendate", ""))
            summary = ""
            topic = _classify_topic(title, summary)
            relevance = _score_relevance(title, summary, topic, "gdelt", pub)
            tags = _impact_tags(title, summary)
            action = _ACTION_MAP.get(topic)
            items.append({
                "id": _item_id(link, title),
                "title": title,
                "source": art.get("domain", "GDELT"),
                "published": pub,
                "url": link,
                "summary": summary,
                "topic": topic,
                "relevance": relevance,
                "tags": tags,
                "action": action,
                "provider": "gdelt",
            })
        return items
    except Exception:
        return []


# ── public API ────────────────────────────────────────────────────────────────

def meta_get(key: str):
    try:
        row = _conn().execute("SELECT val FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None
    except Exception:
        return None


def _meta_set(key: str, val: str):
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO meta(key,val) VALUES(?,?)", (key, val))


def refresh(force: bool = False) -> None:
    """Fetch from all sources and persist new items."""
    with _lock:
        status: dict[str, Any] = {}
        now_iso = dt.datetime.now().isoformat(timespec="seconds")

        # collect existing titles from last 5 days to detect near-dups
        cutoff = (dt.datetime.now() - dt.timedelta(days=5)).isoformat()
        try:
            rows = _conn().execute("SELECT title FROM items WHERE published >= ?", (cutoff,)).fetchall()
            existing_titles = [r[0] for r in rows]
        except Exception:
            existing_titles = []

        all_items: list[dict] = []

        # Google News — staggered to respect rate limits
        gn_ok = gn_fail = 0
        for query, default_topic in _GN_QUERIES:
            try:
                items = _fetch_google_news(query, default_topic)
                all_items.extend(items)
                gn_ok += 1
            except Exception:
                gn_fail += 1
            time.sleep(1.2)  # ~14 queries staggered over ~17 s
        status["Google News"] = {"ok": gn_ok, "fail": gn_fail, "error": ""}

        # Federal Register
        fr_ok = fr_fail = 0
        try:
            fr_items = _fetch_federal_register()
            all_items.extend(fr_items)
            fr_ok = len(_FR_QUERIES)
        except Exception as e:
            fr_fail = len(_FR_QUERIES)
            status["Federal Register"] = {"ok": 0, "fail": fr_fail, "error": str(e)[:120]}
        if "Federal Register" not in status:
            status["Federal Register"] = {"ok": fr_ok, "fail": fr_fail, "error": ""}

        # GDELT (optional)
        try:
            gdelt_items = _fetch_gdelt()
            all_items.extend(gdelt_items)
            status["GDELT"] = {"ok": len(gdelt_items), "fail": 0, "error": ""}
        except Exception as e:
            status["GDELT"] = {"ok": 0, "fail": 1, "error": str(e)[:120]}

        # persist new items
        for item in all_items:
            _save_item(item, existing_titles)
            if item["title"] not in existing_titles:
                existing_titles.append(item["title"])

        _meta_set("last_fetch", now_iso)
        _meta_set("status", json.dumps(status))


def load(limit: int = 100, days: int = 30, topics=None,
         relevance=None, statuses=None, text: str = "") -> list[dict]:
    try:
        cutoff = (dt.datetime.now() - dt.timedelta(days=days)).isoformat()
        sql = "SELECT * FROM items WHERE published >= ?"
        params: list[Any] = [cutoff]
        if topics:
            sql += f" AND topic IN ({','.join('?'*len(topics))})"
            params.extend(topics)
        if relevance:
            sql += f" AND relevance IN ({','.join('?'*len(relevance))})"
            params.extend(relevance)
        if statuses:
            sql += f" AND status IN ({','.join('?'*len(statuses))})"
            params.extend(statuses)
        if text:
            sql += " AND (title LIKE ? OR summary LIKE ?)"
            params += [f"%{text}%", f"%{text}%"]
        sql += " ORDER BY published DESC LIMIT ?"
        params.append(limit)
        rows = _conn().execute(sql, params).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["tags"] = json.loads(d.get("tags") or "[]")
            d["action"] = json.loads(d.get("action") or "null")
            out.append(d)
        return out
    except Exception:
        return []


def set_status(item_id: str, status: str) -> None:
    try:
        with _conn() as c:
            c.execute("UPDATE items SET status=? WHERE id=?", (status, item_id))
    except Exception:
        pass


def log(action: str, item_id: str | None = None, detail: str = "") -> None:
    try:
        with _conn() as c:
            c.execute("INSERT INTO changelog(ts,who,action,detail) VALUES(?,?,?,?)",
                      (dt.datetime.now().isoformat(timespec="seconds"), "app", action, detail[:200]))
    except Exception:
        pass


def changelog(limit: int = 50) -> list[dict]:
    try:
        rows = _conn().execute("SELECT ts,who,action,detail FROM changelog ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []
