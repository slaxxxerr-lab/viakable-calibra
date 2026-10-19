"""market_data.py — live metal prices and FX via Yahoo Finance (free, no API key).

Sources:
  Copper  : HG=F  (COMEX, USD/lb)  × 2,204.62  → USD/metric ton
  Aluminum: ALI=F (COMEX, USD/t)
  USD/MXN : MXN=X (Yahoo FX)

All network calls use requests (already a Streamlit dependency).
Results are cached to market.db (SQLite) in the project root.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    import requests as _req
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

# ── Public constants ──────────────────────────────────────────────────────────
SERIES = ("cu", "al", "fx")

_TICKERS: dict[str, str] = {
    "cu": "HG=F",   # Copper futures USD/lb
    "al": "ALI=F",  # Aluminum futures USD/t
    "fx": "MXN=X",  # USD/MXN
}
_CONV: dict[str, float] = {
    "cu": 2204.62,  # lb → metric ton
    "al": 1.0,
    "fx": 1.0,
}
_LABELS: dict[str, str] = {"cu": "Cobre", "al": "Aluminio", "fx": "USD/MXN"}

_DB_PATH = Path(__file__).parent.parent / "market.db"
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
_TIMEOUT = 8


# ── Quote dataclass ───────────────────────────────────────────────────────────
@dataclass
class Quote:
    series:     str   = ""
    value:      float = float("nan")
    day_pct:    object = None      # float or None  (daily % change)
    change_abs: str   = ""
    ts:         str   = ""
    status:     str   = "none"     # "ok" | "stale" | "none"
    errors:     list  = field(default_factory=list)
    source:     str   = "Yahoo Finance"

    # back-compat alias
    @property
    def change_pct(self):
        return self.day_pct


# ── SQLite helpers ────────────────────────────────────────────────────────────
def _init_db() -> None:
    con = sqlite3.connect(_DB_PATH)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS quotes (
            series TEXT PRIMARY KEY,
            value  REAL,
            chg    REAL,
            ts     TEXT
        );
        CREATE TABLE IF NOT EXISTS daily (
            series TEXT,
            date   TEXT,
            value  REAL,
            PRIMARY KEY (series, date)
        );
        CREATE TABLE IF NOT EXISTS monthly (
            series TEXT,
            ym     TEXT,
            value  REAL,
            PRIMARY KEY (series, ym)
        );
    """)
    con.commit()
    con.close()


def _save_quote(series: str, value: float, chg: float | None, ts: str) -> None:
    con = sqlite3.connect(_DB_PATH)
    con.execute("INSERT OR REPLACE INTO quotes VALUES (?,?,?,?)", (series, value, chg, ts))
    con.commit()
    con.close()


def _save_daily(series: str, rows: list[tuple[str, float]]) -> None:
    if not rows:
        return
    con = sqlite3.connect(_DB_PATH)
    con.executemany("INSERT OR IGNORE INTO daily VALUES (?,?,?)",
                    [(series, d, v) for d, v in rows])
    con.commit()
    con.close()


# ── Yahoo Finance fetch ───────────────────────────────────────────────────────
def _yahoo_quote(symbol: str) -> tuple[float, float | None, list[tuple[str, float]]]:
    """Return (current, change_pct, daily_rows) from Yahoo Finance v8 API."""
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
           f"?interval=1d&range=90d")
    r = _req.get(url, headers=_HEADERS, timeout=_TIMEOUT)
    r.raise_for_status()
    data = r.json()["chart"]["result"][0]

    meta    = data.get("meta", {})
    ts_list = data.get("timestamp", [])
    closes  = data["indicators"]["quote"][0].get("close", [])

    # build daily rows
    daily_rows: list[tuple[str, float]] = []
    for ts, c in zip(ts_list, closes):
        if c is None:
            continue
        date_str = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
        daily_rows.append((date_str, c))

    # current price: prefer regularMarketPrice from meta, else last close
    current = meta.get("regularMarketPrice")
    if current is None:
        valid = [c for c in closes if c is not None]
        if not valid:
            raise ValueError(f"No closes for {symbol}")
        current = valid[-1]

    prev_close = meta.get("chartPreviousClose") or meta.get("previousClose")
    chg = ((current - prev_close) / prev_close * 100) if prev_close else None

    return float(current), chg, daily_rows


# ── Public API ────────────────────────────────────────────────────────────────
def fetch_all(_token=None) -> dict[str, Quote]:
    """Fetch live quotes for all series. Falls back to cached values on error."""
    if not _HAS_REQUESTS:
        return {s: Quote(series=s, errors=["requests not installed"]) for s in SERIES}

    try:
        _init_db()
    except Exception:
        pass

    cached = load_quotes()
    quotes: dict[str, Quote] = {}

    for s in SERIES:
        ticker = _TICKERS[s]
        conv   = _CONV[s]
        try:
            raw, chg, daily = _yahoo_quote(ticker)
            val = raw * conv
            ts  = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
            try:
                _save_quote(s, val, chg, ts)
                _save_daily(s, [(d, v * conv) for d, v in daily])
            except Exception:
                pass
            quotes[s] = Quote(
                series=s, value=val, day_pct=chg,
                change_abs=f"{chg:+.1f}%" if chg is not None else "",
                ts=ts, status="ok", source="Yahoo Finance",
            )
        except Exception as exc:
            fallback = cached.get(s) or Quote(series=s)
            fallback.status = "stale"
            fallback.errors = [str(exc)]
            quotes[s] = fallback

    return quotes


def load_quotes() -> dict[str, Quote]:
    """Return last saved quotes from SQLite."""
    try:
        _init_db()
        con  = sqlite3.connect(_DB_PATH)
        rows = con.execute("SELECT series, value, chg, ts FROM quotes").fetchall()
        con.close()
        return {
            r[0]: Quote(series=r[0], value=r[1] or float("nan"),
                        day_pct=r[2], ts=r[3] or "", status="stale")
            for r in rows
        }
    except Exception:
        return {}


def save_quotes(_q: dict) -> None:
    pass  # individual saves happen inside fetch_all


def ensure_history(_token=None) -> dict:
    """Fetch 90-day daily history on first call (best-effort)."""
    try:
        _init_db()
        con = sqlite3.connect(_DB_PATH)
        n   = con.execute("SELECT COUNT(*) FROM daily").fetchone()[0]
        con.close()
        if n > 0:
            return {"rows": n}
        # fetch history for all series
        for s in SERIES:
            try:
                _, _, daily = _yahoo_quote(_TICKERS[s])
                _save_daily(s, [(d, v * _CONV[s]) for d, v in daily])
            except Exception:
                pass
    except Exception:
        pass
    return {}


def upgrade_monthly(_fred_key=None, _av_key=None) -> None:
    """Aggregate daily history to monthly averages (background thread)."""
    try:
        _init_db()
        con = sqlite3.connect(_DB_PATH)
        for s in SERIES:
            rows = con.execute(
                "SELECT strftime('%Y-%m', date) as ym, AVG(value) FROM daily "
                "WHERE series=? GROUP BY ym", (s,)
            ).fetchall()
            if rows:
                con.executemany(
                    "INSERT OR REPLACE INTO monthly VALUES (?,?,?)",
                    [(s, ym, v) for ym, v in rows],
                )
        con.commit()
        con.close()
    except Exception:
        pass


def load_daily(series: str) -> pd.DataFrame:
    """Return daily history DataFrame with columns ['date', 'value'] (date as Timestamp)."""
    try:
        _init_db()
        con = sqlite3.connect(_DB_PATH)
        df  = pd.read_sql(
            "SELECT date, value FROM daily WHERE series=? ORDER BY date",
            con, params=(series,),
        )
        con.close()
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception:
        return pd.DataFrame(columns=["date", "value"])


def load_monthly(series: str = "") -> dict:
    """Return {series: pd.Series(value, index=ym)} for each series."""
    try:
        _init_db()
        con = sqlite3.connect(_DB_PATH)
        out: dict = {}
        for s in ([series] if series else SERIES):
            rows = con.execute(
                "SELECT ym, value FROM monthly WHERE series=? ORDER BY ym", (s,)
            ).fetchall()
            if rows:
                ym_list, v_list = zip(*rows)
                out[s] = pd.Series(list(v_list), index=list(ym_list), name=s)
        con.close()
        return out
    except Exception:
        return {}


def alerts(quotes: dict, daily: dict) -> list:
    """Return price-alert dicts for moves ≥3% daily or ≥8% over 30 days.

    Each dict: series ("cu"|"al"), pct (float), period ("día"|"mes"), level ("alta"|"media").
    """
    out: list[dict] = []
    for s in ("cu", "al"):
        q = (quotes or {}).get(s)
        if q and q.status == "ok" and q.day_pct is not None and abs(q.day_pct) >= 3.0:
            out.append({"series": s, "pct": q.day_pct, "period": "día", "level": "alta"})
    for s in ("cu", "al"):
        df = (daily or {}).get(s)
        try:
            if df is None or len(df) < 22:
                continue
            recent    = float(df["value"].iloc[-1])
            month_ago = float(df["value"].iloc[-22])
            if month_ago and abs(recent / month_ago - 1) * 100 >= 8:
                chg = (recent / month_ago - 1) * 100
                out.append({"series": s, "pct": chg, "period": "mes", "level": "media"})
        except Exception:
            pass
    return out


def say(msg: str) -> None:
    pass
