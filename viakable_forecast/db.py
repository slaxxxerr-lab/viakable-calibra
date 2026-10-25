"""db.py — SQLite persistence for Viakable Calibra.

Saves/loads the processed monthly sales DataFrame so data survives
page refreshes without re-uploading the file.

Tables:
  uploads   — one row per uploaded dataset (metadata)
  monthly   — monthly aggregated rows (the Prepared.monthly DataFrame)
  app_state — key/value pairs (data_id, source name, etc.)
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from contextlib import contextmanager

import pandas as pd

logger = logging.getLogger(__name__)

# ── Database path ─────────────────────────────────────────────────────────────
# Relative to the app root. On Streamlit Cloud this persists across page
# refreshes within the same container but resets on redeploy.
_DB_DIR = Path(__file__).parent.parent / "data"
_DB_PATH = _DB_DIR / "calibra.db"

# Monthly DataFrame columns (must match data.prepare() output)
_MONTHLY_COLS = ["ym", "cal", "seg", "ch", "cli", "fam", "prod", "prod_name", "mkt_raw", "mkt", "qty"]

# ── Schema ────────────────────────────────────────────────────────────────────
_SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    source      TEXT    NOT NULL DEFAULT '',
    upload_time TEXT    NOT NULL,
    n_rows      INTEGER NOT NULL DEFAULT 0,
    n_months    INTEGER NOT NULL DEFAULT 0,
    gauges      TEXT    NOT NULL DEFAULT '[]',
    tons        REAL    NOT NULL DEFAULT 0,
    notes       TEXT    NOT NULL DEFAULT '[]',
    data_id     TEXT    NOT NULL DEFAULT '',
    active      INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS monthly (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    upload_id   INTEGER NOT NULL REFERENCES uploads(id) ON DELETE CASCADE,
    ym          TEXT    NOT NULL,
    cal         TEXT    NOT NULL DEFAULT '',
    seg         TEXT    NOT NULL DEFAULT '',
    ch          TEXT    NOT NULL DEFAULT '',
    cli         TEXT    NOT NULL DEFAULT '',
    fam         TEXT    NOT NULL DEFAULT '',
    prod        TEXT    NOT NULL DEFAULT '',
    prod_name   TEXT    NOT NULL DEFAULT '',
    mkt_raw     TEXT    NOT NULL DEFAULT '',
    mkt         TEXT    NOT NULL DEFAULT '',
    qty         REAL    NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_monthly_upload ON monthly(upload_id);

CREATE TABLE IF NOT EXISTS app_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL DEFAULT ''
);
"""


# ── Connection helper ─────────────────────────────────────────────────────────

def _ensure_dir() -> None:
    _DB_DIR.mkdir(parents=True, exist_ok=True)


@contextmanager
def _conn():
    _ensure_dir()
    con = sqlite3.connect(str(_DB_PATH), timeout=10)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init_db() -> None:
    """Create tables if they don't exist."""
    with _conn() as con:
        con.executescript(_SCHEMA)


# ── Save ──────────────────────────────────────────────────────────────────────

def save_prep(
    name: str,
    source: str,
    monthly: pd.DataFrame,
    notes: list[str],
    data_id: str,
) -> int:
    """Persist a processed monthly DataFrame. Returns the new upload_id."""
    init_db()
    n_months = int(monthly["ym"].nunique()) if "ym" in monthly.columns else 0
    gauges = sorted(monthly["cal"].unique().tolist()) if "cal" in monthly.columns else []
    tons = float(monthly["qty"].sum()) if "qty" in monthly.columns else 0.0
    n_rows = len(monthly)
    import datetime
    upload_time = datetime.datetime.now().isoformat(timespec="seconds")

    with _conn() as con:
        cur = con.execute(
            """INSERT INTO uploads (name, source, upload_time, n_rows, n_months, gauges, tons, notes, data_id, active)
               VALUES (?,?,?,?,?,?,?,?,?,1)""",
            (name, source, upload_time, n_rows, n_months,
             json.dumps(gauges), tons, json.dumps(notes), data_id),
        )
        upload_id = cur.lastrowid

        # Build rows for monthly table
        m = monthly.copy()
        m["ym"] = m["ym"].astype(str)
        for col in _MONTHLY_COLS:
            if col not in m.columns:
                m[col] = "" if col != "qty" else 0.0
        rows = [
            (upload_id,) + tuple(str(r[c]) if c != "qty" else float(r[c]) for c in _MONTHLY_COLS)
            for _, r in m[_MONTHLY_COLS].iterrows()
        ]
        con.executemany(
            f"INSERT INTO monthly (upload_id, {', '.join(_MONTHLY_COLS)}) VALUES ({', '.join(['?']*(len(_MONTHLY_COLS)+1))})",
            rows,
        )
        # Mark all other uploads as inactive
        con.execute("UPDATE uploads SET active=0 WHERE id != ?", (upload_id,))
        # Save data_id to app_state
        con.execute("INSERT OR REPLACE INTO app_state(key,value) VALUES ('data_id',?)", (data_id,))
        con.execute("INSERT OR REPLACE INTO app_state(key,value) VALUES ('source',?)", (source,))
        con.execute("INSERT OR REPLACE INTO app_state(key,value) VALUES ('name',?)", (name,))

    logger.info("Saved upload %d: %s (%d rows, %d months)", upload_id, name, n_rows, n_months)
    return upload_id


# ── Load ──────────────────────────────────────────────────────────────────────

def load_prep() -> "tuple[pd.DataFrame, list[str], str, str] | None":
    """
    Load the most recent active dataset from SQLite.
    Returns (monthly_df, notes, data_id, name) or None if nothing is saved.
    """
    try:
        init_db()
        with _conn() as con:
            row = con.execute(
                "SELECT id, name, source, notes, data_id FROM uploads WHERE active=1 ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            upload_id, name, source, notes_json, data_id = row
            notes = json.loads(notes_json) if notes_json else []

            rows = con.execute(
                f"SELECT {', '.join(_MONTHLY_COLS)} FROM monthly WHERE upload_id=?",
                (upload_id,),
            ).fetchall()

        if not rows:
            return None

        monthly = pd.DataFrame(rows, columns=_MONTHLY_COLS)
        monthly["ym"] = pd.to_datetime(monthly["ym"])
        monthly["qty"] = pd.to_numeric(monthly["qty"], errors="coerce").fillna(0.0)
        return monthly, notes, data_id, name
    except Exception as e:
        logger.warning("Could not load from DB: %s", e)
        return None


# ── File management ───────────────────────────────────────────────────────────

def list_uploads() -> list[dict]:
    """Return metadata for all uploads (newest first)."""
    try:
        init_db()
        with _conn() as con:
            rows = con.execute(
                "SELECT id, name, source, upload_time, n_months, gauges, tons, active FROM uploads ORDER BY id DESC"
            ).fetchall()
        result = []
        for r in rows:
            result.append({
                "id": r[0], "name": r[1], "source": r[2], "upload_time": r[3],
                "n_months": r[4], "gauges": json.loads(r[5] or "[]"),
                "tons": r[6], "active": bool(r[7]),
            })
        return result
    except Exception:
        return []


def set_active(upload_id: int) -> None:
    """Make a previous upload the active dataset."""
    with _conn() as con:
        con.execute("UPDATE uploads SET active=0")
        con.execute("UPDATE uploads SET active=1 WHERE id=?", (upload_id,))


def delete_upload(upload_id: int) -> None:
    """Delete an upload and its monthly rows (CASCADE)."""
    with _conn() as con:
        con.execute("DELETE FROM uploads WHERE id=?", (upload_id,))
    logger.info("Deleted upload %d", upload_id)


def clear_all() -> None:
    """Remove all saved data."""
    with _conn() as con:
        con.execute("DELETE FROM monthly")
        con.execute("DELETE FROM uploads")
        con.execute("DELETE FROM app_state")
    logger.info("Cleared all saved data")


def has_data() -> bool:
    """Return True if at least one active upload exists."""
    try:
        init_db()
        with _conn() as con:
            n = con.execute("SELECT COUNT(*) FROM uploads WHERE active=1").fetchone()[0]
        return n > 0
    except Exception:
        return False
