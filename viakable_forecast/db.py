"""db.py — SQLite persistence for Viakable Calibra.

Single source of truth for the database schema. Schema changes are handled
through versioned migrations that run automatically on startup.

Tables
------
schema_version  — tracks which migrations have been applied
uploads         — one row per uploaded dataset (metadata)
monthly         — monthly aggregated rows (Prepared.monthly DataFrame)
app_state       — key/value pairs (data_id, source name, etc.)
"""
from __future__ import annotations

import datetime
import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)

# ── Database path ─────────────────────────────────────────────────────────────
_DB_DIR  = Path(__file__).parent.parent / "data"
_DB_PATH = _DB_DIR / "calibra.db"
_LOG_DIR = Path(__file__).parent.parent / "logs"

# ── Monthly DataFrame column definitions ─────────────────────────────────────
# Each entry: (name, sqlite_type, not_null, default)
# THIS IS THE SINGLE DEFINITION. db.py and only db.py owns this list.
_MONTHLY_SCHEMA: list[tuple[str, str, bool, str | None]] = [
    ("ym",        "TEXT", True,  None),
    ("cal",       "TEXT", True,  "''"),
    ("seg",       "TEXT", True,  "''"),
    ("ch",        "TEXT", True,  "''"),
    ("cli",       "TEXT", True,  "''"),
    ("fam",       "TEXT", True,  "''"),
    ("prod",      "TEXT", True,  "''"),
    ("prod_name", "TEXT", True,  "''"),
    ("mkt_raw",   "TEXT", True,  "''"),
    ("mkt",       "TEXT", True,  "''"),
    ("metal",     "TEXT", True,  "'Cobre'"),
    ("qty",       "REAL", True,  "0"),
]
_MONTHLY_COLS = [col for col, *_ in _MONTHLY_SCHEMA]


# ── Schema version 0: base DDL (tables that existed before migration system) ──

_BASE_DDL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    applied_at  TEXT    NOT NULL,
    description TEXT    NOT NULL DEFAULT ''
);

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

# ── Migration steps ───────────────────────────────────────────────────────────
# Each entry: (version, description, list_of_sql_statements)
# - Use ALTER TABLE ... ADD COLUMN for additive changes (SQLite supports this).
# - For renames/removals/type changes, use the rebuild approach (see _rebuild_table).
# - Statements are executed in order within a single transaction.
# - Fill-in UPDATE statements run after the ALTER so existing rows get values.

_MIGRATIONS: list[tuple[int, str, list[str]]] = [
    (1, "Base schema (uploads, monthly without metal, app_state)", []),
    (2, "Add metal column to monthly; derive from fam for existing rows", [
        "ALTER TABLE monthly ADD COLUMN metal TEXT NOT NULL DEFAULT 'Cobre'",
        # Derive metal for existing rows: families containing 'AL' → Aluminio
        "UPDATE monthly SET metal = 'Aluminio' WHERE lower(fam) LIKE '%al%'",
    ]),
]

_CURRENT_VERSION = max(v for v, *_ in _MIGRATIONS)


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _ensure_dirs() -> None:
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    _LOG_DIR.mkdir(parents=True, exist_ok=True)


def _log_migration(msg: str) -> None:
    logger.info(msg)
    try:
        log_file = _LOG_DIR / "calibra.log"
        with log_file.open("a", encoding="utf-8") as f:
            f.write(f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {msg}\n")
    except Exception:
        pass


@contextmanager
def _conn(path: Path | None = None):
    """Context manager that opens a connection, commits on success, rolls back on error."""
    _ensure_dirs()
    con = sqlite3.connect(str(path or _DB_PATH), timeout=15)
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


def _table_columns(con: sqlite3.Connection, table: str) -> list[str]:
    """Return the list of column names for a table (empty list if table doesn't exist)."""
    rows = con.execute(f"PRAGMA table_info({table})").fetchall()
    return [r[1] for r in rows]


def _schema_version(con: sqlite3.Connection) -> int:
    """Return the current schema version, or 0 if the version table doesn't exist yet."""
    try:
        tbl_exists = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
        ).fetchone()
        if not tbl_exists:
            return 0
        row = con.execute("SELECT MAX(version) FROM schema_version").fetchone()
        return int(row[0]) if row and row[0] is not None else 0
    except Exception:
        return 0


# ── Migration engine ──────────────────────────────────────────────────────────

def migrate_db(db_path: Path | None = None) -> list[str]:
    """
    Apply all pending migrations and return a list of human-readable change descriptions.
    Safe to call multiple times — already-applied migrations are skipped.
    """
    applied: list[str] = []
    _ensure_dirs()
    path = db_path or _DB_PATH

    with _conn(path) as con:
        # Create base schema (idempotent; IF NOT EXISTS guards)
        con.executescript(_BASE_DDL)
        current = _schema_version(con)

        for version, description, stmts in _MIGRATIONS:
            if version <= current:
                continue
            _log_migration(f"Applying migration v{version}: {description}")
            for sql in stmts:
                try:
                    con.execute(sql)
                    _log_migration(f"  OK: {sql[:80]}")
                except sqlite3.OperationalError as e:
                    # "duplicate column name" means the column was already added — skip
                    if "duplicate column" in str(e).lower() or "already exists" in str(e).lower():
                        _log_migration(f"  SKIP (already exists): {sql[:80]}")
                    else:
                        raise
            con.execute(
                "INSERT OR REPLACE INTO schema_version(version, applied_at, description) VALUES (?,?,?)",
                (version, datetime.datetime.now().isoformat(timespec="seconds"), description),
            )
            applied.append(f"v{version}: {description}")
            _log_migration(f"Migration v{version} done.")

    return applied


def init_db(db_path: Path | None = None) -> list[str]:
    """Create tables and apply all pending migrations. Call this before any read/write."""
    return migrate_db(db_path)


# ── Save ──────────────────────────────────────────────────────────────────────

def save_prep(
    name: str,
    source: str,
    monthly: pd.DataFrame,
    notes: list[str],
    data_id: str,
    db_path: Path | None = None,
) -> int:
    """
    Persist a processed monthly DataFrame in a single atomic transaction.
    Returns the new upload_id. Raises on failure (no partial rows are left).
    """
    # Ensure schema is current before writing
    init_db(db_path)

    # Collect metadata
    n_months = int(monthly["ym"].nunique()) if "ym" in monthly.columns else 0
    gauges   = sorted(monthly["cal"].unique().tolist()) if "cal" in monthly.columns else []
    tons     = float(monthly["qty"].sum())              if "qty" in monthly.columns else 0.0
    n_rows   = len(monthly)
    upload_time = datetime.datetime.now().isoformat(timespec="seconds")

    # Prepare monthly rows: only columns defined in _MONTHLY_SCHEMA, in order
    m = monthly.copy()
    m["ym"] = m["ym"].astype(str)

    # Ensure every required column exists; warn about missing ones
    for col, col_type, required, _ in _MONTHLY_SCHEMA:
        if col not in m.columns:
            if required:
                logger.warning("Column '%s' missing from DataFrame; filling with default.", col)
            m[col] = 0.0 if col_type == "REAL" else ""

    # Derive metal from fam if the column is empty (shouldn't happen, but belt-and-suspenders)
    if "metal" in m.columns and m["metal"].eq("").all():
        m["metal"] = m["fam"].str.contains("AL", case=False, na=False).map(
            {True: "Aluminio", False: "Cobre"}
        )

    # Ignore extra columns that aren't in the schema (log them)
    extra = [c for c in m.columns if c not in _MONTHLY_COLS and c != "upload_id"]
    if extra:
        logger.debug("Ignoring extra DataFrame columns (not in DB schema): %s", extra)

    with _conn(db_path or _DB_PATH) as con:
        # 1. Insert upload metadata
        cur = con.execute(
            """INSERT INTO uploads
               (name, source, upload_time, n_rows, n_months, gauges, tons, notes, data_id, active)
               VALUES (?,?,?,?,?,?,?,?,?,1)""",
            (name, source, upload_time, n_rows, n_months,
             json.dumps(gauges), tons, json.dumps(notes), data_id),
        )
        upload_id = cur.lastrowid

        # 2. Insert monthly rows (only schema columns, in schema order)
        col_list = ", ".join(_MONTHLY_COLS)
        placeholders = ", ".join(["?"] * (len(_MONTHLY_COLS) + 1))  # +1 for upload_id
        sql = f"INSERT INTO monthly (upload_id, {col_list}) VALUES ({placeholders})"

        def _val(row, col: str, col_type: str):
            v = row[col]
            if col_type == "REAL":
                return float(v) if v is not None and v == v else 0.0
            return str(v) if v is not None else ""

        rows = [
            (upload_id,) + tuple(_val(r, col, typ) for col, typ, *_ in _MONTHLY_SCHEMA)
            for _, r in m[_MONTHLY_COLS].iterrows()
        ]
        con.executemany(sql, rows)

        # 3. Mark all other uploads as inactive
        con.execute("UPDATE uploads SET active=0 WHERE id != ?", (upload_id,))

        # 4. Update app_state
        for key, val in (("data_id", data_id), ("source", source), ("name", name)):
            con.execute("INSERT OR REPLACE INTO app_state(key,value) VALUES (?,?)", (key, val))

    _log_migration(f"Saved upload {upload_id}: {name!r} ({n_rows} rows, {n_months} months)")
    return upload_id


# ── Load ──────────────────────────────────────────────────────────────────────

def load_prep(db_path: Path | None = None) -> "tuple[pd.DataFrame, list[str], str, str] | None":
    """
    Load the most recent active dataset.
    Returns (monthly_df, notes, data_id, name) or None if nothing is saved.
    """
    try:
        init_db(db_path)
        with _conn(db_path or _DB_PATH) as con:
            row = con.execute(
                "SELECT id, name, source, notes, data_id FROM uploads WHERE active=1 ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            upload_id, name, source, notes_json, data_id = row
            notes = json.loads(notes_json) if notes_json else []

            # Load only columns that actually exist in the DB (handles old schemas)
            db_cols = _table_columns(con, "monthly")
            read_cols = [c for c in _MONTHLY_COLS if c in db_cols]
            missing   = [c for c in _MONTHLY_COLS if c not in db_cols]
            if missing:
                logger.warning("DB monthly table missing columns (will be derived): %s", missing)

            col_sql = ", ".join(read_cols)
            rows = con.execute(
                f"SELECT {col_sql} FROM monthly WHERE upload_id=?", (upload_id,)
            ).fetchall()

        if not rows:
            return None

        monthly = pd.DataFrame(rows, columns=read_cols)
        monthly["ym"]  = pd.to_datetime(monthly["ym"])
        monthly["qty"] = pd.to_numeric(monthly["qty"], errors="coerce").fillna(0.0)

        # Fill derived columns not in the DB
        if "metal" not in monthly.columns or monthly["metal"].eq("").all():
            monthly["metal"] = monthly.get("fam", pd.Series("", index=monthly.index)
                                           ).str.contains("AL", case=False, na=False).map(
                {True: "Aluminio", False: "Cobre"}
            )

        return monthly, notes, data_id, name
    except Exception as e:
        logger.warning("Could not load from DB: %s", e)
        return None


# ── File management ───────────────────────────────────────────────────────────

def list_uploads(db_path: Path | None = None) -> list[dict]:
    """Return metadata for all uploads (newest first)."""
    try:
        init_db(db_path)
        with _conn(db_path or _DB_PATH) as con:
            rows = con.execute(
                "SELECT id, name, source, upload_time, n_months, gauges, tons, active "
                "FROM uploads ORDER BY id DESC"
            ).fetchall()
        return [
            {"id": r[0], "name": r[1], "source": r[2], "upload_time": r[3],
             "n_months": r[4], "gauges": json.loads(r[5] or "[]"),
             "tons": r[6], "active": bool(r[7])}
            for r in rows
        ]
    except Exception:
        return []


def set_active(upload_id: int, db_path: Path | None = None) -> None:
    with _conn(db_path or _DB_PATH) as con:
        con.execute("UPDATE uploads SET active=0")
        con.execute("UPDATE uploads SET active=1 WHERE id=?", (upload_id,))


def delete_upload(upload_id: int, db_path: Path | None = None) -> None:
    with _conn(db_path or _DB_PATH) as con:
        con.execute("DELETE FROM uploads WHERE id=?", (upload_id,))
    _log_migration(f"Deleted upload {upload_id}")


def clear_all(db_path: Path | None = None) -> None:
    with _conn(db_path or _DB_PATH) as con:
        con.execute("DELETE FROM monthly")
        con.execute("DELETE FROM uploads")
        con.execute("DELETE FROM app_state")
    _log_migration("Cleared all saved data")


def has_data(db_path: Path | None = None) -> bool:
    try:
        init_db(db_path)
        with _conn(db_path or _DB_PATH) as con:
            n = con.execute("SELECT COUNT(*) FROM uploads WHERE active=1").fetchone()[0]
        return n > 0
    except Exception:
        return False


def schema_info(db_path: Path | None = None) -> dict:
    """Return current schema version and column info (for diagnostics)."""
    try:
        init_db(db_path)
        with _conn(db_path or _DB_PATH) as con:
            ver = _schema_version(con)
            cols = _table_columns(con, "monthly")
        return {"version": ver, "monthly_columns": cols, "expected_columns": _MONTHLY_COLS}
    except Exception as e:
        return {"error": str(e)}
