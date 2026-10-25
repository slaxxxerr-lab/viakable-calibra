"""
Tests for viakable_forecast/db.py — schema migrations, save/load round-trips,
atomicity and migration idempotency.
"""
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from viakable_forecast import db as DB
from viakable_forecast.db import (
    _MONTHLY_COLS, _MIGRATIONS, _CURRENT_VERSION,
    init_db, migrate_db, save_prep, load_prep, clear_all, has_data, schema_info,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

def _tmp_db() -> Path:
    """Return a path to a fresh temp file (caller must delete)."""
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    Path(f.name).unlink()   # start empty
    return Path(f.name)


def _make_monthly(n_months: int = 6, gauges=("6", "10", "12")) -> pd.DataFrame:
    months = pd.date_range("2024-01-01", periods=n_months, freq="MS")
    rows = []
    rng = np.random.default_rng(42)
    for m in months:
        for cal in gauges:
            fam = "AL BT Cal 6" if cal == "6" else f"CU BT Cal {cal}"
            rows.append({
                "ym": m, "cal": cal, "seg": "CONS", "ch": "VK",
                "cli": "Cliente A", "fam": fam, "prod": f"P{cal}",
                "prod_name": f"Cable {cal} AWG", "mkt_raw": "MX",
                "mkt": "Nacional",
                "metal": "Aluminio" if cal == "6" else "Cobre",
                "qty": round(float(rng.integers(50, 300)), 2),
            })
    return pd.DataFrame(rows)


# ── test_empty_db ─────────────────────────────────────────────────────────────

def test_empty_db_initialises():
    """Starting from an empty database creates all tables at current schema version."""
    path = _tmp_db()
    try:
        applied = init_db(path)
        info = schema_info(path)
        assert info["version"] == _CURRENT_VERSION
        assert "metal" in info["monthly_columns"]
        assert set(info["monthly_columns"]) >= set(_MONTHLY_COLS)
    finally:
        path.unlink(missing_ok=True)


# ── test_migration_from_v1 ────────────────────────────────────────────────────

def _create_v1_db(path: Path) -> None:
    """Create a database with the OLD schema (no metal column) and seed some rows."""
    con = sqlite3.connect(str(path))
    con.executescript("""
        CREATE TABLE schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT ''
        );
        INSERT INTO schema_version VALUES (1, '2026-10-01T00:00:00', 'Base schema');

        CREATE TABLE uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT '',
            upload_time TEXT NOT NULL,
            n_rows INTEGER NOT NULL DEFAULT 0,
            n_months INTEGER NOT NULL DEFAULT 0,
            gauges TEXT NOT NULL DEFAULT '[]',
            tons REAL NOT NULL DEFAULT 0,
            notes TEXT NOT NULL DEFAULT '[]',
            data_id TEXT NOT NULL DEFAULT '',
            active INTEGER NOT NULL DEFAULT 1
        );
        INSERT INTO uploads VALUES (1,'old.xlsx','Subir',
            '2026-10-01T00:00:00',12,2,'["10","12"]',1000,'[]','did_v1',1);

        CREATE TABLE monthly (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            upload_id INTEGER NOT NULL,
            ym TEXT NOT NULL,
            cal TEXT NOT NULL DEFAULT '',
            seg TEXT NOT NULL DEFAULT '',
            ch TEXT NOT NULL DEFAULT '',
            cli TEXT NOT NULL DEFAULT '',
            fam TEXT NOT NULL DEFAULT '',
            prod TEXT NOT NULL DEFAULT '',
            prod_name TEXT NOT NULL DEFAULT '',
            mkt_raw TEXT NOT NULL DEFAULT '',
            mkt TEXT NOT NULL DEFAULT '',
            qty REAL NOT NULL DEFAULT 0
        );
        INSERT INTO monthly (upload_id, ym, cal, fam, qty)
            VALUES (1,'2024-01-01','10','CU BT Cal 10',500.0);
        INSERT INTO monthly (upload_id, ym, cal, fam, qty)
            VALUES (1,'2024-01-01','12','CU BT Cal 12',500.0);

        CREATE TABLE app_state (key TEXT PRIMARY KEY, value TEXT NOT NULL DEFAULT '');
    """)
    con.commit()
    con.close()


def test_migration_from_v1_adds_metal_column():
    """DB with v1 schema (no metal) migrates to current version automatically."""
    path = _tmp_db()
    try:
        _create_v1_db(path)

        # Verify old schema before migration
        con = sqlite3.connect(str(path))
        old_cols = [r[1] for r in con.execute("PRAGMA table_info(monthly)").fetchall()]
        con.close()
        assert "metal" not in old_cols, "Precondition: old DB should not have metal column"

        # Run migration
        applied = migrate_db(path)
        assert any("v2" in a for a in applied), f"Expected v2 migration; got: {applied}"

        # Verify new schema
        info = schema_info(path)
        assert info["version"] == _CURRENT_VERSION
        assert "metal" in info["monthly_columns"]

        # Verify existing rows got metal derived from fam
        con = sqlite3.connect(str(path))
        rows = con.execute("SELECT cal, fam, metal FROM monthly").fetchall()
        con.close()
        for cal, fam, metal in rows:
            assert metal in ("Cobre", "Aluminio"), f"Unexpected metal value: {metal!r}"
            if "AL" in fam.upper():
                assert metal == "Aluminio"
            else:
                assert metal == "Cobre"
    finally:
        path.unlink(missing_ok=True)


def test_migration_keeps_existing_rows():
    """After migration, previously inserted rows are intact."""
    path = _tmp_db()
    try:
        _create_v1_db(path)
        migrate_db(path)

        con = sqlite3.connect(str(path))
        n = con.execute("SELECT COUNT(*) FROM monthly").fetchone()[0]
        total = con.execute("SELECT SUM(qty) FROM monthly").fetchone()[0]
        con.close()
        assert n == 2, f"Expected 2 rows, got {n}"
        assert abs(total - 1000.0) < 1e-6, f"Expected 1000t, got {total}"
    finally:
        path.unlink(missing_ok=True)


def test_migration_idempotent():
    """Running migrate_db() twice changes nothing the second time."""
    path = _tmp_db()
    try:
        applied_1 = migrate_db(path)
        applied_2 = migrate_db(path)
        assert applied_2 == [], f"Second migration should be no-op, got: {applied_2}"
        info = schema_info(path)
        assert info["version"] == _CURRENT_VERSION
    finally:
        path.unlink(missing_ok=True)


# ── test_save_load ────────────────────────────────────────────────────────────

def test_save_and_load_round_trip():
    """save_prep → load_prep returns the same DataFrame."""
    path = _tmp_db()
    try:
        monthly = _make_monthly(12)
        uid = save_prep("test.xlsx", "Subir", monthly, ["nota1"], "did_test", path)
        assert uid > 0

        result = load_prep(path)
        assert result is not None
        df, notes, did, name = result
        assert len(df) == len(monthly)
        assert abs(df["qty"].sum() - monthly["qty"].sum()) < 0.1
        assert notes == ["nota1"]
        assert did == "did_test"
        assert "metal" in df.columns
    finally:
        path.unlink(missing_ok=True)


def test_metal_column_correctly_derived():
    """AWG-6 aluminum family rows get metal='Aluminio', copper rows get 'Cobre'."""
    path = _tmp_db()
    try:
        monthly = _make_monthly(6, ("6", "10", "12"))
        save_prep("x.xlsx", "S", monthly, [], "d1", path)
        df, *_ = load_prep(path)
        al_rows = df[df["cal"] == "6"]
        cu_rows = df[df["cal"].isin(["10", "12"])]
        assert (al_rows["metal"] == "Aluminio").all(), "AWG-6 should be Aluminio"
        assert (cu_rows["metal"] == "Cobre").all(), "AWG-10/12 should be Cobre"
    finally:
        path.unlink(missing_ok=True)


def test_save_is_atomic_no_partial_rows():
    """If the monthly bulk insert fails, neither the upload row nor any monthly rows remain."""
    path = _tmp_db()
    try:
        init_db(path)
        monthly = _make_monthly(3)

        # Install a trigger that fires on every INSERT to monthly and issues
        # RAISE(ROLLBACK) — this rolls back the entire transaction, including the
        # upload row that was already inserted earlier in the same _conn() block.
        con = sqlite3.connect(str(path))
        con.execute("""
            CREATE TRIGGER monthly_always_fail
            BEFORE INSERT ON monthly
            BEGIN
                SELECT RAISE(ROLLBACK, 'Simulated failure during monthly insert');
            END
        """)
        con.commit()
        con.close()

        with pytest.raises(Exception):
            save_prep("x.xlsx", "S", monthly, [], "bad", path)

        # Both tables must be empty — the transaction should have been rolled back
        con = sqlite3.connect(str(path))
        n_up = con.execute("SELECT COUNT(*) FROM uploads").fetchone()[0]
        n_mo = con.execute("SELECT COUNT(*) FROM monthly").fetchone()[0]
        con.close()
        assert n_up == 0, f"Orphan upload row after failed save: {n_up}"
        assert n_mo == 0, f"Partial monthly rows after failed save: {n_mo}"
    finally:
        path.unlink(missing_ok=True)


def test_load_returns_none_on_empty_db():
    """load_prep() returns None when there are no saved uploads."""
    path = _tmp_db()
    try:
        result = load_prep(path)
        assert result is None
    finally:
        path.unlink(missing_ok=True)


def test_clear_all_removes_data():
    """clear_all() leaves the tables empty."""
    path = _tmp_db()
    try:
        save_prep("x.xlsx", "S", _make_monthly(6), [], "d1", path)
        assert has_data(path)
        clear_all(path)
        assert not has_data(path)
        assert load_prep(path) is None
    finally:
        path.unlink(missing_ok=True)


def test_extra_dataframe_columns_are_ignored():
    """Columns in the DataFrame that aren't in the schema don't cause an error."""
    path = _tmp_db()
    try:
        monthly = _make_monthly(3)
        monthly["extra_column"] = "should_be_ignored"
        uid = save_prep("x.xlsx", "S", monthly, [], "d1", path)
        assert uid > 0
        df, *_ = load_prep(path)
        assert "extra_column" not in df.columns
    finally:
        path.unlink(missing_ok=True)


def test_migration_from_v1_then_save():
    """After migrating from v1 schema, a new upload succeeds."""
    path = _tmp_db()
    try:
        _create_v1_db(path)
        migrate_db(path)
        monthly = _make_monthly(6)
        uid = save_prep("new.xlsx", "Subir", monthly, [], "did_new", path)
        assert uid > 0
        df, _, did, _ = load_prep(path)
        assert did == "did_new"
        assert "metal" in df.columns
    finally:
        path.unlink(missing_ok=True)
