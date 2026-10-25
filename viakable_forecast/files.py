"""files.py — file-handling utilities for multi-file upload.

The full engine version lives on the production machine (Mac).
This version implements all functions needed by app.py so the
Streamlit Cloud deployment works without the full engine.
"""
from __future__ import annotations

import io
from pathlib import Path


# ── File type detection ───────────────────────────────────────────────────────

def kind_of(name: str) -> str:
    """Return 'pdf', 'excel', or 'csv' based on the file's extension."""
    ext = Path(name).suffix.lower().lstrip(".")
    if ext == "pdf":
        return "pdf"
    if ext in ("xlsx", "xls"):
        return "excel"
    return "csv"


def sheet_names(content: bytes) -> list[str]:
    """Return the list of sheet names in an Excel file (bytes)."""
    try:
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=True)
        names = wb.sheetnames
        wb.close()
        return names
    except Exception:
        return ["Sheet1"]


# ── Multi-file combination ────────────────────────────────────────────────────

def combine_sales(all_tabs: list) -> tuple:
    """Combine a list of (name, DataFrame) into one deduplicated DataFrame.

    Returns (combined_df, summary_dict, notes_list).
    combined_df is None when no valid DataFrames are supplied.
    """
    import pandas as pd

    dfs = []
    file_stats = []
    for name, df in all_tabs:
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        n_before = len(df)
        dfs.append(df)
        file_stats.append({"file": name, "rows": n_before})

    if not dfs:
        return None, {}, []

    combined = pd.concat(dfs, ignore_index=True)
    n_dupes = combined.duplicated().sum()
    combined = combined.drop_duplicates().reset_index(drop=True)

    summary = {
        "files":  len(dfs),
        "rows":   len(combined),
        "dupes":  int(n_dupes),
        "stats":  file_stats,
    }
    notes: list[str] = []
    if n_dupes:
        notes.append(f"Se eliminaron {n_dupes} filas duplicadas entre archivos.")

    return combined, summary, notes


# ── Stubs for functions that require the full engine ─────────────────────────

def _unavailable(*_a, **_kw):
    raise NotImplementedError("Motor no disponible en este equipo.")


combine         = _unavailable   # legacy alias — use combine_sales
read_pdf        = _unavailable
normalize       = _unavailable
quality_summary = _unavailable   # defined in app.py directly; not called via F
