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


# ── File reading ──────────────────────────────────────────────────────────────

def read_table(name: str, content: bytes, sheet=0) -> "pd.DataFrame":
    """Read an Excel or CSV file from bytes and return a DataFrame.

    `sheet` can be a sheet index (int) or sheet name (str); ignored for CSV.
    """
    import pandas as pd

    file_kind = kind_of(name)
    buf = io.BytesIO(content)

    if file_kind == "excel":
        try:
            df = pd.read_excel(buf, sheet_name=sheet, header=0, engine="openpyxl")
        except Exception:
            buf.seek(0)
            df = pd.read_excel(buf, sheet_name=0, header=0, engine="openpyxl")
    else:
        # Try common CSV encodings
        for enc in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                buf.seek(0)
                df = pd.read_csv(buf, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            buf.seek(0)
            df = pd.read_csv(buf, encoding="latin-1", errors="replace")

    # Strip leading/trailing whitespace from column names
    df.columns = [str(c).strip() for c in df.columns]
    return df


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


# ── Column normalization (light stub) ─────────────────────────────────────────

# Canonical column names expected by data.py
CANON: dict[str, str] = {
    "date":  "Fecha / Date",
    "qty":   "Cantidad / Tons",
    "cal":   "Calibre / Gauge",
    "prod":  "Producto / Product",
    "seg":   "Segmento / Segment",
    "fam":   "Familia / Family",
}


def normalize_sales(df: "pd.DataFrame") -> tuple:
    """Best-effort pass-through: return (df, []) when the DataFrame looks like
    sales data, or (None, [missing_keys]) when required columns are absent.

    The full engine version does heavy column-name matching; this version
    returns the raw DataFrame unchanged and lets data.py's auto-detect handle it.
    """
    import pandas as pd

    if df is None or df.empty:
        return None, list(CANON.keys())
    return df, []


def guess_columns(df: "pd.DataFrame") -> dict:
    """Guess which columns hold calibre and price data (for PDF price lists)."""
    cols = list(df.columns)
    cal_guess = next((c for c in cols if "cal" in str(c).lower() or "gauge" in str(c).lower()), cols[0] if cols else None)
    pr_guess  = next((c for c in cols if any(k in str(c).lower() for k in ("precio", "price", "mxn", "usd", "valor"))), cols[-1] if cols else None)
    return {"cal": cal_guess, "price": pr_guess}


def prices_from_table(df: "pd.DataFrame", cal_col: str, price_col: str) -> dict:
    """Extract {calibre: price} from a price-list DataFrame."""
    import pandas as pd

    if cal_col not in df.columns or price_col not in df.columns:
        return {}
    result = {}
    for _, row in df.iterrows():
        cal = str(row[cal_col]).strip()
        try:
            result[cal] = float(str(row[price_col]).replace(",", "").replace("$", "").strip())
        except (ValueError, TypeError):
            pass
    return result


# ── Stubs for functions that require the full engine ─────────────────────────

def _unavailable(*_a, **_kw):
    raise NotImplementedError("Motor no disponible en este equipo.")


def pdf_tables(content: bytes) -> tuple:
    """Extract tables from a PDF. Requires pdfplumber (full engine only)."""
    try:
        import pdfplumber
    except ImportError:
        return [], 0

    tables = []
    no_table_pages = 0
    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                page_tables = page.extract_tables()
                if not page_tables:
                    if page.extract_text():
                        no_table_pages += 1
                    continue
                for j, raw in enumerate(page_tables, 1):
                    import pandas as pd
                    if not raw or len(raw) < 2:
                        continue
                    header = [str(c).strip() if c else f"Col{k}" for k, c in enumerate(raw[0])]
                    rows = [[str(cell).strip() if cell else "" for cell in r] for r in raw[1:]]
                    df = pd.DataFrame(rows, columns=header)
                    tables.append({"pagina": i, "numero": j, "tabla": df})
    except Exception:
        pass
    return tables, no_table_pages


combine         = _unavailable   # legacy alias — use combine_sales
read_pdf        = _unavailable
normalize       = _unavailable
quality_summary = _unavailable   # defined in app.py directly; not called via F
