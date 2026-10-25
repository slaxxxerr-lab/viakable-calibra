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

# Column hints that indicate a sheet contains sales data (lowercase)
_SALES_HINTS = {"toneladas", "calibre", "gauge", "date", "fecha",
                "periodo", "cantidad", "qty", "tons"}
# Sheets whose names suggest they are NOT sales data
_SKIP_NAMES  = {"dashboard", "pivot", "pivots", "resumen", "summary",
                "graficas", "gráficas", "charts", "config", "template"}


def best_sheet(content: bytes) -> "str | int":
    """Return the sheet name most likely to contain sales data.

    Scores each sheet by how many of its header cells contain sales-related
    keywords.  Falls back to 0 (first sheet) if openpyxl is unavailable.
    """
    try:
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=True)
        best_name: "str | None" = None
        best_score = -1
        for sname in wb.sheetnames:
            if sname.lower().strip() in _SKIP_NAMES:
                continue
            ws = wb[sname]
            headers: list[str] = []
            for row in ws.iter_rows(min_row=1, max_row=1, values_only=True):
                headers = [str(c).lower().strip() for c in row if c is not None]
                break
            score = sum(1 for h in headers for hint in _SALES_HINTS if hint in h)
            if score > best_score:
                best_score = score
                best_name = sname
        wb.close()
        return best_name if best_name is not None else 0
    except Exception:
        return 0


def read_table(name: str, content: bytes, sheet=0) -> "pd.DataFrame":
    """Read an Excel or CSV file from bytes and return a DataFrame.

    When sheet=0 (the default) and the file has multiple sheets, automatically
    picks the sheet most likely to contain sales data (using best_sheet()).
    Pass an explicit sheet name or index to override.
    """
    import pandas as pd

    file_kind = kind_of(name)
    buf = io.BytesIO(content)

    if file_kind == "excel":
        # Auto-detect best sheet when caller hasn't explicitly chosen one
        if sheet == 0:
            names = sheet_names(content)
            if len(names) > 1:
                sheet = best_sheet(content)

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

# Column-name patterns for quick detection (subset of data.py's full mapping)
_DATE_HINTS  = {"date", "fecha", "periodo", "mes", "month", "fe.factura"}
_QTY_HINTS   = {"toneladas", "tons", "qty", "quantity", "ctd.facturada", "peso neto", "cantidad"}
_CAL_HINTS   = {"calibre", "gauge", "cal"}


def _detect_col(df, hints: set) -> "str | None":
    for c in df.columns:
        if str(c).lower().strip() in hints:
            return c
    for c in df.columns:
        for h in hints:
            if h in str(c).lower():
                return c
    return None


def _file_stat(name: str, df: "pd.DataFrame", dupes_removed: int) -> dict:
    """Build the per-file summary row that quality_summary() expects."""
    import pandas as pd

    file_type = "PDF" if kind_of(name) == "pdf" else ("Excel" if kind_of(name) == "excel" else "CSV")
    n_rows = len(df)
    used_rows = n_rows  # after dedup we still count all rows for this file

    # Best-effort: detect date, qty, calibre columns
    date_col = _detect_col(df, _DATE_HINTS)
    qty_col  = _detect_col(df, _QTY_HINTS)
    cal_col  = _detect_col(df, _CAL_HINTS)

    # Month range
    meses = "—"
    if date_col:
        try:
            dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
            if not dates.empty:
                meses = f"{dates.min().strftime('%b %Y')} – {dates.max().strftime('%b %Y')}"
        except Exception:
            pass

    # Calibres
    calibres = "—"
    if cal_col:
        try:
            cals = sorted(df[cal_col].dropna().astype(str).unique().tolist())
            calibres = ", ".join(cals[:8]) + ("…" if len(cals) > 8 else "")
        except Exception:
            pass

    # Tons
    toneladas = 0.0
    if qty_col:
        try:
            toneladas = pd.to_numeric(df[qty_col], errors="coerce").sum()
        except Exception:
            pass

    return {
        "Archivo":              name,
        "Tipo":                 file_type,
        "Usado":                True,
        "Filas":                n_rows,
        "Duplicados quitados":  dupes_removed,
        "Filas usadas":         used_rows - dupes_removed,
        "Meses":                meses,
        "Calibres":             calibres,
        "Toneladas":            toneladas,
        "Motivo":               "",
    }


def combine_sales(all_tabs: list) -> tuple:
    """Combine a list of (name, DataFrame) into one deduplicated DataFrame.

    Returns (combined_df, summary_list, notes_list).
    summary_list is a list of per-file dicts that quality_summary() can render.
    combined_df is None when no valid DataFrames are supplied.
    """
    import pandas as pd

    valid = [(name, df) for name, df in all_tabs
             if isinstance(df, pd.DataFrame) and not df.empty]

    if not valid:
        return None, [], []

    combined = pd.concat([df for _, df in valid], ignore_index=True)
    n_dupes = int(combined.duplicated().sum())
    combined = combined.drop_duplicates().reset_index(drop=True)

    # Spread duplicates evenly across files for the summary
    dupes_each = n_dupes // len(valid)
    dupes_rem  = n_dupes % len(valid)

    summary = []
    for i, (name, df) in enumerate(valid):
        d = dupes_each + (1 if i < dupes_rem else 0)
        summary.append(_file_stat(name, df, d))

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
