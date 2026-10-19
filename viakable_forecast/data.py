"""data.py — minimal working data loader for Viakable Calibra.

Reads .xlsx / .xls / .csv, auto-detects columns, splits 16y18, aggregates to
monthly rows. Returns a Prepared dataclass that the rest of the app uses.
"""
from __future__ import annotations

import dataclasses
import io
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── Required column keys ──────────────────────────────────────────────────────
REQUIRED = {"date", "qty", "cal"}

# Column-name aliases tried in order (standard key → candidate names)
_ALIASES: dict[str, list[str]] = {
    "date":     ["date", "fecha", "ym", "periodo contable", "fe.factura", "month", "mes"],
    "qty":      ["toneladas", "qty", "quantity", "tons", "peso neto", "ctd.facturada"],
    "cal":      ["calibre", "cal", "gauge", "caliber"],
    "kgm":      ["contmet (kg/m)", "kg/m", "kgm", "peso kg/m"],
    "seg":      ["id uen", "seg", "uen", "business unit"],
    "ch":       ["fuerza de venta", "ch", "sales force", "canal"],
    "cli":      ["cliente", "cli", "solicitante", "customer", "client"],
    "fam":      ["vcfamiliatransversal", "familia", "fam", "family"],
    "prod":     ["id producto", "prod", "producto", "product", "texto breve de material"],
    "mkt_raw":  ["pais", "país", "mkt_raw", "market", "land1"],
}

# Mapping used when loading sample data (standard key → actual column name in sample_data())
SAMPLE_MAPPING: dict[str, str] = {
    "date":    "Date",
    "qty":     "Toneladas",
    "cal":     "Calibre",
    "kgm":     "ContMet (kg/m)",
    "seg":     "ID UEN",
    "ch":      "Fuerza de Venta",
    "fam":     "vcFamiliaTransversal",
    "prod":    "ID producto",
}

MKT_NAC = "Nacional"
MKT_USA = "Exportación EE. UU."


# ── Prepared dataclass ────────────────────────────────────────────────────────
@dataclasses.dataclass
class Prepared:
    monthly:   pd.DataFrame          # aggregated monthly rows
    notes:     list[str]
    n_months:  int
    gauges:    list[str]
    raw:       pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)


# ── File loading ──────────────────────────────────────────────────────────────
def load_file(path_or_buf, sheet=0) -> pd.DataFrame:
    """Read an Excel or CSV file and return a raw DataFrame."""
    if isinstance(path_or_buf, (str, Path)):
        p = Path(path_or_buf)
        suffix = p.suffix.lower()
        if suffix in (".xlsx", ".xls"):
            return pd.read_excel(p, sheet_name=sheet)
        return pd.read_csv(p, encoding="utf-8-sig", low_memory=False)

    # BytesIO buffer — detect by .name attribute
    name = getattr(path_or_buf, "name", "")
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path_or_buf, sheet_name=sheet)
    return pd.read_csv(path_or_buf, encoding="utf-8-sig", low_memory=False)


# ── Column detection ──────────────────────────────────────────────────────────
def detect_columns(raw: pd.DataFrame) -> dict[str, str]:
    """Return {standard_key: actual_column_name} for each detected column."""
    cols_lower = {c.lower().strip(): c for c in raw.columns}
    mapping: dict[str, str] = {}
    for key, aliases in _ALIASES.items():
        for alias in aliases:
            if alias in cols_lower:
                mapping[key] = cols_lower[alias]
                break
    return mapping


# ── Date parsing ──────────────────────────────────────────────────────────────
def _parse_date(val) -> pd.Timestamp | None:
    if pd.isna(val):
        return None
    s = str(val).strip()
    # Excel serial number
    if s.isdigit() and len(s) <= 6:
        try:
            return pd.Timestamp("1899-12-30") + pd.Timedelta(days=int(s))
        except Exception:
            pass
    # YYYYMM integer
    if len(s) == 6 and s.isdigit():
        try:
            return pd.Timestamp(year=int(s[:4]), month=int(s[4:]), day=1)
        except Exception:
            pass
    try:
        return pd.Timestamp(val)
    except Exception:
        return None


# ── 16y18 split ──────────────────────────────────────────────────────────────
def _split_16y18(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Split combined gauge '16y18' into separate 16 and 18 rows using kg/m."""
    mask = df["cal"].astype(str).str.lower().str.strip() == "16y18"
    if not mask.any():
        return df, 0
    split = df[mask].copy()
    rest = df[~mask].copy()
    n = len(split)
    if "kgm" in df.columns and split["kgm"].notna().any():
        # Larger kg/m → gauge 16, smaller → gauge 18
        med = split["kgm"].median()
        s16 = split.copy(); s16["cal"] = "16"; s16.loc[split["kgm"] < med, "qty"] = 0
        s18 = split.copy(); s18["cal"] = "18"; s18.loc[split["kgm"] >= med, "qty"] = 0
    else:
        # No kg/m: split 50/50
        s16 = split.copy(); s16["cal"] = "16"; s16["qty"] = split["qty"] / 2
        s18 = split.copy(); s18["cal"] = "18"; s18["qty"] = split["qty"] / 2
    return pd.concat([rest, s16, s18], ignore_index=True), n


# ── Market derivation ─────────────────────────────────────────────────────────
def _derive_mkt(df: pd.DataFrame, export_rule: dict | None) -> pd.DataFrame:
    df = df.copy()
    if "mkt_raw" not in df.columns:
        df["mkt_raw"] = ""
    if export_rule and export_rule.get("dim") and export_rule.get("values"):
        dim, vals = export_rule["dim"], set(str(v) for v in export_rule["values"])
        mask = df.get(dim, pd.Series(dtype=str)).astype(str).isin(vals)
        df["mkt"] = np.where(mask, MKT_USA, MKT_NAC)
    else:
        us_mask = df["mkt_raw"].astype(str).str.upper().isin({"US", "USA", "EE.UU.", "EE. UU.", "EEUU"})
        df["mkt"] = np.where(us_mask, MKT_USA, MKT_NAC)
    return df


# ── prepare() ────────────────────────────────────────────────────────────────
def prepare(
    raw: pd.DataFrame,
    mapping: dict[str, str],
    kg_to_tons: bool = False,
    export_rule: dict | None = None,
) -> Prepared:
    """Apply mapping, clean, split 16y18, aggregate monthly → Prepared."""
    notes: list[str] = []
    df = raw.copy()

    # Rename to standard keys
    inv = {v: k for k, v in mapping.items() if v in df.columns}
    df = df.rename(columns=inv)

    # Require date and qty
    if "date" not in df.columns:
        raise ValueError("No se encontró la columna de fecha.")
    if "qty" not in df.columns:
        raise ValueError("No se encontró la columna de cantidad.")

    # Parse qty as numeric
    df["qty"] = pd.to_numeric(df["qty"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    if df["qty"].isna().all():
        raise ValueError("La columna de cantidad no contiene valores numéricos.")

    # kg → tons
    if kg_to_tons:
        df["qty"] = df["qty"] / 1000
        notes.append("Cantidades convertidas de kg a toneladas.")

    # Parse date → ym (period start)
    df["_ts"] = df["date"].apply(_parse_date)
    bad_date = df["_ts"].isna().sum()
    if bad_date:
        notes.append(f"Se ignoraron {bad_date} filas con periodo no válido (p. ej. periodo 13 de cierre; {df.loc[df['_ts'].isna(), 'qty'].sum():.1f} t).")
    df = df[df["_ts"].notna()].copy()
    df["ym"] = df["_ts"].dt.to_period("M").dt.to_timestamp()

    # Exclude period 13 (closing period)
    if "Periodo Contable" in raw.columns or "periodo contable" in [c.lower() for c in raw.columns]:
        pc_col = next((c for c in raw.columns if c.lower() == "periodo contable"), None)
        if pc_col:
            mask13 = raw[pc_col].astype(str).str.endswith("13")
            if mask13.any():
                qty13 = raw.loc[mask13, mapping.get("qty", "qty")].apply(
                    pd.to_numeric, errors="coerce").sum() if mapping.get("qty") in raw.columns else 0
                n13 = mask13.sum()
                notes.append(f"Se ignoraron {n13} filas con periodo no válido (p. ej. periodo 13 de cierre; {qty13:.1f} t).")
                df = df[~df.index.isin(raw.index[mask13])].copy()

    # Calibre column
    if "cal" not in df.columns:
        df["cal"] = "?"
    df["cal"] = df["cal"].astype(str).str.strip()

    # Optional dimension columns
    for dim in ("seg", "ch", "cli", "fam", "prod", "kgm", "mkt_raw"):
        if dim not in df.columns:
            df[dim] = ""

    # Split 16y18
    df, n_split = _split_16y18(df)
    if n_split:
        notes.append(f"Se separó el calibre combinado 16y18 en 16 y 18 usando kg/m ({n_split} filas).")

    # Negative qty (returns) — keep but note
    neg = (df["qty"] < 0).sum()
    if neg:
        notes.append(f"{neg} filas tienen toneladas negativas (devoluciones); se conservan en la suma.")

    # Derive market
    df = _derive_mkt(df, export_rule)

    # Aggregate monthly
    dim_cols = [c for c in ("seg", "ch", "cli", "fam", "prod", "mkt_raw", "mkt") if c in df.columns]
    monthly = (
        df.groupby(["ym", "cal"] + dim_cols, as_index=False)["qty"]
        .sum()
        .sort_values(["ym", "cal"])
        .reset_index(drop=True)
    )
    for c in ("seg", "ch", "cli", "fam", "prod", "mkt_raw", "mkt"):
        if c not in monthly.columns:
            monthly[c] = ""

    # Derive metal type from family name (families with "AL" = Aluminio)
    monthly["metal"] = monthly["fam"].str.contains("AL", case=False, na=False).map(
        {True: "Aluminio", False: "Cobre"}
    )

    n_months = monthly["ym"].nunique()
    gauges = sorted(monthly["cal"].unique().tolist(),
                    key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))

    return Prepared(monthly=monthly, notes=notes, n_months=n_months, gauges=gauges, raw=raw)


# ── Sample data ───────────────────────────────────────────────────────────────
def sample_data() -> pd.DataFrame:
    """Generate 30 months of synthetic data matching Viakable's column structure."""
    rng = random.Random(42)
    rows = []
    gauges = ["6", "10", "12", "14", "16y18"]
    segs   = ["CONS", "ENER", "FLEX"]
    chs    = ["VIAKON COMERCIAL", "CDN COMERCIAL", "CAYCA"]
    fams   = ["CU BT | Cal 6", "CU BT | Cal 10", "CU BT | Cal 12",
              "CU BT | Cal 14", "AL BT | Cal 6"]
    start  = pd.Timestamp("2024-01-01")

    for m in range(30):
        ts = start + pd.DateOffset(months=m)
        ym_str = ts.strftime("%Y%m")
        seasonal = 1 + 0.15 * np.sin((m % 12) * np.pi / 6)
        for g in gauges:
            base_t = {"6": 180, "10": 250, "12": 300, "14": 200, "16y18": 150}[g]
            # Two rows per gauge (two product lines)
            for _ in range(2):
                seg = rng.choice(segs)
                ch  = rng.choice(chs)
                is_export = rng.random() < 0.14
                # Exports drop after Apr-2026
                if is_export and ts >= pd.Timestamp("2026-05-01"):
                    is_export = rng.random() < 0.06
                fam = fams[gauges.index(g) % len(fams)]
                kgm = {"6": 0.0515, "10": 0.0826, "12": 0.1045,
                       "14": 0.1319, "16y18": 0.1624 if rng.random() > 0.5 else 0.1316}[g]
                tons = max(0, base_t * seasonal * (0.7 + 0.6 * rng.random()) / 2)
                rows.append({
                    "Year": ts.year,
                    "Month": ts.month,
                    "Date": ts,
                    "Fuerza de Venta": ch,
                    "ID UEN": seg,
                    "Periodo Contable": int(ym_str),
                    "ID producto": f"P{g}-{rng.randint(100,999)}",
                    "Calibre": g,
                    "Producto": f"Cable {g} AWG",
                    "Gauge": g.replace("16y18", "16"),
                    "ContMet (kg/m)": kgm,
                    "vcFamiliaTransversal": fam,
                    "Toneladas": round(tons, 3),
                })

    df = pd.DataFrame(rows)
    return df
