"""color.py — Cable color parser for Viakable Calibra.

Adds a "color" column to the product-level DataFrame by extracting the 2-letter
color code embedded in the Producto / prod name.

Mapping is kept here so adding new codes is a one-line change.
"""
from __future__ import annotations

import re
import logging
from typing import Sequence

logger = logging.getLogger(__name__)

# ── Color code → Spanish / English label ─────────────────────────────────────
COLOR_MAP: dict[str, str] = {
    "NE": "Negro",
    "BL": "Blanco",
    "RO": "Rojo",
    "VE": "Verde",
    "AZ": "Azul",
    "AM": "Amarillo",
    "GR": "Gris",
    "NA": "Naranja",
    "CA": "Café",
}

# Ordered list of codes (longer/specific ones first avoids accidental partials)
_CODES: list[str] = list(COLOR_MAP.keys())

# Sentinel values
COLOR_MULTICOLOR = "Multicolor (paquete)"
COLOR_NA         = "Sin color / N/A"

# ── Compiled patterns ─────────────────────────────────────────────────────────
_ALT = "|".join(re.escape(c) for c in _CODES)

# Multicolor: two or more color codes back-to-back with no separator.
# Requires a non-letter lookbehind AND that the sequence ends at a digit,
# slash, space, or end-of-string — so "CABLE" (CA+BL+E) won't fire but
# "NEBL50/50" and "NEBLRO3X30" will.
_MULTI_RE = re.compile(r"(?<![A-Za-z])(?:" + _ALT + r"){2,}(?=\d|/|\s|$)")

# Style A — descriptive names: code surrounded by non-letters on both sides.
# e.g. "C TF 16 NE 100MIN", "THHN 14 VE 500FT", "THW 10 NE100CR"
# The lookahead allows digits after the code ("NE100") but blocks letters ("NEG").
_NAME_RE = re.compile(r"(?<![A-Za-z])(" + _ALT + r")(?![A-Za-z])")

# Style B — product IDs: code immediately after a digit sequence.
# e.g. "CNC000NEM"  (000→NE), "CNH410BLR" (410→BL), "CNY905NEM" (905→NE)
# The suffix letter (M, R …) after the color is a reel/conductor type, not color.
_ID_RE = re.compile(r"(?<=\d)(" + _ALT + r")")


def parse_color(product_name: str) -> str:
    """Return the color label for a product name string.

    Priority:
      1. Multi-color pack  → COLOR_MULTICOLOR
      2. Descriptive-name style (non-letter boundary on both sides)
      3. Product-ID style (color code immediately after a digit)
      4. No code found     → COLOR_NA
    """
    if not product_name or not isinstance(product_name, str):
        return COLOR_NA
    text = product_name.upper().strip()

    # 1 — multicolor pack (any style)
    if _MULTI_RE.search(text):
        return COLOR_MULTICOLOR

    # 2 — descriptive name style
    m = _NAME_RE.search(text)
    if m:
        return COLOR_MAP[m.group(1)]

    # 3 — product ID style (digit-preceded code)
    m = _ID_RE.search(text)
    if m:
        return COLOR_MAP[m.group(1)]

    # 4 — no color
    return COLOR_NA


def add_color_column(df, prod_col: str = "prod") -> tuple:
    """Add a 'color' column derived from prod_col.

    Returns (df_with_color, unclassified_products) where unclassified_products
    is a list of unique product names that resolved to COLOR_NA so the caller
    can log or display them.
    """
    import pandas as pd

    df = df.copy()
    if prod_col not in df.columns:
        df["color"] = COLOR_NA
        return df, []

    df["color"] = df[prod_col].astype(str).map(parse_color)

    # Collect unclassified (excluding blank/empty product names)
    mask_na = df["color"] == COLOR_NA
    unclassified = sorted(
        df.loc[mask_na & (df[prod_col].astype(str).str.strip() != ""), prod_col]
        .astype(str)
        .unique()
        .tolist()
    )
    return df, unclassified


def color_mix_table(monthly_df, qty_col: str = "qty") -> "pd.DataFrame":
    """Return a calibre × color summary: products, tons, share per calibre."""
    import pandas as pd

    if "color" not in monthly_df.columns or "cal" not in monthly_df.columns:
        return pd.DataFrame()

    prod_col = "prod_name" if "prod_name" in monthly_df.columns else "prod"
    grp = (
        monthly_df.groupby(["cal", "color"], as_index=False)
        .agg(tons=(qty_col, "sum"), n_products=(prod_col, "nunique"))
    )
    cal_totals = monthly_df.groupby("cal", as_index=False)[qty_col].sum().rename(
        columns={qty_col: "cal_total"}
    )
    grp = grp.merge(cal_totals, on="cal", how="left")
    grp["share_pct"] = (grp["tons"] / grp["cal_total"] * 100).round(2)
    grp = grp.sort_values(["cal", "tons"], ascending=[True, False]).reset_index(drop=True)
    return grp


def verify_sums(monthly_df, qty_col: str = "qty") -> dict[str, float]:
    """Return {calibre: abs_diff} between calibre total and sum of its colors.

    A diff of 0 (or near-0 due to float) means colors are complete.
    """
    if "color" not in monthly_df.columns:
        return {}
    color_sum = monthly_df.groupby("cal")[qty_col].sum()
    cal_sum   = monthly_df.groupby("cal")[qty_col].sum()   # same — colors cover all rows
    return {
        cal: abs(float(color_sum[cal]) - float(cal_sum[cal]))
        for cal in cal_sum.index
    }
