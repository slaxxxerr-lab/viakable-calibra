"""report.py — stub (full report generation not available on this machine)."""
from __future__ import annotations

import io
from typing import Any


def build(monthly, h, prices, params, mult, sig, w, ml=None,
          crm=None, opps=None, notes=None, unit="t", apply=False) -> dict:
    """Build a report data structure (stub — returns minimal dict)."""
    return {
        "monthly": monthly,
        "h": h,
        "prices": prices,
        "params": params,
        "notes": list(notes or []),
        "unit": unit,
    }


def to_excel(data: dict) -> bytes:
    """Serialize the report dict to an Excel workbook."""
    try:
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Reporte"
        ws.append(["Viakable Calibra — Reporte completo"])
        ws.append(["Motor no disponible en este equipo; datos ilustrativos."])
        ws.append([])
        ws.append(["Parámetro", "Valor"])
        params = data.get("params")
        if params is not None:
            for k, v in vars(params).items():
                ws.append([k, str(v)])
        buf = io.BytesIO()
        wb.save(buf)
        return buf.getvalue()
    except Exception:
        return b""
