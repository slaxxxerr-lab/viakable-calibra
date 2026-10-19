"""summary.py — stub (full report generation not available on this machine)."""
from __future__ import annotations


def to_pdf(title: str, subtitle: str, blocks: list, footer: str = "") -> bytes:
    """Return a minimal plain-text PDF stub (engine not available)."""
    text = f"{title}\n{subtitle}\n\n"
    for b in blocks:
        label = b.get("label", "")
        value = b.get("value", "")
        text += f"{label}: {value}\n"
    if footer:
        text += f"\n{footer}"
    # Minimal hand-written PDF so the download button works without dependencies
    body = text.encode("latin-1", errors="replace")
    pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842]\n"
        b"   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length " + str(len(body) + 50).encode() + b" >>\nstream\n"
        b"BT /F1 10 Tf 40 800 Td\n" + body[:200] + b"\nET\nendstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 6\n0000000000 65535 f\ntrailer\n<< /Size 6 /Root 1 0 R >>\n"
        b"startxref\n0\n%%EOF"
    )
    return pdf


def to_xlsx(title: str, subtitle: str, blocks: list, footer: str = "") -> bytes:
    """Return an Excel workbook with the summary blocks."""
    try:
        import io
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Resumen"
        ws.append([title])
        ws.append([subtitle])
        ws.append([])
        for b in blocks:
            ws.append([b.get("label", ""), b.get("value", ""), b.get("detail", "")])
        if footer:
            ws.append([])
            ws.append([footer])
        buf = io.BytesIO()
        wb.save(buf)
        return buf.getvalue()
    except Exception:
        return b""
