"""crm.py — in-memory stub for Viakable Calibra (no SQLite on this machine).

All constants and function signatures match what app.py expects.
Data lives only in memory for the session; nothing is persisted.
"""
from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
ROLES = ["Administrador", "Planeación", "Comercial", "Consulta"]

STAGES = [
    ("Prospecto",   10),
    ("Calificada",  25),
    ("Cotización",  50),
    ("Negociación", 75),
    ("Ganada",     100),
    ("Perdida",      0),
]
OPEN_STAGES = {s for s, p in STAGES if 0 < p < 100}
PROB = {s: p for s, p in STAGES}

ACTION_LABEL = {
    "ver_datos":         "Ver datos",
    "editar_datos":      "Editar datos de ventas",
    "ver_pronostico":    "Ver pronóstico",
    "editar_consenso":   "Editar consenso",
    "congelar":          "Congelar pronóstico",
    "ver_clientes":      "Ver clientes y oportunidades",
    "editar_clientes":   "Editar clientes y oportunidades",
    "ver_senales":       "Ver señales",
    "editar_senales":    "Editar señales y escenarios",
    "ver_equipo":        "Ver equipo y bitácora",
    "editar_equipo":     "Editar equipo",
    "asistente_escenario": "Aplicar escenario desde el asistente",
    "anomalias":           "Revisar anomalías de ritmo",
    "oportunidades":       "Ver oportunidades",
    "clientes":            "Ver clientes",
    "tareas":              "Gestionar tareas",
}

_PERMISSIONS: dict[str, set[str]] = {
    "Administrador": set(ACTION_LABEL),
    "Planeación":    {"ver_datos", "ver_pronostico", "editar_consenso", "congelar", "ver_clientes",
                      "editar_clientes", "ver_senales", "editar_senales", "ver_equipo", "asistente_escenario",
                      "anomalias", "oportunidades", "clientes", "tareas"},
    "Comercial":     {"ver_datos", "ver_pronostico", "ver_clientes", "editar_clientes", "ver_senales",
                      "oportunidades", "clientes", "tareas"},
    "Consulta":      {"ver_datos", "ver_pronostico", "ver_clientes", "ver_senales", "oportunidades", "clientes"},
}

# ── In-memory store ───────────────────────────────────────────────────────────
_customers:  list[dict] = []
_opps:       list[dict] = []
_activities: list[dict] = []
_audit:      list[dict] = []
_learned:    dict[str, str] = {}
_frozen:     list[dict] = []
_decisions:  list[dict] = []
_seeded:     bool = False
_id_counter: int = 1


def _next_id() -> int:
    global _id_counter
    _id_counter += 1
    return _id_counter


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


# ── Permissions ───────────────────────────────────────────────────────────────
def can(role: str, action: str) -> bool:
    return action in _PERMISSIONS.get(role, set())


# ── Seed sample data ──────────────────────────────────────────────────────────
def seed(entities: list[dict], gauges: list[str], next_month) -> None:
    global _seeded
    if _seeded:
        return
    _seeded = True
    sample_custs = [
        {"id": 1, "name": "Distribuidor Norte", "type": "Cliente", "country": "MX",
         "data_key": "", "owner": "sistema", "status": "Activo", "notes": "Ejemplo", "sample": True},
        {"id": 2, "name": "Exportador Sur", "type": "Cliente", "country": "US",
         "data_key": "", "owner": "sistema", "status": "Activo", "notes": "Ejemplo", "sample": True},
    ]
    _customers.extend(sample_custs)
    sample_opps = [
        {"id": 10, "customer_id": 1, "name": "Proyecto Cableado Norte", "stage": "Cotización",
         "gauge": gauges[0] if gauges else "12", "tons_month": 5.0, "months": 6,
         "start": str(next_month)[:7], "country": "MX", "owner": "sistema", "sample": True},
        {"id": 11, "customer_id": 2, "name": "Licitación Export 2026", "stage": "Negociación",
         "gauge": gauges[0] if gauges else "10", "tons_month": 3.0, "months": 4,
         "start": str(next_month)[:7], "country": "US", "owner": "sistema", "sample": True},
    ]
    _opps.extend(sample_opps)


# ── Customers ─────────────────────────────────────────────────────────────────
def customers() -> list[dict]:
    return list(_customers)


def customer(cid: int) -> dict | None:
    return next((c for c in _customers if c["id"] == cid), None)


def save_customer(data: dict, who: str, role: str) -> int:
    existing = next((c for c in _customers if c.get("id") == data.get("id")), None)
    if existing:
        existing.update(data)
        return existing["id"]
    nid = data.get("id") or _next_id()
    _customers.append({**data, "id": nid, "sample": False})
    _audit.append({"ts": _now(), "who": who, "role": role, "action": "guardar cliente",
                   "section": "clientes", "ref": str(nid), "detail": data.get("name", "")})
    return nid


# ── Opportunities ─────────────────────────────────────────────────────────────
_OPP_DEFAULTS = {
    "customer_id": None, "name": "", "stage": "Prospecto", "gauge": "",
    "tons_month": 0.0, "months": 1, "start": "", "close_date": "",
    "country": "MX", "owner": "sistema", "notes": "", "sample": False,
}


def _fill_opp(o: dict) -> dict:
    result = {**_OPP_DEFAULTS, **o}
    if not result["close_date"]:
        try:
            start = pd.Timestamp(str(result["start"]) + "-01")
            months = int(result["months"])
            close = start + pd.DateOffset(months=months)
            result["close_date"] = close.strftime("%Y-%m-%d")
        except Exception:
            result["close_date"] = ""
    return result


def opps() -> list[dict]:
    return [_fill_opp(o) for o in _opps]


def save_opp(data: dict, who: str, role: str) -> int:
    existing = next((o for o in _opps if o.get("id") == data.get("id")), None)
    if existing:
        existing.update(data)
        return existing["id"]
    nid = data.get("id") or _next_id()
    _opps.append({**data, "id": nid, "sample": False})
    _audit.append({"ts": _now(), "who": who, "role": role, "action": "guardar oportunidad",
                   "section": "oportunidades", "ref": str(nid), "detail": data.get("name", "")})
    return nid


def move_opp(oid: int, new_stage: str, who: str, role: str) -> None:
    o = next((o for o in _opps if o["id"] == oid), None)
    if o:
        o["stage"] = new_stage
        _audit.append({"ts": _now(), "who": who, "role": role, "action": "mover oportunidad",
                       "section": "oportunidades", "ref": str(oid), "detail": new_stage})


def pipeline_summary(opps_list: list[dict]) -> dict:
    total = sum(o["tons_month"] * o["months"] * PROB.get(o["stage"], 0) / 100
                for o in opps_list if o.get("stage") in PROB)
    open_n    = sum(1 for o in opps_list if o.get("stage") in OPEN_STAGES)
    open_tons = sum(o["tons_month"] * o["months"]
                    for o in opps_list if o.get("stage") in OPEN_STAGES)
    won_n  = sum(1 for o in opps_list if o.get("stage") == "Ganada")
    lost_n = sum(1 for o in opps_list if o.get("stage") == "Perdida")
    closed = won_n + lost_n
    win_rate = (won_n / closed * 100) if closed else None
    this_month = dt.date.today().strftime("%Y-%m")
    closing_list = [o for o in opps_list
                    if o.get("stage") in OPEN_STAGES and
                    str(o.get("start", ""))[:7] == this_month]
    closing_n    = len(closing_list)
    closing_tons = sum(o["tons_month"] for o in closing_list)
    weighted_month = total / 6 if total else 0.0
    return {
        "weighted_tons":  round(total, 1),
        "weighted_month": round(weighted_month, 1),
        "count":          len(opps_list),
        "open_n":         open_n,
        "open_tons":      round(open_tons, 1),
        "won_n":          won_n,
        "lost_n":         lost_n,
        "win_rate":       win_rate,
        "closing_n":      closing_n,
        "closing_tons":   round(closing_tons, 1),
    }


def crm_arrays(opps_list: list[dict], first_month, horizon: int = 12) -> dict:
    """Return {gauge: {"MX": array, "US": array}} of weighted opportunity tons."""
    import numpy as np
    result: dict[str, dict] = {}
    for o in opps_list:
        if o.get("stage") not in OPEN_STAGES:
            continue
        prob = PROB.get(o["stage"], 0) / 100
        g = str(o.get("gauge", ""))
        if not g:
            continue
        arr = np.zeros(horizon)
        m = min(int(o.get("months", 1)), horizon)
        arr[:m] = o.get("tons_month", 0) * prob
        mkt = "US" if o.get("country") == "US" else "MX"
        d = result.setdefault(g, {})
        d[mkt] = d.get(mkt, np.zeros(horizon)) + arr
    return result


# ── Activities / tasks ────────────────────────────────────────────────────────
def activities(customer_id=None, tasks_only=False, open_only=False) -> list[dict]:
    out = list(_activities)
    if customer_id is not None:
        out = [a for a in out if a.get("customer_id") == customer_id]
    if tasks_only:
        out = [a for a in out if a.get("is_task")]
    if open_only:
        out = [a for a in out if not a.get("done")]
    return out


def add_activity(data: dict, who: str, role: str) -> int:
    nid = _next_id()
    _activities.append({**data, "id": nid, "done": False, "ts": _now()})
    return nid


def complete_task(task_id: int, who: str, role: str) -> None:
    a = next((a for a in _activities if a["id"] == task_id), None)
    if a:
        a["done"] = True
        _audit.append({"ts": _now(), "who": who, "role": role, "action": "completar tarea",
                       "section": "clientes", "ref": str(task_id), "detail": a.get("note", "")})


def overdue_tasks() -> list[dict]:
    today = dt.date.today().isoformat()
    return [a for a in _activities if a.get("is_task") and not a.get("done")
            and a.get("due") and a["due"] < today]


# ── Audit ─────────────────────────────────────────────────────────────────────
def audit(who: str, role: str, action: str, section: str, ref: str, detail: str) -> None:
    _audit.append({"ts": _now(), "who": who, "role": role, "action": action,
                   "section": section, "ref": ref, "detail": detail})


def audit_log(limit: int = 200) -> list[dict]:
    return list(reversed(_audit))[:limit]


# ── Learning / feedback ───────────────────────────────────────────────────────
def learn_all() -> dict[str, str]:
    return dict(_learned)


def learn_add(tokens: list[str], intent: str) -> None:
    key = " ".join(sorted(tokens))
    _learned[key] = intent


def feedback_add(route: str, source: str, ok: bool) -> None:
    _audit.append({"ts": _now(), "who": "usuario", "role": "—",
                   "action": f"feedback {'👍' if ok else '👎'}",
                   "section": "asistente", "ref": route, "detail": source})


# ── Decisions (anomalies) ─────────────────────────────────────────────────────
def decisions() -> list[dict]:
    return list(_decisions)


def add_decision(key: str, pair: str, kind: str, status: str,
                 who: str, role: str, note: str = "") -> None:
    existing = next((d for d in _decisions if d["key"] == key and d["pair"] == pair), None)
    if existing:
        existing.update({"status": status, "note": note, "ts": _now()})
    else:
        _decisions.append({"key": key, "pair": pair, "kind": kind, "status": status,
                            "note": note, "who": who, "ts": _now()})


# ── Frozen forecasts ──────────────────────────────────────────────────────────
def frozen_list() -> list[dict]:
    return list(_frozen)


def freeze(name: str, period: str, payload: dict, who: str, role: str) -> int:
    nid = _next_id()
    _frozen.append({"id": nid, "name": name, "period": period,
                    "payload": payload, "who": who, "ts": _now()})
    _audit.append({"ts": _now(), "who": who, "role": role, "action": "congelar pronóstico",
                   "section": "pronóstico", "ref": str(nid), "detail": name})
    return nid


def delete_frozen(fid: int, who: str, role: str) -> None:
    global _frozen
    _frozen = [f for f in _frozen if f["id"] != fid]
    _audit.append({"ts": _now(), "who": who, "role": role, "action": "borrar pronóstico congelado",
                   "section": "pronóstico", "ref": str(fid), "detail": ""})
