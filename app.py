"""Viakable Calibra · Pronóstico, planeación y CRM. Interfaz Streamlit (el motor está en viakable_forecast/, el estilo en ui.py)."""
import copy
import dataclasses
import datetime as dt
import hashlib
import io
import json
import re
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import settings as ST
import ui
from i18n import set_lang, set_unit, t
from ui import Raw, esc
from viakable_forecast import atypical as AT
from viakable_forecast import assistant as AS
from viakable_forecast import kb as KBM
from viakable_forecast import llm as LLM
from viakable_forecast import router as RT
from viakable_forecast import wiki as WIKI
from viakable_forecast import crm as CRM
from viakable_forecast import db as DB
from viakable_forecast import insights as I
from viakable_forecast import data as D
from viakable_forecast import files as F
from viakable_forecast import market_data as MD
from viakable_forecast import news as NEWS
from viakable_forecast import engine as E
from viakable_forecast import methods as M
from viakable_forecast import metals as MT
from viakable_forecast import ml as MLm
from viakable_forecast import planning as PL
from viakable_forecast import risk as RK
from viakable_forecast import similar as SM
from viakable_forecast.config import (APP_NAME, DCLASS_NOTE, DEFAULT_PRICES, FOOTER, METHOD_DESC, MKT_NAC, MKT_USA, PASS_THROUGH_DEFAULT,
                                      PLAN_DEFAULTS, SIGNAL_INFO, SIGNAL_PRESETS, SIGNALS,
                                      TARIFF_CURRENT, TARIFF_ELASTICITY, TARIFF_RATE_PCT, TARIFF_SECTION,
                                      TARIFF_EFFECTIVE, TARIFF_VERIFIED, TARIFF_SOURCE)

st.set_page_config(page_title=f"{APP_NAME} · Pronóstico", page_icon="🔴", layout="wide", initial_sidebar_state="collapsed")

LOCAL_CSV = Path(__file__).parent / "Viakable_Tons_Sales.csv"   # solo existe en tu computadora (no se publica)
SRC_LOCAL, SRC_SAMPLE, SRC_UPLOAD = "Mis datos de Viakable", "Datos de ejemplo", "Subir mi archivo"
FIELD_LABELS = {
    "date": "Fecha o periodo *", "qty": "Cantidad *", "cal": "Calibre *", "kgm": "Peso kg/m (separa calibres combinados)",
    "seg": "Unidad de negocio", "ch": "Fuerza de venta", "cli": "Cliente", "fam": "Familia", "prod": "ID de producto",
    "prod_name": "Nombre del producto", "mkt_raw": "País o mercado destino",
}
DIM_LABELS = {"ch": "Fuerza de venta", "seg": "Unidad de negocio", "cli": "Cliente", "fam": "Familia", "prod": "Producto", "mkt_raw": "País"}


def adv() -> bool:
    """Modo avanzado: muestra métodos, consenso, machine learning, señales y escenarios."""
    return ST.cfg()["mode"] == "adv"


def mes_largo(ts) -> str:
    return ui.mes(ts, True)


NOTE_PATTERNS = [
    (r"Se ignoraron (\d+) filas con periodo no válido \(p\. ej\. periodo 13 de cierre; ([\d.\-]+) t\)\.", "Ignored {0} rows with an invalid period (e.g. closing period 13; {1} t)."),
    (r"Se ignoraron (\d+) filas sin cantidad numérica\.", "Ignored {0} rows without a numeric quantity."),
    (r"Cantidades convertidas de kg a toneladas\.", "Quantities converted from kg to tons."),
    (r"Se separó el calibre combinado 16y18 en 16 y 18 usando kg/m \((\d+) filas\)\.", "Combined gauge 16y18 was split into 16 and 18 using kg/m ({0} rows)."),
    (r"Hay calibres combinados que no pude separar \(falta la columna kg/m\)\.", "There are combined gauges I could not split (kg/m column missing)."),
    (r"(\d+) filas tienen toneladas negativas \(devoluciones\); se conservan en la suma\.", "{0} rows have negative tons (returns); they are kept in the sum."),
]


def note_t(n: str) -> str:
    """Traduce las notas de calidad que genera el motor (siempre en español) cuando la interfaz está en inglés."""
    if n.startswith("ATIPICOS:"):
        return t("Se corrigieron {n} meses atípicos con el valor esperado; los modelos entrenan con esos datos.", n=n.split(":")[1])
    if n.startswith("EDICIONES:"):
        return t("Se corrigieron {n} valores de ventas dentro de la app; los cálculos ya los incluyen.", n=n.split(":")[1])
    if ST.cfg()["lang"] != "en":
        return n
    for pat, en in NOTE_PATTERNS:
        mt = re.fullmatch(pat, n)
        if mt:
            return en.format(*mt.groups())
    return n


# ================================================================== estado persistente
def store() -> dict:
    """Valores de los controles; sobreviven al cambiar de sección (Streamlit borra los widgets que no se dibujan)."""
    return st.session_state.setdefault("store", {})


def sync_store():
    """Copia a `store` los valores más recientes de los controles al inicio de cada ejecución.

    Streamlit ya trae el valor nuevo del control que se movió, pero `store` se actualizaba cuando el control se dibujaba, más abajo en
    el script. Sin esto, los cálculos hechos antes de dibujar el control usaban el valor anterior (un clic de retraso)."""
    s = store()
    for k in list(s.keys()):
        if k in st.session_state:
            s[k] = st.session_state[k]


def sv(key, default=None):
    return store().get(key, default)


def wid(fn, label, key, default, options=None, **kw):
    """Dibuja un control cuyo valor se recuerda entre secciones."""
    if key not in st.session_state:
        v = store().get(key, default)
        if options is not None:
            ok = all(x in options for x in v) if isinstance(v, list) else v in options
            v = v if ok else default
        st.session_state[key] = v
    out = fn(label, key=key, **({"options": options} if options is not None else {}), **kw)
    store()[key] = out
    return out


# ================================================================== datos
@st.cache_data(show_spinner=False)
def read_raw(kind: str, name: str = "", content: bytes = b"", sheet=0) -> pd.DataFrame:
    if kind == SRC_LOCAL:
        return D.load_file(str(LOCAL_CSV))
    if kind == SRC_SAMPLE:
        return D.sample_data()
    buf = io.BytesIO(content)
    buf.name = name
    return D.load_file(buf, sheet=sheet)


@st.cache_data(show_spinner=False)
def run_prepare(raw: pd.DataFrame, mapping: tuple, kg_to_tons: bool, rule_dim, rule_vals: tuple) -> D.Prepared:
    rule = {"dim": rule_dim, "values": list(rule_vals)} if rule_dim and rule_vals else None
    return D.prepare(raw, dict(mapping), kg_to_tons=kg_to_tons, export_rule=rule)


def make_did(kind, name, sheet, kg_on, rule_dim, rule_vals) -> str:
    """Identificador del conjunto de datos cargado (cambia solo si cambia el archivo, la hoja, las unidades o la regla de exportación)."""
    return f"{kind}|{name}|{sheet}|{kg_on}|{rule_dim}|{tuple(rule_vals)}"


def base_prep() -> D.Prepared:
    """Datos cargados (sin ediciones). Carga desde SQLite si existe; si no, usa CSV local o datos de ejemplo."""
    if "prep" not in st.session_state:
        # 1. Try SQLite first (survives page refreshes)
        saved = DB.load_prep()
        if saved is not None:
            monthly_df, notes, data_id, saved_name = saved
            gauges = sorted(monthly_df["cal"].unique().tolist(),
                            key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 0))
            n_months = int(monthly_df["ym"].nunique())
            prep = D.Prepared(monthly=monthly_df, notes=notes, n_months=n_months, gauges=gauges)
            st.session_state["prep"] = prep
            st.session_state["source"] = SRC_UPLOAD
            st.session_state["data_id"] = data_id
            st.session_state["_db_loaded"] = saved_name
        else:
            # 2. Fall back to local CSV or sample data
            kind = SRC_LOCAL if LOCAL_CSV.exists() else SRC_SAMPLE
            raw = read_raw(kind)
            mp = D.SAMPLE_MAPPING if kind == SRC_SAMPLE else D.detect_columns(raw)
            st.session_state["prep"] = run_prepare(raw, tuple(mp.items()), False, None, ())
            st.session_state["source"] = kind
            st.session_state["data_id"] = make_did(kind, "", 0, False, None, ())
    return st.session_state["prep"]


def apply_edits(monthly: pd.DataFrame, edits: dict) -> pd.DataFrame:
    """Aplica correcciones de ventas por mes y calibre. edits: {('2026-06', '12'): toneladas}.

    Si el mes y calibre ya existen, se reescalan todos sus renglones (se conserva la mezcla de clientes y productos).
    Si no existen, se agrega un renglón con los atributos más recientes del calibre."""
    if not edits:
        return monthly
    d = monthly.copy()
    new_rows = []
    for (ym, cal), tons in edits.items():
        ts = pd.Timestamp(ym)
        mask = (d["ym"] == ts) & (d["cal"] == cal)
        cur = float(d.loc[mask, "qty"].sum())
        if mask.any() and cur != 0:
            d.loc[mask, "qty"] = d.loc[mask, "qty"] * (tons / cur)
        elif mask.any():
            d.loc[d.index[mask][0], "qty"] = tons
        else:
            src = d[d["cal"] == cal]
            tpl = (src.sort_values("ym").iloc[-1] if len(src) else d.iloc[-1]).copy()
            tpl["ym"], tpl["qty"], tpl["cal"], tpl["mkt"] = ts, tons, cal, MKT_NAC
            new_rows.append(tpl)
    if new_rows:
        d = pd.concat([d, pd.DataFrame(new_rows)], ignore_index=True)
        d["ym"] = pd.to_datetime(d["ym"])
    return d


@st.cache_data(show_spinner=False)
def cached_edit(monthly: pd.DataFrame, edits_key: tuple) -> pd.DataFrame:
    return apply_edits(monthly, dict(edits_key))


@st.cache_data(show_spinner=False)
def cached_detect(monthly: pd.DataFrame) -> pd.DataFrame:
    return AT.detect_all(monthly)


@st.cache_data(show_spinner=False)
def cached_correct(monthly: pd.DataFrame, corr_key: tuple) -> pd.DataFrame:
    """Aplica las correcciones de meses atípicos que el usuario decidió (detectando sobre los datos tal como llegan)."""
    return AT.apply_corrections(monthly, AT.detect_all(monthly), {k: {"a": "corregir"} for k in corr_key})


def corrected_keys() -> tuple:
    return tuple(sorted(k for k, v in store().get("atyp_dec", {}).items() if v.get("a") == "corregir"))


def monthly_edited() -> pd.DataFrame:
    """Ventas con las correcciones de cifras hechas en 'Datos' pero SIN corregir meses atípicos (sobre estos datos se detectan los atípicos)."""
    base = base_prep()
    edits = store().get("data_edits", {})
    return cached_edit(base.monthly, tuple(sorted(edits.items()))) if edits else base.monthly


def default_prep() -> D.Prepared:
    """Datos activos: los cargados, más las correcciones de cifras de 'Datos' y los meses atípicos que se decidió corregir."""
    base = base_prep()
    m = monthly_edited()
    notes = list(base.notes)
    if store().get("data_edits"):
        notes.append(f"EDICIONES:{len(store()['data_edits'])}")
    keys = corrected_keys()
    if keys:
        m = cached_correct(m, keys)
        notes.append(f"ATIPICOS:{len(keys)}")
    if m is base.monthly:
        return base
    n = len(pd.date_range(m["ym"].min(), m["ym"].max(), freq="MS"))
    return dataclasses.replace(base, monthly=m, n_months=n, notes=notes)


@st.cache_data(show_spinner=False)
def cached_ml(monthly, windows, specs=None, exog=None):
    """Entrena todos los modelos globales (principal, XGBoost, bosque, Ridge, red neuronal). `specs`: configuración ya afinada por familia."""
    ml = MLm.fit_everything(monthly, windows, dict(specs) if specs else None, exog)
    if isinstance(ml, dict):
        ml["trained_at"] = dt.datetime.now()      # queda en la caché junto con el modelo: es la hora real del entrenamiento
    return ml


@st.cache_data(show_spinner=False)
def cached_refit(monthly, spec_key, exog, future, _ml=None):
    return MLm.refit_final(monthly, _ml, exog, future)


@st.cache_data(ttl=21600, show_spinner=False)
def auto_monthly(_stamp=None):
    return MD.load_monthly()


def price_history():
    """Histórico mensual de precios de metal (columnas ym, cu, al): el que subió el usuario o, si no hay, el automático (COMEX/ALI o FRED)."""
    up = st.session_state.get("mt_hist")
    if up is not None:
        return up
    if not store().get("mt_auto_hist", True):
        return None
    try:
        ensure_hist()
        df, _ = auto_monthly()
    except Exception:
        return None
    return df if len(df) >= 18 else None


def get_ml(m, windows, edited=None):
    """Modelo de ML. Se afina y entrena una sola vez por conjunto de datos y se guarda en caché.

    Si solo se corrigieron unas cifras en 'Datos', se reutiliza la configuración ya afinada (reentrena en segundos);
    el precio del metal solo entra al modelo si mejoró la validación cruzada."""
    did = st.session_state.get("data_id", "")
    specs = st.session_state.setdefault("ml_spec", {})
    edited = bool(store().get("data_edits")) or bool(corrected_keys()) if edited is None else edited
    exog = MLm.exog_for(m, price_history())
    with st.spinner(t("Entrenando el modelo de machine learning (solo se hace una vez por conjunto de datos)…")):
        ml = cached_ml(m, windows, tuple(sorted(specs[did].items())) if (edited and did in specs) else None, exog)
        if ml and did not in specs:
            specs[did] = ml["specs"]
        if ml and ml.get("exog") and ml["exog"]["kept"]:
            p = metal_paths(store())
            k = 1 + store().get("pl_mchg", 0) / 100
            ml = cached_refit(m, str(ml["spec"]), exog, {"cu": [x * k for x in p["cu"]], "al": [x * k for x in p["al"]]}, _ml=ml)
    if isinstance(ml, dict) and ml.get("trained_at"):
        st.session_state["_ml_at"] = ml["trained_at"]
    return ml


@st.cache_data(show_spinner=False)
def cached_project(monthly, units, market, horizon, method, measure, mode, mult, sig, w, prices, _ml=None, crm=None, apply_signals=False):
    return E.project(monthly, [list(u) for u in units], market, horizon, method, measure, mode, mult, sig, w, prices, crm=crm, ml=_ml, apply_signals=apply_signals)


@st.cache_data(show_spinner=False)
def cached_matrix(monthly, units, market, measure, prices, _ml=None):
    return E.accuracy_matrix(monthly, [list(u) for u in units], market, measure, prices, _ml)


@st.cache_data(show_spinner=False)
def cached_bt(monthly, gauges, market, measure, prices, _ml=None):
    return E.selection_backtest(monthly, list(gauges), market, measure, prices, _ml)


@st.cache_data(show_spinner=False)
def cached_totals(monthly, gauges, market, horizon, measure, prices, bt, _ml=None):
    return E.method_totals(monthly, list(gauges), market, horizon, measure, prices, _ml, bt)


@st.cache_data(show_spinner=False)
def cached_plan(monthly, gauges, horizon, params, prices, mult, sig, w, overrides, _ml=None, crm=None):
    return PL.plan(monthly, list(gauges), horizon, PL.PlanParams(**params), prices, mult, sig, w, _ml, overrides, crm=crm)


# ---- parámetros compartidos (señales, arancel, precios, planeación). `s` = diccionario de valores (por omisión, los actuales)
SNAP_PREFIX = ("pl_", "t_", "chg_", "sens_", "lead_", "sigon_", "sig_w", "seg_service", "mt_", "seg_mt_", "wi_", "sig_apply")


def _s(s):
    return store() if s is None else s


def metal_paths(s=None) -> dict:
    """Camino mensual de precios (12 meses): {'cu': USD/t, 'al': USD/t, 'fx': MXN por USD}.

    Para el estado actual se calcula con los controles de Metales; un escenario guardado trae su propio camino ('mt_path')."""
    live = s is None or s is store()
    s = _s(s)
    if not live and s.get("mt_path"):
        return s["mt_path"]
    hist = price_history()
    path = MT.make_path(s.get("mt_kind", "flat"), float(s.get("pl_cu", PLAN_DEFAULTS["cu_lme"])), float(s.get("pl_al", PLAN_DEFAULTS["al_lme"])),
                        float(s.get("pl_fx", PLAN_DEFAULTS["fx"])), float(s.get("mt_cu_chg", 0)), float(s.get("mt_al_chg", 0)), float(s.get("mt_fx_chg", 0)),
                        s.get("seg_mt_shape", "lineal"), s.get("mt_manual"), hist)
    if live:
        s["mt_path"] = path
    return path


def cu_link_change(s=None) -> float:
    """Cambio % del precio del cobre (primeros 6 meses del camino contra hoy, con el escenario de metal) que entra a la señal del cobre."""
    s = _s(s)
    if not s.get("mt_link", True):
        return 0.0
    p = metal_paths(s)
    k = 1 + s.get("pl_mchg", 0) / 100
    return float(np.clip(MT.path_change([x * k for x in p["cu"]], float(s.get("pl_cu", PLAN_DEFAULTS["cu_lme"]))), -30, 30))


def cu_signal_total(s=None) -> float:
    """Cambio que entra a la señal del cobre: el del camino de precios más el del precio en vivo contra su promedio de 90 días."""
    return cu_link_change(s) + live_signal_change(s)


def get_sig(s=None) -> dict:
    """Señales activas: {nombre: {chg, sens, lead}}. Una señal apagada no mueve el pronóstico.

    La señal del cobre suma el cambio que sale del camino de precios de Metales (si está activado el vínculo)."""
    s = _s(s)
    out = {}
    for name, base in SIGNALS.items():
        on = s.get(f"sigon_{name}", True)
        chg = float(s.get(f"chg_{name}", 0))
        if name == "Precio del cobre":
            chg += cu_signal_total(s)
        out[name] = {"chg": chg if on else 0.0, "sens": float(s.get(f"sens_{name}", base["sens"])),
                     "lead": int(s.get(f"lead_{name}", base["lead"]))}
    return out


def opp_tons() -> dict:
    """Toneladas por mes de las oportunidades del CRM, ponderadas por etapa: {calibre: {'MX': arreglo, 'US': arreglo}}. Se lee en cada ejecución (cambia al instante)."""
    try:
        m = default_prep().monthly
        rows = [o for o in CRM.opps() if not o["sample"] or store().get("crm_use_sample", False)]
        return CRM.crm_arrays(rows, m["ym"].max() + pd.offsets.MonthBegin(1), 12)
    except Exception:
        return {}


def get_crm(s=None, measure=E.TON, prices=None) -> dict:
    """Oportunidades del CRM más el pedido adicional de la simulación (what-if): {calibre: {'MX': t por mes, 'US': t por mes}}, en toneladas o en pesos."""
    s = _s(s)
    out = {g: {k: v.copy() for k, v in d.items()} for g, d in opp_tons().items()}
    tons = float(s.get("wi_opp_t", 0) or 0)
    cal = s.get("wi_opp_cal")
    if tons and cal:
        arr = np.zeros(12)
        arr[:int(s.get("wi_opp_m", 3))] = tons
        d = out.setdefault(str(cal), {})
        d["MX"] = d.get("MX", np.zeros(12)) + arr
    if measure == E.MXN:
        p = prices or DEFAULT_PRICES
        fx = float(s.get("pl_fx", PLAN_DEFAULTS["fx"]))
        for g, d in out.items():
            for k in d:
                d[k] = d[k] * (float(p["usd_t"].get(g, 0)) * fx if k == "US" else float(p["mxn_t"].get(g, 0)))
    return out


def apply_signals_on(s=None) -> bool:
    return bool(_s(s).get("sig_apply", True))


def get_w(s=None) -> float:
    return _s(s).get("sig_w", 30) / 100


def get_tariff(s=None) -> dict:
    s = _s(s)
    return {"current": float(s.get("t_cur", TARIFF_CURRENT)), "scenario": float(s.get("t_sce", TARIFF_CURRENT)),
            "elasticity": float(s.get("t_ela", TARIFF_ELASTICITY))}


def get_mult(s=None) -> float:
    tf = get_tariff(s)
    return E.tariff_multiplier(tf["scenario"], tf["current"], tf["elasticity"])


def get_prices(s=None) -> dict:
    s = _s(s)
    p = copy.deepcopy(s.get("prices", DEFAULT_PRICES))
    p["fx"] = float(s.get("pl_fx", PLAN_DEFAULTS["fx"]))
    return p


def get_params(s=None) -> dict:
    s, d = _s(s), PLAN_DEFAULTS
    return {"service": int(s.get("seg_service", d["service"])), "lead_time": float(s.get("pl_lt", d["lead_time"])),
            "hold_pct": float(s.get("pl_hold", d["hold_pct"])), "metal_pct": float(s.get("pl_metalpct", d["metal_pct"])),
            "cu_lme": float(s.get("pl_cu", d["cu_lme"])), "al_lme": float(s.get("pl_al", d["al_lme"])),
            "metal_chg": float(s.get("pl_mchg", d["metal_chg"])), "fx": float(s.get("pl_fx", d["fx"])),
            "conv_cu": float(s.get("pl_ccu", d["conv_cu"])), "conv_al": float(s.get("pl_cal", d["conv_al"])),
            "exp_lt": float(s.get("pl_explt", d["exp_lt"])), "freight": float(s.get("pl_freight", d["freight"])),
            "ddp": bool(s.get("pl_ddp", d["ddp"])), "tariff_scen": float(s.get("t_sce", TARIFF_CURRENT)),
            "base_method": s.get("pl_base", E.AUTO), "cu_path": tuple(round(x, 4) for x in metal_paths(s)["cu"]),
            "al_path": tuple(round(x, 4) for x in metal_paths(s)["al"]), "fx_path": tuple(round(x, 6) for x in metal_paths(s)["fx"]),
            "pass_through": float(s.get("mt_pass", PASS_THROUGH_DEFAULT)), "apply_signals": apply_signals_on(s),
            "cu_ref": ref_value(s, "cu"), "al_ref": ref_value(s, "al"), "fx_ref": ref_value(s, "fx")}


def apply_preset(name: str):
    """Aplica un escenario de mercado: fija el cambio esperado de cada señal."""
    pr = SIGNAL_PRESETS[name]
    for s in SIGNALS:
        store()[f"chg_{s}"] = pr.get(s, 0)
        if f"chg_{s}" in st.session_state:
            st.session_state[f"chg_{s}"] = pr.get(s, 0)


# ================================================================== precios de mercado en vivo (Fase 6C)
def secret(section: str, key: str):
    """Lee una llave de .streamlit/secrets.toml (None si no existe el archivo o la llave)."""
    try:
        v = st.secrets[section][key]
        return str(v).strip() or None
    except Exception:
        return None


_UPGRADE_STARTED = {"done": False}


@st.cache_data(ttl=900, show_spinner=False)
def fetch_market(token) -> dict:
    """Precios actuales de las tres series. Se guarda en caché 15 minutos para no saturar las fuentes; guarda el resultado en SQLite."""
    q = MD.fetch_all(token)
    try:
        MD.save_quotes(q)
    except Exception as e:                  # si la base está ocupada se siguen mostrando los precios en vivo
        MD.say(f"no se pudo guardar en market.db: {type(e).__name__}: {e}")
    return {"quotes": q, "fetched": dt.datetime.now()}


@st.cache_data(ttl=21600, show_spinner=False)
def ensure_hist(_stamp=None) -> dict:
    """Histórico diario y mensual en la base (primera vez y una vez al día)."""
    out = MD.ensure_history()
    if not _UPGRADE_STARTED["done"]:           # mejora opcional con FRED / Alpha Vantage, en segundo plano para no bloquear
        _UPGRADE_STARTED["done"] = True
        threading.Thread(target=MD.upgrade_monthly, args=(secret("fred", "api_key"), secret("alphavantage", "api_key")), daemon=True).start()
    return out


def live_snapshot() -> dict:
    """Precios en vivo; si una fuente falla, el último precio guardado (marcado 'stale'); si no hay nada, status 'none'. Nunca lanza error."""
    try:
        res = fetch_market(secret("banxico", "token"))
        if all(x.status == "none" for x in res["quotes"].values()) and (dt.datetime.now() - res["fetched"]).total_seconds() > 60:
            fetch_market.clear()                     # una falla pasajera no se queda en caché 15 minutos: se reintenta cada minuto
            res = fetch_market(secret("banxico", "token"))
        q, fetched = dict(res["quotes"]), res["fetched"]
    except Exception as e:
        MD.say(f"fallo al consultar precios: {type(e).__name__}: {e}")
        q, fetched = {}, None
    saved = MD.load_quotes() if any(s not in q or q[s].status == "none" for s in MD.SERIES) else {}
    for s in MD.SERIES:
        if s not in q or q[s].status == "none":
            if s in saved:
                saved[s].errors = (q[s].errors if s in q else []) + saved[s].errors
                q[s] = saved[s]
            elif s not in q:
                q[s] = MD.Quote(s, float("nan"), None, "", "", "none", ["sin conexión"])
    return {"q": q, "fetched": fetched}


def live_sig(snap) -> tuple:
    return tuple((s, round(q.value, 4) if np.isfinite(q.value) else None, q.status) for s, q in sorted(snap["q"].items()))


def px_mode(series: str) -> str:
    return store().get(f"seg_px_{series}", "live")


def px_live(series: str) -> bool:
    """¿Se está usando el precio en vivo de esta serie (hay dato y no se eligió precio manual)?"""
    q = live_snapshot()["q"].get(series)
    return px_mode(series) == "live" and q is not None and q.status in ("live", "stale") and bool(np.isfinite(q.value))


PX_KEY = {"cu": "pl_cu", "al": "pl_al", "fx": "pl_fx"}
REF_DEFAULT = {"cu": PLAN_DEFAULTS["cu_lme"], "al": PLAN_DEFAULTS["al_lme"], "fx": PLAN_DEFAULTS["fx"]}


def apply_live():
    """Pone el precio en vivo como precio actual de cada metal y del tipo de cambio (salvo los que estén en 'Precio manual')."""
    ensure_hist()
    snap = live_snapshot()
    for s, key in PX_KEY.items():
        q = snap["q"].get(s)
        if px_live(s):
            v = round(float(q.value), 4 if s == "fx" else 2)
            store()[key] = v
            st.session_state[key] = v
        elif q is not None and q.status in ("live", "stale") and np.isfinite(q.value):
            # Even when px_live() is False (manual mode), seed the store with the DB value
            # so the what-if panel never shows 1.00 or NaN on first load
            if key not in store() or store()[key] < 5.0:
                v = round(float(q.value), 4 if s == "fx" else 2)
                store().setdefault(key, v)
    # Hard-guard: FX must never be below 5 MXN/USD (obviously wrong)
    if store().get("pl_fx", 0) < 5.0:
        store()["pl_fx"] = PLAN_DEFAULTS["fx"]
        st.session_state["pl_fx"] = PLAN_DEFAULTS["fx"]
    st.session_state["_live_applied"] = live_sig(snap)


def ref_value(s, series: str) -> float:
    """Precio de metal con el que se fijaron los precios de lista (por omisión los valores ilustrativos de la app; el usuario puede cambiarlos)."""
    v = _s(s).get(f"mt_ref_{series}")
    return float(v) if v else REF_DEFAULT[series]


@st.cache_data(ttl=3600, show_spinner=False)
def cu_avg90() -> float | None:
    d = MD.load_daily("cu")
    d = d[d["date"] >= pd.Timestamp.now().normalize() - pd.Timedelta(days=90)]
    return float(d["value"].mean()) if len(d) >= 20 else None


def live_signal_change(s=None) -> float:
    """Cambio % del precio en vivo del cobre contra su promedio de 90 días, para la señal del cobre (solo con precio en vivo)."""
    s = _s(s)
    if not s.get("mt_live_sig", True) or s.get("seg_px_cu", "live") != "live":
        return 0.0
    q = live_snapshot()["q"].get("cu")
    avg = cu_avg90()
    if q is None or q.status == "none" or not avg or not np.isfinite(q.value):
        return 0.0
    return float(np.clip((float(s.get("pl_cu", q.value)) / avg - 1) * 100, -30, 30))


# ------------------------------------------------------------------ rótulo de precios y alertas
def fmt_chg(p) -> str:
    return "" if p is None else f"{'▲' if p >= 0 else '▼'} {p:+.1f}%"


def ticker_html(snap) -> str:
    q, fetched = snap["q"], snap["fetched"]
    parts, srcs, stale_dates = [], [], []
    for s, label in (("cu", t("Cobre")), ("al", t("Aluminio")), ("fx", "USD/MXN")):
        x = q.get(s)
        if x is None or x.status == "none" or not np.isfinite(x.value):
            parts.append(f'<span class="tk"><span class="tkl">{esc(label)}</span> <b>—</b></span>')
            continue
        manual = px_mode(s) == "manual"
        shown = store().get(PX_KEY[s], x.value) if manual else x.value
        txt = f"${shown:,.0f}" if s != "fx" else f"{shown:.2f}"
        unit = " USD/t" if s != "fx" else ""
        chg = "" if manual else f' <span class="tkc">{fmt_chg(x.day_pct)}{" " + esc(t("hoy")) if s == "cu" else ""}</span>'
        badge = f' <span class="tag">{esc(t("Precio manual"))}</span>' if manual else ""
        live_ref = f' <span class="tkc">{esc(t("en vivo"))} {x.value:,.0f}</span>' if (manual and s != "fx") else (f' <span class="tkc">{esc(t("en vivo"))} {x.value:.2f}</span>' if manual else "")
        parts.append(f'<span class="tk"><span class="tkl">{esc(label)}</span> <b data-px="{s}" data-v="{shown}">{txt}{unit}</b>{chg}{badge}{live_ref}</span>')
        short = x.source.split(" ")[0].replace("COMEX", "COMEX").replace("ALI=F", "ALI")
        srcs.append(short)
        if x.status == "stale":
            stale_dates.append(x.errors[-1] if x.errors else "")
    src = " / ".join(dict.fromkeys(srcs)) or "—"
    when = f"{fetched:%H:%M}" if fetched else "—"
    warn = ""
    none_all = all(q.get(s) is None or q[s].status == "none" for s in MD.SERIES)
    if none_all:
        warn = f' <span class="tkw">⚠ {esc(t("Sin conexión y sin precios guardados: se usan los precios que escribiste."))}</span>'
        when = "—"
    elif stale_dates or not fetched:
        warn = f' <span class="tkw">⚠ {esc(t("Sin conexión o sin respuesta: se usa el último precio guardado."))}</span>'
    tip = esc(t("COMEX es una referencia cercana al LME, no el precio oficial del LME (requiere licencia de pago)."))
    return (f'<div class="ticker" title="{tip}">{" · ".join(parts)} · <span class="tkl">{esc(t("Fuente"))}: {esc(src)}</span> · '
            f'<span class="tkl">{esc(t("actualizado"))} {when}</span>{warn}</div>')


def _stamp(ts) -> str:
    """'11 oct 21:40' (o 'Oct 11 21:40' en inglés) para el indicador de frescura."""
    if ts is None:
        return "—"
    mo = (ui.MESES_EN if ST.cfg()["lang"] == "en" else ui.MESES_ES)[ts.month - 1].lower()
    return f"{mo.capitalize()} {ts.day} {ts:%H:%M}" if ST.cfg()["lang"] == "en" else f"{ts.day} {mo} {ts:%H:%M}"


def freshness_html() -> str:
    """Datos hasta, hora de entrenamiento de los modelos, hora de los precios y de las noticias. Se pone ámbar lo que lleva mucho sin actualizarse."""
    now = dt.datetime.now()
    try:
        data_to = ui.mes(default_prep().monthly["ym"].max(), True)
    except Exception:
        data_to = "—"
    trained = st.session_state.get("_ml_at")
    snap = live_snapshot()
    px = snap.get("fetched")
    saved = False
    if px is None or any(q.status != "live" for q in snap["q"].values()):
        saved = True
        tss = [dt.datetime.fromisoformat(q.ts) for q in snap["q"].values() if getattr(q, "ts", "")]
        px = px or (max(tss) if tss else None)
    try:
        nw = dt.datetime.fromisoformat(NEWS.meta_get("last_fetch")) if NEWS.meta_get("last_fetch") else None
    except Exception:
        nw = None

    def cell(label, text, old=False):
        return f'<span class="fr{" old" if old else ""}"><span class="frl">{esc(label)}</span> {esc(text)}</span>'
    px_txt = (f"{px:%H:%M}" if px and px.date() == now.date() else _stamp(px)) + (f" ({t('guardado')})" if saved else "")
    nw_txt = f"{nw:%H:%M}" if nw and nw.date() == now.date() else _stamp(nw)
    return ('<div class="fresh">' + cell(t("Datos hasta"), data_to) + cell(t("Modelos entrenados"), _stamp(trained)) + cell(t("Precios"), px_txt, saved or (px is None) or (now - px).total_seconds() > 3600)
            + cell(t("Noticias"), nw_txt, nw is None or (now - nw).total_seconds() > 7200) + "</div>")


def alert_impact(series: str, pct: float) -> float:
    """Estimación rápida del cambio en el margen de los próximos 6 meses si el precio se queda en ese movimiento (sin entrenar modelos)."""
    m = default_prep().monthly
    last = m[m["ym"] > m["ym"].max() - pd.DateOffset(months=6)]
    metal_name = "Aluminio" if series == "al" else "Cobre"
    if "metal" in last.columns:
        tons = last.loc[last["metal"] == metal_name, "qty"].sum()
    else:
        # Derive on the fly if column was stripped (e.g. old SQLite row)
        metal_mask = last["fam"].str.contains("AL", case=False, na=False) if "fam" in last.columns else pd.Series(False, index=last.index)
        tons = last.loc[metal_mask if series == "al" else ~metal_mask, "qty"].sum()
    tons = tons or last["qty"].sum()   # fallback: total tons if filter returns zero
    p = get_params()
    price = float(store().get(PX_KEY[series], 0))
    cost = tons * p["metal_pct"] / 100 * price * float(store().get("pl_fx", PLAN_DEFAULTS["fx"]))
    return -cost * pct / 100 * (1 - p["pass_through"] / 100)


def market_alert_texts(snap) -> list[str]:
    daily = {s: MD.load_daily(s) for s in ("cu", "al")}
    out = []
    for a in MD.alerts(snap["q"], daily):
        name = t("el cobre") if a["series"] == "cu" else t("el aluminio")
        verb = t("sube") if a["pct"] >= 0 else t("baja")
        per = t("hoy") if a["period"] == "día" else t("en el último mes")
        dm = alert_impact(a["series"], a["pct"])
        eff = t("baja") if dm < 0 else t("sube")
        txt = t("Alerta de mercado: {m} {v} {p}% {per}. Si se mantiene, el margen de los próximos 6 meses {e} unos {x} (estimado, con traspaso de {pt}%).",
                m=name, v=verb, p=f"{abs(a['pct']):.1f}", per=per, e=eff, x=ui.mxn(abs(dm)), pt=f"{get_params()['pass_through']:.0f}")
        out.append(txt)
    return out


def market_alerts_html(snap) -> str:
    return "".join(f'<div class="alertbar">{esc(x)}</div>' for x in market_alert_texts(snap))


@st.fragment(run_every="15m")
def ticker():
    """Rótulo de precios. Se actualiza solo cada 15 minutos mientras la app está abierta; si cambian los precios, recalcula toda la app."""
    snap = live_snapshot()
    c1, c2 = st.columns([8, 1.4], vertical_alignment="center")
    c1.html(ticker_html(snap))
    c2.button(t("Actualizar precios"), key="px_refresh", width="stretch", on_click=fetch_market.clear)
    al = market_alerts_html(snap)
    if al:
        st.html(al)
    if live_sig(snap) != st.session_state.get("_live_applied"):
        st.rerun()


# ================================================================== controles compartidos
def cal_chips(gauges):
    """Chips de calibre: 'Todos', cada calibre y '16 + 18'. 'Todos' y '16 + 18' son excluyentes."""
    opts = ["Todos"] + list(gauges) + (["16 + 18"] if {"16", "18"} <= set(gauges) else [])
    key = "chips_cal"
    if key not in st.session_state:
        v = [x for x in sv(key, ["Todos"]) if x in opts] or ["Todos"]
        st.session_state[key], st.session_state["_cal_prev"] = v, v

    def _fix():
        new = list(st.session_state.get(key) or [])
        prev = st.session_state.get("_cal_prev", ["Todos"])
        added = [x for x in new if x not in prev]
        if not new or "Todos" in added:
            new = ["Todos"]
        elif "16 + 18" in added:
            new = ["16 + 18"]
        else:
            new = [x for x in new if x not in ("Todos", "16 + 18")]
            if len(new) == len(gauges):
                new = ["Todos"]
        st.session_state[key] = new
        st.session_state["_cal_prev"] = new

    st.pills(t("Calibre"), opts, selection_mode="multi", key=key, on_change=_fix, format_func=t)
    sel = st.session_state.get(key) or ["Todos"]
    store()[key] = sel
    st.caption(t("Puedes elegir varios calibres a la vez."))
    if "Todos" in sel:
        return [[g] for g in gauges]
    if "16 + 18" in sel:
        return [["16", "18"]]
    return [[g] for g in sorted(sel, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 0))]


# Hex/CSS background for each color name (used only for the swatch legend)
_COLOR_HEX: dict[str, str] = {
    "Negro":               "#1a1a1a",
    "Blanco":              "#d8d8d8",
    "Rojo":                "#d93030",
    "Verde":               "#1f9e50",
    "Azul":                "#1d7fe8",
    "Amarillo":            "#e6b800",
    "Gris":                "#8a8a8a",
    "Naranja":             "#e87820",
    "Café":                "#7b4b2a",
    "Multicolor (paquete)":"linear-gradient(90deg,#d93030 0%,#1f9e50 35%,#1d7fe8 70%,#e6b800 100%)",
    "Sin color / N/A":     "#cccccc",
}

# Colored Unicode squares shown inside the pill chip for each color
_COLOR_SQUARE: dict[str, str] = {
    "Negro":               "⬛",
    "Blanco":              "⬜",
    "Rojo":                "🟥",
    "Verde":               "🟩",
    "Azul":                "🟦",
    "Amarillo":            "🟨",
    "Gris":                "🔲",
    "Naranja":             "🟧",
    "Café":                "🟫",
    "Multicolor (paquete)":"🌈",
    "Sin color / N/A":     "○",
}


def _color_swatches(colors: list[str]):
    """Inline legend: colored dot + label for each color in the list."""
    if not colors:
        return
    parts = []
    for c in colors:
        bg = _COLOR_HEX.get(c, "#cccccc")
        border = "border:1px solid rgba(128,128,128,.25);" if c in ("Blanco", "Sin color / N/A") else ""
        parts.append(
            f'<span class="acc" style="white-space:nowrap">'
            f'<i style="background:{bg};{border}border-radius:50%;width:8px;height:8px;flex-shrink:0"></i>'
            f'{esc(t(c))}</span>'
        )
    st.html(
        '<div style="display:flex;flex-wrap:wrap;gap:4px 12px;padding:2px 0 6px">'
        + "".join(parts) + "</div>"
    )


def color_chips(monthly, cal_selection: list) -> "set[str] | None":
    """Color filter chips, cascading from the calibre selection.

    Returns a set of selected color labels, or None when 'Todos' is selected.
    """
    if "color" not in monthly.columns:
        return None

    sel_gauges = tuple(sorted(g for u in cal_selection for g in u))

    # When calibre selection changes, drop colors that no longer exist
    if st.session_state.get("_color_gauges") != sel_gauges:
        m_cal = monthly[monthly["cal"].isin(sel_gauges)]
        valid_now = set(m_cal["color"].unique())
        key = "chips_color"
        current = st.session_state.get(key, ["Todos"])
        kept = [c for c in current if c == "Todos" or c in valid_now]
        st.session_state[key] = kept if kept else ["Todos"]
        st.session_state["_color_gauges"] = sel_gauges

    # Available colors in the selected calibres, ordered by tons desc
    m_cal = monthly[monthly["cal"].isin(sel_gauges)]
    order = (
        m_cal.groupby("color")["qty"].sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    from viakable_forecast.color import COLOR_MULTICOLOR, COLOR_NA
    priority   = [c for c in order if c not in (COLOR_NA, COLOR_MULTICOLOR)]
    extras     = [c for c in (COLOR_MULTICOLOR, COLOR_NA) if c in order]
    avail      = priority + extras

    if not avail:
        return None

    opts = ["Todos"] + avail
    key  = "chips_color"
    if key not in st.session_state:
        v = [x for x in sv(key, ["Todos"]) if x in opts] or ["Todos"]
        st.session_state[key], st.session_state["_color_prev"] = v, v

    def _fix_color():
        new  = list(st.session_state[key] or [])
        prev = st.session_state.get("_color_prev", ["Todos"])
        added = [x for x in new if x not in prev]
        if not new or "Todos" in added:
            new = ["Todos"]
        else:
            new = [x for x in new if x != "Todos"]
        st.session_state[key]       = new
        st.session_state["_color_prev"] = new

    def _fmt_color(c):
        if c == "Todos":
            return t("Todos")
        sq = _COLOR_SQUARE.get(c, "●")
        return f"{sq} {t(c)}"

    with st.container(height=130, border=False):
        st.pills(t("Color"), opts, selection_mode="multi", key=key, on_change=_fix_color,
                 format_func=_fmt_color)
    sel = st.session_state[key] or ["Todos"]
    store()[key] = sel

    return None if "Todos" in sel else set(sel)


def price_control(series: str, label: str, key: str, default: float, step: float, min_value: float):
    """Precio de un metal (o del tipo de cambio): en vivo por omisión, o manual (por ejemplo un precio de contrato)."""
    q = live_snapshot()["q"].get(series)
    ok = q is not None and q.status in ("live", "stale") and bool(np.isfinite(q.value))
    st.html(f'<div style="font-weight:700;font-size:.93rem;margin-top:6px">{esc(label)}</div>')
    mode = wid(st.radio, t("Fuente de {x}", x=label), f"seg_px_{series}", "live", options=["live", "manual"], horizontal=True, label_visibility="collapsed",
               format_func=lambda k: t("En vivo") if k == "live" else t("Precio manual")) if ok else "manual"
    live = ok and mode == "live"
    val = wid(st.number_input, label, key, default, min_value=min_value, step=step, disabled=live, label_visibility="collapsed")
    if live:
        when = q.ts[:16].replace("T", " ") if q.ts else ""
        st.caption(t("En vivo: {s} · {w}", s=q.source, w=when) + (" · " + t("último precio guardado") if q.status == "stale" else ""))
    else:
        st.html(f'<span class="tag" style="margin:0">{esc(t("Precio manual"))}</span>')
    return val


def px_disabled(series: str) -> bool:
    return px_live(series)


def selection_controls(prep, market_label="Mercado"):
    """Mercado, calibres, color y horizonte: los mismos controles en Pronóstico y Planeación."""
    m = prep.monthly
    mk = [x for x in (MKT_NAC, MKT_USA) if x in set(m["mkt"])]
    market = None
    if len(mk) > 1:
        market = wid(st.radio, t(market_label), "seg_mkt", "Todos", options=["Todos", *mk], horizontal=True, format_func=t)
        market = None if market == "Todos" else market
    units = cal_chips(prep.gauges)
    color_filter = color_chips(m, units)
    horizon = wid(st.radio, t("Horizonte"), "seg_h", 6, options=[3, 6, 12], horizontal=True, format_func=lambda x: t("{n} meses", n=x))
    return units, market, horizon, color_filter


def layout():
    with st.container(key="grid"):
        return st.columns([290, 1100], gap="large")


def scope_text(units, market, n_gauges):
    allg = [g for u in units for g in u]
    if len(allg) == n_gauges:
        base = t("en todos los calibres")
    elif len(units) == 1:
        base = t("de calibre {g} AWG", g=" + ".join(units[0]))
    else:
        base = t("en la selección actual")
    return base + (f" · {t(market)}" if market else "")


def money_unit():
    return "MXN"


# ================================================================== plantillas de Excel
def sales_template() -> bytes:
    """Plantilla de ventas: encabezados reconocidos por la app y unas filas de ejemplo."""
    en = ST.cfg()["lang"] == "en"
    raw = D.sample_data()
    # sample_data() uses original file column names; map to what this function expects
    raw = raw.rename(columns={
        "Date": "Fecha", "Fuerza de Venta": "Canal", "ID UEN": "Unidad",
        "vcFamiliaTransversal": "Familia",
    })
    # sample_data has both "Producto" (description) and "ID producto"; keep description as Producto
    if "ID producto" in raw.columns and "Producto" in raw.columns:
        raw = raw.drop(columns=["ID producto"])
    if "Cliente" not in raw.columns:
        raw["Cliente"] = ""
    if "País" not in raw.columns:
        raw["País"] = "MX"
    raw = raw[raw["Fecha"].isin(sorted(raw["Fecha"].unique())[:8])].groupby(["Fecha", "Calibre"], as_index=False).head(1).copy()   # 8 meses, un renglón por calibre
    raw["Fecha"] = pd.to_datetime(raw["Fecha"]).dt.strftime("%Y-%m")
    names = {"Fecha": ("Fecha", "Date"), "Toneladas": ("Toneladas", "Tons"), "Calibre": ("Calibre", "Gauge"), "Unidad": ("Unidad de negocio", "Business unit"),
             "Canal": ("Fuerza de venta", "Sales force"), "Cliente": ("Cliente", "Customer"), "Familia": ("Familia", "Family"),
             "Producto": ("ID producto", "Product ID"), "País": ("País", "Country")}
    raw = raw.rename(columns={k: v[1 if en else 0] for k, v in names.items()})[[v[1 if en else 0] for v in names.values()]]
    notes = pd.DataFrame({t("Cómo llenar la plantilla"): [
        t("Una fila por mes, calibre y cliente. Las columnas de fecha, cantidad y calibre son obligatorias."),
        t("La fecha puede ser 2026-03, 03/2026, mar-2026 o una fecha de Excel."),
        t("Las cantidades van en toneladas, o en kilogramos si marcas esa opción al cargar el archivo."),
        t("Borra las filas de ejemplo y pega tus ventas."), ]})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        raw.to_excel(xw, sheet_name=t("Ventas"), index=False)
        notes.to_excel(xw, sheet_name=t("Instrucciones"), index=False)
    return buf.getvalue()


def indicator_template() -> bytes:
    """Plantilla de indicadores para calibrar señales: una columna de fecha y una por indicador."""
    months = pd.date_range("2024-01-01", periods=24, freq="MS")
    rng = np.random.default_rng(3)
    df = pd.DataFrame({t("Fecha"): months.strftime("%Y-%m")})
    for name in SIGNALS:
        df[t(name)] = (100 + np.cumsum(rng.normal(0.4, 1.5, len(months)))).round(1)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name=t("Indicadores"), index=False)
    return buf.getvalue()


# ================================================================== varios archivos (Excel, CSV y PDF)
@st.cache_data(show_spinner=False)
def cached_table(name: str, content: bytes, sheet=0) -> pd.DataFrame:
    return F.read_table(name, content, sheet)


@st.cache_data(show_spinner=False)
def cached_pdf_tables(content: bytes):
    return F.pdf_tables(content)


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s)


def pdf_review(pdfs):
    """Tablas encontradas en los PDF. Nada se usa hasta que el usuario las revise y diga qué son. Devuelve las tablas de ventas confirmadas."""
    sales = []
    for name, content in pdfs:
        try:
            tabs, no_table = cached_pdf_tables(content)
        except Exception as e:
            st.error(t("No pude leer el PDF {f}: {e}", f=name, e=e))
            continue
        with st.expander(t("PDF: {f} · {n} tablas", f=name, n=len(tabs)), expanded=bool(tabs)):
            if not tabs:
                st.info(t("No encontré tablas con texto en este PDF. Si es una imagen escaneada, hace falta pasarlo a Excel o CSV."))
                continue
            if no_table:
                st.caption(t("{n} páginas tienen texto pero ninguna tabla.", n=no_table))
            for tb in tabs:
                key = f"{_safe(name)}_p{tb['pagina']}_t{tb['numero']}"
                st.html(f'<p class="syncnote" style="margin-top:14px"><b>{esc(t("Página {p}, tabla {n}", p=tb["pagina"], n=tb["numero"]))}</b></p>')
                saved = store().get(f"pdfdf_{key}", tb["tabla"])
                ed = st.data_editor(saved if list(saved.columns) == list(tb["tabla"].columns) else tb["tabla"], hide_index=True, width="stretch", key=f"pdf_ed_{key}")
                store()[f"pdfdf_{key}"] = ed
                kinds = ["ignorar", "precios", "ventas"]
                lab = {"ignorar": "No usar", "precios": "Lista de precios", "ventas": "Tabla de ventas"}
                kind = wid(st.radio, t("Qué es esta tabla"), f"pdfkind_{key}", "ignorar", options=kinds, horizontal=True, format_func=lambda k: t(lab[k]))
                if kind == "precios":
                    g = F.guess_columns(ed)
                    cols = list(ed.columns)
                    c = st.columns(3)
                    with c[0]:
                        cal_c = wid(st.selectbox, t("Columna del calibre"), f"pdfcal_{key}", g["cal"] or cols[0], options=cols)
                    with c[1]:
                        pr_c = wid(st.selectbox, t("Columna del precio"), f"pdfpr_{key}", g["price"] or cols[-1], options=cols)
                    with c[2]:
                        cur = wid(st.radio, t("Moneda"), f"pdfcur_{key}", "mxn_t", options=["mxn_t", "usd_t"], horizontal=True, format_func=lambda k: "MXN/t" if k == "mxn_t" else "USD/t")
                    found = F.prices_from_table(ed, cal_c, pr_c)
                    if found:
                        st.html(ui.table([t("Calibre"), t("Precio")], [[ui.cal_label(g_), f"{v:,.0f} {'MXN/t' if cur == 'mxn_t' else 'USD/t'}"] for g_, v in found.items()], num=(1,)))

                        def _use(found=found, cur=cur):
                            pc = copy.deepcopy(store().get("prices", DEFAULT_PRICES))
                            pc[cur].update(found)
                            store()["prices"] = pc
                            store()["ov_ver"] = sv("ov_ver", 0) + 1
                            store()["pdf_prices_msg"] = len(found)
                        st.button(t("Usar estos precios en Planeación"), key=f"pdfuse_{key}", on_click=_use)
                    else:
                        st.warning(t("No encontré calibres con precio numérico en esas columnas."))
                elif kind == "ventas":
                    norm, info = F.normalize_sales(ed)
                    if norm is None:
                        st.warning(t("A esta tabla le faltan columnas de ventas: {x}.", x=", ".join(F.CANON[k] for k in info)))
                    else:
                        st.html(f'<p class="hint">{esc(t("Lista para combinar con tus otros archivos de ventas."))}</p>')
                        sales.append((f"{name} p{tb['pagina']} t{tb['numero']}", ed.copy()))
    if store().get("pdf_prices_msg"):
        st.success(t("Se actualizaron {n} precios de venta. Ya se usan en Planeación.", n=store().pop("pdf_prices_msg")))
    return sales


def quality_summary(summary, notes):
    """Resumen de calidad por archivo: filas, duplicados quitados, meses, calibres y toneladas."""
    st.html(ui.h2(t("Resumen de los archivos")))
    rows = []
    for r in summary:
        used = r.get("Usado")
        rows.append([Raw(f"<b>{esc(r['Archivo'])}</b>"), r["Tipo"], t("Sí") if used else Raw(f'<span class="neg">{esc(t("No"))}</span>'), f"{r['Filas']:,}",
                     f"{r.get('Duplicados quitados', 0):,}" if used else "—", f"{r.get('Filas usadas', 0):,}" if used else "—", r.get("Meses", "—") if used else "—",
                     str(r.get("Calibres", "—")) if used else "—", f"{r['Toneladas']:,.0f}" if used else "—", motivo_t(r.get("Motivo", ""))])
    st.html(ui.table([t("Archivo"), t("Tipo"), t("Usado"), t("Filas"), t("Duplicados quitados"), t("Filas usadas"), t("Meses"), t("Calibres"), t("Toneladas"), t("Motivo")],
                     rows, num=(3, 4, 5, 8), min_w=980))
    if notes:
        st.html("<ul class='hint'>" + "".join(f"<li>{esc(file_note_t(n))}</li>" for n in notes) + "</ul>")


def motivo_t(s: str) -> str:
    return t("Faltan columnas: {x}", x=s.split(":", 1)[1].strip()) if s.startswith("Faltan columnas:") else s


def file_note_t(n: str) -> str:
    """Traduce las notas de combinación de archivos (el módulo las genera en español)."""
    if ST.cfg()["lang"] != "en":
        return n
    for pat, en in ((r"Se quitaron (\d+) filas repetidas entre archivos \(periodos que se traslapan\)\.", "Removed {0} rows repeated across files (overlapping periods)."),
                    (r"(.+): no trae (.+); esas columnas quedan vacías en sus filas\.", "{0}: has no {1}; those columns stay empty in its rows."),
                    (r"(.+): trae (.+), que el primer archivo no tiene\.", "{0}: has {1}, which the first file does not have.")):
        mt = re.fullmatch(pat, n)
        if mt:
            return en.format(*mt.groups())
    return n


def upload_block():
    """Carga de varios archivos. Devuelve (tabla de ventas, nombre para mostrar, firma de los archivos) o (None, '', '')."""
    st.html(f'<p class="hint" style="margin-bottom:6px">{esc(t("Haz clic en el botón o arrastra tu archivo directamente al recuadro de abajo."))}</p>')
    ups = st.file_uploader(t("Arrastra aquí tus archivos"), type=["xlsx", "xls", "csv", "pdf"], accept_multiple_files=True, key="up_multi",
                           help=t("Puedes subir varios archivos de ventas con la misma estructura (por año o por planta) y PDF con tablas. Solo son obligatorias la fecha, la cantidad y el calibre."))
    if ups:
        st.session_state["uploads"] = [(f.name, f.getvalue()) for f in ups]
    files = st.session_state.get("uploads", [])
    if not files:
        st.html(f'<p class="hint">{esc(t("Solo son obligatorias la fecha, la cantidad y el calibre; el resto de columnas sirve para filtrar."))}</p>')
        return None, "", ""
    if not ups and st.button(t("Quitar los archivos subidos"), key="up_clear"):
        st.session_state.pop("uploads", None)
        st.rerun()
    tables = [(n, c) for n, c in files if F.kind_of(n) != "pdf"]
    pdfs = [(n, c) for n, c in files if F.kind_of(n) == "pdf"]
    sheets = {}
    multi = [(n, c) for n, c in tables if F.kind_of(n) == "excel" and len(F.sheet_names(c)) > 1]
    if multi:
        with st.expander(t("Hojas de los archivos Excel")):
            for n, c in multi:
                sheets[n] = st.selectbox(t("Hoja de {f}", f=n), F.sheet_names(c), key=f"sheet_{_safe(n)}")
    dfs = []
    for n, c in tables:
        try:
            dfs.append((n, cached_table(n, c, sheets.get(n, 0))))
        except Exception as e:
            st.error(t("No pude leer {f}: {e}", f=n, e=e))
    extra = pdf_review(pdfs) if pdfs else []
    all_tabs = dfs + extra
    if not all_tabs:
        st.info(t("Sube al menos un archivo de ventas (Excel o CSV) o confirma una tabla de ventas de un PDF."))
        return None, "", ""
    combined, summary, notes = F.combine_sales(all_tabs)
    quality_summary(summary, notes)
    if combined is None:
        st.error(t("Ningún archivo trae fecha, cantidad y calibre."))
        return None, "", ""
    sig = hashlib.md5(b"".join(c for _, c in tables) + "|".join(f"{n}:{len(d)}" for n, d in extra).encode() + str(sheets).encode()).hexdigest()[:10]
    if len(all_tabs) == 1:
        return all_tabs[0][1], all_tabs[0][0], sig
    return combined, t("{n} archivos", n=len(all_tabs)), sig


# ================================================================== diagnóstico de carga
def _diag_upload(raw: pd.DataFrame, prep, mapping: dict, kind: str, file_name: str):
    """Diagnostic panel shown in Avanzado mode after a file is loaded."""
    import traceback as _tb

    st.markdown(f"**{t('Archivo')}:** `{file_name or kind}`")
    st.markdown(f"**{t('Filas leídas')}:** {len(raw):,} · **{t('Columnas')}:** {len(raw.columns)}")

    # Column types
    col_rows = [[c, str(raw[c].dtype), str(raw[c].iloc[0]) if len(raw) else "—"]
                for c in raw.columns[:20]]
    st.markdown(f"**{t('Columnas del archivo')} (primeras 20):**")
    col_data = pd.DataFrame(col_rows, columns=[t("Columna"), t("Tipo"), t("Ejemplo")])
    st.dataframe(col_data, hide_index=True, use_container_width=True)

    # Detected mapping
    map_rows = [[k, v or "—"] for k, v in mapping.items()]
    st.markdown(f"**{t('Mapeo detectado')}:**")
    st.dataframe(pd.DataFrame(map_rows, columns=[t("Campo estándar"), t("Columna del archivo")]),
                 hide_index=True, use_container_width=True)

    # Post-prepare stats
    m = prep.monthly
    st.markdown(f"**{t('Rango de fechas')}:** {m['ym'].min().strftime('%b %Y')} – {m['ym'].max().strftime('%b %Y')}")
    st.markdown(f"**{t('Meses únicos')}:** {prep.n_months}")
    st.markdown(f"**{t('Calibres encontrados')}:** {', '.join(prep.gauges)}")
    if "color" in m.columns:
        colors = sorted(m["color"].unique().tolist())
        st.markdown(f"**{t('Colores encontrados')}:** {', '.join(colors)}")
    st.markdown(f"**{t('Productos únicos')}:** {m['prod'].nunique()}")
    st.markdown(f"**{t('Filas mensuales tras agregar')}:** {len(m):,}")
    st.markdown(f"**{t('Toneladas totales')}:** {m['qty'].sum():,.1f}")

    # Notes from prepare
    if prep.notes:
        st.markdown(f"**{t('Notas del procesamiento')}:**")
        for n in prep.notes:
            st.markdown(f"- {n}")


# ================================================================== gestión de archivos guardados
def _file_manager_block():
    """Muestra los archivos guardados en SQLite y permite eliminarlos o volver a uno anterior."""
    uploads = DB.list_uploads()
    if not uploads:
        return
    with st.expander(t("Datos guardados (sobreviven al recargar la página)"), expanded=False):
        st.caption(t("Los datos que subiste se guardan localmente para que no tengas que volver a subirlos cada vez que recargues la página. "
                     "Solo aplica a esta sesión del servidor."))
        for u in uploads:
            gauges_str = ", ".join(u["gauges"])
            col1, col2, col3 = st.columns([4, 1, 1])
            status = "✅ " if u["active"] else ""
            label = f"{status}{u['name'] or u['source']} · {u['n_months']} {t('meses')} · {gauges_str} · {u['tons']:,.0f} t"
            col1.markdown(label)
            if not u["active"] and col2.button(t("Usar"), key=f"db_use_{u['id']}", use_container_width=True):
                try:
                    DB.set_active(u["id"])
                    st.session_state.pop("prep", None)
                    st.session_state.pop("data_id", None)
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
            if col3.button(t("Borrar"), key=f"db_del_{u['id']}", use_container_width=True):
                try:
                    DB.delete_upload(u["id"])
                    st.session_state.pop("prep", None)
                    st.session_state.pop("data_id", None)
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
        if st.button(t("Borrar todos los datos guardados"), key="db_clear_all", type="secondary"):
            DB.clear_all()
            st.session_state.pop("prep", None)
            st.session_state.pop("data_id", None)
            st.rerun()


# ================================================================== sección Datos
def section_datos():
    st.html(ui.headline(t("Carga tus ventas y el pronóstico se recalcula solo."),
                        t("Sube uno o varios Excel o CSV con el historial de venta, y PDF con tablas si las tienes. La app detecta calibre, unidad de negocio, "
                          "fuerza de venta, cliente, familia y producto, y agrupa todo por mes.")))
    # Banner when data was auto-loaded from SQLite on this page load
    db_loaded_name = st.session_state.pop("_db_loaded", None)
    if db_loaded_name:
        st.info(t("✓ Datos de \"{n}\" cargados automáticamente. Sube un nuevo archivo para reemplazarlos.", n=db_loaded_name), icon="💾")
    options = ([SRC_LOCAL] if LOCAL_CSV.exists() else []) + [SRC_SAMPLE, SRC_UPLOAD]
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        kind = wid(st.radio, t("Fuente de datos"), "seg_src", options[0], options=options, horizontal=True, format_func=t)
    c2.download_button(t("Descargar plantilla de ventas"), sales_template(), file_name=t("plantilla-ventas-viakable.xlsx"),
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_tpl_sales", width="stretch")
    c3.download_button(t("Descargar plantilla de indicadores"), indicator_template(), file_name=t("plantilla-indicadores-viakable.xlsx"),
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_tpl_ind", width="stretch")
    name, sheet, up_sig, raw = "", 0, "", None
    if kind == SRC_UPLOAD:
        raw, name, up_sig = upload_block()
        if raw is None:
            return
    else:
        try:
            raw = read_raw(kind, "", b"", 0)
        except Exception as e:
            st.error(t("No pude leer el archivo: {e}", e=e))
            return
    sid = f"{kind}|{up_sig or name}|{sheet}"
    detected = D.SAMPLE_MAPPING if kind == SRC_SAMPLE else D.detect_columns(raw)
    none_lbl = t("(ninguna)")
    with st.expander(t("Revisa las columnas detectadas (puedes corregirlas)")):
        cols = [none_lbl] + list(raw.columns)
        # Pre-seed session state so selectboxes default to the detected column.
        # Also override a stale "(ninguna)" entry if the detected mapping now has a real value.
        for k in FIELD_LABELS:
            wkey = f"map_{sid}_{k}"
            d0 = detected.get(k)
            if d0 and d0 in cols:
                cur = st.session_state.get(wkey)
                if cur is None or cur == none_lbl:
                    st.session_state[wkey] = d0
        mapping, grid = {}, st.columns(4)
        for i, (k, label) in enumerate(FIELD_LABELS.items()):
            d = detected.get(k)
            sel = grid[i % 4].selectbox(t(label), cols, index=cols.index(d) if d in cols else 0, key=f"map_{sid}_{k}")
            mapping[k] = None if sel == none_lbl else sel
        kg_on = st.checkbox(t("Las cantidades están en kilogramos (convertir a toneladas)"), key=f"kg_{sid}")
        prev_rows = []
        for k, label in FIELD_LABELS.items():
            col = mapping.get(k)
            ex = ", ".join(str(v) for v in raw[col].dropna().astype(str).head(3)) if col else "—"
            prev_rows.append([Raw(f"<b>{esc(t(label))}</b>"), col or "—", Raw(f'<span class="hint">{esc(ex)}</span>')])
        st.html(ui.table([t("Campo"), t("Columna del archivo"), t("Ejemplos")], prev_rows))
    miss = [t(FIELD_LABELS[k]) for k in D.REQUIRED if not mapping.get(k)]
    if miss:
        st.error(t("Falta indicar: {x}", x=", ".join(miss)))
        return
    mp = tuple(mapping.items())
    try:
        prep0 = run_prepare(raw, mp, kg_on, None, ())
    except ValueError as e:
        st.error(str(e))
        return

    st.html(ui.h2(t("Ventas de exportación a EE. UU."), t("Tus datos no traen país ni cliente. Indica qué valores cuentan como exportación a EE. UU.; "
                                                        "por omisión no hay ninguna regla y todo se considera Nacional.")))
    avail = {k: DIM_LABELS[k] for k in DIM_LABELS if (prep0.monthly[k].astype(str).str.strip() != "").any()}
    c1, c2 = st.columns([1, 3])
    dim_lbl = c1.selectbox(t("Identificar por"), [none_lbl] + [t(v) for v in avail.values()], key=f"rdim_{sid}")
    rule_dim, rule_vals = None, ()
    if dim_lbl != none_lbl:
        rule_dim = next(k for k, v in avail.items() if t(v) == dim_lbl)
        vals = sorted(prep0.monthly[rule_dim].astype(str).unique())
        lbl = t("Valores que son exportación a EE. UU.")
        with c2:
            if len(vals) <= 40:
                rule_vals = tuple(st.pills(lbl, vals, selection_mode="multi", key=f"chips_rule_{sid}_{rule_dim}") or [])
            else:
                rule_vals = tuple(st.multiselect(lbl, vals, key=f"rval_{sid}_{rule_dim}"))
    prep = run_prepare(raw, mp, kg_on, rule_dim, rule_vals)
    did = make_did(kind, up_sig or name, sheet, kg_on, rule_dim, rule_vals)
    if st.session_state.get("data_id") != did:           # datos nuevos: se descartan las correcciones y la configuración del ML
        store().pop("data_edits", None)
        store()["edit_ver"] = store().get("edit_ver", 0) + 1
        # Reset gauge/color filter chips so stale selections don't filter out new data
        for _k in ("chips_cal", "chips_color", "_color_gauges"):
            st.session_state.pop(_k, None)
    st.session_state.update(prep=prep, source=kind, data_id=did)
    # Persist to SQLite so data survives page refreshes
    if kind == SRC_UPLOAD:
        try:
            DB.save_prep(name, kind, prep.monthly, list(prep.notes), did)
        except Exception as _db_err:
            _db_err_str = str(_db_err)
            # Friendly Spanish message; data still shows for this session
            st.warning(
                t("Los datos se cargaron correctamente para esta sesión, pero no se pudieron guardar "
                  "en la base de datos local (necesitarás volver a subir el archivo la próxima vez que recargues la página).")
            )
            if adv():
                with st.expander(t("Detalles técnicos del error de guardado"), expanded=False):
                    st.code(_db_err_str)
    base_data = prep
    prep = default_prep()                                  # con las correcciones hechas en esta pestaña

    m = prep.monthly
    last12 = m[m["ym"] > m["ym"].max() - pd.DateOffset(months=12)]
    exp_share = last12.loc[last12["mkt"] == MKT_USA, "qty"].sum() / max(last12["qty"].sum(), 1e-9) * 100
    u = ui.UNIT
    st.html(ui.status([(name if kind == SRC_UPLOAD else t(kind), t("Fuente")), (str(prep.n_months), t("Meses")),
                       (f"{mes_largo(m['ym'].min())} – {mes_largo(m['ym'].max())}", t("Periodo")), (", ".join(prep.gauges), t("Calibres")),
                       (str(m["prod"].nunique() or "—"), t("Productos")), (f"{m['qty'].sum():,.0f} {u}", t("Toneladas totales")),
                       (f"{exp_share:.1f}%", t("Exportación (12 m)"))]))
    st.html(ui.h2(t("Notas de calidad de los datos")))
    st.html("<ul>" + "".join(f"<li>{esc(note_t(n))}</li>" for n in (prep.notes or [t("Sin observaciones.")])) + "</ul>")
    c1, c2 = st.columns([1, 2], gap="large")
    by = m.groupby("cal")["qty"].sum().reindex(prep.gauges)
    with c1:
        st.html(ui.table([t("Calibre"), t("Toneladas")], [[ui.cal_label(g), f"{v:,.1f}"] for g, v in by.items()], num=(1,)))
    with c2:
        tot = m.groupby("ym")["qty"].sum()
        st.plotly_chart(ui.bar_fig([ui.mes(x) for x in tot.index], {t("Toneladas"): tot.values}, {t("Toneladas"): ui.RED}, height=260),
                        width="stretch", theme=None, config={"displayModeBar": False})
    data_editor_block(base_data)

    if adv():
        with st.expander(t("Diagnóstico de carga"), expanded=False):
            _diag_upload(raw, base_data, dict(mp), kind, name)

    # ── Datos guardados (gestión de archivos) ─────────────────────────────────
    _file_manager_block()

    st.html(ui.h2(t("Compatible con SAP"), t("Mientras no haya conexión directa, la app lee exportaciones estándar de SAP S/4HANA y reconoce sus nombres de columna.")))
    st.html("<ul class='hint'>" + "".join(f"<li>{esc(t(x))}</li>" for x in (
        "Facturas (VF05, o SE16N sobre VBRK/VBRP): Fe.factura, Material, Texto breve de material, Solicitante, Ctd.facturada o Peso neto.",
        "Si el peso viene en kilogramos, marca la casilla de kilogramos y se convierte a toneladas.",
        "País del cliente (LAND1) para separar la exportación a EE. UU.")) + "</ul>")
    st.html(ui.h2(t("Tus ventas se quedan en tu navegador"), t("El archivo se procesa en tu computadora o en la sesión de la app y no se publica. "
                                                              "La versión publicada arranca con datos de ejemplo y cada persona sube su propio archivo.")))



# ================================================================== corregir ventas dentro de la app
def horizon_default() -> int:
    return int(sv("seg_h", 6))


def total_forecast(m, ml, prices=None) -> float:
    """Pronóstico total (automático, todos los calibres) a H meses con el estado actual de los controles."""
    prices = prices or get_prices()
    gauges = sorted({g for g in m["cal"].unique()}, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 0))
    pr = cached_project(m, tuple((g,) for g in gauges), None, horizon_default(), E.AUTO, E.TON, "sum", get_mult(), get_sig(), get_w(), prices, ml,
                        get_crm(None, E.TON, prices), apply_signals_on())
    return float(pr.fc.sum())


def data_editor_block(base: D.Prepared):
    """Tabla editable de toneladas por mes y calibre. Las correcciones se guardan aparte y se aplican a todos los cálculos."""
    st.html(ui.h2(t("Corrige o agrega ventas"), t("Cambia las toneladas de un mes y calibre, o escribe el mes siguiente. El pronóstico, el plan y las gráficas "
                                                  "se recalculan al instante. Los cambios no modifican tu archivo.")))
    mb = base.monthly
    gauges = list(base.gauges)
    months = list(pd.date_range(mb["ym"].min(), mb["ym"].max(), freq="MS"))
    piv = mb.groupby(["ym", "cal"])["qty"].sum().unstack("cal").reindex(index=months, columns=gauges).fillna(0.0)
    edits = dict(store().get("data_edits", {}))
    extra = sorted({pd.Timestamp(k[0]) for k in edits if pd.Timestamp(k[0]) > months[-1]})
    all_m = months + [x for x in extra if x not in months]
    all_m = sorted(set(all_m))
    nxt = all_m[-1] + pd.offsets.MonthBegin(1)
    tbl = pd.DataFrame(np.nan, index=all_m + [nxt], columns=gauges)
    tbl.loc[months, gauges] = piv.values
    for (ym, cal), v in edits.items():
        tbl.loc[pd.Timestamp(ym), cal] = v
    keep = st.radio(t("Meses a mostrar"), [12, 24, 0], index=0, horizontal=True, key="seg_rows", format_func=lambda x: t("Todos") if x == 0 else t("{n} meses", n=x))
    shown = tbl.iloc[-(keep + 1):] if keep else tbl
    view = shown.copy()
    view.insert(0, t("Mes"), [x.strftime("%Y-%m") for x in view.index])
    view.columns = [view.columns[0]] + [ui.cal_label(g) for g in gauges]
    ver = sv("edit_ver", 0)
    ed = st.data_editor(view, hide_index=True, width="stretch", height=min(35 * len(view) + 40, 520), key=f"ed_sales_{ver}_{keep}_{ST.cfg()['lang']}", disabled=[view.columns[0]],
                        column_config={ui.cal_label(g): st.column_config.NumberColumn(format="%.1f", step=0.1) for g in gauges})
    changed, new_edits = False, dict(edits)
    for i, ym in enumerate(shown.index):
        for g in gauges:
            v, old = ed.iloc[i][ui.cal_label(g)], shown.iloc[i][g]
            if pd.isna(v) or (not pd.isna(old) and abs(v - old) < 1e-9):
                continue
            key = (ym.strftime("%Y-%m"), g)
            base_v = piv.loc[ym, g] if ym in piv.index else None
            if base_v is not None and abs(v - base_v) < 1e-9:
                new_edits.pop(key, None)
            else:
                new_edits[key] = float(v)
            changed = True
    if changed:
        store()["data_edits"] = new_edits
        store()["edit_ver"] = ver + 1
        st.rerun()
    if not edits:
        st.html(f'<p class="hint">{esc(t("Escribe sobre cualquier número para corregirlo. En la fila del mes siguiente puedes capturar su venta cuando la tengas."))}</p>')
        return
    # calibres vacíos en un mes nuevo
    for ym in extra:
        miss = [ui.cal_label(g) for g in gauges if pd.isna(tbl.loc[ym, g])]
        if miss:
            st.warning(t("En {m} faltan: {x}. Los calibres vacíos cuentan como 0 t.", m=mes_largo(ym), x=", ".join(miss)))
    # impacto de las correcciones
    m_edit = default_prep().monthly
    ml_e = get_ml(m_edit, E.windows_for(m_edit))
    ml_b = get_ml(mb, E.windows_for(mb), edited=False)
    f_new, f_old = total_forecast(m_edit, ml_e), total_forecast(mb, ml_b)
    d = (f_new / f_old - 1) * 100 if f_old else 0.0
    u = ui.UNIT
    st.html(ui.kpis([(str(len(edits)), t("Valores corregidos")), (f"{f_new:,.0f} {u}", t("Pronóstico {h} meses con las correcciones", h=horizon_default())),
                     (f"{f_old:,.0f} {u}", t("Pronóstico {h} meses sin ellas", h=horizon_default())),
                     (Raw(f'<span class="{"neg" if d < 0 else ""}">{d:+.1f}%</span>'), t("Cambio por las correcciones"))]))
    rows = []
    for (ym, cal), v in sorted(edits.items()):
        old = piv.loc[pd.Timestamp(ym), cal] if pd.Timestamp(ym) in piv.index else None
        rows.append([ym, ui.cal_label(cal), f"{old:,.1f}" if old is not None else "—", f"{v:,.1f}",
                     f"{v - old:+,.1f}" if old is not None else f"{v:+,.1f}"])
    st.html(ui.table([t("Mes"), t("Calibre"), t("Antes"), t("Ahora"), t("Diferencia")], rows, num=(2, 3, 4)))
    if st.button(t("Quitar todas las correcciones"), key="edits_reset"):
        store().pop("data_edits", None)
        store()["edit_ver"] = ver + 1
        st.rerun()


def edits_note():
    n = len(store().get("data_edits", {}))
    if n:
        st.html(f'<p class="hint" style="margin:0 0 10px">{esc(t("Incluye {n} correcciones de ventas hechas en la pestaña Datos.", n=n))}</p>')


# ================================================================== sección Pronóstico
MEAS = {"Toneladas": E.TON, "Ventas $": E.MXN}
MODE_LBL = {"sum": "Suma de calibres", "direct": "Total directo"}
HDR_TOP, HDR_OTH = "__top__", "__oth__"


def method_label(i, bt):
    if i == E.AUTO:
        return t("Automático: el más certero")
    if i == HDR_TOP:
        return "── " + t("Más certeros para esta selección") + " ──"
    if i == HDR_OTH:
        return "── " + t("Otros métodos") + " ──"
    r = bt[bt["Método"] == i]
    if len(r):
        few = "" if bool(r.iloc[0]["Elegible"]) else " · " + t("pocas ventanas")
        return f"{t(i)} — {ui.acc_txt(ui.acc_key(r.iloc[0]['Error %']))} ({r.iloc[0]['Precisión %']:.1f}%){few}"
    return t(i)


def forecast_excel(pr, lo, hi, methods_df, unit, horizon) -> bytes:
    """Excel con el pronóstico del total, el pronóstico por calibre y la comparación de métodos."""
    months = [mes_largo(x) for x in pr.index]
    main = pd.DataFrame({t("Mes"): months, t("Pronóstico ({u})", u=unit): np.round(pr.fc, 2), t("Mínimo"): np.round(lo, 2), t("Máximo"): np.round(hi, 2)})
    rows = []
    for d in pr.detail:
        r = {t("Calibre"): d["label"], t("Método"): t(d["method"]), t("Precisión %"): round(max(0, 100 - d["error"]), 1) if np.isfinite(d["error"]) else None,
             t("Tipo de demanda"): t(d["demand"])}
        r.update({mth: round(float(v), 2) for mth, v in zip(months, d["fc"])})
        r[t("Total")] = round(float(np.sum(d["fc"])), 2)
        rows.append(r)
    by = pd.DataFrame(rows)
    mt = methods_df.copy()
    mt["Método"] = mt["Método"].map(t)
    mt = mt.rename(columns={"Error %": t("Error MAPE %"), "Sesgo %": t("Sesgo %"), "Total": t("Total {h} m ({u})", h=horizon, u=unit)})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        main.to_excel(xw, sheet_name=t("Pronóstico"), index=False)
        by.to_excel(xw, sheet_name=t("Por calibre"), index=False)
        mt.to_excel(xw, sheet_name=t("Métodos"), index=False)
    return buf.getvalue()


def section_pronostico():
    prep = default_prep()
    m = prep.monthly
    ml = get_ml(m, E.windows_for(m))
    prices, mult, sig, w = get_prices(), get_mult(), get_sig(), get_w()
    A = adv()
    u = ui.UNIT
    left, right = layout()
    with left:
        units, market, horizon, color_filter = selection_controls(prep)
        meas = wid(st.radio, t("Ver pronóstico en"), "seg_meas", "Toneladas", options=list(MEAS), horizontal=True, format_func=t)
        measure = MEAS[meas]
        allg = tuple(g for x in units for g in x)
        m_view = m if color_filter is None else m[m["color"].isin(color_filter)]
        bt = cached_bt(m_view, allg, market, measure, prices, ml)
        method, mode = E.AUTO, "sum"
        best_name = E.best_method(bt) if len(bt) else E.AUTO
        if A:
            elig = list(bt.loc[bt["Elegible"], "Método"])
            top = elig[:3]
            rest = [x for x in bt["Método"] if x not in top] + [E.SIGNAL]
            ids = [E.AUTO, HDR_TOP, *top, HDR_OTH, *rest]
            method = wid(st.selectbox, t("Método principal"), "pf_method", E.AUTO, options=ids, format_func=lambda i: method_label(i, bt))
            if method in (HDR_TOP, HDR_OTH):
                method = st.session_state["pf_method"] = store()["pf_method"] = E.AUTO
            if method not in (E.AUTO, best_name) and len(bt):
                st.caption(t("Hay un método más certero para esta selección: {m} ({p}% de precisión).", m=t(best_name),
                             p=f"{bt.loc[bt['Método'] == best_name, 'Precisión %'].iloc[0]:.1f}"))
            if len(units) > 1:
                mode = wid(st.radio, t("Total de calibres"), "seg_mode", "sum", options=list(MODE_LBL), horizontal=True, format_func=lambda x: t(MODE_LBL[x]))
                st.caption(t("Suma de calibres hace que el total cuadre con la tabla por calibre."))
            st.html(ui.group_title(t("Modelo por señales")))
            wid(st.slider, t("Peso del histórico"), "sig_w", 30, min_value=0, max_value=100, step=5, format="%d%%")
            st.caption(t("Con 0% la base solo usa los últimos tres meses."))
            st.pills(t("Escenario de mercado"), list(SIGNAL_PRESETS), selection_mode="single", key="chips_preset", format_func=t,
                     on_change=lambda: apply_preset(st.session_state["chips_preset"]) if st.session_state.get("chips_preset") else None)
        export_slot = st.container()

    pr = cached_project(m_view, tuple(tuple(x) for x in units), market, horizon, method, measure, mode, mult, sig, w, prices, ml, get_crm(None, measure, prices), apply_signals_on())
    money = measure == E.MXN
    unit = "MXN" if money else u
    fv = (lambda v: ui.mxn(v)) if money else (lambda v: f"{v:,.0f} {u}")
    fc, hist = pr.fc, pr.hist
    lo, hi = pr.lo, pr.hi
    prev = float(pr.table["Mismos meses año anterior"].sum())
    yoy = (fc.sum() / prev - 1) * 100 if prev > 0 else None
    auto = pr if method == E.AUTO else cached_project(m_view, tuple(tuple(x) for x in units), market, horizon, E.AUTO, measure, mode, mult, sig, w, prices, ml, get_crm(None, measure, prices), apply_signals_on())
    sum_mode = mode == "sum" and len(units) > 1
    main_name = t(MODE_LBL["sum"]) if sum_mode else t(pr.method)
    cls = M.demand_class(hist.values)[0]

    with right:
        scope = scope_text(units, market, len(prep.gauges))
        if color_filter:
            scope += " · " + ", ".join(sorted(color_filter))
        head = (t("Esperamos ventas por {v} {scope} en los próximos {h} meses con el método {m}.", v=fv(fc.sum()), scope=scope, h=horizon, m=main_name) if money
                else t("Esperamos {v} {scope} en los próximos {h} meses con el método {m}.", v=fv(fc.sum()), scope=scope, h=horizon, m=main_name))
        k = ui.acc_key(auto.error)
        sub = (t("Suma de calibres: el total suma cada calibre con su método más certero.") if sum_mode
               else t("{m} es el método más certero aquí: {p}% de precisión ({a}).", m=t(auto.method), p=f"{max(0, 100 - auto.error):.1f}", a=ui.acc_txt(k).lower()))
        if method not in (E.AUTO, auto.method) and not sum_mode:
            sub += " " + t("Estás viendo {m} como método principal.", m=t(method))
        if mult != 1.0:
            base_pr = cached_project(m_view, tuple(tuple(x) for x in units), market, horizon, method, measure, mode, 1.0, sig, w, prices, ml, get_crm(None, measure, prices), apply_signals_on())
            d_t = fc.sum() - base_pr.fc.sum()
            tf = get_tariff()
            sub += " " + t("Con arancel de {s}% en lugar de {c}%, la exportación a EE. UU. cambia {x}% y el total queda {d} {dir}.",
                           s=f"{tf['scenario']:.0f}", c=f"{tf['current']:.0f}", x=f"{(mult - 1) * 100:+.1f}", d=fv(abs(d_t)),
                           dir=t("arriba") if d_t >= 0 else t("abajo"))
        if yoy is not None:
            sub += " " + t("Queda {p}% {dir} del mismo periodo del año anterior.", p=f"{abs(yoy):.1f}", dir=t("arriba") if yoy >= 0 else t("abajo"))
        edits_note()
        whatif_panel(prep)
        levers = changed_levers()
        if levers:
            bs = neutral(store())
            bpr = cached_project(m_view, tuple(tuple(x) for x in units), market, horizon, method, measure, mode, get_mult(bs), get_sig(bs), get_w(bs), get_prices(bs), ml,
                                 get_crm(bs, measure, get_prices(bs)), apply_signals_on(bs))
            whatif_strip([delta_chip(t("Pronóstico {h} meses", h=horizon), float(fc.sum()), float(bpr.fc.sum()), fv)], levers)
        else:
            whatif_strip([], [])
        with st.container(key="result"):
            st.html(ui.headline(head, sub))
            st.html(f'<p class="hint" style="margin:-14px 0 18px">{esc(t("Tipo de demanda:"))} {ui.dclass_html(cls)} · {esc(t(DCLASS_NOTE.get(cls, "")))}</p>')
            acc = ui.acc_html(pr.error, short=True)
            st.html(ui.kpis([(fv(fc[0]), t("Venta pronosticada {mes} ({u})", mes=mes_largo(pr.index[0]), u=unit)),
                             (fv(fc.sum()), t("Venta total {h} meses ({u})", h=horizon, u=unit)),
                             (Raw(f"{max(0, 100 - pr.error):.1f}%"), t("Precisión en prueba histórica"), acc),
                             (f"{fv(lo[0])} – {fv(hi[0])}", t("Rango del 80% del próximo mes"))], cols=4))
        fig_slot = st.container()
        ctl_slot = st.container()

        cons = None
        if A and len(units) == 1 and len(units[0]) == 1:
            cons = consensus_block(units[0][0], market, measure, horizon, method, fc, pr.index, unit)
        if len(pr.detail) > 1 or len(units) > 1:
            calibre_table(pr, horizon, fv, unit)
        if color_filter is None and "color" in m.columns and not money:
            _color_forecast_section(m, pr, allg, horizon, unit)
        if A:
            waterfall_block(horizon)
        else:
            with st.expander(t("Por qué cambió el pronóstico")):
                waterfall_block(horizon)
        if A:
            explain_block(ml, units, market, horizon, pr.index, len(prep.gauges))
        m_raw = monthly_edited()
        det = cached_detect(m_raw)
        if A:
            atypical_block(m_raw, det, allg, E.windows_for(m_raw))
        methods_df = pd.DataFrame()
        if A:
            methods_df = compare_table(m_view, allg, market, horizon, measure, prices, ml, pr, bt, w, sig, mult, mode, units, method, fv, unit)
            ml_block(ml, m, units, market, measure, prices)
            lowvol_block(m, allg)
            tracking_block(m, ml, allg)

        with ctl_slot:
            c1, c2 = st.columns([1, 3])
            show_trend = wid(st.toggle, t("Tendencia de 12 meses"), "tg_trend", True)
            cmp_names = []
            if A:
                opts = [x for x in list(bt["Método"])[:5] if x != pr.method]
                with c2:
                    cmp_names = st.pills(t("Comparar con otros métodos"), opts, selection_mode="multi", key="chips_cmp", format_func=t) or []
        marks = atypical_marks(det, allg, hist) if A else []
        compare = [(t(n), cached_project(m_view, tuple(tuple(x) for x in units), market, horizon, n, measure, mode, mult, sig, w, prices, ml, get_crm(None, measure, prices), apply_signals_on()).fc) for n in cmp_names]
        with fig_slot:
            st.plotly_chart(ui.forecast_fig(list(hist.index), hist.values, list(pr.index), fc, lo, hi, unit, main_name, cons, money=money,
                                            trend=show_trend, compare=compare, marks=marks), width="stretch", theme=None, config={"displayModeBar": False})

    with export_slot:
        if not methods_df.empty:
            ex_methods = methods_df
        else:
            ex_methods = bt[["Método", "Error %", "Sesgo %"]].assign(Total=[np.nan] * len(bt))
        st.download_button(t("Exportar a Excel"), forecast_excel(pr, lo, hi, ex_methods, unit, horizon), file_name=t("pronostico-viakable.xlsx"),
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_forecast", width="stretch")


def consensus_block(cal, market, measure, horizon, method, fc, idx, unit):
    st.html(ui.h2(t("Pronóstico de consenso"), t("Ajusta el pronóstico estadístico del calibre {c} AWG con lo que sabe el equipo comercial. "
                                                "Cada ajuste necesita una justificación.", c=cal)))
    div = 1e6 if measure == E.MXN else 1
    ck = f"{cal}|{market}|{measure}|{horizon}|{method}"
    saved = store().setdefault("consensus", {}).get(ck, {"adj": [0.0] * horizon, "why": [""] * horizon})
    months = [mes_largo(x) for x in idx]
    c_mes, c_adj, c_why = t("Mes"), t("Ajuste %"), t("Justificación")
    c_stat = t("Estadístico ({u})", u="M$" if div > 1 else ui.UNIT)
    df = pd.DataFrame({c_mes: months, c_stat: (fc / div).round(2), c_adj: saved["adj"], c_why: saved["why"]})
    ed = st.data_editor(df, hide_index=True, width="stretch", key=f"cons_{ck}_{ST.cfg()['lang']}", disabled=[c_mes, c_stat],
                        column_config={c_adj: st.column_config.NumberColumn(min_value=-100, max_value=300, step=1),
                                       c_why: st.column_config.TextColumn(width="large")})
    adj, why = ed[c_adj].fillna(0).tolist(), ed[c_why].fillna("").tolist()
    store()["consensus"][ck] = {"adj": adj, "why": why}
    out, bad = E.consensus(fc, adj, why)
    if bad:
        st.warning(t("No se aplicó el ajuste de: {x} (falta justificación).", x=", ".join(months[i] for i in bad)))
    if not any(a != 0 for a in adj):
        return None
    st.html(ui.kpis([(f"{out.sum() / div:,.1f}" + (" M$" if div > 1 else f" {ui.UNIT}"), t("Total con consenso"),
                      t("{p}% contra el estadístico", p=f"{(out.sum() / fc.sum() - 1) * 100:+.1f}"))], cols=4))
    return out


def _color_forecast_section(monthly, pr, selected_gauges, horizon, unit):
    """Pronóstico por color — top-down reconciled color breakdown.

    Shown when color_filter is None (Todos colors) and data has a color column.
    For each calibre the calibre-level forecast is distributed by historical
    color shares so all color forecasts sum exactly to the calibre total.
    """
    from viakable_forecast.color import color_forecast_breakdown, COLOR_NA, COLOR_MULTICOLOR

    cdf = color_forecast_breakdown(monthly, pr.detail, list(selected_gauges))
    if cdf.empty or cdf["fc_total"].sum() == 0:
        return

    with st.expander(t("Pronóstico por color"), expanded=False):
        st.html(f'<p class="sub">'
                + esc(t("Distribución del pronóstico por color según el mix histórico de cada calibre. "
                         "Los totales por calibre cuadran exactamente con el pronóstico principal."))
                + "</p>")

        # Build the Plotly stacked-bar chart (calibres on x, stacked by color)
        import plotly.graph_objects as go

        cals = cdf["cal"].unique().tolist()
        colors_order = (
            cdf.groupby("color")["fc_total"].sum()
            .sort_values(ascending=False)
            .index.tolist()
        )
        # Put Sin color / N/A and Multicolor at the end
        priority = [c for c in colors_order if c not in (COLOR_NA, COLOR_MULTICOLOR)]
        extras   = [c for c in (COLOR_MULTICOLOR, COLOR_NA) if c in colors_order]
        colors_order = priority + extras

        fig = go.Figure()
        for color in colors_order:
            row = cdf[cdf["color"] == color]
            y_vals = [float(row.loc[row["cal"] == cal, "fc_total"].sum()) for cal in cals]
            hex_c = _COLOR_HEX.get(color, "#aaaaaa")
            # gradient colors: use first hex stop
            if "gradient" in hex_c:
                hex_c = "#7b4b2a"
            fig.add_trace(go.Bar(
                name=t(color),
                x=[f"{cal} AWG" for cal in cals],
                y=y_vals,
                marker_color=hex_c,
                hovertemplate=f"{t(color)}: %{{y:,.1f}} {unit}<extra></extra>",
            ))
        fig.update_layout(
            barmode="stack",
            height=240,
            margin=dict(l=0, r=0, t=8, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(size=11),
            xaxis=dict(showgrid=False),
            yaxis=dict(title=unit, gridcolor="rgba(128,128,128,.15)"),
        )
        st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})

        # Summary table: color | share % | tons per calibre
        rows_html = []
        for color in colors_order:
            bg = _COLOR_HEX.get(color, "#cccccc")
            border = "border:1px solid rgba(128,128,128,.25);" if color in ("Blanco", COLOR_NA) else ""
            swatch = (f'<i style="display:inline-block;width:8px;height:8px;border-radius:50%;'
                      f'background:{bg};{border}margin-right:6px;flex-shrink:0"></i>')
            label  = esc(t(color))
            # One column per calibre
            cells = ""
            for cal in cals:
                row = cdf[(cdf["color"] == color) & (cdf["cal"] == cal)]
                if row.empty:
                    cells += '<td class="num">—</td>'
                else:
                    share = row["share_pct"].iloc[0]
                    tons  = row["fc_total"].iloc[0]
                    cells += f'<td class="num">{share:.1f}%&nbsp;&nbsp;{tons:,.0f}&nbsp;{esc(unit)}</td>'
            rows_html.append(
                f"<tr><td><span style='display:inline-flex;align-items:center'>{swatch}{label}</span></td>{cells}</tr>"
            )

        th_cals = "".join(f"<th>{cal} AWG</th>" for cal in cals)
        table = (
            '<div class="tablewrap"><table>'
            f"<thead><tr><th>{esc(t('Color'))}</th>{th_cals}</tr></thead>"
            "<tbody>" + "".join(rows_html) + "</tbody>"
            "</table></div>"
        )
        st.html(table)


def calibre_table(pr, horizon, fv, unit):
    st.html(ui.h2(t("Pronóstico por calibre"), t("Cada calibre se pronostica por separado con su método más certero. El cambio compara contra los mismos "
                                                "{h} meses del año anterior.", h=horizon)))
    det, tb = pr.detail, pr.table
    l12 = [float(d["y"][-12:].sum()) for d in det]
    tot12 = sum(l12) or 1
    mx = max(v / tot12 * 100 for v in l12) if l12 else 1
    rows = []
    for d, tr, v in zip(det, tb.itertuples(), l12):
        chg = tr._8
        rows.append([Raw(f"<b>{esc(ui.cal_label(d['label']) if d['label'].replace(' + ', '').isdigit() else d['label'])}</b> {ui.dclass_html(d['demand'])}"),
                     fv(v), Raw(f"<b>{esc(fv(d['fc'].sum()))}</b>"),
                     Raw(f'<span class="{"neg" if chg < 0 else ""}">{chg:+.1f}%</span>') if np.isfinite(chg) else "—",
                     ui.mix_bar(v / tot12 * 100, mx), ui.spark_svg(d["y"], d["fc"]), t(d["method"]), ui.acc_html(d["error"])])
    rows.append([Raw(f"<b>{esc(t('Total'))}</b>"), Raw(f"<b>{esc(fv(tot12))}</b>"), Raw(f"<b>{esc(fv(pr.fc.sum()))}</b>"), "", "", "", "", ""])
    st.html(ui.table([t("Calibre"), t("Últimos 12 m ({u})", u=unit), t("Próximos {h} m ({u})", h=horizon, u=unit), t("Cambio"), t("Mezcla"), t("Tendencia"),
                      t("Método"), t("Certeza")], rows, num=(1, 2, 3), min_w=900))


def compare_table(m, allg, market, horizon, measure, prices, ml, pr, bt, w, sig, mult, mode, units, method, fv, unit):
    wins = E.windows_for(m)
    st.html(ui.h2(t("Comparativo de métodos"), t("Ordenados del más al menos certero. Cada método se evaluó con validación cruzada de origen móvil: {w} ventanas "
                                                "de {h} meses, entrenando solo con datos anteriores a cada una. El error es el promedio de las ventanas.", w=len(wins), h=wins[0][1])))
    tot = cached_totals(m, allg, market, horizon, measure, prices, bt, ml)
    sig_pr = cached_project(m, tuple(tuple(x) for x in units), market, horizon, E.SIGNAL, measure, "direct" if len(units) > 1 else mode, mult, sig, w, prices, ml, get_crm(None, measure, prices), apply_signals_on())
    sr = E.signal_row(pr.hist.values, w)
    allrows = pd.concat([bt, pd.DataFrame([sr])]).reset_index(drop=True)
    allrows["Elegible"] = allrows["Elegible"].fillna(True).astype(bool)
    allrows["_o"] = (~allrows["Elegible"]).astype(int)
    allrows = allrows.sort_values(["_o", "Error %"], na_position="last").reset_index(drop=True)
    best = E.best_method(bt)
    rows, main, totals = [], [], []
    for i, r in allrows.iterrows():
        name = r["Método"]
        tv = sig_pr.fc.sum() if name == E.SIGNAL else tot.get(name)
        totals.append(tv if tv is not None else np.nan)
        tag = Raw(f'<span class="tag">{esc(t("Recomendado"))}</span>') if name == best else ""
        if not r["Elegible"]:
            tag = Raw(f'<span class="tag" style="border-color:var(--line);color:var(--muted)">{esc(t("pocas ventanas"))}</span>')
        wgt = f"{r['Peso ensamble'] * 100:.0f}%" if pd.notna(r.get("Peso ensamble")) else "—"
        wgs = f"{r['Peso apilado'] * 100:.0f}%" if pd.notna(r.get("Peso apilado")) else "—"
        spread = f"{r['Error mín %']:.1f}–{r['Error máx %']:.1f}%" if pd.notna(r.get("Error mín %")) else "—"
        rows.append([Raw(f"<b>{esc(t(name))}</b>{tag}"), t(E.TYPE_OF.get(name, "")), Raw(f'<span class="wrap" style="color:var(--muted)">{esc(t(METHOD_DESC.get(name, "")))}</span>'),
                     ui.acc_html(r["Error %"]), f"{r['Error %']:.1f}%", spread, f"{r['Sesgo %']:+.1f}%", wgt, wgs, fv(tv) if tv is not None else "—"])
        if (method == E.AUTO and name == best) or name == method:
            main.append(i)
    st.html(ui.table([t("Método"), t("Tipo"), t("Cómo funciona"), t("Certeza"), t("Error MAPE"), t("Mejor y peor ventana"), t("Sesgo"), t("Peso en el ensamble"),
                      t("Peso en el apilado"), t("Total {h} m ({u})", h=horizon, u=unit)], rows, num=(4, 5, 6, 7, 8, 9), min_w=1180, main_rows=main))
    wl = " · ".join(f"{t('hace')} {k} {t('m')}: {e:.1f}%" for (k, _), e in zip(wins, pr.window_errors))
    if wl:
        st.html(f'<p class="syncnote">{esc(t("Error del total en cada ventana"))}: {esc(wl)}</p>')
    st.html(f'<p class="syncnote">{esc(t("Sesgo positivo = el método se pasa. Muy certero <5% · Certero <10% · Aceptable <20% · Poco confiable ≥20% de error. Holt-Winters y la regresión necesitan 24 o más meses de entrenamiento y solo entran al ranking cuando tienen al menos 3 ventanas. El ensamble pondera cada método por el inverso de su error."))}</p>')
    return allrows[["Método", "Error %", "Sesgo %"]].assign(Total=totals)


def spec_label(spec) -> str:
    kind, p = spec
    if kind in ("gbr", "hgb"):
        return t("{k} · {n} · prof. {d} · tasa {lr}", k="GradientBoosting" if kind == "gbr" else "HistGradientBoosting", n=p[0], d=p[1], lr=p[2])
    if kind == "xgb":
        return t("{k} · {n} · prof. {d} · tasa {lr}", k="XGBoost", n=p[0], d=p[1], lr=p[2])
    if kind == "rf":
        return t("Bosque aleatorio · {n} árboles · hoja mín. {l} · variables {f}", n=p[0], l=p[1], f=p[2])
    if kind == "ridge":
        return t("Ridge · alfa {a}", a=p[0])
    return t("MLP · capas {h} · alfa {a}", h=str(p[0]), a=p[1])


def ml_block(ml, m, units, market, measure, prices):
    st.html(ui.h2(t("Modelo de machine learning"), t("Gradient boosting entrenado con todos los productos a la vez. Aprende de la venta reciente, la "
                                                    "estacionalidad, el calibre, la familia y el mercado, y compite contra los métodos estadísticos con la misma prueba histórica.")))
    if ml is None:
        st.info(t("El modelo necesita al menos 14 meses de historia."))
        return
    mx = cached_matrix(m, tuple(tuple(x) for x in units), market, measure, prices, ml)
    vs = E.ml_vs_stat(mx)
    tot = vs[vs["Calibre"] == "Total"].iloc[0]
    f = ml["final"]
    st.html(ui.kpis([(f"{max(0, 100 - tot['Error ML %']):.1f}%", t("Precisión del ML en esta selección"), ui.acc_html(tot["Error ML %"], True)),
                     (f"{max(0, 100 - tot['Error estadístico %']):.1f}%", t("Mejor estadístico: {m}", m=t(tot["Mejor estadístico"]))),
                     (f"{f.n_series:,}", t("Series de producto usadas")), (f"{f.n_samples:,}", t("Observaciones de entrenamiento"), spec_label(ml["spec"]))]))
    d = tot["Error estadístico %"] - tot["Error ML %"]
    st.markdown(t("El modelo de ML le gana al mejor método estadístico por {d} puntos en esta selección.", d=f"{d:.1f}") if d > 0 else
                t("En esta selección, {m} es {d} puntos más certero que el ML. Con más historia, el ML suele mejorar.", m=t(tot["Mejor estadístico"]), d=f"{-d:.1f}"))
    g = vs[vs["Calibre"] != "Total"]
    none = t("ninguno")
    wins = ", ".join(g.loc[g["Gana"] == "Machine learning", "Calibre"]) or none
    loses = ", ".join(g.loc[g["Gana"] != "Machine learning", "Calibre"]) or none
    st.markdown(t("**Gana en los calibres:** {w}. **No gana en:** {l}.", w=wins, l=loses))
    rows = [[Raw(f"<b>{esc(r.Calibre)}</b>"), f"{r._2:.1f}%", t(r._3), f"{r._4:.1f}%", f"{r._5:+.1f}", t(r.Gana)] for r in vs.itertuples()]
    st.html(ui.table([t("Calibre"), t("Error ML"), t("Mejor estadístico"), t("Error estadístico"), t("Diferencia (pts)"), t("Gana")], rows, num=(1, 3, 4), min_w=700))
    with st.expander(t("Modelos probados y elegido")):
        st.html(f'<p class="hint">{esc(t("Se probaron {n} configuraciones de GradientBoosting y HistGradientBoosting con validación cruzada de origen móvil y se conservó la mejor: {d}.", n=len(ml["tuning"]), d=spec_label(ml["spec"])))}</p>')
        st.html(ui.table([t("Modelo"), t("Error de validación")], [[spec_label(r["spec"]), f"{r['Error validación %']:.1f}%"] for _, r in ml["tuning"].iterrows()], num=(1,)))
        st.html(f'<p class="syncnote">{esc(t("Cada ventana se evalúa con la configuración que se habría elegido sin ver esa ventana, y las variables nunca usan datos posteriores al mes que se predice."))}</p>')
    st.html(ui.h2(t("Qué variables usa más el modelo")))
    imp = f.importance
    st.plotly_chart(ui.importance_fig([t(x) for x in imp.index], list(imp.values * 100)), width="stretch", theme=None, config={"displayModeBar": False})
    st.html(f'<p class="syncnote">{esc(t("Importancia = reducción del error que aporta cada variable en los árboles. El modelo pronostica cada producto y luego suma, así que el total y los calibres cuadran por construcción. Con pocos meses de historia se apoya mucho en el tamaño de la serie."))}</p>')
    if st.button(t("Reentrenar modelo")):
        cached_ml.clear()
        st.rerun()


# ================================================================== sección Planeación
def kg(v):
    return f"{v:,.0f} {ui.UNIT}"


def section_planeacion():
    prep = default_prep()
    m = prep.monthly
    ml = get_ml(m, E.windows_for(m))
    has_exp = (m["mkt"] == MKT_USA).any()
    A = adv()
    u = ui.UNIT
    left, right = layout()
    with left:
        units, market, horizon, color_filter = selection_controls(prep)
        wid(st.radio, t("Nivel de servicio"), "seg_service", 95, options=[90, 95, 98, 99], horizontal=True, format_func=lambda v: f"{v}%")
        st.caption(t("Probabilidad de no quedarse sin inventario durante el tiempo de reposición."))
        wid(st.slider, t("Tiempo de reposición (meses)"), "pl_lt", 1.5, min_value=0.25, max_value=4.0, step=0.25)
        st.caption(t("Meses entre pedir metal y tener cable terminado."))
        wid(st.slider, t("Costo anual de mantener inventario"), "pl_hold", 25, min_value=5, max_value=50, step=1, format="%d%%")
        st.caption(t("Capital, almacén, seguros y mermas como porcentaje del valor."))
        if A:
            st.html(ui.group_title(t("Metal y márgenes")))
            wid(st.number_input, t("Cobre (USD/t)"), "pl_cu", 9840.0, min_value=0.0, step=10.0, disabled=px_disabled("cu"))
            wid(st.number_input, t("Aluminio (USD/t)"), "pl_al", 2600.0, min_value=0.0, step=10.0, disabled=px_disabled("al"))
            if px_disabled("cu") or px_disabled("al"):
                st.caption(t("Precio en vivo; para escribir uno manual ve a Metales."))
            wid(st.slider, t("Escenario de precio del metal"), "pl_mchg", 0, min_value=-30, max_value=30, step=1, format="%+d%%")
            st.caption(t("Mueve el costo del metal sin cambiar tu precio de venta: muestra cuánto margen se pierde o se gana."))
            wid(st.number_input, t("Tipo de cambio (MXN por USD)"), "pl_fx", 18.5, min_value=1.0, step=0.1, disabled=px_disabled("fx"))
            wid(st.number_input, t("Costo de conversión cobre (MXN/t)"), "pl_ccu", 25000.0, min_value=0.0, step=500.0)
            wid(st.number_input, t("Costo de conversión aluminio (MXN/t)"), "pl_cal", 15000.0, min_value=0.0, step=500.0)
            st.caption(t("Energía, mano de obra, aislamiento y empaque por tonelada."))
            wid(st.slider, t("Metal en las toneladas"), "pl_metalpct", 100, min_value=10, max_value=100, step=1, format="%d%%")
            st.caption(t("Si tus toneladas ya son contenido metálico (ContMet), déjalo en 100%."))
            if has_exp:
                st.html(ui.group_title(t("Exportación a EE. UU.")))
                wid(st.slider, t("Tiempo extra de reposición (meses)"), "pl_explt", 0.5, min_value=0.0, max_value=2.0, step=0.25)
                wid(st.number_input, t("Flete a EE. UU. (USD/t)"), "pl_freight", 120.0, min_value=0.0, step=10.0)
                wid(st.checkbox, t("Viakable paga el arancel (DDP)"), "pl_ddp", False)
            st.html(ui.group_title(t("Pronóstico base")))
            wid(st.selectbox, t("Método del plan"), "pl_base", E.AUTO, options=[E.AUTO, E.SIGNAL],
                format_func=lambda i: t("Automático: el más certero") if i == E.AUTO else t("Modelo por señales"))

    gauges = tuple(g for x in units for g in x)
    m_view = m if color_filter is None else m[m["color"].isin(color_filter)]
    prices, mult, sig, w, params = get_prices(), get_mult(), get_sig(), get_w(), get_params()
    ver = sv("ov_ver", 0)
    ov = {"metal": sv("ov_metal", {}), "inv": sv("ov_inv", {}), "cap": sv("ov_cap", {})}
    res = cached_plan(m_view, gauges, horizon, params, prices, mult, sig, w, ov, ml, get_crm(None, E.TON, prices))
    rows = res.rows if market is None else res.rows[res.rows["mkt"] == market]
    tot = PL.totals(res, market)
    months = [ui.mes(x) for x in res.months]

    with right:
        st.html(ui.headline(t("Plan de inventario y compras de metal para {h} meses", h=horizon) + (f" · {t(market)}" if market else ""),
                            t("Convierte el pronóstico de cada calibre y mercado en inventario de seguridad, producción, compras de metal, margen y uso "
                              "de capacidad. Edita precios, inventario y capacidad en las tablas.")))
        if rows.empty:
            st.warning(t("No hay ventas para esa selección."))
            return
        edits_note()
        levers = changed_levers()
        if levers:
            bs = neutral(store())
            res_b = cached_plan(m_view, gauges, horizon, get_params(bs), get_prices(bs), get_mult(bs), get_sig(bs), get_w(bs), ov, ml, get_crm(bs, E.TON, get_prices(bs)))
            tb_ = PL.totals(res_b, market)
            whatif_strip([delta_chip(t("Volumen {h} meses", h=horizon), tot["vol"], tb_["vol"], kg), delta_chip(t("Ingreso {h} meses", h=horizon), tot["rev"], tb_["rev"], ui.mxn),
                          delta_chip(t("Margen bruto"), tot["margin"], tb_["margin"], ui.mxn),
                          delta_chip(t("Inventario de seguridad"), tot["ss"], tb_["ss"], kg, good_up=False)], levers)
        else:
            whatif_strip([], [])
        st.html(ui.kpis([(kg(tot["ss"]), t("Inventario de seguridad")), (ui.mxn(tot["ss_value"]), t("Valor del inventario de seguridad")),
                         (Raw(f'<span class="{"neg" if tot["freed"] < 0 else ""}">{esc(ui.mxn(tot["freed"]))}</span>'), t("Capital liberado vs método tradicional"),
                          t("Error promedio {a}% contra {b}% del método tradicional", a=f"{tot['err']:.1f}", b=f"{tot['err_trad']:.1f}")),
                         (ui.mxn(tot["saving"]), t("Ahorro anual por mantener menos inventario"))]))
        st.html(ui.kpis([(ui.mxn(tot["rev"]), t("Ingreso pronosticado {h} meses", h=horizon)),
                         (Raw(f'<span class="{"neg" if tot["margin"] < 0 else ""}">{esc(ui.mxn(tot["margin"]))}</span>'), t("Margen bruto pronosticado"),
                          t("{p}% · cada +10% en el metal: -{x}", p=f"{tot['margin_pct']:.1f}", x=ui.mxn(tot["margin_lost_10"]))),
                         (kg(tot["cu"]), t("Cobre requerido {h} meses", h=horizon)), (kg(tot["al"]), t("Aluminio requerido {h} meses", h=horizon))]))
        over = tot["over_months"]
        if A:
            st.html(ui.kpis([(kg(tot["exp"]) if has_exp else "—", t("Exportación EE. UU. {h} meses", h=horizon),
                              t("{p}% del total", p=f"{tot['exp'] / max(tot['vol'], 1e-9) * 100:.1f}") if has_exp else ""),
                             (ui.mxn(tot["metal_cost"]), t("Costo del metal del pronóstico")),
                             (Raw(f'<span class="{"neg" if over else ""}">{over}</span>'), t("Meses con capacidad excedida"))], cols=3))

        st.html(ui.h2(t("Requerimiento mensual de metal"), t("Toneladas de metal a comprar cada mes según la producción sugerida.")))
        cu = sum(a for a, mt in zip(rows["metal_req"], rows["metal"]) if mt == "cu") if (rows["metal"] == "cu").any() else np.zeros(horizon)
        al = sum(a for a, mt in zip(rows["metal_req"], rows["metal"]) if mt == "al") if (rows["metal"] == "al").any() else np.zeros(horizon)
        st.plotly_chart(ui.bar_fig(months, {t("Cobre"): cu, t("Aluminio"): al}, {t("Cobre"): ui.RED, t("Aluminio"): ui.GRAY}),
                        width="stretch", theme=None, config={"displayModeBar": False})

        if A:
            st.html(ui.h2(t("Parámetros por calibre"), t("Inventario de seguridad = z × 1.25 × error × demanda media × √tiempo de reposición. El método "
                                                         "tradicional necesitaría más inventario por ser menos certero.")))
            tb = []
            for r in rows.itertuples():
                mk_b = ui.badge(t("EE. UU."), "fill") if r.is_exp else t(r.mkt)
                tb.append([Raw(f"<b>{esc(ui.cal_label(r.cal))}</b>"), mk_b, t("Aluminio") if r.metal == "al" else t("Cobre"), f"{r.mean:,.1f}", ui.acc_html(r.error, True),
                           Raw(f"<b>{r.ss:,.1f}</b>"), f"{r.rop:,.1f}", f"{r.inv:,.1f}", f"{r.prod_t:,.0f}",
                           f"{r.price:,.0f} " + ("USD/t" if r.is_exp else "MXN/t"), ui.mxn(r.rev),
                           Raw(f'<span class="{"neg" if r.margin < 0 else ""}">{esc(ui.mxn(r.margin))}</span>'),
                           Raw(f'<span class="{"neg" if r.freed < 0 else ""}">{esc(ui.mxn(r.freed))}</span>')])
            st.html(ui.table([t("Calibre"), t("Mercado"), t("Metal"), t("Demanda mensual ({u})", u=u), t("Certeza"), t("Inv. seguridad ({u})", u=u),
                              t("Punto de reorden ({u})", u=u), t("Inventario actual ({u})", u=u), t("Producción {h} m ({u})", h=horizon, u=u), t("Precio"),
                              t("Ingreso MXN"), t("Margen MXN"), t("Capital liberado MXN")], tb, num=(3, 5, 6, 7, 8, 9, 10, 11, 12), min_w=1240))
            st.html(f'<p class="syncnote">{esc(t("Precios ilustrativos: edítalos con los de Viakable."))}</p>')
            edit_rows(rows, ver, prices)

        cap = res.capacity
        st.html(ui.h2(t("Capacidad de planta por calibre"), t("Producción requerida de todos los mercados como porcentaje de la capacidad mensual. Por omisión, la "
                                                              "capacidad es 10% arriba del mejor mes de los últimos dos años; edítala con la real.")))
        cr = []
        for c in cap.sort_values("cal", key=lambda s: s.map(lambda x: int(x) if str(x).isdigit() else 99)).itertuples():
            cells = [Raw(f"<b>{esc(ui.cal_label(c.cal))}</b>" + (f' <span class="badge warn">{esc(t("Cuello de botella"))}</span>' if c.over else "")), f"{c.cap:,.1f}"]
            cells += [Raw(f'<span style="{"color:var(--accent);font-weight:700" if x > 100 else "font-weight:700" if x > 90 else ""}">{x:.0f}%</span>') for x in c.util]
            cells.append(Raw(f'<b class="neg">{c.deficit:,.0f}</b>') if c.deficit > 0 else "—")
            cr.append(cells)
        st.html(ui.table([t("Calibre"), t("Capacidad mensual ({u})", u=u), *months, t("Déficit ({u})", u=u)], cr, num=tuple(range(1, horizon + 3)), min_w=820))
        if A:
            edit_caps(cap, ver)

            st.html(ui.h2(t("Producción sugerida por mes"), t("Producción necesaria para cubrir el pronóstico y terminar cada mes con el inventario de seguridad.")))
            byc = {}
            for r in res.rows.itertuples():
                byc[r.cal] = byc.get(r.cal, np.zeros(horizon)) + r.prod
            pr_rows = [[Raw(f"<b>{esc(ui.cal_label(c))}</b>"), *[f"{x:,.0f}" for x in v], Raw(f"<b>{v.sum():,.0f}</b>")]
                       for c, v in sorted(byc.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 99)]
            main = []
            for key, lab in (("cu", "Cobre"), ("al", "Aluminio")):
                v = sum((a for a, mt in zip(res.rows["metal_req"], res.rows["metal"]) if mt == key), np.zeros(horizon))
                if np.any(v):
                    main.append(len(pr_rows))
                    pr_rows.append([Raw(f"<b>{esc(t(lab))} ({u})</b>"), *[Raw(f"<b>{x:,.0f}</b>") for x in v], Raw(f"<b>{v.sum():,.0f}</b>")])
            st.html(ui.table([t("Calibre"), *months, t("Total")], pr_rows, num=tuple(range(1, horizon + 2)), min_w=760, main_rows=main))
            scenarios_block(m, gauges, horizon, market, tot, res, ml)


def edit_rows(rows, ver, prices):
    """Tabla editable: metal, inventario actual y precio de cada renglón. Al editar, se guarda y se recalcula."""
    with st.expander(t("Editar metal, inventario actual y precio de cada renglón")):
        c_cal, c_mkt, c_met, c_inv, c_pr = t("Calibre"), t("Mercado"), t("Metal"), t("Inventario actual ({u})", u=ui.UNIT), t("Precio (MXN/t o USD/t)")
        cu_l, al_l = t("Cobre"), t("Aluminio")
        df = pd.DataFrame({c_cal: [ui.cal_label(c) for c in rows["cal"]], c_mkt: [t(x) for x in rows["mkt"]],
                           c_met: [al_l if x == "al" else cu_l for x in rows["metal"]],
                           c_inv: [round(float(x), 2) for x in rows["inv"]],
                           c_pr: [round(float(x)) for x in rows["price"]]})
        sig_ = "_".join(f"{c}{k}" for c, k in zip(rows["cal"], rows["mkt"]))
        ed = st.data_editor(df, hide_index=True, width="stretch", key=f"ed_rows_{ver}_{sig_}_{ST.cfg()['lang']}_{ui.UNIT}", disabled=[c_cal, c_mkt],
                            column_config={c_met: st.column_config.SelectboxColumn(options=[cu_l, al_l]),
                                           c_inv: st.column_config.NumberColumn(min_value=0.0, step=1.0),
                                           c_pr: st.column_config.NumberColumn(min_value=0, step=500)})
        if not ed.equals(df):
            ovm, ovi, pc = dict(sv("ov_metal", {})), dict(sv("ov_inv", {})), copy.deepcopy(sv("prices", DEFAULT_PRICES))
            for (_, r), cal, mk_, is_exp in zip(ed.iterrows(), rows["cal"], rows["mkt"], rows["is_exp"]):
                ovm[(cal, mk_)] = "al" if r[c_met] == al_l else "cu"
                ovi[(cal, mk_)] = float(r[c_inv])
                pc["usd_t" if is_exp else "mxn_t"][cal] = float(r[c_pr])
            store().update(ov_metal=ovm, ov_inv=ovi, prices=pc, ov_ver=ver + 1)
            st.rerun()


def edit_caps(cap, ver):
    with st.expander(t("Editar la capacidad mensual de cada calibre")):
        c_cal, c_cap = t("Calibre"), t("Capacidad mensual ({u})", u=ui.UNIT)
        df = pd.DataFrame({c_cal: [ui.cal_label(c) for c in cap["cal"]], c_cap: [round(float(x), 1) for x in cap["cap"]]})
        ed = st.data_editor(df, hide_index=True, width="stretch", key=f"ed_cap_{ver}_{'_'.join(cap['cal'])}_{ST.cfg()['lang']}_{ui.UNIT}", disabled=[c_cal],
                            column_config={c_cap: st.column_config.NumberColumn(min_value=0.1, step=1.0)})
        if not ed.equals(df):
            caps = dict(sv("ov_cap", {}))
            for c, v in zip(cap["cal"], ed[c_cap]):
                caps[c] = float(v)
            store().update(ov_cap=caps, ov_ver=ver + 1)
            st.rerun()


def scenarios_block(m, gauges, horizon, market, tot, res, ml):
    st.html(ui.h2(t("Escenarios guardados"), t("Guarda la combinación actual de señales, arancel, precio del metal y parámetros, y compárala contra otros.")))
    scen = store().setdefault("scenarios", [])

    def _save():
        name = st.session_state.get("scen_name", "").strip()
        if not name:
            st.session_state["_scen_msg"] = "Escribe un nombre para el escenario."
            return
        snap = {k: copy.deepcopy(v) for k, v in store().items() if k.startswith(SNAP_PREFIX) or k in ("prices", "pl_base")}
        scen[:] = [x for x in scen if x["name"] != name] + [{"name": name, "snap": snap}]
        st.session_state["scen_name"] = ""
        st.session_state["_scen_msg"] = "Escenario guardado."

    c1, c2 = st.columns([4, 1])
    c1.text_input(t("Nombre del escenario"), key="scen_name", placeholder=t("Nombre del escenario, ej. Arancel 25% y cobre +10%"), label_visibility="collapsed")
    c2.button(t("Guardar escenario"), type="primary", on_click=_save, width="stretch")
    if st.session_state.get("_scen_msg"):
        st.caption(t(st.session_state.pop("_scen_msg")))

    def metrics(snap):
        r = cached_plan(m, gauges, horizon, get_params(snap), get_prices(snap), get_mult(snap), get_sig(snap), get_w(snap),
                        {"metal": snap.get("ov_metal", sv("ov_metal", {})), "inv": sv("ov_inv", {}), "cap": sv("ov_cap", {})}, ml, get_crm(snap, E.TON, get_prices(snap)))
        return PL.scenario_metrics(r)

    cur = {"vol": tot["vol"], "exp": tot["exp"], "rev": tot["rev"], "margin": tot["margin"], "ss": tot["ss"], "metal": tot["cu"] + tot["al"],
           "over": tot["over_months"]}
    def describe(snap):
        tf = get_tariff(snap)
        return " · ".join([t("arancel {s}%", s=f"{tf['scenario']:.0f}"), t("metal {s}%", s=f"{snap.get('pl_mchg', 0):+.0f}"),
                          t("TC {s}", s=snap.get("pl_fx", PLAN_DEFAULTS["fx"]))])

    entries = [(t("Actual (sin guardar)"), "", cur)] + [(x["name"], describe(x["snap"]), metrics(x["snap"])) for x in scen]
    rows = []
    for name, desc, mt in entries:
        nm = Raw(f"<b>{esc(name)}</b>" + (f'<br><span class="syncnote">{esc(desc)}</span>' if desc else ""))
        rows.append([nm, f"{mt['vol']:,.0f}", f"{mt['exp']:,.0f}" if mt["exp"] else "—", f"{mt['metal']:,.0f}", ui.mxn(mt["rev"]),
                     Raw(f"{esc(ui.mxn(mt['margin']))}<br><span class='syncnote'>{mt['margin'] / mt['rev'] * 100 if mt['rev'] else 0:.1f}%</span>"),
                     f"{mt['ss']:,.1f}", f"{mt['over']}"])
    u = ui.UNIT
    st.html(ui.table([t("Escenario"), t("Volumen {h} m ({u})", h=horizon, u=u), t("Exportación ({u})", u=u), t("Metal ({u})", u=u), t("Ingreso MXN"), t("Margen"),
                      t("Inv. seguridad ({u})", u=u), t("Meses sin capacidad")], rows, num=(1, 2, 3, 4, 5, 6, 7), min_w=900, main_rows=(0,)))
    for i, x in enumerate(scen):
        def _apply(snap=x["snap"]):
            for k, v in snap.items():
                store()[k] = copy.deepcopy(v)
                if k in st.session_state:
                    st.session_state[k] = copy.deepcopy(v)
            store()["ov_ver"] = sv("ov_ver", 0) + 1

        def _del(name=x["name"]):
            scen[:] = [y for y in scen if y["name"] != name]

        c = st.columns([3, 1, 1, 3])
        c[0].caption(x["name"])
        c[1].button(t("Aplicar"), key=f"apply_{i}", on_click=_apply)
        c[2].button(t("Eliminar"), key=f"del_{i}", on_click=_del)


# ================================================================== sección Señales
def section_senales():
    st.html(ui.headline(t("Señales que anticipan pedidos antes de que lleguen."),
                        t("Cada señal tiene un tiempo de anticipación y una sensibilidad: cuánto se mueve la demanda por cada 1% de cambio en la señal. "
                          "Solo afectan al modelo por señales. Los valores de lectura son de ejemplo.")))
    st.pills(t("Escenario de mercado"), list(SIGNAL_PRESETS), selection_mode="single", key="chips_preset", format_func=t,
             on_change=lambda: apply_preset(st.session_state["chips_preset"]) if st.session_state.get("chips_preset") else None)
    wid(st.slider, t("Peso del histórico en el modelo por señales"), "sig_w", 30, min_value=0, max_value=100, step=5, format="%d%%")
    mx = [n for n, b in SIGNALS.items() if b["mkt"] == "MX"]
    us = [n for n, b in SIGNALS.items() if b["mkt"] == "US"]
    st.html(ui.h2(t("Mercado nacional")))
    st.html('<div class="sigtop"></div>')
    for row in range(0, len(mx), 3):
        cols = st.columns(3)
        for c, name in zip(cols, mx[row:row + 3]):
            with c:
                signal_card(name)
    st.html(ui.h2(t("Exportación a Estados Unidos")))
    m = default_prep().monthly
    has = (m["mkt"] == MKT_USA).any()
    if has:
        last12 = m[m["ym"] > m["ym"].max() - pd.DateOffset(months=12)]
        sh = last12.loc[last12["mkt"] == MKT_USA, "qty"].sum() / max(last12["qty"].sum(), 1e-9) * 100
        st.html(f'<p class="hint">{esc(t("Las exportaciones son {p}% de tu volumen de los últimos 12 meses. Estas señales solo mueven esa parte.", p=f"{sh:.1f}"))}</p>')
    else:
        st.html(f'<p class="hint">{esc(t("Tu archivo aún no distingue ventas de exportación. Márcalas en la pestaña Datos para que estas señales tengan efecto."))}</p>')
    st.html('<div class="sigtop"></div>')
    cols = st.columns(3)
    with cols[0]:
        tariff_card()
    for c, name in zip(cols[1:], us[:2]):
        with c:
            signal_card(name)
    for row in range(2, len(us), 3):
        for c, name in zip(st.columns(3), us[row:row + 3]):
            with c:
                signal_card(name)


def signal_card(name):
    base, info = SIGNALS[name], SIGNAL_INFO[name]
    on = sv(f"sigon_{name}", True)
    idx = list(SIGNALS).index(name)
    with st.container(key=f"sigcard_{idx}_{'on' if on else 'off'}"):
        c1, c2 = st.columns([5, 1])
        c1.html(f'<h3 style="margin:0;font-size:1.05em;line-height:1.3">{esc(t(name))}</h3>')
        with c2:
            on = wid(st.toggle, t("Usar señal: {n}", n=t(name)), f"sigon_{name}", True, label_visibility="collapsed")
        lead = int(sv(f"lead_{name}", base["lead"]))
        st.html(f'<p class="hint" style="margin:0">{esc(t(info[0]))}</p><div class="sigmeta"><span>{esc(t("Anticipación"))} <b>{esc(t("{n} meses", n=lead) if lead != 1 else t("1 mes"))}</b></span>'
                f'<span>{esc(t("Lectura actual"))} <b>{esc(t(info[1]))}</b></span><span>{esc(t("Fuente"))} <b>{esc(t(info[2]))}</b></span></div>')
        wid(st.slider, t("Cambio esperado"), f"chg_{name}", 0, min_value=-30, max_value=30, step=1, format="%+d%%", disabled=not on)
        if name == "Precio del cobre" and abs(cu_link_change()) > 0.05:
            st.caption(t("Suma {x}% que viene del camino de precios de Metales.", x=f"{cu_link_change():+.1f}"))
        if name == "Precio del cobre" and abs(live_signal_change()) > 0.05:
            st.caption(t("Suma {x}% porque el precio en vivo está así contra su promedio de 90 días.", x=f"{live_signal_change():+.1f}"))
        wid(st.slider, t("Sensibilidad"), f"sens_{name}", float(base["sens"]), min_value=-1.0, max_value=1.0, step=0.05, disabled=not on)
        with st.expander(t("Ajustes avanzados")):
            wid(st.number_input, t("Meses para efecto completo"), f"lead_{name}", int(base["lead"]), min_value=1, max_value=12, step=1)
        if name in TOPIC_OF_SIGNAL:
            news_note(TOPIC_OF_SIGNAL[name])


def tariff_card():
    eff = (get_mult() - 1) * 100
    with st.container(key="sigcard_tariff_on"):
        st.html(f'<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px"><h3 style="margin:0;font-size:1.05em;line-height:1.3">'
                f'{esc(t("Arancel de EE. UU. (Sección 232)"))}</h3><span class="badge {"warn" if eff < 0 else "ok"}">{eff:+.1f}%</span></div>'
                f'<p class="hint" style="margin:0">{esc(t("Desde abril de 2026 se cobra sobre el valor total del cable importado. Mueve la demanda de exportación según la elasticidad precio."))}</p>')
        wid(st.slider, t("Arancel actual, ya reflejado en tus ventas"), "t_cur", TARIFF_RATE_PCT, min_value=0, max_value=60, step=5, format="%d%%")
        wid(st.slider, t("Arancel en el escenario"), "t_sce", TARIFF_RATE_PCT, min_value=0, max_value=60, step=5, format="%d%%")

        def _set(v):
            store()["t_sce"] = v
            st.session_state["t_sce"] = v
        b = st.columns(3)
        b[0].button(t("T-MEC 0%"), key="tq0", on_click=_set, args=(0,))
        b[1].button("25%", key="tq25", on_click=_set, args=(25,))
        b[2].button("50%", key="tq50", on_click=_set, args=(50,))
        wid(st.slider, t("Elasticidad precio de la demanda"), "t_ela", 0.6, min_value=0.0, max_value=3.0, step=0.1)
        st.html(f'<p class="syncnote">{esc(t("Efecto en exportaciones = ((1 + arancel escenario) / (1 + arancel actual)) elevado a la −elasticidad."))}</p>')
        news_note("Arancel / T-MEC")


# ================================================================== qué pasa si… (comparación contra la base)
def neutral(s: dict) -> dict:
    """Estado 'base': sin cambios en señales, arancel vigente, sin escenario de metal, camino de precios plano y precios de venta de referencia."""
    n = copy.deepcopy(dict(s))
    for name in SIGNALS:
        n[f"chg_{name}"] = 0
    # Sliders store integer percentages; t_cur defaults to TARIFF_RATE_PCT (50), not the decimal 0.50
    n["t_sce"] = int(n.get("t_cur", TARIFF_RATE_PCT))
    n["pl_mchg"], n["mt_kind"], n["wi_opp_t"], n["mt_manual"] = 0, "flat", 0.0, None
    n["mt_pass"] = PASS_THROUGH_DEFAULT
    n["prices"] = copy.deepcopy(DEFAULT_PRICES)
    n.pop("mt_path", None)
    return n


def changed_levers(s=None) -> list[str]:
    """Qué se movió contra la base, en texto corto."""
    s = _s(s)
    out = []
    tf = get_tariff(s)
    if abs(tf["scenario"] - tf["current"]) > 1e-9:
        out.append(t("arancel {a}% (vigente {b}%)", a=f"{tf['scenario']:.0f}", b=f"{tf['current']:.0f}"))
    for name in SIGNALS:
        c = float(s.get(f"chg_{name}", 0))
        if c and s.get(f"sigon_{name}", True):
            out.append(f"{t(name)} {c:+.0f}%")
    if s.get("pl_mchg", 0):
        out.append(t("metal {x}%", x=f"{s['pl_mchg']:+.0f}"))
    if s.get("mt_kind", "flat") != "flat":
        out.append(t("camino de precios del metal"))
    if s.get("prices") and s["prices"] != DEFAULT_PRICES:
        out.append(t("precios de venta editados"))
    if float(s.get("wi_opp_t", 0) or 0) > 0:
        out.append(t("pedido adicional"))
    return out


def delta_chip(label: str, now: float, base: float, fmt_fn, pct_mode=True, good_up=True):
    d = (now / base - 1) * 100 if (pct_mode and base) else now - base
    cls = "neg" if (d < 0) == good_up and d != 0 else ""
    txt = f"{d:+.1f}%" if pct_mode else (("+" if d >= 0 else "-") + ui.mxn(abs(d)))
    return (Raw(f'<span class="{cls}">{esc(txt)}</span>'), label, t("antes: {x}", x=fmt_fn(base)))


def whatif_strip(items: list, levers: list[str]):
    """Indicador de qué cambió: cada pieza dice cuánto se movió contra la base y la lista de palancas activas."""
    if not levers:
        st.html(f'<p class="hint" style="margin:0 0 10px">{esc(t("Sin cambios contra la base: señales en 0%, arancel vigente y metal sin escenario."))}</p>')
        return
    st.html(f'<p class="hint" style="margin:0 0 6px">{esc(t("Cambios contra la base"))}: <b>{esc(", ".join(levers))}</b></p>' + ui.kpis(items, cols=len(items)))


def reset_whatif():
    n = neutral(store())
    _reset_keys = ("chg_", "t_sce", "pl_mchg", "wi_")
    _reset_exact = {"prices", "mt_kind", "mt_manual", "mt_pass"}
    for k, v in n.items():
        if k.startswith(_reset_keys) or k in _reset_exact:
            store()[k] = v
            if k in st.session_state and k not in ("prices",):
                st.session_state[k] = v
    store()["wi_opp_t"] = 0.0
    st.session_state["wi_opp_t"] = 0.0
    store().pop("wi_banner", None)
    st.session_state["wi_open"] = False
    store()["ov_ver"] = sv("ov_ver", 0) + 1
    # Re-apply live FX/metals so "Volver a la base" picks up current market prices
    try:
        apply_live()
    except Exception:
        pass


def whatif_panel(prep):
    """Panel '¿Qué pasa si…?': palancas comunes en un solo lugar. Usan los mismos controles que Señales, Planeación y Metales."""
    levers = changed_levers()
    b = store().get("wi_banner")
    if b:
        st.html(f'<div class="alertbar"><b>{esc(t("Escenario sugerido por una noticia"))}:</b> {esc(b["title"])}<br>{esc(t("Cambios pre-llenados"))}: {esc(", ".join(b["changes"]))}. '
                f'{esc(t("Revisa los valores abajo: nada más cambia hasta que lo ajustes tú, y puedes volver a la base."))}'
                + (f'<br>{esc(t("Tus datos no distinguen exportaciones a EE. UU.: el arancel no moverá el pronóstico hasta que las marques en Datos."))}' if not (default_prep().monthly["mkt"] == MKT_USA).any() and any("arancel" in c or "tariff" in c for c in b["changes"]) else "")
                + '</div>')
    with st.expander(t("¿Qué pasa si…?"), expanded=bool(levers) or bool(st.session_state.get("wi_open"))):
        # ── live market context for captions ───────────────────────────────────
        snap = live_snapshot()
        q = snap["q"]
        def _q(s): return q.get(s)
        def _val(s, fallback): return float(_q(s).value) if _q(s) and _q(s).status != "none" and np.isfinite(_q(s).value) else fallback
        def _src(s): return (f"{_q(s).source} · {(_q(s).ts or '')[:10]}" if _q(s) and _q(s).status != "none" else t("último guardado")) if _q(s) else "—"
        def _stale(s): return _q(s) and _q(s).status == "stale"

        live_fx  = _val("fx", PLAN_DEFAULTS["fx"])
        live_cu  = _val("cu", PLAN_DEFAULTS["cu_lme"])
        live_al  = _val("al", PLAN_DEFAULTS["al_lme"])

        # Ensure pl_fx is initialised to the live rate (not 1.00 from a corrupted session)
        if store().get("pl_fx", 0) < 5.0:
            store()["pl_fx"] = live_fx
            st.session_state["pl_fx"] = live_fx

        import datetime as _dt
        try:
            days_since_verified = (_dt.date.today() - _dt.date.fromisoformat(TARIFF_VERIFIED)).days
        except Exception:
            days_since_verified = 0

        c = st.columns(4, gap="medium")
        with c[0]:
            wid(st.slider, t("Arancel en el escenario"), "t_sce", TARIFF_RATE_PCT, min_value=0, max_value=60, step=5, format="%d%%")
            tariff_note = f"Vigente: {TARIFF_RATE_PCT}% · Sección {TARIFF_SECTION} · verificado {TARIFF_VERIFIED}"
            if days_since_verified > 30 and adv():
                tariff_note += f" · ⚠️ {t('Verifica que el arancel siga vigente.')}"
            st.caption(tariff_note)
        with c[1]:
            pct = wid(st.slider, t("Precio del metal"), "pl_mchg", 0, min_value=-30, max_value=30, step=1, format="%+d%%")
            cu_shown = live_cu * (1 + pct / 100)
            al_shown = live_al * (1 + pct / 100)
            metal_lbl = f"Cobre: ${live_cu:,.0f} · Aluminio: ${live_al:,.0f} USD/t · {_src('cu')}"
            if _stale("cu"):
                metal_lbl += f" · ⚠️ {t('precio antiguo')}"
            st.caption(metal_lbl)
            if pct != 0:
                st.caption(f"{pct:+d}% → Cobre ${cu_shown:,.0f} · Aluminio ${al_shown:,.0f} USD/t")
        with c[2]:
            fx_src = _src("fx")
            if _stale("fx"):
                fx_src += f" · ⚠️ {t('dato antiguo')}"
            wid(st.number_input, t("Tipo de cambio (MXN por USD)"), "pl_fx", live_fx, min_value=5.0, max_value=30.0, step=0.05)
            st.caption(f"{fx_src}")
        with c[3]:
            wid(st.slider, t("Traspaso del metal al precio de venta"), "mt_pass", PASS_THROUGH_DEFAULT,
                min_value=0, max_value=100, step=5, format="%d%%")
            st.caption(t("Supuesto de la empresa: {n}%", n=int(PASS_THROUGH_DEFAULT)))

        gs = list(prep.gauges)
        # Default calibre = currently selected gauge (first non-Todos selection)
        cur_sel = store().get("chips_cal") or ["Todos"]
        default_cal = gs[0] if "Todos" in cur_sel else (cur_sel[0] if cur_sel[0] in gs else gs[0])
        d = st.columns([1.3, 1, 1, 1, 1], gap="medium") if adv() else [None, None, *st.columns(3, gap="medium")]
        if adv():
            with d[0]:
                name = wid(st.selectbox, t("Señal"), "wi_sig", list(SIGNALS)[0], options=list(SIGNALS), format_func=t)
            with d[1]:
                wid(st.slider, t("Cambio esperado"), f"chg_{name}", 0, min_value=-30, max_value=30, step=1, format="%+d%%")
        with d[2]:
            wid(st.selectbox, t("Pedido adicional en el calibre"), "wi_opp_cal", default_cal, options=gs, format_func=ui.cal_label)
        with d[3]:
            wid(st.number_input, t("Toneladas por mes"), "wi_opp_t", 0.0, min_value=0.0, step=5.0)
        with d[4]:
            wid(st.number_input, t("Durante (meses)"), "wi_opp_m", 3, min_value=1, max_value=12, step=1)
        e = st.columns([3, 1], vertical_alignment="center")
        if adv():
            e[0].toggle(t("Aplicar señales y pedidos al pronóstico principal"), value=sv("sig_apply", True), key="sig_apply",
                        on_change=lambda: store().__setitem__("sig_apply", st.session_state["sig_apply"]))
        e[1].button(t("Volver a la base"), key="wi_reset", on_click=reset_whatif, width="stretch")
        st.caption(t("Cada cambio recalcula el pronóstico, el plan, los márgenes y los precios de Metales al instante."))


# ================================================================== sección Metales
def price_template() -> bytes:
    months = pd.date_range("2024-01-01", periods=30, freq="MS")
    rng = np.random.default_rng(5)
    df = pd.DataFrame({t("Fecha"): months.strftime("%Y-%m"),
                       t("Cobre (USD/t)"): (9000 + np.cumsum(rng.normal(25, 150, len(months)))).round(0),
                       t("Aluminio (USD/t)"): (2400 + np.cumsum(rng.normal(6, 40, len(months)))).round(0)})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name=t("Precios"), index=False)
    return buf.getvalue()


def load_price_file(up, generic):
    try:
        buf = io.BytesIO(up.getvalue())
        buf.name = up.name
        st.session_state["mt_hist"] = MT.parse_history(D.load_file(buf), generic)
        st.session_state["mt_hist_name"] = up.name
        return None
    except Exception as e:
        return str(e)


def metal_impact(metal: str, pct: float, m, gauges, horizon, ml) -> dict:
    """Plan con el precio de un metal movido pct % en todos los meses contra el plan actual (todo lo demás igual)."""
    s1 = store()
    s2 = copy.deepcopy(dict(s1))
    s2["mt_path"] = MT.shock(metal_paths(s1), metal, pct)
    out = []
    for s in (s1, s2):
        res = cached_plan(m, gauges, horizon, get_params(s), get_prices(s), get_mult(s), get_sig(s), get_w(s),
                          {"metal": sv("ov_metal", {}), "inv": sv("ov_inv", {}), "cap": sv("ov_cap", {})}, ml, get_crm(s, E.TON, get_prices(s)))
        out.append(PL.totals(res, None))
    return {"base": out[0], "shock": out[1]}


def section_metales():
    prep = default_prep()
    m = prep.monthly
    ml = get_ml(m, E.windows_for(m))
    gauges = tuple(prep.gauges)
    hist = price_history()
    left, right = layout()
    with left:
        horizon = wid(st.radio, t("Horizonte"), "seg_h", 6, options=[3, 6, 12], horizontal=True, format_func=lambda x: t("{n} meses", n=x))
        st.html(ui.group_title(t("Precios actuales")))
        cu0 = price_control("cu", t("Cobre (USD/t)"), "pl_cu", 9840.0, 10.0, 0.0)
        al0 = price_control("al", t("Aluminio (USD/t)"), "pl_al", 2600.0, 10.0, 0.0)
        fx0 = price_control("fx", t("Tipo de cambio (MXN por USD)"), "pl_fx", 18.5, 0.1, 1.0)
        st.caption(t("COMEX es una referencia cercana al LME, no el precio oficial del LME (requiere licencia de pago)."))
        if adv():
            with st.expander(t("Referencia de los precios de lista")):
                st.caption(t("Los precios de lista de Planeación se fijaron con estos precios de metal (los valores ilustrativos de la app). Si el precio actual es distinto, el traspaso mueve el precio de venta. Cámbialos por los de tu lista real."))
                wid(st.number_input, t("Cobre de referencia (USD/t)"), "mt_ref_cu", ref_value(None, "cu"), min_value=0.0, step=10.0)
                wid(st.number_input, t("Aluminio de referencia (USD/t)"), "mt_ref_al", ref_value(None, "al"), min_value=0.0, step=10.0)
                wid(st.number_input, t("Tipo de cambio de referencia"), "mt_ref_fx", ref_value(None, "fx"), min_value=1.0, step=0.1)

        wid(st.slider, t("Traspaso del metal al precio de venta"), "mt_pass", PASS_THROUGH_DEFAULT, min_value=0, max_value=100, step=5, format="%d%%")
        st.caption(t("Qué parte de un cambio en el costo del metal se traslada al precio de lista. Con 0% el margen absorbe todo. Supuesto de la empresa: {n}%.", n=int(PASS_THROUGH_DEFAULT)))
        st.html(ui.group_title(t("Camino de precios, 12 meses")))
        kinds = ["flat", "pct", "manual"] + (["trend"] if hist is not None else [])
        klabel = {"flat": "Precio actual constante", "pct": "Escenario en %", "manual": "Tabla mes por mes", "trend": "Tendencia del histórico"}
        if st.session_state.get("mt_kind") not in kinds + [None]:
            st.session_state["mt_kind"] = "flat"
        kind = wid(st.selectbox, t("Cómo definirlo"), "mt_kind", "flat", options=kinds, format_func=lambda k: t(klabel[k]))
        if kind == "pct":
            wid(st.slider, t("Cobre: cambio en 12 meses"), "mt_cu_chg", 0, min_value=-40, max_value=40, step=1, format="%+d%%")
            wid(st.slider, t("Aluminio: cambio en 12 meses"), "mt_al_chg", 0, min_value=-40, max_value=40, step=1, format="%+d%%")
            wid(st.slider, t("Tipo de cambio: cambio en 12 meses"), "mt_fx_chg", 0, min_value=-20, max_value=20, step=1, format="%+d%%")
            wid(st.radio, t("Forma del cambio"), "seg_mt_shape", "lineal", options=["lineal", "inmediato"], horizontal=True, format_func=t)
        if adv():
            wid(st.toggle, t("Usar el camino de precios en la señal del cobre"), "mt_link", True)
            wid(st.toggle, t("Usar el precio en vivo en la señal del cobre"), "mt_live_sig", True)
            wid(st.toggle, t("Usar el histórico automático de precios (COMEX/ALI)"), "mt_auto_hist", True)
            st.caption(t("Si el cobre sube, la señal del cobre reduce un poco la demanda (sensibilidad en Señales)."))
            st.html(ui.group_title(t("Histórico de precios")))
            generic = st.radio(t("Si el archivo trae una sola columna de precio, es de"), ["cu", "al"], horizontal=True, key="seg_mt_generic",
                               format_func=lambda k: t("Cobre") if k == "cu" else t("Aluminio"))
            up = st.file_uploader(t("Sube fecha y precio (.xlsx o .csv)"), type=["xlsx", "xls", "csv"], key="mt_file")
            if up is not None and st.session_state.get("mt_hist_name") != up.name:
                err = load_price_file(up, generic)
                if err:
                    st.error(err)
                else:
                    st.rerun()
            if hist is not None:
                st.caption(t("Histórico cargado: {n} meses ({a} a {b}).", n=len(hist), a=ui.mes(hist["ym"].min(), True), b=ui.mes(hist["ym"].max(), True)))
                if st.button(t("Quitar histórico"), key="mt_hist_del"):
                    for k in ("mt_hist", "mt_hist_name"):
                        st.session_state.pop(k, None)
                    st.rerun()
            st.download_button(t("Descargar plantilla de precios"), price_template(), file_name=t("plantilla-precios-metal.xlsx"),
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_tpl_px", width="stretch")
        st.html(ui.group_title(t("Compra de metal")))
        ahead = wid(st.slider, t("Anticipación máxima (meses)"), "mt_ahead", 6, min_value=1, max_value=12, step=1) if adv() else sv("mt_ahead", 6)
        hold = wid(st.slider, t("Costo anual de mantener inventario"), "pl_hold", 25, min_value=5, max_value=50, step=1, format="%d%%")

    months12 = MT.next_months(m["ym"].max())
    # tabla manual: se inicia con el camino actual la primera vez
    if kind == "manual" and not store().get("mt_manual"):
        p0 = MT.make_path("flat", cu0, al0, fx0)
        store()["mt_manual"] = [{"cu": p0["cu"][i], "al": p0["al"][i], "fx": p0["fx"][i]} for i in range(12)]
    path = metal_paths()
    prices, mult, sig, w = get_prices(), get_mult(), get_sig(), get_w()
    params = get_params()
    ov = {"metal": sv("ov_metal", {}), "inv": sv("ov_inv", {}), "cap": sv("ov_cap", {})}
    crm = get_crm(None, E.TON, prices)
    res = cached_plan(m, gauges, horizon, params, prices, mult, sig, w, ov, ml, crm)
    tot = PL.totals(res)
    base_s = neutral(store())
    res_b = cached_plan(m, gauges, horizon, get_params(base_s), get_prices(base_s), get_mult(base_s), get_sig(base_s), get_w(base_s), ov, ml, get_crm(base_s, E.TON, prices))
    tot_b = PL.totals(res_b)
    k = 1 + sv("pl_mchg", 0) / 100

    with right:
        end_cu = path["cu"][-1] * k
        chg_cu = (end_cu / cu0 - 1) * 100 if cu0 else 0
        st.html(ui.headline(t("El cobre está en {a} USD/t y el camino lo lleva a {b} USD/t en 12 meses ({c}).", a=f"{cu0:,.0f}", b=f"{end_cu:,.0f}", c=f"{chg_cu:+.1f}%"),
                            t("Cambia el precio actual, el camino mes por mes o un escenario en %. El costo del metal, el margen, el inventario y las compras se recalculan al instante.")))
        news_note("Precios de metales")
        whatif_strip([delta_chip(t("Ingreso {h} meses", h=horizon), tot["rev"], tot_b["rev"], ui.mxn),
                      delta_chip(t("Margen bruto"), tot["margin"], tot_b["margin"], ui.mxn),
                      delta_chip(t("Costo del metal"), tot["metal_cost"], tot_b["metal_cost"], ui.mxn, good_up=False),
                      delta_chip(t("Valor del inventario de seguridad"), tot["ss_value"], tot_b["ss_value"], ui.mxn, good_up=False)], changed_levers())
        st.html(ui.kpis([(ui.mxn(tot["metal_cost"]), t("Costo del metal {h} meses", h=horizon)),
                         (Raw(f'<span class="{"neg" if tot["margin"] < 0 else ""}">{esc(ui.mxn(tot["margin"]))}</span>'), t("Margen bruto"), f"{tot['margin_pct']:.1f}%"),
                         (ui.mxn(tot["rev"]), t("Ingreso en pesos {h} meses", h=horizon)),
                         (ui.mxn(tot["ss_value"]), t("Valor del inventario de seguridad"))]))
        c1, c2 = st.columns(2, gap="large")
        for col, key, nm, now in ((c1, "cu", t("Cobre"), cu0), (c2, "al", t("Aluminio"), al0)):
            hs = hist.set_index("ym")[key].dropna().tail(36) if hist is not None and hist[key].notna().any() else None
            with col:
                st.html(ui.h2(nm, t("USD por tonelada")))
                st.plotly_chart(ui.metal_fig(hs, now, months12, [x * k for x in path[key]], nm), width="stretch", theme=None, config={"displayModeBar": False})
        # tabla del camino (editable en modo manual)
        tbl = pd.DataFrame({t("Mes"): [mes_largo(x) for x in months12], t("Cobre (USD/t)"): [round(x * (k if kind != "manual" else 1), 1) for x in path["cu"]],
                            t("Aluminio (USD/t)"): [round(x * (k if kind != "manual" else 1), 1) for x in path["al"]], t("Tipo de cambio"): [round(x, 3) for x in path["fx"]]})
        if kind == "manual":
            st.html(ui.h2(t("Tabla de precios mes por mes"), t("Escribe los precios esperados. El primer mes es el siguiente a tu última venta.")))
            ver = sv("mt_ver", 0)
            ed = st.data_editor(tbl, hide_index=True, width="stretch", key=f"ed_path_{ver}_{ST.cfg()['lang']}", disabled=[tbl.columns[0]],
                                column_config={tbl.columns[1]: st.column_config.NumberColumn(min_value=0.0, step=10.0), tbl.columns[2]: st.column_config.NumberColumn(min_value=0.0, step=10.0),
                                               tbl.columns[3]: st.column_config.NumberColumn(min_value=1.0, step=0.05)})
            if not ed.equals(tbl):
                store()["mt_manual"] = [{"cu": float(r.iloc[1]), "al": float(r.iloc[2]), "fx": float(r.iloc[3])} for _, r in ed.iterrows()]
                store()["mt_ver"] = ver + 1
                st.rerun()
        else:
            with st.expander(t("Ver el camino mes por mes")):
                st.html(ui.table(list(tbl.columns), [[r.iloc[0], f"{r.iloc[1]:,.0f}", f"{r.iloc[2]:,.0f}", f"{r.iloc[3]:.2f}"] for _, r in tbl.iterrows()], num=(1, 2, 3)))

        # impacto de un movimiento del precio
        st.html(ui.h2(t("Si el precio del metal se mueve"), t("Compara el plan actual contra el mismo plan con el precio de un metal movido en todos los meses.")))
        ci = st.columns([1.3, 1.2, 2.5], gap="large", vertical_alignment="bottom")
        mt_sel = ci[0].radio(t("Metal"), ["cu", "al"], horizontal=True, key="seg_mt_shock_metal", format_func=lambda x: t("Cobre") if x == "cu" else t("Aluminio"))
        pct = ci[1].number_input(t("Cambio %"), min_value=-50.0, max_value=100.0, value=10.0, step=5.0, key="mt_shock_pct")
        imp = metal_impact(mt_sel, pct, m, gauges, horizon, ml)
        dm = (imp["shock"]["margin"] - imp["base"]["margin"])
        dr = (imp["shock"]["rev"] / imp["base"]["rev"] - 1) * 100 if imp["base"]["rev"] else 0.0
        dv = (imp["shock"]["vol"] / imp["base"]["vol"] - 1) * 100 if imp["base"]["vol"] else 0.0
        metal_name = t("el cobre") if mt_sel == "cu" else t("el aluminio")
        verb_p = t("sube") if pct >= 0 else t("baja")
        verb_m = t("baja") if dm < 0 else t("sube")
        verb_r = t("sube") if dr >= 0 else t("baja")
        st.html(f'<p class="headline" style="font-size:clamp(18px,2vw,24px);max-width:60ch">{esc(t("Si {m} {v1} {p}%, el margen {v2} {x} y la venta en pesos {v3} {r}%.", m=metal_name, v1=verb_p, p=f"{abs(pct):.0f}", v2=verb_m, x=ui.mxn(abs(dm)), v3=verb_r, r=f"{abs(dr):.1f}"))}</p>'
                f'<p class="subline">{esc(t("Con traspaso de {pt}% del alza al precio de lista, el volumen cambia {dv}% por la señal del cobre.", pt=f"{sv("mt_pass", 50):.0f}", dv=f"{dv:+.2f}"))}</p>')

        # recomendación de compra
        st.html(ui.h2(t("¿Comprar ahora o después?"), t("Para cada mes de producción se elige el momento de compra de menor costo, considerando el camino de precios, el plan de producción y el costo de mantener inventario.")))
        rows_cu = res.rows
        for key, nm, now in (("cu", t("Cobre"), cu0), ("al", t("Aluminio"), al0)):
            req = np.zeros(horizon)
            for r in rows_cu.itertuples():
                if r.metal == key:
                    req = req + np.asarray(r.metal_req)
            if req.sum() <= 0:
                continue
            pp = [x * k for x in path[key]]
            plan_b = MT.purchase_plan(req, now, pp, fx0, path["fx"], hold, ahead)
            sched = plan_b["schedule"]
            now_t = plan_b["buy_now"]
            if now_t > 1e-6:
                cov = next((s_["covers"] for s_ in sched if s_["t"] == 0), [])
                head = t("{n}: compra hoy {x} t, que cubren {c}.", n=nm, x=f"{now_t:,.0f}", c=t("los meses {a} a {b}", a=min(cov), b=max(cov)) if len(cov) > 1 else t("el mes {a}", a=cov[0]))
            else:
                head = t("{n}: compra mes a mes. El camino no justifica adelantar compras.", n=nm)
            sav = plan_b["saving"]
            st.html(f'<h3 style="margin:18px 0 4px;font-size:1.05em">{esc(head)}</h3>'
                    f'<p class="hint">{esc(t("Ahorro estimado contra comprar mes a mes: {x}. Costo recomendado {a}; mes a mes {b}; todo hoy {c}.", x=ui.mxn(sav), a=ui.mxn(plan_b["cost_opt"]), b=ui.mxn(plan_b["cost_jit"]), c=ui.mxn(plan_b["cost_now"])))}</p>')
            lab = lambda tt: t("Hoy") if tt == 0 else mes_largo(months12[tt - 1])
            st.html(ui.table([t("Momento de compra"), t("Toneladas"), t("Precio (USD/t)"), t("Cubre los meses")],
                             [[lab(s_["t"]), f"{s_['tons']:,.1f}", f"{s_['price']:,.0f}", ", ".join(str(c) for c in s_["covers"])] for s_ in sched], num=(1, 2)))
        st.html(f'<p class="syncnote">{esc(t("Simplificaciones: costo de mantener = porcentaje anual del valor, sin descuentos por volumen, coberturas ni límites de bodega. Los precios son ilustrativos hasta que captures los tuyos."))}</p>')

        # ¿ayuda el precio del metal al machine learning?
        if adv():
            st.html(ui.h2(t("Precio del metal en el machine learning")))
            info = ml.get("exog") if ml else None
            if hist is None:
                st.html(f'<p class="hint">{esc(t("Sube un histórico de precios para probar si el precio del cobre y del aluminio mejora el machine learning. Solo se usa si baja el error de validación."))}</p>')
            elif info is None:
                st.html(f'<p class="hint">{esc(t("El histórico no cubre al menos 85% de los meses de ventas, así que no se pudo probar."))}</p>')
            else:
                verdict = t("Se conserva: mejora la validación.") if info["kept"] else t("No se usa: no mejora la validación.")
                st.html(ui.kpis([(f"{info['err_without']:.1f}%", t("Error de validación sin el precio")), (f"{info['err_with']:.1f}%", t("Error de validación con el precio")),
                                 (f"{info['windows_better']} / {info['windows']}", t("Ventanas donde mejora")), (verdict, t("Decisión"))]))
                st.html(f'<p class="syncnote">{esc(t("Se prueba con la mejor configuración y se conserva solo si mejora al menos 0.25 puntos y en la mitad de las ventanas. Los precios futuros de cada ventana se suponen planos."))}</p>')


# ================================================================== por qué el modelo pronostica esto (SHAP)
FEATURE_PLAIN = {
    "Rezago 1 mes": "la venta del mes pasado", "Rezago 2 meses": "la venta de hace 2 meses", "Rezago 3 meses": "la venta de hace 3 meses",
    "Mismo mes año anterior": "la venta del mismo mes del año pasado", "Promedio 3 meses": "el promedio de los últimos 3 meses",
    "Tendencia reciente": "la tendencia reciente", "Mes del año": "la temporada (el mes del año)", "Calibre": "el calibre",
    "Familia de producto": "la familia de producto", "Exportación": "si es exportación", "Tamaño de la serie (log)": "el tamaño del producto",
}


def explain_block(ml, units, market, horizon, idx, n_gauges):
    """Qué variables suben o bajan el pronóstico de un mes (valores SHAP del XGBoost), en español sencillo."""
    res = ((ml or {}).get("extra") or {}).get("xgb")
    st.html(ui.h2(t("Por qué el modelo pronostica esto"), t("Explica el pronóstico del modelo XGBoost para la selección actual. Cada barra dice cuántas toneladas "
                                                           "suma o resta una variable contra el valor base del modelo (valores SHAP).")))
    if res is None or res["final"].model is None:
        st.html(f'<p class="hint">{esc(t("XGBoost no está disponible en este equipo, así que no hay explicación."))}</p>')
        return
    gauges = [g for x in units for g in x]
    step = wid(st.selectbox, t("Mes a explicar"), "ex_step", 0, options=list(range(horizon)), format_func=lambda i: mes_largo(idx[i]))
    ex = MLm.explain(res["final"], gauges, market, step)
    if ex is None:
        st.html(f'<p class="hint">{esc(t("No hay datos suficientes para explicar esa selección."))}</p>')
        return
    c = ex["contrib"].sort_values()
    ups, downs = c[c > 0.05].sort_values(ascending=False).head(3), c[c < -0.05].head(3)
    fmt_list = lambda s: ", ".join(f"{t(FEATURE_PLAIN.get(k, k))} ({v:+,.0f} {ui.UNIT})" for k, v in s.items())
    u = ui.UNIT
    txt = t("Para {mes}, el modelo XGBoost espera {f} {u} {scope}. Parte de una base de {b} {u}.", mes=mes_largo(idx[step]), f=f"{ex['fc']:,.0f}", u=u,
            scope=scope_text(units, market, n_gauges), b=f"{ex['base']:,.0f}")
    if len(ups):
        txt += " " + t("Lo que más lo sube: {x}.", x=fmt_list(ups))
    if len(downs):
        txt += " " + t("Lo que más lo baja: {x}.", x=fmt_list(downs))
    st.html(f'<p class="subline" style="max-width:90ch">{esc(txt)}</p>')
    st.plotly_chart(ui.explain_fig([t(k) for k in c.index], c.values), width="stretch", theme=None, config={"displayModeBar": False})
    st.html(f'<p class="syncnote">{esc(t("Es la explicación de este modelo, no del método principal. Los valores suman el pronóstico: base + todas las barras = pronóstico del modelo."))}</p>')


# ================================================================== meses atípicos
ACT_LABEL = {"mantener": "Mantener", "corregir": "Corregir", "evento": "Marcar evento"}


@st.cache_data(show_spinner=False)
def cached_cvtest(monthly: pd.DataFrame, windows: tuple) -> pd.DataFrame:
    return AT.cv_test(monthly, list(windows))


def atypical_marks(det, gauges, hist) -> list:
    """Puntos para la gráfica: meses atípicos de los calibres seleccionados, con la decisión tomada."""
    dec = store().get("atyp_dec", {})
    out = []
    g = det[(det["nivel"] == "Calibre") & (det["id"].isin([str(x) for x in gauges]))]
    for r in g.itertuples():
        if r.ym not in hist.index:
            continue
        d = dec.get(r.key, {})
        note = f" · {d['n']}" if d.get("n") else ""
        out.append({"ym": r.ym, "y": float(hist.loc[r.ym]), "kind": d.get("a", "mantener"),
                    "text": f"{ui.cal_label(r.id)}: " + t("real {a} t, esperado {e} t", a=f"{r.actual:,.0f}", e=f"{r.expected:,.0f}") + note})
    return out


def atypical_block(m_raw, det, gauges, windows):
    """Lista de meses atípicos de la selección con la decisión de cada uno, y la prueba de si corregir mejora la precisión."""
    st.html(ui.h2(t("Meses atípicos"), t("Meses que se alejan mucho de lo esperado según la temporada de esa serie (puntaje z robusto mayor a 3.5). Nada se corrige solo: "
                                          "elige qué hacer con cada uno y los modelos se recalculan con los datos corregidos.")))
    gs = [str(x) for x in gauges]
    cal_of = m_raw.assign(_p=AT.product_id(m_raw)).groupby("_p")["cal"].first()
    g_rows = det[(det["nivel"] == "Calibre") & det["id"].isin(gs)]
    p_all = det[(det["nivel"] == "Producto") & det["id"].map(cal_of).isin(gs)].copy()
    p_all["dev"] = (p_all["actual"] - p_all["expected"]).abs()
    p_rows = p_all.sort_values("dev", ascending=False).head(30)
    dec = store().get("atyp_dec", {})
    ver = sv("atyp_ver", 0)
    opts = [t(v) for v in ACT_LABEL.values()]
    inv = {t(v): k for k, v in ACT_LABEL.items()}

    def editor(rows, title, key):
        if rows.empty:
            st.html(f'<p class="hint">{esc(t("No hay meses atípicos en {x}.", x=title))}</p>')
            return
        c_lvl, c_id, c_m, c_a, c_e, c_z, c_d, c_n = t("Nivel"), t("Calibre o producto"), t("Mes"), t("Real"), t("Esperado"), t("Puntaje z"), t("Decisión"), t("Nota")
        df = pd.DataFrame({c_lvl: [t(x) for x in rows["nivel"]], c_id: [ui.cal_label(i) if n_ == "Calibre" else i for i, n_ in zip(rows["id"], rows["nivel"])],
                           c_m: [mes_largo(x) for x in rows["ym"]], c_a: rows["actual"].round(1).to_numpy(), c_e: rows["expected"].round(1).to_numpy(), c_z: rows["z"].round(1).to_numpy(),
                           c_d: [t(ACT_LABEL[dec.get(k, {}).get("a", "mantener")]) for k in rows["key"]], c_n: [dec.get(k, {}).get("n", "") for k in rows["key"]]})
        ed = st.data_editor(df, hide_index=True, width="stretch", key=f"ed_atyp_{key}_{ver}_{ST.cfg()['lang']}", disabled=[c_lvl, c_id, c_m, c_a, c_e, c_z],
                            column_config={c_d: st.column_config.SelectboxColumn(options=opts, required=True), c_n: st.column_config.TextColumn(width="large")})
        if not ed.equals(df):
            new = {k: dict(v) for k, v in dec.items()}
            for k, (_, r) in zip(rows["key"], ed.iterrows()):
                a_, n_ = inv.get(r[c_d], "mantener"), str(r[c_n] or "")
                if a_ == "mantener" and not n_:
                    new.pop(k, None)
                else:
                    new[k] = {"a": a_, "n": n_}
            store()["atyp_dec"], store()["atyp_ver"] = new, ver + 1
            st.rerun()

    editor(g_rows, t("los calibres seleccionados"), "cal")
    with st.expander(t("Productos con meses atípicos ({n}, los 30 de mayor desviación)", n=len(p_all))):
        editor(p_rows, t("los productos de esta selección"), "prod")
    n_fix = len(corrected_keys())
    if n_fix:
        st.html(f'<p class="hint">{esc(t("{n} meses corregidos con el valor esperado. Los modelos, el plan y las gráficas ya los usan.", n=n_fix))}</p>'
                f'<p class="syncnote">{esc(t("Aviso: con meses corregidos, la precisión de validación se mide contra los datos ya corregidos, así que es algo optimista. La prueba de abajo, que no usa información futura y mide contra lo vendido, es la confiable."))}</p>')
    # ¿corregir mejora la precisión?
    with st.expander(t("¿Corregir los atípicos mejora la precisión?")):
        res = cached_cvtest(m_raw, tuple(windows))
        if res.empty:
            st.html(f'<p class="hint">{esc(t("No hay suficiente historia para probarlo."))}</p>')
            return
        rows = []
        for r in res.sort_values("Selección", key=lambda s: s.map(lambda x: (x == "Total", int(x) if str(x).isdigit() else 0))).itertuples(index=False):
            d = r[5]
            verdict = t("mejora") if d > 0.2 else t("empeora") if d < -0.2 else t("sin efecto")
            rows.append([Raw(f"<b>{esc(t('Total') if r[0] == 'Total' else ui.cal_label(r[0]))}</b>"), f"{r[1]:.1f}%", f"{r[2]:.1f}%",
                         Raw(f'<span class="{"neg" if d < -0.2 else ""}">{d:+.2f} {esc(t("pts"))}</span>'), str(r[3]), verdict])
        st.html(ui.table([t("Selección"), t("Error con datos crudos"), t("Error con datos corregidos"), t("Cambio"), t("Meses corregidos"), t("Veredicto")], rows, num=(1, 2, 3, 4)))
        tot = res[res["Selección"] == "Total"]
        if len(tot):
            d = float(tot["Mejora (pts)"].iloc[0])
            st.html(f'<p class="subline" style="max-width:90ch">{esc(t("En el total, corregir todos los atípicos detectados cambia el error en {d} puntos ({v}).", d=f"{d:+.2f}", v=t("mejora") if d > 0.2 else t("empeora") if d < -0.2 else t("sin efecto apreciable")))}</p>')
        st.html(f'<p class="syncnote">{esc(t("Prueba: en cada ventana de validación se detectan y corrigen los atípicos usando solo datos anteriores a esa ventana, se pronostica con cinco métodos estadísticos y el error se mide contra lo que de verdad se vendió. Es un promedio de los cinco métodos."))}</p>')


# ================================================================== aprender de series similares
@st.cache_data(show_spinner=False)
def cached_similar_eval(monthly: pd.DataFrame, windows: tuple) -> dict:
    return SM.evaluate_products(monthly, list(windows))


@st.cache_data(show_spinner=False)
def cached_product_groups(monthly: pd.DataFrame):
    Y, meta, months = SM.product_table(monthly)
    infos = [SM.span_info(r) for r in Y]
    meta = meta.assign(target=[SM.is_target(i) for i in infos], span=[i["span"] for i in infos], zero=[i["zero_share"] for i in infos], tons=Y.sum(axis=1))
    return meta


def similar_block(m):
    st.html(ui.h2(t("Aprender de series similares"), t("Los productos con historia corta o muchos ceros toman prestada la temporada y la tendencia de su grupo (mismo calibre, metal y familia). "
                                                     "Se prueba fuera de muestra si eso mejora el pronóstico.")))
    wins = E.windows_for(m)
    meta = cached_product_groups(m)
    n_t = int(meta["target"].sum())
    st.html(ui.kpis([(str(len(meta)), t("Productos")), (str(meta["g1"].nunique()), t("Grupos (calibre, metal y familia)")), (str(n_t), t("Con historia corta o muchos ceros")),
                     (f"{meta.loc[meta['target'], 'tons'].sum() / max(meta['tons'].sum(), 1e-9) * 100:.1f}%", t("De las toneladas totales"))]))
    ev = cached_similar_eval(m, tuple(wins))
    if not ev["n"]:
        st.html(f'<p class="hint">{esc(t("Ningún producto cumple las condiciones para evaluarlo."))}</p>')
        return
    cal = ev["calibre"]
    tot = cal[cal["cal"] == "Total"].iloc[0]
    d = float(tot["Mejora de la mezcla (pts)"])
    verdict = t("mejora") if d > 0.5 else t("empeora") if d < -0.5 else t("no cambia de forma apreciable")
    st.html(ui.headline(t("Con tus datos, usar el patrón del grupo {v} el pronóstico: {a}% de error sumando productos contra {b}% con el patrón propio.", v=verdict,
                          a=f"{tot['Error mezcla %']:.1f}", b=f"{tot['Error propio %']:.1f}"),
                        t("Sumando los pronósticos de todos los productos de cada calibre, la mezcla cambia el error en {d} puntos. Por producto el error es muy alto en ambos casos porque son series muy pequeñas.", d=f"{d:+.2f}")))
    rows = [[Raw(f"<b>{esc(t('Total') if r.cal == 'Total' else ui.cal_label(r.cal))}</b>"), f"{r._1:,.0f} {ui.UNIT}", f"{r._2:.1f}%", f"{r._3:.1f}%",
             Raw(f'<span class="{"neg" if r._4 < -0.05 else ""}">{r._4:+.2f} {esc(t("pts"))}</span>')] for r in cal.itertuples(index=False)]
    st.html(ui.table([t("Calibre"), t("Toneladas reales"), t("Error con patrón propio"), t("Error con patrón del grupo"), t("Cambio")], rows, num=(1, 2, 3, 4)))
    pr_ = ev["resumen"].iloc[0]
    st.html(f'<p class="syncnote">{esc(t("Solo en los {n} casos de productos con historia corta o muchos ceros, el error por producto pasa de {a}% a {b}% con la mezcla.", n=int(pr_["Series y ventanas"]), a=f"{pr_["Error propio %"]:.0f}", b=f"{pr_["Error mezcla %"]:.0f}"))}</p>')
    # ¿algún calibre entra al ranking con este método?
    applic = [g for g in sorted(m["cal"].unique()) if E.similar_parts(m, [g], None, E.TON, None, 6)[0] is not None and any(x is not None for x in E.similar_parts(m, [g], None, E.TON, None, 6)[0])]
    msg = t("El método 'Aprendizaje de series similares' compite en el ranking de: {x}.", x=", ".join(ui.cal_label(g) for g in applic)) if applic else \
        t("Ningún calibre tiene historia corta o muchos ceros, así que el método 'Aprendizaje de series similares' no compite en el ranking. Se activa solo cuando un calibre sea nuevo o intermitente.")
    st.html(f'<p class="hint">{esc(msg)}</p>')


# ================================================================== Laboratorio de modelos
def lab_time(v) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return "<0.01 s" if v < 0.01 else f"{v:.2f} s" if v < 10 else f"{v:.0f} s"


def section_laboratorio():
    prep = default_prep()
    m = prep.monthly
    ml = get_ml(m, E.windows_for(m))
    prices = get_prices()
    mk = [x for x in (MKT_NAC, MKT_USA) if x in set(m["mkt"])]
    left, right = layout()
    with left:
        market = None
        if len(mk) > 1:
            market = wid(st.radio, t("Mercado"), "seg_mkt", "Todos", options=["Todos", *mk], horizontal=True, format_func=t)
            market = None if market == "Todos" else market
        st.caption(t("Todos los modelos se entrenan con todos los productos a la vez, con las mismas variables y la misma validación cruzada de origen móvil."))
        if st.button(t("Reentrenar todos los modelos"), key="lab_retrain"):
            cached_ml.clear()
            st.session_state.get("ml_spec", {}).clear()
            st.rerun()
    units = [[g] for g in prep.gauges]
    allg = tuple(prep.gauges)
    mx = cached_matrix(m, tuple(tuple(x) for x in units), market, E.TON, prices, ml)          # error % por método (filas) y selección (columnas)
    bt = cached_bt(m, allg, market, E.TON, prices, ml)
    secs = bt.set_index("Método")["Tiempo s"]
    cols = list(mx.columns)
    win = {c: mx[c].idxmin() for c in cols if mx[c].notna().any()}
    with right:
        total_w = win.get("Total", "")
        st.html(ui.headline(t("Laboratorio de modelos: {w} es el más certero en el total con {p}% de precisión.", w=t(total_w), p=f"{max(0, 100 - mx.loc[total_w, 'Total']):.1f}"),
                            t("Cada modelo se evalúa contra los últimos meses reales con validación cruzada de origen móvil. La precisión es 100 menos el error MAPE; "
                              "no se usa R².")))
        by_type = {}
        for c in cols:
            by_type.setdefault(E.TYPE_OF.get(win.get(c, ""), ""), []).append(c)
        st.html(ui.kpis([(str(len(mx)), t("Modelos evaluados")), (t(E.TYPE_OF.get(total_w, "")), t("Tipo del ganador en el total")),
                         (str(len([c for c in cols if c != "Total" and E.TYPE_OF.get(win.get(c, "")) != "Estadístico"])), t("Calibres que gana un modelo que no es estadístico")),
                         (f"{ml['time_all']:.0f} s", t("Entrenamiento de los modelos de ML (una vez)"), t("después queda en caché"))]))
        order = mx.sort_values("Total", na_position="last").index
        rows = []
        for name in order:
            cells = [Raw(f"<b>{esc(t(name))}</b>"), t(E.TYPE_OF.get(name, ""))]
            won = []
            for c in cols:
                e_ = mx.loc[name, c]
                if not np.isfinite(e_):
                    cells.append("—")
                    continue
                p_ = f"{max(0, 100 - e_):.1f}%"
                if win.get(c) == name:
                    cells.append(Raw(f'<span class="win">{p_}</span>'))
                    won.append(c)
                else:
                    cells.append(p_)
            cells.append(lab_time(secs.get(name, np.nan) if name not in E.ML_FAMILY else ml_time_of(ml, name)))
            cells.append(", ".join(ui.cal_label(c) if c.isdigit() else t(c) for c in won) or "—")
            rows.append(cells)
        heads = [t("Modelo"), t("Tipo")] + [t("Total") if c == "Total" else ui.cal_label(c) for c in cols] + [t("Tiempo de entrenamiento"), t("Gana en")]
        main_rows = [i for i, nme in enumerate(order) if nme == total_w]
        st.html(ui.table(heads, rows, num=tuple(range(2, 2 + len(cols))), min_w=1100, main_rows=main_rows))
        st.html(f'<p class="syncnote">{esc(t("Precisión de validación (más alto es mejor). Subrayado = el mejor de esa columna. Los modelos con pocas ventanas no entran. Esta precisión es optimista porque el ganador se elige mirando los mismos meses; la prueba de abajo es la confiable."))}</p>')

        # ganador por calibre
        st.html(ui.h2(t("Ganador por calibre")))
        wrows = []
        for c in cols:
            col = mx[c].dropna().sort_values()
            if len(col) < 2:
                continue
            wrows.append([Raw(f"<b>{esc(t('Total') if c == 'Total' else ui.cal_label(c))}</b>"), t(col.index[0]), t(E.TYPE_OF.get(col.index[0], "")), f"{max(0, 100 - col.iloc[0]):.1f}%",
                          t(col.index[1]), f"{col.iloc[1] - col.iloc[0]:+.1f} {t('pts')}"])
        st.html(ui.table([t("Selección"), t("Ganador"), t("Tipo"), t("Precisión"), t("Segundo lugar"), t("Ventaja")], wrows, num=(3, 5)))

        # configuración elegida de cada familia
        with st.expander(t("Configuración elegida de cada modelo de machine learning")):
            fam_rows = [[t(E.ML), spec_label(ml["spec"]), f"{ml['honest_err']:.1f}%", lab_time(ml["time"])]]
            for fam, r in (ml.get("extra") or {}).items():
                fam_rows.append([t(MLm.MODEL_NAMES[fam]), spec_label(r["spec"]), f"{r['honest_err']:.1f}%", lab_time(r["time"])])
            st.html(ui.table([t("Modelo"), t("Configuración"), t("Error de validación"), t("Tiempo de entrenamiento")], fam_rows, num=(2, 3)))
            st.html(f'<p class="syncnote">{esc(t("Para cada ventana se usa la configuración que se habría elegido sin ver esa ventana. El tiempo incluye afinar, validar y entrenar el modelo final."))}</p>')
        out_of_sample_block()
        similar_block(m)


def ml_time_of(ml, name) -> float:
    return float(E.ml_models(ml)[name].get("time", np.nan))


def out_of_sample_block():
    """Resultados guardados de evaluacion_modelos.py: error real fuera de muestra de cada modelo."""
    p = Path(__file__).parent / "resultados_modelos.json"
    st.html(ui.h2(t("Prueba fuera de muestra"), t("Se repite todo el proceso como si estuviéramos 6 a 12 meses antes del final y se compara contra lo que de verdad se vendió. "
                                                  "Es la medida confiable de qué modelo gana.")))
    if not p.exists():
        st.html(f'<p class="hint">{esc(t("Todavía no hay resultados. Corre evaluacion_modelos.py para generarlos."))}</p>')
        return
    import json
    d = json.loads(p.read_text(encoding="utf-8"))
    meta, tab = d["meta"], pd.DataFrame(d["tabla"])
    cols = [c for c in tab.columns if c not in ("Método", "Tipo", "Orígenes", "Promedio calibres")]
    tab = tab.sort_values("Total")
    best = {c: tab.loc[tab[c].idxmin(), "Método"] for c in cols}
    rows = []
    for _, r in tab.iterrows():
        cells = [Raw(f"<b>{esc(t(r['Método']))}</b>"), t(r["Tipo"])]
        for c in cols:
            v = f"{r[c]:.1f}%"
            cells.append(Raw(f'<span class="win">{v}</span>') if best[c] == r["Método"] else v)
        cells.append(f"{r['Promedio calibres']:.1f}%")
        cells.append(f"{int(r['Orígenes'])} / {len(meta['origenes'])}")
        rows.append(cells)
    st.html(ui.table([t("Modelo"), t("Tipo")] + [t("Total") if c == "Total" else ui.cal_label(c) for c in cols] + [t("Promedio calibres"), t("Orígenes con resultado")], rows,
                     num=tuple(range(2, 4 + len(cols))), min_w=1100))
    st.html(f'<p class="syncnote">{esc(t("Error MAPE fuera de muestra, promedio de {n} orígenes (archivo {f}, {d}). Menor es mejor y el subrayado marca al mejor. Los orígenes se traslapan, así que no son independientes.", n=len(meta["origenes"]), f=meta["archivo"], d=meta["fecha"]))}</p>')



# ================================================================== noticias y regulación (Fase 6D)
_NEWS_BG = {"running": False}


def news_background(force: bool = False):
    """Actualiza las noticias en segundo plano (nunca bloquea la pantalla) si pasaron más de 30 minutos."""
    if _NEWS_BG["running"]:
        return
    last = NEWS.meta_get("last_fetch")
    if not force and last and (dt.datetime.now() - dt.datetime.fromisoformat(last)).total_seconds() < NEWS.TTL_MINUTES * 60:
        return
    _NEWS_BG["running"] = True
    def run():
        try:
            NEWS.refresh(force=force)
        finally:
            _NEWS_BG["running"] = False
    threading.Thread(target=run, daemon=True).start()


def ensure_news():
    """Si todavía no hay noticias guardadas, las trae una vez esperando (unos segundos); si no, solo revisa si toca actualizar en segundo plano."""
    if not NEWS.load(limit=1):
        with st.spinner(t("Buscando noticias y regulación (solo la primera vez)…")):
            waited = 0.0
            while _NEWS_BG["running"] and waited < 60:             # ya hay una actualización en segundo plano: se espera a que termine
                time.sleep(0.5)
                waited += 0.5
            if not NEWS.load(limit=1):
                NEWS.refresh(force=True)
    else:
        news_background()


@st.fragment(run_every="30m")
def news_timer():
    """Cada 30 minutos pide actualizar las noticias; si llegaron nuevas, recarga la pantalla para mostrarlas."""
    news_background()
    seen = st.session_state.get("_news_seen")
    now = NEWS.meta_get("last_fetch")
    if seen is not None and now and now != seen:
        st.session_state["_news_seen"] = now
        st.rerun()


TOPIC_OF_SIGNAL = {"Precio del cobre": "Precios de metales", "Licitaciones CFE": "Normativa México", "Naves industriales": "Industria y nearshoring",
                   "Permisos de construcción": "Construcción y vivienda", "Centros de datos": "Industria y nearshoring"}
REL_RANK = {"Alta": 0, "Media": 1, "Baja": 2}


def latest_news(topic=None, n: int = 1, days: int = 30, min_rel=("Alta", "Media")):
    rows = NEWS.load(days=days, topics=[topic] if topic else None, relevance=list(min_rel), statuses=["nuevo", "relevante", "revisado"])
    rows.sort(key=lambda r: (REL_RANK[r["relevance"]], -dt.datetime.fromisoformat(r["published"]).timestamp()))
    return rows[:n]


def rel_badge(r: str) -> str:
    return f'<span class="badge {"warn" if r == "Alta" else "ok" if r == "Media" else ""}">{esc(t(r))}</span>'


def summary_t(row) -> str:
    """Traduce el resumen propio (basado en reglas) cuando la interfaz está en inglés; los extractos de los feeds se dejan como vienen."""
    s = row["summary"]
    for base in NEWS.WHY.values():
        if s.startswith(base):
            rest = s[len(base):]
            tags = ", ".join(t(x) for x in row["tags"])
            return t(base) + (" " + t("Podría tocar: {x}.", x=tags) if rest.strip() and row["tags"] else "")
    return s


def news_card_html(r, compact: bool = False) -> str:
    d = dt.datetime.fromisoformat(r["published"])
    src = esc(r["source"])
    tags = "".join(f'<span class="tag">{esc(t(x))}</span>' for x in r["tags"])
    st_ = f'<span class="badge fill">{esc(t(r["status"]))}</span> ' if r["status"] in ("relevante", "revisado", "ignorado") else ""
    also = f' · {esc(t("{n} fuentes más", n=r["also"]))}' if r.get("also") else ""
    body = "" if compact else f'<p class="newssum">{esc(summary_t(r))}</p>'
    return (f'<div class="newscard"><div class="newshead">{rel_badge(r["relevance"])} {st_}<span class="badge">{esc(t(r["topic"]))}</span>'
            f'<span class="newsmeta">{d:%d %b %Y} · {src}{also}</span></div>'
            f'<a class="newstitle" href="{esc(r["url"])}" target="_blank" rel="noopener noreferrer">{esc(r["title"])}</a>{body}'
            + (f'<div class="newstags">{tags}</div>' if tags and not compact else "") + "</div>")


def news_note(topic: str):
    """Última noticia relevante del tema, para mostrarla junto a la tarjeta o la sección relacionada."""
    rows = latest_news(topic, 1)
    if rows:
        st.html(f'<div class="newsnote"><span class="tkl">{esc(t("Última noticia relacionada"))}</span>{news_card_html(rows[0], compact=True)}</div>')


def apply_simulation(item_id: str):
    """Pre-llena el panel ¿Qué pasa si…? con el escenario que sugiere una noticia. Se ejecuta solo cuando la persona pulsa el botón."""
    rows = [r for r in NEWS.load(limit=2000) if r["id"] == item_id]
    if not rows or not rows[0]["action"] or not rows[0]["action"].get("sim"):
        return
    r, sim, ch = rows[0], rows[0]["action"]["sim"], []

    def put(key, val):
        store()[key] = val
        if key in st.session_state:
            st.session_state[key] = val
    if "tariff" in sim:
        cur = float(store().get("t_cur", TARIFF_CURRENT))
        val = max(0.0, cur - 25) if sim["tariff"] == "down" else min(60.0, cur + 10)
        put("t_sce", int(val))
        ch.append(t("arancel {a}% (vigente {b}%)", a=f"{val:.0f}", b=f"{cur:.0f}") + (" · " + t("prueba de estrés: la noticia no dice si sube o baja") if sim["tariff"] == "stress" else ""))
    if "metal" in sim:
        put("pl_mchg", int(sim["metal"]))
        ch.append(t("metal {x}%", x=f"{sim['metal']:+d}"))
    if "signal" in sim:
        name, val = sim["signal"]
        put(f"chg_{name}", int(val))
        ch.append(f"{t(name)} {val:+d}%")
    store()["wi_banner"] = {"id": item_id, "title": r["title"], "text": r["action"]["text"], "changes": ch}
    st.session_state["nav"] = "Pronóstico"
    st.session_state["wi_open"] = True
    NEWS.log("simular escenario", item_id, "; ".join(ch) + " · " + r["title"][:80])


def mark_item(item_id: str, status: str):
    NEWS.set_status(item_id, status)


def item_buttons(r, prefix: str):
    """Botones de cada noticia: marcar y, si hay escenario sugerido, simularlo."""
    sim = bool(r["action"] and r["action"].get("sim"))
    c = st.columns([1, 1, 1, 1.6, 2.4])
    c[0].button(t("Relevante"), key=f"{prefix}_rel_{r['id']}", on_click=mark_item, args=(r["id"], "relevante"), disabled=r["status"] == "relevante")
    c[1].button(t("Ya revisado"), key=f"{prefix}_rev_{r['id']}", on_click=mark_item, args=(r["id"], "revisado"), disabled=r["status"] == "revisado")
    c[2].button(t("Ignorar"), key=f"{prefix}_ign_{r['id']}", on_click=mark_item, args=(r["id"], "ignorado"), disabled=r["status"] == "ignorado")
    if sim:
        c[3].button(t("Simular este escenario"), key=f"{prefix}_sim_{r['id']}", type="primary", on_click=apply_simulation, args=(r["id"],))


def attention_items():
    """Qué atender: movimientos de mercado y noticias de relevancia alta sin revisar, con su acción sugerida."""
    out = []
    try:
        snap = live_snapshot()
        out += [{"kind": "mercado", "text": txt, "action": t("Revisar el escenario de precio del metal y la recomendación de compra en Metales."), "section": "Metales"} for txt in market_alert_texts(snap)]
    except Exception:
        pass
    seen_topics = {}
    for r in NEWS.load(days=14, relevance=["Alta"], statuses=["nuevo", "relevante"]):
        if seen_topics.get(r["topic"], 0) >= 2:                      # varias fuentes repiten el mismo suceso: se muestran las 2 primeras de cada tema
            continue
        seen_topics[r["topic"]] = seen_topics.get(r["topic"], 0) + 1
        out.append({"kind": "noticia", "row": r})
    return out


def source_status_block():
    stt = json.loads(NEWS.meta_get("status") or "{}")
    last = NEWS.meta_get("last_fetch")
    st.html(f'<p class="hint">{esc(t("Última actualización: {x}.", x=last.replace("T", " ")[:16] if last else "—"))}</p>')
    for src, s in stt.items():
        if src.startswith("_"):
            continue
        if s["fail"] and not s["ok"]:
            if src == "GDELT":
                st.caption(t("GDELT (fuente opcional) no respondió o limitó el uso; no afecta al resto."))
            else:
                st.warning(t("{s} no respondió ({e}). Se muestran las últimas noticias guardadas.", s=src, e=re.sub(r"\(.*", "", s["error"])[:90].strip() or t("sin conexión")))
        elif s["fail"]:
            st.info(t("{s}: {a} de {b} búsquedas respondieron.", s=src, a=s["ok"], b=s["ok"] + s["fail"]))
    if not stt:
        st.info(t("Sin estado de las fuentes todavía."))


def section_noticias():
    ensure_news()
    left, right = layout()
    topics, rels = list(NEWS.TOPICS), ["Alta", "Media", "Baja"]
    with left:
        st.button(t("Actualizar"), key="news_refresh", type="primary", width="stretch", on_click=lambda: news_background(force=True))
        wid(st.pills, t("Tema"), "chips_news_topic", topics, options=topics, selection_mode="multi", format_func=t)
        wid(st.pills, t("Relevancia"), "chips_news_rel", ["Alta", "Media"], options=rels, selection_mode="multi", format_func=t)
        days = wid(st.radio, t("Fecha"), "seg_news_days", 30, options=[7, 30, 60], horizontal=True, format_func=lambda d: t("{n} días", n=d))
        sts = wid(st.pills, t("Estado"), "chips_news_status", ["nuevo", "relevante", "revisado"], options=list(NEWS.STATUSES), selection_mode="multi", format_func=t)
        q = wid(st.text_input, t("Buscar"), "news_q", "")
        st.caption(t("Se actualiza sola cada 30 minutos mientras la app está abierta. Solo se guardan titular, fuente, fecha, enlace y un resumen corto."))
    sel_t, sel_r, sel_s = store().get("chips_news_topic") or topics, store().get("chips_news_rel") or rels, store().get("chips_news_status") or list(NEWS.STATUSES)
    rows = NEWS.load(days=days, topics=sel_t, relevance=sel_r, statuses=sel_s, text=q.strip())
    rows.sort(key=lambda r: (REL_RANK[r["relevance"]], -dt.datetime.fromisoformat(r["published"]).timestamp()))
    allr = NEWS.load(days=days, statuses=["nuevo", "relevante", "revisado"])
    with right:
        n_alta = sum(1 for r in allr if r["relevance"] == "Alta")
        st.html(ui.headline(t("Noticias y regulación: {n} en los últimos {d} días, {a} de relevancia alta.", n=len(allr), d=days, a=n_alta),
                            t("Fuentes: Federal Register (oficial de EE. UU.), Google News en español e inglés y GDELT cuando responde. El DOF no tiene un feed accesible, así que sus publicaciones se buscan a través de las noticias.")))
        st.html(ui.kpis([(str(sum(1 for r in allr if r["relevance"] == "Alta")), t("Relevancia alta")), (str(sum(1 for r in allr if r["relevance"] == "Media")), t("Relevancia media")),
                         (str(len([r for r in allr if r["status"] == "nuevo"])), t("Sin revisar")), (str(len([r for r in allr if r["provider"] == "federal_register"])), t("Oficiales (Federal Register)"))]))
        source_status_block()
        shown = int(st.session_state.get("news_shown", 25))
        if not rows:
            st.html(f'<p class="hint">{esc(t("Ninguna noticia coincide con los filtros."))}</p>')
        for r in rows[:shown]:
            st.html(news_card_html(r) + (f'<div class="alertbar" style="margin-top:6px"><b>{esc(t("Qué hacer"))}:</b> {esc(t(r["action"]["text"]))}</div>' if r["action"] and r["relevance"] != "Baja" else ""))
            item_buttons(r, "nw")
        if len(rows) > shown:
            st.button(t("Mostrar más ({n} restantes)", n=len(rows) - shown), key="news_more", on_click=lambda: st.session_state.__setitem__("news_shown", shown + 25))
        if adv():
            with st.expander(t("Bitácora de cambios")):
                lg = NEWS.changelog(50)
                st.html(ui.table([t("Fecha"), t("Quién"), t("Acción"), t("Detalle")], [[x["ts"].replace("T", " "), x["who"], t(x["action"]), x["detail"]] for x in lg], min_w=700) if lg else
                        f'<p class="hint">{esc(t("Sin cambios registrados."))}</p>')


# ================================================================== CRM, equipo, alertas e Inicio (Fase 6E)
def role() -> str:
    return ST.cfg().get("role", "Administrador")


def who() -> str:
    n = ST.cfg().get("user", "").strip()
    return n or t("Usuario")


def can(action: str) -> bool:
    return CRM.can(role(), action)


def deny_note(action: str):
    st.html(f'<p class="hint">{esc(t("Tu rol ({r}) no permite: {a}. Cámbialo en Personalizar o en Equipo.", r=t(role()), a=t(CRM.ACTION_LABEL[action]).lower()))}</p>')


def ensure_crm():
    """Siembra clientes, oportunidades y tareas de ejemplo la primera vez y agrega como clientes las entidades nuevas de los datos."""
    did = st.session_state.get("data_id", "")
    if st.session_state.get("_crm_ready") == did:
        return
    m = default_prep().monthly
    dim = I.entity_dim(m)
    ents = []
    if dim:
        last12 = m[m["ym"] > m["ym"].max() - pd.DateOffset(months=12)]
        for v in sorted(m[dim].astype(str).unique()):
            if not v.strip():
                continue
            d = last12[last12[dim].astype(str) == v]
            us, tot = d.loc[d["mkt"] == MKT_USA, "qty"].sum(), d["qty"].sum()
            ents.append({"name": v, "type": "Cliente" if dim == "cli" else "Fuerza de venta", "country": "US" if tot > 0 and us / tot > 0.5 else "MX", "data_key": v})
    CRM.seed(ents, list(base_prep().gauges), m["ym"].max() + pd.offsets.MonthBegin(1))
    have = {c["data_key"] for c in CRM.customers() if c["data_key"]}
    for e in ents:
        if e["data_key"] not in have:
            CRM.save_customer({**e, "owner": "", "status": "Activo", "notes": "Creado a partir de los datos de ventas."}, "sistema", "Administrador")
    st.session_state["_crm_ready"] = did


def opp_value_mxn(r) -> float:
    p = get_prices()
    price = p["usd_t"].get(r["gauge"], 0) * p["fx"] if r.get("country") == "US" else p["mxn_t"].get(r["gauge"], 0)
    return float(r["tons_month"] * r["months"] * price)


def series_forecast(y: np.ndarray, h: int):
    """Pronóstico de una serie (p. ej. un cliente) con el método más certero y su rango del 80 % de la validación cruzada."""
    cv = E.cv_run(y)
    method = E.best_method(cv.table)
    fc = E.forecast(y, h, method, cv.table)
    ok = [j for j, f in enumerate(cv.fcs[method]) if f is not None]
    lo, hi = E.interval([cv.actual[j] for j in ok], [cv.fcs[method][j] for j in ok], fc, inflate=E.INFLATE_BY_WINDOWS.get(len(ok), 1.0)) if ok else (fc * .85, fc * 1.15)
    err = float(cv.table.loc[cv.table["Método"] == method, "Error %"].iloc[0])
    return fc, lo, hi, method, err


def nav_go(section: str):
    st.session_state["nav"] = section


def stage_badge(stage: str) -> str:
    kind = "fill" if stage == "Ganada" else "warn" if stage == "Perdida" else "ok"
    return f'<span class="badge {kind}">{esc(t(stage))}</span>'


def customer_stats(m, dim):
    """Para cada cliente con datos: toneladas de 12 meses, cambio contra los 12 previos, 'compra a la baja' y exportación."""
    out = {}
    if not dim:
        return out
    tr = I.entity_trends(m, dim).set_index("entidad")
    last = m["ym"].max()
    for key, d in m.assign(_k=m[dim].astype(str)).groupby("_k"):
        a = d[d["ym"] > last - pd.DateOffset(months=12)]["qty"].sum()
        b = d[(d["ym"] > last - pd.DateOffset(months=24)) & (d["ym"] <= last - pd.DateOffset(months=12))]["qty"].sum()
        out[key] = {"t12": float(a), "chg": (a / b - 1) * 100 if b > 0 else None, "baja": bool(tr.loc[key, "baja"]) if key in tr.index else False}
    return out


# ------------------------------------------------------------------ Clientes
def section_clientes():
    ensure_crm()
    prep = default_prep()
    m = prep.monthly
    dim = I.entity_dim(m)
    cust = CRM.customers()
    stats = customer_stats(m, dim)
    opps_all = CRM.opps()
    left, right = layout()
    with left:
        types = sorted({c["type"] for c in cust})
        wid(st.pills, t("Tipo"), "chips_cu_type", types, options=types, selection_mode="multi", format_func=t)
        wid(st.pills, t("País"), "chips_cu_country", ["MX", "US"], options=["MX", "US"], selection_mode="multi", format_func=lambda x: t("México") if x == "MX" else t("EE. UU."))
        q = wid(st.text_input, t("Buscar"), "cu_q", "")
        if can("clientes"):
            with st.expander(t("Nuevo cliente")):
                with st.form("new_cu", clear_on_submit=True):
                    n = st.text_input(t("Nombre"))
                    ty = st.selectbox(t("Tipo"), ["Cliente", "Distribuidor", "Constructora", "Fuerza de venta"], format_func=t)
                    co = st.radio(t("País"), ["MX", "US"], horizontal=True, format_func=lambda x: t("México") if x == "MX" else t("EE. UU."))
                    key = st.text_input(t("Valor en los datos de ventas (opcional)"), help=t("Si el nombre es igual al que trae tu archivo, se enlazan las ventas."))
                    if st.form_submit_button(t("Guardar cliente"), type="primary") and n.strip():
                        CRM.save_customer({"name": n.strip(), "type": ty, "country": co, "data_key": key.strip(), "owner": who()}, who(), role())
                        st.rerun()
        else:
            deny_note("clientes")
    sel_t, sel_c = store().get("chips_cu_type") or types, store().get("chips_cu_country") or ["MX", "US"]
    rows = [c for c in cust if c["type"] in sel_t and c["country"] in sel_c and (not q or q.lower() in c["name"].lower())]
    od = CRM.overdue_tasks()
    with right:
        st.html(ui.headline(t("Clientes: {n} registrados, {b} compran a la baja.", n=len(cust), b=sum(1 for c in cust if stats.get(c["data_key"], {}).get("baja"))),
                            t("Perfil de cada cliente con su pronóstico, mezcla de calibres, oportunidades, actividades y tareas. Si tus datos no traen cliente, la fuerza de venta hace de cliente.")))
        st.html(ui.kpis([(str(len(cust)), t("Clientes")), (str(len(opps_all)), t("Oportunidades")), (str(len(CRM.activities(tasks_only=True, open_only=True))), t("Tareas abiertas")),
                         (Raw(f'<span class="{"neg" if od else ""}">{len(od)}</span>'), t("Tareas vencidas"))]))
        table_rows = []
        for c in rows:
            s = stats.get(c["data_key"]) if c["data_key"] else None
            badges = (f' <span class="badge fill">{esc(t("EE. UU."))}</span>' if c["country"] == "US" else "") + (f' <span class="badge warn">{esc(t("Compra a la baja"))}</span>' if s and s["baja"] else "") \
                + (f' <span class="badge">{esc(t("ejemplo"))}</span>' if c["sample"] else "")
            n_op = sum(1 for o in opps_all if o["customer_id"] == c["id"] and o["stage"] in CRM.OPEN_STAGES)
            table_rows.append([Raw(f"<b>{esc(c['name'])}</b>{badges}"), t(c["type"]), f"{s['t12']:,.0f} {ui.UNIT}" if s else "—",
                               Raw(f'<span class="{"neg" if s and s["chg"] is not None and s["chg"] < 0 else ""}">{s["chg"]:+.1f}%</span>') if s and s["chg"] is not None else "—", str(n_op)])
        st.html(ui.table([t("Cliente"), t("Tipo"), t("Ventas 12 meses"), t("Cambio vs. 12 meses previos"), t("Oportunidades abiertas")], table_rows, num=(2, 3, 4), min_w=720))
        if not rows:
            return
        ids = [c["id"] for c in rows]
        cid = wid(st.selectbox, t("Ver el perfil de"), "cu_sel", ids[0], options=ids, format_func=lambda i: next(c["name"] for c in cust if c["id"] == i))
        profile(CRM.customer(cid), m, dim, stats, opps_all)


def profile(c, m, dim, stats, opps_all):
    s = stats.get(c["data_key"]) if c["data_key"] else None
    badges = (f'<span class="badge fill">{esc(t("EE. UU."))}</span> ' if c["country"] == "US" else f'<span class="badge">{esc(t("México"))}</span> ') + \
             (f'<span class="badge warn">{esc(t("Compra a la baja"))}</span>' if s and s["baja"] else "")
    st.html(f'<h2 class="vk">{esc(c["name"])} {badges}</h2><p class="sub">{esc(t(c["type"]))} · {esc(c["owner"] or "—")} · {esc(t(c["status"]))}</p>')
    if s:
        sub = m[m[dim].astype(str) == c["data_key"]]
        y = E.series(m, None, None, E.TON, None, **{dim: [c["data_key"]]})
        total12 = float(sub[sub["ym"] > m["ym"].max() - pd.DateOffset(months=12)]["qty"].sum())
        share = total12 / max(float(m[m["ym"] > m["ym"].max() - pd.DateOffset(months=12)]["qty"].sum()), 1e-9) * 100
        lastbuy = y[y > 0].index.max()
        st.html(ui.kpis([(f"{total12:,.0f} {ui.UNIT}", t("Ventas últimos 12 meses")), (Raw(f'<span class="{"neg" if s["chg"] is not None and s["chg"] < 0 else ""}">{s["chg"]:+.1f}%</span>') if s["chg"] is not None else "—", t("Cambio contra los 12 meses previos")),
                         (f"{share:.1f}%", t("Del volumen total")), (mes_largo(lastbuy) if pd.notna(lastbuy) else "—", t("Última compra"))]))
        if len(y) >= 12 and (y > 0).sum() >= 6:
            fc, lo, hi, method, err = series_forecast(y.values, 6)
            idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=6, freq="MS")
            st.html(ui.h2(t("Pronóstico de este cliente"), t("Método {m} con {p}% de precisión en la validación. Rango del 80%.", m=t(method), p=f"{max(0, 100 - err):.0f}")))
            st.plotly_chart(ui.forecast_fig(list(y.index), y.values, list(idx), fc, lo, hi, ui.UNIT, t(method), height=300), width="stretch", theme=None, config={"displayModeBar": False})
        mix = sub[sub["ym"] > m["ym"].max() - pd.DateOffset(months=12)].groupby("cal")["qty"].sum().sort_values(ascending=False)
        if len(mix):
            mx = float(mix.max())
            st.html(ui.h2(t("Mezcla de calibres")))
            st.html(ui.table([t("Calibre"), t("Toneladas 12 meses"), t("Mezcla")], [[ui.cal_label(g), f"{v:,.1f}", ui.mix_bar(v / mix.sum() * 100, mx / mix.sum() * 100)] for g, v in mix.items()], num=(1,)))
    else:
        st.html(f'<p class="hint">{esc(t("Este cliente no tiene ventas enlazadas en los datos."))}</p>')
    mine = [o for o in opps_all if o["customer_id"] == c["id"]]
    st.html(ui.h2(t("Oportunidades")))
    if mine:
        st.html(ui.table([t("Oportunidad"), t("Etapa"), t("Calibre"), t("t por mes"), t("Meses"), t("Cierre estimado")],
                         [[o["name"], Raw(stage_badge(o["stage"])), ui.cal_label(o["gauge"]), f"{o['tons_month']:,.1f}", str(o["months"]), o["close_date"] or "—"] for o in mine], num=(3, 4)))
    else:
        st.html(f'<p class="hint">{esc(t("Sin oportunidades."))}</p>')
    acts = CRM.activities(customer_id=c["id"])
    st.html(ui.h2(t("Actividades y tareas")))
    today = dt.date.today().isoformat()
    if not acts:
        st.html(f'<p class="hint">{esc(t("Sin actividades."))}</p>')
    for a in acts[:20]:
        od = a["is_task"] and not a["done"] and a["due"] and a["due"] < today
        badge = (f'<span class="badge warn">{esc(t("Vencida"))}</span>' if od else f'<span class="badge fill">{esc(t("Hecha"))}</span>' if a["is_task"] and a["done"] else
                 f'<span class="badge ok">{esc(t("Tarea"))}</span>' if a["is_task"] else f'<span class="badge">{esc(t(a["type"]))}</span>')
        due = f' · {esc(t("vence"))} {a["due"]}' if a["is_task"] and a["due"] else ""
        st.html(f'<div class="tl"><span class="tlm">{esc(a["date"])}{due}</span> {badge} {esc(a["note"])}</div>')
        if a["is_task"] and not a["done"] and can("tareas"):
            st.button(t("Marcar hecha"), key=f"done_{a['id']}", on_click=lambda i=a["id"]: CRM.complete_task(i, who(), role()))
    if can("tareas"):
        with st.expander(t("Registrar actividad o tarea")):
            with st.form(f"act_{c['id']}", clear_on_submit=True):
                ty = st.selectbox(t("Tipo"), ["Llamada", "Visita", "Correo", "Reunión", "Tarea"], format_func=t)
                note = st.text_input(t("Descripción"))
                due = st.date_input(t("Vence (solo tareas)"), value=dt.date.today() + dt.timedelta(days=7))
                if st.form_submit_button(t("Guardar"), type="primary") and note.strip():
                    CRM.add_activity({"customer_id": c["id"], "type": ty, "note": note.strip(), "is_task": int(ty == "Tarea"), "due": due.isoformat() if ty == "Tarea" else None, "owner": who()}, who(), role())
                    st.rerun()
    else:
        deny_note("tareas")
    if can("clientes"):
        with st.expander(t("Editar cliente")):
            with st.form(f"edit_{c['id']}"):
                n = st.text_input(t("Nombre"), c["name"])
                co = st.radio(t("País"), ["MX", "US"], index=0 if c["country"] == "MX" else 1, horizontal=True, format_func=lambda x: t("México") if x == "MX" else t("EE. UU."))
                stt = st.selectbox(t("Estado"), ["Activo", "En riesgo", "Inactivo"], index=["Activo", "En riesgo", "Inactivo"].index(c["status"]) if c["status"] in ("Activo", "En riesgo", "Inactivo") else 0, format_func=t)
                notes = st.text_area(t("Notas"), c["notes"] or "")
                if st.form_submit_button(t("Guardar cambios"), type="primary"):
                    CRM.save_customer({**c, "name": n.strip() or c["name"], "country": co, "status": stt, "notes": notes}, who(), role())
                    st.rerun()


# ------------------------------------------------------------------ Oportunidades
def move_cb(oid: int, key: str):
    new = st.session_state.get(key)
    if new and can("oportunidades"):
        CRM.move_opp(oid, new, who(), role())


def section_oportunidades():
    ensure_crm()
    m = default_prep().monthly
    opps_all = CRM.opps()
    cust = {c["id"]: c for c in CRM.customers()}
    gauges = list(base_prep().gauges)
    summ = CRM.pipeline_summary(opps_all)
    first = m["ym"].max() + pd.offsets.MonthBegin(1)
    arr = CRM.crm_arrays(opps_all, first, 12)
    w6 = sum(float(v[:6].sum()) for d in arr.values() for v in d.values())
    left, right = layout()
    with left:
        gsel = wid(st.pills, t("Calibre"), "chips_op_cal", gauges, options=gauges, selection_mode="multi", format_func=ui.cal_label)
        st.caption(t("Las oportunidades abiertas entran al pronóstico ponderadas por la probabilidad de su etapa; las ganadas cuentan al 100% mientras no aparezcan en tus ventas."))
        wid(st.toggle, t("Usar también las oportunidades de ejemplo en el pronóstico"), "crm_use_sample", False)
        if can("oportunidades"):
            with st.expander(t("Nueva oportunidad")):
                opp_form(None, cust, gauges, first)
        else:
            deny_note("oportunidades")
    sel = store().get("chips_op_cal") or gauges
    with right:
        st.html(ui.headline(t("Oportunidades: {a} abiertas suman {b} t de pipeline y {c} t ponderadas.", a=summ["open_n"], b=f"{summ['open_tons']:,.0f}", c=f"{summ['weighted_tons']:,.0f}"),
                            t("Mover una oportunidad de etapa o cambiar sus toneladas actualiza el pronóstico al instante: las abiertas aportan {x} t a los próximos 6 meses.", x=f"{w6:,.0f}")))
        val_open = sum(opp_value_mxn(o) for o in opps_all if o["stage"] in CRM.OPEN_STAGES)
        st.html(ui.kpis([(f"{summ['open_tons']:,.0f} {ui.UNIT}", t("Pipeline abierto"), f"{ui.mxn(val_open)} · {summ['open_n']} " + t("oportunidades")), (f"{summ['weighted_tons']:,.0f} {ui.UNIT}", t("Pipeline ponderado"),
                         t("{x} t por mes", x=f"{summ['weighted_month']:,.1f}")), (f"{summ['win_rate']:.0f}%" if summ["win_rate"] is not None else "—", t("Tasa de éxito"), t("{a} ganadas, {b} perdidas", a=summ["won_n"], b=summ["lost_n"])),
                         (str(summ["closing_n"]), t("Cierran este mes"), f"{summ['closing_tons']:,.0f} {ui.UNIT}")]))
        cols = st.columns(len(CRM.STAGES), gap="small")
        names = [s for s, _ in CRM.STAGES]
        for col, (stage, prob) in zip(cols, CRM.STAGES):
            cards = [o for o in opps_all if o["stage"] == stage and o["gauge"] in sel]
            with col:
                st.html(f'<div class="kcol"><b>{esc(t(stage))}</b> <span class="syncnote">{prob}% · {len(cards)} · {sum(o["tons_month"] for o in cards):,.0f} {ui.UNIT}/{esc(t("mes"))}</span></div>')
                for o in cards:
                    cu = cust.get(o["customer_id"], {})
                    us = f' <span class="badge fill">{esc(t("EE. UU."))}</span>' if cu.get("country") == "US" else ""
                    st.html(f'<div class="kcard"><b>{esc(o["name"])}</b><br><span class="syncnote">{esc(cu.get("name", "—"))}{us}</span><br>'
                            f'{esc(ui.cal_label(o["gauge"]))} · {o["tons_month"]:,.1f} {ui.UNIT}/{esc(t("mes"))} × {o["months"]}<br><span class="syncnote">{esc(t("cierra"))} {esc(o["close_date"] or "—")}</span></div>')
                    k = f"mv_{o['id']}"
                    st.session_state[k] = o["stage"]
                    st.selectbox(t("Mover a"), names, key=k, label_visibility="collapsed", format_func=t, on_change=move_cb, args=(o["id"], k), disabled=not can("oportunidades"))
        if can("oportunidades") and opps_all:
            st.html(ui.h2(t("Editar una oportunidad")))
            oid = wid(st.selectbox, t("Oportunidad"), "op_sel", opps_all[0]["id"], options=[o["id"] for o in opps_all], format_func=lambda i: next(o["name"] for o in opps_all if o["id"] == i))
            opp_form(next(o for o in opps_all if o["id"] == oid), cust, gauges, first)


def opp_form(o, cust, gauges, first):
    months = [(first + pd.DateOffset(months=k)).strftime("%Y-%m") for k in range(-2, 12)]
    with st.form(f"opp_{o['id'] if o else 'new'}", clear_on_submit=not o):
        ids = list(cust)
        name = st.text_input(t("Nombre"), o["name"] if o else "")
        cu = st.selectbox(t("Cliente"), ids, index=ids.index(o["customer_id"]) if o and o["customer_id"] in ids else 0, format_func=lambda i: cust[i]["name"])
        c1, c2 = st.columns(2)
        stage = c1.selectbox(t("Etapa"), [s for s, _ in CRM.STAGES], index=[s for s, _ in CRM.STAGES].index(o["stage"]) if o else 0, format_func=t)
        gauge = c2.selectbox(t("Calibre"), gauges, index=gauges.index(o["gauge"]) if o and o["gauge"] in gauges else 0, format_func=ui.cal_label)
        c3, c4, c5 = st.columns(3)
        tons = c3.number_input(t("Toneladas por mes"), min_value=0.0, step=1.0, value=float(o["tons_month"]) if o else 10.0)
        mths = c4.number_input(t("Meses"), min_value=1, max_value=24, step=1, value=int(o["months"]) if o else 6)
        start = c5.selectbox(t("Primer mes"), months, index=months.index(o["start"]) if o and o["start"] in months else 2)
        close = st.date_input(t("Cierre estimado"), value=dt.date.fromisoformat(o["close_date"]) if o and o["close_date"] else dt.date.today() + dt.timedelta(days=30))
        notes = st.text_area(t("Notas"), o["notes"] if o else "")
        if st.form_submit_button(t("Guardar cambios") if o else t("Crear oportunidad"), type="primary") and name.strip():
            CRM.save_opp({"id": o["id"] if o else None, "customer_id": cu, "name": name.strip(), "stage": stage, "gauge": gauge, "tons_month": tons, "months": mths, "start": start,
                          "close_date": close.isoformat(), "owner": who(), "notes": notes}, who(), role())
            st.rerun()


# ------------------------------------------------------------------ Equipo
def section_equipo():
    ensure_crm()
    st.html(ui.headline(t("Equipo: roles, permisos y bitácora de cambios."), t("Elige con qué rol ves la app. Los roles ordenan quién puede hacer qué; no son seguridad formal.")))
    c = st.columns(2, gap="large")
    with c[0]:
        r = st.selectbox(t("Ver como"), CRM.ROLES, index=CRM.ROLES.index(role()), format_func=t, key="eq_role")
        if r != role():
            ST.put("role", r)
            CRM.audit(who(), r, "cambiar rol", "equipo", "", t("Ahora se ve como {r}", r=t(r)))
            st.rerun()
    with c[1]:
        nm = st.text_input(t("Tu nombre (aparece en la bitácora)"), ST.cfg().get("user", ""), key="eq_user")
        if nm != ST.cfg().get("user", ""):
            ST.put("user", nm)
    acts = list(CRM.ACTION_LABEL)
    rows = [[Raw(f"<b>{esc(t(CRM.ACTION_LABEL[a]))}</b>"), *[("✓" if CRM.can(r_, a) else "—") for r_ in CRM.ROLES]] for a in acts]
    st.html(ui.h2(t("Permisos por rol")))
    st.html(ui.table([t("Acción"), *[t(r_) for r_ in CRM.ROLES]], rows, num=(1, 2, 3, 4)))
    st.html(ui.h2(t("Bitácora de cambios"), t("Quién, qué y cuándo. Incluye clientes, oportunidades, tareas, anomalías, pronósticos congelados y noticias.")))
    log = [{"ts": x["ts"], "who": f"{x['who']} ({t(x['role'])})", "action": x["action"], "detail": x["detail"]} for x in CRM.audit_log(400)]
    log += [{"ts": x["ts"], "who": x["who"], "action": x["action"], "detail": x["detail"]} for x in NEWS.changelog(200)]
    log.sort(key=lambda x: x["ts"], reverse=True)
    q = st.text_input(t("Filtrar la bitácora"), key="eq_q")
    log = [x for x in log if not q or q.lower() in (x["action"] + x["detail"] + x["who"]).lower()]
    st.html(ui.table([t("Fecha"), t("Quién"), t("Acción"), t("Detalle")], [[x["ts"].replace("T", " "), x["who"], t(x["action"]), x["detail"]] for x in log[:150]], min_w=760) if log else
            f'<p class="hint">{esc(t("Sin cambios registrados."))}</p>')


# ================================================================== Inicio completo, anomalías, bajo volumen y seguimiento
def anomaly_task(key: str, text: str, entity: str, pair: str, kind: str):
    """'Confirmado — dar seguimiento': guarda la decisión y crea una tarea de CRM para el cliente."""
    if not can("anomalias"):
        return
    cust = next((c for c in CRM.customers() if c["data_key"] == entity or c["name"] == entity), None)
    if cust is None:
        cid = CRM.save_customer({"name": entity, "type": "Fuerza de venta", "country": "MX", "data_key": entity, "owner": who(), "status": "Activo", "notes": "Creado desde una anomalía."}, who(), role())
    else:
        cid = cust["id"]
    CRM.add_activity({"customer_id": cid, "type": "Tarea", "is_task": 1, "due": (dt.date.today() + dt.timedelta(days=7)).isoformat(), "note": t("Dar seguimiento: ") + text, "owner": who(), "anomaly_key": key}, who(), role())
    CRM.add_decision(key, pair, kind, "confirmado", who(), role(), "tarea creada")


def anomaly_decide(key: str, pair: str, kind: str, status: str):
    if can("anomalias"):
        CRM.add_decision(key, pair, kind, status, who(), role())


@st.cache_data(show_spinner=False)
def cached_rhythm(monthly: pd.DataFrame, dim, tol_key: tuple, gf: float) -> pd.DataFrame:
    return I.rhythm_anomalies(monthly, dim, dict(tol_key), gf)


def pending_anomalies(m, level: str = "Calibre"):
    """Anomalías de ritmo de pedidos sin decisión, ya ajustadas por lo aprendido de las falsas alarmas."""
    dim = I.entity_dim(m)
    if dim is None:
        return pd.DataFrame()
    decs = CRM.decisions()
    tol, gf = I.learned_tolerance(decs)
    an = cached_rhythm(m, dim, tuple(sorted(tol.items())), gf)
    done = {d["key"] for d in decs}
    an = an[~an["key"].isin(done)]
    return an[an["level"] == level].reset_index(drop=True) if len(an) else an


def anomaly_queue(m, limit: int = 6):
    """Cola de revisión: cada anomalía se confirma (crea tarea), se descarta como falsa alarma o se marca como ya resuelta."""
    an = pending_anomalies(m)
    st.html(ui.h2(t("Cola de revisión de anomalías"), t("Aprende el ritmo normal de pedidos de cada cliente (o fuerza de venta) y calibre, y avisa cuando se rompe. Tú decides: lo que marques como falsa alarma sube el umbral de esa pareja.")))
    if an.empty:
        st.html(f'<p class="hint">{esc(t("Sin anomalías pendientes de revisar."))}</p>')
    else:
        for i, r in an.head(limit).iterrows():
            kind = t("Retraso") if r["kind"] == "retraso" else t("Cantidad")
            st.html(f'<div class="newscard"><div class="newshead"><span class="badge warn">{esc(kind)}</span><span class="newsmeta">{esc(t("en riesgo"))}: {r["at_risk"]:,.1f} t</span></div>'
                    f'<span class="newstitle">{esc(_anom_text(r))}</span></div>')
            c = st.columns([2.2, 1.2, 1.2, 3])
            dis = not can("anomalias")
            c[0].button(t("Confirmado — dar seguimiento"), key=f"an_ok_{i}", type="primary", disabled=dis, on_click=anomaly_task, args=(r["key"], _anom_text(r), r["entity"], r["pair"], r["kind"]))
            c[1].button(t("Falsa alarma"), key=f"an_fa_{i}", disabled=dis, on_click=anomaly_decide, args=(r["key"], r["pair"], r["kind"], "falsa"))
            c[2].button(t("Ya resuelto"), key=f"an_re_{i}", disabled=dis, on_click=anomaly_decide, args=(r["key"], r["pair"], r["kind"], "resuelto"))
        if len(an) > limit:
            st.caption(t("Hay {n} anomalías más por calibre en la cola.", n=len(an) - limit))
    if not can("anomalias"):
        deny_note("anomalias")
    prod = pending_anomalies(m, "Producto")
    if len(prod):
        with st.expander(t("Anomalías por producto ({n})", n=len(prod))):
            st.html(ui.table([t("Descripción"), t("En riesgo (t)")], [[_anom_text(r), f"{r['at_risk']:,.1f}"] for _, r in prod.head(40).iterrows()], num=(1,)))
            st.caption(t("Para decidir sobre un producto, revisa su calibre en la cola de arriba."))
    decs = CRM.decisions()
    if decs:
        fa = sum(d["status"] == "falsa" for d in decs)
        st.html(f'<p class="syncnote">{esc(t("{n} decisiones guardadas ({f} falsas alarmas). Cada falsa alarma sube 50% el umbral de esa pareja, hasta el triple.", n=len(decs), f=fa))}</p>')


def _anom_text(r) -> str:
    """Texto de la anomalía en el idioma de la interfaz (el motor lo genera en español)."""
    if ST.cfg()["lang"] != "en":
        return r["text"]
    item = f"gauge {r['item']} AWG" if r["level"] == "Calibre" else f"product {r['item']}"
    lo = f"{r['last_order']:%Y-%m}"
    if r["kind"] == "retraso":
        every = "every month" if r["normal_gap"] <= 1.01 else f"every {r['normal_gap']:.0f} months"
        return f"{r['entity']} normally buys {item} {every} and has gone {r['now_gap']} months without buying (last order: {lo})."
    return f"{r['entity']} normally orders {r['normal_qty']:.1f} {ui.UNIT} of {item} per order and the last one was {r['last_qty']:.1f} {ui.UNIT} ({r['last_qty'] / r['normal_qty'] * 100:.0f}% of normal, {lo})."


def business_alerts(m, pr, res, frozen_rows) -> list[dict]:
    """'Qué atender' de negocio: calibres que caen, pronósticos poco confiables, exportación a la baja, capacidad, clientes a la baja, tareas vencidas, anomalías y desviaciones."""
    out = []
    tr = I.gauge_trends(m)
    for r in tr[(tr["cambio"] < -8) & (tr["anterior"] >= 5)].itertuples():
        out.append({"sev": "Alta" if r.cambio < -15 else "Media", "text": t("El calibre {g} vende {p}% menos que hace un año (últimos 3 meses).", g=ui.cal_label(r.calibre), p=f"{abs(r.cambio):.0f}"),
                    "action": t("Revisar el pronóstico y las señales de ese calibre."), "section": "Pronóstico"})
    tot12 = float(m[m["ym"] > m["ym"].max() - pd.DateOffset(months=12)]["qty"].sum())
    for d in pr.detail:
        if np.isfinite(d["error"]) and d["error"] >= 20:
            vol = float(d["y"][-12:].sum())
            if vol / max(tot12, 1e-9) >= 0.01:
                out.append({"sev": "Media", "text": t("El pronóstico del calibre {g} es poco confiable ({p}% de precisión).", g=ui.cal_label(d["label"]), p=f"{max(0, 100 - d['error']):.0f}"),
                            "action": t("Usar un método distinto o ajustar el consenso con una justificación."), "section": "Pronóstico"})
    ex = I.export_trend(m)
    if ex and ex["cambio"] is not None and ex["cambio"] < -15:
        out.append({"sev": "Alta", "text": t("La exportación a EE. UU. cae {p}% contra el año anterior (últimos 3 meses).", p=f"{abs(ex['cambio']):.0f}"), "action": t("Revisar el escenario arancelario."), "section": "Señales"})
    cap = res.capacity
    over = [r.cal for r in cap.itertuples() if r.over > 0] if len(cap) else []
    if over:
        out.append({"sev": "Alta", "text": t("Capacidad excedida en: {x}.", x=", ".join(ui.cal_label(g) for g in over)), "action": t("Revisar la capacidad y el plan de producción."), "section": "Planeación"})
    dim = I.entity_dim(m)
    for r in I.entity_trends(m, dim).query("baja").head(3).itertuples():
        out.append({"sev": "Media", "text": t("{c} compra {p}% menos que en los 6 meses previos.", c=r.entidad, p=f"{abs(r.cambio):.0f}"), "action": t("Llamar al cliente y revisar sus oportunidades."), "section": "Clientes"})
    od = CRM.overdue_tasks()
    if od:
        out.append({"sev": "Alta", "text": t("{n} tareas vencidas; la más antigua: {x}", n=len(od), x=od[0]["note"][:90]), "action": t("Completarlas o reprogramarlas."), "section": "Clientes"})
    n_an = len(pending_anomalies(m)) if adv() else 0
    if n_an:
        out.append({"sev": "Media", "text": t("{n} anomalías del ritmo de pedidos esperan revisión.", n=n_an), "action": t("Revisarlas en la cola de abajo."), "section": "Inicio"})
    for f in (frozen_rows if adv() else []):
        cmp_ = I.frozen_compare(f["payload"], m)
        tot = cmp_[cmp_["calibre"] == "Total"] if len(cmp_) else cmp_
        if len(tot) and np.isfinite(tot["error_est"].iloc[0]) and tot["error_est"].iloc[0] > 15:
            out.append({"sev": "Media", "text": t("El pronóstico congelado «{n}» se desvía {p}% de lo vendido.", n=f["name"], p=f"{tot['error_est'].iloc[0]:.0f}"), "action": t("Revisar qué cambió desde que se congeló."), "section": "Pronóstico"})
    return out


KPI_GO = {"fc": "Pronóstico", "acc": "Laboratorio", "mar": "Planeación", "pipe": "Oportunidades", "task": "Clientes", "news": "Noticias"}


def tile(key: str, value, label: str, sub: str, section: str):
    """Mosaico de KPI en el que se puede hacer clic: lleva al detalle."""
    with st.container(key=f"tile_{key}"):
        st.html(f'<div class="tile"><b>{value if isinstance(value, Raw) else ui.count_html(value, "tile:" + key)}</b><span>{esc(label)}</span><small>{esc(sub)}</small></div>')
        st.button("→", key=f"tilebtn_{key}", on_click=nav_go, args=(section,))


def section_inicio():
    ensure_news()
    ensure_crm()
    prep = default_prep()
    m = prep.monthly
    ml = get_ml(m, E.windows_for(m))
    prices, mult, sig, w, params = get_prices(), get_mult(), get_sig(), get_w(), get_params()
    gauges = tuple(prep.gauges)
    h = 6
    pr = cached_project(m, tuple((g,) for g in gauges), None, h, E.AUTO, E.TON, "sum", mult, sig, w, prices, ml, get_crm(None, E.TON, prices), apply_signals_on())
    res = cached_plan(m, gauges, h, params, prices, mult, sig, w, {"metal": sv("ov_metal", {}), "inv": sv("ov_inv", {}), "cap": sv("ov_cap", {})}, ml, get_crm(None, E.TON, prices))
    tot = PL.totals(res)
    summ = CRM.pipeline_summary(CRM.opps())
    od = CRM.overdue_tasks()
    frozen_rows = CRM.frozen_list()
    alerts = business_alerts(m, pr, res, frozen_rows)
    st.html(ui.headline(t("Esperamos {v} {u} en los próximos {h} meses.", v=f"{pr.fc.sum():,.0f}", u=ui.UNIT, h=h),
                        t("Inicio resume el pronóstico, el plan, el CRM y lo que hay que atender hoy. Toca un mosaico para ver su detalle.")))
    st.button(t("Resumen del mes"), key="btn_summary", type="primary", on_click=lambda: store().__setitem__("show_summary", not store().get("show_summary", False)))
    if store().get("show_summary", False):
        summary_panel(m, pr, res, alerts, h, ml)
    insights_box(h)
    news_hi = len(NEWS.load(days=14, relevance=["Alta"], statuses=["nuevo", "relevante"]))
    with st.container(key="tiles"):
        cols = st.columns(6, gap="small")
        with cols[0]:
            tile("fc", f"{pr.fc.sum():,.0f} {ui.UNIT}", t("Pronóstico {h} meses", h=h), t("rango {a} – {b}", a=f"{pr.lo.sum():,.0f}", b=f"{pr.hi.sum():,.0f}"), KPI_GO["fc"])
        with cols[1]:
            tile("acc", f"{max(0, 100 - pr.error):.1f}%", t("Precisión de validación"), t("ver el laboratorio") if adv() else t("ver el pronóstico"), KPI_GO["acc"] if adv() else "Pronóstico")
        with cols[2]:
            tile("mar", f"{tot['margin_pct']:.1f}%", t("Margen bruto"), ui.mxn(tot["margin"]), KPI_GO["mar"])
        with cols[3]:
            tile("pipe", f"{summ['weighted_tons']:,.0f} {ui.UNIT}", t("Pipeline ponderado"), t("{n} abiertas", n=summ["open_n"]), KPI_GO["pipe"])
        with cols[4]:
            tile("task", Raw(f'<span class="{"neg" if od else ""}">{len(od)}</span>'), t("Tareas vencidas"), t("ir a Clientes"), KPI_GO["task"])
        with cols[5]:
            tile("news", str(news_hi), t("Noticias de relevancia alta"), t("ir a Noticias"), KPI_GO["news"])
    st.plotly_chart(ui.forecast_fig(list(pr.hist.index), pr.hist.values, list(pr.index), pr.fc, pr.lo, pr.hi, ui.UNIT, t("Suma de calibres"), height=330), width="stretch", theme=None, config={"displayModeBar": False})
    # qué atender
    items = alerts + [{"sev": "Alta" if True else "Media", "text": x, "action": t("Revisar el escenario de precio del metal y la recomendación de compra en Metales."), "section": "Metales", "kind": "mercado"}
                      for x in _safe_market_texts()]
    news_items = []
    seen = {}
    for r in NEWS.load(days=14, relevance=["Alta"], statuses=["nuevo", "relevante"]):
        if seen.get(r["topic"], 0) >= 2:
            continue
        seen[r["topic"]] = seen.get(r["topic"], 0) + 1
        news_items.append(r)
    st.html(ui.h2(t("Qué atender"), t("Alertas de negocio, de mercado y de noticias, cada una con la acción que sugiere. Nada cambia el pronóstico hasta que tú lo confirmas.")))
    with st.container(key="alerts"):
        if not items and not news_items:
            st.html(f'<p class="hint">{esc(t("Sin alertas por ahora."))}</p>')
        for i, it in enumerate(sorted(items, key=lambda x: 0 if x["sev"] == "Alta" else 1)[:10]):
            st.html(f'<div class="alertbar{" soft" if it["sev"] != "Alta" else ""}">{esc(it["text"])}<br><b>{esc(t("Qué hacer"))}:</b> {esc(it["action"])}</div>')
            if it["section"] != "Inicio":
                st.button(t("Ir a {s}", s=t(it["section"])), key=f"go_{i}", on_click=nav_go, args=(it["section"],))
        for i, r in enumerate(news_items):
            st.html(news_card_html(r) + (f'<div class="alertbar"><b>{esc(t("Qué hacer"))}:</b> {esc(t(r["action"]["text"]))}</div>' if r["action"] else ""))
            item_buttons(r, f"in{i}")
    if adv():
        anomaly_queue(m)
    st.html(ui.h2(t("Últimas noticias relevantes")))
    rows = latest_news(None, 4)
    st.html("".join(news_card_html(r, compact=True) for r in rows) if rows else f'<p class="hint">{esc(t("Todavía no hay noticias guardadas."))}</p>')


def _safe_market_texts() -> list[str]:
    try:
        return market_alert_texts(live_snapshot())
    except Exception:
        return []


# ------------------------------------------------------------------ bajo volumen y seguimiento (en Pronóstico, modo avanzado)
@st.cache_data(show_spinner=False)
def cached_lowvol(monthly: pd.DataFrame, gauges: tuple) -> pd.DataFrame:
    return I.low_volume(monthly, list(gauges))


def lowvol_block(m, gauges):
    st.html(ui.h2(t("Productos de bajo volumen"), t("Productos intermitentes o pequeños (menos de 12 t en 12 meses). Su demanda es difícil de pronosticar mes a mes; Croston-SBA y TSB estiman cuánto vender en 6 meses.")))
    lv = cached_lowvol(m, tuple(str(g) for g in gauges))
    if lv.empty:
        st.html(f'<p class="hint">{esc(t("No hay productos de bajo volumen en esta selección."))}</p>')
        return
    rows = [[Raw(f"<b>{esc(r.producto)}</b>"), ui.cal_label(r.calibre), esc(r.familia), f"{r.ventas12:,.1f} {ui.UNIT}", f"{r.meses_con_venta} / 24", ui.dclass_html(r.clase), f"{r.croston:,.1f} {ui.UNIT}", f"{r.tsb:,.1f} {ui.UNIT}"]
            for r in lv.itertuples()]
    st.html(ui.table([t("Producto"), t("Calibre"), t("Familia"), t("Ventas 12 meses"), t("Meses con venta"), t("Tipo de demanda"), t("Croston-SBA 6 m"), t("TSB 6 m")], rows, num=(3, 6, 7), min_w=940))
    st.html(f'<p class="syncnote">{esc(t("Se muestran los {n} de mayor volumen. Meses con venta cuenta los últimos 24.", n=len(lv)))}</p>')


def freeze_payload(m, ml, h) -> dict:
    prices, mult, sig, w = get_prices(), get_mult(), get_sig(), get_w()
    out = {"horizon": h, "gauges": {}}
    try:
        raw = waterfall_raw(h)
        out["factors"] = {"months": raw["months"], "f": {k: [float(x) for x in v] for k, v in raw["factors"].items()}}
    except Exception:
        pass
    for g in sorted(set(m["cal"])):
        pr = cached_project(m, ((g,),), None, h, E.AUTO, E.TON, "sum", mult, sig, w, prices, ml, get_crm(None, E.TON, prices), apply_signals_on())
        cons = pr.fc
        ck = f"{g}|None|{E.TON}|{h}|{E.AUTO}"
        c = store().get("consensus", {}).get(ck)
        if c:
            cons, _ = E.consensus(pr.fc, c["adj"], c["why"])
        out["gauges"][g] = {"months": [f"{x:%Y-%m}" for x in pr.index], "stat": [float(v) for v in pr.fc], "cons": [float(v) for v in cons]}
    return out


def do_freeze(m, ml, h):
    name = st.session_state.get("freeze_name", "").strip() or t("Pronóstico {d}", d=dt.date.today().isoformat())
    CRM.freeze(name, f"{m['ym'].max():%Y-%m}", freeze_payload(m, ml, h), who(), role())
    st.session_state["freeze_name"] = ""


def tracking_block(m, ml, gauges):
    """Congelar el pronóstico, compararlo con lo que se vende después y simular un pronóstico hecho hace 6 meses."""
    st.html(ui.h2(t("Seguimiento del pronóstico"), t("Congela el pronóstico de hoy y compáralo después con lo vendido. El valor agregado del pronóstico (FVA) es el error del estadístico menos el error del consenso: positivo significa que el consenso ayudó.")))
    h = horizon_default()
    c = st.columns([3, 1.4], vertical_alignment="bottom")
    c[0].text_input(t("Nombre del pronóstico congelado"), key="freeze_name", placeholder=t("Ej. Pronóstico de octubre"), disabled=not can("congelar"))
    c[1].button(t("Congelar pronóstico"), key="freeze_btn", type="primary", disabled=not can("congelar"), on_click=do_freeze, args=(m, ml, h), width="stretch")
    if not can("congelar"):
        deny_note("congelar")
    fr = CRM.frozen_list()
    if not fr:
        st.html(f'<p class="hint">{esc(t("Todavía no hay pronósticos congelados."))}</p>')
    for f in fr:
        cmp_ = I.frozen_compare(f["payload"], m)
        st.html(f'<p><b>{esc(f["name"])}</b> <span class="syncnote">· {esc(t("congelado el {d} con datos hasta {m}", d=f["frozen_at"][:10], m=f["last_data"]))}</span></p>')
        if cmp_.empty:
            st.html(f'<p class="hint">{esc(t("Todavía no hay ventas posteriores para compararlo."))}</p>')
        else:
            rows = [[Raw(f"<b>{esc(t('Total') if r.calibre == 'Total' else ui.cal_label(r.calibre))}</b>"), str(r.meses), f"{r.real:,.0f} {ui.UNIT}", f"{r.estadistico:,.0f} {ui.UNIT}", f"{r.consenso:,.0f} {ui.UNIT}",
                     f"{r.error_est:.1f}%", f"{r.error_con:.1f}%", f"{r.fva:+.2f} {t('pts')}"] for r in cmp_.itertuples()]
            st.html(ui.table([t("Calibre"), t("Meses con datos"), t("Real"), t("Estadístico"), t("Consenso"), t("Error estadístico"), t("Error consenso"), t("FVA")], rows, num=(1, 2, 3, 4, 5, 6, 7), min_w=820))
        if can("congelar"):
            st.button(t("Borrar este pronóstico congelado"), key=f"fz_del_{f['id']}", on_click=lambda i=f["id"]: CRM.delete_frozen(i, who(), role()))
    with st.expander(t("Seguimiento de prueba: un pronóstico hecho hace 6 meses")):
        tr = I.tracking_test(m, sorted(set(m["cal"])), 6)
        if tr.empty:
            st.html(f'<p class="hint">{esc(t("No hay suficiente historia para simularlo."))}</p>')
        else:
            rows = [[Raw(f"<b>{esc(t('Total') if r.calibre == 'Total' else ui.cal_label(r.calibre))}</b>"), f"{r.real:,.0f} {ui.UNIT}", f"{r.pronostico:,.0f} {ui.UNIT}", f"{r.error:.1f}%", f"{r.sesgo:+.1f}%", t(r.metodo) if r.metodo else ""] for r in tr.itertuples()]
            st.html(ui.table([t("Calibre"), t("Vendido"), t("Pronóstico hecho hace 6 meses"), t("Error"), t("Sesgo"), t("Método")], rows, num=(1, 2, 3, 4), min_w=720))
            st.html(f'<p class="syncnote">{esc(t("Es el pronóstico estadístico (sin machine learning ni consenso) que se habría hecho con los datos de hace 6 meses, medido contra lo que de verdad se vendió."))}</p>')


# ================================================================== 6F: portafolio, sensibilidad, cascada y resumen del mes
def margin_per_ton(res) -> dict:
    r = res.rows
    if r.empty:
        return {}
    g = r.groupby("cal")[["margin", "vol"]].sum()
    return {k: float(v["margin"] / v["vol"]) for k, v in g.iterrows() if v["vol"] > 0}


@st.cache_data(show_spinner=False)
def cached_abc(monthly: pd.DataFrame, margin_t: tuple) -> pd.DataFrame:
    return I.abc_xyz(monthly, dict(margin_t))


@st.cache_data(show_spinner=False)
def cached_sens(monthly: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    return I.price_sensitivity(monthly, hist)


SUG_BADGE = {"Impulsar": "fill", "Revisar": "warn", "Considerar descontinuar": "neg", "Mantener": ""}


def _motivo_t(s: str) -> str:
    mt = re.match(r"Sin ventas hace (\d+) meses y muy poco volumen\.", s)
    return t("Sin ventas hace {n} meses y muy poco volumen.", n=mt.group(1)) if mt else t(s)


def section_portafolio():
    prep = default_prep()
    m = prep.monthly
    ml = get_ml(m, E.windows_for(m))
    gauges = tuple(prep.gauges)
    res = cached_plan(m, gauges, 12, get_params(), get_prices(), get_mult(), get_sig(), get_w(), {"metal": sv("ov_metal", {}), "inv": sv("ov_inv", {}), "cap": sv("ov_cap", {})}, ml, get_crm(None, E.TON, get_prices()))
    df = cached_abc(m, tuple(sorted(margin_per_ton(res).items())))
    if df.empty:
        st.html(f'<p class="hint">{esc(t("No hay productos con ventas en los últimos 12 meses."))}</p>')
        return
    a_share = df.loc[df["abc"] == "A", "vol12"].sum() / df["vol12"].sum() * 100
    st.html(ui.headline(t("{n} productos vendieron en los últimos 12 meses; los {a} de clase A son {p}% del volumen.", n=len(df), a=int((df["abc"] == "A").sum()), p=f"{a_share:.0f}"),
                        t("ABC ordena por volumen (A: 80% del volumen, B: el siguiente 15%, C: el resto). XYZ mide qué tan estable es la demanda mes a mes (X estable, Y variable, Z errática). El margen por producto es una aproximación: toneladas por el margen de su calibre.")))
    cnt = df["sugerencia"].value_counts()
    st.html(ui.kpis([(str(int(cnt.get(k, 0))), t(k)) for k in ("Impulsar", "Revisar", "Considerar descontinuar", "Mantener")]))
    # matriz ABC-XYZ
    st.html(ui.h2(t("Matriz ABC–XYZ"), t("Número de productos y porcentaje del volumen de 12 meses en cada cruce.")))
    tv = df["vol12"].sum()
    rows = []
    for a in "ABC":
        row = [Raw(f"<b>{a}</b>")]
        for x in "XYZ":
            d = df[(df["abc"] == a) & (df["xyz"] == x)]
            row.append(f"{len(d)} · {d['vol12'].sum() / tv * 100:.0f}%")
        rows.append(row)
    st.html(ui.table([t("Clase"), t("X estable"), t("Y variable"), t("Z errática")], rows, num=(1, 2, 3)))
    # lista
    st.html(ui.h2(t("Productos y sugerencias"), t("Las sugerencias son reglas simples para orientar la revisión, no una decisión. Revisa cada producto antes de actuar.")))
    sel = st.pills(t("Mostrar"), ["Impulsar", "Revisar", "Considerar descontinuar", "Mantener"], selection_mode="multi", default=["Impulsar", "Revisar", "Considerar descontinuar"], key="chips_sug", format_func=t) or []
    d = df[df["sugerencia"].isin(sel)] if sel else df
    rows = [[Raw(f"<b>{esc(r.producto)}</b>"), ui.cal_label(r.calibre), f"{r.vol12:,.1f} {ui.UNIT}", f"{r.vol12 / tv * 100:.1f}%", ui.mxn(r.margen12) if np.isfinite(r.margen12) else "—",
             f"{r.crec:+.0f}%" if np.isfinite(r.crec) else "—", f"{r.error:.0f}%" if np.isfinite(r.error) else "—", ui.dclass_html(r.clase), f"{r.abc}{r.xyz}",
             ui.badge(t(r.sugerencia), SUG_BADGE.get(r.sugerencia, "")), _motivo_t(r.motivo)] for r in d.head(80).itertuples()]
    st.html(ui.table([t("Producto"), t("Calibre"), t("Volumen 12 m"), t("Del total"), t("Margen 12 m"), t("Crecimiento"), t("Error de referencia"), t("Demanda"), t("ABC–XYZ"), t("Sugerencia"), t("Motivo")], rows, num=(2, 3, 4, 5, 6), min_w=1100)
            if rows else f'<p class="hint">{esc(t("No hay productos con esa sugerencia."))}</p>')
    st.html(f'<p class="syncnote">{esc(t("Crecimiento: últimos 12 meses contra los 12 previos. Error de referencia: error de un promedio móvil de 3 meses sobre los últimos meses, como medida de qué tan difícil es pronosticar el producto. Se muestran hasta 80 productos."))}</p>')
    if adv():
        sensitivity_block(m)


def sensitivity_block(m):
    st.html(ui.h2(t("Sensibilidad del volumen al precio del metal"), t("Cómo se movió el volumen cuando se movió el precio del cobre o del aluminio, con el histórico de precios guardado. Compara cada mes contra el mismo mes del año anterior.")))
    hist = price_history()
    if hist is None or hist.empty:
        st.html(f'<p class="hint">{esc(t("No hay histórico de precios del metal. Cárgalo en Metales o activa el histórico automático."))}</p>')
        return
    sn = cached_sens(m, hist)
    if sn.empty:
        st.html(f'<p class="hint">{esc(t("El histórico de precios y el de ventas se traslapan muy poco para estimar la relación."))}</p>')
        return
    keep = sn[((sn["grupo"] == "Total") & (sn["metal"] == "Cobre")) | ((sn["grupo"] == "Cobre") & (sn["metal"] == "Cobre")) | ((sn["grupo"] == "Aluminio") & (sn["metal"] == "Aluminio"))]
    clear = keep[(keep["t"].abs() >= 2) & (keep["confianza"] != "Baja")]
    if clear.empty:
        st.html(f'<p class="headline" style="font-size:clamp(18px,2vw,24px);max-width:62ch">{esc(t("No se ve una relación confiable entre el precio del metal y el volumen."))}</p>')
    else:
        r = clear.iloc[clear["t"].abs().argmax()]
        st.html(f'<p class="headline" style="font-size:clamp(18px,2vw,24px);max-width:62ch">{esc(t("Si el precio de {m} sube 10%, el volumen de {g} cambia {x}% (rezago de {l} meses).", m=t(r["metal"]).lower(), g=t(r["grupo"]).lower(), x=f"{r.elasticidad * 10:+.1f}", l=int(r.rezago)))}</p>')
    rows = [[t(r.grupo), t(r.metal), f"{int(r.rezago)}", f"{r.elasticidad:+.2f}", f"±{1.96 * r.error_est:.2f}", f"{int(r.n)}", t(r.confianza), t(r.lectura)] for r in keep.itertuples()]
    st.html(ui.table([t("Volumen de"), t("Precio de"), t("Rezago (meses)"), t("Elasticidad"), t("Margen de error (95%)"), t("Puntos"), t("Confianza"), t("Lectura")], rows, num=(2, 3, 4, 5), min_w=860))
    n_min = int(keep["n"].min())
    st.html(f'<p class="syncnote">{esc(t("Elasticidad: % de cambio del volumen por cada 1% de cambio del precio del metal. Con solo {n} puntos comparables y precios que suben junto con las ventas, esta relación no prueba causa y efecto. Úsala como una pista, no como una regla; el precio de lista propio no está en los datos.", n=n_min))}</p>')


# ---- cascada: por qué cambió el pronóstico
def latest_frozen():
    fr = CRM.frozen_list()
    return fr[0] if fr else None


FACTORS = [("atipicos", "Meses atípicos corregidos", "Efecto de las correcciones de meses atípicos que decidiste aplicar."),
           ("senales", "Señales de mercado", "Vivienda, construcción, tipo de cambio y demás señales activas (sin el cobre)."),
           ("metales", "Precio de los metales", "Efecto del precio del cobre (en vivo y del camino de Metales) sobre la demanda."),
           ("arancel", "Arancel", "Escenario arancelario sobre la parte de exportación."),
           ("opps", "Oportunidades", "Toneladas ponderadas de las oportunidades abiertas y ganadas del CRM."),
           ("consenso", "Ajustes de consenso", "Ajustes manuales con justificación en Pronóstico.")]


def waterfall_raw(h: int, only=None) -> dict:
    """Pronóstico base mes a mes y la aportación mensual de cada factor, aplicados en orden. Es una descomposición secuencial: cada efecto se mide con los anteriores ya aplicados."""
    prices, w, on = get_prices(), get_w(), apply_signals_on()
    base_m, cur_m = monthly_edited(), default_prep().monthly
    gauges = sorted(set(cur_m["cal"]) if not only else set(only), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 0))
    units = tuple((g,) for g in gauges)
    sig, mult, crm = get_sig(), get_mult(), get_crm(None, E.TON, prices)
    sig_nm = {**sig, "Precio del cobre": {**sig["Precio del cobre"], "chg": 0.0}}

    def proj(monthly, sg, mu, cr, ap):
        return cached_project(monthly, units, None, h, E.AUTO, E.TON, "sum", mu, sg, w, prices, get_ml(monthly, E.windows_for(monthly)), cr, ap)
    p = [proj(base_m, sig, 1.0, {}, False), proj(cur_m, sig, 1.0, {}, False), proj(cur_m, sig_nm, 1.0, {}, on), proj(cur_m, sig, 1.0, {}, on),
         proj(cur_m, sig, mult, {}, on), proj(cur_m, sig, mult, crm, on)]
    cons_d = np.zeros(h)
    for d, g in zip(p[-1].detail, gauges):
        c = store().get("consensus", {}).get(f"{g}|None|{E.TON}|{h}|{E.AUTO}")
        if c:
            cons_d += E.consensus(d["fc"], c["adj"], c["why"])[0] - d["fc"]
    arr = [np.asarray(x.fc, float) for x in p]
    f = {"atipicos": arr[1] - arr[0], "senales": arr[2] - arr[1], "metales": arr[3] - arr[2], "arancel": arr[4] - arr[3], "opps": arr[5] - arr[4], "consenso": cons_d}
    return {"months": [f"{x:%Y-%m}" for x in p[-1].index], "base": arr[0], "factors": f, "final": arr[5] + cons_d}


def waterfall_data(h: int, only=None) -> dict:
    raw = waterfall_raw(h, only)
    months, fz = raw["months"], latest_frozen()
    mask = np.ones(h, bool)
    steps, start, start_label, frozen = [], float(raw["base"].sum()), t("Pronóstico base"), False
    S = lambda x: float(np.asarray(x)[mask].sum())
    if fz:
        fm = {g: v for g, v in fz["payload"]["gauges"].items() if not only or g in only}
        fmonths = next(iter(fm.values()))["months"] if fm else []
        mask = np.array([x in fmonths for x in months])
        if mask.any():
            frozen = True
            fcons = np.zeros(h)
            for gd in fm.values():
                for j, mo in enumerate(gd["months"]):
                    if mo in months:
                        fcons[months.index(mo)] += gd["cons"][j]
            start_label, start = t("Congelado «{n}»", n=fz["name"]), float(fcons[mask].sum())
            fzf = fz["payload"].get("factors") if not only else None
            end = S(raw["final"])
            if fzf and fzf.get("months"):
                fidx = {mo: j for j, mo in enumerate(fzf["months"])}
                deltas = []
                for key, lab, why in FACTORS:
                    old = np.array([fzf["f"][key][fidx[mo]] if mo in fidx else 0.0 for mo in months])
                    deltas.append((lab, S(raw["factors"][key] - old), why))
                steps.append((t("Datos nuevos y reentrenamiento"), end - start - sum(d[1] for d in deltas), t("Ventas nuevas y reajuste del modelo: lo que queda después de los cambios en los factores.")))
                steps += [(t(a), v, t(b)) for a, v, b in deltas]
            else:
                steps.append((t("Todos los cambios desde el congelado"), end - start, t("Este pronóstico se congeló sin el detalle de factores, así que no se puede separar.")))
    if not frozen:
        mask = np.ones(h, bool)
        start = float(raw["base"].sum())
        steps = [(t(lab), S(raw["factors"][key]), t(why)) for key, lab, why in FACTORS]
    return {"start_label": start_label, "start": start, "steps": steps, "end": start + sum(s[1] for s in steps), "frozen": frozen, "n_months": int(mask.sum()), "h": h}


def waterfall_block(h: int):
    st.html(ui.h2(t("Por qué cambió el pronóstico"), t("Parte del pronóstico base (o del último congelado) y suma el efecto de cada factor hasta llegar al pronóstico actual. Es una descomposición en orden: cada efecto se mide con los anteriores ya aplicados.")))
    wd = waterfall_data(h)
    labels = [wd["start_label"]] + [s[0] for s in wd["steps"]] + [t("Pronóstico actual")]
    st.plotly_chart(ui.waterfall_fig(labels, [s[1] for s in wd["steps"]], wd["start"], height=360), width="stretch", theme=None, config={"displayModeBar": False})
    big = [s for s in wd["steps"] if abs(s[1]) >= 0.5]
    if big:
        top = max(big, key=lambda s: abs(s[1]))
        st.html(f'<p class="hint">{esc(t("El mayor movimiento es {f}: {v} t.", f=top[0].lower(), v=f"{top[1]:+,.0f}"))}</p>')
    else:
        st.html(f'<p class="hint">{esc(t("Ningún factor mueve el pronóstico por ahora: es el pronóstico estadístico base."))}</p>')
    st.html(ui.table([t("Factor"), t("Aporta (t)"), t("Qué es")], [[Raw(f"<b>{esc(s[0])}</b>"), f"{s[1]:+,.1f}", s[2]] for s in wd["steps"]], num=(1,), min_w=620))
    st.html(f'<p class="syncnote">{esc(t("Compara contra el pronóstico congelado más reciente ({n} meses en común). Congela uno en Seguimiento del pronóstico para ver qué cambió desde entonces.", n=wd["n_months"]) if wd["frozen"] else t("Todavía no hay un pronóstico congelado, así que se compara contra el pronóstico estadístico base. Congela uno en Seguimiento del pronóstico para ver qué cambió desde entonces."))}</p>')


# ---- resumen del mes
def buy_reco_lines(res, horizon: int) -> list[str]:
    path = metal_paths()
    k = 1 + sv("pl_mchg", 0) / 100
    fx0, hold, ahead = float(sv("pl_fx", 18.5)), sv("pl_hold", 25), sv("mt_ahead", 6)
    out = []
    for key, nm, now in (("cu", t("Cobre"), float(sv("pl_cu", 9840.0))), ("al", t("Aluminio"), float(sv("pl_al", 2600.0)))):
        req = np.zeros(horizon)
        for r in res.rows.itertuples():
            if r.metal == key:
                req = req + np.asarray(r.metal_req)[:horizon]
        if req.sum() <= 0:
            continue
        pl = MT.purchase_plan(req, now, [x * k for x in path[key]], fx0, path["fx"], hold, ahead)
        if pl["buy_now"] > 1e-6:
            out.append(t("{n}: comprar hoy {x} t; ahorro estimado {s} contra comprar mes a mes.", n=nm, x=f"{pl['buy_now']:,.0f}", s=ui.mxn(pl["saving"])))
        else:
            out.append(t("{n}: comprar mes a mes; el camino de precios no justifica adelantar.", n=nm))
    return out


def build_summary(m, pr, res, alerts, h: int) -> list[dict]:
    """Bloques del resumen del mes (texto corto)."""
    B = []
    tot = PL.totals(res)
    B.append({"title": t("Pronóstico"), "lines": [
        t("Esperamos {v} t en los próximos {h} meses (rango {a} a {b}).", v=f"{pr.fc.sum():,.0f}", h=h, a=f"{pr.lo.sum():,.0f}", b=f"{pr.hi.sum():,.0f}"),
        t("Precisión de validación: {p}% (error {e}%, medido contra meses reales con validación cruzada).", p=f"{max(0, 100 - pr.error):.1f}", e=f"{pr.error:.1f}"),
        t("Ingreso estimado {r}; margen bruto {g} ({p}%).", r=ui.mxn(tot["rev"]), g=ui.mxn(tot["margin"]), p=f"{tot['margin_pct']:.1f}")]})
    wd = waterfall_data(h)
    ch = [t("{f}: {v} t", f=s[0], v=f"{s[1]:+,.0f}") for s in sorted(wd["steps"], key=lambda s: -abs(s[1])) if abs(s[1]) >= 0.5][:4]
    B.append({"title": t("Qué cambió y por qué"), "lines": [t("Se parte de: {a} ({v} t).", a=wd["start_label"], v=f"{wd['start']:,.0f}")] + (ch or [t("Ningún factor mueve el pronóstico.")])})
    top = sorted(alerts, key=lambda x: 0 if x["sev"] == "Alta" else 1)[:5]
    B.append({"title": t("Qué atender"), "lines": [x["text"] for x in top] or [t("Sin alertas por ahora.")]})
    an = pending_anomalies(m) if adv() else pd.DataFrame()
    if adv():
        B.append({"title": t("Anomalías por revisar"), "lines": ([t("{n} anomalías esperan revisión; las de mayor riesgo:", n=len(an))] + [_anom_text(r) for _, r in an.sort_values("at_risk", ascending=False).head(3).iterrows()]) if len(an) else [t("Sin anomalías pendientes de revisar.")]})
    snap = live_snapshot()
    ml_ = []
    for s, nm in (("cu", t("Cobre")), ("al", t("Aluminio"))):
        q = snap["q"].get(s)
        if q is not None and np.isfinite(q.value):
            ml_.append(f"{nm}: ${q.value:,.0f} USD/t" + (f" ({q.day_pct:+.1f}% {t('hoy')})" if q.day_pct is not None else ""))
    fxq = snap["q"].get("fx")
    if fxq is not None and np.isfinite(fxq.value):
        ml_.append(f"USD/MXN: {fxq.value:.2f}")
    B.append({"title": t("Metales y compra"), "lines": [" · ".join(ml_) if ml_ else t("Sin precios disponibles.")] + buy_reco_lines(res, h) + _safe_market_texts()[:2]})
    nw = latest_news(None, 3)
    B.append({"title": t("Noticias y regulación relevantes"), "lines": [f"{r['published'][:10]} · {r['source']} · {r['title']}" for r in nw] or [t("Todavía no hay noticias guardadas.")]})
    ops = sorted([o for o in CRM.opps() if o["stage"] in CRM.OPEN_STAGES], key=lambda o: -o["tons_month"] * o["months"] * CRM.PROB[o["stage"]])[:5]
    cu = {c["id"]: c["name"] for c in CRM.customers()}
    tb = {"headers": [t("Oportunidad"), t("Cliente"), t("Etapa"), t("Toneladas"), t("Ponderadas")],
          "rows": [[o["name"], cu.get(o["customer_id"], "—"), t(o["stage"]), round(o["tons_month"] * o["months"], 1), round(o["tons_month"] * o["months"] * CRM.PROB[o["stage"]] / 100, 1)] for o in ops]}
    B.append({"title": t("Principales oportunidades"), "lines": [t("{a} · {b} t en total, {c} t ponderadas", a=f"{o['name']} ({cu.get(o['customer_id'], '—')}, {t(o['stage'])})", b=f"{o['tons_month'] * o['months']:,.0f}", c=f"{o['tons_month'] * o['months'] * CRM.PROB[o['stage']] / 100:,.0f}") for o in ops] or [t("Sin oportunidades abiertas.")], "table": tb if ops else None})
    return B


@st.cache_data(show_spinner=False)
def cached_report(monthly, h, prices, params, mult, sig, w, crm, opps, notes, unit, apply, lang, _ml=None) -> bytes:
    from viakable_forecast import report as RP
    return RP.to_excel(RP.build(monthly, h, prices, PL.PlanParams(**params), mult, sig, w, _ml, crm, opps, notes, unit, apply))


def summary_panel(m, pr, res, alerts, h: int, ml=None):
    """Resumen de una página en pantalla y descargas en PDF y Excel."""
    from viakable_forecast import summary as SUMM
    B = build_summary(m, pr, res, alerts, h)
    title = t("Resumen del mes")
    sub = f"{APP_NAME} · {ui.mes(pd.Timestamp(dt.date.today()), True)} · " + t("datos hasta {d}", d=ui.mes(m["ym"].max(), True))
    with st.container(key="summary"):
        st.html(ui.h2(title, sub))
        for b in B:
            st.html(f'<h3 style="margin:14px 0 4px;font-size:1.02em">{esc(b["title"])}</h3>' + "".join(f'<p class="hint" style="margin:2px 0">{esc(x)}</p>' for x in b["lines"]))
        st.html(f'<p class="syncnote">{esc(t(FOOTER))}</p>')
    c = st.columns([1, 1, 1.4, 3])
    c[0].download_button(t("Descargar PDF"), SUMM.to_pdf(title, sub, B, t(FOOTER)), file_name=f"resumen-{dt.date.today():%Y-%m}.pdf", mime="application/pdf", key="dl_sum_pdf", width="stretch")
    c[1].download_button(t("Descargar Excel"), SUMM.to_xlsx(title, sub, B, t(FOOTER)), file_name=f"resumen-{dt.date.today():%Y-%m}.xlsx",
                         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_sum_xlsx", width="stretch")
    prices = get_prices()
    rows = [o for o in CRM.opps() if not o["sample"] or store().get("crm_use_sample", False)]
    c[2].download_button(t("Descargar reporte completo (Excel)"), cached_report(m, h, prices, get_params(), get_mult(), get_sig(), get_w(), get_crm(None, E.TON, prices), rows, list(default_prep().notes), ui.UNIT,
                                                                           apply_signals_on(), ST.cfg()["lang"], ml), file_name=f"reporte-calibra-{dt.date.today():%Y-%m}.xlsx",
                         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_full_xlsx", width="stretch")


# ================================================================== modo presentación
PRES_SECS = 15                      # 8 pasos x 15 s: unos 2 minutos
PRES_STEPS = [
    ("Pronóstico actual", "Este es el pronóstico de los próximos 6 meses con los datos de hoy."),
    ("Precio del cobre en vivo", "El cobre se mueve cada día. Aquí se simula una alza de 10%: el margen y el pronóstico se recalculan al instante."),
    ("Una noticia regulatoria", "La app vigila aranceles y normas. Esta es la noticia más relevante guardada; tú decides qué hacer con ella."),
    ("Escenario: exención arancelaria del T-MEC", "Qué pasaría si las exportaciones a EE. UU. quedaran exentas de arancel."),
    ("Capital liberado", "Un pronóstico más preciso necesita menos inventario de seguridad, y eso libera capital de trabajo."),
    ("Riesgo y cobertura", "El margen no es un solo número: se simulan demanda, metales y arancel para ver qué tan malo puede ser el peor 5% y cuánto lo reduce fijar parte de las compras."),
    ("Anomalías y alertas", "Lo que necesita atención hoy, ordenado por urgencia."),
    ("Por qué cambió el pronóstico", "Cada factor suma o resta al pronóstico base. Aquí se ve el efecto de los dos escenarios."),
]
PRES_KEYS = ("pl_mchg", "t_sce")


def pres_put(key, val):
    store()[key] = val
    if key in st.session_state:
        st.session_state[key] = val


def pres_scenario(step: int, base: dict):
    """Estado del escenario para cada paso: cobre +10% desde el paso 2 y arancel en 0% desde el paso 4."""
    pres_put("pl_mchg", 10 if step >= 1 else base["pl_mchg"])
    pres_put("t_sce", 0 if step >= 3 else base["t_sce"])


def pres_start():
    base = {"pl_mchg": store().get("pl_mchg", 0), "t_sce": store().get("t_sce", store().get("t_cur", TARIFF_CURRENT)), "nav": st.session_state.get("nav", "Inicio")}
    st.session_state["present"] = {"step": 0, "t0": time.time(), "paused": False, "base": base}
    st.session_state["drawer"] = False
    pres_scenario(0, base)


def pres_exit():
    p = st.session_state.pop("present", None)
    if p:
        pres_scenario(-1, p["base"])
        st.session_state["nav"] = p["base"].get("nav", "Inicio")


def pres_go(delta: int):
    p = st.session_state.get("present")
    if not p:
        return
    s = p["step"] + delta
    if s >= len(PRES_STEPS):
        pres_exit()
        return
    p["step"], p["t0"] = max(0, s), time.time()
    pres_scenario(p["step"], p["base"])


def pres_toggle_pause():
    p = st.session_state["present"]
    p["paused"] = not p["paused"]
    p["t0"] = time.time()


@st.fragment(run_every=2)
def pres_clock():
    """Avanza solo cada PRES_SECS segundos mientras no esté en pausa."""
    p = st.session_state.get("present")
    if p and not p["paused"] and time.time() - p["t0"] >= PRES_SECS - 0.5:
        pres_go(1)
        st.rerun()


def pres_data(h: int = 6) -> dict:
    """Pronóstico y plan actuales (con el escenario del paso) y los de la base, para mostrar las diferencias."""
    prep = default_prep()
    m = prep.monthly
    ml = get_ml(m, E.windows_for(m))
    gauges = tuple(prep.gauges)
    ov = {"metal": sv("ov_metal", {}), "inv": sv("ov_inv", {}), "cap": sv("ov_cap", {})}
    out = {"m": m, "gauges": gauges, "h": h}
    for tag, s in (("", store()), ("_b", neutral(store()))):
        pr_ = get_prices(s)
        out["pr" + tag] = cached_project(m, tuple((g,) for g in gauges), None, h, E.AUTO, E.TON, "sum", get_mult(s), get_sig(s), get_w(s), pr_, ml, get_crm(s, E.TON, pr_), apply_signals_on(s))
        out["res" + tag] = cached_plan(m, gauges, h, get_params(s), pr_, get_mult(s), get_sig(s), get_w(s), ov, ml, get_crm(s, E.TON, pr_))
        out["tot" + tag] = PL.totals(out["res" + tag])
    return out


def pres_stage():
    p = st.session_state["present"]
    i = p["step"]
    d = pres_data()
    pr, pr_b, tot, tot_b, res = d["pr"], d["pr_b"], d["tot"], d["tot_b"], d["res"]
    title, cap = PRES_STEPS[i]
    snap = live_snapshot()
    q = snap["q"].get("cu")
    dfc, dmar = float(pr.fc.sum() - pr_b.fc.sum()), tot["margin"] - tot_b["margin"]
    # texto corto de cada paso
    if i == 0:
        text = t("Esperamos {v} {u} en los próximos {h} meses, con {p}% de precisión de validación.", v=f"{pr.fc.sum():,.0f}", u=ui.UNIT, h=d["h"], p=f"{max(0, 100 - pr.error):.1f}")
    elif i == 1:
        px = f"${q.value:,.0f} USD/t" if q is not None and np.isfinite(q.value) else "—"
        day = f" ({q.day_pct:+.1f}% {t('hoy')})" if q is not None and q.day_pct is not None else ""
        text = t("El cobre está en {x}{d}. Con +10%, el margen cambia {m} y el pronóstico {f} {u}.", x=px, d=day, m=ui.mxn(dmar), f=f"{dfc:+,.0f}", u=ui.UNIT)
    elif i == 2:
        text = ""
    elif i == 3:
        ex = tot["exp"]
        text = (t("Con arancel de 0%, la exportación pasa de {a} a {b} {u} y el margen cambia {m}.", a=f"{tot_b['exp']:,.0f}", b=f"{ex:,.0f}", u=ui.UNIT, m=ui.mxn(tot["margin"] - tot_b["margin"])) if (ex or tot_b["exp"])
                else t("Tus datos no traen ventas de exportación definidas, así que el arancel no mueve nada. Define la regla en Datos para ver este efecto."))
    elif i == 4:
        text = t("El inventario de seguridad libera {x} de capital y ahorra {y} al año en costo de mantener.", x=ui.mxn(tot["freed"]), y=ui.mxn(tot["saving"]))
    elif i == 5:
        rs = risk_summary(d["h"], 5000)
        text = (t("Margen esperado {m}; en el peor 5% queda en {p} y hay {q}% de probabilidad de perder dinero. Fijar {x}% de las compras lo deja en {y}.", m=ui.mxn(rs["s"]["mean"]), p=ui.mxn(rs["s"]["p5"]), q=f"{rs['s']['prob_below'] * 100:.0f}", x=f"{rs['frac'] * 100:.0f}", y=ui.mxn(rs["hedged"]["p5"]))
                if rs else "")
    elif i == 6:
        text = ""
    else:
        text = t("De {a} a {b} {u} con los dos escenarios.", a=f"{pr_b.fc.sum():,.0f}", b=f"{pr.fc.sum():,.0f}", u=ui.UNIT)
    secs = 0 if (p["paused"] or not ui.ANIM) else PRES_SECS
    with st.container(key="pres"):
        st.html(f'<div class="pres-card"><div class="pres-top"><span class="tour-step">{esc(t("Paso {a} de {b}", a=i + 1, b=len(PRES_STEPS)))}</span>'
                f'<span class="pres-bar"><i style="animation-duration:{secs}s" class="{"run" if secs else ""}"></i></span></div>'
                f'<h3>{esc(t(title))}</h3><p>{esc(t(cap))}</p>' + (f'<p class="pres-big">{esc(text)}</p>' if text else "") + "</div>")
        c = st.columns([1, 1, 1, 1, 3])
        c[0].button(t("Anterior"), key="pres_prev", on_click=pres_go, args=(-1,), disabled=i == 0)
        c[1].button(t("Siguiente") if i < len(PRES_STEPS) - 1 else t("Terminar"), key="pres_next", type="primary", on_click=pres_go, args=(1,))
        c[2].button(t("Continuar") if p["paused"] else t("Pausar"), key="pres_pause", on_click=pres_toggle_pause)
        c[3].button(t("Salir"), key="pres_exit", on_click=pres_exit)
    # los mismos cuatro indicadores en todos los pasos: cuentan hasta su nuevo valor
    st.html(ui.kpis([(f"{pr.fc.sum():,.0f} {ui.UNIT}", t("Pronóstico {h} meses", h=d["h"]), t("{x} contra la base", x=f"{dfc:+,.0f}")),
                     (f"{max(0, 100 - pr.error):.1f}%", t("Precisión de validación")),
                     (ui.mxn(tot["margin"]), t("Margen bruto"), t("{x} contra la base", x=ui.mxn(dmar))),
                     (ui.mxn(tot["freed"]), t("Capital liberado"), f"{tot['freed'] / tot['ss_value'] * 100:.0f}% " + t("del inventario de seguridad") if tot["ss_value"] else "")]))
    if i == 4:
        rows = res.rows.groupby("cal")["freed"].sum()
        st.plotly_chart(ui.bar_fig([ui.cal_label(g) for g in rows.index], {t("Capital liberado (MXN)"): [float(v) for v in rows.values]}, {t("Capital liberado (MXN)"): ui.RED}, height=330, unit="MXN"),
                        width="stretch", theme=None, config={"displayModeBar": False})
    elif i == 5 and rs:
        st.plotly_chart(ui.hist_fig(rs["o"]["margin"] / 1e6, [(rs["s"]["p5"] / 1e6, "P5", ui.FG), (rs["s"]["mean"] / 1e6, t("Esperado"), ui.RED)], height=330), width="stretch", theme=None, config={"displayModeBar": False})
    elif i == 7:
        wd = waterfall_data(d["h"])
        labels = [wd["start_label"]] + [s[0] for s in wd["steps"]] + [t("Pronóstico actual")]
        st.plotly_chart(ui.waterfall_fig(labels, [s[1] for s in wd["steps"]], wd["start"], height=360), width="stretch", theme=None, config={"displayModeBar": False})
    else:
        st.plotly_chart(ui.forecast_fig(list(pr.hist.index), pr.hist.values, list(pr.index), pr.fc, pr.lo, pr.hi, ui.UNIT, t("Suma de calibres"), height=330), width="stretch", theme=None, config={"displayModeBar": False})
    if i == 2:
        rows = latest_news(None, 1, days=90)
        st.html(news_card_html(rows[0]) + (f'<div class="alertbar"><b>{esc(t("Qué hacer"))}:</b> {esc(t(rows[0]["action"]["text"]))}</div>' if rows[0]["action"] else "") if rows
                else f'<p class="hint">{esc(t("Todavía no hay noticias guardadas."))}</p>')
    if i == 6:
        al = business_alerts(d["m"], pr, res, CRM.frozen_list()) + [{"sev": "Alta", "text": x, "action": t("Revisar Metales."), "section": "Metales"} for x in _safe_market_texts()]
        n_an = len(pending_anomalies(d["m"]))
        st.html("".join(f'<div class="alertbar{" soft" if x["sev"] != "Alta" else ""}">{esc(x["text"])}<br><b>{esc(t("Qué hacer"))}:</b> {esc(x["action"])}</div>' for x in sorted(al, key=lambda x: 0 if x["sev"] == "Alta" else 1)[:4])
                + f'<p class="syncnote">{esc(t("{n} anomalías del ritmo de pedidos esperan revisión.", n=n_an))}</p>')


# ================================================================== 8B: asistente Calibra (herramientas)
ASST = "asst"








def _tool_sel(a: dict):
    prep = default_prep()
    allg = list(prep.gauges)
    g = [str(x) for x in (a.get("gauges") or allg)]
    bad = [x for x in g if x not in allg]
    if bad:
        raise ValueError(f"Calibres no válidos: {bad}. Los disponibles son {allg}.")
    mk = a.get("market") or None
    if mk not in (None, MKT_NAC, MKT_USA):
        raise ValueError(f"Mercado no válido. Usa '{MKT_NAC}' o '{MKT_USA}'.")
    h = int(a.get("horizon") or 6)
    if h not in (3, 6, 12):
        raise ValueError("El horizonte debe ser 3, 6 o 12 meses.")
    return prep, g, mk, h


def _r(x, n=1):
    return None if x is None or not np.isfinite(float(x)) else round(float(x), n)


def tool_eval(s, g, h):
    """Pronóstico y plan de una selección con el estado `s` (el actual o uno simulado)."""
    prep = default_prep()
    m = prep.monthly
    ml = get_ml(m, E.windows_for(m))
    prices = get_prices(s)
    crm = get_crm(s, E.TON, prices)
    pr = cached_project(m, tuple((x,) for x in g), None, h, E.AUTO, E.TON, "sum", get_mult(s), get_sig(s), get_w(s), prices, ml, crm, apply_signals_on(s))
    res = cached_plan(m, tuple(g), h, get_params(s), prices, get_mult(s), get_sig(s), get_w(s), {"metal": sv("ov_metal", {}), "inv": sv("ov_inv", {}), "cap": sv("ov_cap", {})}, ml, crm)
    return pr, PL.totals(res), res


def tool_forecast(a):
    prep, g, mk, h = _tool_sel(a)
    measure = E.MXN if a.get("measure") == "pesos" else E.TON
    m = prep.monthly
    ml = get_ml(m, E.windows_for(m))
    prices = get_prices()
    pr = cached_project(m, tuple((x,) for x in g), mk, h, E.AUTO, measure, "sum", get_mult(), get_sig(), get_w(), prices, ml, get_crm(None, measure, prices), apply_signals_on())
    unit = "MXN" if measure == E.MXN else ui.UNIT
    rows = [{"gauge": r["Calibre"], "forecast_total": _r(r["Pronóstico"], 0), "method": r["Método"], "accuracy_pct": _r(r["Precisión %"]), "demand_type": r["Demanda"],
             "change_vs_same_months_last_year_pct": _r(r["Cambio %"])} for _, r in pr.table.iterrows()]
    res = {"unit": unit, "horizon_months": h, "market": mk or "todos", "gauges": g, "months": [f"{x:%Y-%m}" for x in pr.index], "forecast": [_r(v, 0) for v in pr.fc],
           "low_80": [_r(v, 0) for v in pr.lo], "high_80": [_r(v, 0) for v in pr.hi], "total": _r(pr.fc.sum(), 0), "method": pr.method, "validation_accuracy_pct": _r(max(0, 100 - pr.error)),
           "by_gauge": rows, "note": "La precisión es 100 menos el error porcentual de validación cruzada."}
    fig = ui.forecast_fig(list(pr.hist.index), pr.hist.values, list(pr.index), pr.fc, pr.lo, pr.hi, unit, t("Suma de calibres"), height=300, money=measure == E.MXN)
    tb = {"headers": [t("Calibre"), t("Método"), t("Precisión %"), f"{t('Pronóstico')} ({unit})", t("Cambio %")],
          "rows": [[ui.cal_label(r["gauge"]) if len(rows) > 1 or "+" not in str(r["gauge"]) else r["gauge"], t(r["method"]), r["accuracy_pct"], f"{r['forecast_total']:,.0f}", r["change_vs_same_months_last_year_pct"]] for r in rows]}
    return res, {"type": "forecast", "title": t("Pronóstico {h} meses", h=h), "fig": fig, "table": tb}


def tool_whatif_changes(a: dict):
    clip = lambda v, lo, hi: float(min(max(float(v), lo), hi))
    ch, desc = {}, []
    if a.get("metal_pct") is not None:
        v = clip(a["metal_pct"], -30, 30)
        ch["pl_mchg"] = int(round(v))
        desc.append(t("cobre y aluminio {x}%", x=f"{ch['pl_mchg']:+d}"))
    pct = False
    for k, key, lab, lim in (("copper_pct", "mt_cu_chg", "cobre", 40), ("aluminum_pct", "mt_al_chg", "aluminio", 40), ("fx_pct", "mt_fx_chg", "tipo de cambio", 20)):
        if a.get(k) is not None:
            ch[key] = int(round(clip(a[k], -lim, lim)))
            desc.append(f"{t(lab)} {ch[key]:+d}%")
            pct = True
    if pct:
        ch["mt_kind"], ch["seg_mt_shape"] = "pct", "inmediato"
    if a.get("tariff_pct") is not None:
        ch["t_sce"] = int(5 * round(clip(a["tariff_pct"], 0, 60) / 5))
        desc.append(t("arancel {a}% (vigente {b}%)", a=ch["t_sce"], b=f"{float(sv('t_cur', TARIFF_CURRENT)):.0f}"))
    if a.get("pass_through_pct") is not None:
        ch["mt_pass"] = int(5 * round(clip(a["pass_through_pct"], 0, 100) / 5))
        desc.append(t("traspaso del metal {x}%", x=ch["mt_pass"]))
    eo = a.get("extra_order")
    if eo:
        gs = list(default_prep().gauges)
        if str(eo.get("gauge")) not in gs:
            raise ValueError(f"Calibre del pedido no válido. Disponibles: {gs}.")
        ch["wi_opp_cal"], ch["wi_opp_t"], ch["wi_opp_m"] = str(eo["gauge"]), float(max(0, eo["tons_per_month"])), int(min(max(int(eo["months"]), 1), 12))
        desc.append(t("pedido adicional de {x} {u} por mes en calibre {g} durante {m} meses", x=f"{ch['wi_opp_t']:,.0f}", u=ui.UNIT, g=eo["gauge"], m=ch["wi_opp_m"]))
    for name, v in (a.get("signals") or {}).items():
        if name not in SIGNALS:
            raise ValueError(f"Señal desconocida '{name}'. Opciones: {list(SIGNALS)}.")
        ch[f"chg_{name}"] = int(round(clip(v, -30, 30)))
        desc.append(f"{t(name)} {ch[f'chg_{name}']:+d}%")
    if not ch:
        raise ValueError("Indica al menos un cambio para simular.")
    return ch, desc


def tool_whatif(a):
    h = int(a.get("horizon") or 6)
    if h not in (3, 6, 12):
        raise ValueError("El horizonte debe ser 3, 6 o 12 meses.")
    ch, desc = tool_whatif_changes(a)
    g = list(default_prep().gauges)
    s0 = copy.deepcopy(dict(store()))
    s0.pop("mt_path", None)
    s1 = copy.deepcopy(s0)
    s1.update(ch)
    pr0, t0, _ = tool_eval(s0, g, h)
    pr1, t1, _ = tool_eval(s1, g, h)
    pid = hashlib.md5(json.dumps(ch, sort_keys=True).encode()).hexdigest()[:8]
    asst_state()["proposals"][pid] = {"changes": ch, "desc": desc, "applied": False}
    items = [("forecast", t("Pronóstico {h} meses", h=h) + f" ({ui.UNIT})", pr0.fc.sum(), pr1.fc.sum(), 0), ("revenue", t("Ingreso (MXN)"), t0["rev"], t1["rev"], 0), ("margin", t("Margen bruto (MXN)"), t0["margin"], t1["margin"], 0),
             ("margin_pct", t("Margen bruto %"), t0["margin_pct"], t1["margin_pct"], 1), ("metal_cost", t("Costo del metal (MXN)"), t0["metal_cost"], t1["metal_cost"], 0),
             ("safety_stock_value", t("Valor del inventario de seguridad"), t0["ss_value"], t1["ss_value"], 0), ("capital_freed", t("Capital liberado"), t0["freed"], t1["freed"], 0)]
    res = {"proposal_id": pid, "applied": False, "changes": desc, "horizon_months": h, "comparison": {k: {"current": _r(a0, n), "scenario": _r(b0, n), "difference": _r(b0 - a0, n)} for k, _, a0, b0, n in items},
           "note": "Escenario simulado, NO aplicado. La persona debe pulsar Aplicar escenario. Los precios de lista y costos son ilustrativos."}
    fig = ui.forecast_fig(list(pr1.hist.index), pr1.hist.values, list(pr1.index), pr1.fc, pr1.lo, pr1.hi, ui.UNIT, t("Escenario"), height=300, compare=[(t("Actual"), pr0.fc)])
    tb = {"headers": [t("Indicador"), t("Actual"), t("Escenario"), t("Diferencia")], "rows": [[lab, f"{a0:,.{n}f}", f"{b0:,.{n}f}", f"{b0 - a0:+,.{n}f}"] for _, lab, a0, b0, n in items]}
    return res, {"type": "whatif", "id": pid, "title": t("Escenario propuesto"), "desc": desc, "fig": fig, "table": tb}


def tool_signals(a):
    cur = get_sig()
    return {"signals": [{"name": k, "current_change_pct": _r(v["chg"]), "sensitivity": v["sens"], "lead_months": v["lead"]} for k, v in cur.items()]}, None


def tool_explain(a):
    prep, g, mk, h = _tool_sel(a)
    m = prep.monthly
    ml = get_ml(m, E.windows_for(m))
    xg = ((ml or {}).get("extra") or {}).get("xgb")
    if xg is None or xg["final"].model is None:
        raise ValueError("XGBoost no está disponible en este equipo, así que no hay explicación SHAP.")
    step = int(a.get("month_index") or 0)
    ex = MLm.explain(xg["final"], g, mk, step)
    if ex is None:
        raise ValueError("No hay datos suficientes para explicar esa selección.")
    c = ex["contrib"].sort_values()
    idx = pd.date_range(m["ym"].max() + pd.offsets.MonthBegin(1), periods=12, freq="MS")
    res = {"month": f"{idx[step]:%Y-%m}", "model_forecast": _r(ex["fc"], 0), "base_value": _r(ex["base"], 0), "unit": ui.UNIT,
           "raises": {FEATURE_PLAIN.get(k, k): _r(v, 0) for k, v in c[c > 0.05].sort_values(ascending=False).head(4).items()},
           "lowers": {FEATURE_PLAIN.get(k, k): _r(v, 0) for k, v in c[c < -0.05].head(4).items()}, "note": "Es la explicación del modelo XGBoost, no del método principal."}
    return res, {"type": "fig", "title": t("Por qué el modelo pronostica esto") + f" · {ui.mes(idx[step], True)}", "fig": ui.explain_fig([t(FEATURE_PLAIN.get(k, k)) for k in c.index], c.values)}


def tool_waterfall(a):
    prep, g, mk, h = _tool_sel(a)
    only = g if len(g) < len(prep.gauges) else None
    wd = waterfall_data(h, only)
    labels = [wd["start_label"]] + [s[0] for s in wd["steps"]] + [t("Pronóstico actual")]
    res = {"horizon_months": h, "gauges": g, "unit": ui.UNIT, "starting_point": wd["start_label"], "start": _r(wd["start"], 0), "end": _r(wd["end"], 0), "compared_with_frozen": wd["frozen"],
           "factors": [{"factor": s[0], "contribution": _r(s[1], 1), "what_it_is": s[2]} for s in wd["steps"]],
           "note": "Descomposición secuencial: cada efecto se mide con los anteriores ya aplicados." + ("" if not only else " Con una selección de calibres no se compara contra el congelado por factor.")}
    return res, {"type": "fig", "title": t("Por qué cambió el pronóstico"), "fig": ui.waterfall_fig(labels, [s[1] for s in wd["steps"]], wd["start"], height=330)}


def tool_alerts(a):
    d = pres_data()
    al = business_alerts(d["m"], d["pr"], d["res"], CRM.frozen_list()) + [{"sev": "Alta", "text": x, "action": "Revisar Metales."} for x in _safe_market_texts()]
    n_an = len(pending_anomalies(d["m"]))
    al = sorted(al, key=lambda x: 0 if x["sev"] == "Alta" else 1)[:10]
    res = {"alerts": [{"severity": x["sev"], "text": x["text"], "suggested_action": x["action"]} for x in al], "anomalies_waiting_review": n_an}
    return res, {"type": "table", "title": t("Qué atender"), "headers": [t("Urgencia"), t("Alerta"), t("Qué hacer")], "rows": [[t(x["sev"]), x["text"], x["action"]] for x in al]}


def tool_prices(a):
    snap = live_snapshot()
    out, rows = {}, []
    for s, nm in (("cu", "cobre"), ("al", "aluminio"), ("fx", "USD/MXN")):
        q = snap["q"].get(s)
        if q is None or not np.isfinite(q.value):
            out[nm] = None
            continue
        out[nm] = {"value": _r(q.value, 2), "unit": "USD/t" if s != "fx" else "MXN por USD", "day_change_pct": _r(q.day_pct), "source": q.source, "time": q.ts, "status": q.status, "manual_override": px_mode(s) == "manual"}
        rows.append([nm.capitalize() if s != "fx" else nm, f"{q.value:,.2f}", f"{q.day_pct:+.1f}%" if q.day_pct is not None else "—", q.source, (q.ts or "")[:16].replace("T", " "), t(q.status)])
    out["alerts"] = _safe_market_texts()
    out["note"] = "COMEX/ALI son una referencia cercana al LME, no su precio oficial. status stale = último precio guardado."
    return out, {"type": "table", "title": t("Precios de mercado"), "headers": [t("Serie"), t("Precio"), t("Cambio del día"), t("Fuente"), t("Hora"), t("Estado")], "rows": rows}


def tool_news(a):
    rows = latest_news(a.get("topic"), int(a.get("n") or 5), days=60, min_rel=("Alta",) if a.get("min_relevance") == "Alta" else ("Alta", "Media"))
    res = {"news": [{"title": r["title"], "source": r["source"], "date": r["published"][:10], "topic": r["topic"], "relevance": r["relevance"], "tags": r["tags"],
                     "suggested_action": (r["action"] or {}).get("text"), "url": r["url"]} for r in rows],
           "note": "Clasificación automática por reglas; una persona decide. Ninguna noticia cambia el pronóstico sola."}
    return res, {"type": "table", "title": t("Noticias y regulación"), "headers": [t("Fecha"), t("Titular"), t("Tema"), t("Relevancia")], "rows": [[r["published"][:10], r["title"], t(r["topic"]), t(r["relevance"])] for r in rows]}


def tool_plan(a):
    prep, g, mk, h = _tool_sel(a)
    pr, tot, res = tool_eval(copy.deepcopy(dict(store())), g, h)
    by = res.rows.groupby("cal")[["vol", "rev", "margin", "ss"]].sum()
    cap = res.capacity
    out = {"horizon_months": h, "unit": ui.UNIT, "totals": {"volume": _r(tot["vol"], 0), "revenue_mxn": _r(tot["rev"], 0), "gross_margin_mxn": _r(tot["margin"], 0), "gross_margin_pct": _r(tot["margin_pct"]), "metal_cost_mxn": _r(tot["metal_cost"], 0),
                                                                    "safety_stock": _r(tot["ss"], 1), "safety_stock_value_mxn": _r(tot["ss_value"], 0), "capital_freed_mxn": _r(tot["freed"], 0), "copper_needed_tons": _r(tot["cu"], 1),
                                                                    "aluminum_needed_tons": _r(tot["al"], 1), "months_over_capacity": tot["over_months"]},
           "by_gauge": [{"gauge": k, "volume": _r(v["vol"], 0), "revenue_mxn": _r(v["rev"], 0), "margin_mxn": _r(v["margin"], 0), "safety_stock": _r(v["ss"], 1)} for k, v in by.iterrows()],
           "capacity": [{"gauge": r.cal, "capacity_per_month": _r(r.cap, 0), "peak_utilization_pct": _r(max(r.util), 0), "months_over": int(sum(1 for x in r.util if x > 100))} for r in cap.itertuples()] if len(cap) else [],
           "note": "Precios de lista, costos, capacidad y elasticidades son ilustrativos."}
    tb = {"type": "table", "title": t("Plan {h} meses", h=h), "headers": [t("Calibre"), f"{t('Volumen')} ({ui.UNIT})", t("Ingreso (MXN)"), t("Margen (MXN)"), t("Inventario de seguridad")],
          "rows": [[ui.cal_label(k), f"{v['vol']:,.0f}", ui.mxn(v["rev"]), ui.mxn(v["margin"]), f"{v['ss']:,.1f}"] for k, v in by.iterrows()]}
    return out, tb


def tool_customers(a):
    m = default_prep().monthly
    dim = I.entity_dim(m)
    tr = I.entity_trends(m, dim).sort_values("pct12", ascending=False).head(int(a.get("top") or 8))
    pipe = CRM.pipeline_summary([o for o in CRM.opps() if not o["sample"] or store().get("crm_use_sample", False)])
    out = {"customer_dimension": "cliente" if dim == "cli" else "fuerza de venta (no hay columna de cliente)", "top": [{"name": r.entidad, "last_6m": _r(r.ultimos6, 0), "previous_6m": _r(r.previos6, 0), "change_pct": _r(r.cambio), "buying_less": bool(r.baja),
                                                                                                                    "share_of_12m_pct": _r(r.pct12)} for r in tr.itertuples()],
           "pipeline": {"open_opportunities": pipe["open_n"], "open_tons": _r(pipe["open_tons"], 0), "weighted_tons": _r(pipe["weighted_tons"], 0), "win_rate_pct": _r(pipe["win_rate"], 0) if pipe["win_rate"] is not None else None}}
    return out, {"type": "table", "title": t("Clientes"), "headers": [t("Cliente"), t("Últimos 6 meses"), t("6 meses previos"), t("Cambio %"), t("Compra a la baja")],
                 "rows": [[r.entidad, f"{r.ultimos6:,.0f}", f"{r.previos6:,.0f}", f"{r.cambio:+.0f}%" if np.isfinite(r.cambio) else "—", t("Sí") if r.baja else ""] for r in tr.itertuples()]}


def tool_summary(a):
    d = pres_data()
    al = business_alerts(d["m"], d["pr"], d["res"], CRM.frozen_list()) + [{"sev": "Alta", "text": x, "action": "", "section": "Metales"} for x in _safe_market_texts()]
    B = build_summary(d["m"], d["pr"], d["res"], al, d["h"])
    return {"summary": [{"title": b["title"], "lines": b["lines"]} for b in B], "note": "Resumen listo para una junta de S&OP; se puede descargar en PDF o Excel."}, {"type": "summary", "blocks": B}


def tool_risk(a):
    h = int(a.get("horizon") or 6)
    if h not in (3, 6, 12):
        raise ValueError("El horizonte debe ser 3, 6 o 12 meses.")
    old = store().get("rk_hedge")
    if a.get("hedge_pct") is not None:
        store()["rk_hedge"] = int(10 * round(min(max(float(a["hedge_pct"]), 0), 100) / 10))     # solo para este cálculo
    try:
        rs = risk_summary(h, 5000)
    finally:
        if old is None:
            store().pop("rk_hedge", None)
        else:
            store()["rk_hedge"] = old
    if rs is None:
        raise ValueError("No hay un plan que simular.")
    s, hd, ctx, o = rs["s"], rs["hedged"], rs["ctx"], rs["o"]
    p = ctx["setup"].params
    k = min(int(sv("rk_cov", 3)), h)
    nvs = {}
    for key, nm, price in (("cu", "cobre", p.cu_lme), ("al", "aluminio", p.al_lme)):
        need = o["metal_need"][key][:, :k].sum(axis=1)
        if need.mean() > 1e-6:
            nv = RK.newsvendor(need, price * p.fx, float(sv("rk_rush", 8)), float(p.hold_pct), k)
            nvs[nm] = {"buy_now_tons": _r(nv["q_star"], 0), "coverage_months": k, "need_p10": _r(nv["p10"], 0), "need_median": _r(nv["median"], 0), "need_p90": _r(nv["p90"], 0), "critical_ratio_pct": _r(nv["critical_ratio"] * 100, 0),
                       "shortage_cost_per_ton_mxn": _r(nv["under_cost"], 0), "excess_cost_per_ton_mxn": _r(nv["over_cost"], 0)}
    res = {"horizon_months": h, "simulations": rs["n"], "margin_mxn": {"expected": _r(s["mean"], 0), "p5": _r(s["p5"], 0), "p50": _r(s["p50"], 0), "p95": _r(s["p95"], 0), "at_risk_expected_minus_p5": _r(s["at_risk"], 0)},
           "prob_margin_below_threshold_pct": _r(s["prob_below"] * 100, 1), "threshold_mxn": _r(rs["thr"], 0),
           "hedge": {"fixed_pct": _r(rs["frac"] * 100, 0), "months": min(int(sv("rk_hm", h)), h), "margin_p5_with_hedge_mxn": _r(hd["p5"], 0), "margin_at_risk_with_hedge_mxn": _r(hd["at_risk"], 0)},
           "optimal_purchase": nvs, "calibration": {"band_inflation": _r(ctx["cal"]["inflate"], 1), "coverage_p10_p90_pct": _r(ctx["cal"]["coverage"], 0), "coverage_before_pct": _r(ctx["cal"]["coverage_raw"], 0)},
           "note": "Supuestos ajustables en la sección Riesgo; precios y costos ilustrativos; con pocas ventanas de validación las colas son una aproximación."}
    fig = ui.hist_fig(o["margin"] / 1e6, [(s["p5"] / 1e6, "P5", ui.FG), (s["mean"] / 1e6, t("Esperado"), ui.RED)], height=280)
    return res, {"type": "fig", "title": t("Distribución del margen"), "fig": fig}


TOOLS = {"get_risk_analysis": tool_risk, "get_forecast": tool_forecast, "run_whatif": tool_whatif, "get_signals_list": tool_signals, "explain_forecast": tool_explain, "why_forecast_changed": tool_waterfall,
                "get_alerts": tool_alerts, "get_live_prices": tool_prices, "get_news": tool_news, "get_plan": tool_plan, "get_customers": tool_customers, "get_executive_summary": tool_summary}


def tool_dispatch(name: str, args: dict):
    fn = TOOLS.get(name)
    if fn is None:
        raise ValueError(f"Herramienta desconocida: {name}")
    return fn(args or {})


def asst_apply(pid: str):
    """Aplica un escenario propuesto por el asistente. Solo ocurre al pulsar el botón; queda en la bitácora."""
    p = asst_state()["proposals"].get(pid)
    if not p or p["applied"] or not can("planeacion"):
        return
    for k, v in p["changes"].items():
        pres_put(k, v)
    p["applied"] = True
    store()["wi_banner"] = {"id": pid, "title": t("Escenario del asistente"), "text": t("Lo propuso el asistente y lo aplicaste tú."), "changes": p["desc"]}
    CRM.audit(who(), role(), "asistente: escenario aplicado", "asistente", pid, "; ".join(p["desc"]))


def asst_reset():
    st.session_state.pop(ASST, None)


def asst_ask(q: str, orig: str | None = None):
    S = asst_state()
    S["pending"] = q
    S["learn_from"] = orig          # si venía de una pregunta no entendida, se aprende qué quiso decir (solo palabras de contenido)


def asst_render(art: dict, key: str):
    """Muestra el gráfico o la tabla que produjo una herramienta."""
    ty = art["type"]
    if ty in ("forecast", "whatif", "fig", "table"):
        if art.get("title"):
            st.html(f'<h3 style="margin:8px 0 4px;font-size:1.02em">{esc(art["title"])}</h3>')
    if ty == "whatif" and art.get("desc"):
        st.html(f'<p class="hint">{esc(", ".join(art["desc"]))}</p>')
    if art.get("fig") is not None:
        st.plotly_chart(art["fig"], width="stretch", theme=None, config={"displayModeBar": False}, key=f"cf_{key}")
    tb = art.get("table") if ty in ("forecast", "whatif") else (art if ty == "table" else None)
    if tb:
        st.html(ui.table(tb["headers"], tb["rows"], num=tuple(range(1, len(tb["headers"]))) if ty != "table" else (), min_w=520))
    if ty == "whatif":
        p = asst_state()["proposals"].get(art["id"])
        if p:
            c = st.columns([1.3, 3])
            if p["applied"]:
                c[0].html(f'<span class="badge fill">{esc(t("Aplicado"))}</span>')
            else:
                c[0].button(t("Aplicar escenario"), key=f"cap_{key}", type="primary", on_click=asst_apply, args=(art["id"],), disabled=not can("planeacion"))
                if not can("planeacion"):
                    c[1].html(f'<p class="hint">{esc(t("Tu rol no permite aplicar escenarios."))}</p>')
    if ty == "summary":
        from viakable_forecast import summary as SUMM
        for b in art["blocks"]:
            st.html(f'<h3 style="margin:10px 0 2px;font-size:1.02em">{esc(b["title"])}</h3>' + "".join(f'<p class="hint" style="margin:2px 0">{esc(x)}</p>' for x in b["lines"]))
        title, sub = t("Resumen del mes"), f"{APP_NAME} · {ui.mes(pd.Timestamp(dt.date.today()), True)}"
        c = st.columns([1, 1, 3])
        c[0].download_button(t("Descargar PDF"), SUMM.to_pdf(title, sub, art["blocks"], t(FOOTER)), file_name=f"resumen-{dt.date.today():%Y-%m}.pdf", mime="application/pdf", key=f"dp_{key}", width="stretch")
        c[1].download_button(t("Descargar Excel"), SUMM.to_xlsx(title, sub, art["blocks"], t(FOOTER)), file_name=f"resumen-{dt.date.today():%Y-%m}.xlsx",
                             mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"dx_{key}", width="stretch")








def tool_accuracy(a):
    """Precisión de validación del total y de cada calibre (100 menos el error de validación cruzada) y el método elegido."""
    prep = default_prep()
    allg = list(prep.gauges)
    m = prep.monthly
    ml = get_ml(m, E.windows_for(m))
    prices = get_prices()
    h = int(a.get("horizon") or 6)
    pr = cached_project(m, tuple((g,) for g in allg), None, h, E.AUTO, E.TON, "sum", get_mult(), get_sig(), get_w(), prices, ml, get_crm(None, E.TON, prices), apply_signals_on())
    rows = [{"gauge": r["Calibre"], "accuracy_pct": _r(r["Precisión %"]), "method": r["Método"], "evaluation": r["Evaluación"], "demand_type": r["Demanda"]} for _, r in pr.table.iterrows()]
    res = {"total_accuracy_pct": _r(max(0, 100 - pr.error)), "total_error_pct": _r(pr.error), "by_gauge": rows, "method_total": pr.method,
           "note": "Precisión = 100 menos el error porcentual de validación cruzada de origen móvil contra meses reales (no se usa R²)."}
    tb = {"type": "table", "title": t("Precisión de validación"), "headers": [t("Calibre"), t("Precisión %"), t("Método"), t("Evaluación")],
          "rows": [[ui.cal_label(r["gauge"]), r["accuracy_pct"], t(r["method"]), t(r["evaluation"])] for r in rows]}
    return res, tb


TOOLS["get_accuracy"] = tool_accuracy


# ================================================================== asistente Calibra (sin IA externa)
from contextlib import contextmanager


def asst_state() -> dict:
    return st.session_state.setdefault(ASST, {"msgs": [], "proposals": {}, "pending": None, "ctx": None})


@contextmanager
def answer_lang(lang: str):
    """Las respuestas salen en el idioma de la pregunta, sin cambiar el de la interfaz."""
    import i18n as _i18n
    old = _i18n.LANG
    set_lang(lang)
    ui.LANG_EN = lang == "en"
    try:
        yield
    finally:
        set_lang(old)
        ui.LANG_EN = old == "en"


def asst_customers() -> list[str]:
    names = [c["name"] for c in CRM.customers()]
    try:
        m = default_prep().monthly
        dim = I.entity_dim(m)
        if dim:
            names += [str(x) for x in m[dim].dropna().unique()]
    except Exception:
        pass
    return list(dict.fromkeys(names))


def _scope(e) -> str:
    parts = []
    if e.gauges:
        parts.append(t("del calibre {g}", g=" + ".join(e.gauges)) if len(e.gauges) == 1 else t("de los calibres {g}", g=" + ".join(e.gauges)))
    else:
        parts.append(t("de todos los calibres"))
    if e.market:
        parts.append(t(e.market).lower())
    return " ".join(parts)


def _call(name: str, args: dict):
    return tool_dispatch(name, args)


def _tariff_level(e) -> float:
    cur = float(sv("t_cur", TARIFF_CURRENT))
    if e.tariff_zero:
        return 0.0
    if e.pct is None:
        return cur
    return float(e.pct) if e.pct_abs else min(60.0, max(0.0, cur + float(e.pct)))


def asst_ask_text(slot: str, e) -> str:
    if slot == "lever":
        return t("¿Qué quieres simular: el precio del cobre o del aluminio, el arancel, una señal de mercado o un pedido adicional?")
    if slot == "pct":
        what = t("del cobre") if e.metals == ["cu"] else t("del aluminio") if e.metals == ["al"] else t("del arancel") if e.tariff else t("de ese cambio")
        return t("¿De cuánto es el cambio {w}? Dime un porcentaje, por ejemplo 10%.", w=what)
    if slot == "gauge":
        return t("¿De qué calibre es el pedido adicional? Por ejemplo, calibre 12.")
    if slot == "tons":
        return t("¿Cuántas toneladas por mes trae el pedido? Por ejemplo, 20 t.")
    return t("¿Durante cuántos meses? Por ejemplo, 4 meses.")


def _horizon_for(e) -> int:
    h = e.horizon or 6
    if e.month:
        last = default_prep().monthly["ym"].max()
        idx = 0 if e.month == -1 else (e.month - last.month - 1) % 12
        h = max(h, 3 if idx < 3 else 6 if idx < 6 else 12)
    return h


def ans_forecast(e):
    if e.market and not (default_prep().monthly["mkt"] == e.market).any():
        return t("Tus datos no tienen ventas de {m}, así que no hay pronóstico para ese mercado. Define la regla de exportación en Datos.", m=t(e.market)), []
    h = _horizon_for(e)
    res, art = _call("get_forecast", {"gauges": e.gauges or None, "market": e.market, "horizon": h})
    u = res["unit"]
    lo, hi = sum(res["low_80"]), sum(res["high_80"])
    txt = t("Esperamos {v} {u} {s} en los próximos {h} meses (rango {a} a {b}), con el método {m} y {p}% de precisión de validación.", v=f"{res['total']:,.0f}", u=u, s=_scope(e), h=h, a=f"{lo:,.0f}", b=f"{hi:,.0f}", m=t(res["method"]), p=f"{res['validation_accuracy_pct']:.0f}")
    if e.month:
        i = 0 if e.month == -1 else next((k for k, mo in enumerate(res["months"]) if int(mo[5:7]) == e.month), None)
        if i is not None:
            mo = pd.Timestamp(res["months"][i] + "-01")
            txt += " " + t("Para {mes}: {v} {u} (rango {a} a {b}).", mes=ui.mes(mo, True), v=f"{res['forecast'][i]:,.0f}", u=u, a=f"{res['low_80'][i]:,.0f}", b=f"{res['high_80'][i]:,.0f}")
    ch = [r for r in res["by_gauge"] if r["change_vs_same_months_last_year_pct"] is not None]
    if len(res["by_gauge"]) == 1 and ch:
        txt += " " + t("Es {c}% {d} el mismo periodo del año anterior.", c=f"{abs(ch[0]['change_vs_same_months_last_year_pct']):.0f}", d=t("más que en") if ch[0]["change_vs_same_months_last_year_pct"] >= 0 else t("menos que en"))
    return txt, [art]


def ans_whatif(e):
    h = e.horizon or 6
    args = {"horizon": h}
    if e.pct is not None and (e.metals or e.both_metals):
        if e.both_metals or (set(e.metals) == {"cu", "al"}):
            args["metal_pct"] = e.pct
        else:
            args["copper_pct" if e.metals == ["cu"] else "aluminum_pct"] = e.pct
    if e.fx and e.pct is not None and not (e.metals or e.tariff):
        args["fx_pct"] = e.pct
    if e.tariff:
        args["tariff_pct"] = _tariff_level(e)
    if e.signals and e.pct is not None:
        args["signals"] = {n: e.pct for n in e.signals}
    if e.extra_order and e.tons is not None and e.gauges and e.months_n:
        args["extra_order"] = {"gauge": e.gauges[0], "tons_per_month": e.tons, "months": e.months_n}
    res, art = _call("run_whatif", args)
    c = res["comparison"]
    f, mg = c["forecast"], c["margin"]
    txt = t("Con {d}, el pronóstico a {h} meses pasa de {a} a {b} {u} ({x}), el margen bruto de {m0} a {m1} ({dm}) y el costo del metal cambia {mc}.", d=", ".join(res["changes"]), h=h, a=f"{f['current']:,.0f}", b=f"{f['scenario']:,.0f}", u=ui.UNIT,
            x=f"{f['difference']:+,.0f}", m0=ui.mxn(mg["current"]), m1=ui.mxn(mg["scenario"]), dm=ui.mxn(mg["difference"]), mc=ui.mxn(c["metal_cost"]["difference"]))
    if e.tariff and not (default_prep().monthly["mkt"] == MKT_USA).any():
        txt += " " + t("Tus datos no traen ventas de exportación definidas, así que el arancel no mueve nada; defínelas en Datos.")
    txt += " " + t("Es solo una simulación: pulsa Aplicar escenario si quieres usarla.")
    return txt, [art]


def ans_why(e):
    h = e.horizon or 6
    res, art = _call("why_forecast_changed", {"gauges": e.gauges or None, "horizon": h})
    top = [x for x in sorted(res["factors"], key=lambda x: -abs(x["contribution"])) if abs(x["contribution"]) >= 0.5][:3]
    if top:
        txt = t("El pronóstico {s} de {h} meses es {b} {u}. Partiendo de {a} ({v} {u}), lo que más lo mueve: {f}.", s=_scope(e), h=h, b=f"{res['end']:,.0f}", u=res["unit"], a=res["starting_point"], v=f"{res['start']:,.0f}",
                f=", ".join(f"{x['factor'].lower()} ({x['contribution']:+,.0f} {res['unit']})" for x in top))
    else:
        txt = t("Ningún factor mueve el pronóstico {s}: es el pronóstico estadístico base de {v} {u}.", s=_scope(e), v=f"{res['start']:,.0f}", u=res["unit"])
    if len(e.gauges) == 1:
        tr = I.gauge_trends(default_prep().monthly)
        r = tr[tr["calibre"] == e.gauges[0]]
        if len(r) and np.isfinite(r["cambio"].iloc[0]):
            c = float(r["cambio"].iloc[0])
            txt += " " + t("Además, las ventas de los últimos 3 meses del calibre {g} están {c}% {d} mismo periodo del año anterior.", g=e.gauges[0], c=f"{abs(c):.0f}", d=t("por encima del") if c >= 0 else t("por debajo del"))
    return txt, [art]


def ans_alerts(e):
    res, art = _call("get_alerts", {})
    al = res["alerts"]
    if not al:
        return t("Sin alertas por ahora."), [art]
    hi = [x for x in al if x["severity"] == "Alta"]
    first = al[0]
    txt = t("Alertas: {n} en total, {k} urgentes. La más urgente: {x} Qué hacer: {a}", n=len(al), k=len(hi), x=first["text"], a=first["suggested_action"])
    if res["anomalies_waiting_review"]:
        txt += " " + t("También hay {n} anomalías del ritmo de pedidos por revisar en Inicio.", n=res["anomalies_waiting_review"])
    return txt, [art]


def ans_prices(e):
    res, art = _call("get_live_prices", {})
    bits = []
    for key, nm in (("cobre", t("El cobre")), ("aluminio", t("el aluminio"))):
        d = res.get(key)
        if d and (not e.metals or ("cu" if key == "cobre" else "al") in e.metals):
            bits.append(f"{nm} {t('está en')} ${d['value']:,.0f} USD/t" + (f" ({d['day_change_pct']:+.1f}% {t('hoy')})" if d["day_change_pct"] is not None else ""))
    fx = res.get("USD/MXN")
    if fx and (not e.metals or e.fx):
        bits.append(f"USD/MXN {t('está en')} {fx['value']:.2f}")
    if not bits:
        return t("No tengo precios disponibles ahora; la app usa los últimos guardados."), [art]
    stale = any(isinstance(v, dict) and v.get("status") != "live" for v in res.values())
    txt = ", ".join(bits) + "." + (" " + t("Son los últimos precios guardados (sin conexión).") if stale else "")
    if res.get("alerts"):
        txt += " " + res["alerts"][0]
    return txt, [art]


def ans_news(e):
    topic = "Arancel / T-MEC" if e.tariff else ("Precios de metales" if (e.metals or e.both_metals) else None)
    res, art = _call("get_news", {"topic": topic, "n": 5})
    items = res["news"]
    if not items:
        return t("Todavía no hay noticias guardadas con ese filtro."), []
    r = items[0]
    txt = t("La noticia más relevante: «{ti}» ({s}, {d}). Tema: {tp}; relevancia {rl}.", ti=r["title"], s=r["source"], d=r["date"], tp=t(r["topic"]), rl=t(r["relevance"]))
    if r["suggested_action"]:
        txt += " " + t("Qué hacer: {a}", a=t(r["suggested_action"]))
    txt += " " + t("Hay {n} noticias relevantes recientes; ninguna cambia el pronóstico sola.", n=len(items))
    return txt, [art]


def ans_plan(e):
    h = e.horizon or 6
    res, art = _call("get_plan", {"gauges": e.gauges or None, "horizon": h})
    tt = res["totals"]
    txt = t("Plan de {h} meses {s}: ingreso {r}, margen bruto {m} ({p}%), inventario de seguridad {ss} {u}. Necesitas {cu} {u} de cobre y {al} {u} de aluminio.", h=h, s=_scope(e), r=ui.mxn(tt["revenue_mxn"]), m=ui.mxn(tt["gross_margin_mxn"]), p=f"{tt['gross_margin_pct']:.1f}",
            ss=f"{tt['safety_stock']:,.0f}", u=res["unit"], cu=f"{tt['copper_needed_tons']:,.0f}", al=f"{tt['aluminum_needed_tons']:,.0f}")
    if tt["months_over_capacity"]:
        txt += " " + t("Hay {n} meses por encima de la capacidad.", n=tt["months_over_capacity"])
    else:
        txt += " " + t("Ningún mes pasa de la capacidad.")
    arts = [art]
    if (e.metals or e.both_metals) or re.search(r"(compr|buy|purchase)", asst_state().get("last_text", "")):
        rk, rart = _call("get_risk_analysis", {"horizon": h})
        for nm, d in rk["optimal_purchase"].items():
            txt += " " + t("Compra hoy {q} {u} de {m}: cubre {k} meses y la necesidad va de {a} a {b} {u}.", q=f"{d['buy_now_tons']:,.0f}", u=res["unit"], m=t(nm), k=d["coverage_months"], a=f"{d['need_p10']:,.0f}", b=f"{d['need_p90']:,.0f}")
        arts.append(rart)
    return txt, arts


def ans_customers(e):
    res, art = _call("get_customers", {"top": 100})
    top = res["top"]
    if e.customers:
        r = next((x for x in top if x["name"] == e.customers[0]), None)
        if r:
            ch = r["change_pct"]
            return (t("{c}: {a} {u} en los últimos 6 meses contra {b} en los 6 previos ({x}%).", c=r["name"], a=f"{r['last_6m']:,.0f}", u=ui.UNIT, b=f"{r['previous_6m']:,.0f}", x=f"{ch:+.0f}" if ch is not None else "—")
                    + (" " + t("Compra a la baja.") if r["buying_less"] else "")), [art]
    low = [x for x in top if x["buying_less"]]
    if low:
        txt = t("{n} compran a la baja (más de 20% menos que en los 6 meses previos): {l}.", n=len(low), l=", ".join(f"{x['name']} ({x['change_pct']:+.0f}%)" for x in low[:4]))
    else:
        txt = t("Ningún cliente cae más de 20% contra los 6 meses previos.")
    pp = res["pipeline"]
    txt += " " + t("El pipeline abierto es de {a} {u} ({b} {u} ponderadas).", a=f"{pp['open_tons']:,.0f}", u=ui.UNIT, b=f"{pp['weighted_tons']:,.0f}")
    if res["customer_dimension"] != "cliente":
        txt += " " + t("Tus datos no traen cliente; se usa la fuerza de venta.")
    return txt, [art]


def ans_summary(e):
    res, art = _call("get_executive_summary", {})
    first = res["summary"][0]["lines"][0]
    return t("Preparé el resumen del mes. {x} Abajo lo puedes descargar en PDF o Excel.", x=first), [art]


def ans_accuracy(e):
    res, art = _call("get_accuracy", {"horizon": e.horizon or 6})
    rows = [r for r in res["by_gauge"] if r["accuracy_pct"] is not None and "+" not in str(r["gauge"])]
    if e.gauges:
        sel = [r for r in res["by_gauge"] if str(r["gauge"]) in e.gauges and r["accuracy_pct"] is not None]
        if sel:
            txt = " ".join(t("El calibre {g} tiene {p}% de precisión de validación con el método {m} (se mide contra meses reales con validación cruzada).", g=r["gauge"], p=f"{r['accuracy_pct']:.0f}", m=t(r["method"])) for r in sel)
            return txt + " " + t("La del total es {p}%.", p=f"{res['total_accuracy_pct']:.0f}"), [art]
    txt = t("La precisión de validación del total es {p}% (error {e}%), medida contra meses reales con validación cruzada.", p=f"{res['total_accuracy_pct']:.0f}", e=f"{res['total_error_pct']:.0f}")
    if rows:
        b, w = max(rows, key=lambda r: r["accuracy_pct"]), min(rows, key=lambda r: r["accuracy_pct"])
        txt += " " + t("El calibre más preciso es el {a} ({pa}%) y el menos, el {b} ({pb}%).", a=b["gauge"], pa=f"{b['accuracy_pct']:.0f}", b=w["gauge"], pb=f"{w['accuracy_pct']:.0f}")
    return txt, [art]


def ans_risk(e):
    res, art = _call("get_risk_analysis", {"horizon": e.horizon or 6})
    mg = res["margin_mxn"]
    txt = t("En {n} simulaciones el margen esperado es {m}; en el peor 5% queda en {p} (margen en riesgo {r}) y hay {q}% de probabilidad de quedar bajo {u}.", n=f"{res['simulations']:,}", m=ui.mxn(mg["expected"]), p=ui.mxn(mg["p5"]),
            r=ui.mxn(mg["at_risk_expected_minus_p5"]), q=f"{res['prob_margin_below_threshold_pct']:.0f}", u=ui.mxn(res["threshold_mxn"]))
    hd = res["hedge"]
    txt += " " + t("Fijar {x}% de las compras deja el peor 5% en {y}.", x=f"{hd['fixed_pct']:.0f}", y=ui.mxn(hd["margin_p5_with_hedge_mxn"]))
    return txt, [art]


def ans_explain(e):
    idx = 0
    if e.month and e.month != -1:
        last = default_prep().monthly["ym"].max()
        idx = (e.month - last.month - 1) % 12
    res, art = _call("explain_forecast", {"gauges": e.gauges or None, "market": e.market, "month_index": idx})
    ups = ", ".join(f"{k} ({v:+,.0f})" for k, v in list(res["raises"].items())[:3]) or "—"
    dn = ", ".join(f"{k} ({v:+,.0f})" for k, v in list(res["lowers"].items())[:3]) or "—"
    return t("Para {mes}, el modelo XGBoost espera {f} {u} desde una base de {b}. Lo que más lo sube: {a}. Lo que más lo baja: {d}.", mes=res["month"], f=f"{res['model_forecast']:,.0f}", u=res["unit"], b=f"{res['base_value']:,.0f}", a=ups, d=dn), [art]


def ans_help(e):
    return t("Puedo contestar sobre: pronósticos por calibre y mercado, escenarios (precio del cobre o aluminio, arancel, señales, pedidos), por qué cambió el pronóstico, alertas, precios en vivo, noticias, plan y compra de metal, riesgo, clientes que compran menos, precisión y el resumen para la junta. Prueba con estas preguntas."), []


def ans_smalltalk(e):
    return t("Hola, soy el asistente de Calibra. Pregúntame por el pronóstico, un escenario, los precios, las noticias o lo que hay que atender."), []


ANSWERS = {"forecast": ans_forecast, "whatif": ans_whatif, "why": ans_why, "alerts": ans_alerts, "prices": ans_prices, "news": ans_news, "plan": ans_plan, "customers": ans_customers, "summary": ans_summary,
           "accuracy": ans_accuracy, "risk": ans_risk, "explain": ans_explain, "help": ans_help, "smalltalk": ans_smalltalk}


def asst_respond_app(p) -> dict:
    """Respuesta de negocio (ruta a): plantillas con cifras del motor y, si hace falta, una pregunta corta o sugerencias."""
    with answer_lang(p.lang):
        if p.intent is None:
            return {"text": t("No entendí bien la pregunta. ¿Quisiste decir alguna de estas?"), "suggest": AS.suggestions(p), "artifacts": [], "tools": [], "source": [], "orig": p.text}
        if p.missing:
            return {"text": asst_ask_text(p.missing[0], p.ents), "artifacts": [], "ask": True, "tools": [], "source": []}
        try:
            txt, arts = ANSWERS[p.intent](p.ents)
        except Exception as ex:
            return {"text": t("No pude calcularlo: {x}", x=str(ex)[:200]), "artifacts": [], "tools": [], "error": True, "source": []}
        out = {"text": txt, "artifacts": arts, "tools": [], "source": ["app"]}
        if p.intent in ("help",):
            out["suggest"] = [AS.EXAMPLES[i][1 if p.lang == "en" else 0] for i in ("forecast", "whatif", "alerts")]
        return out


# ---- IA (Ollama local o nube gratuita) con reglas de fundamentación
def llm_secret(sec: str, key: str):
    return secret(sec, key)


def llm_allowed() -> tuple:
    return ("ollama", "gemini", "groq") if ST.cfg().get("asst_ai", "on") == "on" else ()


def llm_ready() -> bool:
    return bool(llm_allowed()) and LLM.any_available(llm_secret)


SYS_OPEN = {"es": ("Eres el asistente de Viakable Calibra, una app de pronóstico y planeación de una empresa mexicana que fabrica cable eléctrico. Responde en español de México, breve (máximo 120 palabras), claro y directo, "
                   "sin signos de exclamación y sin adornos. REGLAS: no inventes cifras de la empresa (ventas, precios, márgenes, inventarios, clientes): si la pregunta pide datos de Viakable, di que los consulta la app. "
                   "Si no estás seguro de algo o no lo sabes, dilo. No des asesoría legal ni financiera definitiva."),
            "en": ("You are the assistant of Viakable Calibra, a forecasting and planning app for a Mexican electrical cable manufacturer. Answer in English, briefly (at most 120 words), clearly and directly, "
                   "with no exclamation marks. RULES: never invent company figures (sales, prices, margins, inventory, customers): if asked for Viakable data, say the app looks it up. "
                   "If you are not sure or do not know, say so. Do not give definitive legal or financial advice.")}
SYS_MIXED = {"es": ("Eres el asistente de Viakable Calibra. Escribe una explicación breve (máximo 130 palabras) en español de México, clara y sin signos de exclamación, de cómo el tema de la pregunta podría afectar a la empresa. "
                    "USA ÚNICAMENTE las cifras del bloque CIFRAS (salen del motor de la app); no inventes ni calcules otras. Si una cifra no está ahí, habla de forma cualitativa. "
                    "Aclara que los escenarios de estrés son supuestos ilustrativos y no predicciones."),
             "en": ("You are the assistant of Viakable Calibra. Write a short explanation (at most 130 words) in English, clear and with no exclamation marks, of how the topic of the question could affect the company. "
                    "USE ONLY the figures in the FIGURES block (they come from the app's engine); do not invent or compute others. If a figure is not there, speak qualitatively. "
                    "State that stress scenarios are illustrative assumptions, not predictions.")}
_NUM = re.compile(r"\$?\d[\d,]*(?:\.\d+)?\s*(?:%|M\b|t\b|toneladas|tons)?")


def _num_key(s: str) -> str:
    return re.sub(r"[^\d.]", "", s.replace(",", "")).rstrip(".")


def numbers_ok(text: str, facts: str) -> tuple[bool, list]:
    """Revisa que la IA no haya inventado cifras: todo número de 2 o más dígitos (o con %, $ o decimales) debe aparecer en las cifras que se le pasaron."""
    allowed = {_num_key(m.group(0)) for m in _NUM.finditer(facts)}
    bad = []
    for m in _NUM.finditer(text):
        raw = m.group(0)
        k = _num_key(raw)
        if not k:
            continue
        simple = k.isdigit() and len(k) <= 1 and not any(c in raw for c in "%$")
        year = k.isdigit() and len(k) == 4 and 1900 <= int(k) <= 2100
        if simple or year or k in allowed:
            continue
        bad.append(raw.strip())
    return (not bad), bad


def strip_business_numbers(text: str) -> str:
    """En una respuesta general, quita las oraciones con cifras de dinero o toneladas: esas cifras solo las da el motor."""
    sents = re.split(r"(?<=[.!?])\s+", text)
    keep = [s for s in sents if not re.search(r"(\$\s?\d|\d\s?(MXN|USD|pesos|dólares|dollars|toneladas|tons?\b|millones de pesos))", s, re.I)]
    return " ".join(keep) if keep else text


def history_for_llm(S: dict, n: int = 2) -> list[dict]:
    """Últimos turnos (solo texto, sin datos personales) para que la IA entienda seguimientos."""
    out = []
    for m in S["msgs"][-2 * n:]:
        if m.get("source") and "ia" not in m["source"] and "wiki" not in m["source"]:
            continue
        out.append({"role": m["role"], "content": RT.redact(m["text"])[:600]})
    return out


def respond_kb(route, q, lang) -> dict:
    with answer_lang(lang):
        title, body = KBM.answer(route.entry, lang)
        return {"text": f"{title}. {body}", "artifacts": [], "source": ["kb"], "tools": []}


def respond_wiki(route, q, lang, S) -> dict | None:
    if ST.cfg().get("asst_wiki", "on") != "on":
        return None
    r = WIKI.lookup(q, lang)
    if not r:
        return None
    ans = WIKI.short_answer(r.get("lead") or r["extract"], q, focus=r.get("focus"))
    note = ""
    if llm_ready():
        try:
            sysmsg = ("Reescribe con tus propias palabras, en español de México, en máximo 2 oraciones cortas, la respuesta a la pregunta usando SOLO la información del EXTRACTO. No agregues datos que no estén ahí. Sin signos de exclamación."
                      if lang == "es" else "Rewrite in your own words, in English, in at most 2 short sentences, the answer to the question using ONLY the information in the EXTRACT. Do not add facts that are not there. No exclamation marks.")
            o = LLM.chat(sysmsg, [{"role": "user", "content": f"{'PREGUNTA' if lang == 'es' else 'QUESTION'}: {RT.redact(q)}\n{'EXTRACTO' if lang == 'es' else 'EXTRACT'}: {(r.get('lead') or r['extract'])[:1800]}"}], llm_secret, max_tokens=160, allow=llm_allowed())
            ok, bad = numbers_ok(o["text"], r.get("lead") or r["extract"])
            if ok:
                ans, note = o["text"], f"{o['provider']} · {o['model']}"
        except LLM.LLMUnavailable:
            pass
    return {"text": ans, "artifacts": [], "source": ["wiki"], "url": r["url"], "url_title": f"{r['title']} ({r['lang']}.wikipedia.org)", "ai_note": note, "tools": [], "extract_only": not note}


def respond_open(route, q, lang, S) -> dict:
    with answer_lang(lang):
        if not llm_ready():
            return {"text": t("Esta es una pregunta abierta y para contestarla necesito una IA, que no está disponible ahora. Puedes instalar Ollama (gratis, corre en tu Mac) o poner una llave gratuita de Gemini o Groq; las instrucciones están abajo. Mientras tanto contesto datos de la app, metodología y hechos de Wikipedia."),
                    "artifacts": [], "source": [], "tools": [], "no_llm": True, "suggest": [AS.EXAMPLES[i][1 if lang == "en" else 0] for i in ("forecast", "whatif", "alerts")]}
        msgs = history_for_llm(S) + [{"role": "user", "content": RT.redact(q)}]
        try:
            o = LLM.chat(SYS_OPEN[lang], msgs, llm_secret, max_tokens=360, allow=llm_allowed())
        except LLM.LLMUnavailable as ex:
            return {"text": t("No pude contactar a la IA ahora ({e}). Intenta de nuevo en un momento.", e=str(ex)[:160]), "artifacts": [], "source": [], "tools": [], "error": True}
        txt = strip_business_numbers(o["text"])
        return {"text": txt, "artifacts": [], "source": ["ia"], "ai_note": f"{o['provider']} · {o['model']}", "cloud": not o["local"], "tools": []}


def mixed_facts(q: str, p) -> tuple[dict, list]:
    """Cifras del motor para una pregunta que mezcla un tema general con la empresa. Los escenarios de estrés son supuestos del asistente, no predicciones."""
    e = p.ents if p else AS.extract(q)
    low = AS.norm(q)
    facts, arts = {}, []
    us = bool(re.search(r"\b(eeuu|usa|us|estados unidos|united states|america)\b", low)) or e.market == AS.MKT_USA
    mx = bool(re.search(r"\b(mexico|mexicano\w*|nacional)\b", low))
    macro = bool(re.search(r"(recesion|recession|crisis|desaceler|slowdown|inflacion|inflation|tasas|interest|guerra|war|huelga|strike|sequia)", low))
    try:
        if us or e.tariff or re.search(r"exporta|export", low):
            if (default_prep().monthly["mkt"] == AS.MKT_USA).any():
                res, art = _call("get_forecast", {"market": AS.MKT_USA, "horizon": 6})
                facts["pronostico_exportacion_6_meses_t"] = res["total"]
                arts.append(art)
            else:
                facts["nota"] = "Los datos no tienen ventas de exportación definidas, así que el motor no puede estimar un efecto en exportaciones."
        sig = {}
        if macro and (us or not mx):
            sig.update({"Inicios de vivienda EE. UU.": -10, "PMI manufacturero EE. UU.": -10})
        if macro and (mx or not us):
            sig.update({"PMI manufacturero México": -10, "Permisos de construcción": -10, "Naves industriales": -10})
        args = {"horizon": 6}
        if sig:
            args["signals"] = sig
        if e.tariff:
            args["tariff_pct"] = _tariff_level(e) if e.pct is not None else min(60.0, float(sv("t_cur", TARIFF_CURRENT)) + 10)
        if "cu" in e.metals or "al" in e.metals or e.both_metals:
            pct = e.pct if e.pct is not None else (-10.0 if e.direction < 0 else 10.0)
            args["metal_pct" if (e.both_metals or len(e.metals) == 2) else ("copper_pct" if "cu" in e.metals else "aluminum_pct")] = pct
        if len(args) > 1:
            res, art = _call("run_whatif", args)
            c = res["comparison"]
            facts["escenario_de_estres_supuesto"] = ", ".join(res["changes"])
            facts["pronostico_6_meses_t"] = {"actual": c["forecast"]["current"], "escenario": c["forecast"]["scenario"], "diferencia": c["forecast"]["difference"]}
            facts["margen_bruto_mxn"] = {"actual": c["margin"]["current"], "escenario": c["margin"]["scenario"], "diferencia": c["margin"]["difference"]}
            facts["ingreso_mxn"] = {"actual": c["revenue"]["current"], "escenario": c["revenue"]["scenario"]}
            arts.append(art)
        if not facts or list(facts) == ["nota"]:
            res, art = _call("get_risk_analysis", {"horizon": 6})
            facts["margen_esperado_mxn"] = res["margin_mxn"]["expected"]
            facts["margen_peor_5_por_ciento_mxn"] = res["margin_mxn"]["p5"]
            facts["probabilidad_margen_bajo_umbral_pct"] = res["prob_margin_below_threshold_pct"]
            arts.append(art)
    except Exception as ex:
        facts["error_del_motor"] = str(ex)[:160]
    return facts, arts


def respond_mixed(route, q, lang, S) -> dict:
    with answer_lang(lang):
        facts, arts = mixed_facts(q, route.parse)
        fjson = json.dumps(facts, ensure_ascii=False, default=str)
        base = {"artifacts": arts, "tools": [], "source": ["app"]}
        if llm_ready():
            try:
                content = f"{'PREGUNTA' if lang == 'es' else 'QUESTION'}: {RT.redact(q)}\n{'CIFRAS' if lang == 'es' else 'FIGURES'}: {fjson}"
                o = LLM.chat(SYS_MIXED[lang], history_for_llm(S) + [{"role": "user", "content": content}], llm_secret, max_tokens=380, allow=llm_allowed())
                ok, bad = numbers_ok(o["text"], fjson)
                if not ok:                                              # un segundo intento, más estricto
                    o = LLM.chat(SYS_MIXED[lang] + (" Tu respuesta anterior incluyó cifras que no estaban en CIFRAS; no uses ningún número fuera de ese bloque." if lang == "es" else " Your previous answer used figures that were not in FIGURES; use no number outside that block."),
                                 [{"role": "user", "content": content}], llm_secret, max_tokens=380, allow=llm_allowed())
                    ok, bad = numbers_ok(o["text"], fjson)
                if ok:
                    return {**base, "text": o["text"], "source": ["app", "ia"], "ai_note": f"{o['provider']} · {o['model']}", "cloud": not o["local"]}
                note = t("La IA incluyó cifras que no salen del motor ({x}) y las descarté; abajo van las cifras del motor.", x=", ".join(bad[:3]))
            except LLM.LLMUnavailable:
                note = ""
        else:
            note = ""
        lines = []
        if "nota" in facts:
            lines.append(facts["nota"])
        if "pronostico_6_meses_t" in facts:
            f, m = facts["pronostico_6_meses_t"], facts["margen_bruto_mxn"]
            lines.append(t("Con {d}, el pronóstico a 6 meses pasa de {a} a {b} {u} y el margen bruto de {m0} a {m1}. Es un escenario de estrés supuesto, no una predicción.", d=facts["escenario_de_estres_supuesto"], a=f"{f['actual']:,.0f}", b=f"{f['escenario']:,.0f}", u=ui.UNIT, m0=ui.mxn(m["actual"]), m1=ui.mxn(m["escenario"])))
        elif "margen_esperado_mxn" in facts:
            lines.append(t("Margen esperado {m}; en el peor 5% queda en {p}.", m=ui.mxn(facts["margen_esperado_mxn"]), p=ui.mxn(facts["margen_peor_5_por_ciento_mxn"])))
        lines.append(note or t("Para explicar cómo y por qué afecta ese tema necesito una IA, que no está disponible ahora; estas son las cifras del motor."))
        return {**base, "text": " ".join(lines)}


def asst_respond(route, q: str, S: dict) -> dict:
    """Despacha según la ruta y etiqueta la fuente. Las preguntas de negocio nunca pasan por la IA."""
    lang = route.parse.lang if route.parse else AS.detect_lang(q, ST.cfg()["lang"])
    if route.kind == "app":
        return asst_respond_app(route.parse)
    if route.kind == "kb":
        return respond_kb(route, q, lang)
    if route.kind == "wiki":
        r = respond_wiki(route, q, lang, S)
        if r:
            return r
        if llm_ready():
            return respond_open(route, q, lang, S)
        with answer_lang(lang):
            return {"text": t("No encontré un artículo claro de Wikipedia para esa pregunta. Prueba con otras palabras."), "artifacts": [], "source": [], "tools": [], "suggest": AS.suggestions(route.parse or AS.parse(q), 3) if route.parse else []}
    if route.kind == "mixed":
        return respond_mixed(route, q, lang, S)
    return respond_open(route, q, lang, S)


def asst_process(q: str):
    S = asst_state()
    cust = asst_customers()
    prev = S.get("ctx")
    p = AS.merge_pending(prev, q, cust) if prev else None
    if p is not None:
        route = RT.Route("app", 1.0, {}, p, reason="completa la pregunta anterior")
    else:
        route = RT.route(q, cust, ST.cfg()["lang"], CRM.learn_all(), S.get("last_intent"))
    S["last_text"] = AS.norm(q)
    # bitácora: las preguntas de negocio se guardan (sin correos ni teléfonos); las generales solo dejan constancia de la ruta
    if route.kind == "app":
        CRM.audit(who(), role(), "pregunta al asistente", "asistente", "", RT.redact(q)[:200])
    else:
        CRM.audit(who(), role(), "pregunta general al asistente", "asistente", "", t("ruta {r} (el texto no se guarda)", r=route.kind))
    S["msgs"].append({"role": "user", "text": q})
    r = asst_respond(route, q, S)
    if route.kind == "app" and route.parse and route.parse.intent:
        S["last_intent"] = route.parse.intent
        if S.get("learn_from"):
            toks = RT.content_tokens(S["learn_from"])
            if len(toks.split()) >= 2:
                CRM.learn_add(toks, route.parse.intent)
        S["ctx"] = route.parse if route.parse.missing else None
    else:
        S["ctx"] = None
        S["last_intent"] = None
    S["learn_from"] = None
    if r.get("suggest") and not r.get("source"):
        r["orig_q"] = q
    S["msgs"].append({"role": "assistant", **r, "route": route.kind, "conf": route.conf, "lang": route.parse.lang if route.parse else None, "intent": route.parse.intent if route.parse else None})


def asst_insights(h: int = 6) -> list[str]:
    """3 a 5 observaciones en lenguaje sencillo; cada cifra sale del motor. Se recalculan con cada cambio."""
    out = []
    try:
        prep = default_prep()
        m = prep.monthly
        ml = get_ml(m, E.windows_for(m))
        prices = get_prices()
        gauges = list(prep.gauges)
        pr = cached_project(m, tuple((g,) for g in gauges), None, h, E.AUTO, E.TON, "sum", get_mult(), get_sig(), get_w(), prices, ml, get_crm(None, E.TON, prices), apply_signals_on())
        tr = I.gauge_trends(m)
        tr = tr[(tr["anterior"] >= 5) & np.isfinite(tr["cambio"])]
        if len(tr):
            r = tr.iloc[tr["cambio"].abs().argmax()]
            if abs(r["cambio"]) >= 5:
                out.append(t("El calibre {g} vende {p}% {d} que hace un año (últimos 3 meses).", g=r["calibre"], p=f"{abs(r['cambio']):.0f}", d=t("más") if r["cambio"] > 0 else t("menos")))
        try:
            snap = live_snapshot()
            q = snap["q"].get("cu")
            if q is not None and q.day_pct is not None and abs(q.day_pct) >= 1.0:
                dm = alert_impact("cu", q.day_pct)
                out.append(t("El cobre {v} {p}% hoy; si se mantiene, el margen de 6 meses {e} unos {x} (estimado).", v=t("sube") if q.day_pct > 0 else t("baja"), p=f"{abs(q.day_pct):.1f}", e=t("baja") if dm < 0 else t("sube"), x=ui.mxn(abs(dm))))
        except Exception:
            pass
        dim = I.entity_dim(m)
        low = I.entity_trends(m, dim).query("baja")
        if len(low):
            r = low.sort_values("cambio").iloc[0]
            out.append(t("{c} compra {p}% menos que en los 6 meses previos.", c=r["entidad"], p=f"{abs(r['cambio']):.0f}"))
        an = pending_anomalies(m)
        if len(an):
            out.append(_anom_text(an.sort_values("at_risk", ascending=False).iloc[0]))
        hist = pr.hist
        prev = hist.reindex(pr.index - pd.DateOffset(years=1)).sum() if len(hist) else 0
        if prev > 0:
            c = (pr.fc.sum() / prev - 1) * 100
            out.append(t("El pronóstico a {h} meses es {x} {u}, {c}% {d} mismo periodo del año pasado.", h=h, x=f"{pr.fc.sum():,.0f}", u=ui.UNIT, c=f"{abs(c):.0f}", d=t("por encima del") if c >= 0 else t("por debajo del")))
        if adv():
            rs = risk_summary(h, 2000)
            if rs:
                out.append(t("Hay {p}% de probabilidad de que el margen de {h} meses quede bajo {u}; en el peor 5% queda en {m}.", p=f"{rs['s']['prob_below'] * 100:.0f}", h=h, u=ui.mxn(rs["thr"]), m=ui.mxn(rs["s"]["p5"])))
        od = CRM.overdue_tasks()
        if od:
            out.append(t("Tareas vencidas en el CRM: {n}.", n=len(od)))
    except Exception:
        pass
    return out[:5]


def insights_box(h: int = 6, link: bool = True):
    items = asst_insights(h)
    if not items:
        return
    with st.container(key="insights"):
        st.html(ui.h2(t("Lo que notó el asistente"), t("Observaciones en lenguaje sencillo, calculadas con el motor. Se actualizan con cada cambio.")) + "<ul class=\"insights\">" + "".join(f"<li>{esc(x)}</li>" for x in items) + "</ul>")
        if link:
            st.button(t("Preguntar al asistente"), key="ins_go", on_click=nav_go, args=("Asistente",))


SRC_LABEL = {"app": "Datos de la app", "kb": "Metodología", "wiki": "Wikipedia", "ia": "Respuesta general (IA)"}


def asst_feedback(route: str, source: str, ok: bool, key: str):
    CRM.feedback_add(route or "?", source or "?", ok)
    asst_state()["fb"] = {**asst_state().get("fb", {}), key: ok}


def _llm_note(note: str) -> str:
    if note.startswith("Ollama no responde"):
        return t("Ollama no responde: instálalo y ábrelo (ver instrucciones)")
    if note.startswith("Ollama corre"):
        return t("Ollama corre, pero falta un modelo: ejecuta ollama pull qwen2.5:3b")
    if note.startswith("sin llave"):
        return t("sin llave en secrets.toml")
    return note


def ai_status_panel():
    """Qué IA hay disponible para las preguntas abiertas y cómo conectar una gratis."""
    provs = LLM.providers(llm_secret)
    on = [p for p in provs if p.ok]
    head = t("IA para preguntas abiertas: {x}", x=(f"{on[0].label} · {on[0].model}" if on else t("no disponible")))
    with st.expander(head, expanded=not on and not asst_state()["msgs"]):
        rows = [[p.label, p.model or "—", Raw(f'<span class="badge {"fill" if p.ok else "warn"}">{esc(t("disponible") if p.ok else t("no disponible"))}</span>'), _llm_note(p.note)] for p in provs]
        st.html(ui.table([t("Proveedor"), t("Modelo"), t("Estado"), t("Nota")], rows, min_w=560))
        c = st.columns(2)
        wid(c[0].toggle, t("Usar Wikipedia para hechos generales"), "asst_wiki_on", True)
        wid(c[1].toggle, t("Usar IA para preguntas abiertas"), "asst_ai_on", True)
        for k, v in (("asst_wiki", sv("asst_wiki_on", True)), ("asst_ai", sv("asst_ai_on", True))):
            if ST.cfg().get(k) != ("on" if v else "off"):
                ST.put(k, "on" if v else "off")
        st.markdown(t("""**Ollama (local, gratis, privado):** 1) Instálalo desde https://ollama.com/download (Mac) o con `brew install ollama`. 2) Ábrelo (queda corriendo en http://localhost:11434). 3) En la terminal: `ollama pull qwen2.5:3b` (unos 2 GB; si tu Mac tiene poca memoria, `ollama pull llama3.2:1b`). 4) Reinicia la app. La pregunta nunca sale de tu Mac.

**Nube gratuita (opcional):** Gemini: crea una llave en https://aistudio.google.com/apikey y pon `[gemini]` con `api_key = "..."` en `.streamlit/secrets.toml`. Groq: llave en https://console.groq.com/keys y `[groq]` con `api_key = "..."`. Los límites del nivel gratuito cambian y cada proveedor los muestra en tu cuenta (AI Studio y la consola de Groq); revísalos ahí. En la nube, el texto de la pregunta (sin correos ni teléfonos) se envía al proveedor, que puede usarlo según sus términos."""))
        st.caption(t("Las respuestas de negocio nunca pasan por la IA: las cifras salen del motor. La IA solo contesta temas abiertos o redacta cifras que el motor ya calculó."))


def render_sources(msg: dict):
    src = msg.get("source") or []
    if not src:
        return
    badges = "".join(f'<span class="badge {"fill" if s == "app" else ""}" style="margin-right:6px">{esc(t(SRC_LABEL[s]))}</span>' for s in src)
    st.html(f'<div style="margin:0 0 6px">{badges}</div>')


def render_notes(msg: dict):
    notes = []
    if msg.get("url"):
        notes.append(f'{esc(t("Fuente"))}: <a href="{esc(msg["url"])}" target="_blank" rel="noopener noreferrer">{esc(msg.get("url_title", "Wikipedia"))}</a>' + (f' · {esc(t("extracto de Wikipedia, sin reescribir (no hay IA disponible)"))}' if msg.get("extract_only") else f' · {esc(t("reformulado por"))} {esc(msg.get("ai_note", ""))}'))
    if "ia" in (msg.get("source") or []):
        notes.append(esc(t("Las respuestas generales de IA pueden contener errores; verifica los datos importantes.")) + (f' · {esc(msg.get("ai_note", ""))}' if msg.get("ai_note") else "") + (f' · {esc(t("la pregunta se envió a la nube"))}' if msg.get("cloud") else ""))
    if notes:
        st.html('<p class="syncnote">' + "<br>".join(notes) + "</p>")


def section_asistente():
    S = asst_state()
    st.html(ui.headline(t("Asistente Calibra"), t("Pregunta sobre tu negocio, la metodología de la app o cualquier tema. Los datos de negocio los calcula el motor y nunca una IA; los hechos generales vienen de Wikipedia con su enlace; y las preguntas abiertas las contesta una IA local o gratuita si la conectaste. Cada respuesta dice de dónde salió.")))
    ai_status_panel()
    insights_box(link=False)
    for i, msg in enumerate(S["msgs"]):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.write(msg["text"])
                continue
            render_sources(msg)
            st.write(msg["text"])
            for j, art in enumerate(msg.get("artifacts", [])):
                asst_render(art, f"{i}_{j}")
            render_notes(msg)
            if msg.get("suggest") and i == len(S["msgs"]) - 1:
                cols = st.columns(len(msg["suggest"]), gap="small")
                for k, q in enumerate(msg["suggest"]):
                    cols[k].button(q, key=f"as_sug_{i}_{k}", on_click=asst_ask, args=(q, msg.get("orig_q")), width="stretch")
            if msg.get("source") and i == len(S["msgs"]) - 1:
                fb = S.get("fb", {}).get(str(i))
                c = st.columns([1, 1, 6])
                c[0].button("👍", key=f"fb_up_{i}", on_click=asst_feedback, args=(msg.get("route"), (msg["source"] or ["?"])[-1], True, str(i)), disabled=fb is not None)
                c[1].button("👎", key=f"fb_dn_{i}", on_click=asst_feedback, args=(msg.get("route"), (msg["source"] or ["?"])[-1], False, str(i)), disabled=fb is not None)
                if fb is not None:
                    c[2].caption(t("Gracias: se guarda solo la ruta y la fuente, no tu pregunta."))
    if not S["msgs"]:
        st.html(ui.h2(t("Preguntas sugeridas")))
        cols = st.columns(3, gap="small")
        for i, q in enumerate(ASST_SUGGESTIONS):
            cols[i % 3].button(t(q), key=f"asst_sug_{i}", on_click=asst_ask, args=(t(q),), width="stretch")
    else:
        st.button(t("Nueva conversación"), key="asst_new", on_click=asst_reset)
    q = st.chat_input(t("Escribe tu pregunta"), key="asst_input")
    q = q or S.pop("pending", None)
    S["pending"] = None
    if q:
        with st.spinner(t("Pensando…")):
            asst_process(q)
        st.rerun()


ASST_SUGGESTIONS = ["¿Qué pasa si el cobre sube 10%?", "¿Por qué cambió el pronóstico del calibre 6?", "Prepara el resumen para la junta de S&OP", "¿Qué debo atender hoy?",
                    "¿Qué pasa si se exenta el arancel del T-MEC?", "¿Qué clientes compran menos?"]


# ================================================================== 8C: riesgo y decisiones
@st.cache_data(show_spinner=False)
def cached_vol(hist, fx_daily) -> dict:
    return RK.price_vol(hist, fx_daily)


@st.cache_data(show_spinner=False)
def cached_cal(monthly, gauges, h, _rows) -> dict:
    return RK.calibrate(_rows)


def risk_inputs(h: int):
    """Plan actual (con los precios y escenarios de este momento) y lo que necesita la simulación. Se recalcula con cada cambio; el plan viene de la caché."""
    prep = default_prep()
    m = prep.monthly
    ml = get_ml(m, E.windows_for(m))
    prices, mult = get_prices(), get_mult()
    res = cached_plan(m, tuple(prep.gauges), h, get_params(), prices, mult, get_sig(), get_w(), {"metal": sv("ov_metal", {}), "inv": sv("ov_inv", {}), "cap": sv("ov_cap", {})}, ml, get_crm(None, E.TON, prices))
    if res.rows.empty:
        return None
    tf = get_tariff()
    setup = RK.Setup(res.rows, res.params, mult, tf["current"], tf["elasticity"], h)
    hist = price_history()
    fxd = MD.load_daily("fx")
    vol = cached_vol(hist if hist is not None else pd.DataFrame(columns=["ym", "cu", "al"]), fxd)
    cal = cached_cal(m, tuple(prep.gauges), h, res.rows)
    return {"setup": setup, "res": res, "vol": vol, "cal": cal, "m": m, "gauges": list(prep.gauges)}


def risk_tariff() -> list:
    t_cur = float(sv("t_cur", TARIFF_CURRENT))
    return [(0.0, float(sv("rk_p0", 20))), (t_cur, float(sv("rk_p1", 60))), (min(60.0, t_cur + 10), float(sv("rk_p2", 20)))]


def risk_run(ctx, n: int, hedge=None):
    return RK.run(ctx["setup"], n=n, tariff=risk_tariff(), vol=ctx["vol"], hedge=hedge, inflate=ctx["cal"]["inflate"])


def risk_summary(h: int = 6, n: int = 5000) -> dict | None:
    """Cifras clave de riesgo (para la presentación y el asistente)."""
    ctx = risk_inputs(h)
    if ctx is None:
        return None
    thr = float(sv("rk_thr", 0.0)) * 1e6
    o = risk_run(ctx, n)
    s = RK.summarize(o["margin"], thr)
    p = ctx["setup"].params
    frac = float(sv("rk_hedge", 50)) / 100
    k = min(int(sv("rk_hm", h)), h)
    fx = {"cu": p.cu_lme, "al": p.al_lme}
    hed = RK.run(ctx["setup"], n=n, tariff=risk_tariff(), vol=ctx["vol"], hedge=(np.array([frac if i < k else 0.0 for i in range(h)]), fx, float(sv("rk_fee", 0.0))), inflate=ctx["cal"]["inflate"])
    sh = RK.summarize(hed["margin"], thr)
    return {"ctx": ctx, "o": o, "s": s, "hedged": sh, "frac": frac, "n": n, "thr": thr, "h": h}


def _vol_src(s: str) -> str:
    m = re.match(r"(\d+) meses de historia", s)
    return t("{n} meses de historia", n=m.group(1)) if m else t(s)


def section_riesgo():
    h = horizon_default()
    left, right = layout()
    with left:
        h = wid(st.radio, t("Horizonte"), "seg_h", 6, options=[3, 6, 12], horizontal=True, format_func=lambda x: t("{n} meses", n=x))
        n = wid(st.radio, t("Simulaciones"), "rk_n", 5000, options=[2000, 5000, 10000], horizontal=True, format_func=lambda x: f"{x:,}")
        thr = wid(st.number_input, t("Umbral de margen (millones de MXN)"), "rk_thr", 0.0, step=5.0)
        st.html(ui.group_title(t("Arancel (probabilidad de cada caso)")))
        wid(st.slider, t("Exención (0%)"), "rk_p0", 20, min_value=0, max_value=100, step=5, format="%d%%")
        wid(st.slider, t("Sigue el vigente"), "rk_p1", 60, min_value=0, max_value=100, step=5, format="%d%%")
        wid(st.slider, t("Sube 10 puntos"), "rk_p2", 20, min_value=0, max_value=100, step=5, format="%d%%")
        st.caption(t("Las tres se ajustan a 100%. Son supuestos tuyos; sin ventas de exportación definidas no mueven el margen."))
        st.html(ui.group_title(t("Compra de metal")))
        wid(st.slider, t("Meses que cubre la compra"), "rk_cov", 3, min_value=1, max_value=6, step=1)
        wid(st.slider, t("Recargo por comprar de urgencia"), "rk_rush", 8, min_value=0, max_value=30, step=1, format="%d%%")
        wid(st.slider, t("Costo anual de mantener inventario"), "pl_hold", 25, min_value=5, max_value=50, step=1, format="%d%%")
        st.html(ui.group_title(t("Cobertura")))
        wid(st.slider, t("Compras fijadas al precio de hoy"), "rk_hedge", 50, min_value=0, max_value=100, step=10, format="%d%%")
        wid(st.slider, t("Meses cubiertos"), "rk_hm", 6, min_value=1, max_value=12, step=1)
        wid(st.number_input, t("Costo de la cobertura (% del precio fijado)"), "rk_fee", 0.0, min_value=0.0, max_value=10.0, step=0.1)
    with right:
        ctx = risk_inputs(h)
        if ctx is None:
            st.html(f'<p class="hint">{esc(t("No hay un plan que simular con esta selección."))}</p>')
            return
        sm = risk_summary(h, int(n))
        s, o, setup, vol, cal = sm["s"], sm["o"], ctx["setup"], ctx["vol"], ctx["cal"]
        thr_m = thr * 1e6
        st.html(ui.headline(t("En {n} simulaciones, el margen de {h} meses queda entre {a} y {b} (P5 a P95).", n=f"{int(n):,}", h=h, a=ui.mxn(s["p5"]), b=ui.mxn(s["p95"])),
                            t("Hay {p}% de probabilidad de que el margen quede bajo {u}. Combina la demanda (errores de la validación cruzada), el cobre, el aluminio y el tipo de cambio (volatilidad histórica) y el arancel. Los precios y costos son ilustrativos.",
                              p=f"{s['prob_below'] * 100:.0f}", u=ui.mxn(thr_m))))
        base = RK.deterministic(setup)
        st.html(ui.kpis([(ui.mxn(s["mean"]), t("Margen esperado"), t("plan sin riesgo: {x}", x=ui.mxn(base["margin"]))), (ui.mxn(s["p5"]), t("Margen en el peor 5% (P5)")),
                         (ui.mxn(s["at_risk"]), t("Margen en riesgo (esperado menos P5)")), (f"{s['prob_below'] * 100:.0f}%", t("Probabilidad de margen bajo el umbral"), ui.mxn(thr_m))]))
        c = st.columns(2, gap="large")
        with c[0]:
            st.html(ui.h2(t("Distribución del margen"), t("Millones de MXN. Rayas: P5, esperado y P95.")))
            st.plotly_chart(ui.hist_fig(o["margin"] / 1e6, [(s["p5"] / 1e6, "P5", ui.FG), (s["mean"] / 1e6, t("Esperado"), ui.RED), (s["p95"] / 1e6, "P95", ui.MUTED)], height=300),
                            width="stretch", theme=None, config={"displayModeBar": False})
        with c[1]:
            sr = RK.summarize(o["revenue"])
            st.html(ui.h2(t("Distribución del ingreso"), t("Millones de MXN. P5 {a}, esperado {b}, P95 {c}.", a=f"{sr['p5'] / 1e6:,.0f}", b=f"{sr['mean'] / 1e6:,.0f}", c=f"{sr['p95'] / 1e6:,.0f}")))
            st.plotly_chart(ui.hist_fig(o["revenue"] / 1e6, [(sr["p5"] / 1e6, "P5", ui.FG), (sr["mean"] / 1e6, t("Esperado"), ui.RED), (sr["p95"] / 1e6, "P95", ui.MUTED)], height=300),
                            width="stretch", theme=None, config={"displayModeBar": False})
        st.html(f'<p class="syncnote">{esc(t("Margen en riesgo = margen esperado menos el percentil 5: lo que se puede perder en un mal escenario de cada 20. En el peor 5% el margen promedia {x}.", x=ui.mxn(s["cvar"])))}</p>')
        # volatilidad y supuestos
        with st.expander(t("Supuestos de la simulación")):
            rows = [[t("Cobre"), f"{vol['sigma']['cu'] * 100:.1f}%", _vol_src(vol["source"]["cu"])], [t("Aluminio"), f"{vol['sigma']['al'] * 100:.1f}%", _vol_src(vol["source"]["al"])], ["USD/MXN", f"{vol['sigma']['fx'] * 100:.1f}%", _vol_src(vol["source"]["fx"])]]
            st.html(ui.table([t("Serie"), t("Volatilidad mensual"), t("Origen")], rows, num=(1,)))
            st.html(f'<p class="syncnote">{esc(t("Los precios siguen una caminata aleatoria lognormal alrededor del camino de Metales (sin tendencia propia), con la correlación histórica entre ellos. La demanda es el pronóstico más errores tomados al azar de la validación cruzada de todos los calibres, ensanchados {c}x para que en validación {v}% de los meses reales caiga dentro de P10 a P90 (sin ensanchar: {r}%). Con tan pocas ventanas, las colas son una aproximación.", c=f"{cal['inflate']:.1f}", v=f"{cal['coverage']:.0f}", r=f"{cal['coverage_raw']:.0f}"))}</p>')
            if not any(setup.rows["is_exp"]):
                st.html(f'<p class="syncnote">{esc(t("Tus datos no tienen ventas de exportación definidas, así que el arancel no afecta esta simulación."))}</p>')
        risk_bands_block(ctx, int(n))
        risk_purchase_block(ctx, o)
        risk_hedge_block(ctx, sm)


def risk_bands_block(ctx, n: int):
    st.html(ui.h2(t("Pronóstico probabilístico"), t("P10, P50 y P90 por calibre: en 8 de cada 10 casos la demanda cae entre P10 y P90. Salen de simular la demanda con los errores de la validación cruzada.")))
    setup, cal = ctx["setup"], ctx["cal"]
    opts = ["Todos"] + [g for g in ctx["gauges"]]
    sel = wid(st.radio, t("Calibre"), "rk_gauge", "Todos", options=opts, horizontal=True, format_func=lambda g: t("Todos") if g == "Todos" else ui.cal_label(g))
    rows = setup.rows if sel == "Todos" else setup.rows[setup.rows["cal"] == sel]
    rng = np.random.default_rng(RK.SEED)
    pool = RK.pooled_residuals(setup.rows)
    draws = np.zeros((n, setup.H))
    fc = np.zeros(setup.H)
    for r in rows.itertuples():
        f = np.asarray(r.fc, float)
        base_fc = f / setup.mult0 if r.is_exp and setup.mult0 > 0 else f
        d = RK.simulate_demand(base_fc, pool, n, rng, inflate=cal["inflate"])
        draws += d * (setup.mult0 if r.is_exp else 1.0)
        fc += f
    b = RK.bands(draws)
    m = ctx["m"]
    hist = E.series(m, [sel] if sel != "Todos" else None)
    idx = pd.date_range(m["ym"].max() + pd.offsets.MonthBegin(1), periods=setup.H, freq="MS")
    st.plotly_chart(ui.forecast_fig(list(hist.index), hist.values, list(idx), b["p50"], b["p10"], b["p90"], ui.UNIT, "P50", height=330, compare=[(t("Pronóstico"), fc)]),
                    width="stretch", theme=None, config={"displayModeBar": False})
    tb = [[ui.mes(x), f"{b['p10'][i]:,.0f}", f"{b['p50'][i]:,.0f}", f"{b['p90'][i]:,.0f}", f"{fc[i]:,.0f}"] for i, x in enumerate(idx)]
    tb.append([Raw(f"<b>{esc(t('Total'))}</b>"), f"{b['tot_p10']:,.0f}", f"{b['tot_p50']:,.0f}", f"{b['tot_p90']:,.0f}", f"{fc.sum():,.0f}"])
    st.html(ui.table([t("Mes"), "P10", "P50", "P90", t("Pronóstico")], tb, num=(1, 2, 3, 4), main_rows=(len(tb) - 1,)))
    st.html(f'<p class="syncnote">{esc(t("P50 puede diferir del pronóstico porque en la validación los errores tuvieron un sesgo promedio. Con los errores sin ensanchar, {r}% de los meses reales caía dentro de P10 a P90; ensanchados, {v}%.", r=f"{cal['coverage_raw']:.0f}", v=f"{cal['coverage']:.0f}"))}</p>')


def risk_purchase_block(ctx, o):
    p = ctx["setup"].params
    st.html(ui.h2(t("Compra óptima de metal"), t("Modelo del vendedor de periódicos: compra hoy hasta el punto donde un faltante y un sobrante cuestan lo mismo en promedio.")))
    k = min(int(sv("rk_cov", 3)), ctx["setup"].H)
    rush, hold = float(sv("rk_rush", 8)), float(p.hold_pct)
    for key, nm, price in (("cu", t("Cobre"), p.cu_lme), ("al", t("Aluminio"), p.al_lme)):
        need = o["metal_need"][key][:, :k].sum(axis=1)
        if need.mean() <= 1e-6:
            continue
        pm = price * p.fx
        nv = RK.newsvendor(need, pm, rush, hold, k)
        extra = nv["cost_median"] - nv["cost_opt"]
        c = st.columns([1.1, 1.9], gap="large")
        with c[0]:
            st.html(f'<h3 style="margin:10px 0 4px;font-size:1.1em">{esc(t("{m}: comprar hoy {q} t", m=nm, q=f"{nv['q_star']:,.0f}"))}</h3>'
                    f'<p class="hint">{esc(t("La necesidad de los próximos {k} meses va de {a} a {b} t (mediana {c} t). Un faltante cuesta {u} por tonelada (compra urgente) y un sobrante {v} (mantener inventario {k} meses), así que conviene cubrir hasta el percentil {p}: {q} t. Comprar solo la mediana costaría en promedio {x} más.",
                                                    k=k, a=f"{nv['p10']:,.0f}", b=f"{nv['p90']:,.0f}", c=f"{nv['median']:,.0f}", u=ui.mxn(nv["under_cost"]), v=ui.mxn(nv["over_cost"]), p=f"{nv['critical_ratio'] * 100:.0f}", q=f"{nv['q_star']:,.0f}", x=ui.mxn(extra)))}</p>')
        with c[1]:
            st.plotly_chart(ui.line_fig(nv["curve_q"], nv["curve_cost"] / 1e6, [(nv["q_star"], t("Óptimo"), ui.RED), (nv["median"], t("Mediana"), ui.MUTED)], height=240, xtitle=f"{t('Toneladas compradas hoy')}", ytitle=t("Costo esperado (M MXN)"), yfmt=",.2f"),
                            width="stretch", theme=None, config={"displayModeBar": False})
    st.html(f'<p class="syncnote">{esc(t("Faltante = recargo de comprar de urgencia (ajustable). Sobrante = costo anual de mantener inventario por los meses que cubre la compra. La necesidad es la producción del plan más la desviación de la demanda simulada. No incluye descuentos por volumen ni coberturas: es una guía de cantidad, no de momento (para el momento, ve Metales).")) }</p>')


def risk_hedge_block(ctx, sm):
    setup = ctx["setup"]
    p, h = setup.params, setup.H
    st.html(ui.h2(t("Cobertura del precio del metal"), t("Cuánto riesgo de margen se quita si una parte de las compras de los próximos meses se fija hoy al precio actual.")))
    k = min(int(sv("rk_hm", h)), h)
    fee = float(sv("rk_fee", 0.0))
    fixed = {"cu": p.cu_lme, "al": p.al_lme}
    ratios = tuple(i / 10 for i in range(11))
    hs = RK.hedge_sweep(setup, ratios, k, fixed, fee, n=sm["n"], tariff=risk_tariff(), vol=ctx["vol"], inflate=ctx["cal"]["inflate"])
    r0 = float(hs.loc[hs["cobertura"] == 0, "en_riesgo"].iloc[0])
    sel = float(sm["frac"])
    row = hs.iloc[(hs["cobertura"] - sel).abs().argmin()]
    best = hs.loc[hs["en_riesgo"].idxmin()]
    cut = (1 - row["en_riesgo"] / r0) * 100 if r0 > 0 else 0.0
    if cut >= 0:
        head = t("Fijar {x}% de las compras de los próximos {k} meses baja el margen en riesgo de {a} a {b} ({c}% menos).", x=f"{sel * 100:.0f}", k=k, a=ui.mxn(r0), b=ui.mxn(row["en_riesgo"]), c=f"{cut:.0f}")
    else:
        head = t("Fijar {x}% de las compras de los próximos {k} meses sube el margen en riesgo de {a} a {b} ({c}% más): es más cobertura de la que conviene.", x=f"{sel * 100:.0f}", k=k, a=ui.mxn(r0), b=ui.mxn(row["en_riesgo"]), c=f"{abs(cut):.0f}")
    st.html(f'<p class="headline" style="font-size:clamp(18px,2vw,24px);max-width:64ch">{esc(head)}</p>')
    st.html(ui.kpis([(ui.mxn(r0), t("Margen en riesgo sin cobertura")), (ui.mxn(row["en_riesgo"]), t("Margen en riesgo con {x}%", x=f"{sel * 100:.0f}")),
                     (f"{best['cobertura'] * 100:.0f}%", t("Cobertura con menos riesgo"), t("riesgo {x}", x=ui.mxn(best["en_riesgo"]))),
                     (ui.mxn(row["media"] - float(hs.loc[hs["cobertura"] == 0, "media"].iloc[0])), t("Cambio del margen esperado"), t("por el costo de la cobertura"))]))
    st.plotly_chart(ui.line_fig(hs["cobertura"] * 100, hs["en_riesgo"] / 1e6, [(sel * 100, t("Tu cobertura"), ui.RED), (best["cobertura"] * 100, t("Mínimo riesgo"), ui.MUTED)], height=280, xtitle=t("% de las compras fijado hoy"), ytitle=t("Margen en riesgo (M MXN)"), yfmt=",.0f"),
                    width="stretch", theme=None, config={"displayModeBar": False})
    st.html(f'<p class="syncnote">{esc(t("Cubrir todo no es lo mejor: con un traspaso de {pt}% del metal al precio de venta, ya trasladas esa parte del riesgo a tus clientes, y la cobertura que menos riesgo deja ronda {o}%. Se fija el precio de compra del metal (en USD); el tipo de cambio sigue flotando. Es una simulación, no una oferta de cobertura.", pt=f"{p.pass_through:.0f}", o=f"{max(0, 100 - p.pass_through):.0f}"))}</p>')


# ================================================================== Personalizar, guía rápida y pie
def open_drawer():
    st.session_state["drawer"] = True


def close_drawer():
    st.session_state["drawer"] = False


def toggle_lang():
    """Switch the dashboard language between Spanish and English."""
    new_lang = "en" if ST.cfg()["lang"] == "es" else "es"
    ST.put("lang", new_lang)
    st.session_state["seg_lang"] = new_lang


def pref_radio(label, pref, labels: dict):
    """Botones segmentados que guardan su valor en las preferencias."""
    key = f"seg_{pref}"
    if key not in st.session_state:
        st.session_state[key] = ST.cfg()[pref]
    st.radio(label, list(labels), key=key, horizontal=True, format_func=lambda v: labels[v], on_change=lambda: ST.put(pref, st.session_state[key]))


SWATCHES = {"red": ("#ff0000", "Rojo Viakable"), "blue": ("#48b3f6", "Azul innovación"), "black": ("#000000", "Negro"), "gray": ("#8a8a8a", "Gris")}


def drawer():
    cfg = ST.cfg()
    with st.container(key="scrim"):
        st.button("x", key="scrim_btn", on_click=close_drawer)
    with st.container(key="drawer"):
        c1, c2 = st.columns([3, 1], vertical_alignment="center")
        c1.html(f"<h2>{esc(t('Personalizar'))}</h2>")
        c2.button(t("Cerrar"), key="drawer_close", on_click=close_drawer)
        pref_radio(t("Idioma"), "lang", {"es": "Español", "en": "English"})
        pref_radio(t("Tema"), "theme", {"light": t("Claro"), "dark": t("Oscuro"), "system": t("Sistema")})

        st.html(f'<div class="lab" style="font-weight:700;font-size:.93rem">{esc(t("Color de acento"))}</div>')
        cols = st.columns([1, 1, 1, 1, 3], gap="small", vertical_alignment="center")

        def _pick(color):
            ST.put("accent", color)
            st.session_state["accent_custom"] = color
        for c, (name, (color, title)) in zip(cols, SWATCHES.items()):
            with c, st.container(key=f"sw_{name}" + ("_on" if cfg["accent"].lower() == color else "")):
                st.button(" ", key=f"swb_{name}", help=t(title), on_click=_pick, args=(color,))
        if "accent_custom" not in st.session_state:
            st.session_state["accent_custom"] = cfg["accent"]
        cols[4].color_picker(t("Otro"), key="accent_custom", on_change=lambda: ST.put("accent", st.session_state["accent_custom"]))
        st.caption(t("El rojo es el color institucional de Viakable."))

        fonts = {"brand": "Aeonik / Aptos", "manrope": "Manrope", "plex": "IBM Plex Sans", "system": t("Del sistema")}
        if "sel_font" not in st.session_state:
            st.session_state["sel_font"] = cfg["font"]
        st.selectbox(t("Tipografía"), list(fonts), format_func=fonts.get, key="sel_font", on_change=lambda: ST.put("font", st.session_state["sel_font"]))
        pref_radio(t("Densidad"), "density", {"comfy": t("Cómoda"), "compact": t("Compacta")})
        pref_radio(t("Animaciones"), "anim", {"on": t("Sí"), "off": t("No")})
        st.caption(t("Si tu sistema pide reducir el movimiento, la app no anima aunque estén activadas."))
        if "in_unit" not in st.session_state:
            st.session_state["in_unit"] = cfg["unit"]
        st.text_input(t("Unidad de medida"), key="in_unit", max_chars=12, on_change=lambda: ST.put("unit", st.session_state["in_unit"].strip() or "t"))
        st.caption(t("Ej. t, km, kg, piezas."))
        pref_radio(t("Ver como"), "role", {r: t(r) for r in CRM.ROLES})
        if "in_user" not in st.session_state:
            st.session_state["in_user"] = cfg.get("user", "")
        st.text_input(t("Tu nombre"), key="in_user", max_chars=40, on_change=lambda: ST.put("user", st.session_state["in_user"].strip()))
        st.caption(t("El rol limita lo que puedes cambiar y queda en el registro. No es seguridad formal: cualquiera puede cambiarlo aquí."))
        if ST.ON_CLOUD:
            st.caption(t("En la versión publicada, estas preferencias duran mientras la página siga abierta."))
        else:
            st.caption(t("Se guardan en el archivo ajustes.json de esta carpeta."))


# (sección, elemento resaltado, título, texto). Los pasos de Clientes e Inicio aparecen cuando esas secciones existan.
TOUR = [
    ("Datos", '.st-key-seg_src,[data-testid="stFileUploader"]', "Sube tus ventas",
     "Arrastra aquí el Excel del histórico de venta. La app detecta las columnas, incluso si viene exportado de SAP."),
    ("Pronóstico", ".st-key-seg_meas", "Toneladas o pesos", "Elige si quieres ver el pronóstico en toneladas o en ventas en pesos."),
    ("Pronóstico", ".st-key-chips_cal,.st-key-seg_mkt", "Elige calibres y mercado",
     "Selecciona uno o varios calibres, o el botón 16 + 18. Si exportas, filtra Nacional o EE. UU."),
    ("Pronóstico", ".st-key-result", "Lee el resultado",
     "Aquí ves la venta pronosticada, el total del horizonte y qué tan certero es el pronóstico. La app elige sola el método más certero."),
    ("Planeación", ".st-key-nav", "Convierte en un plan",
     "Planeación traduce el pronóstico en inventario de seguridad, compras de cobre y aluminio, margen y uso de capacidad."),
    ("Metales", ".ticker", "Precios en vivo",
     "Aquí ves el cobre, el aluminio y el tipo de cambio. Se actualizan solos cada 15 minutos y todo el pronóstico, el plan y el margen se recalculan con ellos. Si falla internet, se usa el último precio guardado."),
    ("Clientes", ".st-key-nav", "Da seguimiento a clientes",
     "Registra clientes, oportunidades y actividades. Las oportunidades abiertas alimentan el pronóstico."),
    ("Noticias", ".st-key-news_refresh", "Noticias y regulación",
     "La app busca cada 30 minutos aranceles, T-MEC, normas y precios de metales, y los clasifica por relevancia. Tú decides qué hacer: ninguna noticia cambia el pronóstico sola."),
    ("Inicio", ".st-key-alerts", "Qué atender",
     "Cada vez que entres, aquí aparecen las alertas importantes: calibres que caen, capacidad excedida, clientes que compran menos o tareas vencidas."),
    ("Pronóstico", ".st-key-seg_ui_mode", "Simple o avanzado",
     "El modo simple muestra lo esencial. Cambia a avanzado para ver métodos, consenso, machine learning, señales, escenarios y permisos del equipo. En Personalizar cambias idioma, tema, color, tipografía, densidad y unidad. Puedes abrir esta guía otra vez cuando quieras."),
]


def tour_steps():
    return [s for s in TOUR if s[0] in SECTIONS]


def tour_go(delta: int):
    steps = tour_steps()
    i = (st.session_state.get("tour") or 0) + delta
    if i >= len(steps):
        tour_close()
        return
    i = max(0, i)
    st.session_state["tour"] = i
    st.session_state["nav"] = steps[i][0]


def tour_close():
    st.session_state["tour"] = None
    ST.put("tour_done", True)


def tour_start():
    st.session_state["tour"] = -1
    tour_go(1)


def tour_card():
    steps = tour_steps()
    i = st.session_state.get("tour")
    if i is None or not (0 <= i < len(steps)):
        return
    _, _, title, text = steps[i]
    last = i == len(steps) - 1
    with st.container(key="tour"):
        st.html(f'<span class="tour-step">{esc(t("Paso {a} de {b}", a=i + 1, b=len(steps)))}</span><h3>{esc(t(title))}</h3><p>{esc(t(text))}</p>')
        c = st.columns([1, 1, 1])
        if i > 0:
            c[0].button(t("Anterior"), key="tour_prev", on_click=tour_go, args=(-1,))
        c[1].button(t("Terminar") if last else t("Siguiente"), key="tour_next", type="primary", on_click=tour_go, args=(1,))
        c[2].button(t("Cerrar"), key="tour_close", on_click=tour_close)


def do_reset():
    """Restablece las preferencias y vuelve a los datos iniciales (se pierden los archivos subidos en esta sesión)."""
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    ST.reset()


# ================================================================== navegación
SECTIONS = {"Inicio": section_inicio, "Pronóstico": section_pronostico, "Planeación": section_planeacion, "Metales": section_metales, "Riesgo": section_riesgo, "Clientes": section_clientes,
            "Portafolio": section_portafolio, "Asistente": section_asistente, "Oportunidades": section_oportunidades, "Señales": section_senales, "Noticias": section_noticias, "Laboratorio": section_laboratorio, "Datos": section_datos, "Equipo": section_equipo}
ADV_ONLY = {"Señales", "Laboratorio", "Equipo", "Riesgo"}


def main():
    sync_store()
    try:
        apply_live()
    except Exception:                       # un fallo de red o de la base nunca debe romper la app
        pass
    cfg = ST.cfg()
    set_lang(cfg["lang"])
    try:
        ui.REDUCED_MOTION = st.context.cookies.get("vk_rm") == "1"
    except Exception:
        ui.REDUCED_MOTION = False
    sys_dark = getattr(st.context.theme, "type", None) == "dark"
    dark = ui.configure(cfg, sys_dark)
    set_unit(cfg["unit"])
    flip = cfg["theme"] != "system" and dark != sys_dark
    steps = tour_steps()
    if "tour" not in st.session_state:          # la guía se abre sola la primera vez
        st.session_state["tour"] = 0 if not cfg["tour_done"] else None
        if st.session_state["tour"] == 0:
            st.session_state["nav"] = steps[0][0]
    ti = st.session_state.get("tour")
    hl = steps[ti][1] if ti is not None and 0 <= ti < len(steps) else ""
    st.markdown(ui.css(cfg, hl, flip), unsafe_allow_html=True)

    if st.session_state.get("present"):                 # modo presentación: encabezado mínimo y una escena por paso
        with st.container(key="top"):
            st.html(ui.header_html())
        st.html(ui.connector_html())
        st.html(ui.engine_html(ui.ANIM), unsafe_allow_javascript=True)
        fresh_slot = st.empty()
        ticker()
        pres_stage()
        pres_clock()
        fresh_slot.html(freshness_html())
        with st.container(key="footer"):
            st.html(esc(t(FOOTER)))
        return

    nav_opts = [s for s in SECTIONS if s not in ADV_ONLY or adv()]
    if st.session_state.get("nav") not in nav_opts + [None]:
        st.session_state["nav"] = nav_opts[0]
    if "seg_ui_mode" not in st.session_state:
        st.session_state["seg_ui_mode"] = cfg["mode"]
    with st.container(key="top"):
        c = st.columns([3.4, 1.5, 0.95, 1.05, 1.3], vertical_alignment="center")
        c[0].html(ui.header_html())
        c[1].radio(t("Modo"), ["simple", "adv"], key="seg_ui_mode", horizontal=True, label_visibility="collapsed",
                   format_func=lambda v: t("Simple") if v == "simple" else t("Avanzado"), on_change=lambda: ST.put("mode", st.session_state["seg_ui_mode"]))
        c[2].button(t("Guía rápida"), key="btn_tour", on_click=tour_start)
        c[3].button(t("Presentar"), key="btn_present", on_click=pres_start)
        _lang_label = "Translate to English" if cfg["lang"] == "es" else "Traducir a Español"
        c[4].button(_lang_label, key="btn_lang", on_click=toggle_lang)
        choice = wid(st.radio, t("Sección"), "nav", nav_opts[0], options=nav_opts, horizontal=True, label_visibility="collapsed", format_func=t)
    st.html(ui.connector_html())
    st.html(ui.engine_html(ui.ANIM), unsafe_allow_javascript=True)
    fresh_slot = st.empty()
    ticker()
    st.session_state.setdefault("_news_seen", NEWS.meta_get("last_fetch"))
    news_timer()
    SECTIONS[choice]()
    fresh_slot.html(freshness_html())
    with st.container(key="footer"):
        f1, f2 = st.columns([4, 1.4], vertical_alignment="center")
        f1.html(esc(t(FOOTER)))
        f2.button(t("Restablecer ajustes y ventas"), key="reset_btn", on_click=do_reset)
    if st.session_state.get("drawer"):
        drawer()
    tour_card()


main()
