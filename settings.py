"""Preferencias de la interfaz (idioma, tema, color, tipografía, densidad, unidad, modo). Se guardan en ajustes.json.

En Streamlit Community Cloud el disco es compartido y temporal, así que ahí las preferencias viven solo en la sesión.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

PATH = Path(os.environ.get("VIAKABLE_AJUSTES") or Path(__file__).parent / "ajustes.json")   # la variable permite probar sin tocar tus ajustes
ON_CLOUD = str(Path(__file__).resolve()).startswith("/mount/")

DEFAULTS = {"lang": "es", "theme": "system", "accent": "#ff0000", "font": "brand", "density": "comfy", "unit": "t",
            "mode": "simple", "anim": "on", "asst_wiki": "on", "asst_ai": "on", "tour_done": False, "role": "Administrador", "user": ""}
CHOICES = {"lang": ("es", "en"), "theme": ("light", "dark", "system"), "font": ("brand", "manrope", "plex", "system"),
           "density": ("comfy", "compact"), "anim": ("on", "off"), "asst_wiki": ("on", "off"), "asst_ai": ("on", "off"), "mode": ("simple", "adv"), "role": ("Administrador", "Planeación", "Comercial", "Consulta")}


def _clean(raw: dict) -> dict:
    """Mezcla lo guardado con los valores por omisión y descarta lo inválido."""
    out = dict(DEFAULTS)
    for k, v in (raw or {}).items():
        if k not in DEFAULTS:
            continue
        if k in CHOICES and v not in CHOICES[k]:
            continue
        if k == "accent" and not (isinstance(v, str) and len(v) == 7 and v.startswith("#") and all(c in "0123456789abcdefABCDEF" for c in v[1:])):
            continue
        if k == "unit":
            v = str(v).strip()[:12] or "t"
        if k == "tour_done":
            v = bool(v)
        if k == "user":
            v = str(v).strip()[:40]
        out[k] = v
    return out


def load() -> dict:
    if ON_CLOUD or not PATH.exists():
        return dict(DEFAULTS)
    try:
        return _clean(json.loads(PATH.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return dict(DEFAULTS)


def save(cfg: dict) -> None:
    if ON_CLOUD:
        return
    try:
        PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def cfg() -> dict:
    """Preferencias de esta sesión (se leen del archivo una sola vez)."""
    if "cfg" not in st.session_state:
        st.session_state["cfg"] = load()
    return st.session_state["cfg"]


def put(key: str, value) -> None:
    c = cfg()
    c[key] = value
    save(_clean(c))
    st.session_state["cfg"] = _clean(c)


def reset() -> None:
    st.session_state["cfg"] = dict(DEFAULTS)
    save(dict(DEFAULTS))
