"""Textos de la interfaz en español (México) e inglés. La llave es el texto en español; el inglés está en EN.

t("Esperamos {x}", x=5) busca la traducción, y si no existe usa el texto en español.
"""
import re

LANG = "es"


def set_lang(lang: str) -> None:
    global LANG
    LANG = lang if lang in ("es", "en") else "es"


UNIT = "t"
_UNIT_RE = re.compile(r"(?<=[\d\)]) t(?![\w])|\(t\)|^t por mes$|^t per month$")


def set_unit(unit: str) -> None:
    global UNIT
    UNIT = (unit or "t").strip() or "t"


def t(es: str, **kw) -> str:
    """Traduce y da formato. La unidad de medida elegida en Personalizar reemplaza la 't' de toneladas que sigue a un número o va entre paréntesis."""
    s = EN.get(es, es) if LANG == "en" else es
    s = s.format(**kw) if kw else s
    if UNIT != "t":
        s = _UNIT_RE.sub(lambda m: f" {UNIT}" if m.group(0).startswith(" ") else (f"({UNIT})" if m.group(0) == "(t)" else m.group(0).replace("t", UNIT, 1)), s)
    return s


EN: dict[str, str] = {}
try:                                    # las traducciones viven en i18n_en.py para mantener este archivo corto
    from i18n_en import EN as _EN
    EN.update(_EN)
    from i18n_en_crm import EN as _EN2
    EN.update({k: v for k, v in _EN2.items() if k not in _EN})
    from i18n_en_6f import EN as _EN3
    EN.update({k: v for k, v in _EN3.items() if k not in EN})
    from i18n_en_7 import EN as _EN4
    EN.update({k: v for k, v in _EN4.items() if k not in EN})
    from i18n_en_8b import EN as _EN5
    EN.update({k: v for k, v in _EN5.items() if k not in EN})
    from i18n_en_8c import EN as _EN6
    EN.update({k: v for k, v in _EN6.items() if k not in EN})
    from i18n_en_asst import EN as _EN7
    EN.update({k: v for k, v in _EN7.items() if k not in EN})
    from i18n_en_hybrid import EN as _EN8
    EN.update({k: v for k, v in _EN8.items() if k not in EN})
except ImportError:
    pass
