"""Estilo y componentes visuales de la app (basados en referencia/viakable-demand-sensing.html).

Solo presentación: aquí no hay cálculos de pronóstico ni de planeación.
"""
from __future__ import annotations

import html as _h
import re
import textwrap

import numpy as np
import plotly.graph_objects as go

from pathlib import Path

from i18n import t
from viakable_forecast.config import MESES_ES

MESES_EN = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
GRAY, BLUE = "#cbcbcb", "#48b3f6"
PAL_L = {"bg": "#ffffff", "fg": "#000000", "muted": "#5c5c5c", "line": "#cbcbcb", "soft": "#f4f4f4", "inverse-bg": "#000000", "inverse-fg": "#ffffff"}
PAL_D = {"bg": "#000000", "fg": "#ffffff", "muted": "#a8a8a8", "line": "#3a3a3a", "soft": "#151515", "inverse-bg": "#ffffff", "inverse-fg": "#000000"}
FONTS = {"brand": '"Aeonik","Aptos","Manrope",system-ui,-apple-system,"Segoe UI",sans-serif',
         "manrope": '"Manrope",system-ui,sans-serif', "plex": '"IBM Plex Sans",system-ui,sans-serif',
         "system": 'system-ui,-apple-system,"Segoe UI",Roboto,sans-serif'}
# Valores vigentes (los fija configure() en cada ejecución; las gráficas los leen al dibujarse)
RED, FG, BG, MUTED, LINE, SOFT, FONT, UNIT = "#ff0000", "#000000", "#ffffff", "#5c5c5c", "#cbcbcb", "#f4f4f4", FONTS["brand"], "t"
LANG_EN = False
INV_FG = "#ffffff"


DARK = False                # tema efectivo (lo fija configure)
ANIM = True                 # transiciones de las gráficas y animaciones de números (se apaga en Personalizar o con 'reducir movimiento')
REDUCED_MOTION = False      # lo fija la app a partir de la preferencia del sistema (cookie que escribe el motor de animación)

_INJECT_CSS = ""            # CSS completo; lo fija css() y lo inyecta engine_html() en el <head> del documento padre


def configure(cfg: dict, system_dark: bool = False) -> bool:
    """Aplica las preferencias a las gráficas. Devuelve True si el tema efectivo es oscuro."""
    global RED, FG, BG, MUTED, LINE, SOFT, FONT, UNIT, LANG_EN, INV_FG, ANIM, DARK
    dark = cfg["theme"] == "dark" or (cfg["theme"] == "system" and system_dark)
    p = PAL_D if dark else PAL_L
    RED, FG, BG, MUTED, LINE, SOFT, INV_FG = cfg["accent"], p["fg"], p["bg"], p["muted"], p["line"], p["soft"], p["inverse-fg"]
    FONT, UNIT, LANG_EN = FONTS.get(cfg["font"], FONTS["brand"]), cfg["unit"], cfg["lang"] == "en"
    ANIM = cfg.get("anim", "on") == "on" and not REDUCED_MOTION
    DARK = dark
    return dark


class Raw(str):
    """Texto que ya es HTML seguro (no se escapa)."""


esc = lambda s: _h.escape(str(s if s is not None else ""), quote=True)


# ------------------------------------------------------------------ CSS



ENGINE_JS = """<script>
(function(){
  if (window.__vkEngine) return; window.__vkEngine = true;
  var RM = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : {matches:false};
  try { document.cookie = 'vk_rm=' + (RM.matches ? 1 : 0) + ';path=/;max-age=31536000;SameSite=Lax'; } catch(e) {}
  var last = {}, lastPx = {};
  function on(){ var c = document.getElementById('vk-cfg'); return !RM.matches && !(c && c.dataset.anim === 'off'); }
  function fmt(v, d){ return v.toLocaleString('en-US', {minimumFractionDigits:d, maximumFractionDigits:d}); }
  function show(el, v){
    var d = +(el.dataset.dec || 0), s = (v < 0 ? '-' : (el.dataset.plus && v > 0 ? '+' : ''));
    el.textContent = s + (el.dataset.pre || '') + fmt(Math.abs(v), d) + (el.dataset.sp ? ' ' : '') + (el.dataset.suf || '');
  }
  function delta(el, from, to){
    var k = el.nextElementSibling; if (!k || !k.classList.contains('kd')) return;
    var d = to - from, dec = +(el.dataset.dec || 0);
    if (Math.abs(d) < Math.pow(10, -dec) / 2) return;
    k.className = 'kd ' + (d > 0 ? 'up' : 'down') + ' show';
    k.textContent = (d > 0 ? '+' : '-') + fmt(Math.abs(d), dec) + (el.dataset.suf || '').trim();
    clearTimeout(k.__t); k.__t = setTimeout(function(){ k.classList.remove('show'); }, 6000);
  }
  function count(el){
    var to = parseFloat(el.dataset.count), key = el.dataset.k;
    if (el.__v === el.dataset.count) return;
    el.__v = el.dataset.count;
    var from = last[key]; last[key] = to;
    if (from === undefined || from === to) { show(el, to); return; }
    delta(el, from, to);
    if (!on()) { show(el, to); return; }
    var t0 = null, dur = 700;
    if (el.__raf) cancelAnimationFrame(el.__raf);
    (function step(ts){
      if (t0 === null) t0 = ts;
      var p = Math.min(1, (ts - t0) / dur), e = 1 - Math.pow(1 - p, 3);
      show(el, from + (to - from) * e);
      if (p < 1) el.__raf = requestAnimationFrame(step); else show(el, to);
    })(performance.now());
  }
  function norm(c){ var d = document.createElement('div'); d.style.color = c; document.body.appendChild(d); var r = getComputedStyle(d).color; d.remove(); return r; }
  function draw(){
    var accent = norm(getComputedStyle(document.documentElement).getPropertyValue('--accent') || '#ff0000');
    document.querySelectorAll('.js-plotly-plot').forEach(function(gd){
      if (gd.__drawn) return;
      var lines = gd.querySelectorAll('.scatterlayer .trace path.js-line');
      var hit = null;
      lines.forEach(function(p){ if (!hit && getComputedStyle(p).stroke === accent) hit = p; });
      if (!hit) return;
      gd.__drawn = true;
      if (!on()) return;
      hit.setAttribute('pathLength', '1'); hit.classList.add('vk-draw');
      gd.classList.add('vk-first'); setTimeout(function(){ gd.classList.remove('vk-first'); }, 2600);
    });
  }
  function ticker(){
    document.querySelectorAll('[data-px]').forEach(function(el){
      var k = el.dataset.px, v = parseFloat(el.dataset.v);
      if (lastPx[k] !== undefined && lastPx[k] !== v && on()) {
        el.classList.remove('flash-up', 'flash-down'); void el.offsetWidth;
        el.classList.add(v > lastPx[k] ? 'flash-up' : 'flash-down');
      }
      lastPx[k] = v;
    });
  }
  var pend = false;
  function scan(){ pend = false; document.querySelectorAll('[data-count]').forEach(count); ticker(); draw(); }
  new MutationObserver(function(){ if (!pend) { pend = true; requestAnimationFrame(scan); } }).observe(document.body, {childList:true, subtree:true});
  scan();
})();
</script>"""


def engine_html(anim_on: bool) -> str:
    """Motor de animación: números que cuentan, destello de precios, trazo del pronóstico.
    También inyecta <link> de fuentes y CSS en el <head> del documento padre via JS."""
    import json as _json
    # All JS must target window.parent.document — engine_html() is injected via
    # st.html() which sandboxes the script inside an iframe; document alone
    # refers to the iframe, not the Streamlit page.
    font_js = (
        "(function(){"
        "var pd=window.parent?window.parent.document:document;"
        "var fonts=["
        "'https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700;800&display=swap',"
        "'https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap'"
        "];"
        "fonts.forEach(function(href){"
        "if(pd.querySelector('link[href=\"'+href+'\"]'))return;"
        "var l=pd.createElement('link');l.rel='stylesheet';l.href=href;"
        "(pd.head||pd.documentElement).appendChild(l);"
        "});"
        "})();"
    )
    css_js = ""
    if _INJECT_CSS:
        css_js = (
            "(function(){"
            "var pd=window.parent?window.parent.document:document;"
            "var s=pd.getElementById('vk-css');"
            "if(!s){s=pd.createElement('style');s.id='vk-css';"
            "(pd.head||pd.documentElement).appendChild(s);}"
            f"s.textContent={_json.dumps(_INJECT_CSS)};"
            "})();"
        )
    hide_icons_js = (
        "(function(){"
        "var pd=window.parent?window.parent.document:document;"
        "var ICON_TEXTS=['_arrow_right','arrow_right','arrow_down','arrow_drop_down',"
        "  'expand_more','chevron_right','chevron_down','slideshow','settings',"
        "  'keyboard_arrow_down','keyboard_arrow_right','expand_less'];"
        "function hide(el){el.style.cssText='display:none!important;width:0!important;overflow:hidden!important;';}"
        "function isPlotly(n){var p=n;for(var i=0;i<6;i++){if(!p)break;if(p.classList&&(p.classList.contains('js-plotly-plot')||p.classList.contains('plotly')))return true;p=p.parentElement;}return false;}"
        "function scan(root){"
        "  if(isPlotly(root))return;"
        "  var sel=root.querySelectorAll?root:pd;"
        "  sel.querySelectorAll('[class*=\"material-symbols\"],[class*=\"material-icon\"],[data-testid=\"stExpanderToggleIcon\"]').forEach(function(el){if(!isPlotly(el))hide(el);});"
        "  sel.querySelectorAll('span,i,p').forEach(function(el){"
        "    if(!isPlotly(el)&&el.children.length===0&&ICON_TEXTS.indexOf((el.textContent||'').trim())!==-1)hide(el);"
        "  });"
        "}"
        "scan(pd);"
        "var _t=null;"
        "new MutationObserver(function(muts){"
        "  clearTimeout(_t);"
        "  _t=setTimeout(function(){"
        "    muts.forEach(function(m){m.addedNodes.forEach(function(n){if(n.nodeType===1)scan(n);});});"
        "  },50);"
        "}).observe(pd.body||pd.documentElement,{childList:true,subtree:true});"
        "})();"
    )
    js_body = ENGINE_JS.strip().removeprefix("<script>").removesuffix("</script>").strip()
    return (f'<div id="vk-cfg" data-anim="{"on" if anim_on else "off"}" style="display:none"></div>'
            f'<script>{font_js}{css_js}{hide_icons_js}{js_body}</script>')




ASSETS = Path(__file__).parent / "assets"
_CACHE: dict = {}


def _asset(name: str) -> str:
    """Contenido de assets/<name>, releído solo si el archivo cambió."""
    p = ASSETS / name
    key = (name, p.stat().st_mtime if p.exists() else 0)
    if key not in _CACHE:
        _CACHE[key] = p.read_text(encoding="utf-8") if p.exists() else ""
    return _CACHE[key]


def css(cfg: dict, hl: str = "", flip_tables: bool = False) -> str:
    """Todo el CSS en un solo bloque <style> (se inyecta con st.html, no con st.markdown): variables del tema, acento, tipografía y densidad,
    la hoja assets/styles.css y (opcional) el resaltado de la guía rápida."""
    global _INJECT_CSS
    def block(p):
        return ";".join(f"--{k}:{v}" for k, v in p.items())
    if cfg["theme"] == "light":
        root = f":root{{{block(PAL_L)}}}"
    elif cfg["theme"] == "dark":
        root = f":root{{{block(PAL_D)}}}"
    else:
        root = f":root{{{block(PAL_L)}}}@media (prefers-color-scheme:dark){{:root{{{block(PAL_D)}}}}}"
    comp = cfg["density"] == "compact"
    dyn = (f"{root}:root{{--red:#ff0000;--gray:#cbcbcb;--blue:#48b3f6;--accent:{cfg['accent']};--font:{FONTS.get(cfg['font'], FONTS['brand'])};"
           f"--base:{'13.5px' if comp else '15px'};--pad:{0.7 if comp else 1}}}")
    static = _asset("styles.css")
    imports = "".join(ln + "\n" for ln in static.split("\n") if ln.startswith("@import"))      # @import debe ir primero
    body = "\n".join(ln for ln in static.split("\n") if not ln.startswith("@import"))
    extra = ""
    if hl:
        extra += f"{hl}{{outline:3px solid var(--accent)!important;outline-offset:4px}}"
    if flip_tables:      # las tablas editables de Streamlit usan el tema del navegador; si no coincide con el elegido, se invierten
        extra += '[data-testid="stDataFrame"]{filter:invert(1) hue-rotate(180deg)}'
    full_css = f"{imports}{dyn}\n{body}\n{extra}"
    _INJECT_CSS = full_css   # guardado para que engine_html() lo inyecte en el <head> del padre
    return f"<style>{full_css}</style>"


# ------------------------------------------------------------------ bloques HTML
def _logo_uri(name: str) -> str | None:
    p = ASSETS / name
    if not p.exists():
        return None
    key = ("logo", name, p.stat().st_mtime)
    if key not in _CACHE:
        import base64
        _CACHE[key] = "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
    return _CACHE[key]


def header_html() -> str:
    """Logotipo: logo de Viakable + línea gris vertical + "calibra" en minúsculas, y debajo la descripción. Si no hay assets/viakable-logo.png se escribe el nombre como texto.
    En tema oscuro usa assets/viakable-logo-white.png si existe; si no, pone el logo sobre una pequeña área blanca."""
    from viakable_forecast.config import APP_BRAND, APP_PRODUCT
    logo, chip = _logo_uri("viakable-logo.png"), False
    if logo and DARK:
        white = _logo_uri("viakable-logo-white.png")
        logo, chip = (white, False) if white else (logo, True)
    mark = (f'<span class="lg{" chip" if chip else ""}"><img src="{logo}" alt="{esc(APP_BRAND)}"></span>' if logo else f'<span class="lg-text">{esc(APP_BRAND)}</span>')
    name = f'<span class="lk-sep"></span><span class="lk-name">{esc(APP_PRODUCT.lower())}</span>' if APP_PRODUCT else ""
    return f'<div class="lockup">{mark}{name}</div><div class="product">{esc(t("Pronóstico de ventas, demanda y clientes"))}</div>'


def connector_html() -> str:
    return '<div class="connector" aria-hidden="true"><span></span><span></span></div>'


def headline(text: str, sub: str = "") -> str:
    return f'<p class="headline">{esc(text)}</p>' + (f'<p class="subline">{esc(sub)}</p>' if sub else "")


def h2(text: str, sub: str = "") -> str:
    return f'<h2 class="vk">{esc(text)}</h2>' + (f'<p class="sub">{esc(sub)}</p>' if sub else "")


def group_title(text: str) -> str:
    return f'<div class="grouptitle">{esc(text)}</div>'


_NUM = re.compile(r"^([^\d]*?)(\d[\d,]*(?:\.\d+)?)(\D*)$")


def count_html(text: str, key: str) -> Raw:
    """Valor que cuenta hasta su nuevo número cuando cambia (lo anima el motor de animación) y muestra una flechita con el cambio. Sin JavaScript se ve el texto tal cual."""
    m = _NUM.match(str(text).strip())
    if not m or len(str(text)) > 24:
        return Raw(esc(text))
    pre, num, suf = m.groups()
    sign = -1 if "-" in pre else 1
    plus = "+" in pre
    pre = pre.replace("-", "").replace("+", "")
    sp = suf.startswith(" ")
    suf = suf.strip()
    dec = len(num.split(".")[1]) if "." in num else 0
    val = sign * float(num.replace(",", ""))
    return Raw(f'<span class="cnt" data-k="{esc(key)}" data-count="{val}" data-dec="{dec}" data-pre="{esc(pre)}" data-suf="{esc(suf)}"{" data-sp=1" if sp else ""}{" data-plus=1" if plus else ""}>{esc(text)}</span><i class="kd"></i>')


def kpis(items, cols: int | None = None) -> str:
    """items: lista de (valor, etiqueta, nota opcional). Valor y nota pueden ser Raw."""
    cols = cols or len(items)
    cells = "".join(f'<div class="kpi"><b>{v if isinstance(v, Raw) else count_html(v, "k:" + str(lab))}</b><span>{esc(lab)}</span>'
                    + (f'<small>{n if isinstance(n, Raw) else esc(n)}</small>' if n else "") + '</div>'
                    for v, lab, *rest in items for n in [rest[0] if rest else None])
    return f'<div class="kpis" style="grid-template-columns:repeat({cols},1fr)">{cells}</div>'


def status(items) -> str:
    return '<div class="status">' + "".join(f'<div><b>{esc(v)}</b><span>{esc(l)}</span></div>' for v, l in items) + '</div>'


def table(headers, rows, num=(), min_w: int = 0, main_rows=()) -> str:
    """headers: lista de textos; rows: lista de listas de celdas (str se escapa, Raw no); num: índices alineados a la derecha."""
    th = "".join(f'<th class="{"num" if i in num else ""}">{esc(h)}</th>' for i, h in enumerate(headers))
    body = ""
    for r_i, r in enumerate(rows):
        tds = "".join(f'<td class="{"num" if i in num else ""}">{c if isinstance(c, Raw) else esc(c)}</td>' for i, c in enumerate(r))
        body += f'<tr class="{"is-main" if r_i in main_rows else ""}">{tds}</tr>'
    style = f' style="min-width:{min_w}px"' if min_w else ""
    return f'<div class="tablewrap"><table{style}><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def acc_color(k: str) -> str:
    return {"vh": FG, "h": BLUE, "m": GRAY, "l": RED, "na": LINE}[k]


ACC_ES = {"vh": "Muy certero", "h": "Certero", "m": "Aceptable", "l": "Poco confiable", "na": "Sin prueba"}


def acc_txt(k: str) -> str:
    return t(ACC_ES[k])


def acc_key(err) -> str:
    if err is None or not np.isfinite(err):
        return "na"
    return "vh" if err < 5 else "h" if err < 10 else "m" if err < 20 else "l"


def acc_html(err, short: bool = False) -> Raw:
    k = acc_key(err)
    pct = f" · {max(0, 100 - err):.1f}%" if k != "na" and not short else ""
    return Raw(f'<span class="acc"><i style="background:{acc_color(k)}"></i>{esc(acc_txt(k))}{pct}</span>')


def badge(text: str, kind: str = "") -> Raw:
    return Raw(f'<span class="badge {kind}">{esc(text)}</span>')


def dclass_html(cls: str) -> Raw:
    key = {"Estable": "smooth", "Errática": "erratic", "Intermitente": "intermittent", "Irregular": "lumpy"}.get(cls, "smooth")
    return Raw(f'<span class="dc {key}">{esc(t(cls))}</span>')


def mix_bar(pct: float, max_pct: float) -> Raw:
    return Raw(f'<span class="mix" style="width:{pct / max(max_pct, 1e-9) * 70:.0f}px"></span>{pct:.1f}%')


def spark_svg(y, f) -> Raw:
    """Mini gráfica: últimos 18 meses (negro) y pronóstico (rojo) sobre zona azul clara."""
    a = np.asarray(y, float)[-18:]
    f = np.asarray(f, float)
    allv = np.concatenate([a, f])
    mx, mn = max(allv.max(), 1e-9), min(allv.min(), 0) * 0.95
    W, H, pad = 170, 40, 4
    n = len(allv)
    X = lambda i: pad + i / max(n - 1, 1) * (W - 2 * pad)
    Y = lambda v: H - pad - (v - mn) / ((mx - mn) or 1) * (H - 2 * pad - 6)
    hp = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(a))
    fp = " ".join(f"{X(len(a) - 1 + j):.1f},{Y(v):.1f}" for j, v in enumerate([a[-1], *f]))
    xs = X(len(a) - 1)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}"><rect x="{xs:.1f}" y="0" width="{W - xs:.1f}" height="{H}" fill="rgba(72,179,246,.1)"/>'
               f'<path d="M{hp.split()[0]} L{hp.replace(" ", " L")} L{X(len(a) - 1):.1f},{H} L{X(0):.1f},{H} Z" fill="{_rgba(FG, .08)}"/>'
               f'<polyline points="{hp}" fill="none" stroke="{FG}" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/>'
               f'<polyline points="{fp}" fill="none" stroke="{RED}" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>'
               f'<circle cx="{xs:.1f}" cy="{Y(a[-1]):.1f}" r="2.4" fill="{FG}"/><circle cx="{X(n - 1):.1f}" cy="{Y(f[-1] if len(f) else a[-1]):.1f}" r="3" fill="{RED}"/></svg>')
    import base64
    return Raw('<img class="spark" alt="" src="data:image/svg+xml;base64,' + base64.b64encode(svg.encode()).decode() + '">')


# ------------------------------------------------------------------ números
def fmt(n, dec=None) -> str:
    a = abs(n)
    d = dec if dec is not None else (2 if a < 10 else 1 if a < 100 else 0)
    return f"{n:,.{d}f}"


def mxn(n) -> str:
    a = abs(n)
    if a >= 1e6:
        return f"{'-' if n < 0 else ''}${a / 1e6:,.1f} M"
    return f"{'-' if n < 0 else ''}${a:,.0f}"


def pct(n, d=1) -> str:
    return f"{n:+.{d}f}%"


def mes(ts, largo=False) -> str:
    nombre = (MESES_EN if LANG_EN else MESES_ES)[ts.month - 1]
    return f"{nombre} {ts.year}" if largo else f"{nombre} {str(ts.year)[2:]}"


# ------------------------------------------------------------------ gráficas (estilo de la referencia)
def _rgba(hexc, a):
    h = hexc.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


COMPARE_STYLE = [(BLUE, "dash"), ("#8a8a8a", "dashdot"), ("#2d7fb8", "longdash"), ("#b0b0b0", "dot"), ("#5a5a5a", "dash")]


def _layout(height, legend=True, **kw):
    """Diseño común: fondo transparente (se ve el fondo de la página), tipografía y tooltip del tema."""
    d = dict(height=height, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT, color=MUTED, size=12),
             hoverlabel=dict(bgcolor=FG, font=dict(color=BG, family=FONT)))
    if legend:
        d["legend"] = dict(orientation="h", y=-0.12, yanchor="top", x=0, xanchor="left",
                           font=dict(size=12, color=FG), bgcolor="rgba(0,0,0,0)", borderwidth=0)
    if ANIM:
        d["transition"] = dict(duration=550, easing="cubic-in-out")
    d.update(kw)
    return d


def forecast_fig(hist_idx, hist, f_idx, fc, lo, hi, unit="t", main_label=None, cons=None, height=390, money=False, trend=True, compare=None, marks=None):
    """Historia con área suave, pronóstico con puntos (color de acento), banda y zona de pronóstico azul claro, tendencia punteada.

    compare: lista de (nombre, valores del pronóstico) para dibujar otros métodos como líneas punteadas.
    """
    main_label = main_label or t("Pronóstico")
    n = min(36, len(hist))
    hx, hv = list(hist_idx[-n:]), np.asarray(hist[-n:], float)
    allx = hx + list(f_idx)
    labels = [mes(x) for x in allx]
    nh = len(hx)
    full = np.concatenate([np.asarray(hist, float), np.asarray(fc, float)])
    ma = np.array([full[i - 11:i + 1].mean() if i >= 11 else np.nan for i in range(len(full))])[len(hist) - nh:]
    fmtv = (lambda v: mxn(v)) if money else (lambda v: f"{v:,.0f} {unit}")
    fig = go.Figure()
    fig.add_shape(type="rect", xref="x", yref="paper", x0=nh - 0.5, x1=len(allx) - 0.5, y0=0, y1=1, fillcolor=_rgba(BLUE, .07), line_width=0, layer="below")
    for i, x in enumerate(allx):                                   # separadores de año
        if i > 0 and x.month == 1:
            fig.add_shape(type="line", xref="x", yref="paper", x0=i - .5, x1=i - .5, y0=0, y1=1, line=dict(color=_rgba(MUTED, .25), width=1))
            fig.add_annotation(x=i - .5, y=0, yref="paper", text=str(x.year), showarrow=False, xanchor="left", yanchor="bottom", font=dict(size=11, color=MUTED))
    fig.add_shape(type="line", xref="x", yref="paper", x0=nh - .5, x1=nh - .5, y0=0, y1=1, line=dict(color=RED, width=1.5, dash="dot"))
    fig.add_annotation(x=nh - .5, y=1.04, yref="paper", text=f"<b>{t('Real')}</b>", showarrow=False, xanchor="right", xshift=-8, font=dict(size=12, color=MUTED))
    fig.add_annotation(x=nh - .5, y=1.04, yref="paper", text=f"<b>{t('Pronóstico')}</b>", showarrow=False, xanchor="left", xshift=8, font=dict(size=12, color=RED))
    jx = [labels[nh - 1]] + labels[nh:]
    if lo is not None:
        fig.add_trace(go.Scatter(x=jx, y=[hv[-1], *hi], mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=jx, y=[hv[-1], *lo], mode="lines", line=dict(width=0), fill="tonexty", fillcolor=_rgba(BLUE, .22),
                                 name=t("Rango de confianza (80%)"), hoverinfo="skip"))
    lbl_real = t("Demanda real")
    fig.add_trace(go.Scatter(x=labels[:nh], y=hv, mode="lines", name=lbl_real, line=dict(color=FG, width=2.5, shape="spline", smoothing=.6),
                             fill="tozeroy", fillcolor=_rgba(FG, .07), hovertemplate="%{x}: %{customdata}<extra>" + lbl_real + "</extra>",
                             customdata=[fmtv(v) for v in hv]))
    if trend:
        fig.add_trace(go.Scatter(x=labels, y=ma, mode="lines", name=t("Tendencia (promedio 12 meses)"), line=dict(color=MUTED, width=2, dash="dot"), hoverinfo="skip"))
    for k, (nm, vals) in enumerate(compare or []):
        col, dash = COMPARE_STYLE[k % len(COMPARE_STYLE)]
        fig.add_trace(go.Scatter(x=jx, y=[hv[-1], *vals], mode="lines", name=nm, line=dict(color=col, width=2, dash=dash),
                                 hovertemplate="%{x}: %{y:,.0f}<extra>" + nm + "</extra>"))
    kinds = {"mantener": ("circle-open", t("Mes atípico")), "corregir": ("diamond", t("Mes corregido")), "evento": ("square", t("Evento marcado"))}
    for kind, (symbol, nm) in kinds.items():
        pts = [mk for mk in (marks or []) if mk["kind"] == kind and mes(mk["ym"]) in labels[:nh]]
        if pts:
            fig.add_trace(go.Scatter(x=[mes(mk["ym"]) for mk in pts], y=[mk["y"] for mk in pts], mode="markers", name=nm,
                                     marker=dict(symbol=symbol, size=12, color=RED, line=dict(color=RED, width=2.2)),
                                     hovertext=[mk["text"] for mk in pts], hovertemplate="%{x}: %{hovertext}<extra>" + nm + "</extra>"))
    fig.add_trace(go.Scatter(x=jx, y=[hv[-1], *fc], mode="lines+markers", name=main_label, line=dict(color=RED, width=3.5, shape="spline", smoothing=.6),
                             marker=dict(size=[0] + [7] * len(fc), color=RED, line=dict(color=BG, width=1.5)),
                             hovertemplate="%{x}: %{customdata}<extra>" + main_label + "</extra>", customdata=[fmtv(hv[-1]), *[fmtv(v) for v in fc]]))
    if cons is not None:
        fig.add_trace(go.Scatter(x=labels[nh:], y=cons, mode="lines+markers", name=t("Consenso"), line=dict(color=FG, width=2, dash="dash"),
                                 marker=dict(size=6, symbol="diamond", color=FG)))
    fig.add_annotation(x=labels[-1], y=float(fc[-1]), text=f"<b>{fmtv(fc[-1])}</b>", showarrow=False, yshift=14, xanchor="right", font=dict(size=12, color=RED))
    step = 3 if len(allx) > 24 else 2 if len(allx) > 12 else 1
    fig.update_layout(**_layout(height, margin=dict(l=0, r=8, t=34, b=4), hovermode="x unified"))
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=labels, tickmode="array", tickvals=labels[::step], ticktext=labels[::step], showgrid=False, linecolor=FG, tickangle=0, fixedrange=True)
    fig.update_yaxes(gridcolor=_rgba(MUTED, .15), zeroline=False, showline=False, rangemode="tozero", fixedrange=True, tickformat=",.0f" if not money else None,
                     tickprefix="$" if money else None)
    return fig


def bar_fig(labels, series: dict, colors: dict, height=300, unit=None):
    """Barras apiladas en el estilo de la referencia (p. ej. cobre en acento y aluminio en gris)."""
    unit = unit or UNIT
    fig = go.Figure()
    for name, vals in series.items():
        fig.add_trace(go.Bar(x=labels, y=vals, name=name, marker_color=colors[name], hovertemplate="%{x}: %{y:,.0f} " + unit + "<extra>" + name + "</extra>"))
    multi = len(series) > 1
    fig.update_layout(**_layout(height, margin=dict(l=0, r=8, t=10, b=36), barmode="stack", bargap=.35, legend=True))
    step = 3 if len(labels) > 14 else 1
    fig.update_xaxes(showgrid=False, linecolor=FG, fixedrange=True, tickmode="array", tickvals=list(labels)[::step], ticktext=list(labels)[::step], tickangle=0)
    fig.update_yaxes(gridcolor=_rgba(MUTED, .15), zeroline=False, fixedrange=True, tickformat=",.0f")
    return fig


def metal_fig(hist, now, months, path, name, height=300, unit="USD/t"):
    """Precio de un metal: histórico subido (negro), precio actual y camino de los próximos meses (acento con puntos)."""
    fig = go.Figure()
    labels_h = [mes(x) for x in hist.index] if hist is not None and len(hist) else []
    labels_f = [mes(x) for x in months]
    allx = labels_h + labels_f
    if labels_h:
        hv = hist.to_numpy(float)
        fig.add_trace(go.Scatter(x=labels_h, y=hv, mode="lines", name=t("Histórico"), line=dict(color=FG, width=2.5), fill="tozeroy", fillcolor=_rgba(FG, .07),
                                 hovertemplate="%{x}: %{y:,.0f} " + unit + "<extra></extra>"))
        fig.add_shape(type="rect", xref="x", yref="paper", x0=len(labels_h) - .5, x1=len(allx) - .5, y0=0, y1=1, fillcolor=_rgba(BLUE, .07), line_width=0, layer="below")
    jx = ([labels_h[-1]] if labels_h else []) + labels_f
    jy = ([float(hist.iloc[-1])] if labels_h else []) + list(path)
    fig.add_trace(go.Scatter(x=jx, y=jy, mode="lines+markers", name=t("Camino de precios"), line=dict(color=RED, width=3),
                             marker=dict(size=[0] * (1 if labels_h else 0) + [7] * len(path), color=RED, line=dict(color=BG, width=1.5)),
                             hovertemplate="%{x}: %{y:,.0f} " + unit + "<extra></extra>"))
    fig.add_hline(y=now, line=dict(color=MUTED, width=1, dash="dot"), annotation_text=t("Precio actual"), annotation_position="top left",
                  annotation_font=dict(size=11, color=MUTED))
    step = 3 if len(allx) > 24 else 2 if len(allx) > 12 else 1
    fig.update_layout(**_layout(height, margin=dict(l=0, r=8, t=20, b=4), hovermode="x unified", title=None))
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=allx, tickmode="array", tickvals=allx[::step], ticktext=allx[::step], showgrid=False,
                     linecolor=FG, tickangle=0, fixedrange=True)
    vals = list(path) + ([float(x) for x in hist.to_numpy(float)] if labels_h else []) + [now]
    lo_, hi_ = min(vals), max(vals)
    pad = max((hi_ - lo_) * 0.15, hi_ * 0.05)
    fig.update_yaxes(gridcolor=_rgba(MUTED, .15), zeroline=False, fixedrange=True, tickformat=",.0f", range=[max(lo_ - pad, 0), hi_ + pad])
    return fig


def waterfall_fig(labels, vals, start, unit=None, height=340):
    """Cascada: de un pronóstico de partida al actual, con la aportación de cada factor (sube en acento, baja en oscuro)."""
    unit = unit or UNIT
    names = [labels[0]] + list(labels[1:-1]) + [labels[-1]]
    measure = ["absolute"] + ["relative"] * (len(labels) - 2) + ["total"]
    y = [start] + list(vals) + [None]
    fig = go.Figure(go.Waterfall(x=names, y=y, measure=measure, connector=dict(line=dict(color=_rgba(MUTED, .5), width=1)),
                                 increasing=dict(marker=dict(color=RED)), decreasing=dict(marker=dict(color=FG)), totals=dict(marker=dict(color=MUTED)),
                                 text=[f"{start:,.0f}"] + [f"{v:+,.0f}" for v in vals] + [f"{start + sum(vals):,.0f}"], textposition="outside", cliponaxis=False,
                                 hovertemplate="%{x}: %{y:,.0f} " + unit + "<extra></extra>"))
    fig.update_layout(**_layout(height, legend=False, margin=dict(l=0, r=8, t=10, b=4)))
    lo = min(start, start + sum(vals), *[start + sum(vals[:i + 1]) for i in range(len(vals))])
    hi = max(start, start + sum(vals), *[start + sum(vals[:i + 1]) for i in range(len(vals))])
    pad = max((hi - lo) * 0.25, hi * 0.02, 1)
    wrapped = ["<br>".join(textwrap.wrap(n, 14)) for n in names]
    fig.update_xaxes(showgrid=False, linecolor=FG, fixedrange=True, tickangle=0, automargin=True, tickmode="array", tickvals=names, ticktext=wrapped, tickfont=dict(size=11, color=FG))
    fig.update_yaxes(gridcolor=_rgba(MUTED, .15), zeroline=False, fixedrange=True, tickformat=",.0f", range=[max(0, lo - pad * 2), hi + pad])
    return fig


def hist_fig(values, marks=(), height=300, unit="M", color=None):
    """Distribución simulada (histograma) con rayas verticales para los valores que se quieren destacar: marks = [(valor, etiqueta, color)]. Los valores van en millones."""
    color = color or RED
    v = np.asarray(values, float)
    fig = go.Figure(go.Histogram(x=v, nbinsx=40, marker=dict(color=_rgba(color, .55), line=dict(color=BG, width=1)), hovertemplate="%{x:,.1f} " + unit + "<br>%{y} " + t("simulaciones") + "<extra></extra>"))
    for i, (x, lab, col) in enumerate(marks):
        fig.add_vline(x=x, line=dict(color=col or FG, width=2, dash="dot" if i else "solid"))
        fig.add_annotation(x=x, y=1.0 - 0.09 * (i % 3), yref="paper", text=f"<b>{lab}</b>", showarrow=False, xanchor="left", xshift=5, font=dict(size=11, color=col or FG))
    fig.update_layout(**_layout(height, legend=False, margin=dict(l=0, r=8, t=18, b=4), bargap=0.04))
    fig.update_xaxes(showgrid=False, linecolor=FG, fixedrange=True, ticksuffix=f" {unit}", tickformat=",.0f")
    fig.update_yaxes(gridcolor=_rgba(MUTED, .15), zeroline=False, fixedrange=True, showticklabels=False)
    return fig


def line_fig(x, y, marks=(), height=300, xtitle="", ytitle="", xfmt=",.0f", yfmt=",.1f"):
    """Línea simple con puntos y marcas verticales (valor, etiqueta, color)."""
    fig = go.Figure(go.Scatter(x=list(x), y=list(y), mode="lines+markers", line=dict(color=RED, width=2.5), marker=dict(size=6, color=RED),
                               hovertemplate="%{x:" + xfmt + "}: %{y:" + yfmt + "}<extra></extra>"))
    for i, (xv, lab, col) in enumerate(marks):
        fig.add_vline(x=xv, line=dict(color=col or FG, width=1.5, dash="dot"))
        fig.add_annotation(x=xv, y=1.0 - 0.1 * (i % 3), yref="paper", text=f"<b>{lab}</b>", showarrow=False, xanchor="left", xshift=5, font=dict(size=11, color=col or FG))
    fig.update_layout(**_layout(height, legend=False, margin=dict(l=0, r=8, t=18, b=4)))
    fig.update_xaxes(showgrid=False, linecolor=FG, fixedrange=True, tickformat=xfmt, title=dict(text=xtitle, font=dict(size=11, color=MUTED)))
    fig.update_yaxes(gridcolor=_rgba(MUTED, .15), zeroline=False, fixedrange=True, tickformat=yfmt, title=dict(text=ytitle, font=dict(size=11, color=MUTED)))
    return fig


def explain_fig(names, vals, base=None, height=340):
    """Barras horizontales: qué variables suben (acento) o bajan (oscuro) el pronóstico, en toneladas."""
    order = np.argsort(np.abs(vals))
    names, vals = np.asarray(names)[order], np.asarray(vals)[order]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=[RED if v >= 0 else FG for v in vals],
                           hovertemplate="%{y}: %{x:+,.1f} " + UNIT + "<extra></extra>", text=[f"{v:+,.1f}" for v in vals], textposition="outside", cliponaxis=False,
                           textfont=dict(size=11, color=MUTED)))
    fig.update_layout(**_layout(height, legend=False, margin=dict(l=0, r=40, t=10, b=4)))
    fig.update_xaxes(gridcolor=_rgba(MUTED, .15), zeroline=True, zerolinecolor=FG, zerolinewidth=1, fixedrange=True, ticksuffix=f" {UNIT}")
    fig.update_yaxes(fixedrange=True, automargin=True)
    return fig


def importance_fig(names, vals, height=340):
    order = np.argsort(vals)
    fig = go.Figure(go.Bar(x=np.asarray(vals)[order], y=np.asarray(names)[order], orientation="h", marker_color=FG,
                           hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
    fig.update_layout(**_layout(height, legend=False, margin=dict(l=0, r=8, t=10, b=4), xaxis_title=t("% de importancia")))
    fig.update_xaxes(gridcolor=_rgba(MUTED, .15), zeroline=False, fixedrange=True)
    fig.update_yaxes(fixedrange=True, automargin=True)
    return fig


def cal_label(g: str) -> str:
    return f"{g} AWG" if str(g).isdigit() else str(g)
