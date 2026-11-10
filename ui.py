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
PAL_L = {
    "bg": "#F3F3F3", "fg": "#0A0A0A", "muted": "#5C5C5C", "line": "#ECECEC",
    "soft": "#FFFFFF", "inverse-bg": "#0A0A0A", "inverse-fg": "#FFFFFF",
    "surface-2": "#F7F7F7", "text-3": "#9A9A9A",
    "accent-soft": "#FFE8E8", "black-btn": "#0A0A0A",
    "tile-border-color": "transparent",
    "shadow-tile": "0 12px 32px rgba(0,0,0,.06),0 2px 6px rgba(0,0,0,.04)",
    "shadow-sm": "0 2px 8px rgba(0,0,0,.07),0 1px 3px rgba(0,0,0,.04)",
}
PAL_D = {
    "bg": "#0B0B0C", "fg": "#F5F5F5", "muted": "#B4B4B4", "line": "#26262A",
    "soft": "#161618", "inverse-bg": "#F5F5F5", "inverse-fg": "#0B0B0C",
    "surface-2": "#1E1E21", "text-3": "#7A7A7A",
    "accent-soft": "rgba(255,42,42,.14)", "black-btn": "#F5F5F5",
    "tile-border-color": "rgba(255,255,255,.06)",
    "shadow-tile": "0 2px 12px rgba(0,0,0,.35)",
    "shadow-sm": "0 1px 4px rgba(0,0,0,.3)",
}
FONTS = {"brand": '"Inter Tight","Aeonik","Aptos","Manrope",system-ui,-apple-system,"Segoe UI",sans-serif',
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
  try { var _pd=window.parent?window.parent.document:document; var _bg=getComputedStyle(_pd.documentElement).getPropertyValue('--bg').trim(); function _lum(c){var m=c.match(/rgb\\((\\d+),\\s*(\\d+),\\s*(\\d+)\\)/);return m?(0.299*(+m[1])+0.587*(+m[2])+0.114*(+m[3]))/255:(_bg.startsWith('#')?parseInt(_bg.slice(1,3)||'ff',16)/255:1);} var _isDark=_lum(_bg)<0.15; document.cookie='vk_dark='+(_isDark?1:0)+';path=/;max-age=31536000;SameSite=Lax'; } catch(e) {}
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

    # Resolve which document to target — works whether st.html runs in an
    # iframe (try window.parent.document) or directly in the main document.
    _pd = (
        "var _pd;"
        "try{_pd=window.parent?window.parent.document:document;}catch(e){_pd=document;}"
    )

    font_js = (
        "(function(){"
        + _pd +
        "var fonts=["
        "'https://fonts.googleapis.com/css2?family=Inter+Tight:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&family=Manrope:wght@300;400;500;600;700;800&display=swap',"
        "'https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap'"
        "];"
        "fonts.forEach(function(href){"
        "if(_pd.querySelector('link[href=\"'+href+'\"]'))return;"
        "var l=_pd.createElement('link');l.rel='stylesheet';l.href=href;"
        "(_pd.head||_pd.documentElement).appendChild(l);"
        "});"
        "})();"
    )

    css_js = ""
    if _INJECT_CSS:
        css_content = _json.dumps(_INJECT_CSS)
        css_js = (
            "(function(){"
            + _pd +
            "var _hd=_pd.head||_pd.documentElement;"
            "function _injectCSS(){"
            "var s=_pd.getElementById('vk-css');"
            "if(!s){s=_pd.createElement('style');s.id='vk-css';_hd.appendChild(s);}"
            "else if(s!==_hd.lastElementChild){_hd.appendChild(s);}"  # keep at end
            f"s.textContent={css_content};"
            "}"
            "_injectCSS();"
            # Re-append vk-css to end of head whenever emotion inserts a new style
            "new MutationObserver(function(ms){"
            "ms.forEach(function(m){"
            "m.addedNodes.forEach(function(n){"
            "if(n.nodeName==='STYLE'&&n.id!=='vk-css')_injectCSS();"
            "});"
            "});"
            "}).observe(_hd,{childList:true});"
            "})();"
        )

    # Icon fix: apply Material Symbols font to elements that contain only an icon ligature name
    icon_set_js = _json.dumps([
        'expand_more','expand_less','arrow_drop_down','arrow_drop_up',
        'keyboard_arrow_down','keyboard_arrow_up','keyboard_arrow_right','keyboard_arrow_left',
        'close','check','upload','help','settings','info','warning','error',
        'more_horiz','more_vert','menu','search','add','remove','delete','refresh',
        'download','chevron_right','chevron_left','open_in_new','dark_mode','light_mode',
        'palette','notifications','table_rows','bar_chart','star','tune','layers',
        'inventory_2','account_circle','manage_accounts','fact_check','lock','lock_open',
        'visibility','visibility_off','edit','content_copy','done','done_all',
        'arrow_forward','arrow_back','north','south','east','west',
        'keyboard_return','send','attach_file','link','calendar_today','schedule',
        'trending_up','trending_down','trending_flat','show_chart','analytics',
    ])
    icon_js = (
        "(function(){"
        + _pd +
        f"var _ICONS={icon_set_js};"
        "var _IS={}; _ICONS.forEach(function(x){_IS[x]=1;});"
        "function _fixIcons(root){"
        "root.querySelectorAll('span,i').forEach(function(el){"
        "if(el.children.length>0)return;"
        "var t=(el.textContent||'').trim();"
        "if(!t||!_IS[t])return;"
        "var ff=getComputedStyle(el).fontFamily;"
        "if(ff.indexOf('Material')>=0)return;"
        "el.style.setProperty('font-family',\"'Material Symbols Rounded','Material Icons\",'important');"
        "el.style.setProperty('font-variation-settings',\"'FILL' 0,'wght' 300,'GRAD' 0,'opsz' 24\",'important');"
        "el.style.setProperty('line-height','1','important');"
        "el.style.setProperty('letter-spacing','0','important');"
        "el.style.setProperty('text-transform','none','important');"
        "el.style.setProperty('font-size','inherit','important');"
        "});"
        "}"
        "var _ip=false;"
        "function _si(){"
        "if(_ip)return;_ip=true;"
        "requestAnimationFrame(function(){_ip=false;_fixIcons(_pd.body||_pd.documentElement);});"
        "}"
        "_si();"
        "new MutationObserver(_si).observe(_pd.body||_pd.documentElement,{childList:true,subtree:true});"
        "})();"
    )

    # Toggle vk-dark / vk-light class on <body> so CSS rules in styles.css
    # can target dark mode without needing inline-style injection.
    dark_widget_js = (
        "(function(){"
        + _pd +
        "function _isLight(c){"
        "if(!c||c==='transparent'||c==='rgba(0, 0, 0, 0)')return false;"
        "var m=c.match(/[\\d.]+/g);"
        "if(!m||m.length<3)return false;"
        "return(0.299*(+m[0])+0.587*(+m[1])+0.114*(+m[2]))/255>0.4;"
        "}"
        "var _prev=null;"
        "function _sync(){"
        "var bg=getComputedStyle(_pd.documentElement).getPropertyValue('--bg').trim();"
        "if(!bg)return;"
        "var isDark=!_isLight(bg);"
        "if(isDark===_prev)return;"
        "_prev=isDark;"
        "var b=_pd.body;if(!b)return;"
        "b.classList.toggle('vk-dark',isDark);"
        "b.classList.toggle('vk-light',!isDark);"
        "}"
        "_sync();"
        "setInterval(_sync,300);"
        "})();"
    )

    js_body = ENGINE_JS.strip().removeprefix("<script>").removesuffix("</script>").strip()
    return (f'<div id="vk-cfg" data-anim="{"on" if anim_on else "off"}" style="display:none"></div>'
            f'<script>{font_js}{css_js}{icon_js}{dark_widget_js}{js_body}</script>')




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


def logo_pill_html() -> Raw:
    """Logo lockup: rounded square "C." + "calibra" in a pill."""
    return Raw(
        '<div class="vk-logo-pill">'
        '<div class="vk-logo-square">C<span class="dot">.</span></div>'
        '<span class="vk-logo-name">calibra</span>'
        '</div>'
    )


def footer_html(plan_ready: bool = True, run_date: str = "", data_month: str = "") -> Raw:
    """Bottom status bar: disclaimer · floating status pill · branding."""
    dot_cls = "vk-status-dot" if plan_ready else "vk-status-dot pending"
    label = esc(t("Plan listo")) if plan_ready else esc(t("Calculando…"))
    meta_parts = []
    if run_date:
        meta_parts.append(esc(t("Modelos {d}", d=run_date)))
    if data_month:
        meta_parts.append(esc(t("datos a {m}", m=data_month)))
    meta = " · ".join(meta_parts) if meta_parts else ""
    meta_html = f'<span class="vk-status-meta"> · {meta}</span>' if meta else ""
    return Raw(
        '<div class="vk-footer">'
        '<span class="vk-footer-mono">'
        + esc(t("Prototipo académico · Precios ilustrativos · No conecta con SAP")) +
        '</span>'
        '<div class="vk-status-pill">'
        f'<span class="{dot_cls}"></span>'
        f'<span class="vk-status-label">{label}</span>'
        f'{meta_html}'
        '</div>'
        '<span class="vk-footer-mono">Viakable · ITESM GPO 401 · 2026</span>'
        '</div>'
    )


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
    if len(a) == 0 and len(f) == 0:
        return Raw('<span class="spark"></span>')
    if len(a) == 0:
        a = np.zeros(1)
    if len(f) == 0:
        f = np.zeros(1)
    allv = np.concatenate([a, f])
    mx, mn = max(float(allv.max()), 1e-9), min(float(allv.min()), 0) * 0.95
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
             hoverlabel=dict(bgcolor=SOFT, font=dict(color=FG, family=FONT), bordercolor=LINE),
             uirevision="stable")
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
        fig.add_trace(go.Scatter(x=labels[:nh], y=ma[:nh], mode="lines", name=t("Tendencia (promedio 12 meses)"), line=dict(color=MUTED, width=2, dash="dot"), hoverinfo="skip"))
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
        fig.add_trace(go.Bar(
            x=labels, y=vals, name=name, marker_color=colors[name],
            hovertemplate="%{x}: %{y:,.0f} " + unit + "<extra></extra>",
        ))
    fig.update_layout(**_layout(height, margin=dict(l=0, r=8, t=10, b=4), barmode="stack", bargap=.35, legend=False))
    fig.update_layout(showlegend=False, hovermode="x unified", hoverdistance=100)
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


# ── Caliber palette (fixed across all chart types) ────────────────────────────
CAL_COLORS: dict[str, str] = {
    "6":  "#5b9af5",
    "10": "#f0b02e",
    "12": "#30c88a",
    "14": "#a07af5",
    "16": "#20c4be",
    "18": "#f87040",
}


def cal_color(g: str) -> str:
    key = str(g).split()[0].split("+")[0].strip()
    return CAL_COLORS.get(key, MUTED)


# ── Semi-circle gauge tile ─────────────────────────────────────────────────────

def gauge_tile_html(label: str, val_str: str, pct: float, color: str, cap: str = "") -> Raw:
    """SVG semi-circle gauge wrapped in a .dash-tile. pct: 0–1 (needle position)."""
    import math
    CX, CY, R, SW = 80.0, 74.0, 52.0, 11.0

    def _pt(p):
        a = math.pi * (1.0 - float(p))
        return CX + R * math.cos(a), CY - R * math.sin(a)

    p = max(0.0, min(1.0, float(pct)))
    sx, sy = _pt(0)
    ex, ey = _pt(1)
    px, py = _pt(p)

    arc_bg = f"M{sx:.2f},{sy:.2f} A{R},{R} 0 1 1 {ex:.2f},{ey:.2f}"
    large = 1 if p > 0.5 else 0
    arc_fill = f"M{sx:.2f},{sy:.2f} A{R},{R} 0 {large} 1 {px:.2f},{py:.2f}" if p > 0.02 else ""

    svg = (
        f'<svg viewBox="0 0 160 82" width="100%" style="max-width:180px;display:block;margin:0 auto" '
        f'fill="none" xmlns="http://www.w3.org/2000/svg">'
        f'<path d="{arc_bg}" stroke="{LINE}" stroke-width="{SW}" stroke-linecap="round"/>'
        + (f'<path d="{arc_fill}" stroke="{color}" stroke-width="{SW}" stroke-linecap="round"/>' if arc_fill else "")
        + f'<circle cx="{px:.2f}" cy="{py:.2f}" r="5.5" fill="{color}"/>'
        f'<circle cx="{px:.2f}" cy="{py:.2f}" r="2.5" fill="{BG}"/>'
        f'</svg>'
    )
    cap_html = f'<div class="gauge-cap">{esc(cap)}</div>' if cap else ""
    return Raw(
        f'<div class="dash-tile">'
        f'<div class="gauge-lbl">{esc(label)}</div>'
        f'{svg}'
        f'<div class="gauge-val">{val_str}</div>'
        f'{cap_html}'
        f'</div>'
    )


# ── Stacked area chart by caliber ─────────────────────────────────────────────

def stacked_area_fig(detail, hist_dates, fc_dates, lo, hi, unit: str, height: int = 320) -> go.Figure:
    """Stacked area: one fill layer per caliber, history + forecast separated by a dotted line."""
    nh, nf = len(hist_dates), len(fc_dates)
    mk = lambda d: f"{MESES_ES[d.month - 1][:3]} {str(d.year)[2:]}"
    lh = [mk(d) for d in hist_dates]
    lf = [mk(d) for d in fc_dates]
    all_lbl = lh + lf

    traces: list = []
    for d in detail:
        g_key = str(d.get("label", "")).split()[0]
        color = CAL_COLORS.get(g_key, MUTED)
        y_h = list(np.asarray(d["y"], float)[:nh])
        y_f = list(np.asarray(d["fc"], float)[:nf])
        y_all = y_h + y_f
        n = len(all_lbl)
        if len(y_all) < n:
            y_all += [0.0] * (n - len(y_all))
        name = str(d.get("label", g_key))
        traces.append(go.Scatter(
            x=all_lbl, y=y_all[:n],
            stackgroup="one",
            fillcolor=_rgba(color, 0.38),
            line=dict(color=color, width=1.2),
            name=name, mode="lines",
            hovertemplate=f"<b>{esc(name)}</b><br>%{{x}}: %{{y:,.0f}} {esc(unit)}<extra></extra>",
        ))

    fig = go.Figure(traces)
    if lh:
        fig.add_vline(x=lh[-1], line=dict(color=MUTED, width=1.5, dash="dot"))
        fig.add_annotation(x=lh[-1], y=1.0, yref="paper", text=t("Real | Pronóstico"),
                           showarrow=False, xanchor="left", xshift=6,
                           font=dict(size=10, color=MUTED))
    fig.update_layout(**_layout(height, legend=True, margin=dict(l=0, r=8, t=22, b=4)))
    fig.update_xaxes(type="category", showgrid=False, linecolor=LINE, fixedrange=True,
                     categoryorder="array", categoryarray=all_lbl)
    fig.update_yaxes(gridcolor=_rgba(MUTED, .15), zeroline=False, fixedrange=True,
                     rangemode="tozero", ticksuffix=f" {unit}")
    return fig


# ── Donut: what drives the forecast ───────────────────────────────────────────

def donut_html(steps, base_sum: float, final_sum: float, unit: str = "t") -> Raw:
    """Donut of forecast drivers. Positive contributions → colored arcs.
    Negative factors are listed with a red background stripe so they cannot be missed."""
    import math
    GREEN = "#22c55e"
    seg_colors = [MUTED, RED, GREEN, "#f59e0b", "#a07af5", "#f87040", BLUE]

    pos_steps = [(lbl, d) for lbl, d, _ in steps if d > 0]
    all_pos = float(base_sum) + sum(d for _, d in pos_steps)
    if all_pos < 1:
        all_pos = max(float(final_sum), 1.0)
    segs = [("Base", float(base_sum))] + pos_steps

    CX, CY, R_O, R_I = 80, 80, 65, 36
    GAP = 0.05
    start = -math.pi / 2
    arcs: list[str] = []
    for i, (_, val) in enumerate(segs):
        if val <= 0:
            continue
        frac = val / all_pos
        span = max(0.0, frac * (2 * math.pi - len(segs) * GAP))
        if span < 0.03:
            continue
        end = start + span
        large = 1 if span > math.pi else 0
        x1 = CX + R_O * math.cos(start); y1 = CY + R_O * math.sin(start)
        x2 = CX + R_O * math.cos(end);   y2 = CY + R_O * math.sin(end)
        xi1= CX + R_I * math.cos(end);   yi1= CY + R_I * math.sin(end)
        xi2= CX + R_I * math.cos(start); yi2= CY + R_I * math.sin(start)
        c = seg_colors[i % len(seg_colors)]
        arcs.append(
            f'<path d="M{x1:.2f},{y1:.2f} A{R_O},{R_O} 0 {large} 1 {x2:.2f},{y2:.2f} '
            f'L{xi1:.2f},{yi1:.2f} A{R_I},{R_I} 0 {large} 0 {xi2:.2f},{yi2:.2f} Z" '
            f'fill="{c}" opacity="0.88"/>'
        )
        start = end + GAP

    svg = (
        f'<svg viewBox="0 0 160 160" width="130" height="130" fill="none" '
        f'xmlns="http://www.w3.org/2000/svg">'
        + "".join(arcs)
        + f'<text x="{CX}" y="{CY - 5}" text-anchor="middle" '
        f'font-size="14" font-weight="700" fill="{FG}">{final_sum:,.0f}</text>'
        + f'<text x="{CX}" y="{CY + 12}" text-anchor="middle" '
        f'font-size="9" fill="{MUTED}">{esc(unit)}</text>'
        + f'</svg>'
    )

    rows: list[str] = []
    for lbl, delta, _ in steps:
        neg = delta < 0
        arrow = "▼" if neg else "▲"
        cls_row = "factor-row neg" if neg else "factor-row pos"
        cls_arr = "factor-arrow neg" if neg else "factor-arrow pos"
        cls_val = "factor-val neg" if neg else "factor-val pos"
        rows.append(
            f'<div class="{cls_row}">'
            f'<span class="{cls_arr}">{arrow}</span>'
            f'<span class="factor-lbl">{esc(lbl)}</span>'
            f'<span class="{cls_val}">{delta:+,.1f}</span>'
            f'</div>'
        )

    return Raw(
        f'<div class="dash-tile" style="height:100%">'
        f'<div class="donut-hdr">{esc(t("Qué mueve el pronóstico"))}</div>'
        f'<div style="display:flex;align-items:center;gap:6px;padding:2px 6px 4px">'
        f'<div style="flex-shrink:0">{svg}</div>'
        f'<div style="flex:1;min-width:0">' + "".join(rows) + f'</div>'
        f'</div>'
        f'</div>'
    )


# ── Conclusion tile ────────────────────────────────────────────────────────────
# (see also: Hoy components at end of file)

def conclusion_tile_html(
    fc_next: float, lo_next: float, hi_next: float,
    fc_total: float, lo_total: float, hi_total: float,
    precision: float, sesgo: float,
    method_name: str, yoy, horizon: int, unit: str,
) -> Raw:
    """Summary tile: next month + total + YoY with precision and bias."""
    GREEN = "#22c55e"
    prec_col = GREEN if precision >= 85 else ("#f59e0b" if precision >= 70 else RED)
    ses_col  = GREEN if abs(sesgo) < 10 else ("#f59e0b" if abs(sesgo) < 20 else RED)
    yoy_v = float(yoy) if yoy is not None and np.isfinite(float(yoy)) else None
    yoy_str  = (("▲ " if yoy_v >= 0 else "▼ ") + f"{abs(yoy_v):.1f}%") if yoy_v is not None else "—"
    yoy_col  = (GREEN if yoy_v >= 0 else RED) if yoy_v is not None else MUTED

    return Raw(
        f'<div class="dash-tile">'
        f'<div class="concl-grid">'
        f'<div><div class="concl-lbl">{esc(t("Próximo mes"))}</div>'
        f'<div class="concl-val">{fc_next:,.0f}</div>'
        f'<div class="concl-sub">{esc(unit)} · {lo_next:,.0f}–{hi_next:,.0f}</div></div>'
        f'<div><div class="concl-lbl">{esc(t("Total {h} meses", h=horizon))}</div>'
        f'<div class="concl-val">{fc_total:,.0f}</div>'
        f'<div class="concl-sub">{esc(unit)} · {lo_total:,.0f}–{hi_total:,.0f}</div></div>'
        f'<div><div class="concl-lbl">{esc(t("Cambio vs año anterior"))}</div>'
        f'<div class="concl-val" style="color:{yoy_col}">{yoy_str}</div>'
        f'<div class="concl-sub">'
        f'{esc(t("Precisión"))}: <b style="color:{prec_col}">{precision:.1f}%</b> · '
        f'{esc(t("Sesgo"))}: <b style="color:{ses_col}">{sesgo:+.1f}%</b><br>'
        f'{esc(t("Método"))}: {esc(t(method_name))}'
        f'</div></div>'
        f'</div>'
        f'</div>'
    )


# ── Hoy page components ───────────────────────────────────────────────────────

_CAL_TC_L = {"6":"#FF0000","10":"#0A0A0A","12":"#3A3A3A","14":"#6B6B6B","16":"#A0A0A0","18":"#CFCFCF"}
_CAL_TC_D = {"6":"#FF2A2A","10":"#F5F5F5","12":"#CFCFCF","14":"#9A9A9A","16":"#6B6B6B","18":"#444448"}


def cal_tile_color(g: str) -> str:
    key = str(g).split()[0].split("+")[0].strip()
    return (_CAL_TC_D if DARK else _CAL_TC_L).get(key, MUTED)


def _hoy_sparkline_svg(hist_arr, fc_arr, width: int = 280, height: int = 52) -> str:
    h_v = [float(x) for x in list(hist_arr)[-24:]]
    f_v = [float(x) for x in list(fc_arr)]
    all_v = h_v + f_v
    if len(all_v) < 3:
        return ""
    mn, mx = min(all_v), max(all_v)
    if mx <= mn:
        mx = mn + 1.0
    n_h, n_f = len(h_v), len(f_v)
    n = n_h + n_f

    def _x(i): return i / (n - 1) * width if n > 1 else 0.0
    def _y(v): return (1.0 - (v - mn) / (mx - mn)) * height * 0.85 + height * 0.075

    h_pts = [(_x(i), _y(h_v[i])) for i in range(n_h)]
    f_pts = [(_x(n_h + i), _y(f_v[i])) for i in range(n_f)]
    h_poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in h_pts)
    h_closed = f"0,{height} {h_poly} {h_pts[-1][0]:.1f},{height}"
    f_line = " ".join(f"{x:.1f},{y:.1f}" for x, y in f_pts)
    connect = f"{h_pts[-1][0]:.1f},{h_pts[-1][1]:.1f} {f_line}"
    ex, ey = f_pts[-1] if f_pts else h_pts[-1]
    tx = h_pts[-1][0] if h_pts else 0.0

    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'preserveAspectRatio="none" style="display:block">'
        f'<polygon points="{h_closed}" fill="{FG}" opacity=".06"/>'
        f'<polyline points="{h_poly}" fill="none" stroke="{FG}" '
        f'stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/>'
        f'<line x1="{tx:.1f}" y1="0" x2="{tx:.1f}" y2="{height}" '
        f'stroke="{RED}" stroke-width="1" stroke-dasharray="3 2" opacity=".4"/>'
        f'<polyline points="{connect}" fill="none" stroke="{RED}" '
        f'stroke-width="1.5" stroke-dasharray="4 3" stroke-linejoin="round"/>'
        f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="3.5" fill="{RED}"/>'
        f'</svg>'
    )


def hoy_hero_html(fc: float, lo: float, hi: float,
                  precision: float, yoy_pct: float, h: int,
                  hist_arr=None, fc_arr=None,
                  demand_class: str = "Estable",
                  period_label: str = "") -> Raw:
    _G = "#1A9E4B"
    prec_col = _G if precision >= 85 else ("#D97706" if precision >= 70 else RED)
    yoy_sign = "▲" if yoy_pct >= 0 else "▼"
    yoy_col = _G if yoy_pct >= 0 else RED
    eye = esc(t("VENTA ESPERADA") + (f" · {period_label}" if period_label else ""))
    sentence = esc(t("toneladas de cable en los próximos {h} meses.", h=h))
    range_txt = esc(t("Lo más probable es que quede entre {a} y {b} t.", a=f"{lo:,.0f}", b=f"{hi:,.0f}"))
    chip_prec = esc(t("Precisión {p}%", p=f"{precision:.0f}"))
    chip_dem = esc(t("Demanda {d}", d=t(demand_class)))
    chip_yoy = f"{yoy_sign} {abs(yoy_pct):.1f}% {esc(t('vs. año ant.'))}"
    spark = ""
    spark_labels = ""
    if hist_arr is not None and fc_arr is not None and len(fc_arr) > 0:
        spark = _hoy_sparkline_svg(hist_arr, fc_arr)
        if spark:
            spark_labels = (
                f'<div style="display:flex;justify-content:space-between;'
                f'font-family:\'JetBrains Mono\',monospace;font-size:9px;color:{MUTED};'
                f'letter-spacing:.05em;margin-top:4px">'
                f'<span>{esc(t("HISTORIAL"))}</span>'
                f'<span style="color:{RED}">{esc(t("PRONÓSTICO"))} →</span>'
                f'</div>'
            )
    return Raw(
        f'<div class="vk-tile" style="padding:24px 24px 20px">'
        f'<div class="vk-eyebrow" style="margin-bottom:8px">{eye}</div>'
        f'<div class="vk-hero" style="line-height:1;margin-bottom:4px">{fc:,.0f}</div>'
        f'<div style="font-size:17px;font-weight:600;color:{MUTED};letter-spacing:-.01em;margin-bottom:6px">'
        f'{sentence}</div>'
        f'<div style="font-size:13px;color:{MUTED};margin-bottom:16px;line-height:1.5">'
        f'{range_txt}</div>'
        f'<div style="display:flex;gap:7px;flex-wrap:wrap;margin-bottom:{"16px" if spark else "4px"}">'
        f'<span style="background:{_rgba(prec_col,.1)};border-radius:999px;padding:4px 11px;'
        f'font-size:12px;font-weight:700;color:{prec_col}">{chip_prec}</span>'
        f'<span style="background:{_rgba(FG,.06)};border-radius:999px;padding:4px 11px;'
        f'font-size:12px;font-weight:600;color:{MUTED}">{chip_dem}</span>'
        f'<span style="background:{_rgba(yoy_col,.1)};border-radius:999px;padding:4px 11px;'
        f'font-size:12px;font-weight:700;color:{yoy_col}">{chip_yoy}</span>'
        f'</div>'
        + (f'<div style="border-top:1px solid {LINE};padding-top:12px">'
           f'{spark}{spark_labels}</div>' if spark else "")
        + f'</div>'
    )


def _rings_svg(yoy_frac: float, prec_frac: float, safe_frac: float,
               center_label: str, center_val: str) -> str:
    import math
    CX = CY = 90
    specs = [
        (70, 11, RED, max(0.0, min(1.0, float(yoy_frac)))),
        (55, 11, FG, max(0.0, min(1.0, float(prec_frac)))),
        (40, 10, MUTED, max(0.0, min(1.0, float(safe_frac)))),
    ]
    parts: list = []
    for r, sw, color, frac in specs:
        circ = 2 * math.pi * r
        filled = frac * circ
        parts.append(
            f'<circle cx="{CX}" cy="{CY}" r="{r}" fill="none" stroke="{LINE}" stroke-width="{sw}"/>'
        )
        if filled > 0.5:
            parts.append(
                f'<circle cx="{CX}" cy="{CY}" r="{r}" fill="none" stroke="{color}" '
                f'stroke-width="{sw}" stroke-linecap="round" '
                f'stroke-dasharray="{filled:.1f} {circ:.1f}" '
                f'transform="rotate(-90 {CX} {CY})"/>'
            )
    parts.append(
        f'<text x="{CX}" y="{CY-9}" text-anchor="middle" font-family="\'JetBrains Mono\',monospace" '
        f'font-size="9" fill="{MUTED}">{esc(center_label.upper() if center_label else "")}</text>'
        f'<text x="{CX}" y="{CY+12}" text-anchor="middle" font-family="\'Inter Tight\',sans-serif" '
        f'font-size="22" font-weight="800" fill="{FG}">{esc(center_val)}</text>'
        f'<text x="{CX}" y="{CY+27}" text-anchor="middle" font-family="\'JetBrains Mono\',monospace" '
        f'font-size="8" fill="{MUTED}">{esc(UNIT)}</text>'
    )
    return f'<svg viewBox="0 0 180 180" width="180" height="180">{"".join(parts)}</svg>'


def rings_tile_html(yoy_frac: float, prec_frac: float, safe_frac: float,
                    safe_n: int, total_n: int,
                    center_label: str, center_val: str,
                    yoy_pct: float = 0.0, prev_year: int = 2025,
                    worst_cal=None) -> Raw:
    _G = "#1A9E4B"
    yoy_sign = "+" if yoy_pct >= 0 else ""
    prec_pct = int(round(prec_frac * 100))
    if worst_cal:
        wc_key = str(worst_cal).split()[0].split("+")[0].strip()
        worst_note = esc(t("el {c} pide cuidado", c=f"{wc_key} AWG"))
    else:
        worst_note = esc(t("todos confiables"))

    ring_svg = _rings_svg(yoy_frac, prec_frac, safe_frac, center_label, center_val)

    def _leg(dot, title, sub):
        return (
            f'<div style="display:flex;align-items:flex-start;gap:9px">'
            f'<span style="width:11px;height:11px;border-radius:50%;background:{dot};'
            f'flex-shrink:0;margin-top:2px"></span>'
            f'<div><div style="font-size:12px;font-weight:700;color:{FG};line-height:1.3">{title}</div>'
            f'<div style="font-size:11px;color:{MUTED};line-height:1.4">{sub}</div></div>'
            f'</div>'
        )

    rows = (
        _leg(RED,
             esc(t("Vs. {y} · {s}{p}%", y=prev_year, s=yoy_sign, p=f"{abs(yoy_pct):.1f}")),
             esc(t("de lo vendido en el mismo período")))
        + _leg(FG,
               esc(t("Precisión · {p}%", p=prec_pct)),
               esc(t("de acierto en meses pasados")))
        + _leg(MUTED,
               esc(t("Calibres seguros · {s} de {total}", s=safe_n, total=total_n)),
               worst_note)
    )

    return Raw(
        f'<div class="vk-tile" style="padding:20px 20px 24px;display:flex;'
        f'flex-direction:column;align-items:center">'
        f'<div style="width:100%;display:flex;align-items:center;margin-bottom:12px">'
        f'<span class="vk-eyebrow">ANILLOS DEL PRONÓSTICO</span>'
        f'</div>'
        f'<div style="margin:4px 0 16px">{ring_svg}</div>'
        f'<div style="display:flex;flex-direction:column;gap:10px;width:100%">{rows}</div>'
        f'</div>'
    )


def atiende_hoy_html(items: list) -> Raw:
    n_high = sum(1 for x in items if x.get("sev") == "Alta")
    dot_col = RED if n_high > 0 else MUTED
    hdr_txt = esc(t("{n} alertas", n=n_high)) if n_high > 0 else esc(t("Sin alertas críticas"))

    rows = []
    for it in items[:5]:
        is_hi = it.get("sev") == "Alta"
        txt_low = str(it.get("text", "")).lower()
        badge = "Cu" if ("cobre" in txt_low or "copper" in txt_low) else ("AL" if "alumin" in txt_low else "●")
        badge_bg = RED if is_hi else _rgba(MUTED, .2)
        badge_fg = "#FFFFFF" if is_hi else MUTED
        txt = esc(str(it.get("text", ""))[:90])
        rows.append(
            f'<div style="display:flex;align-items:flex-start;gap:8px;padding:7px 0;'
            f'border-bottom:1px solid {LINE}">'
            f'<span style="background:{badge_bg};color:{badge_fg};border-radius:4px;'
            f'padding:2px 5px;font-size:10px;font-weight:700;flex-shrink:0;margin-top:1px">'
            f'{esc(badge)}</span>'
            f'<span style="font-size:12px;color:{FG};line-height:1.5">{txt}</span>'
            f'</div>'
        )

    body = "".join(rows) if rows else (
        f'<div style="font-size:13px;color:{MUTED};padding:12px 0">'
        f'{esc(t("Sin alertas por ahora."))}</div>'
    )
    return Raw(
        f'<div class="vk-tile-sm" style="padding:16px 18px">'
        f'<div style="display:flex;align-items:center;gap:7px;margin-bottom:6px">'
        f'<span style="width:7px;height:7px;border-radius:50%;background:{dot_col};flex-shrink:0"></span>'
        f'<span class="vk-eyebrow">ATIENDE HOY · {hdr_txt}</span>'
        f'</div>'
        f'{body}'
        f'</div>'
    )


def noticias_tile_html(news_items: list) -> Raw:
    rows = []
    for r in news_items[:4]:
        date_str = str(r.get("date") or r.get("published_at") or "")[:10]
        topic = esc(str(r.get("topic", ""))[:22])
        title = esc(str(r.get("title", r.get("headline", "")))[:75])
        source = esc(str(r.get("source", ""))[:25])
        rows.append(
            f'<div style="padding:7px 0;border-bottom:1px solid {LINE}">'
            f'<div style="display:flex;align-items:center;gap:5px;margin-bottom:2px">'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:9px;color:{MUTED}">'
            f'{esc(date_str)}</span>'
            f'<span style="background:{_rgba(FG,.07)};border-radius:3px;padding:1px 5px;'
            f'font-size:10px;color:{MUTED}">{topic}</span>'
            f'</div>'
            f'<div style="font-size:12px;font-weight:600;color:{FG};line-height:1.4">{title}</div>'
            f'<div style="font-size:10px;color:{MUTED};margin-top:1px">{source}</div>'
            f'</div>'
        )
    body = "".join(rows) if rows else (
        f'<div style="font-size:12px;color:{MUTED};padding:8px 0">'
        f'{esc(t("No hay noticias recientes."))}</div>'
    )
    return Raw(
        f'<div class="vk-tile-sm" style="padding:14px 18px">'
        f'<span class="vk-eyebrow" style="display:block;margin-bottom:8px">NOTICIAS DEL SECTOR</span>'
        f'{body}'
        f'</div>'
    )


def caliber_tile_html(g: str, fc_next: float, fc_total: float,
                      mix_pct: float, precision: float,
                      demand_class: str = "Estable", h: int = 6) -> Raw:
    _G, _A = "#1A9E4B", "#D97706"
    color = cal_tile_color(g)
    if precision >= 90:
        badge_col, badge_lbl = _G, t("Muy seguro")
    elif precision >= 80:
        badge_col, badge_lbl = "#2563EB", t("Seguro")
    elif precision >= 70:
        badge_col, badge_lbl = _A, t("Aceptable")
    else:
        badge_col, badge_lbl = RED, t("Con cuidado")
    key = str(g).split()[0].split("+")[0].strip()
    lbl = f"{key} AWG" if key.isdigit() else str(g)
    sub_total = esc(t("{v} t en {h} m", v=f"{fc_total:,.0f}", h=h))
    return Raw(
        f'<div class="vk-tile-sm" style="padding:14px 15px">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">'
        f'<div style="display:flex;align-items:center;gap:6px">'
        f'<span style="width:10px;height:10px;border-radius:50%;background:{color};flex-shrink:0"></span>'
        f'<span style="font-size:12px;font-weight:700;color:{FG}">{esc(lbl)}</span>'
        f'</div>'
        f'<span style="font-size:11px;color:{MUTED}">{mix_pct:.1f}%</span>'
        f'</div>'
        f'<div style="font-size:38px;font-weight:800;letter-spacing:-.03em;color:{FG};'
        f'line-height:1;margin-bottom:2px">{fc_next:,.0f}</div>'
        f'<div style="font-size:11px;color:{MUTED};margin-bottom:6px">'
        f'{esc(UNIT)} · {esc(t("próximo mes"))}</div>'
        f'<div style="font-size:11px;color:{MUTED};margin-bottom:10px">{sub_total}</div>'
        f'<div style="display:flex;align-items:center;gap:5px">'
        f'<span style="width:7px;height:7px;border-radius:50%;background:{badge_col}"></span>'
        f'<span style="font-size:11px;font-weight:600;color:{badge_col}">{esc(badge_lbl)}</span>'
        f'<span style="font-size:11px;color:{MUTED}"> · {precision:.0f}%</span>'
        f'</div>'
        f'</div>'
    )


def quien_compra_tile_html(customers: list) -> Raw:
    _G = "#1A9E4B"
    rows = []
    for c in customers[:5]:
        name = esc(str(c.get("name", "—"))[:22])
        qty = float(c.get("qty", 0))
        yoy_v = float(c.get("yoy", 0.0) or 0.0)
        yoy_sign = "▲" if yoy_v >= 0 else "▼"
        yoy_col = _G if yoy_v >= 0 else RED
        rows.append(
            f'<div style="display:flex;align-items:center;justify-content:space-between;'
            f'padding:6px 0;border-bottom:1px solid {LINE}">'
            f'<span style="font-size:12px;font-weight:600;color:{FG};min-width:0;overflow:hidden;'
            f'text-overflow:ellipsis;white-space:nowrap;flex:1">{name}</span>'
            f'<div style="display:flex;align-items:center;gap:4px;flex-shrink:0;margin-left:5px">'
            f'<span style="font-size:11px;color:{MUTED}">{qty:,.0f} {esc(UNIT)}</span>'
            f'<span style="font-size:11px;font-weight:700;color:{yoy_col}">'
            f'{yoy_sign}{abs(yoy_v):.1f}%</span>'
            f'</div></div>'
        )
    body = "".join(rows) if rows else (
        f'<div style="font-size:12px;color:{MUTED};padding:8px 0">'
        f'{esc(t("Sin datos de clientes."))}</div>'
    )
    return Raw(
        f'<div class="vk-tile-sm" style="padding:14px 15px">'
        f'<span class="vk-eyebrow" style="display:block;margin-bottom:8px">QUIÉN COMPRA · 12M</span>'
        f'{body}'
        f'</div>'
    )


def section_header_html(eyebrow: str, headline: str, sub: str) -> Raw:
    """Full-width section header tile for Pronóstico and other main pages."""
    return Raw(
        f'<div class="vk-tile" style="padding:20px 24px 18px;margin-bottom:4px">'
        f'<span class="vk-eyebrow" style="display:block;margin-bottom:8px">{esc(eyebrow)}</span>'
        f'<div class="vk-headline" style="margin:0 0 6px">{esc(headline)}</div>'
        f'<div style="font-size:13px;color:{MUTED};line-height:1.55">{esc(sub)}</div>'
        f'</div>'
    )
