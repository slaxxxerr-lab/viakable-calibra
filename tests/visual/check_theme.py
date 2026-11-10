"""
Viakable Calibra — automated theme visual check.

Usage:
    .venv/Scripts/python tests/visual/check_theme.py

Starts the Streamlit app, opens each main page in light and dark mode
(using Playwright's native color-scheme emulation — no UI clicking needed),
runs checks (LIGHT_WIDGETS_DARK, ICONS, CONTRAST, INVISIBLE_LABELS),
saves screenshots and a JSON + text report to tests/visual/out/.

Exit code 0 = all checks passed. Non-zero = failures found.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PORT = 8599
BASE_URL = f"http://localhost:{PORT}"
OUT = Path(__file__).parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

PAGES = ["Hoy", "Pronóstico", "Compras de metal", "Clientes"]
THEMES = ["light", "dark"]

# Expected token values per theme (used to verify CSS injection health)
EXPECTED_BG = {"light": "#F3F3F3", "dark": "#0B0B0C"}
EXPECTED_FG = {"light": "#0A0A0A", "dark": "#F5F5F5"}

# JS: CSS health check — verifies token values are set
CSS_HEALTH_JS = """
() => {
    const root = document.documentElement;
    const cs = getComputedStyle(root);
    const iframes = document.querySelectorAll('iframe');
    return {
        bgToken: cs.getPropertyValue('--bg').trim(),
        fgToken: cs.getPropertyValue('--fg').trim(),
        softToken: cs.getPropertyValue('--soft').trim(),
        lineToken: cs.getPropertyValue('--line').trim(),
        accentToken: cs.getPropertyValue('--accent').trim(),
        surf2Token: cs.getPropertyValue('--surface-2').trim(),
        iframeCount: iframes.length,
    };
}
"""

# JS: check for light widget backgrounds in dark mode (brightness > 0.6)
# Excludes Streamlit's own toolbar area (Stop/Deploy/Settings header)
LIGHT_WIDGETS_JS = """
() => {
    function hexToRGB(c) {
        if (!c) return null;
        c = c.trim();
        if (c.startsWith('rgb')) {
            const m = c.match(/[\d.]+/g);
            if (!m || m.length < 3) return null;
            return [+m[0], +m[1], +m[2]];
        }
        if (c.startsWith('#')) {
            const hex = c.slice(1);
            if (hex.length === 3) return [parseInt(hex[0]+hex[0],16), parseInt(hex[1]+hex[1],16), parseInt(hex[2]+hex[2],16)];
            if (hex.length === 6) return [parseInt(hex.slice(0,2),16), parseInt(hex.slice(2,4),16), parseInt(hex.slice(4,6),16)];
        }
        return null;
    }
    function luminance(rgb) {
        if (!rgb) return -1;
        return (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255;
    }
    // Composite rgba(r,g,b,a) over an opaque background color
    function composite(fg, bg) {
        const a = fg[3] !== undefined ? fg[3] : 1;
        return [
            Math.round(a * fg[0] + (1-a) * bg[0]),
            Math.round(a * fg[1] + (1-a) * bg[1]),
            Math.round(a * fg[2] + (1-a) * bg[2]),
        ];
    }
    function bgOf(el) {
        let cur = el;
        let opaqueBg = null;
        while (cur && cur !== document.body.parentElement) {
            const bg = getComputedStyle(cur).backgroundColor;
            if (bg && bg !== 'rgba(0, 0, 0, 0)' && bg !== 'transparent') {
                const rgba = hexToRGB(bg);
                if (!rgba) break;
                const alpha = rgba[3] !== undefined ? rgba[3] : 1;
                if (alpha >= 0.95) { opaqueBg = rgba; break; }
                // Semi-transparent — keep walking up to find opaque ancestor
            }
            cur = cur.parentElement;
        }
        return opaqueBg || hexToRGB(getComputedStyle(document.body).backgroundColor) || [255,255,255];
    }
    // Skip elements inside Streamlit's own header toolbar (Stop/Deploy/Settings buttons)
    function inStreamlitHeader(el) {
        let cur = el;
        while (cur) {
            if (cur.getAttribute && (
                cur.getAttribute('data-testid') === 'stHeader' ||
                cur.getAttribute('data-testid') === 'stStatusWidget' ||
                cur.getAttribute('data-testid') === 'stToolbar'
            )) return true;
            cur = cur.parentElement;
        }
        return false;
    }
    const selectors = [
        'button:not([data-testid="stThemeToggle"])',
        'input', 'textarea', 'select',
        '[role="listbox"]', '[role="option"]',
        '[data-testid="stTextInput"] > div',
        '[data-testid="stSelectbox"] [data-baseweb="select"] > div',
        '[data-testid="stMultiSelect"] [data-baseweb="select"] > div',
        '[data-testid="stSegmentedControl"] > div',
        '[data-testid="stExpander"] summary',
        '[data-testid="stFileUploaderDropzone"]',
        '[data-testid="stPopover"] [data-testid="stPopoverBody"]',
        '[data-baseweb="tab-list"]',
    ];
    const failures = [];
    selectors.forEach(sel => {
        try {
            document.querySelectorAll(sel).forEach(el => {
                if (inStreamlitHeader(el)) return;
                if (!el.offsetParent && el.tagName !== 'BODY') return;
                const rect = el.getBoundingClientRect();
                if (rect.width === 0 || rect.height === 0) return;
                const bg = bgOf(el);
                const lum = luminance(bg);
                if (lum > 0.6) {
                    failures.push({
                        sel,
                        tag: el.tagName,
                        testid: el.getAttribute('data-testid') || '',
                        bg: `rgb(${bg[0]},${bg[1]},${bg[2]})`,
                        lum: lum.toFixed(3),
                        text: (el.textContent || '').trim().slice(0,60),
                    });
                }
            });
        } catch(e) {}
    });
    return failures;
}
"""

# JS: icon ligature text visible (Material Symbols rendered as text)
ICON_CHECK_JS = """
() => {
    const ICON_PATTERN = /^[a-z][a-z0-9]*(_[a-z0-9]+)+$/;
    const ICON_WORDS = new Set([
        'expand_more','expand_less','arrow_drop_down','arrow_drop_up',
        'keyboard_arrow_down','keyboard_arrow_up','keyboard_arrow_right',
        'close','check','upload','help','settings','info','warning','error',
        'more_horiz','more_vert','menu','search','add','remove','delete',
        'refresh','download','chevron_right','chevron_left','open_in_new'
    ]);
    const failures = [];
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    let node;
    while (node = walker.nextNode()) {
        const text = node.textContent.trim();
        if (!text) continue;
        if (!(ICON_PATTERN.test(text) || ICON_WORDS.has(text))) continue;
        const parent = node.parentElement;
        if (!parent) continue;
        const style = getComputedStyle(parent);
        if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') continue;
        const rect = parent.getBoundingClientRect();
        if (rect.width === 0 && rect.height === 0) continue;
        if (style.fontFamily.includes('Material')) continue;
        failures.push({
            text,
            tag: parent.tagName,
            testid: parent.getAttribute('data-testid') || '',
            class: parent.className.slice(0,80),
            fontFamily: style.fontFamily.slice(0,100),
        });
    }
    return failures;
}
"""

# JS: contrast check with alpha compositing.
# Excludes Streamlit's internal header area and known low-contrast design choices
# (active tab with accent color on accent-soft bg).
CONTRAST_JS = """
() => {
    function parseRGBA(c) {
        if (!c) return null;
        const m = c.match(/[\d.]+/g);
        if (!m || m.length < 3) return null;
        return [+m[0], +m[1], +m[2], m[3] !== undefined ? +m[3] : 1];
    }
    function lum(r,g,b) {
        const lin = v => { v=v/255; return v<=0.03928?v/12.92:Math.pow((v+0.055)/1.055,2.4); };
        return 0.2126*lin(r)+0.7152*lin(g)+0.0722*lin(b);
    }
    function contrast(fg, bg) {
        const la = lum(fg[0],fg[1],fg[2]), lb = lum(bg[0],bg[1],bg[2]);
        const hi = Math.max(la,lb), lo = Math.min(la,lb);
        return (hi+0.05)/(lo+0.05);
    }
    // Walk up and composite semi-transparent backgrounds onto opaque ancestors
    function resolvedBg(el) {
        const layers = [];
        let cur = el;
        while (cur && cur !== document.documentElement) {
            const bg = getComputedStyle(cur).backgroundColor;
            if (bg && bg !== 'rgba(0, 0, 0, 0)' && bg !== 'transparent') {
                const rgba = parseRGBA(bg);
                if (rgba) layers.push(rgba);
                if (!rgba || (rgba[3] !== undefined && rgba[3] >= 0.95)) break;
            }
            cur = cur.parentElement;
        }
        if (layers.length === 0) {
            const docBg = parseRGBA(getComputedStyle(document.documentElement).backgroundColor);
            return docBg ? [docBg[0],docBg[1],docBg[2]] : [255,255,255];
        }
        // Composite from bottom (last) to top (first)
        let base = layers[layers.length-1];
        for (let i = layers.length-2; i >= 0; i--) {
            const fg = layers[i];
            const a = fg[3] !== undefined ? fg[3] : 1;
            if (a >= 0.95) { base = [fg[0],fg[1],fg[2]]; continue; }
            base = [
                Math.round(a*fg[0]+(1-a)*base[0]),
                Math.round(a*fg[1]+(1-a)*base[1]),
                Math.round(a*fg[2]+(1-a)*base[2]),
            ];
        }
        return base;
    }
    function inStreamlitHeader(el) {
        let cur = el;
        while (cur) {
            const tid = cur.getAttribute && cur.getAttribute('data-testid');
            if (tid === 'stHeader' || tid === 'stStatusWidget' || tid === 'stToolbar') return true;
            cur = cur.parentElement;
        }
        return false;
    }
    // Skip active/selected nav items — accent color on accent-soft is by design
    function isActiveNavItem(el) {
        if (el.getAttribute('aria-selected') === 'true') return true;
        if (el.getAttribute('aria-pressed') === 'true') return true;
        let cur = el;
        while (cur) {
            if (cur.getAttribute && cur.getAttribute('aria-selected') === 'true') return true;
            cur = cur.parentElement;
        }
        return false;
    }
    const failures = [];
    const textEls = document.querySelectorAll('p, label, span, button, input, [data-testid*="Label"], h1, h2, h3');
    let checked = 0;
    textEls.forEach(el => {
        if (checked > 300) return;
        if (inStreamlitHeader(el)) return;
        const rect = el.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return;
        const cs = getComputedStyle(el);
        if (cs.display === 'none' || cs.visibility === 'hidden') return;
        const text = el.textContent.trim();
        if (!text || text.length < 2) return;
        const fgRGBA = parseRGBA(cs.color);
        if (!fgRGBA) return;
        // Composite fg alpha against resolved background
        const bgSolid = resolvedBg(el);
        const fgA = fgRGBA[3] !== undefined ? fgRGBA[3] : 1;
        const fgComposited = fgA < 0.95
            ? [Math.round(fgA*fgRGBA[0]+(1-fgA)*bgSolid[0]),
               Math.round(fgA*fgRGBA[1]+(1-fgA)*bgSolid[1]),
               Math.round(fgA*fgRGBA[2]+(1-fgA)*bgSolid[2])]
            : [fgRGBA[0],fgRGBA[1],fgRGBA[2]];
        const ratio = contrast(fgComposited, bgSolid);
        const fontSize = parseFloat(cs.fontSize);
        const isBold = parseInt(cs.fontWeight) >= 700;
        const isLarge = fontSize >= 24 || (fontSize >= 18 && isBold);
        const threshold = isLarge ? 3.0 : 4.5;
        if (ratio < threshold) {
            // Skip accent-on-accent-soft (active tab design intent)
            if (isActiveNavItem(el)) return;
            failures.push({
                tag: el.tagName,
                text: text.slice(0,60),
                testid: el.getAttribute('data-testid') || '',
                fg: cs.color,
                bg: `rgb(${bgSolid.join(',')})`,
                ratio: ratio.toFixed(2),
                threshold,
                fontSize: cs.fontSize,
            });
        }
        checked++;
    });
    return failures;
}
"""

# JS: radio/checkbox/pills labels are visible
LABEL_CHECK_JS = """
() => {
    const failures = [];
    const selectors = [
        '[data-testid="stRadio"] label',
        '[data-testid="stCheckbox"] label',
        '[data-testid="stSegmentedControl"] button',
        '[data-testid="stPillsToggleButtonGroup"] button',
        '[data-testid="stToggleButtonGroup"] button',
        '[data-testid="stExpander"] summary',
    ];
    selectors.forEach(sel => {
        try {
            document.querySelectorAll(sel).forEach(el => {
                const cs = getComputedStyle(el);
                if (cs.display === 'none') return;
                const rect = el.getBoundingClientRect();
                if (rect.width === 0 && rect.height === 0) return;
                const text = el.textContent.trim();
                if (!text) return;
                const opacity = parseFloat(cs.opacity);
                const fontSize = parseFloat(cs.fontSize);
                if (opacity < 0.1 || fontSize < 1) {
                    failures.push({sel, text: text.slice(0,40), opacity, fontSize});
                }
            });
        } catch(e) {}
    });
    return failures;
}
"""


def wait_for_app(timeout=60):
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(BASE_URL, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def run_checks():
    """Main test runner using separate browser contexts per color-scheme."""
    from playwright.sync_api import sync_playwright

    all_failures: list[dict] = []
    screenshots: list[str] = []
    css_health_results: list[dict] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)

        for scheme in THEMES:
            # Use Playwright's native color-scheme emulation — no UI clicking needed
            ctx = browser.new_context(
                viewport={"width": 1400, "height": 900},
                color_scheme=scheme,
            )
            page = ctx.new_page()
            page.on("console", lambda m: None)  # suppress noise

            # Initial load — wait for the app to render tokens
            page.goto(BASE_URL, wait_until="networkidle", timeout=60000)
            # Wait until --bg token is set (app has rendered CSS)
            try:
                page.wait_for_function(
                    "() => !!getComputedStyle(document.documentElement).getPropertyValue('--bg').trim()",
                    timeout=30000
                )
            except Exception:
                pass
            time.sleep(2)  # extra time for components to stabilize

            for page_name in PAGES:
                print(f"\n=== {page_name} / {scheme} ===")
                safe_name = page_name.replace(" ", "_").replace("/", "-")

                # Navigate to page by clicking the nav button
                try:
                    btn = page.locator(
                        f'[data-testid="stSegmentedControl"] button:has-text("{page_name}")'
                    )
                    if btn.count() > 0:
                        btn.first.click()
                        try:
                            page.wait_for_load_state("networkidle", timeout=10000)
                        except Exception:
                            pass
                        time.sleep(2)
                except Exception:
                    pass

                # CSS health check
                health = page.evaluate(CSS_HEALTH_JS)
                css_health_results.append({"page": page_name, "theme": scheme, **health})
                bg = health.get("bgToken", "")
                fg = health.get("fgToken", "")
                expected_bg = EXPECTED_BG[scheme]
                expected_fg = EXPECTED_FG[scheme]
                token_ok = bg.lower() == expected_bg.lower() and fg.lower() == expected_fg.lower()
                print(f"  CSS tokens: bg={bg!r} fg={fg!r} {'OK' if token_ok else 'WRONG'}")
                if not token_ok:
                    print(f"    expected: bg={expected_bg!r} fg={expected_fg!r}")

                # Screenshot
                ss_path = str(OUT / f"{safe_name}_{scheme}.png")
                page.screenshot(path=ss_path, full_page=True)
                screenshots.append(ss_path)

                # LIGHT_WIDGETS_DARK (only in dark mode)
                if scheme == "dark":
                    widget_fails = page.evaluate(LIGHT_WIDGETS_JS)
                    for f in widget_fails:
                        all_failures.append({
                            "check": "LIGHT_WIDGETS_DARK",
                            "page": page_name, "theme": scheme, **f
                        })
                    if widget_fails:
                        print(f"  LIGHT_WIDGETS_DARK: {len(widget_fails)} failures")

                # ICONS
                icon_fails = page.evaluate(ICON_CHECK_JS)
                for f in icon_fails:
                    all_failures.append({
                        "check": "ICONS",
                        "page": page_name, "theme": scheme, **f
                    })
                if icon_fails:
                    print(f"  ICONS: {len(icon_fails)} failures")

                # CONTRAST
                contrast_fails = page.evaluate(CONTRAST_JS)
                for f in contrast_fails:
                    all_failures.append({
                        "check": "CONTRAST",
                        "page": page_name, "theme": scheme, **f
                    })
                if contrast_fails:
                    print(f"  CONTRAST: {len(contrast_fails)} failures")

                # INVISIBLE_LABELS
                label_fails = page.evaluate(LABEL_CHECK_JS)
                for f in label_fails:
                    all_failures.append({
                        "check": "INVISIBLE_LABELS",
                        "page": page_name, "theme": scheme, **f
                    })
                if label_fails:
                    print(f"  INVISIBLE_LABELS: {len(label_fails)} failures")

                total_this = len(widget_fails if scheme=="dark" else []) + len(icon_fails) + len(contrast_fails) + len(label_fails)
                print(f"  total failures this page: {total_this}")

            # Interactive elements in dark mode
            if scheme == "dark":
                print("\n=== Interactive elements (dark) ===")
                try:
                    exp = page.locator('[data-testid="stExpander"] summary').first
                    if exp.is_visible():
                        exp.click()
                        time.sleep(0.5)
                        page.screenshot(path=str(OUT / "expander_dark.png"))
                        print("  Expander: opened + screenshot")
                except Exception as e:
                    print(f"  Expander: {e}")

                try:
                    pop = page.locator('[data-testid="stPopover"] > div > button').first
                    if pop.is_visible():
                        pop.click()
                        time.sleep(0.8)
                        page.screenshot(path=str(OUT / "popover_dark.png"))
                        print("  Popover: opened + screenshot")
                        page.keyboard.press("Escape")
                except Exception as e:
                    print(f"  Popover: {e}")

            page.close()
            ctx.close()

        browser.close()

    return all_failures, css_health_results, screenshots


def summarize(failures: list[dict], health: list[dict]) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("VIAKABLE CALIBRA — THEME CHECK REPORT")
    lines.append("=" * 60)

    lines.append("\n── CSS Token Health ──")
    for h in health:
        bg = h.get("bgToken", "")
        fg = h.get("fgToken", "")
        scheme = h["theme"]
        ok = bg.lower() == EXPECTED_BG[scheme].lower() and fg.lower() == EXPECTED_FG[scheme].lower()
        status = "OK" if ok else "FAIL"
        lines.append(f"  [{status}] {h['page']} / {h['theme']}: --bg={bg!r} --fg={fg!r}")

    by_check: dict[str, list] = {}
    for f in failures:
        by_check.setdefault(f["check"], []).append(f)

    lines.append(f"\n── Total failures: {len(failures)} ──")
    for check, fails in sorted(by_check.items()):
        lines.append(f"  {check}: {len(fails)}")

    by_pt: dict[str, int] = {}
    for f in failures:
        key = f"{f['page']} / {f['theme']}"
        by_pt[key] = by_pt.get(key, 0) + 1
    lines.append("\n── By page/theme ──")
    for k, n in sorted(by_pt.items(), key=lambda x: -x[1]):
        lines.append(f"  {k}: {n}")

    lines.append("\n── Top 20 failures ──")
    for i, f in enumerate(failures[:20]):
        check = f["check"]
        page = f["page"]
        theme = f["theme"]
        if check == "LIGHT_WIDGETS_DARK":
            lines.append(f"  {i+1}. [{check}] {page}/{theme} | sel={f.get('sel','')} bg={f.get('bg','')} lum={f.get('lum','')}")
        elif check == "ICONS":
            lines.append(f"  {i+1}. [{check}] {page}/{theme} | text={f.get('text','')} font={f.get('fontFamily','')[:60]}")
        elif check == "CONTRAST":
            lines.append(f"  {i+1}. [{check}] {page}/{theme} | {f.get('tag','')} {f.get('text','')!r} fg={f.get('fg','')} bg={f.get('bg','')} ratio={f.get('ratio','')}")
        elif check == "INVISIBLE_LABELS":
            lines.append(f"  {i+1}. [{check}] {page}/{theme} | sel={f.get('sel','')} text={f.get('text','')!r}")
        else:
            lines.append(f"  {i+1}. [{check}] {page}/{theme} | {f}")

    lines.append("\n" + "=" * 60)
    lines.append(f"RESULT: {'PASS' if not failures else f'FAIL — {len(failures)} issue(s)'}")
    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    # Ensure ajustes.json uses theme: system so prefers-color-scheme drives theming
    ajustes_path = Path(__file__).parent.parent.parent / "ajustes.json"
    if ajustes_path.exists():
        try:
            data = json.loads(ajustes_path.read_text(encoding="utf-8"))
            if data.get("theme") not in ("system", "light"):
                data["theme"] = "system"
                ajustes_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                print("Reset ajustes.json theme -> system")
        except Exception:
            pass

    env = os.environ.copy()
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py",
         "--server.headless", "true",
         "--server.port", str(PORT),
         "--server.runOnSave", "false",
         "--logger.level", "error"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env
    )

    try:
        print(f"Waiting for Streamlit on port {PORT}...")
        if not wait_for_app(60):
            print("ERROR: App did not start in 60s")
            proc.kill()
            sys.exit(2)
        print("App is up.")
        time.sleep(5)  # wait for initial data load / model training

        failures, health, screenshots = run_checks()

        report_json = {"failures": failures, "css_health": health, "screenshots": screenshots}
        (OUT / "report.json").write_bytes(json.dumps(report_json, indent=2).encode("utf-8"))

        summary = summarize(failures, health)
        (OUT / "summary.txt").write_text(summary, encoding="utf-8")
        print(summary)

        sys.exit(1 if failures else 0)

    finally:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except Exception:
            pass


if __name__ == "__main__":
    main()
