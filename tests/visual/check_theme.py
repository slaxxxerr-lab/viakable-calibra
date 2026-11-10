"""
Viakable Calibra — automated theme visual check.

Usage:
    .venv/Scripts/python tests/visual/check_theme.py

Starts the Streamlit app, opens each main page in light and dark mode,
runs JS checks (contrast, light-widgets-in-dark, icon render, invisible labels),
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
from typing import Any

# ── constants ──────────────────────────────────────────────────────────────
PORT = 8599
BASE_URL = f"http://localhost:{PORT}"
OUT = Path(__file__).parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

PAGES = ["Hoy", "Pronóstico", "Compras de metal", "Clientes"]
THEMES = ["light", "dark"]

# JS that checks CSS injection health (also scans iframes)
CSS_HEALTH_JS = """
() => {
    const style = document.getElementById('vk-css');
    const root = document.documentElement;
    const cs = getComputedStyle(root);

    // Scan iframes for vk-css and sandbox info
    const iframes = document.querySelectorAll('iframe');
    const frameInfo = [];
    for (const f of Array.from(iframes).slice(0, 8)) {
        const info = {
            src: (f.src || f.srcdoc || 'no-src').slice(0, 60),
            sandbox: f.getAttribute('sandbox') || 'none',
        };
        try {
            const fd = f.contentDocument;
            info.hasVkCss = !!fd?.getElementById('vk-css');
            const fcs = fd ? getComputedStyle(fd.documentElement) : null;
            info.bgToken = fcs ? fcs.getPropertyValue('--bg').trim() : 'n/a';
        } catch(e) {
            info.blocked = e.message.slice(0, 60);
        }
        frameInfo.push(info);
    }

    return {
        styleExists: !!style,
        styleLength: style ? style.textContent.length : 0,
        bgToken: cs.getPropertyValue('--bg').trim(),
        fgToken: cs.getPropertyValue('--fg').trim(),
        softToken: cs.getPropertyValue('--soft').trim(),
        lineToken: cs.getPropertyValue('--line').trim(),
        accentToken: cs.getPropertyValue('--accent').trim(),
        surf2Token: cs.getPropertyValue('--surface-2').trim(),
        iframeCount: iframes.length,
        frameInfo: frameInfo,
    };
}
"""

# JS that checks for the LIGHT_WIDGETS_DARK failure: in dark mode, widget bg luminance > 0.6
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
    function bgOf(el) {
        let cur = el;
        while (cur && cur !== document.body.parentElement) {
            const bg = getComputedStyle(cur).backgroundColor;
            if (bg && bg !== 'rgba(0, 0, 0, 0)' && bg !== 'transparent') return bg;
            cur = cur.parentElement;
        }
        return getComputedStyle(document.body).backgroundColor || '#ffffff';
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
                if (!el.offsetParent && el.tagName !== 'BODY') return; // not visible
                const rect = el.getBoundingClientRect();
                if (rect.width === 0 || rect.height === 0) return;
                const bg = bgOf(el);
                const lum = luminance(hexToRGB(bg));
                if (lum > 0.6) {
                    failures.push({
                        sel,
                        tag: el.tagName,
                        testid: el.getAttribute('data-testid') || '',
                        bg,
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

# JS that checks for icon render failures (Material Symbol ligature text visible)
ICON_CHECK_JS = """
() => {
    const ICON_PATTERN = /^[a-z][a-z0-9]*(_[a-z0-9]+)+$/;  // e.g. expand_more, arrow_drop_down
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
        // Check if visible
        const style = getComputedStyle(parent);
        if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') continue;
        const rect = parent.getBoundingClientRect();
        if (rect.width === 0 && rect.height === 0) continue;
        // Skip if icon font is already correctly applied
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

# JS that checks contrast for body text
CONTRAST_JS = """
() => {
    function parseRGB(c) {
        if (!c) return null;
        const m = c.match(/[\d.]+/g);
        if (!m || m.length < 3) return null;
        return [+m[0], +m[1], +m[2], m[3] !== undefined ? +m[3] : 1];
    }
    function lum(rgb) {
        const [r,g,b] = rgb.map(v => {
            v = v/255;
            return v <= 0.03928 ? v/12.92 : Math.pow((v+0.055)/1.055, 2.4);
        });
        return 0.2126*r + 0.7152*g + 0.0722*b;
    }
    function contrast(a, b) {
        const la = lum(a), lb = lum(b);
        const hi = Math.max(la,lb), lo = Math.min(la,lb);
        return (hi+0.05)/(lo+0.05);
    }
    function bgOf(el) {
        let cur = el;
        while (cur && cur !== document.documentElement) {
            const cs = getComputedStyle(cur);
            const bg = cs.backgroundColor;
            if (bg && bg !== 'rgba(0, 0, 0, 0)' && bg !== 'transparent') {
                return parseRGB(bg);
            }
            cur = cur.parentElement;
        }
        return parseRGB(getComputedStyle(document.documentElement).backgroundColor) || [255,255,255];
    }
    const failures = [];
    const textEls = document.querySelectorAll('p, label, span, button, input, [data-testid*="Label"], h1, h2, h3');
    let checked = 0;
    textEls.forEach(el => {
        if (checked > 200) return;
        const rect = el.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return;
        const cs = getComputedStyle(el);
        if (cs.display === 'none' || cs.visibility === 'hidden') return;
        const text = el.textContent.trim();
        if (!text || text.length < 2) return;
        const fgRGB = parseRGB(cs.color);
        const bgRGB = bgOf(el);
        if (!fgRGB || !bgRGB) return;
        const ratio = contrast(fgRGB, bgRGB);
        const fontSize = parseFloat(cs.fontSize);
        const isBold = parseInt(cs.fontWeight) >= 700;
        const isLarge = fontSize >= 24 || (fontSize >= 18 && isBold);
        const threshold = isLarge ? 3.0 : 4.5;
        if (ratio < threshold) {
            failures.push({
                tag: el.tagName,
                text: text.slice(0,60),
                testid: el.getAttribute('data-testid') || '',
                fg: cs.color,
                bg: `rgb(${bgRGB.join(',')})`,
                ratio: ratio.toFixed(2),
                threshold,
                fontSize: cs.fontSize,
                fontWeight: cs.fontWeight,
            });
        }
        checked++;
    });
    return failures;
}
"""

# JS that checks radio/checkbox/pills labels are visible
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
    """Main test runner."""
    from playwright.sync_api import sync_playwright

    all_failures: list[dict] = []
    screenshots: list[str] = []
    css_health_results: list[dict] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1400, "height": 900})
        page = context.new_page()

        # Suppress console noise
        page.on("console", lambda m: None)

        def navigate_to(nav_name: str):
            """Click the nav tab with the given name."""
            page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
            time.sleep(2)
            # Try clicking the segmented control button
            try:
                btn = page.locator(f'[data-testid="stSegmentedControl"] button:has-text("{nav_name}")')
                if btn.count() > 0:
                    btn.first.click()
                    time.sleep(1.5)
            except Exception:
                pass

        def get_current_theme() -> str:
            """Return 'dark' if --bg is dark, else 'light'."""
            try:
                bg = page.evaluate(
                    "() => getComputedStyle(document.documentElement).getPropertyValue('--bg').trim().toLowerCase()"
                )
                return "dark" if bg == "#0b0b0c" else "light"
            except Exception:
                return "unknown"

        def set_theme(theme: str):
            """Click the theme toggle to set light or dark mode, verify with CSS var."""
            if get_current_theme() == theme:
                return  # already correct

            # Labels to try (in app order)
            light_labels = ["☀️", "☀", "Claro", "Light", "claro"]
            dark_labels  = ["🌙", "Oscuro", "Dark", "oscuro"]
            labels = light_labels if theme == "light" else dark_labels

            for label in labels:
                try:
                    btn = page.locator(f'button:has-text("{label}")')
                    if btn.count() > 0:
                        btn.first.click()
                        time.sleep(2)
                        if get_current_theme() == theme:
                            return
                except Exception:
                    pass

            # Fallback: look inside popover / ⋯ menu
            try:
                popover_btn = page.locator('[data-testid="stPopover"] > div > button').first
                if popover_btn.is_visible():
                    popover_btn.click()
                    time.sleep(0.5)
                    for label in labels:
                        try:
                            btn = page.locator(f'button:has-text("{label}")')
                            if btn.count() > 0:
                                btn.first.click()
                                time.sleep(2)
                                if get_current_theme() == theme:
                                    page.keyboard.press("Escape")
                                    return
                        except Exception:
                            pass
                    page.keyboard.press("Escape")
            except Exception as e:
                pass

            current = get_current_theme()
            if current != theme:
                print(f"  [warn] set_theme({theme}) failed — still {current}")

        def check_page(page_name: str, theme: str) -> list[dict]:
            """Run all checks on a page+theme combo."""
            failures: list[dict] = []
            safe_name = page_name.replace(" ", "_").replace("/", "-")

            # CSS health
            health = page.evaluate(CSS_HEALTH_JS)
            css_health_results.append({"page": page_name, "theme": theme, **health})
            frame_summary = ""
            if health.get('iframeCount', 0) > 0:
                for fi in health.get('frameInfo', [])[:3]:
                    frame_summary += f" | iframe sandbox={fi.get('sandbox','?')!r} vk={fi.get('hasVkCss','?')} blocked={fi.get('blocked','')!r}"
            print(f"  CSS health: styleExists={health['styleExists']} len={health['styleLength']} bg={health['bgToken']!r} fg={health['fgToken']!r} iframes={health.get('iframeCount',0)}{frame_summary}")

            # Screenshot
            ss_path = str(OUT / f"{safe_name}_{theme}.png")
            page.screenshot(path=ss_path, full_page=True)
            screenshots.append(ss_path)

            # LIGHT_WIDGETS_DARK (only in dark mode)
            if theme == "dark":
                widget_fails = page.evaluate(LIGHT_WIDGETS_JS)
                for f in widget_fails:
                    failures.append({
                        "check": "LIGHT_WIDGETS_DARK",
                        "page": page_name, "theme": theme,
                        **f
                    })

            # ICONS
            icon_fails = page.evaluate(ICON_CHECK_JS)
            for f in icon_fails:
                failures.append({
                    "check": "ICONS",
                    "page": page_name, "theme": theme,
                    **f
                })

            # CONTRAST
            contrast_fails = page.evaluate(CONTRAST_JS)
            for f in contrast_fails:
                failures.append({
                    "check": "CONTRAST",
                    "page": page_name, "theme": theme,
                    **f
                })

            # INVISIBLE_LABELS
            label_fails = page.evaluate(LABEL_CHECK_JS)
            for f in label_fails:
                failures.append({
                    "check": "INVISIBLE_LABELS",
                    "page": page_name, "theme": theme,
                    **f
                })

            return failures

        # Open app
        print(f"Opening {BASE_URL}...")
        page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
        time.sleep(3)

        # First set light mode as baseline
        set_theme("light")

        for page_name in PAGES:
            print(f"\n=== {page_name} ===")
            navigate_to(page_name)
            time.sleep(2)

            # Light mode
            set_theme("light")
            time.sleep(1)
            page_failures = check_page(page_name, "light")
            all_failures.extend(page_failures)
            print(f"  light: {len(page_failures)} failures")

            # Dark mode
            set_theme("dark")
            time.sleep(1.5)
            page_failures = check_page(page_name, "dark")
            all_failures.extend(page_failures)
            print(f"  dark: {len(page_failures)} failures")

        # Test interactive elements in dark mode
        print("\n=== Interactive elements (dark) ===")
        navigate_to("Hoy")
        set_theme("dark")
        time.sleep(1.5)

        # Open an expander if present
        try:
            exp = page.locator('[data-testid="stExpander"] summary').first
            if exp.is_visible():
                exp.click(); time.sleep(0.5)
                page.screenshot(path=str(OUT / "expander_dark.png"))
                print("  Expander: opened + screenshot")
        except Exception as e:
            print(f"  Expander: {e}")

        # Open ⋯ popover
        try:
            pop = page.locator('[data-testid="stPopover"] > div > button').first
            if pop.is_visible():
                pop.click(); time.sleep(0.8)
                page.screenshot(path=str(OUT / "popover_dark.png"))
                print("  Popover: opened + screenshot")
                page.keyboard.press("Escape")
        except Exception as e:
            print(f"  Popover: {e}")

        # Navigate to Pronóstico and open a selectbox
        navigate_to("Pronóstico")
        set_theme("dark")
        time.sleep(2)
        try:
            sel = page.locator('[data-testid="stSelectbox"]').first
            if sel.is_visible():
                sel.click(); time.sleep(0.5)
                page.screenshot(path=str(OUT / "selectbox_dropdown_dark.png"))
                print("  Selectbox dropdown: opened + screenshot")
                page.keyboard.press("Escape")
        except Exception as e:
            print(f"  Selectbox: {e}")

        browser.close()

    return all_failures, css_health_results, screenshots


def summarize(failures: list[dict], health: list[dict]) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("VIAKABLE CALIBRA — THEME CHECK REPORT")
    lines.append("=" * 60)

    # CSS health summary
    lines.append("\n── CSS Injection Health ──")
    for h in health:
        status = "OK" if h.get("styleExists") and h.get("bgToken") else "FAIL"
        lines.append(f"  [{status}] {h['page']} / {h['theme']}: "
                     f"styleExists={h.get('styleExists')} "
                     f"len={h.get('styleLength',0)} "
                     f"--bg={h.get('bgToken')!r} "
                     f"--fg={h.get('fgToken')!r}")

    # Failure summary by check type
    by_check: dict[str, list] = {}
    for f in failures:
        by_check.setdefault(f["check"], []).append(f)

    lines.append(f"\n── Total failures: {len(failures)} ──")
    for check, fails in sorted(by_check.items()):
        lines.append(f"  {check}: {len(fails)}")

    # Failures by page+theme
    by_pt: dict[str, int] = {}
    for f in failures:
        key = f"{f['page']} / {f['theme']}"
        by_pt[key] = by_pt.get(key, 0) + 1
    lines.append("\n── By page/theme ──")
    for k, n in sorted(by_pt.items(), key=lambda x: -x[1]):
        lines.append(f"  {k}: {n}")

    # Top 20 failures
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
            lines.append(f"  {i+1}. [{check}] {page}/{theme} | sel={f.get('sel','')} text={f.get('text','')!r} opacity={f.get('opacity','')}")
        else:
            lines.append(f"  {i+1}. [{check}] {page}/{theme} | {f}")

    lines.append("\n" + "=" * 60)
    lines.append(f"RESULT: {'PASS ✓' if not failures else f'FAIL — {len(failures)} issue(s)'}")
    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    # Start streamlit
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
        time.sleep(3)  # extra wait for full initialization

        failures, health, screenshots = run_checks()

        report_json = {"failures": failures, "css_health": health, "screenshots": screenshots}
        (OUT / "report.json").write_text(json.dumps(report_json, indent=2), encoding="utf-8")

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
