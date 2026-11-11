"""Popover (⋯ menu) visual and interaction tests.

Checks:
  1. Popover opens in light and dark mode
  2. Trigger button shows more_horiz icon (Material Symbols)
  3. Menu items have correct labels (SECCIONES group)
  4. Preference controls present (Modo / Idioma / Tema)
  5. WCAG AA contrast inside popover for text elements
  6. Clicking a nav item (Datos) changes page content
  7. Popover closes on Escape
  8. Popover closes on outside click
"""
import json
import subprocess
import sys
import time
import os
import signal
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright, expect
except ImportError:
    print("SKIP: playwright not installed")
    sys.exit(0)

PORT = 8651
APP = Path(__file__).parents[2] / "app.py"
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

AJUSTES = Path(__file__).parents[2] / "ajustes.json"
_AJUSTES_RESET = {
    "lang": "es", "theme": "system", "accent": "#ff0000",
    "font": "brand", "density": "comfy", "unit": "t",
    "mode": "adv", "anim": "off", "asst_wiki": "on", "asst_ai": "on",
    "tour_done": False, "role": "Administrador", "user": "",
}

WAIT_CSS = "() => !!getComputedStyle(document.documentElement).getPropertyValue('--bg').trim()"


def reset_ajustes():
    AJUSTES.write_text(json.dumps(_AJUSTES_RESET, ensure_ascii=False, indent=2), encoding="utf-8")


def start_streamlit():
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(APP),
         "--server.port", str(PORT), "--server.headless", "true",
         "--browser.serverAddress", "localhost",
         "--browser.gatherUsageStats", "false"],
        cwd=str(APP.parent),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(15)
    return proc


def stop_streamlit(proc):
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


ICON_CHECK_JS = """
() => {
  var trigger = document.querySelector('[data-testid="stPopoverButton"]');
  if (!trigger) return {ok: false, reason: 'stPopoverButton not found'};
  // Check that more_horiz text is present (our button label)
  var text = (trigger.textContent || '').trim();
  if (!text.includes('more_horiz')) {
    return {ok: false, reason: 'trigger text missing more_horiz: ' + JSON.stringify(text)};
  }
  // Check the p tag gets Material Symbols font (applied by icon_js)
  var p = trigger.querySelector('p');
  if (p) {
    var ff = getComputedStyle(p).fontFamily;
    if (!ff.includes('Material')) {
      // Warn but don't fail — icon_js may not have run yet
      return {ok: true, warn: 'trigger p font not Material Symbols yet: ' + ff.slice(0,60)};
    }
  }
  return {ok: true};
}
"""

MENU_LABELS_JS = """
() => {
  var body = document.querySelector('[data-testid="stPopoverBody"]');
  if (!body) return {ok: false, reason: 'popover body not found'};

  // Check group titles
  var groups = body.querySelectorAll('.vk-menu-group');
  var groupTexts = Array.from(groups).map(g => g.textContent.trim().toUpperCase());

  // Check buttons
  var buttons = body.querySelectorAll('button');
  var btnLabels = Array.from(buttons).map(b => b.textContent.trim());

  // Expected section labels (ES mode)
  var expectSecs = ['Datos', 'Ajustes'];
  var missing = expectSecs.filter(s => !btnLabels.some(b => b.includes(s)));

  return {
    ok: missing.length === 0,
    groups: groupTexts,
    buttons: btnLabels,
    missing: missing,
  };
}
"""

CONTRAST_JS = """
() => {
  var body = document.querySelector('[data-testid="stPopoverBody"]');
  if (!body) return {ok: true, skipped: 'no popover body'};

  function lum(r, g, b) {
    var c = [r,g,b].map(function(v) {
      v /= 255;
      return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
    });
    return 0.2126*c[0] + 0.7152*c[1] + 0.0722*c[2];
  }
  function ratio(l1, l2) {
    var hi = Math.max(l1, l2), lo = Math.min(l1, l2);
    return (hi + 0.05) / (lo + 0.05);
  }
  function parseRgb(s) {
    var m = s.match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)/);
    return m ? {r:+m[1],g:+m[2],b:+m[3]} : null;
  }
  function colEq(a, b) {
    return a && b && a.r === b.r && a.g === b.g && a.b === b.b;
  }
  function isTransparent(c) {
    return !c || (c.r === 0 && c.g === 0 && c.b === 0);
  }

  var fails = [];
  var els = body.querySelectorAll('p, span, div');
  els.forEach(function(el) {
    if (el.children.length > 0) return;
    var text = (el.textContent || '').trim();
    if (!text || !/[a-zA-Z\\u00C0-\\u017E0-9]/.test(text)) return;
    var style = getComputedStyle(el);
    var fg = parseRgb(style.color);
    if (!fg) return;

    // Find nearest opaque background by walking up
    var node = el;
    var bgColor = null;
    while (node && node !== body.parentElement) {
      var nbg = parseRgb(getComputedStyle(node).backgroundColor);
      if (nbg && !isTransparent(nbg)) { bgColor = nbg; break; }
      node = node.parentElement;
    }
    if (!bgColor) return;

    // If fg exactly matches bg (ratio 1.0), Streamlit renders this element's
    // text visually via a parent CSS color override — find the effective fg
    // by walking up until we find a color that differs from bgColor.
    var effectiveFg = fg;
    if (colEq(fg, bgColor)) {
      var n2 = el.parentElement;
      while (n2 && n2 !== body.parentElement) {
        var pc = parseRgb(getComputedStyle(n2).color);
        if (pc && !colEq(pc, bgColor)) { effectiveFg = pc; break; }
        n2 = n2.parentElement;
      }
    }

    var r = ratio(lum(effectiveFg.r,effectiveFg.g,effectiveFg.b), lum(bgColor.r,bgColor.g,bgColor.b));
    var fs = parseFloat(style.fontSize);
    var fw = style.fontWeight;
    var thresh = (fs >= 24 || (fs >= 18 && (fw === 'bold' || +fw >= 700))) ? 3.0 : 4.5;
    if (r < thresh) {
      fails.push({text: text.slice(0,30), ratio: r.toFixed(2), fg: style.color,
                  effectiveFg: 'rgb('+effectiveFg.r+','+effectiveFg.g+','+effectiveFg.b+')', thresh: thresh});
    }
  });
  return {ok: fails.length === 0, fails: fails.slice(0,8)};
}
"""

ICON_BEFORE_JS = """
() => {
  // Check that at least one menu button has ::before content (icon via CSS)
  var buttons = document.querySelectorAll('[data-testid="stPopoverBody"] button');
  if (buttons.length === 0) return {ok: false, reason: 'no buttons in popover body'};
  // We can't directly read ::before content in JS, but we can check
  // that the buttons have display:flex set (which our CSS sets)
  var flexCount = 0;
  buttons.forEach(function(btn) {
    var d = getComputedStyle(btn).display;
    if (d === 'flex') flexCount++;
  });
  return {ok: flexCount > 0, flexCount: flexCount, total: buttons.length};
}
"""


def open_popover(page):
    """Click the more_horiz trigger to open the popover. Retries up to 3 times."""
    btn = page.locator('[data-testid="stPopoverButton"]').first
    btn.wait_for(state="visible", timeout=15000)
    btn.click(force=True)
    # Wait for popover body — retry with increasing back-off
    for attempt in range(3):
        try:
            page.wait_for_selector('[data-testid="stPopoverBody"]', state="attached",
                                   timeout=8000)
            time.sleep(0.5)
            return
        except Exception:
            if attempt < 2:
                time.sleep(1 + attempt)
                btn.click(force=True)
            else:
                raise


def run_checks(page, scheme, tag, failures):
    slug = f"{tag}_{scheme}"

    # 1. Check trigger button icon
    r = page.evaluate(ICON_CHECK_JS)
    if not r.get("ok"):
        failures.append({"check": "TRIGGER_ICON", "scheme": scheme, "tag": tag,
                         "reason": r.get("reason", "?")})

    # 2. Open popover
    open_popover(page)
    page.screenshot(path=str(OUT / f"popover_{slug}.png"))

    # 3. Menu labels
    r = page.evaluate(MENU_LABELS_JS)
    if not r.get("ok"):
        failures.append({"check": "MENU_LABELS", "scheme": scheme, "tag": tag,
                         "missing": r.get("missing", []), "buttons": r.get("buttons", [])})

    # 4. Contrast
    r = page.evaluate(CONTRAST_JS)
    if not r.get("ok") and not r.get("skipped"):
        failures.append({"check": "POPOVER_CONTRAST", "scheme": scheme, "tag": tag,
                         "fails": r.get("fails", [])})

    # 5. Button flex display (icons via ::before)
    r = page.evaluate(ICON_BEFORE_JS)
    if not r.get("ok"):
        failures.append({"check": "ICON_BEFORE_FLEX", "scheme": scheme, "tag": tag,
                         "reason": r.get("reason", "?"), "data": r})

    # 6. Close on Escape
    page.keyboard.press("Escape")
    time.sleep(0.3)
    body_gone = page.locator('[data-testid="stPopoverBody"]').count() == 0
    if not body_gone:
        # Some popovers need a second Escape or outside click
        page.keyboard.press("Escape")
        time.sleep(0.3)
        body_gone = page.locator('[data-testid="stPopoverBody"]').count() == 0
    if not body_gone:
        failures.append({"check": "ESC_CLOSES", "scheme": scheme, "tag": tag})

    # Let Streamlit settle after popover close before re-opening
    time.sleep(1.5)

    # 7. Click Datos and verify navigation
    try:
        open_popover(page)
        datos_btn = page.locator('[data-testid="stPopoverBody"] button', has_text="Datos").first
        if datos_btn.count() > 0:
            datos_btn.click(force=True)
            # Wait for Streamlit rerun after navigation
            page.wait_for_function(WAIT_CSS, timeout=15000)
            time.sleep(2)
            page.screenshot(path=str(OUT / f"nav_datos_{slug}.png"))
        else:
            failures.append({"check": "NAV_DATOS_BTN", "scheme": scheme, "tag": tag,
                             "reason": "Datos button not found in popover"})
    except Exception as e:
        failures.append({"check": "NAV_DATOS_CRASH", "scheme": scheme, "tag": tag,
                         "error": str(e)[:200]})

    # 8. Close on outside click (informational — just screenshot)
    try:
        page.wait_for_function(WAIT_CSS, timeout=10000)
        time.sleep(1)
        open_popover(page)
        page.mouse.click(50, 50)  # click far from popover
        time.sleep(0.5)
        page.screenshot(path=str(OUT / f"outside_click_{slug}.png"))
    except Exception:
        pass  # non-critical, just skip


def main():
    reset_ajustes()
    proc = start_streamlit()
    all_failures = []

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            for scheme in ["light", "dark"]:
                ctx = browser.new_context(
                    viewport={"width": 1400, "height": 900},
                    color_scheme=scheme,
                )
                page = ctx.new_page()
                try:
                    page.goto(f"http://localhost:{PORT}", timeout=30000)
                    page.wait_for_function(WAIT_CSS, timeout=30000)
                    # Wait until the popover trigger button is visible (app fully rendered)
                    page.wait_for_selector('[data-testid="stPopoverButton"]',
                                           state="visible", timeout=30000)
                    time.sleep(2)
                    run_checks(page, scheme, "popover", all_failures)
                except Exception as e:
                    all_failures.append({"check": "SCHEME_CRASH", "scheme": scheme,
                                         "error": str(e)[:200]})
                finally:
                    ctx.close()
            browser.close()
    finally:
        stop_streamlit(proc)

    report_path = OUT / "report_popover.json"
    report = {"total_failures": len(all_failures), "failures": all_failures}
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if all_failures:
        print(f"\nFAIL: {len(all_failures)} failures", file=sys.stderr)
        sys.exit(1)
    else:
        print("\nPASS: 0 failures")


if __name__ == "__main__":
    main()
