"""Compara el motor ANTERIOR (un solo backtest de 6 meses) con el motor NUEVO (validación cruzada de origen móvil).

Dos formas de medir, para ser honestos:
  1) "Error reportado": lo que cada motor le dice al usuario (sobre sus propios periodos de prueba). Es optimista porque el
     método se elige mirando esos mismos meses.
  2) "Error real fuera de muestra": se repite TODO el proceso (elegir método, afinar, pronosticar) como si estuviéramos 6, 9 y 12
     meses antes del final, usando solo datos anteriores, y se compara contra lo que de verdad se vendió en los 6 meses siguientes.
     Esta es la medida confiable.
Uso:  python evaluacion_motor.py [archivo.csv]    (escribe resultados_motor.json y resultados_motor.md)
"""
import json
import sys
import time

import numpy as np
import pandas as pd

from viakable_forecast import data as D
from viakable_forecast import engine as E
from viakable_forecast import methods as M
from viakable_forecast import ml as MLm
from viakable_forecast.legacy import engine as OE
from viakable_forecast.legacy import methods as OM
from viakable_forecast.legacy import ml as OML

path = sys.argv[1] if len(sys.argv) > 1 else "Viakable_Tons_Sales.csv"
raw = D.load_file(path)
prep = D.prepare(raw, D.detect_columns(raw))
full = prep.monthly
gauges = prep.gauges
months = list(pd.date_range(full["ym"].min(), full["ym"].max(), freq="MS"))
N = len(months)
ORIGINS = [6, 9, 12]          # meses antes del final donde se "congela" la información
H = 6
ENT = ["Total"] + gauges


def actual_of(ent, k):
    g = gauges if ent == "Total" else [ent]
    y = E.series(full, g).values
    return y[N - k:N - k + H]


def err(ent, k, f):
    return M.error_pct(actual_of(ent, k), np.asarray(f)[:H])


def entity_fc(pr_detail, total_fc, ent):
    if ent == "Total":
        return total_fc
    return next(d["fc"] for d in pr_detail if d["label"] == ent)


UNITS = [[g] for g in gauges]
rep_old, rep_new = {}, {}
out_old = {e: [] for e in ENT}
out_new = {e: [] for e in ENT}
meth = {}            # (método, entidad) -> lista de errores por origen
VARIANTS = [0.0, 1.0, 2.0, 99.0]
var_err = {(v, e): [] for v in VARIANTS for e in ENT}
cov_new, cov_old = [], []
t0 = time.time()

# ---------- 1) error reportado con TODOS los datos
print("Calculando error reportado (todos los datos)…", flush=True)
ml_old = OML.fit_all(full, OE.holdout_size(N))
pr_o = OE.project(full, UNITS, None, 6, OE.AUTO, OE.TON, "sum", 1.0, None, 0.3, None, None, ml_old)
rep_old["Total"] = pr_o.error
for d in pr_o.detail:
    rep_old[d["label"]] = d["error"]
ml_new = MLm.fit_all(full, E.windows_for(full))
pr_n = E.project(full, UNITS, None, 6, E.AUTO, E.TON, "sum", 1.0, None, 0.3, None, None, ml_new)
rep_new["Total"] = pr_n.error
for d in pr_n.detail:
    rep_new[d["label"]] = d["error"]
print(f"   ML nuevo elegido: {ml_new['desc']} ({time.time() - t0:.0f} s)", flush=True)

# ---------- 2) error real fuera de muestra: se repite todo el proceso en cada origen
for k in ORIGINS:
    print(f"Origen: {k} meses antes del final (entrenando con {N - k} meses)…", flush=True)
    cut = months[N - k - 1]
    dk = full[full["ym"] <= cut].copy()
    nk = N - k
    # motor anterior
    mlo = OML.fit_all(dk, OE.holdout_size(nk))
    po = OE.project(dk, UNITS, None, H, OE.AUTO, OE.TON, "sum", 1.0, None, 0.3, None, None, mlo)
    # motor nuevo
    mln = MLm.fit_all(dk, E.windows_for(dk))
    pn = E.project(dk, UNITS, None, H, E.AUTO, E.TON, "sum", 1.0, None, 0.3, None, None, mln)
    for ent in ENT:
        out_old[ent].append(err(ent, k, entity_fc(po.detail, po.fc, ent)))
        out_new[ent].append(err(ent, k, entity_fc(pn.detail, pn.fc, ent)))
    # variantes de la regla automática (mismo ML ya entrenado)
    saved = E.AUTO_MARGIN
    for v in VARIANTS:
        E.AUTO_MARGIN = v
        pv = E.project(dk, UNITS, None, H, E.AUTO, E.TON, "sum", 1.0, None, 0.3, None, None, mln)
        for ent in ENT:
            var_err[(v, ent)].append(err(ent, k, entity_fc(pv.detail, pv.fc, ent)))
    E.AUTO_MARGIN = saved
    # cobertura del rango de confianza: ¿cuántos meses reales cayeron dentro?
    for ent in ENT:
        g = gauges if ent == "Total" else [ent]
        act = actual_of(ent, k)
        pe = pn if ent == "Total" else E.project(dk, [[ent]], None, H, E.AUTO, E.TON, "sum", 1.0, None, 0.3, None, None, mln)
        cov_new.append(((act >= pe.lo[:H] - 1e-9) & (act <= pe.hi[:H] + 1e-9)).mean())
        f_old = entity_fc(po.detail, po.fc, ent)
        e_old = rep_err = po.error if ent == "Total" else next(d["error"] for d in po.detail if d["label"] == ent)
        eb = (e_old if np.isfinite(e_old) else 8.0) / 100
        b = np.array([max(0.03, eb * 1.25) * (1 + 0.08 * i) for i in range(H)])
        cov_old.append(((act >= f_old[:H] * (1 - np.minimum(0.9, b))) & (act <= f_old[:H] * (1 + np.minimum(0.9, b)))).mean())
    # cada método por separado (serie de cada entidad, sin elegir nada)
    for ent in ENT:
        g = gauges if ent == "Total" else [ent]
        yk = E.series(dk, g).values
        res = {}
        for name in M.METHODS:
            f = M.run_method(name, yk, H)
            if f is not None:
                res[name] = f
        bt = E.backtest(yk)
        res[E.ENSEMBLE + " (nuevo)"] = E.forecast(yk, H, E.ENSEMBLE, bt) if (bt["Peso ensamble"].notna().any()) else None
        # ensamble anterior: promedio simple de los 3 mejores estadísticos de un solo backtest
        obt = OE.backtest(yk)
        res["Ensamble 3 mejores (anterior)"] = OE.forecast(yk, H, OE.ENSEMBLE, obt) if len([r for r in obt["Método"] if r in OM.STATISTICAL]) >= 3 else None
        if mlo:
            d_, e_ = OML.selection(mlo["final"], g)
            res["ML anterior (150 árboles)"] = (d_ + e_)[:H]
        if mln:
            d_, e_ = MLm.selection(mln["final"], g)
            res["ML nuevo (afinado)"] = (d_ + e_)[:H]
        for name, f in res.items():
            if f is not None:
                meth.setdefault((name, ent), []).append(err(ent, k, f))
        meth.setdefault(("(motor anterior, AUTO)", ent), []).append(out_old[ent][-1])
        meth.setdefault(("(motor nuevo, AUTO)", ent), []).append(out_new[ent][-1])
    print(f"   listo ({time.time() - t0:.0f} s)", flush=True)

# ---------- tablas
def md(df, index=True):
    """Tabla en formato markdown (sin librerías extra)."""
    d = df.reset_index() if index else df
    head = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n"
    return head + "".join("| " + " | ".join(f"{v:.1f}" if isinstance(v, float) else str(v) for v in r) + " |\n" for r in d.itertuples(index=False))


rows = []
for ent in ENT:
    a_rep, n_rep = rep_old[ent], rep_new[ent]
    a_out, n_out = float(np.nanmean(out_old[ent])), float(np.nanmean(out_new[ent]))
    rows.append({"Selección": ent, "Antes: reportado": a_rep, "Antes: real fuera de muestra": a_out,
                 "Ahora: reportado": n_rep, "Ahora: real fuera de muestra": n_out, "Mejora real (pp)": a_out - n_out})
tab = pd.DataFrame(rows)
mt = pd.DataFrame([{"Método": k[0], "Selección": k[1], "Error fuera de muestra %": float(np.nanmean(v)), "Orígenes": len(v)} for k, v in meth.items()])
piv = mt.pivot_table(index="Método", columns="Selección", values="Error fuera de muestra %")[ENT]
piv["Promedio calibres"] = piv[gauges].mean(axis=1)
piv = piv.sort_values("Total")
per_origin = pd.DataFrame({f"{ent} ({'antes' if w == 0 else 'ahora'})": (out_old if w == 0 else out_new)[ent] for ent in ENT for w in (0, 1)}, index=[f"origen -{k}m" for k in ORIGINS]).T

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
print("\n=== ANTES / DESPUÉS (error %, menor es mejor) ===")
print(tab.round(1).to_string(index=False))
vt = pd.DataFrame({f"AUTO con margen {v:g} pp": {ent: float(np.nanmean(var_err[(v, ent)])) for ent in ENT} for v in VARIANTS}).T
vt["Promedio calibres"] = vt[gauges].mean(axis=1)
print("\n=== Regla del método automático (error real fuera de muestra) ===")
print(vt.round(1).to_string())
print(f"\n=== Cobertura del rango de confianza (80% nominal): nuevo {np.mean(cov_new) * 100:.0f}% | fórmula anterior {np.mean(cov_old) * 100:.0f}% (de {len(cov_new)} casos de 6 meses) ===")
print("\n=== Error real fuera de muestra por origen ===")
print(per_origin.round(1).to_string())
print("\n=== Cada método por separado, error real fuera de muestra (promedio de 3 orígenes) ===")
print(piv.round(1).to_string())

with open("resultados_motor.json", "w") as f:
    json.dump({"antes_despues": tab.to_dict("records"), "por_origen": per_origin.to_dict(), "metodos": piv.to_dict(),
               "ml_nuevo": ml_new["desc"] if ml_new else None, "reglas_auto": vt.to_dict(), "cobertura": {"nuevo": float(np.mean(cov_new)), "anterior": float(np.mean(cov_old))}, "origenes_meses_antes_del_final": ORIGINS}, f, ensure_ascii=False, indent=1, default=float)
with open("resultados_motor.md", "w") as f:
    f.write("# Antes y después del motor (error %, menor es mejor)\n\n" + md(tab, index=False) + "\n## Por origen\n\n" + md(per_origin.round(1))
            + "\n## Regla del método automático\n\n" + md(vt.round(1)) + f"\nCobertura del rango 80%: nuevo {np.mean(cov_new) * 100:.0f}%, anterior {np.mean(cov_old) * 100:.0f}%\n"
            + "\n## Cada método, error real fuera de muestra\n\n" + md(piv.round(1)) + "\n")
print(f"\nTiempo total: {time.time() - t0:.0f} s")
