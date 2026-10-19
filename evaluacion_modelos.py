"""Prueba fuera de muestra de TODOS los modelos (Fase 6A).

Como en evaluacion_motor.py, se repite todo el proceso (afinar, entrenar, pronosticar) como si estuviéramos 6 a 12 meses antes del final,
usando solo datos anteriores, y se compara contra lo que de verdad se vendió en los 6 meses siguientes. El error es MAPE (WAPE si hay
meses en cero). No se reporta R².
Uso:  python evaluacion_modelos.py [archivo.csv]    (escribe resultados_modelos.json y resultados_modelos.md)
"""
import datetime
import json
import sys
import time

import numpy as np
import pandas as pd

from viakable_forecast import data as D
from viakable_forecast import engine as E
from viakable_forecast import methods as M
from viakable_forecast import ml as MLm

path = sys.argv[1] if len(sys.argv) > 1 else "Viakable_Tons_Sales.csv"
raw = D.load_file(path)
prep = D.prepare(raw, D.detect_columns(raw))
full, gauges = prep.monthly, prep.gauges
months = list(pd.date_range(full["ym"].min(), full["ym"].max(), freq="MS"))
N = len(months)
ORIGINS = [6, 7, 8, 9, 10, 11, 12]
H = 6
ENT = ["Total"] + gauges
NEW = {MLm.MODEL_NAMES[f] for f in MLm.EXTRA_FAMILIES} | {E.STACK}
DEFAULT_EXCLUDE = set(E.AUTO_EXCLUDE)
UNITS = [[g] for g in gauges]


def actual(ent, k):
    return E.series(full, gauges if ent == "Total" else [ent]).values[N - k:N - k + H]


res = {}          # (modelo, entidad) -> {origen: error}
t0 = time.time()
for k in ORIGINS:
    dk = full[full["ym"] <= months[N - k - 1]].copy()
    print(f"Origen -{k} meses ({N - k} de entrenamiento)…", flush=True)
    ml = MLm.fit_everything(dk, E.windows_for(dk))
    models = E.ml_models(ml)
    for ent in ENT:
        g = gauges if ent == "Total" else [ent]
        yk = E.series(dk, g).values
        a = actual(ent, k)
        bt = E.selection_backtest(dk, g, None, E.TON, None, ml)
        fcs = {}
        for name in M.METHODS:
            f = M.run_method(name, yk, H)
            if f is not None:
                fcs[name] = f
        fcs[E.ENSEMBLE] = E.forecast(yk, H, E.ENSEMBLE, bt) if bt["Peso ensamble"].notna().any() else None
        for name, r in models.items():
            d, e = MLm.selection(r["final"], g)
            fcs[name] = (d + e)[:H]
        fcs[E.STACK] = E.stack_forecast(bt, yk, H, g, None, E.TON, None, ml) if bt["Peso apilado"].notna().any() else None
        for name, f in fcs.items():
            if f is not None:
                res.setdefault((name, ent), {})[k] = M.error_pct(a, np.asarray(f)[:H])
    # selección automática: regla anterior (sin los modelos nuevos) contra regla con todos
    for label, excl in (("AUTO actual (sin los modelos nuevos)", NEW), ("AUTO con todos los modelos", set())):
        E.AUTO_EXCLUDE = set(excl)
        pr = E.project(dk, UNITS, None, H, E.AUTO, E.TON, "sum", 1.0, None, 0.3, None, None, ml)
        for ent in ENT:
            f = pr.fc if ent == "Total" else next(d["fc"] for d in pr.detail if d["label"] == ent)
            res.setdefault((label, ent), {})[k] = M.error_pct(actual(ent, k), f[:H])
    E.AUTO_EXCLUDE = set(DEFAULT_EXCLUDE)
    print(f"   listo ({time.time() - t0:.0f} s)", flush=True)

# Comparación justa: solo los orígenes donde existen TODOS los modelos individuales (el ML necesita al menos 20 meses de entrenamiento).
names = sorted({k[0] for k in res})
individual = [n for n in names if n not in (E.STACK, "AUTO actual (sin los modelos nuevos)", "AUTO con todos los modelos", M.METHODS and "Regresión desestacionalizada", "Holt-Winters multiplicativo")]
common = sorted(set.intersection(*[set(res[(n, "Total")]) for n in individual]))
print("Orígenes comunes (todos los modelos individuales):", common)


def mean_over(name, ent, origins):
    v = [res[(name, ent)][o] for o in origins if o in res.get((name, ent), {})]
    return float(np.nanmean(v)) if v else np.nan


rows = [{"Método": n, **{e: mean_over(n, e, common) for e in ENT}, "Orígenes": len([o for o in common if o in res[(n, "Total")]])} for n in names]
tab = pd.DataFrame(rows).set_index("Método")
tab["Promedio calibres"] = tab[gauges].mean(axis=1)
tab = tab[tab["Orígenes"] > 0].sort_values("Total")
wins = {e: tab[tab["Orígenes"] == len(common)][e].idxmin() for e in ENT}
type_of = {**E.TYPE_OF, "AUTO actual (sin los modelos nuevos)": "Selección automática", "AUTO con todos los modelos": "Selección automática"}
tab.insert(0, "Tipo", [type_of.get(n, "") for n in tab.index])
# el apilado solo existe donde hay 3+ modelos elegibles: se compara contra sus rivales en esos mismos orígenes
st_orig = sorted(res.get((E.STACK, "Total"), {}))
stack_cmp = {n: mean_over(n, "Total", st_orig) for n in ("Apilado (stacking)", "Ensamble ponderado", "TSB", "ETS automático", "Machine learning (boosting global)", "Ridge con rezagos")} if st_orig else {}
pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 30)
print("\n=== Error real fuera de muestra, promedio de", len(common), "orígenes comunes (menor es mejor) ===")
print(tab.round(1).to_string())
print("\nGanador por selección:", wins)
print("Apilado, en los orígenes donde existe", st_orig, "(error total):", {k: round(v, 1) for k, v in stack_cmp.items()})

meta = {"archivo": path, "fecha": datetime.date.today().isoformat(), "origenes": common, "horizonte": H,
        "nota": "MAPE (WAPE si hay meses en cero) fuera de muestra; los orígenes se traslapan, así que no son independientes.",
        "origenes_comunes": common, "apilado": stack_cmp, "apilado_origenes": st_orig}
with open("resultados_modelos.json", "w") as f:
    json.dump({"meta": meta, "tabla": tab.reset_index().to_dict("records"), "ganadores": wins}, f, ensure_ascii=False, indent=1, default=float)
with open("resultados_modelos.md", "w") as f:
    head = "| Método | " + " | ".join(tab.columns) + " |\n|" + "---|" * (len(tab.columns) + 1) + "\n"
    body = "".join("| " + " | ".join([str(i)] + [f"{v:.1f}" if isinstance(v, float) else str(v) for v in r]) + " |\n" for i, r in zip(tab.index, tab.itertuples(index=False)))
    f.write(f"# Prueba fuera de muestra de todos los modelos\n\n{meta['nota']}\n\nArchivo: {path}. Fecha: {meta['fecha']}. Orígenes: {ORIGINS}.\n\n" + head + body)
print(f"\nTiempo total: {time.time() - t0:.0f} s")
