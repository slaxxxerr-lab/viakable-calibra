"""Verifica que ningún pronóstico usa datos posteriores al punto de corte (sin fuga de datos).

Idea: se alteran a propósito los meses "futuros" (los que quedan después del corte de cada ventana de validación) y se
comprueba que el pronóstico de esa ventana NO cambia. Si cambiara, el modelo estaría mirando el futuro.
Uso:  python verificar_fuga.py [archivo.csv]
"""
import sys

import numpy as np
import pandas as pd

from viakable_forecast import data as D
from viakable_forecast import engine as E
from viakable_forecast import ml as MLm

path = sys.argv[1] if len(sys.argv) > 1 else "Viakable_Tons_Sales.csv"
raw = D.load_file(path)
m = D.prepare(raw, D.detect_columns(raw)).monthly
months = sorted(m["ym"].unique())
rng = np.random.default_rng(0)
ok = True


def alter_future(df, k, new_family=False):
    """Copia de los datos con los últimos k meses multiplicados por factores al azar (y, opcional, una familia nueva solo en el futuro)."""
    cut = months[len(months) - k]
    d = df.copy()
    fut = d["ym"] >= cut
    d.loc[fut, "qty"] = d.loc[fut, "qty"] * rng.uniform(0.2, 5.0, fut.sum())
    if new_family:
        extra = d[fut].head(40).copy()
        extra["fam"] = "FAMILIA_SOLO_FUTURO"
        extra["prod"] = "PRODUCTO_NUEVO_" + extra["prod"].astype(str)
        d = pd.concat([d, extra], ignore_index=True)
    return d


print("1) Machine learning: el pronóstico de cada ventana no cambia si se alteran los meses posteriores al corte")
for spec in (MLm.SPECS[0], MLm.SPECS[-1]):
    for k, h in E.windows_for(m):
        base = MLm.train_forecast(m, holdout=k, steps=h, spec=spec)
        alt = MLm.train_forecast(alter_future(m, k, new_family=True), holdout=k, steps=h, spec=spec)
        # se comparan las series que ya existían (la familia y los productos nuevos del futuro solo agregan series nuevas)
        diff = np.abs(base.fc - alt.fc[alt.meta.index.isin(base.meta.index)]).max()
        print(f"   {MLm.spec_name(spec)[:42]:42s} ventana k={k:2d}: diferencia máxima = {diff:.2e} t", "OK" if diff < 1e-6 else "FUGA")
        ok &= diff < 1e-6

print("2) Métodos estadísticos: el pronóstico de cada ventana no cambia si se alteran los meses posteriores al corte")
y = E.series(m).values
n = len(y)
for k, h in E.windows_for(m):
    y2 = y.copy()
    y2[n - k:] = y2[n - k:] * rng.uniform(0.2, 5.0, k)
    a, b = E._stat_cv(y.tobytes(), ((k, h),)), E._stat_cv(y2.tobytes(), ((k, h),))
    worst = 0.0
    for name in a:
        if name == "__t__" or a[name][0] is None:
            continue
        worst = max(worst, float(np.abs(a[name][0] - b[name][0]).max()))
    print(f"   ventana k={k:2d}: diferencia máxima entre todos los métodos = {worst:.2e} t", "OK" if worst < 1e-9 else "FUGA")
    ok &= worst < 1e-9

print("3) Control: si se altera el PASADO, los pronósticos sí deben cambiar (confirma que la prueba detecta diferencias)")
k, h = E.windows_for(m)[0]
past = m.copy()
cut = months[len(months) - k]
mask = past["ym"] < cut
past.loc[mask, "qty"] = past.loc[mask, "qty"] * rng.uniform(0.2, 5.0, mask.sum())
d = np.abs(MLm.train_forecast(m, holdout=k, steps=h, spec=MLm.SPECS[0]).fc - MLm.train_forecast(past, holdout=k, steps=h, spec=MLm.SPECS[0]).fc).max()
print(f"   diferencia máxima al alterar el pasado = {d:.2f} t", "OK (la prueba detecta cambios)" if d > 1e-3 else "LA PRUEBA NO DETECTA NADA")
ok &= d > 1e-3


print("4) Precio del metal como variable: alterar los precios posteriores al corte no cambia el pronóstico de la ventana")
months_all = list(__import__("pandas").date_range(m["ym"].min(), m["ym"].max(), freq="MS"))
px = {"cu": 9000 + np.cumsum(rng.normal(0, 150, len(months_all))), "al": 2400 + np.cumsum(rng.normal(0, 40, len(months_all)))}
for k, h in E.windows_for(m):
    T0 = len(months_all) - k
    px2 = {key: v.copy() for key, v in px.items()}
    for key in px2:
        px2[key][T0:] = px2[key][T0:] * rng.uniform(0.3, 3.0, k)
    a = MLm.train_forecast(m, holdout=k, steps=h, spec=MLm.SPECS[0], exog=px)
    b = MLm.train_forecast(m, holdout=k, steps=h, spec=MLm.SPECS[0], exog=px2)
    diff = np.abs(a.fc - b.fc).max()
    print(f"   ventana k={k:2d}: diferencia máxima = {diff:.2e} t", "OK" if diff < 1e-9 else "FUGA")
    ok &= diff < 1e-9

print("5) Modelos nuevos (XGBoost, bosque, Ridge, red neuronal): alterar el futuro no cambia el pronóstico de la ventana")
for fam in ("xgb", "rf", "ridge", "mlp"):
    if fam == "xgb" and not MLm.HAVE_XGB:
        continue
    sp = MLm.FAMILY_SPECS[fam][0]
    worst = 0.0
    for k, h in E.windows_for(m):
        base = MLm.train_forecast(m, holdout=k, steps=h, spec=sp)
        alt = MLm.train_forecast(alter_future(m, k, new_family=True), holdout=k, steps=h, spec=sp)
        worst = max(worst, np.abs(base.fc - alt.fc[alt.meta.index.isin(base.meta.index)]).max())
    print(f"   {MLm.MODEL_NAMES[fam]:26s} diferencia máxima en las 4 ventanas = {worst:.2e} t", "OK" if worst < 1e-6 else "FUGA")
    ok &= worst < 1e-6

print("\nRESULTADO:", "sin fuga de datos" if ok else "HAY FUGA, revisar")
sys.exit(0 if ok else 1)
