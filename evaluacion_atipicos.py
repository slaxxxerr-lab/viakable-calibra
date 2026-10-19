"""¿Corregir los meses atípicos mejora el pronóstico? Prueba con el modelo de machine learning (la de métodos estadísticos está en la app).

En cada ventana de validación se detectan los atípicos (por calibre y por producto) usando SOLO datos anteriores a la ventana, se corrigen
todos con su valor esperado, se entrena el modelo con esos datos y se mide el error contra lo que de verdad se vendió.
Uso:  python evaluacion_atipicos.py [archivo.csv]
"""
import sys

import numpy as np
import pandas as pd

from viakable_forecast import atypical as AT
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
spec = MLm.FAMILY_SPECS["gb"][0]
rows = []
for k, h in E.windows_for(full):
    cut = months[N - k - 1]
    tr = full[full["ym"] <= cut].copy()
    det = AT.detect_all(tr)
    cor = AT.apply_corrections(tr, det, {key: {"a": "corregir"} for key in det["key"]})
    det_g = det[det["nivel"] == "Calibre"]
    cor_g = AT.apply_corrections(tr, det_g, {key: {"a": "corregir"} for key in det_g["key"]})
    res = {}
    for label, d in (("crudo", tr), ("corregido (todos)", cor), ("corregido (solo calibres)", cor_g)):
        r = MLm.train_forecast(d, holdout=0, steps=h, spec=spec)
        res[label] = r
    for ent in ["Total"] + gauges:
        g = gauges if ent == "Total" else [ent]
        act = E.series(full, g).values[N - k:N - k + h]
        for label, r in res.items():
            dd, ee = MLm.selection(r, g)
            rows.append({"ventana": k, "sel": ent, "tipo": label, "error": M.error_pct(act, (dd + ee)[:h]), "corregidos": len(det)})
df = pd.DataFrame(rows)
tab = df.pivot_table(index="sel", columns="tipo", values="error", aggfunc="mean")
tab["Mejora todos (pts)"] = tab["crudo"] - tab["corregido (todos)"]
tab["Mejora solo calibres (pts)"] = tab["crudo"] - tab["corregido (solo calibres)"]
print("Atípicos corregidos por ventana:", df.groupby("ventana")["corregidos"].first().to_dict())
print(tab.round(2).loc[["Total"] + gauges].to_string())
