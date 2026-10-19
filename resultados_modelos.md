# Prueba fuera de muestra de todos los modelos

MAPE (WAPE si hay meses en cero) fuera de muestra; los orígenes se traslapan, así que no son independientes.

Archivo: Viakable_Tons_Sales.csv. Fecha: 2026-10-09. Orígenes: [6, 7, 8, 9, 10, 11, 12].

| Método | Tipo | Total | 6 | 10 | 12 | 14 | 16 | 18 | Orígenes | Promedio calibres |
|---|---|---|---|---|---|---|---|---|---|---|
| TSB | Estadístico | 7.0 | 21.0 | 6.5 | 10.7 | 9.6 | 9.2 | 30.8 | 5 | 14.6 |
| Promedio móvil 3 meses | Estadístico | 7.2 | 22.3 | 7.4 | 10.6 | 7.5 | 9.0 | 33.1 | 5 | 15.0 |
| SARIMA automático | Estadístico | 7.3 | 22.1 | 6.2 | 11.1 | 10.0 | 9.3 | 31.1 | 5 | 15.0 |
| ETS automático | Estadístico | 7.5 | 24.5 | 6.2 | 10.8 | 10.0 | 8.8 | 31.7 | 5 | 15.3 |
| Suavización exponencial simple | Estadístico | 7.6 | 26.1 | 6.3 | 10.7 | 10.1 | 9.4 | 31.4 | 5 | 15.7 |
| AUTO actual (sin los modelos nuevos) | Selección automática | 7.9 | 22.9 | 7.4 | 10.7 | 9.8 | 13.5 | 31.6 | 5 | 16.0 |
| Ensamble ponderado | Ensamble | 8.2 | 23.3 | 7.6 | 10.9 | 9.5 | 10.1 | 32.2 | 5 | 15.6 |
| Croston-SBA | Estadístico | 9.2 | 20.6 | 8.9 | 11.6 | 12.4 | 11.8 | 29.4 | 5 | 15.8 |
| Machine learning (boosting global) | Machine learning | 10.6 | 20.6 | 9.8 | 12.3 | 10.6 | 11.6 | 39.1 | 5 | 17.4 |
| AUTO con todos los modelos | Selección automática | 10.6 | 22.9 | 10.9 | 13.7 | 9.8 | 13.5 | 31.6 | 5 | 17.1 |
| Bosque aleatorio global | Machine learning | 10.8 | 20.9 | 8.2 | 12.1 | 10.6 | 11.2 | 39.1 | 5 | 17.0 |
| XGBoost global | Machine learning | 11.0 | 20.3 | 9.5 | 13.4 | 11.8 | 12.1 | 34.9 | 5 | 17.0 |
| Ridge con rezagos | Machine learning | 11.4 | 20.1 | 10.4 | 15.3 | 12.5 | 14.1 | 38.8 | 5 | 18.5 |
| Holt amortiguado | Estadístico | 11.8 | 33.6 | 7.4 | 10.4 | 7.2 | 11.4 | 35.1 | 5 | 17.5 |
| Apilado (stacking) | Ensamble | 13.9 | 22.2 | 12.2 | 15.6 | 10.8 | 9.5 | 32.8 | 2 | 17.2 |
| Mismo mes año anterior × crecimiento | Estadístico | 15.3 | 19.8 | 18.4 | 15.9 | 20.1 | 17.3 | 35.1 | 5 | 21.1 |
| Regresión desestacionalizada | Estadístico | 15.9 | 18.2 | 18.3 | 18.1 | 20.2 | 22.1 | 33.4 | 1 | 21.7 |
| Red neuronal (MLP) | Red neuronal | 18.8 | 28.7 | 19.3 | 21.1 | 20.4 | 23.3 | 44.5 | 5 | 26.2 |
| Holt-Winters multiplicativo | Estadístico | 20.3 | 68.9 | 14.0 | 21.6 | 15.6 | 22.0 | 40.1 | 1 | 30.4 |
