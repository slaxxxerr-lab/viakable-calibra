# Antes y después del motor (error %, menor es mejor)

| Selección | Antes: reportado | Antes: real fuera de muestra | Ahora: reportado | Ahora: real fuera de muestra | Mejora real (pp) |
|---|---|---|---|---|---|
| Total | 7.1 | 13.6 | 11.1 | 10.4 | 3.2 |
| 6 | 10.4 | 25.2 | 16.8 | 21.1 | 4.0 |
| 10 | 7.9 | 12.5 | 12.2 | 11.0 | 1.5 |
| 12 | 10.5 | 14.6 | 14.0 | 13.1 | 1.4 |
| 14 | 9.0 | 15.3 | 13.5 | 11.5 | 3.8 |
| 16 | 9.8 | 12.4 | 9.2 | 8.8 | 3.7 |
| 18 | 25.4 | 28.7 | 26.5 | 29.7 | -1.0 |

## Por origen

| index | origen -6m | origen -9m | origen -12m |
|---|---|---|---|
| Total (antes) | 8.4 | 14.8 | 17.6 |
| Total (ahora) | 7.0 | 8.5 | 15.8 |
| 6 (antes) | 26.4 | 28.0 | 21.1 |
| 6 (ahora) | 19.2 | 26.1 | 18.1 |
| 10 (antes) | 9.2 | 12.2 | 15.9 |
| 10 (ahora) | 8.9 | 7.5 | 16.5 |
| 12 (antes) | 11.9 | 13.9 | 17.8 |
| 12 (ahora) | 12.0 | 10.2 | 17.1 |
| 14 (antes) | 9.8 | 20.3 | 16.0 |
| 14 (ahora) | 11.9 | 9.5 | 13.2 |
| 16 (antes) | 12.2 | 15.5 | 9.7 |
| 16 (ahora) | 11.9 | 9.5 | 4.9 |
| 18 (antes) | 25.4 | 35.2 | 25.5 |
| 18 (ahora) | 28.7 | 34.9 | 25.5 |

## Regla del método automático

| index | Total | 6 | 10 | 12 | 14 | 16 | 18 | Promedio calibres |
|---|---|---|---|---|---|---|---|---|
| AUTO con margen 0 pp | 10.3 | 21.1 | 12.0 | 13.4 | 11.7 | 10.2 | 31.0 | 16.6 |
| AUTO con margen 1 pp | 10.2 | 21.1 | 12.3 | 13.3 | 11.3 | 8.8 | 31.0 | 16.3 |
| AUTO con margen 2 pp | 10.4 | 21.1 | 11.0 | 13.1 | 11.5 | 8.8 | 29.7 | 15.9 |
| AUTO con margen 99 pp | 10.5 | 21.5 | 11.0 | 13.1 | 11.4 | 8.8 | 29.4 | 15.9 |

Cobertura del rango 80%: nuevo 71%, anterior 53%

## Cada método, error real fuera de muestra

| Método | Total | 6 | 10 | 12 | 14 | 16 | 18 | Promedio calibres |
|---|---|---|---|---|---|---|---|---|
| TSB | 9.2 | 23.4 | 10.0 | 12.1 | 11.5 | 8.7 | 28.8 | 15.7 |
| ETS automático | 10.0 | 20.9 | 9.8 | 12.9 | 11.7 | 8.1 | 29.2 | 15.4 |
| Suavización exponencial simple | 10.1 | 22.9 | 9.9 | 12.7 | 11.9 | 8.7 | 29.5 | 15.9 |
| (motor nuevo, AUTO) | 10.4 | 21.1 | 11.0 | 13.1 | 11.5 | 8.8 | 29.7 | 15.9 |
| SARIMA automático | 10.6 | 21.2 | 9.6 | 13.1 | 11.4 | 8.4 | 28.4 | 15.4 |
| Promedio móvil 3 meses | 10.8 | 21.1 | 10.9 | 13.3 | 9.9 | 10.2 | 31.0 | 16.1 |
| Croston-SBA | 11.0 | 22.2 | 12.1 | 13.6 | 14.2 | 11.0 | 27.6 | 16.8 |
| ML nuevo (afinado) | 11.1 | 18.6 | 12.6 | 11.9 | 11.2 | 12.4 | 41.8 | 18.1 |
| Ensamble ponderado (nuevo) | 11.1 | 21.5 | 11.0 | 13.1 | 11.4 | 8.8 | 29.4 | 15.9 |
| Ensamble 3 mejores (anterior) | 11.5 | 20.6 | 11.1 | 13.3 | 11.0 | 9.5 | 30.5 | 16.0 |
| ML anterior (150 árboles) | 11.7 | 20.8 | 14.6 | 13.4 | 14.3 | 19.0 | 112.5 | 32.5 |
| Holt amortiguado | 13.6 | 33.8 | 10.4 | 13.1 | 8.9 | 13.3 | 32.1 | 18.6 |
| (motor anterior, AUTO) | 13.6 | 25.2 | 12.5 | 14.6 | 15.3 | 12.4 | 28.7 | 18.1 |
| Regresión desestacionalizada | 15.9 | 18.2 | 18.3 | 18.1 | 20.2 | 22.1 | 33.4 | 21.7 |
| Mismo mes año anterior × crecimiento | 16.5 | 25.1 | 19.5 | 16.1 | 19.2 | 14.8 | 31.4 | 21.0 |
| Holt-Winters multiplicativo | 20.3 | 68.9 | 14.0 | 21.6 | 15.6 | 22.0 | 40.1 | 30.4 |

