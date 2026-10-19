# Viakable Calibra · Pronóstico, planeación y CRM

Aplicación web (Streamlit) que pronostica la venta de cable eléctrico por calibre, la convierte en planes de inventario, compra de metal, margen y capacidad, y suma un CRM ligero con alertas. Está en español (México) y tiene versión en inglés.

> Prototipo académico. Los precios, costos, capacidades y elasticidades son valores ilustrativos y no representan cifras reales de Viakable. El nombre de la app se cambia en una sola línea: `APP_NAME` en `viakable_forecast/config.py`.

## Instalación

Necesitas Python 3.13 o 3.14. En una Mac:

**Opción 1, con doble clic:** abre `iniciar.command`. La primera vez crea el entorno `.venv` e instala las librerías (unos minutos); después solo abre la app en http://localhost:8501. Si macOS bloquea el archivo, haz clic derecho, Abrir.

**Opción 2, en la terminal:**
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

XGBoost necesita una librería de OpenMP en Mac. Si no carga, ejecuta una vez `bash arreglar_openmp_mac.sh`. Sin ella la app funciona, solo sin ese modelo.

## Conectar una IA gratis para las preguntas abiertas (opcional)

Sin esto, el Asistente contesta datos de la app, metodología y hechos de Wikipedia; solo las preguntas abiertas (consejos, tendencias, negociación, etc.) necesitan una IA.

- **Ollama, en tu Mac (recomendado: gratis y privado):** instala la app desde https://ollama.com/download (o `brew install ollama`), ábrela y en la terminal ejecuta `ollama pull qwen2.5:3b` (unos 2 GB; con poca memoria, `ollama pull llama3.2:1b`). Reinicia Calibra: el Asistente lo detecta solo (revisa http://localhost:11434). La pregunta nunca sale de tu computadora.
- **Gemini o Groq, en la nube (opcional):** pide una llave gratuita en https://aistudio.google.com/apikey (Gemini) o https://console.groq.com/keys (Groq) y agrégala en `.streamlit/secrets.toml` en las secciones `[gemini]` o `[groq]` (ver `.streamlit/secrets.toml.example`). Los límites del nivel gratuito cambian con el tiempo y se ven en tu cuenta de cada proveedor (AI Studio y la consola de Groq); consúltalos ahí. En la nube viaja el texto de la pregunta (sin correos, teléfonos ni enlaces), y el proveedor puede usarlo según sus términos.
- Orden de uso: Ollama, Gemini, Groq. En la sección Asistente puedes apagar Wikipedia o la IA.

## Llaves de API (opcionales)

Solo las usan las fuentes de datos de mercado. Copia `.streamlit/secrets.toml.example` como `.streamlit/secrets.toml` y escribe tus llaves. Ese archivo no se sube a GitHub. Sin llaves, la app usa fuentes gratuitas de respaldo.

| Llave | Para qué sirve | Dónde se pide |
|---|---|---|
| `[banxico] token` | Tipo de cambio FIX oficial | https://www.banxico.org.mx/SieAPIRest/service/v1/token |
| `[fred] api_key` | Histórico mensual de cobre y aluminio | https://fred.stlouisfed.org/docs/api/api_key.html |
| `[alphavantage] api_key` | Alternativa para ese histórico | https://www.alphavantage.co/support/#api-key |

Nunca escribas llaves en el código ni las subas a GitHub.

## Cómo se usa

La app abre con una guía de 10 pasos la primera vez; puedes reabrirla con **Guía rápida**. Los datos de Viakable que tengas en la carpeta (`Viakable_Tons_Sales.csv`) se cargan solos; en la versión publicada se empieza con datos de ejemplo y subes tu archivo en **Datos**.

**Modos.** *Simple* muestra lo esencial: inicio, pronóstico, plan, metales, clientes, oportunidades, portafolio, noticias y datos. *Avanzado* agrega métodos y comparativos, consenso, machine learning y SHAP, meses atípicos, señales, escenarios, Laboratorio, anomalías, seguimiento y la sección Equipo. Al pasar a Simple los controles avanzados se esconden, pero conservan sus valores.

**Secciones.**
- **Inicio:** mosaicos de KPI con clic, gráfica del pronóstico, "Qué atender" (calibres que caen, pronósticos poco confiables, exportación, capacidad, clientes que compran menos, tareas vencidas, anomalías, desviaciones contra pronósticos congelados, movimientos de metales y noticias de relevancia alta), cola de anomalías de ritmo de pedidos, últimas noticias y el botón **Resumen del mes**.
- **Pronóstico:** por calibre, mercado y horizonte (3, 6 o 12 meses), en toneladas o pesos, con rango de confianza del 80%, panel "¿Qué pasa si…?", cascada "Por qué cambió el pronóstico" y descarga a Excel.
- **Planeación:** inventario de seguridad, producción, compras de cobre y aluminio, margen y capacidad, con escenarios guardados.
- **Metales:** precios en vivo, camino de 12 meses, traspaso al precio de venta, impacto de un movimiento del metal y recomendación de comprar ahora o después.
- **Clientes y Oportunidades:** perfil por cliente, oportunidades por etapa, actividades y tareas. Mover una oportunidad o cambiar sus toneladas actualiza el pronóstico al instante.
- **Portafolio:** análisis ABC–XYZ con sugerencias (impulsar, revisar, considerar descontinuar) y sensibilidad del volumen al precio del metal.
- **Noticias:** monitor de aranceles, T-MEC, normas y metales.
- **Datos:** subir archivos, revisar columnas, regla de exportación, corregir cifras y plantillas.
- **Asistente:** contesta casi cualquier tema con una arquitectura híbrida y sin API de pago. Enruta cada pregunta: datos de negocio y escenarios los calcula el motor y los redacta con plantillas; conceptos y metodología salen de una base de conocimiento; hechos generales, de Wikipedia con su enlace; y preguntas abiertas, de una IA local (Ollama) o gratuita en la nube (Gemini o Groq) si la conectaste. La IA nunca genera cifras de negocio y cada respuesta lleva su etiqueta (Datos de la app, Metodología, Wikipedia o Respuesta general (IA)). Los escenarios solo se aplican con Aplicar escenario. Ver "Conectar una IA (gratis)" más abajo.
- **Riesgo (modo avanzado):** pronóstico probabilístico P10, P50 y P90 por calibre, simulación Monte Carlo del margen (demanda, cobre, aluminio, tipo de cambio y arancel) con margen en riesgo y probabilidad de quedar bajo un umbral, compra óptima de cobre y aluminio (modelo del vendedor de periódicos) y el efecto de fijar parte de las compras al precio de hoy.
- **Avanzado:** Señales, Laboratorio de modelos y Equipo (roles y bitácora).

**Personalizar** (botón arriba a la derecha): idioma (español o inglés), tema (claro, oscuro, sistema), color de acento, tipografía, densidad, unidad de medida, rol y nombre. Se guardan en `ajustes.json`; en la versión publicada duran mientras la página siga abierta.

**Todo se recalcula en vivo:** cualquier cambio (un dato, un precio, un escenario, una oportunidad) actualiza todas las secciones conectadas.

## Experiencia dinámica

- **Animaciones sutiles:** los números de los indicadores cuentan hasta su nuevo valor y muestran una flechita con el cambio; el rótulo de precios destella en verde o rojo cuando un precio se actualiza; la línea del pronóstico se dibuja al cargar y las gráficas se mueven suavemente cuando cambian los datos. Se pueden apagar en Personalizar (Animaciones) y se apagan solas si tu sistema pide reducir el movimiento.
- **Modo presentación** (botón Presentar): una historia guiada de unos 2 minutos con avance automático: pronóstico actual, precio del cobre en vivo y su efecto (simula +10%), una noticia regulatoria, exención arancelaria del T-MEC (arancel 0%), capital liberado, anomalías y alertas, y por qué cambió el pronóstico. Tiene Anterior, Siguiente, Pausar y Salir; al salir se restauran tus valores. Funciona sin internet con los precios y noticias guardados. Con datos sin exportación, el paso del arancel lo dice y no mueve nada.
- **Indicador de frescura** bajo el encabezado: datos hasta, modelos entrenados, hora de los precios y de las noticias (en ámbar si llevan mucho sin actualizarse).
- **Pantallas pequeñas:** se adapta a tableta y teléfono (menú con desplazamiento horizontal, mosaicos en dos columnas).

## Reportes y plantillas

- **Resumen del mes** (Inicio): una página con pronóstico, qué cambió y por qué, alertas, anomalías, metales y compra, noticias y oportunidades. Se ve en pantalla y se descarga en PDF o Excel.
- **Reporte completo en Excel** (Inicio, dentro del resumen): resumen, pronóstico total y por calibre, plan, capacidad, portafolio, oportunidades y notas de calidad de datos. Sale en el idioma elegido.
- **Pronóstico a Excel** (Pronóstico): pronóstico del total, por calibre y comparación de métodos.
- **Plantillas en Excel** (Datos y Metales): ventas, indicadores y precios de metal.
- **Línea de comandos**, sin abrir la app:
  ```
  python -m viakable_forecast.report ventas.xlsx --horizonte 6 --salida reporte.xlsx --exportacion ch "VALUE" --idioma es
  ```
  Usa precios y parámetros de referencia (no los precios en vivo), solo métodos estadísticos y no incluye CRM.

## Metodología

- **Validación cruzada de origen móvil.** Cada método se evalúa con ventanas que cortan 6, 9, 12 y 15 meses antes del final, entrenando solo con datos anteriores y pronosticando 6 meses. La precisión es 100 menos el error porcentual (MAPE o WAPE) contra meses reales; nunca se reporta R². Si un cambio no mejora la precisión, la app lo dice.
- **Métodos.** Promedio móvil, suavización exponencial, Holt amortiguado, Holt-Winters, regresión desestacionalizada, Croston-SBA, TSB, ETS y SARIMA automáticos (statsmodels), modelo por señales y un ensamble ponderado por el inverso del error. El método automático usa el ensamble salvo que otro le gane por 2 puntos o más. Con 30 meses de historia, los métodos estadísticos simples suelen ganarle al machine learning; la app lo muestra.
- **Rango de confianza del 80%.** Sale de los errores reales de las ventanas de validación, no de una fórmula fija; con menos de 3 ventanas se ensancha.
- **Machine learning y Laboratorio de modelos.** Gradient boosting, HistGradientBoosting, XGBoost, bosque aleatorio, Ridge con rezagos y red neuronal (MLP), más un apilado ponderado por error de validación. Se entrenan con todos los productos a la vez y se verifica que no usen datos futuros (`python verificar_fuga.py`). El bosque aleatorio no puede predecir por encima de lo visto en el entrenamiento. El Laboratorio compara todos por calibre, tiempo y tipo.
- **SHAP.** En Pronóstico avanzado, una explicación en español de qué variables suben o bajan el pronóstico de un mes.
- **Meses atípicos.** Se detectan por calibre y producto con un puntaje z robusto contra el patrón de temporada. Eliges Mantener, Corregir o Marcar evento, y la app mide con validación cruzada si corregir mejora la precisión (el efecto es mixto).
- **Series similares.** Los productos con poca historia toman prestado el patrón de su grupo; en estos datos no cambia la precisión de forma apreciable.
- **Señales.** Indicadores externos (vivienda, construcción, tipo de cambio, cobre y otros) con una sensibilidad y un rezago supuestos, hasta que las calibres con tus propios indicadores en Señales. Son supuestos, no medidas.
- **Arancel.** Un escenario de arancel multiplica solo la parte de exportación a EE. UU. según una elasticidad supuesta. Los datos reales no traen exportación: defínela en Datos con una regla (por ejemplo, una fuerza de venta) para que el arancel mueva el pronóstico.
- **Precios de metales en vivo.** Cobre (COMEX `HG=F`), aluminio (`ALI=F`) y USD/MXN (Banxico FIX o Yahoo), cada 15 minutos. Se guarda un histórico diario en `market.db`. Alertas por movimientos de más de 3% en un día o 8% en un mes, con su impacto estimado en el margen. El precio del metal solo entra al machine learning si baja el error de validación.
- **Monitor de noticias.** Cada 30 minutos consulta el Federal Register, Google News (español e inglés) y GDELT; clasifica por tema y relevancia, etiqueta lo que podría afectar y sugiere una acción. Guarda solo titular, fuente, fecha, enlace y un resumen corto. Ninguna noticia ni precio en vivo cambia el pronóstico solo: una persona decide con "Simular este escenario".
- **CRM y anomalías.** Las oportunidades abiertas entran al pronóstico ponderadas por la probabilidad de su etapa (Prospecto 10%, Calificada 25%, Cotización 50%, Negociación 75%, Ganada 100%). La app aprende el ritmo normal de pedidos de cada cliente (o fuerza de venta) y calibre y avisa cuando se rompe; cada falsa alarma sube el umbral de esa pareja.
- **Portafolio.** ABC por volumen (A: 80%, B: 15%, C: 5%), XYZ por variabilidad de la demanda, crecimiento, margen aproximado y error de referencia. La sensibilidad al precio compara cambios anuales suavizados del volumen y del metal con rezagos de 0, 3 y 6 meses.
- **Cascada.** Parte del pronóstico estadístico base (o del último congelado) y suma, en orden, los meses atípicos corregidos, las señales, el precio de los metales, el arancel, las oportunidades y los ajustes de consenso. Es una descomposición secuencial: cada efecto se mide con los anteriores ya aplicados.
- **Seguimiento.** Se puede congelar un pronóstico y compararlo después con lo vendido; el valor agregado (FVA) es el error del estadístico menos el error del consenso.

## Logotipo y estilos

- El encabezado muestra el logotipo de Viakable, una línea gris vertical y "calibra". Pon el logo en `assets/viakable-logo.png` (y, si lo tienes, `assets/viakable-logo-white.png` para el tema oscuro; sin él, el logo va sobre una pequeña área blanca). Si no hay archivo, se muestra el nombre como texto.
- Todo el CSS vive en `assets/styles.css` y se inyecta de una sola vez con `st.html`. Los cambios de CSS se ven al recargar la página; los de `ui.py` requieren reiniciar Streamlit.

## Si los precios en vivo no cargan

Ejecuta `python -m viakable_forecast.market_data`: prueba cada fuente (Yahoo Finance y, si hay token, Banxico) y muestra el error exacto. Los mismos errores salen en la terminal donde corre Streamlit con el prefijo `[precios]`. Una falla pasajera se reintenta cada minuto; no se queda 15 minutos en caché.

## Riesgo y decisiones

- **Pronóstico probabilístico:** la demanda se simula con el pronóstico más errores tomados al azar de la validación cruzada (relativos al nivel de cada ventana). Con solo 3 o 4 ventanas por calibre, esa distribución propia dejaba dentro de P10 a P90 apenas la mitad de los meses reales cuando se omitía una ventana; por eso se comparten los errores entre calibres y se ensanchan hasta cubrir cerca de 80% en la misma comprobación. La sección muestra ambos porcentajes. Las colas siguen siendo una aproximación.
- **Monte Carlo (5,000 simulaciones):** los precios siguen una caminata aleatoria lognormal alrededor del camino de Metales, con la volatilidad y la correlación de los últimos 60 meses de historia guardada; el arancel se reparte entre exención, vigente y +10 puntos con probabilidades que tú pones. Con la semilla fija, los mismos datos dan los mismos resultados. La economía por calibre replica la de Planeación (se verifica que sin ruido dé el mismo margen).
- **Margen en riesgo:** margen esperado menos el percentil 5. **Compra óptima:** cantidad que cubre el percentil faltante / (faltante + sobrante) de la necesidad simulada de metal; faltante = recargo de compra urgente (ajustable) y sobrante = costo de mantener inventario durante los meses que cubre. **Cobertura:** fijar un porcentaje de las compras al precio de hoy; como el traspaso al precio de venta ya cubre una parte, la cobertura que menos riesgo deja ronda 100% menos el traspaso.
- Los precios, costos y probabilidades son supuestos; no es asesoría financiera ni una oferta de cobertura.

## Notas de calidad de los datos

- **Periodo 13 (202513):** es un cierre contable con 0 toneladas. Se ignora.
- **Calibre combinado "16y18":** se separa con el peso del metal (kg/m): el mayor es calibre 16 y el menor, 18.
- **Toneladas negativas:** hay 7 filas con devoluciones. Se conservan en las sumas.
- **Productos SLZA58 a SLZA61:** su nombre dice "THW 12" pero la columna Calibre dice 14 (y su kg/m es el del 14). Se usa la columna Calibre.
- **Sin cliente ni país:** los datos no los traen, y el archivo es mensual. La fuerza de venta (VIAKON COMERCIAL, CDN COMERCIAL, CAYCA) hace de cliente, y el ritmo de pedidos se mide en meses.
- **Aluminio vs. cobre:** una familia que contiene "AL" (p. ej. "AL BT | Cal 6 ...") es aluminio; el resto es cobre.
- Con datos reales casi no hay anomalías de ritmo de pedidos: la señal es poca por ser mensual y por fuerza de venta.

## Limitaciones

- **Precios ilustrativos.** Hasta que captures los tuyos, precios de lista, costos de conversión, capacidad y elasticidades son supuestos; con ellos el margen ronda 0% con 50% de traspaso del metal.
- **COMEX no es el LME.** El cobre y el aluminio en vivo son contratos de COMEX/ALI, una referencia cercana al LME pero no su precio oficial (que requiere licencia de pago).
- **Las noticias se clasifican automáticamente.** Con reglas de palabras clave. Pueden fallar; por eso una persona decide. El DOF no tiene un feed accesible y se busca a través de las noticias.
- **Los roles no son seguridad formal.** Ordenan quién puede hacer qué y quedan en la bitácora, pero cualquiera puede cambiar su rol en Personalizar. No hay inicio de sesión.
- **Sin conexión a SAP.** Los datos se cargan con archivos (la app reconoce encabezados de SAP).
- **Poca historia.** Con 30 meses, la estacionalidad anual se estima con solo dos ciclos y las predicciones por producto son inciertas.
- **Precio y volumen.** Con 16 puntos comparables, no se ve una relación confiable entre el precio del metal y el volumen, y aunque se viera no probaría causa y efecto.
- **Internet.** Cada fuente externa tiene respaldo y la app nunca se rompe sin internet: usa el último precio y las noticias guardadas, y avisa.
- **Privacidad.** En la versión publicada no hay datos reales de Viakable: empieza con datos de ejemplo y tus ventas se quedan en tu navegador.

## Pruebas y scripts

- `python verificar_fuga.py`: verifica que el machine learning no use datos futuros.
- `python evaluacion_motor.py`, `evaluacion_modelos.py`, `evaluacion_atipicos.py`: pruebas fuera de muestra; los resultados quedan en `resultados_*.md`.
- Las bases `crm.db`, `market.db` y `news.db`, `ajustes.json` y `.streamlit/secrets.toml` no se suben a GitHub.

## Estructura

- `app.py` (interfaz y secciones), `ui.py` (estilo y gráficas), `settings.py` (preferencias), `i18n*.py` (traducciones).
- `viakable_forecast/`: `data`, `methods`, `engine`, `ml`, `planning`, `metals`, `market_data`, `news`, `crm`, `insights`, `atypical`, `similar`, `files`, `summary`, `report`, `config`.
