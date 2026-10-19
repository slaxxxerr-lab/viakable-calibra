# Viakable · Sales Forecasting, Planning and CRM (Python)

## Context
- I'm Saul, a Global Business student at Tecnológico de Monterrey. This is a university project for a supply chain course.
- I'm on a Mac with Python 3.14. I'm not experienced with the terminal: run commands yourself and explain briefly what each step does.
- The tool forecasts sales for Viakable (Mexican electrical cable manufacturer) by wire gauge (calibre), turns the forecast into inventory and metal purchasing plans, and includes a light CRM.
- All user-facing text (UI, reports, messages) must be in **Spanish (Mexico)**. Code, variable names and comments can be in English.
- Prototype for academics: default prices, costs, capacity and elasticities are illustrative. Say so in the UI footer.

## Working rules
- Build in phases (below). Finish and test each phase before starting the next. Show me results after each phase and wait for my OK.
- Use a virtual environment `.venv` in this folder. If a library fails on Python 3.14, tell me and use Python 3.13.
- Never delete files without asking. Keep functions small and documented.
- Test every phase with my real file `Viakable Tons Sales.xlsx` (in this folder) and with generated sample data.
- Libraries: pandas, numpy, scikit-learn, openpyxl, streamlit, plotly. Nothing else unless you ask me.

## My data (`Viakable Tons Sales.xlsx`, sheet "Sheet 1", ~5,940 rows, Jan-2024 to Jun-2026)
Columns: `Year, Month, Date, Fuerza de Venta, ID UEN, Periodo Contable, ID producto, Calibre, Producto, Gauge, ContMet (kg/m), vcFamiliaTransversal, Toneladas`.
- `Toneladas` = quantity (metal tons). `Date` = month. `Periodo Contable` like 202403; period 13 (202513) is a closing period with 0 tons → ignore.
- `Calibre` values: 6, 10, 12, 14 and **"16y18"** (combined). Split 16y18 using `ContMet (kg/m)`: the row with the larger kg/m is gauge 16, the smaller is 18.
- `ID UEN` = business unit (CONS, ENER, FLEX). `Fuerza de Venta` = sales force (VIAKON COMERCIAL, CDN COMERCIAL, CAYCA).
- `vcFamiliaTransversal` = product family. A family containing "AL" (e.g. "AL BT | Cal 6 ...") means aluminum; otherwise copper.
- There is NO customer or country column yet. The app must let me mark which values (e.g. a sales force) are **exports to the USA**.

## Architecture
```
app.py                    Streamlit UI
viakable_forecast/
  config.py               constants, signals, presets, roles, default parameters
  data.py                 load Excel/CSV, auto-detect columns, clean, monthly aggregation
  methods.py              statistical + intermittent methods, error metrics, demand class
  ml.py                   global gradient boosting model
  engine.py               backtest, method selection, signals, tariff, consensus, reconciliation
  planning.py             safety stock, production, metal, margin, capacity, scenarios
  crm.py                  SQLite CRM + audit log
  insights.py             alerts, low-volume products, forecast tracking, signal calibration
  report.py               command-line Excel report
requirements.txt, README.md, iniciar.command (double-click launcher for Mac)
```

## Phase 1 · Data (data.py)
- Read .xlsx/.xls/.csv. Auto-detect columns by name (Spanish/English and typical SAP export names: "Fe.factura", "Ctd.facturada", "Peso neto", "Solicitante", "Texto breve de material", "LAND1"). Let the user override the mapping.
- Parse many date formats: Excel dates, 202609, 2026-09, 15/09/2026, "sep-2026". Parse numbers with commas.
- Option to convert kg → tons.
- Aggregate to monthly rows: `ym, cal, seg, ch, cli, fam, prod, mkt_raw, qty`, plus derived `mkt` ("Nacional" or "Exportación EE. UU."), from a country column or from an export rule `{dim, values}`.
- Require at least 6 months. Generate sample data (30 months, gauges 6/10/12/14/16/18, a few customers, ~14% exports with a drop after Apr-2026).
- **Done when:** loading my file reports 30 months, gauges 6, 10, 12, 14, 16, 18, and a note that 16y18 was split.

## Phase 2 · Forecast methods (methods.py)
Implement on a monthly numpy array:
- Same month last year × growth (traditional / seasonal naive), 3-month moving average, simple exponential smoothing (α grid), Holt with damped trend (φ=0.9), Holt-Winters multiplicative (needs 24+ months), linear regression on deseasonalized data (seasonal index = ratio to centered 12-month moving average).
- Intermittent demand: **Croston-SBA** and **TSB**.
- **ETS automático** (statsmodels Holt-Winters family: trend none/additive/damped, seasonality none/additive/multiplicative, chosen by AICc) and **SARIMA automático** (statsmodels SARIMAX; a few small ARIMA/SARIMA orders, chosen by one-step in-sample error). Both are in the validation ranking like any other method.
- Error: MAPE; if the holdout has zero months, use WAPE. Also bias %.
- Accuracy labels: error <5% "Muy certero", <10% "Certero", <20% "Aceptable", else "Poco confiable"; precision = 100 − error.
- Demand class (Syntetos-Boylan) with ADI and CV²: Estable, Errática, Intermitente, Irregular.

## Phase 3 · Engine (engine.py)
- `series(selection, measure)` for any filter: gauges (multi-select, including "16 + 18" together), market, business unit, sales force, customer, family, product. Measure in tons or in **sales MXN** (tons × price per gauge and market).
- **Validation (v2): rolling-origin cross-validation** instead of one holdout. Windows are cut 6, 9, 12 and 15 months before the end (needs 12+ training months; 30 months of data gives 4 windows, training with 24, 21, 18 and 15 months), each forecasting 6 months using only earlier data. A method's error is the **mean of its window errors**; the ranking uses that mean, so the choice does not depend on one period. Holt-Winters and the deseasonalized regression need 24+ training months, so they only rank when they have at least 3 windows (shown as "pocas ventanas" otherwise). "Automatic" picks the lowest mean error among eligible methods, **but keeps the weighted ensemble unless another method beats it by 2+ points** (selection among many methods is noisy; combining is more robust). The user can still pick any method and the UI says when a more accurate one exists.
- **Ensemble = inverse-error weighted average** of the 5 most accurate eligible statistical methods (weight proportional to 1 / validation error). Its own window errors are computed with weights that leave that window out, so it is not graded on data it was tuned with.
- **Signal-based model:** base = last 3 months deseasonalized (with a "history weight" slider blending in the traditional method) × (1 + Σ signal effects) + CRM opportunities. Effect = sensitivity × expected change % × min(1, month / lead time). Mexican signals only move the domestic share; US signals only move the export share.
- Signals: construction permits, CFE tenders, industrial buildings, data centers, distributor sell-out, PMI, copper price, MXN/USD (domestic); US housing starts, US PMI, exchange rate for exports (US). Presets: "Tendencia actual", "Auge nearshoring", "Desaceleración", "Cobre al alza".
- **US tariff (Section 232):** multiplier = ((1+scenario)/(1+current))^(−elasticity), default current 50%, elasticity 0.6, applied to the export share of ANY method.
- **Consensus:** for a single gauge, monthly % adjustments with a required justification.
- **Reconciliation:** default total = sum of each gauge forecast (each with its best method), so totals match the gauge table. Option for direct total.
- **Confidence range (80 %) from real errors**: ratios actual/forecast − 1 collected from the cross-validation windows (pooled by months 1-2, 3-4, 5-6; +8 % width per month beyond 6), taken from the sum of the gauge forecasts. With fewer than 3 windows the range is widened (x1.5 with 2 windows, x3 with 1) because it covered only 35-65 % of real months. Out-of-sample coverage with 3+ windows was ~81 %.
- **Done when:** my file gives a 6-month total around 4,000 t with ~90% accuracy, and a table by gauge with method, accuracy and demand class.

## Phase 4 · Machine learning (ml.py)
- scikit-learn `GradientBoostingRegressor` (150 trees, depth 3, lr 0.08, subsample 0.8) trained on ALL product series at once.
- Features (scaled by each series' 6-month average): lags 1–3, same month last year, 3-month average, recent trend, month of year, gauge, family, export flag, log of size. Sample weight = series size. Recursive multi-step forecast.
- Sum product forecasts for any selection. Validate the same way (same windows) and compete in the ranking. Show feature importance (permutation importance for HistGradientBoosting).
- **No data leakage** (checked by `verificar_fuga.py`): lags, same-month-last-year, 3-month average, trend and the per-series scale only use months before the month being predicted; sample weights use the same past-only scale; the product-family code is built only from series that already had sales before the cut; hyperparameters are chosen only from validation folds. The script alters the future on purpose and confirms the forecast of each window does not change.

## Phase 5 · Planning (planning.py)
- Per gauge and market: safety stock = z × 1.25 × error × mean demand × √lead time (z by service level 90/95/98/99%); reorder point; suggested production month by month to end each month at safety stock; copper and aluminum tons.
- Money: price MXN/t (domestic) or USD/t (export, minus freight, minus tariff if DDP), metal cost from LME price × FX × scenario, conversion cost; revenue, gross margin, margin lost per +10% metal.
- Capital freed = (safety stock with traditional method error − with best method) × unit cost; annual saving = × holding cost %.
- Capacity per gauge (default 10% above best month of last 24, editable): utilization %, shortfall, bottleneck months.
- Save named scenarios and compare them side by side.

## Phase 6 · CRM and insights (crm.py, insights.py)
- SQLite: customers (with country MX/US), opportunities (stages Prospecto 10%, Calificada 25%, Cotización 50%, Negociación 75%, Ganada 100%, Perdida 0%), activities and tasks, consensus adjustments, roles, audit log (who, what, when). Seed sample data.
- Open opportunities feed the signal-based forecast weighted by probability.
- Roles: Administrador, Planeación, Comercial, Consulta, with permissions per action.
- Alerts ("Qué atender"): gauges falling >8% vs last year, unreliable forecasts, exports falling, capacity exceeded, customers buying >20% less, overdue tasks, deviations vs frozen forecasts.
- Low-volume products table with demand class and Croston/TSB forecasts.
- Freeze forecasts and later compare with actuals (forecast value added = statistical error − consensus error). Include a "test tracking" that simulates a forecast made 6 months ago.
- Signal calibration: upload monthly indicators, compare year-over-year changes with each gauge, test leads 0–6 months, report correlation, sensitivity and evidence (strong ≥0.6, moderate ≥0.4).

## Phase 7 · Streamlit app (app.py) and report
- Sidebar: Simple / Avanzado mode, section menu, "ver como" role selector. Simple mode shows only essentials.
- Sections: Inicio (alerts + KPIs + chart), Pronóstico, Planeación, Clientes, Oportunidades (columns by stage), Señales, Equipo (roles + audit log), Datos (upload, column mapping, export rule).
- Forecast chart (plotly): up to 36 months of history with light area fill, shaded forecast zone, red forecast line with monthly points, confidence band, dotted 12-month moving-average trend line.
- Viakable brand: red #ff0000, black #000000, gray #cbcbcb, accent blue #48b3f6 (innovation only), white background. Font stack "Aeonik", "Aptos", "Manrope", sans-serif. Short, direct copy without exclamation marks.
- Download button for a full Excel report. `report.py` CLI: `python -m viakable_forecast.report file.xlsx --horizonte 6 --salida reporte.xlsx --exportacion ch "VALUE"`.
- `iniciar.command`: double-click launcher that activates `.venv` and runs Streamlit.
- README in Spanish with installation, usage, methodology and limitations (no live SAP connection, roles are not formal security, signals are assumptions until calibrated).

## Methodology changes (v2, accuracy upgrade)
- `evaluacion_motor.py` compares the previous engine (frozen in `viakable_forecast/legacy/`) with the current one. It reports two numbers: the error each engine *reports*, and the real **out-of-sample error** of the whole procedure (select, tune, forecast), repeated as if standing 6, 9 and 12 months before the end. Only the second is trustworthy; the first is optimistic whenever the method is chosen on the months it is graded on.
- Results are saved in `resultados_motor.md` / `.json`. Honest summary with 30 months: the previous engine *reported* ~7 % error but its real out-of-sample error was ~14 %; the new engine reports ~11 % and really delivers ~10 %. Gauge 18 AWG (about 0.4 t/month) is too small and noisy for any model. There is a limit to what 30 months can show: 3 origins only, and per-origin results vary a lot.

## Live updates, metal prices and what-if (v3)
- **Everything recalculates at once.** Control values are synced at the start of every run (`sync_store`), so a change is reflected in that same run in the forecast, KPIs, charts, planning and margins. The ML is cached by data content; if only some sales figures were corrected in the app, the already tuned model configuration is reused (retrains in seconds, tuning only happens when a new file is loaded).
- **Editable sales in "Datos":** table of tons by month and gauge (plus a row for the next month). Edits are stored separately from the file (month + gauge totals; existing rows are rescaled so the customer/product mix is kept, new months get a row with the gauge's latest attributes) and shown with their impact on the forecast.
- **"¿Qué pasa si…?" panel (Pronóstico):** tariff, metal price %, exchange rate, pass-through %, one signal, and an extra order (gauge, t/month, months). It reuses the same controls as Señales, Planeación and Metales. A strip shows what moved against the *base* (signals 0%, current tariff, no metal scenario, flat prices, reference sales prices) with the delta (e.g. "Pronóstico +3.2% vs. base"). A toggle applies signal effects and extra orders to the main forecast (not only to the signal-based model).
- **Metales section:** current LME copper/aluminum (USD/t) and exchange rate; a 12-month path (constant, % scenario linear/immediate, month-by-month table, or damped trend of an uploaded price history); `% pass-through` (default 50%): selling price(month) = list price + pass-through x (metal cost(month) - current metal cost). Planning uses monthly metal cost and price (metal cost, margin, inventory value and revenue). The copper path feeds the "Precio del cobre" signal (average change of the first 6 months vs today) so demand reacts. A shock card answers "Si el cobre sube X%, el margen baja $Y M y la venta en pesos cambia Z%". A purchase recommendation chooses, for each production month, the cheapest purchase moment (today or any month up to a maximum advance), with cost = price x FX x (1 + monthly holding cost x months held), compared with buying month by month and buying everything today.
- **Metal price in the ML:** an uploaded history is tested as two extra features (3-month change and level vs 12-month average of the series' own metal, lagged one month, future prices flat inside validation windows) and kept only if validation error drops by 0.25+ points and in half of the windows. Verified leak-free in `verificar_fuga.py`.

## Phase 6A · Model lab (done)
- **New global models** (`ml.py`), all trained on every product series at once with the same features, sample weights and rolling-origin windows as the main model, each with a small grid and the leave-one-window-out choice of configuration: **XGBoost** (`xgboost`; on Mac without Homebrew run `bash arreglar_openmp_mac.sh` once, it reuses scikit-learn's OpenMP library; if XGBoost cannot load the app skips it), **Random Forest** (cannot predict above values seen in training), **Ridge** (lags + month and family as one-hot categories, standardized with training data only), **MLPRegressor** (small neural network, no TensorFlow/PyTorch).
- **Stacking** (`engine.cv_run`): the three best of {ML models, neural network, weighted statistical ensemble} weighted by inverse validation error (weights for each window computed without that window).
- **SHAP** (`ml.explain`): exact TreeSHAP values from XGBoost (`pred_contribs`), summed over the selected products and expressed in tons, shown in Pronóstico in plain Spanish for any month of the horizon. The `shap` library was not needed.
- **"Laboratorio" view (Avanzado):** model, type (Estadístico / Machine learning / Red neuronal / Ensamble), accuracy for the total and each gauge, training time, winner per column highlighted, winner by gauge, tuned configuration per model, and the saved out-of-sample test.
- **Honest result** (`evaluacion_modelos.py`, `resultados_modelos.md`): over 5 comparable origins, simple statistical methods (TSB, moving average, SARIMA, ETS) were best on the total (7-7.5 % error); the ML models were 10.6-11.4 %, the neural network 18.8 %, stacking and "AUTO with the new models" did worse than the existing AUTO. The new models are available to pick manually, but **AUTO excludes them** (`engine.AUTO_EXCLUDE`) because adding them made out-of-sample accuracy worse. They only win on isolated gauges (6 and 10 AWG in validation) and not out of sample. Caveats: 30 months only; ML models need 20+ months of training so they have fewer origins; stacking only exists where 3+ models have enough windows.
- No R² anywhere (permutation importance uses mean absolute error).

## Phase 6B · Data intelligence (done)
- **Several files at once** (`files.py`, "Datos"): upload any mix of Excel, CSV and PDF. Sales files with different column names are normalized to standard names, combined, and **rows repeated across files are removed** (overlapping periods; repeated rows inside one file are real sales and are kept). A quality summary lists rows, duplicates removed, months, gauges and tons per file, plus structure differences. Combining the three yearly files of the test CSV reproduced exactly 30 months and 18,210 t.
- **PDF tables** (`pdfplumber`): each table is shown editable and nothing is used until the user says what it is: "Lista de precios" (choose gauge and price columns and currency, then "Usar estos precios" updates the sales prices used by Planeación) or "Tabla de ventas" (combined with the other sales files). Scanned PDFs (images) are not readable.
- **Atypical months** (`atypical.py`): per gauge and per product, expected value = 12-month centered median x seasonal factor of that calendar month; robust modified z-score (median/MAD) above 3.5 and a deviation of at least 15 % flag a month. Each flagged month gets a decision: Mantener, Corregir (replace with the expected value, product first and gauge last, keeping the customer mix) or Marcar evento with a note. Marks appear on the forecast chart. Models, planning and charts retrain on the corrected data (fast path: the tuned ML configuration is reused). `cv_test` shows whether correcting helps: in each validation window outliers are detected and corrected using only earlier data, five statistical methods forecast, and error is measured against what really sold.
- **Similar series** (`similar.py`): products are grouped by gauge + metal + family (falling back to gauge + metal, then metal). Products with short history (< 18 months since first sale) or > 30 % zero months blend their own level (Croston-SBA or SES) with the group's seasonality and recent trend; weight on the own level = nonzero months / 24 (0.25 to 0.9). For gauges the method "Aprendizaje de series similares" uses the other gauges of the same metal as peers; it only competes when the gauge has short history or many zeros, and AUTO excludes it.
- **Honest results** (30 months, test CSV): correcting all detected gauge-level outliers lowered the total error of statistical methods by 0.7 points but raised gauge 6's by 0.9; with the ML model the total did not change (-0.1) and only correcting at gauge level helped some gauges slightly; correcting every product-level flag was worse (zeros and tiny products are not errors). Blending with the group pattern changed the summed-product error by +0.04 points (no real improvement); per product both errors are above 100 % because the series are tiny. Conclusion: use corrections case by case, not automatically.
- `evaluacion_atipicos.py` reproduces the ML check of corrections. Note: once the user corrects months, validation accuracy is measured against corrected data (somewhat optimistic); the `cv_test` table is the reliable measure.

## Phase 6C · Live market data (done)
- **`market_data.py`** (no Streamlit inside): one provider per source with a fallback chain, a SQLite database (`market.db`, git-ignored) and alerts. USD/MXN: Banxico SIE FIX (series SF43718, needs a token) then Yahoo `MXN=X`. Copper: COMEX `HG=F` x 2,204.62 (USD/lb to USD/t). Aluminum: `ALI=F` (USD/t). Monthly history: FRED (key), FRED public CSV, Alpha Vantage (key), and finally the monthly mean of the daily closes. Real fetch at the time of writing: copper ~14,770 USD/t, aluminum ~3,181 USD/t, USD/MXN ~18.43. FRED's public CSV timed out from this machine, so the history comes from the Yahoo daily closes (since 2000 for copper, 2014 for aluminum).
- **Keys** go only in `.streamlit/secrets.toml` (template in `secrets.toml.example`; the real file is in `.gitignore`): `[banxico] token`, `[fred] api_key`, `[alphavantage] api_key`. Without them the free fallback is used.
- **Behavior:** `st.fragment(run_every="15m")` ticker under the header ("Cobre $X USD/t ▲ +1.2% hoy · Aluminio ... · USD/MXN ... · Fuente ... · actualizado HH:MM") plus an "Actualizar precios" button (an `on_click` callback that clears the cache; a `st.rerun()` inside the button branch looped forever). Results are cached 15 minutes and saved. If a source fails the last saved price is used with a visible warning; with no data at all the typed prices are used. Nothing breaks without internet (tested by forcing every source to fail).
- **Manual override** per series (Metales): "En vivo" or "Precio manual" (a contract price), with a visible "Precio manual" badge in the ticker. COMEX is labeled as a close reference to the LME, not the official LME price.
- **Everything connected updates:** the live price becomes the current price (`pl_cu`, `pl_al`, `pl_fx`) at the start of every run, so metal cost, margin, inventory value, sales in pesos (pass-through relative to the *reference* metal prices the list prices were set with, editable in Metales, default 9,840 / 2,600 / 18.5), the purchase recommendation and the forecast react. The copper signal adds the live price vs its 90-day average (live mode only). When the fragment finds new prices it reruns the whole app.
- **Alerts:** copper or aluminum moving more than ±3% in a day or ±8% in 30 days (daily history in the database), with an estimated margin impact for the next 6 months (last 6 months' tons of that metal x price x FX x move x (1 - pass-through)). Today's real data raised one: aluminum -9.4% in a month.
- **Daily history** is saved in the database on every fetch; the monthly series feeds the ML feature test automatically (also uploaded files, which take precedence). Honest result with the real COMEX history: error without the price 13.1%, with the price 13.1%, better in 1 of 4 windows, so the feature is NOT used.
- **Warning:** with the real copper price the illustrative list prices give ~0% margin at 50% pass-through (12% at 100%, -18% at 0%). Enter the real list prices and reference prices.

## Phase 6D · News and regulation monitor (done)
- **`news.py`** (no Streamlit inside): providers with fallbacks and a SQLite database (`news.db`, git-ignored). **Federal Register** API (official; 4 queries: Section 232 notices from Commerce/BIS, presidential documents on copper/aluminum, USTR notices on USMCA, copper 232 rules; "Foreign-Trade Zone" notices are dropped as noise), **Google News RSS** (14 searches in Spanish and English, `when:30d`; the feed only gives headline, outlet, date and link) and **GDELT** (optional, one query, 5 s timeout, 1 h back-off after a 429). **DOF** (dof.gob.mx): its server certificate does not match the host name (`curl` and Python both reject it) and it has no stable free feed, so DOF items are only reachable through news searches; certificate verification is not disabled.
- **Processing:** duplicates removed by link and by near-identical headline within 5 days (official source kept, others counted as "N fuentes más"); only headline, source, date, link and a short summary are saved (the feed's abstract, or a summary written by the app from topic and impact tags). Topic by weighted keyword rules (Arancel / T-MEC, Normativa México, Precios de metales, Construcción y vivienda, Industria y nearshoring, Competencia, Otros); relevance Alta/Media/Baja from topic weight, copper/aluminum/cable terms, official source, event words, recency and a penalty for tariffs unrelated to the business; impact tags (exportaciones, calibres de edificación, aluminio calibre 6, costos de metal, demanda nacional); suggested action with an optional scenario pre-fill.
- **App:** "Inicio" (minimal: "Qué atender" with market alerts from 6C and high-relevance news, each with its action and buttons; latest relevant news) and "Noticias" (filters by topic, relevance, date, status and text; "Actualizar" button; auto-refresh every 30 minutes with `st.fragment(run_every="30m")` and a background thread so the screen never blocks; source status with warnings and the last saved items when a source fails). Latest related news appears in the tariff card and the copper, CFE tenders, permits, industrial buildings and data center signal cards, and in Metales. Buttons per item: Relevante, Ya revisado, Ignorar (all written to a change log shown in the section) and **Simular este escenario**, which pre-fills the "¿Qué pasa si…?" panel (tariff, metal price or a signal) and shows a banner; the person confirms by reviewing it, and "Volver a la base" undoes it. If the data has no exports, the banner warns that a tariff scenario will not move the forecast.
- **Rate limits and terms:** at most 14 Google News searches, 4 Federal Register requests and 1 GDELT request per 30 minutes, staggered; no article text is copied.


## Fase 6E: CRM, alertas y anomalías de ritmo (hecha)
- `viakable_forecast/crm.py` (SQLite crm.db, env VIAKABLE_CRM_DB): clientes, oportunidades por etapa, actividades/tareas, decisiones de anomalías, pronósticos congelados y bitácora. Roles Administrador, Planeación, Comercial, Consulta con permisos (`can`); no son seguridad formal.
- `viakable_forecast/insights.py`: tendencias, bajo volumen (Croston-SBA/TSB), anomalías de ritmo de pedidos por cliente (o fuerza de venta) y calibre, tolerancia aprendida de las falsas alarmas, FVA y seguimiento de prueba.
- Secciones nuevas: Clientes, Oportunidades, Equipo (avanzado). Inicio: mosaicos clicables, gráfica, "Qué atender", cola de anomalías, noticias. Pronóstico avanzado: bajo volumen y seguimiento (congelar).
- Las oportunidades abiertas entran ponderadas por etapa; las de ejemplo solo con el interruptor. Resolución mensual: el ritmo se mide en meses.
- Hallazgo honesto: con los datos reales casi no hay anomalías de ritmo (1 por producto, 0 por calibre).


## Fase 6F: insights de negocio (hecha)
- `insights.abc_xyz` (ABC por volumen, XYZ por variabilidad, margen aproximado = toneladas × margen por tonelada del calibre, crecimiento, error de referencia, sugerencia Impulsar/Revisar/Considerar descontinuar/Mantener) y `insights.price_sensitivity` (elasticidad volumen-precio del metal con cambios anuales suavizados y rezagos 0/3/6; confianza Baja con menos de 24 puntos).
- Sección Portafolio (matriz ABC–XYZ, lista con sugerencias, sensibilidad al precio). Cascada "Por qué cambió el pronóstico" en Pronóstico: base estadística + atípicos + señales + metales + arancel + oportunidades + consenso; si hay pronóstico congelado, parte de él (el congelado guarda sus factores) y agrega "Datos nuevos y reentrenamiento" como residual.
- "Resumen del mes" en Inicio (pantalla, PDF escrito a mano en `summary.py` sin librerías extra, Excel con openpyxl).
- Honesto: con 16 puntos comparables no hay relación confiable precio-volumen; los márgenes por producto son ilustrativos (≈0 con 50% de traspaso).

## Fase 7: acabado (hecha)
- Nombre `APP_NAME = "Viakable Calibra"` en `viakable_forecast/config.py` (encabezado, pestaña, resumen, reporte, README). Cambiar el nombre es cambiar esa línea.
- Modo Simple oculta: cola de anomalías, SHAP, meses atípicos, señales del what-if, parámetros avanzados de Metales (histórico, vínculos con la señal, precios de referencia, prueba de ML), sensibilidad al precio, bitácora de noticias y las secciones Señales, Laboratorio y Equipo.
- Guía rápida de 10 pasos (incluye precios en vivo y noticias). Reporte completo en Excel (`viakable_forecast/report.py`, también por línea de comandos). `iniciar.command`. La unidad elegida en Personalizar sustituye la "t" en textos y tablas (`i18n.set_unit`).

## Fase 8: experiencia dinámica (hecha)
- Motor de animación (`ui.ENGINE_JS`, se instala con `st.html(..., unsafe_allow_javascript=True)`): los KPI cuentan hasta su nuevo valor con flechita de cambio (`ui.count_html`, automático en `ui.kpis` y en los mosaicos), el rótulo de precios destella verde o rojo al cambiar (`data-px`), y la línea del pronóstico se dibuja al cargar. Las gráficas se mueven con `layout.transition` de Plotly (550 ms). No se usó streamlit-echarts.
- Respeta `prefers-reduced-motion`: el motor escribe la cookie `vk_rm` y la app la lee en la siguiente carga; también hay "Animaciones Sí/No" en Personalizar (`anim` en ajustes).
- Modo presentación (botón Presentar): 7 pasos de 17 s con avance automático (fragmento con `run_every`), Anterior, Siguiente, Pausar y Salir. Los pasos 2 y 4 aplican escenarios (cobre +10%, arancel 0%) que se restauran al salir. Funciona sin internet con precios y noticias guardados.
- Indicador de frescura en el encabezado (datos, modelos, precios, noticias; ámbar si está viejo). Responsivo para tableta y teléfono (`ui.RESP_CSS`).
- Para ver cambios de CSS en `ui.py` hay que reiniciar Streamlit (no recarga módulos importados).


## Fase 8B: Asistente Calibra (híbrido, sin API de pago)
- Arquitectura: `router.py` clasifica cada pregunta en a) negocio (motor), b) metodología (`kb.py`), c) hechos generales (`wiki.py`, API oficial de Wikipedia, español y luego inglés), d) abierta (`llm.py`) o mixta (cifras del motor + IA que solo las redacta). El enrutador combina reglas de pregunta, el intérprete de negocio de `assistant.py`, la similitud TF-IDF (letras y palabras, scikit-learn) con la base de conocimiento y con preguntas de ejemplo, y devuelve una confianza por ruta. No hay un modelo de embeddings en la app: TF-IDF cumple ese papel sin descargas.
- `llm.py` (sin Streamlit): proveedores en orden: Ollama local (http://localhost:11434, modelo pequeño qwen2.5 o llama3.2), Gemini y Groq (niveles gratuitos, solo con llave en `.streamlit/secrets.toml`). Sin ninguno, el asistente contesta negocio, metodología y Wikipedia, y dice que las preguntas abiertas necesitan la IA.
- Fundamentación: la IA nunca genera cifras de negocio. Preguntas de negocio: plantillas con cifras del motor (nunca pasan por la IA). Preguntas mixtas: el motor calcula (pronóstico de exportación, escenario de estrés supuesto de -10% en señales, arancel o metal, riesgo) y la IA solo redacta con un bloque CIFRAS; `numbers_ok` descarta la respuesta si trae una cifra que no estaba ahí (un reintento y, si falla, solo van las cifras del motor). En respuestas abiertas se quitan las oraciones con cifras de dinero o toneladas.
- Etiquetas: "Datos de la app", "Metodología", "Wikipedia" (con enlace) y "Respuesta general (IA)" con aviso de que puede contener errores. Idioma de la pregunta (ES/EN).
- Privacidad: a Wikipedia solo van palabras clave; a la nube va el texto sin correos, teléfonos ni enlaces (`router.redact`); en Ollama nada sale de la Mac. En la bitácora, las preguntas de negocio se guardan (redactadas); las generales solo dejan la ruta.
- Aprendizaje: si una pregunta no se entiende y se hace clic en una sugerencia, se guardan solo sus palabras de contenido (`assistant_learn` en crm.db) y se reutiliza la intención en preguntas parecidas. Los 👍/👎 guardan solo ruta y fuente (`assistant_feedback`).
- Escenarios: solo se aplican con "Aplicar escenario" (Administrador y Planeación) y quedan en la bitácora, igual que antes. Observaciones automáticas en Inicio y en la sección.
- Pruebas (60 preguntas: 20 negocio, 10 metodología, 15 hechos, 15 abiertas): enrutamiento 46/60 a la primera (antes de ajustar), 60/60 después de ajustar reglas (ya no es limpia); en un grupo nuevo de 27 preguntas: 23/27 antes de ajustar, 27/27 después de corregir las 4 fallas. Respuestas de Wikipedia flojas (no contestan la cifra): velocidad de la luz, población de Japón y quién inventó la bombilla. Las 15 abiertas solo se probaron con un Ollama simulado (conexión, privacidad, guarda de cifras); la calidad real de la IA no está verificada porque no hay Ollama ni llaves en este equipo.

## Fase 8C: Riesgo y decisiones (hecha)
- `viakable_forecast/risk.py` (solo numpy): pronóstico probabilístico (bootstrap de los errores de la validación cruzada, relativos al nivel de cada ventana, con persistencia entre meses), simulación Monte Carlo conjunta (demanda, cobre, aluminio y USD/MXN con volatilidad y correlación de los últimos 60 meses de historia, arancel por casos), margen en riesgo (esperado menos P5), probabilidad de margen bajo un umbral, compra óptima por vendedor de periódicos (cuantil faltante/(faltante+sobrante)) y barrido de cobertura (fracción de compras fijada al precio de hoy; el traspaso al precio de venta sigue al mercado). La economía por renglón replica `planning.plan` y se verifica contra `PL.totals` (`risk.deterministic`).
- Honesto: con 3 o 4 ventanas la distribución propia de cada calibre cubría solo ~50% de los meses reales en P10 a P90 (omitiendo una ventana); se comparten los errores entre calibres y se ensanchan (`risk.calibrate`) hasta cubrir ~80%. Las colas siguen siendo una aproximación.
- Sección Riesgo (solo avanzado): controles de simulaciones, umbral, probabilidades de arancel, compra y cobertura; todo se recalcula con cada cambio de precios, escenarios o datos (plan en caché; la simulación tarda milésimas). Paso "Riesgo y cobertura" en el modo presentación (8 pasos de 15 s) y herramienta `get_risk_analysis` del asistente.
- El plan ahora guarda `cv_a` y `cv_f` por renglón y `Projection.detail` también (no cambia ningún cálculo).
