# calibra · Handoff and remaining steps

Put this file in the project folder (`viakable-vsc`, next to `CLAUDE.md`).
In Claude Code, start with:

```
Read CLAUDE.md and NEXT_STEPS.md. NEXT_STEPS.md has the project status and the remaining steps. Confirm you understand where we are, then wait for my first prompt.
```

---

## Project status (October 2026)

- **App:** calibra, a Viakable sales forecasting, planning and CRM web app in Streamlit. University project for Digital Supply Chains (Tec de Monterrey).
- **Done:** phases 1–8, which cover:
  - data loading, forecast methods, engine, ML, model lab and SHAP
  - planning, data intelligence, live metal prices and FX, regulatory news
  - CRM and alerts, business insights
  - finishing: modes, guided tour, personalization, the "calibra" logo lockup
  - animations and presentation mode
- **Removed:** the Anthropic-API copilot (Phase 8B). Replaced by "Asistente Calibra", which uses no paid API.
- **Remaining:**
  1. Hybrid assistant
  2. Connect Gemini
  3. Phase 9, tests
  4. Phase 10, publish
- **Rules for every step:**
  - Real-time updates everywhere.
  - Reference design: `referencia/viakable-demand-sensing.html`.
  - Spanish UI with an English option.
  - Never report R²; use MAPE/WAPE on future months.
  - Business numbers come only from the engine.
  - Results are reported honestly.
  - Real sales data is never published.

---

## Step 1 · Hybrid assistant

```
Make "Asistente Calibra" able to answer questions on almost any topic, with a hybrid architecture and no paid API:

1. Router: classify each question into (a) business data/forecast/scenario → app engine, (b) app concepts/methodology → knowledge base, (c) general facts → Wikipedia, (d) open/general questions on any topic → LLM. Use the existing embedding model for routing, with confidence scores.

2. Wikipedia (free): for general factual questions, search the Spanish Wikipedia (fall back to English) with the official API, show a short answer in your own words plus the article link as the source. You have my permission to use requests.

3. LLM providers with a pluggable design (llm.py), tried in this order:
   - Ollama local (http://localhost:11434) with a small model (qwen2.5 or llama3.2). Explain how to install it and download the model on my Mac.
   - Optional free-tier cloud provider (e.g. Google Gemini or Groq) if I add a key in .streamlit/secrets.toml; tell me how to get the key and what the free limits are.
   - If none is available, answer with (a), (b) and (c) only and say open questions need the LLM.

4. Grounding rules:
   - The LLM never generates business numbers. For questions that mix topics (e.g. "¿cómo afectaría una recesión en EE. UU. a nuestras exportaciones?"), get the numbers from the engine (export forecast, tariff scenario) and pass them to the LLM only to write the explanation.
   - Label each answer by source: "Datos de la app", "Metodología", "Wikipedia" or "Respuesta general (IA)", with a short note that general AI answers may contain errors.
   - Answer in the user's language (ES/EN), short and clear, in the Viakable tone.

5. Keep conversation context, the learning loop, scenario confirmation and the change log as before. Don't log full general questions with personal data.

6. Test with 60 questions: 20 business, 10 methodology, 15 general facts, 15 open topics (economy, construction, negotiation, supply chain, Mexico, technology). Report which source answered each one and any wrong answer. Wait for my OK.
```

---

## Step 2 · Connect Gemini

First, put your key **only** in `.streamlit/secrets.toml`:

```toml
[gemini]
api_key = "your-key-here"
```

Check that it works:

```
Check that the Gemini key in .streamlit/secrets.toml works: make one minimal test request and tell me only "OK" or the error message. Never print or display the key.
```

Then:

```
I created a free Gemini API key (Google AI Studio) and added it to .streamlit/secrets.toml under [gemini] api_key. Use Gemini as the LLM provider for "Asistente Calibra" (open/general questions and for writing explanations), keeping the hybrid architecture:

1. Use the official Gemini API (check the current docs for the recommended free-tier Flash model ID; don't hardcode an outdated one; make the model configurable in secrets). Install the official Google Gen AI SDK or use requests; you have my permission.
2. Privacy (free tier data may be used by Google): never send raw sales files, product-level or customer-level data to Gemini. Send only the user's question and, when needed, small aggregated numbers already computed by the engine. Add a visible note in the assistant: "Las preguntas generales se procesan con Google Gemini (plan gratuito)."
3. Business numbers still come only from the engine; Gemini only writes the explanation around them. Label answers by source ("Datos de la app", "Metodología", "Wikipedia", "Respuesta general (Gemini)").
4. Rate limits: handle "too many requests" errors gracefully (wait and retry once, then show a friendly message and answer with the engine/knowledge base/Wikipedia if possible). Cache identical questions for the session. Add a simple per-session question limit setting.
5. If the key is missing or invalid, keep the assistant working without the LLM and show how to add the key.
6. Make sure .streamlit/secrets.toml is in .gitignore, and add the [gemini] section to secrets.toml.example without the real key.
7. Test with the 60-question set (business, methodology, general facts, open topics) and report which source answered each and any wrong answer. Wait for my OK.
```

Check it yourself: ask "¿Cómo afectaría una recesión en EE. UU. a nuestras exportaciones?". The export number should match the one in Pronóstico.

---

## Step 3 · Phase 9, tests

```
Continue with Phase 9 from CLAUDE.md:
1. Automated tests (pytest) for the engine: total equals sum of gauges, 16y18 split matches the Gauge column, cross-validation never uses future data, metal price changes move margin in the right direction, tariff exemption increases exports, odd files give clear errors, and the app keeps working when prices, news, Wikipedia or Gemini fail (simulate no internet and an invalid key).
2. Assistant tests: every business number in an answer matches the engine; no raw sales, product-level or customer-level data is sent to Gemini; scenarios only apply after "Aplicar escenario"; the rate-limit message works.
3. Run the whole app with my CSV and with sample data, in both modes and languages, and fix anything broken.
4. Performance: tell me which actions are slow and optimize them.
5. Final report: what works, known limitations, and the accuracy summary per model and gauge, for my presentation. Wait for my OK.
```

---

## Step 4 · Phase 10, publish

Before this step, delete your current Gemini key in AI Studio and create a new one. The old key was pasted in a chat, so it shouldn't go into the public version.

```
Continue with Phase 10: help me publish the app on Streamlit Community Cloud.
1. .gitignore must exclude .venv, databases, .streamlit/secrets.toml and my real data (Viakable_Tons_Sales.csv / .xlsx). Show me the list of files that WILL be uploaded before pushing.
2. Guide me step by step to create the GitHub repository (I may need to create an account and install git) and deploy on share.streamlit.io.
3. Remind me to create a NEW Gemini key for the online version, and show me how to add it and the other keys (Banxico, FRED) in the "Secrets" section of Streamlit Cloud.
4. Confirm live prices, news and the assistant work online, and that the app starts with sample data only.
5. Give me the final public link.
```

---

| Step | What | Status |
|---|---|---|
| 1 | Hybrid assistant | ⏳ Next |
| 2 | Connect Gemini | ⏳ |
| 3 | Phase 9: tests | ⏳ |
| 4 | Phase 10: publish | ⏳ |
