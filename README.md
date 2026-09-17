# CapEx AI Agent — Zimbabwe Capital Projects Finance Intelligence Platform

A professional Streamlit application that evaluates Zimbabwe capital projects
end-to-end: **live macro economics** (inflation, ZiG/USD, RBZ policy and
lending rates, construction price index), one shared discounted cash-flow
engine, capital-budgeting metrics, geometric risk status, Monte-Carlo
simulation, real options, MILP portfolio selection, deterministic AI
explainability, TTS audio, and a Board-ready PDF management report.

> **IMPORTANT DISCLAIMER** — This platform is a decision-support tool.
> All demo projects contained in `data/demo_projects.csv` are
> **SYNTHETIC / DEMONSTRATION records**. The ML lab is trained on
> **synthetic demonstration data**. Macro indicators that are not fetched
> live are clearly labelled `fallback_estimate` and are **never** disguised
> as official Zimbabwe statistics. Outputs are planning aids, not bankable
> valuations.

---

## 1. Quick start

```bash
# inside the project folder
python -m venv .venv
.venv\Scripts\activate            # Windows PowerShell
pip install -r requirements.txt

streamlit run ui/app.py
```

Open the printed URL (default http://localhost:8501).

**Demo login** (shown clearly in the app, credentials are for
demonstration only):
- Username: `admin`
- Password: `admin123`

On first run you will see the cinematic intro (8 slides + countdown), then
the login panel, then the time-neutral welcome and a choice between
**AI GUIDED MODE** and **MANUAL MODE**.

## 2. Verify the install

```bash
python -m py_compile core/*.py ui/*.py tests/*.py
python tests/test_all.py          # prints PASS/FAIL and exits non-zero on failure
$env:PYTHONPATH="."
python -c "import ui.app"         # import smoke test (UI won't run under bare python)
python -c "from core import financial_engine, macro_engine, risk_engine, ai_explainer, portfolio_optimizer, optional_pricing, tts_speech, report_builder, email_delivery"
```

## 3. Architecture

| Module | Purpose |
|---|---|
| `core/data_model.py` | Canonical `ProjectInput`, `MacroContext`, `PortfolioSettings` dataclasses — the single project definition every engine consumes |
| `core/financial_engine.py` | ONE shared cash-flow table; all metrics (NPV, IRR, MIRR, payback, discounted payback, ARR, PI, DCF, EAA, break-even) derive from that same table. Also `stress_case`, `sensitivity_scan`, `tornado_scan`, `build_scenarios`, `sensitivity_summary` |
| `core/macro_engine.py` | Live Zimbabwe macro fetchers (RBZ/FRED/World Bank/exchangerate) with per-indicator `{value, source, observation_date, retrieved_at, status}` transparency; `build_macro_context()` |
| `core/risk_engine.py` | Geometric status classifier (GREEN hexagon / AMBER diamond / RED triangle), stress-test lab, sensitivity tornado, scenario matrix, `monte_carlo_npv`, Black-Scholes-style real options |
| `core/optional_pricing.py` | Re-export/wrapper for Monte-Carlo and real-options pricing; standalone `bs_call` / `bs_put` |
| `core/portfolio_optimizer.py` | MILP (PuLP + CBC) portfolio selection with greedy fallback; `optimize_portfolio`, `budget_sensitivity`, `efficient_frontier`, `build_project_table` |
| `core/ai_explainer.py` | Deterministic explainability — `explain_metric`, `build_explanation` (plain / technical / narrative), `explain_graph` for every chart |
| `core/tts_speech.py` | gTTS MP3 bytes + offline browser SpeechSynthesis widget (`speech_button_html`) with Play/Pause/Resume/Stop/Mute/Restart/speed |
| `core/report_builder.py` | fpdf2 Board PDF + plain-text report, Latin-1 safe, no emoji, `->` for arrows |
| `core/email_delivery.py` | SMTP delivery reading credentials from env/Streamlit secrets; validates and masks the recipient |
| `ui/app.py` | The complete Streamlit app (intro, login, 15 modules, robot, charts) |
| `tests/test_all.py` | Standalone PASS/FAIL suite incl. edge cases |

## 4. Data sources

Every macro indicator is tagged:
- `live` — actually retrieved from a public API (RBZ/FRED/World Bank/exchangerate.host).
- `live_proxy` — a clearly-labelled proxy series (e.g. US construction index used when ZIMSTAT is not exposed via JSON).
- `fallback_estimate` — a cached default used because the live source was unreachable. **Always labelled, never presented as official Zimbabwe data.**

Cached fetches live in `data/cache/` and expire after 12 hours; the Settings
page can force a refresh.

## 5. Demo mode

Log in (admin/admin123) and on the Project Input page press
**"Load SYNTHETIC demo projects"** to bring in six demonstration Zimbabwe
projects (road, solar, mining, manufacturing, water and aviation). All are
clearly marked as synthetic. The Portfolio Optimization page uses these by
default.

## 6. E-mail configuration

The Final Report page can e-mail the PDF. Configure SMTP in the environment
(`SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASS`) or in a Streamlit
secrets file. See `.env.example`. Without credentials the app returns a
clear, readable error and never hard-codes secrets.

## 7. Testing

`python tests/test_all.py` runs 37 checks: NPV/IRR/MIRR/payback/ARR/PI/DCF/
EAA/break-even, cumulative flows, scenarios, stress, sensitivity, Monte-Carlo,
real options, Black-Scholes parity, portfolio (incl. empty and zero-budget),
e-mail validation/masking/no-credentials, PDF and text report generation,
TTS widget, AI explainer, macro context methods, plus edge cases (negative
revenue, zero capex, zero WACC/PI, short 1-year life). It exits non-zero if
anything fails.

## 8. Known limitations

- The macro engine degrades gracefully to labelled fallback estimates when
  APIs are unreachable (network-dependent).
- The ML lab uses **synthetic** training data; it is a demonstration of
  methodology, not a real credit-risk model.
- Real-option values are illustrative Black-Scholes approximates.
- gTTS audio requires internet access; the browser speech widget works
  fully offline.

## 9. Licence / provenance

Demonstration data is synthetic. All analysis code in this repository is
provided for evaluation purposes. Validate any real capital decision with
primary ZIMSTAT/RBZ data and professional project-finance review.