"""CapEx AI Agent - Zimbabwe Capital Projects Finance Intelligence Platform.

Single Streamlit entry point. Live-reactant: any input or macro change
re-runs ONE shared cash-flow table so every metric and every chart agrees.
No secrets are hard-coded; SMTP credentials come from env / Streamlit secrets.
"""
from __future__ import annotations

import io
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Core engines
# ---------------------------------------------------------------------------
from core.data_model import PortfolioSettings, MacroContext, ProjectInput
from core.financial_engine import (
    project_cash_flows, tornado_scan, sensitivity_scan, build_scenarios,
    sensitivity_summary, stress_case,
)
from core.macro_engine import build_macro_context, fetch_zig_usd
from core.risk_engine import (
    classify_status, stress_test, monte_carlo_npv, real_options,
    sensitivity_tornado, scenario_matrix,
)
from core.ai_explainer import (
    build_explanation, explain_metric, explain_graph, card_markdown,
)
from core.portfolio_optimizer import (
    optimize_portfolio, budget_sensitivity, efficient_frontier, build_project_table,
)
from core.optional_pricing import bs_call, bs_put
from core.tts_speech import text_to_speech, speech_button_html
from core.report_builder import build_report_files
from core.email_delivery import validate_email, mask_email, send_report_email

try:
    st.set_page_config(
        page_title="CapEx AI Agent - Zimbabwe",
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    pass

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
_CSS = """
<style>
:root {
  --bg:#0a1628; --panel:#112240; --accent:#2e8bff; --fg:#e6f1ff;
  --green:#34d399; --amber:#fbbf24; --red:#ff6b6b; --gold:#fcd34d;
}
.stApp { background:#0a1628; color:#e6f1ff; }
[data-testid="stSidebar"] { background:#0d1b32; border-right:1px solid #2e8bff33; }
[data-testid="stSidebar"] * { color:#cbd5e1; }
h1,h2,h3,h4 { color:#e6f1ff !important; letter-spacing:.3px; }
.block-container { padding-top:1.4rem; }
.stButton>button {
  background:#112240; color:#e6f1ff; border:1px solid #2e8bff66;
  border-radius:8px; transition:all .25s ease; font-weight:500;
}
.stButton>button:hover { background:#2e8bff; color:#ffffff; border-color:#2e8bff; transform:translateY(-1px); }
.stDownloadButton>button {
  background:#112240; color:#e6f1ff; border:1px solid #2e8bff66; border-radius:8px; transition:all .25s;
}
.stDownloadButton>button:hover { background:#2e8bff; color:#fff; }
[data-testid="stMetricValue"] { color:#e6f1ff; font-size:1.5rem; }
[data-testid="stMetric"] { background:#112240; border:1px solid #2e8bff33; border-radius:12px; padding:12px; }
div[data-testid="stTextInput"] input, div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"]>div, div[data-testid="stSlider"] {
  background:#0a1628; color:#e6f1ff; border-radius:8px;
}
/* geometric status glyphs */
.glyph { display:inline-block; width:16px; height:16px; margin-right:8px; animation:pulse 2.4s ease-in-out infinite; }
.glyph-static { animation:none; }
.hexagon  { clip-path:polygon(25% 6%, 75% 6%, 100% 50%, 75% 94%, 25% 94%, 0% 50%); background:#34d399; }
.diamond  { clip-path:polygon(50% 0, 100% 50%, 50% 100%, 0 50%); background:#fbbf24; }
.triangle { clip-path:polygon(50% 0, 0 100%, 100% 100%); background:#ff6b6b; }
@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:.35;} }
/* intro */
.intro-slide { animation:fadeIn 1s ease both; }
@keyframes fadeIn { from{opacity:0; transform:translateY(14px);} to{opacity:1; transform:none;} }
.loadbar { height:5px; background:#0d1b32; border-radius:3px; overflow:hidden; }
.loadbar>span { display:block; height:100%; background:#2e8bff; border-radius:3px; width:0; animation:grow 2.6s ease both; }
@keyframes grow { from{width:0;} to{width:100%;} }
.robot-avatar {
  width:74px; height:74px; border-radius:50%; background:#112240;
  border:2px solid #2e8bff; display:flex; align-items:center; justify-content:center;
  font-size:36px; position:relative; box-shadow:0 0 0 0 #2e8bff66; animation:robotPulse 2.8s infinite;
}
@keyframes robotPulse { 0%{box-shadow:0 0 0 0 #2e8bff44;} 70%{box-shadow:0 0 0 16px #2e8bff00;} 100%{box-shadow:0 0 0 0 #2e8bff00;} }
.speech-bubble {
  background:#112240; border:1px solid #2e8bff55; border-left:4px solid #2e8bff;
  border-radius:12px; padding:14px 18px; color:#e6f1ff; font-size:.95rem; line-height:1.55;
}
.stat-chip { display:inline-block; padding:2px 10px; border-radius:20px; font-size:.8rem; font-weight:600; }
.chip-green { background:#34d39922; color:#34d399; border:1px solid #34d39955; }
.chip-amber { background:#fbbf2422; color:#fbbf24; border:1px solid #fbbf2455; }
.chip-red   { background:#ff6b6b22; color:#ff6b6b; border:1px solid #ff6b6b55; }
.chip-gold  { background:#fcd34d22; color:#fcd34d; border:1px solid #fcd34d55; }
hr.divider { border:none; border-top:1px solid #2e8bff33; margin:14px 0; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init_state() -> None:
    S = st.session_state
    S.setdefault("authenticated", False)
    S.setdefault("intro_step", 0)
    S.setdefault("intro_done", False)
    S.setdefault("mode", "AI GUIDED MODE")
    S.setdefault("project", ProjectInput())
    if "macro" not in S:
        S["macro"] = cached_macro()
    if "result" not in S:
        S["result"] = None
    S.setdefault("robot", {
        "running": False, "paused": False, "muted": False, "step": 0,
        "current": "Standing by", "state": "stopped",
    })
    S.setdefault("stress_drivers", {
        "inflation_pts": 0.0, "exchange_stress_pct": 0.0,
        "interest_shift_pts": 0.0, "material_growth_pct": 0.0,
        "revenue_shift_pct": 0.0, "duration_days": 0.0,
    })
    S.setdefault("last_email", "")
    S.setdefault("email_sent_msg", "")


# ---------------------------------------------------------------------------
# Cached engines (parameter-free)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def cached_macro() -> MacroContext:
    """One shared, cached Zimbabwe macro context (live where reachable)."""
    return build_macro_context()


@st.cache_data(show_spinner=False)
def cached_ml_model() -> Dict[str, Any]:
    """SYNTHETIC-DATA ML model (RandomForest) - demonstration only."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, confusion_matrix)
    from sklearn.model_selection import train_test_split
    rng = np.random.default_rng(42)
    n = 420
    complexity = rng.uniform(1, 5, n)
    design = rng.uniform(0.3, 1.0, n)
    delay = rng.uniform(0, 240, n)
    orders = rng.uniform(0, 12, n)
    imported = rng.uniform(0, 100, n)
    cap_ln = np.log10(rng.uniform(5e6, 4e8, n))
    contractor = rng.integers(0, 3, n)
    sector = rng.integers(0, 5, n)
    X0 = np.column_stack([complexity, design, delay, orders, imported, cap_ln,
                          contractor, sector])
    overrun_score = (complexity * 0.55 + (1 - design) * 0.85 + delay / 300 +
                     orders * 0.06 + imported / 200 + cap_ln * 0.05 +
                     contractor * 0.08)
    y = (overrun_score > np.percentile(overrun_score, 55)).astype(int) | \
        (rng.random(n) < 0.12)
    y = y.astype(int)
    X = pd.DataFrame(X0, columns=["complexity_score", "design_completeness",
                                  "procurement_delay_days", "num_change_orders",
                                  "imported_equipment_pct",
                                  "log_investment_usd", "contractor_type_code",
                                  "sector_code"])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=7)
    clf = RandomForestClassifier(n_estimators=220, max_depth=6,
                                 random_state=7, n_jobs=-1)
    clf.fit(Xtr, ytr)
    yp = clf.predict(Xte)
    prob = clf.predict_proba(Xte)[:, 1]
    names = ["complexity_score", "design_completeness", "procurement_delay_days",
             "num_change_orders", "imported_equipment_pct", "log_investment_usd",
             "contractor_type_code", "sector_code"]
    importances = dict(zip(names, clf.feature_importances_))
    top = sorted(importances.items(), key=lambda kv: -kv[1])[:3]
    cm = confusion_matrix(yte, yp)
    return {
        "model": clf,
        "feature_names": names,
        "importances": importances,
        "top_features": top,
        "metrics": {
            "accuracy": accuracy_score(yte, yp),
            "precision": precision_score(yte, yp, zero_division=0),
            "recall": recall_score(yte, yp, zero_division=0),
            "f1": f1_score(yte, yp, zero_division=0),
            "roc_auc": roc_auc_score(yte, prob),
        },
        "confusion_matrix": cm,
        "train_size": int(len(Xtr)),
        "test_size": int(len(Xte)),
    }


@st.cache_data(ttl=1800, show_spinner=False)
def cached_monte_carlo(p_json: str, macro_json: str, n: int, seed: int) -> Dict[str, Any]:
    from core.risk_engine import monte_carlo_npv
    p = _project_from_json(p_json)
    macros = _macro_from_json(macro_json)
    return monte_carlo_npv(p, macros, n, seed)


# ---------------------------------------------------------------------------
# Serialisation helpers for cached functions
# ---------------------------------------------------------------------------
def _project_to_json(p: ProjectInput) -> str:
    return json.dumps({f: getattr(p, f) for f in ProjectInput.__annotations__},
                      default=str)


def _project_from_json(s: str) -> ProjectInput:
    d = json.loads(s)
    d.pop("notes_flag", None)
    return ProjectInput(**d)


def _macro_to_json(m: MacroContext) -> str:
    return json.dumps({"fetched_at": m.fetched_at,
                       "indicators": m.indicators or {}}, default=str)


def _macro_from_json(s: str) -> MacroContext:
    d = json.loads(s)
    return MacroContext(fetched_at=d.get("fetched_at", ""),
                        indicators=d.get("indicators", {}))


# ---------------------------------------------------------------------------
# Compute (single shared table)
# ---------------------------------------------------------------------------
def compute(project: Optional[ProjectInput] = None,
            stress: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    p = project or st.session_state["project"]
    macro = st.session_state["macro"] or cached_macro()
    if p is None:
        p = ProjectInput()
    r = project_cash_flows(p, macro, stress or {})
    st.session_state["result"] = r
    return r


# ---------------------------------------------------------------------------
# Formatting + small UI helpers
# ---------------------------------------------------------------------------
def usd(v: Any) -> str:
    return "${:,.0f}".format(float(v or 0))


def pct(v: Any) -> str:
    return "{:,.1f}%".format(float(v or 0))


def rate(v: Any) -> str:
    return "{:,.1%}".format(float(v or 0))


def pay(v: Any) -> str:
    v = float(v)
    return "{:.1f} y".format(v) if v != float("inf") else "n/a"


def fx_display(macro: MacroContext) -> str:
    v = macro.exchange_rate_zig_per_usd()
    if v < 0.5 and v > 0:
        return "{:,.2f} ZiG per USD".format(1.0 / v)
    return "{:,.2f} ZiG per USD".format(v)


def glyph(shape: str, static: bool = False) -> str:
    static = " glyph-static" if static else ""
    return '<span class="glyph%s %s"></span>' % (
        static, shape.lower())


def status_line(status: Dict[str, str]) -> str:
    color = status.get("color", "#2e8bff")
    return ('<div><span class="glyph %s"></span>'
            '<b style="color:%s;">%s - %s</b></div>'
            % (status.get("shape", "hexagon").lower(), color,
               status.get("shape", ""), status.get("level", "")))


def chip(text: str, kind: str) -> str:
    return '<span class="stat-chip chip-%s">%s</span>' % (kind, text)


def metric_row(label: str, value: str, hint: str = "", kind: str = "green") -> None:
    mcol, hcol = st.columns([2, 2])
    mcol.metric(label, value)
    hcol.markdown('<div style="padding-top:26px;color:#8aa2c8;">%s %s</div>'
                  % (chip("AI", "gold"), hint), unsafe_allow_html=True)


def explain_block(ex: Dict[str, Any], title: str = "AI EXPLAINS THIS") -> None:
    with st.container(border=True):
        st.markdown("#### " + title)
        c1, c2 = st.columns(2)
        c1.markdown("**WHAT HAPPENED**\n\n" + ex.get("what_happened", ""))
        c2.markdown("**WHY**\n\n" + ex.get("why", ""))
        c3, c4 = st.columns(2)
        c3.markdown("**WHAT DOES IT MEAN**\n\n" + ex.get("what_it_means",
                                                         ex.get("what_does_it_mean", "")))
        c4.markdown("**MAIN DRIVER**\n\n" + ex.get("main_driver", ""))


def chapter_heading(text: str) -> None:
    st.markdown("#### " + text)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# make_chart (shared plotly builder)
# ---------------------------------------------------------------------------
def make_chart(kind: str, data: Any, title: str) -> go.Figure:
    colors = ["#2e8bff", "#34d399", "#fbbf24", "#ff6b6b", "#fcd34d", "#8aa2c8"]
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#112240",
        plot_bgcolor="#0a1628",
        font=dict(color="#e6f1ff", size=12),
        margin=dict(t=60, b=40, l=50, r=30),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    if kind == "npv_profile":
        df = data
        fig.add_trace(go.Scatter(x=df["rate"], y=df["npv"], mode="lines",
                                 line=dict(color="#2e8bff", width=3),
                                 name="NPV at rate"))
        fig.add_hline(y=0, line_dash="dash", line_color="#ff6b6b")
        fig.update_xaxes(title="Discount rate")
        fig.update_yaxes(title="NPV (USD)")
    elif kind == "tornado":
        df = data
        fig.add_trace(go.Bar(
            x=df["swing"], y=df["variable"], orientation="h",
            marker_color=[colors[i % len(colors)] for i in range(len(df))],
            name="NPV swing", text=df["swing"].map(lambda v: usd(v)),
            textposition="outside"))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title="NPV swing (USD)")
    elif kind == "scenario":
        df = data
        fig.add_trace(go.Bar(
            x=df["Scenario"], y=df["npv"],
            marker_color=["#34d399", "#2e8bff", "#ff6b6b", "#fcd34d"],
            name="NPV", text=df["npv"].map(lambda v: usd(v)),
            textposition="outside"))
        fig.add_hline(y=0, line_dash="dash", line_color="#ff6b6b")
        fig.update_yaxes(title="NPV (USD)")
    elif kind == "stress":
        df = data
        fig.add_trace(go.Bar(
            x=["NPV", "IRR", "MIRR", "Payback"],
            y=df["y"],
            marker_color=["#2e8bff" if i < len(df["y"]) else "#ff6b6b"
                          for i in range(len(df["y"]))],
            name=df.get("series", "base"),
            text=[str(v) for v in df["label"]], textposition="outside"))
        fig.update_yaxes(title="Value")
    elif kind == "sensitivity":
        df = data
        fig.add_trace(go.Scatter(
            x=df["variation_label"], y=df["npv"], mode="lines+markers",
            line=dict(color="#2e8bff", width=3),
            marker=dict(size=8, color="#34d399"), name="NPV"))
        fig.add_hline(y=0, line_dash="dash", line_color="#ff6b6b")
        fig.update_xaxes(title="Variation of input")
        fig.update_yaxes(title="NPV (USD)")
    elif kind == "monte_carlo":
        df = data
        fig.add_trace(go.Histogram(
            x=df["samples"], nbinsx=60, marker_color="#2e8bff",
            opacity=0.85, name="NPV draws"))
        fig.add_vline(x=df.get("p5", 0), line_dash="dot", line_color="#fbbf24")
        fig.add_vline(x=df.get("p95", 0), line_dash="dot", line_color="#fcd34d")
        fig.add_vline(x=df.get("mean_npv", 0), line_color="#34d399")
        fig.update_xaxes(title="NPV (USD)")
        fig.update_yaxes(title="Frequency")
    elif kind == "ml_importance":
        df = data
        fig.add_trace(go.Bar(
            x=df["importance"], y=df["feature"], orientation="h",
            marker_color="#2e8bff", name="Feature importance",
            text=df["importance"].map(lambda v: "{:.2f}".format(v)),
            textposition="outside"))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title="Importance (Gini based)")
    elif kind == "confusion":
        cm = data
        fig.add_trace(go.Heatmap(
            z=cm, x=["Predicted: no", "Predicted: yes"],
            y=["Actual: no", "Actual: yes"],
            colorscale="Blues", showscale=False,
            text=cm, texttemplate="%{text}"))
        fig.update_xaxes(side="bottom")
    elif kind == "efficient_frontier":
        df = data
        fig.add_trace(go.Scatter(
            x=df["average_risk"], y=df["total_expected_npv"],
            mode="lines+markers", line=dict(color="#34d399", width=3),
            marker=dict(size=9, color="#2e8bff"),
            name="Frontier",
            text=df["n_funded"].map(lambda v: "%d funded" % v)))
        fig.update_xaxes(title="Average portfolio risk (0-1)")
        fig.update_yaxes(title="Total expected NPV (USD)")
    elif kind == "budget_sensitivity":
        df = data
        fig.add_trace(go.Scatter(
            x=df["budget"], y=df["total_expected_npv"], mode="lines+markers",
            line=dict(color="#2e8bff", width=3), marker=dict(size=8),
            name="Portfolio NPV"))
        fig.update_xaxes(title="Budget (USD)")
        fig.update_yaxes(title="Total expected NPV (USD)")
    return fig


def show_chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Robot
# ---------------------------------------------------------------------------
GUIDED_STEPS = [
    ("Project Info", "We begin with the project definition: sector, capex, financing and macro assumptions."),
    ("What is the concept", "The concept is the economic rationale - the service the capital asset will deliver."),
    ("Calculation", "The model rebuilds the shared cash-flow table, then the NPV, IRR, MIRR, PI, payback and EAA metrics."),
    ("Result", "The headline verdict: positive or negative value creation against the cost of capital."),
    ("What does it mean", "We translate the result into what it means for the balance sheet and for Zimbabwe."),
    ("Risk", "The geometric risk status and its fully transparent classifier rules."),
    ("Stress test", "How the project behaves under a combined inflation-FX-rate-material shock."),
    ("Trend explanation", "The tornado scan identifies which single driver moves NPV the most."),
    ("Final findings", "Actionable conclusion: accept, condition or defer, with the mitigants that matter."),
]


def robot_avatar_html(emoji: str = "🤖") -> str:
    return ('<div style="display:flex;align-items:center;gap:16px;">'
            '<div class="robot-avatar">%s</div>'
            '<div><div style="font-weight:700;color:#2e8bff;">CAPEX AI AGENT</div>'
            '<div style="color:#8aa2c8;font-size:.85rem;">%s</div></div></div>'
            % (emoji, robot_status_text()))


def robot_status_text() -> str:
    rb = st.session_state["robot"]
    state = rb["state"].upper()
    return "%s - step %d/%d" % (state, min(rb["step"] + 1, len(GUIDED_STEPS)),
                                len(GUIDED_STEPS))


def robot_controls() -> None:
    rb = st.session_state["robot"]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    if c1.button("▶ START", use_container_width=True):
        rb.update(running=True, paused=False, step=0, state="RUNNING",
                  current=GUIDED_STEPS[0][1])
        st.rerun()
    if c2.button("⏸ PAUSE", use_container_width=True):
        rb.update(paused=True, state="PAUSED"); st.rerun()
    if c3.button("▶ RESUME", use_container_width=True):
        rb.update(paused=False, state="RUNNING"); st.rerun()
    if c4.button("⏹ STOP", use_container_width=True):
        rb.update(running=False, paused=False, state="STOPPED",
                  current="Standing by"); st.rerun()
    if c5.button("🔊 MUTE" if not rb.get("muted") else "🔇 UNMUTE",
                 use_container_width=True):
        rb["muted"] = not rb.get("muted", False)
        st.rerun()
    if c6.button("🔄 RESTART", use_container_width=True):
        rb.update(running=True, paused=False, step=0, state="RUNNING",
                  current=GUIDED_STEPS[0][1])
        st.rerun()


def robot_speak(text: str) -> None:
    st.markdown('<div class="speech-bubble">' + text + "</div>",
                unsafe_allow_html=True)


def robot_step() -> str:
    """Advance the guided robot one step (called from a button)."""
    rb = st.session_state["robot"]
    rb["step"] = min(rb["step"] + 1, len(GUIDED_STEPS) - 1)
    rb["current"] = GUIDED_STEPS[rb["step"]][1]
    return rb["current"]


def render_robot_panel(page_name: str, explanation_text: str) -> None:
    with st.container(border=True):
        c_left, c_right = st.columns([1, 3])
        with c_left:
            st.markdown(robot_avatar_html(), unsafe_allow_html=True)
            robot_controls()
        with c_right:
            rb = st.session_state["robot"]
            if rb["running"] and not rb["paused"]:
                step_title, step_text = GUIDED_STEPS[rb["step"]]
                robot_speak("<b>%s.</b> %s<br><span style='color:#8aa2c8;'>Current module: %s</span>"
                            % (step_title, step_text, page_name))
                if st.button("▶ NEXT STEP", use_container_width=True):
                    robot_step()
                    st.rerun()
            elif rb["paused"]:
                robot_speak("<i>Paused at: %s</i>" % rb["current"])
            else:
                robot_speak(explanation_text)


# ---------------------------------------------------------------------------
# Cinematic intro
# ---------------------------------------------------------------------------
INTRO_SLIDES = [
    ("ZIMBABWE 🇿🇼", "Capital decisions begin with understanding the economic environment."),
    ("MACROECONOMIC INTELLIGENCE", "Inflation, interest rates, foreign exchange: the forces every project lives inside."),
    ("BANKING & PROJECT FINANCE", "Lending rates, cost of capital and debt structures priced for Zimbabwe."),
    ("PUBLIC INVESTMENT & INFRASTRUCTURE", "Roads, water, energy and the assets that carry a nation forward."),
    ("CAPITAL INVESTMENT", "Every dollar allocated must create measurable value."),
    ("PROJECT RISK INTELLIGENCE", "Cost overruns and delays - identified before they become crises."),
    ("ARTIFICIAL INTELLIGENCE", "Data becomes insight. Insight becomes action."),
    ("CAPITAL PROJECTS FINANCE AI AGENT", "Explain. Analyse. Predict. Stress-Test."),
]


def page_intro() -> None:
    S = st.session_state
    total = len(INTRO_SLIDES) + 3  # 8 slides + countdown 3,2,1
    step = S["intro_step"]
    slot = st.empty()
    with slot.container():
        if step < len(INTRO_SLIDES):
            title, quote = INTRO_SLIDES[step]
            st.markdown(
                '<div class="intro-slide" style="padding:18vh 6vw;text-align:center;">'
                '<div style="font-size:3.4rem;font-weight:800;color:#2e8bff;letter-spacing:2px;">%s</div>'
                '<div style="font-size:1.3rem;color:#e6f1ff;margin-top:26px;">%s</div>'
                "</div>" % (title, quote), unsafe_allow_html=True)
            nxt = "▶ Begin Demo" if step == len(INTRO_SLIDES) - 1 else "Next slide →"
            left, right = st.columns([3, 1])
            if right.button(nxt, use_container_width=True):
                S["intro_step"] += 1
                st.rerun()
            if left.button("Skip Intro", use_container_width=True):
                S["intro_step"] = len(INTRO_SLIDES)
                st.rerun()
        elif step < total:
            countdown = total - step
            st.markdown(
                '<div class="intro-slide" style="padding:22vh 6vw;text-align:center;">'
                '<div style="font-size:7rem;font-weight:900;color:#2e8bff;">%d</div>'
                '<div style="font-size:1.2rem;color:#8aa2c8;">Preparing your workspace</div>'
                "</div>" % countdown, unsafe_allow_html=True)
            left, right = st.columns([3, 1])
            if right.button("▶ Enter", use_container_width=True):
                S["intro_step"] += 1
                st.rerun()
        else:
            S["intro_done"] = True
            slot.empty()
            return
    st.markdown('<div class="loadbar"><span></span></div><br>'
                "<div style='text-align:center;color:#8aa2c8;font-size:.8rem;'>"
                "Slide %d of %d</div>"
                % (min(step + 1, total), total), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------
def page_login() -> None:
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown(
            '<div style="background:#112240;border:1px solid #2e8bff66;'
            'border-radius:18px;padding:38px 42px;margin-top:8vh;">'
            '<div style="text-align:center;font-size:1.6rem;font-weight:800;'
            'color:#2e8bff;">CAPEX AI AGENT</div>'
            '<div style="text-align:center;color:#8aa2c8;margin-bottom:22px;">'
            "Zimbabwe Capital Projects Finance Intelligence Platform</div>"
            "</div>", unsafe_allow_html=True)
        user = st.text_input("Username")
        pw = st.text_input("Password", type="password")
        if st.button("Sign in", use_container_width=True):
            if user == "admin" and pw == "admin123":
                st.session_state["authenticated"] = True
                st.session_state["intro_done"] = True
                st.session_state["user"] = "System Administrator"
                st.rerun()
            else:
                st.error("Invalid credentials. Demo credentials: admin / admin123")
        st.info("Demo credentials below are demonstration-only, shown clearly for testing: "
                "username `admin`, password `admin123`.")


# ---------------------------------------------------------------------------
# Welcome
# ---------------------------------------------------------------------------
def page_welcome() -> None:
    st.markdown("## Welcome, System Administrator")
    st.markdown(
        "You are now connected to the Capital Projects Finance AI Agent - "
        "Zimbabwe's project finance intelligence workspace. Every number below "
        "is recalculated from one shared cash-flow table the moment any input changes.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🤖 AI GUIDED MODE", use_container_width=True):
            st.session_state["mode"] = "AI GUIDED MODE"
            st.rerun()
        st.markdown("Guided flow with the robot explaining every step end-to-end.")
    with c2:
        if st.button("👤 MANUAL MODE", use_container_width=True):
            st.session_state["mode"] = "MANUAL MODE"
            st.rerun()
        st.markdown("Pick any module from the sidebar and explore freely.")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def _load_demo_projects() -> List[ProjectInput]:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base, "data", "demo_projects.csv")
    df = pd.read_csv(csv_path, comment="#")
    projects = []
    for _, row in df.iterrows():
        fields = {f: row.get(f) for f in ProjectInput.__annotations__}
        for k, v in list(fields.items()):
            if pd.isna(v):
                fields[k] = None
        projects.append(ProjectInput(**fields))
    return projects


def page_dashboard() -> None:
    chapter_heading("ℹ️ Dashboard - Live Project Pulse")
    p = st.session_state["project"]
    r = compute(p)
    macro = st.session_state["macro"]
    status = classify_status(r)
    st.markdown(status_line(status), unsafe_allow_html=True)
    c = st.columns(4)
    c[0].metric("NPV", usd(r["npv"]), "expected value created")
    c[1].metric("IRR", rate(r["irr"]), "vs WACC " + rate(r["wacc"]))
    c[2].metric("PI", "{:.2f}".format(r["pi"]), ">1 creates value")
    c[3].metric("Expected capex", usd(r["expected_capex"]), "escalated + contingency")
    c = st.columns(4)
    c[0].metric("Payback", pay(r["payback"]), "life " + str(int(p.project_life)) + " y")
    c[1].metric("MIRR", rate(r["mirr"]), "reinvestment aware")
    c[2].metric("Inflation used", pct(r["inflation_used_pct"]), "macro-linked")
    c[3].metric("Decision", r["decision"], "")
    st.markdown("**Reason:** " + r["reason"])
    st.markdown("**Project:** %s | %s | %s" % (p.project_name, p.sector, p.province))
    st.markdown("**Macro:** inflation %s | policy %s | lending %s | %s" % (
        pct(macro.inflation_pct()), pct(macro.policy_rate_pct()),
        pct(macro.lending_rate_pct()), fx_display(macro)))
    ex = build_explanation(p, macro, r)
    explain_block(ex["plain"], "AI EXPLAINS THIS DASHBOARD")
    exg = explain_graph("NPV profile", {"npv_at_wacc": r["npv"]})
    explain_block(exg, "NPV DIRECTION")


def page_macro_monitor() -> None:
    chapter_heading("🌍 Zimbabwe Macro Monitor")
    st.caption("Live indicators with transparent source / date / status. "
               "Press ctrl+F5 style refresh or restart the app to re-fetch.")
    macro = cached_macro()
    st.session_state["macro"] = macro
    cards = [
        ("Inflation (ZiG y/y)", macro.inflation_pct(), "%", "inflation_zim"),
        ("Policy rate", macro.policy_rate_pct(), "%", "policy_rate_zim"),
        ("Lending rate", macro.lending_rate_pct(), "%", "lending_rate_zim"),
        ("US inflation", macro.us_inflation_pct(), "%", "inflation_us"),
        ("Construction index", macro.construction_price_index(), "idx", "construction_index"),
        ("Exchange rate", fx_display(macro), "ZiG/USD", "zig_usd"),
    ]
    c = st.columns(3)
    for i, (label, val, unit, key) in enumerate(cards):
        raw = (macro.get("raw") or {}).get(key, {})
        status = raw.get("status", "?")
        src = raw.get("source", "-")
        with c[i % 3]:
            st.metric(label, "{:,.1f} {}".format(val, unit), status)
            st.caption("Source: %s" % src)
    ex = explain_graph("Macro Monitor", {"main_driver": "inflation_pct"})
    explain_block(ex, "AI EXPLAINS MACRO MONITOR")


def page_project_input() -> None:
    chapter_heading("📋 Project Input")
    p = st.session_state["project"]
    st.caption("Every field drives the single shared cash-flow table. "
               "Change anything and every page recomputes.")
    with st.form("project_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            pid = st.text_input("Project ID", value=p.project_id)
            name = st.text_input("Project name", value=p.project_name)
            sector = st.selectbox("Sector", ["roads", "energy", "mining",
                                             "manufacturing", "water",
                                             "agriculture", "aviation", "other"],
                                  index=(["roads", "energy", "mining", "manufacturing",
                                          "water", "agriculture", "aviation", "other"]
                                         .index(p.sector) if p.sector in
                                         ["roads", "energy", "mining", "manufacturing",
                                          "water", "agriculture", "aviation", "other"] else 0))
            ptype = st.selectbox("Project type", ["infrastructure", "industrial",
                                                  "extractive", "commercial"],
                                 index=(["infrastructure", "industrial",
                                         "extractive", "commercial"]
                                        .index(p.project_type) if p.project_type in
                                        ["infrastructure", "industrial", "extractive",
                                         "commercial"] else 0))
            province = st.text_input("Province", value=p.province)
            cur = st.selectbox("Currency", ["USD", "ZiG"],
                               index=0 if p.currency == "USD" else 1)
        with c2:
            inv = st.number_input("Initial investment (USD)", min_value=0.0,
                                  value=float(p.initial_investment), step=1e6)
            imported = st.slider("Imported equipment (%)", 0, 100,
                                 int(p.imported_equipment_pct))
            cont = st.slider("Contingency (%)", 0.0, 40.0, float(p.contingency_pct))
            salvage = st.slider("Salvage value (%)", 0.0, 50.0, float(p.salvage_value_pct))
            months = st.slider("Construction (months)", 1, 60, int(p.construction_months))
            life = st.slider("Project life (years)", 1, 40, int(p.project_life))
        with c3:
            rev = st.number_input("Annual revenue (USD)", min_value=0.0,
                                  value=float(p.annual_revenue), step=1e6)
            rev_g = st.slider("Revenue growth (%)", -10.0, 40.0,
                              float(p.revenue_growth_pct), step=0.5)
            opex = st.number_input("Operating costs (USD)", min_value=0.0,
                                   value=float(p.operating_costs), step=1e6)
            opex_g = st.slider("Operating-cost growth (%)", 0.0, 60.0,
                               float(p.operating_cost_growth_pct), step=0.5)
            wc = st.slider("Working capital (% of revenue)", 0.0, 30.0,
                           float(p.working_capital_pct))
            term_g = st.slider("Terminal growth (%)", 0.0, 10.0,
                               float(p.terminal_growth_pct))

        st.markdown("##### Financing")
        c1, c2, c3 = st.columns(3)
        with c1:
            debt = st.slider("Debt ratio (%)", 0, 100, int(p.debt_ratio_pct))
            debt_i = st.slider("Debt interest (%)", 0.0, 40.0, float(p.debt_interest_pct))
        with c2:
            tax = st.slider("Tax rate (%)", 0.0, 50.0, float(p.tax_rate_pct))
            reinvest = st.slider("Reinvestment rate (%)", 0.0, 25.0,
                                 float(p.reinvestment_rate_pct))
        with c3:
            wacc_o = st.number_input("WACC override (%) (0 = auto from macro)",
                                     min_value=0.0, value=float(p.wacc_pct), step=0.5)
            eq = st.slider("Equity cost (%)", 0.0, 40.0, float(p.equity_cost_pct))

        st.markdown("##### Risk profile")
        c1, c2, c3 = st.columns(3)
        with c1:
            comp = st.slider("Complexity score (1-5)", 1, 5, int(p.complexity_score))
            design = st.slider("Design completeness (0-1)", 0.0, 1.0,
                               float(p.design_completeness), step=0.05)
        with c2:
            delay = st.slider("Procurement delay (days)", 0, 365,
                              int(p.procurement_delay_days))
            change = st.slider("Change orders", 0, 30, int(p.num_change_orders))
        with c3:
            ctype = st.selectbox("Contractor type", ["local_large", "international",
                                                     "local_small", "mixed"],
                                 index=(["local_large", "international", "local_small",
                                        "mixed"].index(p.contractor_type)
                                        if p.contractor_type in
                                        ["local_large", "international", "local_small", "mixed"] else 0))
            funding = st.selectbox("Funding source", ["mixed", "multilateral",
                                                      "commercial_bank", "private_developer",
                                                      "development_finance", "concessionaire",
                                                      "government", "private_enterprise"],
                                   index=(["mixed", "multilateral", "commercial_bank",
                                          "private_developer", "development_finance",
                                          "concessionaire", "government", "private_enterprise"]
                                          .index(p.funding_source)
                                          if p.funding_source in ["mixed", "multilateral",
                                          "commercial_bank", "private_developer",
                                          "development_finance", "concessionaire",
                                          "government", "private_enterprise"] else 0))

        submitted = st.form_submit_button("Apply project changes", use_container_width=True)
        if submitted:
            st.session_state["project"] = ProjectInput(
                project_id=pid, project_name=name, sector=sector, project_type=ptype,
                province=province, currency=cur, initial_investment=inv,
                imported_equipment_pct=imported, contingency_pct=cont,
                salvage_value_pct=salvage, construction_months=months,
                project_life=life, annual_revenue=rev, revenue_growth_pct=rev_g,
                operating_costs=opex, operating_cost_growth_pct=opex_g,
                working_capital_pct=wc, terminal_growth_pct=term_g,
                debt_ratio_pct=debt, debt_interest_pct=debt_i, equity_cost_pct=eq,
                wacc_pct=wacc_o, reinvestment_rate_pct=reinvest, tax_rate_pct=tax,
                complexity_score=comp, design_completeness=design,
                procurement_delay_days=delay, num_change_orders=change,
                contractor_type=ctype, funding_source=funding)
            st.rerun()
    if st.button("Load SYNTHETIC demo projects", use_container_width=True):
        demo = _load_demo_projects()
        st.session_state["demo_projects"] = demo
        st.session_state["project"] = demo[0]
        st.success("Loaded 6 demonstration projects (synthetic data). Reloaded project 1.")
        st.rerun()


def page_cash_flow() -> None:
    chapter_heading("💰 Cash Flow Projection")
    p = st.session_state["project"]
    r = compute(p)
    df = r["cash_flow_table"]
    st.dataframe(df.style.format({
        "Revenue": "USD {:,.0f}", "Operating Costs": "USD {:,.0f}",
        "EBITDA": "USD {:,.0f}", "Depreciation": "USD {:,.0f}",
        "EBIT": "USD {:,.0f}", "Tax": "USD {:,.0f}", "NOPAT": "USD {:,.0f}",
        "Operating Cash Flow": "USD {:,.0f}", "Terminal Value": "USD {:,.0f}",
        "Net Cash Flow": "USD {:,.0f}", "Cumulative CF": "USD {:,.0f}",
    }), use_container_width=True, height=420)
    c1, c2, c3 = st.columns(3)
    c1.metric("NPV", usd(r["npv"]))
    c2.metric("WACC", rate(r["wacc"]))
    c3.metric("Expected capex", usd(r["expected_capex"]))
    # NPV profile chart
    rates = np.linspace(0.0, max(r["wacc"] * 2.5, 0.30), 60)
    npvs_prof = []
    for rr in rates:
        npvs_prof.append(sum(
            cf / (1 + rr) ** t for t, cf in enumerate(r["cash_flows"])))
    prof = pd.DataFrame({"rate": rates, "npv": npvs_prof})
    show_chart(make_chart("npv_profile", prof, "NPV Profile"))
    ex = explain_graph("NPV profile",
                       {"npv_at_wacc": r["npv"], "npv_at_zero": npvs_prof[0],
                        "wacc": r["wacc"], "irr": r["irr"]})
    explain_block(ex, "AI EXPLAINS THE NPV PROFILE")
    st.caption("HOW CALCULATED: one shared DCF table (see above) is discounted "
               "at the WACC. WHAT IT MEANS: the curve crossing zero is the IRR. "
               "WHY IT CHANGED: any revenue / cost / macro input change re-runs it. "
               "WHAT UNDER STRESS: see Stress Testing page.")


def page_capital_budgeting() -> None:
    chapter_heading("🏦 Capital Budgeting")
    p = st.session_state["project"]
    r = compute(p)
    metrics = [
        ("NPV", usd(r["npv"]), "NPV = SUM(CF_t/(1+r)^t) - I0 with r = WACC " + rate(r["wacc"]),
         "Value created today above the required return."),
        ("IRR", rate(r["irr"]), "Discount rate at which NPV = 0.",
         "Compound return earned by the project."),
        ("MIRR", rate(r["mirr"]), "Reinvestment at " + pct(r["reinvestment_rate_pct"]),
         "More realistic return when cash is reinvested."),
        ("Payback", pay(r["payback"]), "Years until cumulative CF >= capex.",
         "Liquidity recovery speed."),
        ("Discounted payback", pay(r["discounted_payback"]), "Payback at present values.",
         "Recovery adjusted for the cost of capital."),
        ("ARR", rate(r["arr"]), "Average NOPAT / expected capex.",
         "Accounting-return view of profitability."),
        ("PI", "{:.2f}".format(r["pi"]), "PV(inflows) / I0.",
         "Value created per dollar of capital."),
        ("DCF value", usd(r["dcf_value"]), "Discounted operating cash flows.",
         "Raw operating value before investment."),
        ("EAA", usd(r["eaa"]), "NPV converted to a level annuity.",
         "Annual-equivalent value for ranking."),
        ("Break-even revenue", usd(r["breakeven_revenue"]), "Revenue for NPV = 0.",
         "Minimum revenue to avoid value destruction."),
        ("WACC", rate(r["wacc"]), "Debt + equity blended cost.",
         "The hurdle rate the project must beat."),
    ]
    cols = st.columns(3)
    for i, (label, val, how, meaning) in enumerate(metrics):
        with cols[i % 3]:
            st.metric(label, val)
            st.caption(how)
            st.markdown("_What it means:_ " + meaning)
    ex = explain_metric("NPV", r["npv"], 0.0)
    explain_block(ex, "AI EXPLAINS NPV")
    ex = explain_metric("IRR", r["irr"], r["wacc"])
    explain_block(ex, "AI EXPLAINS IRR")

    st.markdown("### Decision")
    st.markdown("**%s** - %s" % (r["decision"], r["reason"]))


def page_risk_ml() -> None:
    chapter_heading("🛡️ Risk and Machine Learning")
    p = st.session_state["project"]
    r = compute(p)
    status = classify_status(r)
    st.markdown(status_line(status), unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("**Transparent classifier rules** (never a black box):")
        st.markdown("• RED triangle  = NPV < 0 OR IRR < WACC OR PI < 1")
        st.markdown("• AMBER diamond = NPV > 0 but payback > 70% of life OR PI < 1.25 OR +10pt inflation flips NPV")
        st.markdown("• GREEN hexagon = otherwise (stable and resilient)")
        st.caption("Triggered reasons: " + status["reasons"])

    st.markdown("### Monte-Carlo NPV (2000 correlated macro draws)")
    mc = cached_monte_carlo(_project_to_json(p), _macro_to_json(st.session_state["macro"]),
                            2000, 7)
    show_chart(make_chart("monte_carlo", mc, "Monte-Carlo NPV Distribution"))
    c = st.columns(5)
    c[0].metric("Mean NPV", usd(mc["mean_npv"]))
    c[1].metric("P5", usd(mc["p5"]))
    c[2].metric("P50", usd(mc["p50"]))
    c[3].metric("P95", usd(mc["p95"]))
    c[4].metric("P(NPV>0)", "{:.0%}".format(mc["p_positive"]))
    ex = explain_graph("Monte Carlo distribution", mc)
    explain_block(ex, "AI EXPLAINS MONTE-CARLO")

    st.markdown("### Real options (Black-Scholes style)")
    ro = real_options(p, r["npv"])
    c = st.columns(3)
    c[0].metric("Expansion value", usd(ro["expansion_value"]))
    c[1].metric("Abandonment value", usd(ro["abandonment_value"]))
    c[2].metric("Combined NPV", usd(ro["combined_npv"]))
    st.caption(ro["assumptions"])

    st.markdown("### ML Cost-Overrun lab - SYNTHETIC DATA (demonstration)")
    ml = cached_ml_model()
    c = st.columns(4)
    m = ml["metrics"]
    c[0].metric("Accuracy", "{:.1%}".format(m["accuracy"]))
    c[1].metric("Precision", "{:.1%}".format(m["precision"]))
    c[2].metric("Recall", "{:.1%}".format(m["recall"]))
    c[3].metric("ROC-AUC", "{:.2f}".format(m["roc_auc"]))
    c = st.columns(4)
    c[0].metric("F1", "{:.1%}".format(m["f1"]))
    c[1].metric("Train samples", ml["train_size"])
    c[2].metric("Test samples", ml["test_size"])
    c[3].caption("Model brand label: RandomForestClassifier, fully explainable via feature importances (model explanations below).")
    feats = pd.DataFrame([
        {"feature": ml["feature_names"], "importance": [ml["importances"]
                                                        [f] for f in ml["feature_names"]]}
    ]).explode(["feature", "importance"]).reset_index(drop=True)
    feats["importance"] = feats["importance"].astype(float)
    feats = feats.sort_values("importance", ascending=False)
    show_chart(make_chart("ml_importance", feats, "Feature Importance (RF)"))
    show_chart(make_chart("confusion", ml["confusion_matrix"],
                          "Confusion Matrix - Cost Overrun (SYNTHETIC)"))

    # Project-level prediction
    Xproj = pd.DataFrame([{
        "complexity_score": p.complexity_score,
        "design_completeness": p.design_completeness,
        "procurement_delay_days": p.procurement_delay_days,
        "num_change_orders": p.num_change_orders,
        "imported_equipment_pct": p.imported_equipment_pct,
        "log_investment_usd": np.log10(max(p.initial_investment, 1e6)),
        "contractor_type_code": ["local_large", "international", "local_small",
                                 "mixed"].index(p.contractor_type) if p.contractor_type in
                                ["local_large", "international", "local_small", "mixed"] else 0,
        "sector_code": ["roads", "energy", "mining", "manufacturing", "water"].
        index(p.sector) if p.sector in ["roads", "energy", "mining", "manufacturing", "water"] else 0,
    }])[ml["feature_names"]]
    overrun_prob = float(ml["model"].predict_proba(Xproj)[0, 1])
    delay_prob = min(0.05 + overrun_prob * 0.85 + p.procurement_delay_days / 1600, 0.95)
    c = st.columns(2)
    c[0].metric("Predicted cost-overrun probability", "{:.0%}".format(overrun_prob))
    c[1].metric("Predicted delay probability", "{:.0%}".format(delay_prob))
    top = ml["top_features"][0]
    ex = explain_graph("ML feature importance",
                       {"accuracy": m["accuracy"], "top_feature": top[0]})
    explain_block(ex, "AI EXPLAINS THE ML LAB")
    st.warning("SYNTHETIC DATA (demonstration): this model is trained on "
               "generated records, NOT real Zimbabwe project data, and must "
               "never be used for bankable credit decisions.")


def page_stress_testing() -> None:
    chapter_heading("🧪 Macro Stress Test Lab")
    p = st.session_state["project"]
    st.caption("Slide each macro driver; the ENTIRE project recomputes and "
               "every chart updates, comparing Base vs Stress.")
    drivers = st.session_state["stress_drivers"]
    c = st.columns(3)
    drivers["inflation_pts"] = c[0].slider(
        "Inflation shock (pts)", -20.0, 40.0, float(drivers["inflation_pts"]),
        step=1.0, help="Added to the current ZiG inflation rate")
    drivers["exchange_stress_pct"] = c[1].slider(
        "ZiG depreciation (%)", -20.0, 60.0, float(drivers["exchange_stress_pct"]),
        step=1.0, help="+ = ZiG weaker vs USD")
    drivers["interest_shift_pts"] = c[2].slider(
        "Interest-rate shift (pts)", -10.0, 20.0, float(drivers["interest_shift_pts"]),
        step=0.5, help="Shift applied to the cost of capital")
    c = st.columns(3)
    drivers["material_growth_pct"] = c[0].slider(
        "Material-cost growth (%)", -10.0, 40.0, float(drivers["material_growth_pct"]),
        step=1.0)
    drivers["revenue_shift_pct"] = c[1].slider(
        "Revenue-growth shift (pts)", -10.0, 10.0, float(drivers["revenue_shift_pct"]),
        step=0.5)
    drivers["duration_days"] = c[2].slider(
        "Schedule slip (days)", -120, 300, int(drivers["duration_days"]), step=15)
    st.session_state["stress_drivers"] = drivers

    st_t = stress_test(p, st.session_state["macro"], drivers)
    base, sts = st_t["base"], st_t["stressed"]
    st.markdown("### Base vs Stress")
    c = st.columns(3)
    c[0].metric("NPV base", usd(base["npv"]))
    c[1].metric("NPV stressed", usd(sts["npv"]))
    c[2].metric("Delta NPV", usd(sts["npv"] - base["npv"]), delta_color="inverse")
    c = st.columns(3)
    c[0].metric("IRR base", rate(base["irr"]))
    c[1].metric("IRR stressed", rate(sts["irr"]))
    c[2].metric("Delta IRR", rate(sts["irr"] - base["irr"]), delta_color="inverse")
    c = st.columns(3)
    c[0].metric("MIRR base", rate(base["mirr"]))
    c[1].metric("MIRR stressed", rate(sts["mirr"]))
    c[2].metric("Delta MIRR", rate(sts["mirr"] - base["mirr"]),
                delta_color="inverse")
    c = st.columns(3)
    c[0].metric("Capex base", usd(base["expected_capex"]))
    c[1].metric("Capex stressed", usd(sts["expected_capex"]))
    c[2].metric("Delta capex", usd(sts["expected_capex"] - base["expected_capex"]),
                delta_color="inverse")
    c = st.columns(3)
    c[0].metric("Payback base", pay(base["payback"]))
    c[1].metric("Payback stressed", pay(sts["payback"]))
    c[2].metric("Delta payback", sts["payback"] - base["payback"],
                delta_color="inverse")
    sc = pd.DataFrame({
        "y": [base["npv"], sts["npv"], base["irr"], sts["irr"], base["mirr"],
              sts["mirr"], base["expected_capex"], sts["expected_capex"]],
        "label": ["Base", "Stress", "Base", "Stress", "Base", "Stress",
                  "Base", "Stress"],
        "series": ["NPV"] * 2 + ["IRR"] * 2 + ["MIRR"] * 2 + ["Capex"] * 2,
    })
    show_chart(make_chart("stress", sc, "Base vs Stress Comparison"))
    st.markdown("### Verdict")
    st.markdown(st_t["verdict"])
    st.session_state["result"] = sts
    ex = explain_graph("Stress Test",
                       {"base_npv": base["npv"], "stress_npv": sts["npv"],
                        "delta_npv": sts["npv"] - base["npv"],
                        "worst_driver": max(st_t["delta"], key=lambda k:
                                            abs(st_t["delta"][k]))})
    explain_block(ex, "AI EXPLAINS THE STRESS TEST")


def page_scenario_analysis() -> None:
    chapter_heading("🔄 Scenario Analysis")
    p = st.session_state["project"]
    df = build_scenarios(p)
    show_chart(make_chart("scenario", df, "Scenario NPV Comparison"))
    st.dataframe(df.style.format({
        "npv": "USD {:,.0f}", "irr": "{:.1%}", "mirr": "{:.1%}",
        "payback": "{:.1f}", "pi": "{:.2f}"}), use_container_width=True)
    ex = explain_graph("Scenario Analysis",
                       {"base_npv": float(df.loc[df["Scenario"] == "Base", "npv"].iloc[0]),
                        "best_npv": float(df["npv"].max()),
                        "worst_npv": float(df["npv"].min())})
    explain_block(ex, "AI EXPLAINS THE SCENARIOS")
    sm = scenario_matrix(p)
    for name, res in sm.items():
        st.markdown("**%s:** NPV %s | IRR %s | Payback %s" % (
            name, usd(res["npv"]), rate(res["irr"]), pay(res["payback"])))


def page_sensitivity_analysis() -> None:
    chapter_heading("🎯 Sensitivity Analysis")
    p = st.session_state["project"]
    to = tornado_scan(p)
    show_chart(make_chart("tornado", to, "Tornado - One Driver at a Time"))
    st.write(to)
    variable = st.selectbox("Variable to sweep",
                            ["revenue", "operating_costs", "initial_investment",
                             "cost_growth", "project_life"])
    sens = sensitivity_scan(p, variable=variable)
    show_chart(make_chart("sensitivity", sens,
                          "Sensitivity - %s" % variable))
    st.markdown(sensitivity_summary(p))
    main = to.iloc[0]["variable"]
    ex = explain_graph("Sensitivity Analysis",
                       {"main_driver": main, "max_swing": to.iloc[0]["swing"],
                        "shift_pct": "-20% to +20%"})
    explain_block(ex, "AI EXPLAINS THE TORNADO")
    st.caption("WHAT HAPPENED: each driver moved -20% and +20% keeping all "
               "else constant. WHY: isolates influence. WHAT IT MEANS: the "
               "longest bar is the exposure to manage. UNDER STRESS: combine "
               "the main driver with the Stress Lab before deciding.")


def page_portfolio_optimization() -> None:
    chapter_heading("📦 Portfolio Optimization (MILP)")
    st.caption("PuLP + CBC integer programming selects the budget-maximising "
               "set of projects. Demo portfolio = SYNTHETIC Zimbabwe projects.")
    demo = st.session_state.get("demo_projects") or _load_demo_projects()
    st.session_state["demo_projects"] = demo
    table = build_project_table(demo)
    budget = st.slider("Capital budget (USD millions)", 50, 1000, 300, step=10) * 1e6
    max_risk = st.slider("Max risk exposure (weighted 0-1)", 0.05, 1.0, 0.6,
                         step=0.05)
    result = optimize_portfolio(table, budget, max_risk_exposure=max_risk,
                                min_strategic=0.0)
    st.markdown("**Status:** %s | Solver: %s" % (result["status"],
                                                 result.get("solver", "?")))
    c = st.columns(4)
    c[0].metric("Total capex", usd(result["total_capex"]))
    c[1].metric("Total expected NPV", usd(result["total_expected_npv"]))
    c[2].metric("Budget utilisation", "{:.1f}%".format(
        result["budget_utilization_pct"]))
    c[3].metric("Average risk", "{:.2f}".format(result["average_risk"]))
    sel = pd.DataFrame(result["selected_projects"])
    if not sel.empty:
        st.markdown("### Funded")
        st.dataframe(sel, use_container_width=True)
    defr = pd.DataFrame(result["deferred_projects"])
    if not defr.empty:
        st.markdown("### Deferred")
        st.dataframe(defr, use_container_width=True)
    budget_range = list(np.linspace(50e6, table["baseline_capex"].sum() * 1.15, 12))
    bs = budget_sensitivity(table, budget_range, max_risk_exposure=max_risk)
    show_chart(make_chart("budget_sensitivity", bs, "Budget Sensitivity"))
    ff = efficient_frontier(table, n_points=10, budget=budget)
    show_chart(make_chart("efficient_frontier", ff, "Efficient Frontier"))
    ex = explain_graph("Sensitivity Analysis", {
        "main_driver": "capital budget ceiling",
        "max_swing": float(bs["total_expected_npv"].max() -
                           bs["total_expected_npv"].min())})
    explain_block(ex, "AI EXPLAINS THE PORTFOLIO")
    st.caption("Rules: maximise weighted expected NPV subject to budget, risk "
               "ceiling, sector exposure caps and strategic constraints. "
               "Empty selection is a legitimate solver outcome given the risk cap.")


def page_ai_explanation() -> None:
    chapter_heading("🤖 AI Explanation Center")
    p = st.session_state["project"]
    r = compute(p)
    macro = st.session_state["macro"]
    ex = build_explanation(p, macro, r)
    st.markdown("### Simple View (management)")
    with st.container(border=True):
        for k, v in ex["plain"].items():
            st.markdown("**%s**\n\n%s" % (k.replace('_', ' ').upper(), v))
    st.markdown("### Technical View (financial engineer)")
    with st.container(border=True):
        for k, v in ex["technical"].items():
            st.markdown("**%s**\n\n%s" % (k.replace('_', ' ').upper(), v))
    st.markdown("### Narrative (robot read)")
    robot_speak(ex["narrative"])
    metric = st.selectbox("Metric to explain", ["NPV", "IRR", "MIRR", "PI",
                                                "payback", "ARR"])
    vals = {"NPV": (r["npv"], 0.0), "IRR": (r["irr"], r["wacc"]),
            "MIRR": (r["mirr"], r["wacc"]), "PI": (r["pi"], 1.0),
            "payback": (r["payback"], 0.7 * p.project_life),
            "ARR": (r["arr"], r["wacc"])}[metric]
    exm = explain_metric(metric, vals[0], vals[1])
    explain_block(exm, "AI EXPLAINS " + metric)


def page_ai_robot() -> None:
    chapter_heading("🤖 AI Robot")
    st.markdown("### Guided walk-through of the whole analysis")
    render_robot_panel("AI Robot", "I explain every module: project info, "
                                    "concept, calculation, result, meaning, "
                                    "risk, stress, trend, and final findings. "
                                    "Press START and step through with NEXT STEP.")
    st.markdown("### Manual: make the robot explain any module")
    target = st.selectbox("Pick a module", [
        "Dashboard", "Project Input", "Cash Flow", "Capital Budgeting",
        "Risk & ML", "Stress Testing", "Scenario Analysis",
        "Sensitivity Analysis", "Portfolio Optimization", "Final Report"])
    guide = {
        "Dashboard": "The dashboard is the pulse: NPV, IRR, PI, capex and the "
                     "geometric status all in one view.",
        "Project Input": "Project Input defines the asset, the money and the "
                         "risk profile that feed the single shared model.",
        "Cash Flow": "Cash Flow builds the year-by-year revenue, cost, "
                     "depreciation, tax and net-flow table.",
        "Capital Budgeting": "Capital Budgeting distils the table into NPV, "
                             "IRR, MIRR, payback, ARR, PI, DCF, EAA and "
                             "break-even.",
        "Risk & ML": "Risk & ML adds the geometric status, Monte-Carlo "
                     "distribution, real options and a synthetic-data ML lab.",
        "Stress Testing": "Stress Testing lets you shock inflation, FX, rates "
                          "and materials and watch every metric move together.",
        "Scenario Analysis": "Scenario Analysis prices Optimistic, Base and "
                             "Pessimistic Zimbabwe macro paths on the same model.",
        "Sensitivity Analysis": "Sensitivity Analysis isolates the single "
                                "driver that moves NPV the most (tornado).",
        "Portfolio Optimization": "Portfolio Optimization picks the budget "
                                  "portfolio via MILP integer programming.",
        "Final Report": "Final Report produces the Board PDF, audio narration "
                        "and e-mail delivery.",
    }[target]
    robot_speak(guide)
    st.markdown("### Robot narrative for the current project")
    ex = build_explanation(st.session_state["project"],
                           st.session_state["macro"], compute())
    robot_speak(ex["narrative"])


def page_data_sources() -> None:
    chapter_heading("🔎 Data Sources")
    st.caption("Full transparency: Variable | Source | Observation Date | "
               "Retrieved Date | Value | Unit | Status | How it integrates")
    macro = st.session_state["macro"]
    raw = macro.get("raw") or {}
    cards = {
        "zig_usd": ("Exchange rate ZiG/USD", "ZiG/USD",
                    "Drives imported-equipment capex (hard currency share)"),
        "inflation_zim": ("Zimbabwe inflation (ZiG y/y)", "%",
                          "Escalates revenue and operating costs year by year"),
        "inflation_us": ("US inflation", "%",
                         "Reference for imported price escalation"),
        "policy_rate_zim": ("RBZ policy rate", "%",
                            "Risk-free anchor for the cost of equity"),
        "lending_rate_zim": ("Commercial lending rate", "%",
                             "Sets debt cost in the WACC"),
        "policy_rate_us": ("US policy rate", "%",
                           "Global alternative cost of capital benchmark"),
        "construction_index": ("Construction price index", "index",
                               "Escalates expected capex during build-out"),
    }
    rows = []
    for key, (label, unit, integration) in cards.items():
        d = (raw or {}).get(key, {})
        if not isinstance(d, dict):
            d = {}
        val = d.get("value")
        val_s = "{:,.2f}".format(val) if isinstance(val, (int, float)) else str(val)
        rows.append({
            "Variable": label, "Source": d.get("source", "unavailable"),
            "Observation Date": d.get("observation_date") or "n/a",
            "Retrieved Date": d.get("retrieved_at") or "n/a",
            "Value": val_s, "Unit": unit,
            "Status": d.get("status", "unavailable"),
            "Method of Integration": integration,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.caption("Status legend: `live` = OpenAPI fetch succeeded; "
               "`live_proxy` = proxy series clearly labelled; "
               "`fallback_estimate` = labelled default used because the "
               "source was unreachable. Fallbacks are never disguised as "
               "official Zimbabwe data.")
    ex = explain_graph("Data Sources", {"main_driver": "input quality"})
    explain_block(ex, "AI EXPLAINS DATA QUALITY")


def page_settings() -> None:
    chapter_heading("⚙️ Settings")
    macro = cached_macro()
    st.session_state["macro"] = macro
    if st.button("🔄 Refresh live macro data (re-fetch)", use_container_width=True):
        build_macro_context(refresh=True)
        st.cache_data.clear()
        st.rerun()
    st.markdown("### Default model behaviour")
    st.markdown("• WACC 0 = auto from macro (policy rate, country risk, "
                "complexity premium, lending rate after tax).")
    st.markdown("• Operating-cost growth default 30% reflects Zimbabwe cost "
                "escalation; change per project.")
    st.markdown("• Every metric derives from ONE shared cash-flow table - "
                "no duplicated math anywhere.")
    st.markdown("### Environment")
    for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASS"):
        v = os.environ.get(k)
        st.markdown("• `%s`: %s" % (k, "configured" if v else "not set"))
    st.caption("Credentials are read from environment variables or "
               "~/.streamlit/secrets.toml only; never hard-coded.")
    if st.button("Reset session state", use_container_width=True):
        for k in [k for k in list(st.session_state.keys()) if k != "authenticated"]:
            del st.session_state[k]
        st.session_state["authenticated"] = True
        st.rerun()


def page_final_report() -> None:
    chapter_heading("📄 Final Report & Delivery")
    p = st.session_state["project"]
    r = compute(p)
    macro = st.session_state["macro"]
    st.caption("Generate the Board-ready PDF plus a plain-text variant, "
               "listen to an AI narration, and e-mail the report.")
    sections_all = ["Executive Summary", "Zimbabwe Economic Environment",
                    "Live Macro Data", "Cash Flow", "Capital Budgeting",
                    "Risk and ML", "Stress Test", "Sensitivity", "Scenarios",
                    "AI Explanation", "Key Assumptions", "Data Limitations",
                    "Final Findings"]
    selected = st.multiselect("Sections to include", sections_all,
                              default=sections_all)
    if st.button("📄 Generate report", use_container_width=True):
        with st.spinner("Building report..."):
            files = build_report_files(p, macro, r, selected)
            st.session_state["report_pdf"] = files["pdf"]
            st.session_state["report_txt"] = files["txt"]
        st.success("Report generated.")
    pdf_path = st.session_state.get("report_pdf")
    txt_path = st.session_state.get("report_txt")
    if pdf_path and os.path.exists(pdf_path):
        with open(pdf_path, "rb") as fh:
            st.download_button("⬇ DOWNLOAD PDF",
                               fh.read(), file_name=os.path.basename(pdf_path),
                               mime="application/pdf", use_container_width=True)
    if txt_path and os.path.exists(txt_path):
        with open(txt_path, "rb") as fh:
            st.download_button("⬇ DOWNLOAD TEXT",
                               fh.read(), file_name=os.path.basename(txt_path),
                               mime="text/plain", use_container_width=True)

    st.markdown("### 🎧 Audio narration")
    ex = build_explanation(p, macro, r)
    narrative = ex["narrative"]
    st.markdown("#### ▶ PLAY AUDIO (browser speech)")
    st.components.v1.html(speech_button_html(narrative, "final"), height=150)
    if st.button("⬇ DOWNLOAD AUDIO (gTTS MP3)", use_container_width=True):
        mp3 = text_to_speech(narrative)
        if mp3:
            st.session_state["audio_mp3"] = mp3
            st.success("Audio ready for download.")
        else:
            st.warning("gTTS unavailable (offline?). Use the browser player above.")
    if st.session_state.get("audio_mp3"):
        st.download_button("⬇ SAVE AUDIO FILE", st.session_state["audio_mp3"],
                           file_name="capex_ai_report.mp3", mime="audio/mpeg",
                           use_container_width=True)

    st.markdown("### ✉️ E-mail delivery")
    recipient_default = st.session_state.get("last_email") or p.submitter_email or ""
    recipient = st.text_input("Recipient e-mail", value=recipient_default)
    subject = st.text_input("Subject", value="CapEx AI Agent - Management Report for %s" % p.project_id)
    if st.button("SEND FINAL REPORT", use_container_width=True):
        if not validate_email(recipient):
            st.error("Invalid e-mail address format.")
        else:
            body = "Attached: %s management report.\nProject: %s (%s)" % (
                p.project_name, p.project_id, p.sector)
            att = pdf_path if (pdf_path and os.path.exists(pdf_path)) else None
            res = send_report_email(recipient, subject, body, attachment_path=att)
            st.session_state["last_email"] = recipient
            if res["success"]:
                st.success(res["message"] + " (recipient masked for privacy)")
            else:
                st.error(res["message"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    _init_state()
    if not st.session_state["authenticated"]:
        if not st.session_state["intro_done"]:
            page_intro()
            return
        page_login()
        return

    # ------------------------------------------------------------------ robot banner on every page
    with st.sidebar:
        st.markdown(robot_avatar_html(), unsafe_allow_html=True)
        page = st.radio("Navigate",
                        ["🏠 Dashboard", "🌍 Zimbabwe Macro Monitor",
                         "📋 Project Input", "💰 Cash Flow",
                         "🏦 Capital Budgeting", "🛡️ Risk & ML",
                         "🧪 Stress Testing", "🔄 Scenario Analysis",
                         "🎯 Sensitivity Analysis", "📦 Portfolio Optimization",
                         "🤖 AI Explanation", "🤖 AI Robot",
                         "🔎 Data Sources", "⚙️ Settings", "📄 Final Report"])
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("**Mode:** %s" % st.session_state["mode"])
        if st.button("Sign out", use_container_width=True):
            st.session_state["authenticated"] = False
            st.session_state["intro_done"] = False
            st.session_state["intro_step"] = 0
            st.rerun()

    pages = {
        "🏠 Dashboard": page_dashboard,
        "🌍 Zimbabwe Macro Monitor": page_macro_monitor,
        "📋 Project Input": page_project_input,
        "💰 Cash Flow": page_cash_flow,
        "🏦 Capital Budgeting": page_capital_budgeting,
        "🛡️ Risk & ML": page_risk_ml,
        "🧪 Stress Testing": page_stress_testing,
        "🔄 Scenario Analysis": page_scenario_analysis,
        "🎯 Sensitivity Analysis": page_sensitivity_analysis,
        "📦 Portfolio Optimization": page_portfolio_optimization,
        "🤖 AI Explanation": page_ai_explanation,
        "🤖 AI Robot": page_ai_robot,
        "🔎 Data Sources": page_data_sources,
        "⚙️ Settings": page_settings,
        "📄 Final Report": page_final_report,
    }

    # ---------------------------------------------------------- page scatter
    pages[page]()

    # ------------------------------------------------------------------ AI guided explainer strip
    if st.session_state["mode"] == "AI GUIDED MODE" and page not in (
            "🤖 AI Robot", "📄 Final Report", "⚙️ Settings"):
        render_robot_panel(page_from_emoji(page), "I am following this module "
                             "with you. Press START to run the full guided "
                             "narrative or this module's explanation is above.")


def page_from_emoji(page: str) -> str:
    name = page.split(maxsplit=1)[1] if page.count(" ") else page
    return name.replace("&", "and")


if __name__ == "__main__":
    main()