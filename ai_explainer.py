"""Deterministic AI explainability engine.

Every chart and every number in the app can be explained in accessible,
plain English that cites the ACTUAL computed figures. Nothing here is a
neural net - it is deterministic, auditable and transparent so a
non-finance reader understands WHAT happened, WHY, WHAT it means and which
single driver matters most.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from core.data_model import MacroContext, ProjectInput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _usd(v: float) -> str:
    return "${:,.0f}".format(v) if abs(v) >= 0.5 else "${:,.2f}".format(v)


def _pct(v: float) -> str:
    return "{:,.1f}%".format(v)


def _rate(v: float) -> str:
    return "{:,.1%}".format(v)


def _first_letter(text: str) -> str:
    return text[0].upper() + text[1:] if text else text


# ---------------------------------------------------------------------------
# Single-metric explanation
# ---------------------------------------------------------------------------

def explain_metric(name: str, value: float, threshold: float,
                   higher_is_better: bool = True) -> Dict[str, str]:
    """Explain one metric vs a threshold in the 4-card AI format.

    Parameters
    ----------
    name            : human label, e.g. "Net Present Value (NPV)"
    value           : the computed value
    threshold       : the decision threshold it is compared against
    higher_is_better: True when a value ABOVE threshold is good
    """
    good = value >= threshold if higher_is_better else value <= threshold
    direction = "above" if value >= 0 else "below"
    status = "GOOD" if good else "AT RISK"

    if name.lower() in ("npv", "net present value"):
        what = ("NPV is %s, which is %s the zero mark required to create value."
                % (_usd(value), direction))
        why = ("It is the present value of every future cash flow minus the "
               "initial capital outlay, using the %s cost of capital." % _rate(threshold))
        main_driver = ("The difference between revenue and operating cost "
                       "growth is what moves NPV most.")
    elif name.lower() in ("irr", "internal rate of return"):
        ev = "above" if good else "below"
        what = "IRR is %s, %s the %s cost of capital." % (_rate(value), ev, _rate(threshold))
        why = ("IRR is the discount rate at which the project's cash flows "
               "exactly pay back the investment - the break-even return.")
        main_driver = ("Timing and size of early cash flows dominate IRR.")
    elif name.lower() in ("mirr", "modified internal rate of return"):
        what = "MIRR is %s, giving a reinvestment-aware view of return." % _rate(value)
        why = ("MIRR assumes positive cash flows are reinvested at the "
               "reinvestment rate rather than at the IRR itself.")
        main_driver = "MIRR is most sensitive to the reinvestment rate you set."
    elif "payback" in name.lower():
        what = "Payback period is %s years." % ("{:.1f}".format(value)
                                                if value != float("inf") else "not reached")
        why = ("It is the time needed for cumulative cash inflows to recover "
               "the initial investment.")
        main_driver = "The size of year-one cash flow governs payback."
    elif name.lower() in ("pi", "profitability index"):
        what = "Profitability Index is %s, %s the 1.0 value-creation line." % (
            "{:.2f}".format(value), "above" if good else "below")
        why = ("PI is the present value of positive cash flows divided by the "
               "capital invested; above 1 means value is created per dollar.")
        main_driver = "PI is the ratio between cash in and cash out - driven by revenue."
    else:
        what = "%s is %s." % (name, _usd(value) if abs(value) > 100 else "{:,.2f}".format(value))
        why = "This value is compared against a threshold of %s." % _usd(threshold)
        main_driver = "Inspect the project cash-flow table for its main driver."

    return {
        "what_happened": _first_letter(what),
        "why": _first_letter(why),
        "what_it_means": (
            "For decision-making: this metric is %s. %s"
            % (status,
               "It passes the investment rule." if good
               else "It fails the investment rule and warrants mitigation or rejection.")
        ),
        "main_driver": _first_letter(main_driver),
    }


# ---------------------------------------------------------------------------
# Full project / result explanation
# ---------------------------------------------------------------------------

def build_explanation(project: ProjectInput,
                      macro: Optional[MacroContext],
                      result: Dict[str, Any]) -> Dict[str, Any]:
    """Build the three views (plain / technical / narrative) for one result."""
    macro = macro or MacroContext()

    npv = float(result.get("npv", 0.0))
    irr = float(result.get("irr", 0.0))
    mirr = float(result.get("mirr", 0.0))
    wacc = float(result.get("wacc", 0.0))
    payback = float(result.get("payback", float("inf")))
    pi = float(result.get("pi", 0.0))
    capex = float(result.get("expected_capex", project.initial_investment))
    be = float(result.get("breakeven_revenue", 0.0))
    arr = float(result.get("arr", 0.0))
    eaa = float(result.get("eaa", 0.0))
    inf = float(result.get("inflation_used_pct", macro.inflation_pct()))
    decision = str(result.get("decision", "N/A"))

    good = npv >= 0 and irr >= wacc and pi >= 1.0

    # ----------------------------------------------------------- plain view
    if npv >= 0:
        npv_msg = ("The project creates an estimated %s of value today." % _usd(npv))
    else:
        npv_msg = ("The project currently destroys %s of value under these assumptions."
                   % _usd(abs(npv)))

    main_driver = _main_driver(project, result)

    plain = {
        "what_happened": (
            "Based on your inputs, the project shows a Net Present Value of %s "
            "with an Internal Rate of Return of %s against a %s cost of capital, "
            "and a payback period of %s years."
            % (_usd(npv), _rate(irr), _rate(wacc),
               "{:.1f}".format(payback) if payback != float("inf") else "over the project life")
        ),
        "why": (
            "The model projected revenue and operating costs over %s years, "
            "escalated them for %s inflation, converted capital into %s expected "
            "capex, and discounted everything at the %s weighted-average cost of "
            "capital." % (int(project.project_life), _pct(inf), _usd(capex), _rate(wacc))
        ),
        "what_does_it_mean": npv_msg + (
            " Management verdict: %s." % decision
        ),
        "main_driver": main_driver,
    }

    # ------------------------------------------------------ technical view
    technical = {
        "what_happened": (
            "NPV = SUM(CF_t / (1+r)^t) - I0 = %s where r = WACC %s and I0 = "
            "%s expected capex. IRR = %s (discount rate where NPV = 0); "
            "MIRR = %s; PI = %s; EAA = %s; break-even revenue = %s."
            % (_usd(npv), _rate(wacc), _usd(capex), _rate(irr), _rate(mirr),
               "{:.2f}".format(pi), _usd(eaa), _usd(be))
        ),
        "why": (
            "Cash flows are taxed (NOPAT = EBIT - tax), depreciation is added "
            "back, and terminal value (salvage + perpetual growth) is released "
            "in year %s. Year-0 net cash-flow includes working-capital set-up."
            % int(project.project_life)
        ),
        "what_does_it_mean": (
            "Decision rule: ACCEPT if NPV > 0, IRR > WACC and PI > 1. "
            "Computed decision = %s. Accounting gradient ARR = %s."
            % (decision, _rate(arr))
        ),
        "main_driver": "Sensitivity leg uses tornado_scan() output internally.",
    }

    # --------------------------------------------------------- narrative
    verdict = ("value-creating" if good else "value-destroying")
    narrative = (
        "The %s project in the %s sector was evaluated over a %s-year life at "
        "a %s cost of capital built from Zimbabwe's %s policy rate, escalating "
        "costs at %s inflation. Its %s expected capex, %s annual revenue and %s "
        "annual operating costs produce an NPV of %s and an IRR of %s, so the "
        "project is currently %s (%s). The dominant sensitivity driver is %s, "
        "which is what management should watch and hedge before commitment."
        % (project.project_name, project.sector, int(project.project_life),
           _rate(wacc), _pct(macro.policy_rate_pct()), _pct(inf),
           _usd(capex), _usd(project.annual_revenue), _usd(project.operating_costs),
           _usd(npv), _rate(irr), verdict, decision, main_driver)
    )

    return {"plain": plain, "technical": technical, "narrative": narrative}


def _main_driver(project: ProjectInput, result: Dict[str, Any]) -> str:
    """Cheap deterministic guess at the dominant driver using known structure."""
    try:
        from core.financial_engine import tornado_scan
        df = tornado_scan(project, MacroContext())
        if df is not None and not df.empty:
            return str(df.iloc[0]["variable"])
    except Exception:
        pass
    growth_gap = project.revenue_growth_pct - project.operating_cost_growth_pct
    if growth_gap < 0:
        return ("Operating costs are growing faster than revenue "
                "(cost %s%% vs revenue %s%%)." % (
                    _pct(project.operating_cost_growth_pct),
                    _pct(project.revenue_growth_pct)))
    return "Revenue growth versus operating-cost growth."


# ---------------------------------------------------------------------------
# Graph explanations
# ---------------------------------------------------------------------------

def explain_graph(title: str, data_summary: Dict[str, Any]) -> Dict[str, str]:
    """Return the four AI EXPLAINS THIS cards for one graph type.

    `data_summary` carries whatever the caller knows about the chart, e.g.
      {"max_driver": "Revenue", "delta_npv": -123000.0,
       "p_positive": 0.82, "mean_npv": 50000.0, ...}
    Unknown keys fall back to a neutral, honest sentence.
    """
    t = (title or "").lower()

    # determine the graph kind from the title
    kind = "chart"
    if "tornado" in t:
        kind = "tornado"
    elif "npv profile" in t or "npv profile" in t:
        kind = "npv_profile"
    elif "scenario" in t:
        kind = "scenario"
    elif "sensitivity" in t:
        kind = "sensitivity"
    elif "stress" in t:
        kind = "stress"
    elif "monte carlo" in t or "monte-carlo" in t or "distribution" in t:
        kind = "monte_carlo"
    elif "feature importance" in t or "machine learning" in t or "ml" in t:
        kind = "ml"

    return _EXPLAINERS[kind](data_summary)


def _expl_npv_profile(d: Dict[str, Any]) -> Dict[str, str]:
    r0 = float(d.get("npv_at_wacc", 0.0))
    r5 = float(d.get("npv_at_zero", 0.0))
    irr = float(d.get("irr", 0.0))
    return {
        "what_happened": (
            "The NPV profile plots project value against the discount rate: "
            "at 0%% discount rate NPV is %s and at the %s WACC it is %s."
            % (_usd(r5), _rate(float(d.get("wacc", 0.0))), _usd(r0))
        ),
        "why": "Higher discount rates reduce the worth of future cash flows, so the curve slopes down.",
        "what_it_means": (
            "Where the curve crosses zero (around %s) is the IRR - the "
            "project's break-even return." % _rate(irr)
        ),
        "main_driver": "The discount rate (WACC) drives where the curve sits.",
    }


def _expl_tornado(d: Dict[str, Any]) -> Dict[str, str]:
    main = d.get("main_driver") or d.get("max_driver") or "unknown"
    swing = d.get("max_swing", 0.0)
    return {
        "what_happened": (
            "Each bar shows how NPV moves when one assumption swings down and up "
            "by the tested amount (%s)." % str(d.get("shift_pct", "-20% to +20%"))
        ),
        "why": "One-driver-at-a-time re-runs isolate each assumption's individual impact on NPV.",
        "what_it_means": (
            "%s is the largest risk: it alone moves NPV by about %s."
            % (main, _usd(swing))
        ),
        "main_driver": f"{main}",
    }


def _expl_scenario(d: Dict[str, Any]) -> Dict[str, str]:
    bs = d.get("best_npv", 0.0)
    ws = d.get("worst_npv", 0.0)
    base = d.get("base_npv", 0.0)
    return {
        "what_happened": (
            "Three scenarios were run on the same model: Optimistic NPV %s, "
            "Base NPV %s and Pessimistic NPV %s." % (_usd(bs), _usd(base), _usd(ws))
        ),
        "why": "Each scenario changes inflation, foreign exchange, interest and material-cost assumptions together.",
        "what_it_means": (
            "The spread between best and worst (%s) is the range of plausible "
            "outcomes under different Zimbabwe macro paths." % _usd(abs(bs - ws))
        ),
        "main_driver": "Macro assumptions (inflation, ZiG, rates) differentiate the scenarios.",
    }


def _expl_sensitivity(d: Dict[str, Any]) -> Dict[str, str]:
    main = d.get("main_driver") or d.get("max_driver") or "unknown"
    return {
        "what_happened": (
            "This chart re-runs NPV for each chosen variable across a range and "
            "shows the resulting value spread."
        ),
        "why": "Changing one variable at a time reveals which input the result leans on most.",
        "what_it_means": (
            "%s has the steepest effect - a small change in it moves NPV the most."
            % main
        ),
        "main_driver": f"{main}",
    }


def _expl_stress(d: Dict[str, Any]) -> Dict[str, str]:
    delta = float(d.get("delta_npv", 0.0))
    base = float(d.get("base_npv", 0.0))
    stressed = float(d.get("stress_npv", base + delta))
    return {
        "what_happened": (
            "Under a combined stress shock, NPV moves from %s to %s, a change "
            "of %s." % (_usd(base), _usd(stressed), _usd(delta))
        ),
        "why": "Stress simultaneously raises inflation, weakens the ZiG, lifts interest rates and surges material costs.",
        "what_it_means": (
            "If the project stays positive here it is resilient; if not, it "
            "needs hedges, fixed-price contracts or more contingency."
        ),
        "main_driver": str(d.get("worst_driver", "the dominant stress leg")),
    }


def _expl_monte_carlo(d: Dict[str, Any]) -> Dict[str, str]:
    mean = float(d.get("mean_npv", 0.0))
    p_pos = float(d.get("p_positive", 0.0))
    p5 = float(d.get("p5", 0.0))
    p95 = float(d.get("p95", 0.0))
    return {
        "what_happened": (
            "Simulation of %s plausible futures gives an average NPV of %s, "
            "with %s%% of outcomes positive." % (int(d.get("n", 0)), _usd(mean),
                                                 "{:.0f}".format(p_pos * 100))
        ),
        "why": "Correlated random shocks to inflation, FX, rates, revenue and materials produce a distribution rather than one number.",
        "what_it_means": (
            "You can expect NPV roughly between %s (5th percentile) and %s "
            "(95th percentile)." % (_usd(p5), _usd(p95))
        ),
        "main_driver": str(d.get("risk_class", "the distribution of macro shocks")),
    }


def _expl_ml(d: Dict[str, Any]) -> Dict[str, str]:
    acc = d.get("accuracy", 0.0)
    top = d.get("top_feature") or d.get("main_driver") or "unknown"
    return {
        "what_happened": (
            "A machine-learning model trained on SYNTHETIC demonstration data "
            "was evaluated with %s accuracy." % "{:.0%}".format(acc)
        ),
        "why": "The model learns patterns between project attributes and cost-overrun outcomes.",
        "what_it_means": (
            "%s is the strongest predictor of cost overrun or delay risk."
            % top
        ),
        "main_driver": f"{top}",
    }


def _expl_chart(d: Dict[str, Any]) -> Dict[str, str]:
    return {
        "what_happened": "This chart shows how the selected figures behave across the range tested.",
        "why": "One shared cash-flow table is re-run for each point so every value is consistent.",
        "what_it_means": "Read the shape: steeper curves mean the result is more sensitive to that input.",
        "main_driver": str(d.get("main_driver", "review the inputs shown on the axis")),
    }


_EXPLAINERS = {
    "npv_profile": _expl_npv_profile,
    "tornado": _expl_tornado,
    "scenario": _expl_scenario,
    "sensitivity": _expl_sensitivity,
    "stress": _expl_stress,
    "monte_carlo": _expl_monte_carlo,
    "ml": _expl_ml,
    "chart": _expl_chart,
}


# ---------------------------------------------------------------------------
# Generic card renderer helper (returns markdown-ready strings)
# ---------------------------------------------------------------------------

def card_markdown(explanation: Dict[str, str], title: str = "AI EXPLAINS THIS") -> str:
    """Compose the four-card explanation block as markdown."""
    lines = [
        "#### " + title,
        "**WHAT HAPPENED**",
        explanation.get("what_happened", ""),
        "**WHY**",
        explanation.get("why", ""),
        "**WHAT DOES IT MEAN**",
        explanation.get("what_it_means", explanation.get("what_does_it_mean", "")),
        "**MAIN DRIVER**",
        explanation.get("main_driver", ""),
    ]
    return "\n\n".join(lines)