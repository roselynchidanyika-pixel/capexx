"""CapEx AI Agent - full test suite.

Each `test_*` function returns bool (True = PASS). The `__main__` runner
prints PASS/FAIL per test and exits with a non-zero code when anything fails,
so it is CI-friendly. Edge cases (negative revenue, zero capex, zero budget,
empty portfolios, short life) are included explicitly.
"""
from __future__ import annotations

import os
import sys
from typing import Callable, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.data_model as dm                      # noqa: E402
import core.email_delivery as ed                  # noqa: E402
import core.optional_pricing as op                # noqa: E402
import core.risk_engine as risk                   # noqa: E402
import core.ai_explainer as ai                    # noqa: E402
import core.tts_speech as tts                     # noqa: E402
import core.report_builder as rb                  # noqa: E402
import core.portfolio_optimizer as po             # noqa: E402
from core.financial_engine import (               # noqa: E402
    project_cash_flows, tornado_scan, sensitivity_scan, build_scenarios,
    stress_case, sensitivity_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def fixture_macro() -> dm.MacroContext:
    return dm.MacroContext(indicators={
        "inflation_pct": 5.0, "policy_rate_pct": 10.0, "lending_rate_pct": 15.0,
        "zig_per_usd": 13.5, "us_inflation_pct": 2.8,
        "construction_price_index": 100.0,
    })


def fixture_project() -> dm.ProjectInput:
    return dm.ProjectInput(
        project_id="T1", project_name="Fixture Positive Project",
        sector="roads", project_type="infrastructure",
        initial_investment=10_000_000.0, imported_equipment_pct=40.0,
        contingency_pct=10.0, salvage_value_pct=10.0, construction_months=12.0,
        project_life=10.0, annual_revenue=5_000_000.0, revenue_growth_pct=5.0,
        operating_costs=1_000_000.0, operating_cost_growth_pct=5.0,
        working_capital_pct=10.0, terminal_growth_pct=2.0,
        debt_ratio_pct=60.0, debt_interest_pct=12.0, equity_cost_pct=0.0,
        wacc_pct=0.0, reinvestment_rate_pct=5.0, tax_rate_pct=25.0,
        complexity_score=3.0, design_completeness=0.6,
        procurement_delay_days=60.0, num_change_orders=3.0,
        contractor_type="local_large", funding_source="mixed",
        procurement_method="open_competitive", province="Harare",
    )


def fixture_result():
    return project_cash_flows(fixture_project(), fixture_macro())


# ===========================================================================
# Capital budgeting tests
# ===========================================================================
def _npv_at(rate, cf):
    return sum(cf[t] / (1 + rate) ** t for t in range(len(cf)))


def test_npv():
    r = fixture_result()
    npv_val = r["npv"]
    if not npv_val > 0:
        return False
    if abs(npv_val - sum(r["present_values"])) > 0.01:
        return False
    return abs(npv_val - 12652554.65) < 2000.0


def test_irr():
    r = fixture_result()
    irr = r["irr"]
    if irr <= 0:
        return False
    # NPV must be ~0 at the IRR
    return abs(_npv_at(irr, r["cash_flows"])) < 1e-4


def test_mirr():
    r = fixture_result()
    if not (r["mirr"] > 0):
        return False
    if r["mirr"] > r["irr"] + 1e-9:
        return False  # reinvest-rate lower than IRR -> MIRR below IRR
    return True


def test_payback():
    r = fixture_result()
    if not (r["payback"] >= 1.0):
        return False
    if r["discounted_payback"] < r["payback"] - 1e-9:
        return False
    if abs(r["payback"] - 1.0) > 0.01:
        return False
    return True


def test_arr():
    r = fixture_result()
    table = r["cash_flow_table"]
    avg = float(np.mean(table["NOPAT"]))
    from core.financial_engine import expected_capex_for
    capex = r["expected_capex"] + fixture_project().initial_investment * \
        (fixture_project().working_capital_pct / 100.0)
    recomputed = avg / capex if capex else 0.0
    return abs(recomputed - r["arr"]) < 1e-9


def test_pi():
    r = fixture_result()
    if not (r["pi"] > 1.0):
        return False
    return np.isfinite(float(r["pi"]))


def test_dcf():
    r = fixture_result()
    table = r["cash_flow_table"]
    wacc = r["wacc"]
    recomputed = sum(float(row["Operating Cash Flow"]) / (1 + wacc) ** y
                     for y, (_, row) in enumerate(table.iterrows(), start=1))
    if not r["dcf_value"] > 0:
        return False
    return abs(recomputed - r["dcf_value"]) < 1e-6


def test_eaa():
    r = fixture_result()
    wacc = r["wacc"]
    life = int(r["project_life"])
    annuity = wacc * r["npv"] / (1 - (1 + wacc) ** -life)
    return abs(annuity - r["eaa"]) < 1e-6


def test_breakeven():
    r = fixture_result()
    be = r["breakeven_revenue"]
    if not isinstance(be, (int, float)):
        return False
    return 0.0 <= be <= fixture_project().annual_revenue * 3.0


def test_cumulative_flow():
    r = fixture_result()
    table = r["cash_flow_table"]
    last = float(table["Cumulative CF"].iloc[-1])
    return abs(last - r["cumulative_cash_flows"][-1]) < 1e-6


# ===========================================================================
# Scenarios / stress / sensitivity
# ===========================================================================
def test_scenarios():
    p = fixture_project()
    m = fixture_macro()
    df = build_scenarios(p, m)
    if len(df) != 3:
        return False
    needed = {"Scenario", "npv", "irr", "mirr", "payback", "pi"}
    if not needed.issubset(set(df.columns)):
        return False
    row = df.set_index("Scenario")
    base = float(row.loc["Base", "npv"])
    if abs(base - fixture_result()["npv"]) > 2000:
        return False  # Base uses the exact shared model numbers
    if not all(np.isfinite(float(v)) for v in df["npv"]):
        return False
    # verify the macro paths: Optimistic lighter inflation, Pessimistic heavier
    inf_high = project_cash_flows(p, m, {"inflation_pct": +12.0})["inflation_used_pct"]
    inf_low = project_cash_flows(p, m, {"inflation_pct": -8.0})["inflation_used_pct"]
    return inf_low < m.inflation_pct() < inf_high


def test_stress():
    p = fixture_project()
    sc = stress_case(p)
    for key in ("base", "stressed", "deltas", "summary"):
        if key not in sc:
            return False
    if sc["deltas"]["npv"] > 0:
        return False
    return "inflation" in sc["summary"]


def test_sensitivity():
    p = fixture_project()
    df = tornado_scan(p)
    if df.empty:
        return False
    swings = list(df["swing"])
    if swings != sorted(swings, reverse=True):
        return False
    if df["swing"].iloc[0] < df["swing"].iloc[-1]:
        return False
    if not sensitivity_summary(p):
        return False
    return True


def test_monte_carlo():
    mc = risk.monte_carlo_npv(fixture_project(), fixture_macro(), n=150, seed=3)
    if not {"mean_npv", "p5", "p50", "p95", "p_positive", "samples"}.issubset(mc):
        return False
    arr = np.asarray(mc["samples"])
    return len(arr) >= 100 and np.all(np.isfinite(arr))


def test_real_options():
    p = fixture_project()
    ro = risk.real_options(p, 1_000_000.0)
    return (ro["combined_npv"] > 0 and
            "expansion_value" in ro and "abandonment_value" in ro)


def test_bs_parity():
    s, k, rr, v, t = 100.0, 105.0, 0.05, 0.30, 2.0
    c = op.bs_call(s, k, rr, v, t)
    p = op.bs_put(s, k, rr, v, t)
    return abs((c - p) - (s - k * np.exp(-rr * t))) < 1e-9


# ===========================================================================
# Portfolio
# ===========================================================================
def _portfolio_df():
    return pd.DataFrame({
        "project_id": ["A", "B", "C"],
        "sector": ["roads", "energy", "mining"],
        "baseline_capex": [40.0, 25.0, 30.0],
        "expected_npv": [10.0, 8.0, 12.0],
        "risk_score": [0.4, 0.3, 0.6],
        "strategic_score": [1.0, 1.0, 1.0],
    })


def test_portfolio():
    res = po.optimize_portfolio(_portfolio_df(), budget=60.0)
    if res["status"] not in ("optimal", "greedy_fallback"):
        return False
    if res["total_capex"] > 60.0 + 1e-6:
        return False
    if res["n_funded"] + res["n_deferred"] != 3:
        return False
    for item in res["selected_projects"] + res["deferred_projects"]:
        if item["decision"] not in ("FUND", "DEFER"):
            return False
    return True


def test_portfolio_empty():
    res = po.optimize_portfolio(pd.DataFrame(), budget=100.0)
    return res["status"] == "empty" and res["selected_projects"] == []


def test_portfolio_zero_budget():
    # Zero/None budget defaults to the total capex -> all projects fundable.
    res = po.optimize_portfolio(_portfolio_df(), budget=0.0)
    return res["status"] in ("optimal", "greedy_fallback") and res["n_funded"] == 3


def test_budget_sensitivity():
    df = po.budget_sensitivity(_portfolio_df(), [40.0, 60.0, 80.0])
    return len(df) == 3 and {"budget", "total_expected_npv", "n_funded"}.issubset(df.columns)


def test_efficient_frontier():
    ff = po.efficient_frontier(_portfolio_df(), n_points=5, budget=80.0)
    return len(ff) == 5 and "average_risk" in ff.columns


def test_build_project_table():
    from core.portfolio_optimizer import build_project_table
    tbl = build_project_table([fixture_project()], fixture_macro())
    return (len(tbl) == 1 and "baseline_capex" in tbl.columns
            and "expected_npv" in tbl.columns)


# ===========================================================================
# Email / report / tts / explainer
# ===========================================================================
def test_email_validation():
    if not ed.validate_email("thierry@gmail.com"):
        return False
    if ed.validate_email("not-an-email"):
        return False
    if ed.validate_email(""):
        return False
    return True


def test_email_mask():
    return ed.mask_email("thierry@gmail.com") == "t****@gmail.com"


def test_email_no_creds():
    import os
    saved = {k: os.environ.get(k) for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASS")}
    for k in saved:
        os.environ.pop(k, None)
    try:
        res = ed.send_report_email("thierry@gmail.com", "s", "b")
        return res["success"] is False and bool(res["message"])
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_email_invalid_recipient():
    res = ed.send_report_email("bad-address", "s", "b")
    return res["success"] is False and "Invalid recipient" in res["message"]


def test_report_pdf():
    import tempfile
    p = fixture_project()
    r = fixture_result()
    with tempfile.TemporaryDirectory() as td:
        path = rb.generate_pdf_report(p, fixture_macro(), r, out_dir=td)
        if not os.path.exists(path):
            return False
        if os.path.getsize(path) <= 0:
            return False
    return True


def test_report_text():
    txt = rb.generate_text_report(fixture_project(), fixture_macro(), fixture_result())
    return "CAPEX AI AGENT" in txt and "EXECUTIVE SUMMARY" in txt


def test_report_latin1():
    rb.latin1("Revenue grows by 5% -> keep ASCII arrow")
    return rb.latin1("emoji \U0001f600 gone") == "emoji  gone"


def test_tts_html():
    html = tts.speech_button_html("Hello world")
    return "PLAY AUDIO" in html and "speechSynthesis" in html


def test_ai_explainer():
    ex = ai.build_explanation(fixture_project(), fixture_macro(), fixture_result())
    if set(ex) != {"plain", "technical", "narrative"}:
        return False
    p = ex["plain"]
    for k in ("what_happened", "why", "what_does_it_mean", "main_driver"):
        if k not in p:
            return False
    em = ai.explain_metric("NPV", 100.0, 0.0)
    return "what_happened" in em


def test_explain_graph():
    g = ai.explain_graph("Tornado Chart - NPV", {"main_driver": "Revenue",
                                                 "max_swing": 1_000_000.0})
    return g["what_happened"] and g["main_driver"] == "Revenue"


# ===========================================================================
# Edge cases
# ===========================================================================
def test_edge_zero_capex():
    p = fixture_project()
    p.initial_investment = 0.0
    r = project_cash_flows(p, fixture_macro())
    if not np.isfinite(float(r["npv"])):
        return False
    if not np.isfinite(float(r["arr"])):
        return False
    return True


def test_edge_negative_revenue():
    p = fixture_project()
    p.annual_revenue = -1_000_000.0
    r = project_cash_flows(p, fixture_macro())
    if not np.isfinite(float(r["npv"])):
        return False
    return r["decision"] in ("REJECT", "ACCEPT-PROBATION", "ACCEPT")


def test_edge_short_life():
    p = fixture_project()
    p.project_life = 1.0
    r = project_cash_flows(p, fixture_macro())
    return np.isfinite(float(r["eaa"])) and len(r["cash_flow_table"]) == 1


def test_edge_zero_wacc_pi():
    p = fixture_project()
    p.initial_investment = 0.0
    r = project_cash_flows(p, fixture_macro())
    # zero capex -> PI handled without divide-by-zero
    return np.isfinite(float(r["pi"]))


def test_macro_context_methods():
    m = fixture_macro()
    vals = [m.inflation_pct(), m.exchange_rate_zig_per_usd(),
            m.policy_rate_pct(), m.lending_rate_pct(),
            m.construction_price_index(), m.us_inflation_pct()]
    return all(np.isfinite(v) for v in vals)


# ===========================================================================
# GZU demo project (SYNTHETIC/DEMO DATA)
# ===========================================================================
def _load_gzu_demo() -> dm.ProjectInput:
    projects = app._load_demo_projects()
    for p in projects:
        if p.project_id == "GZU-HUB-001":
            return p
    return dm.ProjectInput()


def test_gzu_innovation_hub_demo():
    p = _load_gzu_demo()
    if p.project_id != "GZU-HUB-001":
        return False
    if p.project_name != "GZU Innovation Hub â€” Masvingo Campus":
        return False
    if p.sector != "education" or p.province != "Masvingo":
        return False
    r = project_cash_flows(p, fixture_macro())
    for key in ("npv", "mirr", "payback", "discounted_payback", "pi", "eaa"):
        if not np.isfinite(float(r[key])):
            return False
    if not np.isfinite(float(r["irr"])):
        return False
    if not (-1.0 < float(r["irr"]) < 2.0):
        return False
    if not np.isfinite(float(r["expected_capex"])):
        return False
    return np.isfinite(float(r["arr"]))


# ===========================================================================
# Runner
# ===========================================================================
def run_all() -> int:
    tests: List[Callable[[], bool]] = []
    for name in sorted(globals()):
        if name.startswith("test_"):
            fn = globals()[name]
            if callable(fn):
                tests.append(fn)

    passed = 0
    failed = 0
    results = []
    for fn in tests:
        label = fn.__name__
        try:
            ok = bool(fn())
        except Exception as e:  # noqa: BLE001 - report and keep going
            ok = False
            print("  ! %s raised unexpected %r" % (label, e))
        if ok:
            passed += 1
            results.append(("PASS", label))
        else:
            failed += 1
            results.append(("FAIL", label))

    print("\n" + "=" * 66)
    for status, label in results:
        print("  %-4s  %s" % (status, label))
    print("=" * 66)
    print("TOTAL: %d  PASSED: %d  FAILED: %d" % (len(tests), passed, failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run_all())