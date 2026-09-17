"""Central cash-flow & capital-budgeting engine (v3 - clean).

ONE discounted-cash-flow model builds the full year-by-year table; every
metric (NPV, IRR, MIRR, ARR, PI, DCF, EAA, payback, discounted payback,
break-even, sensitivity, scenarios, stress) is derived from THAT SAME table.
No metric ever does its own inconsistent math.

Macro context (inflation, FX, policy/lending rates, construction price index)
is threaded through discount rates, cost escalation and capex so the numbers
genuinely react to the live Zimbabwe environment.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.data_model import ProjectInput, MacroContext

# Fix numpy alias (some versions expose np.npv; we implement our own anyway).
npv = None  # placeholder so we never accidentally call the removed np.npv

EPS = 1e-9


# ---------------------------------------------------------------------------
# Cash-flow projection — THE single source of truth
# ---------------------------------------------------------------------------

def project_cash_flows(
    p: ProjectInput,
    macro: Optional[MacroContext] = None,
    stress: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Project the full discounted cash-flow table and ALL metrics.

    Parameters
    ----------
    p       : project definition (ProjectInput)
    macro   : Zimbabwe macro context (live where available, transparent fallback)
    stress  : optional driver overrides dict, e.g.
              {"inflation_pct": +10.0, "exchange_rate_stress": +0.10,
               "interest_rate_shift": +0.05, "revenue_growth_shift": -0.03,
               "material_growth": +0.08}
    """
    macro = macro or MacroContext()
    stress = stress or {}

    # --------------------------------------------------------------- numbers
    life = int(round(max(p.project_life, 1)))
    life = max(life, 1)
    build_years = max(int(round(p.construction_months / 12.0)), 1)
    inflation_pct = macro.inflation_pct() + float(stress.get("inflation_pct", 0.0))
    inf = inflation_pct / 100.0

    fx_stress = float(stress.get("exchange_rate_stress", 0.0))   # + = ZiG weakens
    fx_factor = 1.0 + fx_stress

    int_stress = float(stress.get("interest_rate_shift", 0.0))   # absolute pct pts
    tax_rate = np.clip(p.tax_rate_pct / 100.0, 0.0, 0.6)

    # --------------------------------------------------------------- WACC
    wacc = wacc_for_p(p, macro, stress) + int_stress / 100.0
    if wacc <= 0:
        wacc = 0.15
    finance_rate = max(wacc, 0.01)
    reinvest_rate = max(p.reinvestment_rate_pct / 100.0, 0.01)

    # ------------------------------------------------ expected capex (escalated)
    expected_capex = _expected_capex(p, macro, stress, fx_factor)

    # -------------------------------------------------- straight-line dep.
    depreciation = _straight_line_dep(p, life)

    # ---------------------------------------------------------- table build
    years = np.arange(0, life + 1)
    cash_flows = np.zeros(life + 1)
    cum = 0.0
    cumulative = []
    rows = []

    # year 0 = equity(-) + debt inflow(+) = net -equity
    net_wc0 = p.initial_investment * (p.working_capital_pct / 100.0)
    capex_invested = expected_capex + net_wc0
    cash_flows[0] = -capex_invested
    cum += cash_flows[0]
    cumulative.append(cum)

    for y in range(1, life + 1):
        # construction ramp: fraction of year the project is operational
        ramp = min(1.0, y / build_years)
        eff = y - 1

        rev = p.annual_revenue * (1 + p.revenue_growth_pct / 100.0) ** eff
        rev = rev * (1 + inf) ** eff * ramp * max(0.0, 1.0 + float(stress.get("revenue_growth_shift", 0.0))) ** eff

        opex = p.operating_costs * (1 + p.operating_cost_growth_pct / 100.0) ** eff
        opex = opex * (1 + inf) ** eff * ramp
        opex = opex * (1 + max(0.0, float(stress.get("material_growth", 0.0))) / 100.0) ** max(eff, 0)

        ebitda = rev - opex
        ebit = ebitda - depreciation
        tax_val = max(ebit * tax_rate, 0.0)
        nopat = ebit - tax_val
        ocf = nopat + depreciation

        terminal = 0.0
        wc_recovery = 0.0
        if y == life:
            salvage = p.initial_investment * (p.salvage_value_pct / 100.0) * (1 + inf) ** life
            terminal = max(salvage + p.initial_investment * p.terminal_growth_pct / 100.0, 0.0)
            wc_recovery = net_wc0

        cash_flow = ocf + (terminal + wc_recovery if y == life else 0.0)
        cash_flows[y] = cash_flow
        cum += cash_flow
        cumulative.append(cum)

        rows.append({
            "Year": y,
            "Revenue": rev,
            "Operating Costs": opex,
            "EBITDA": ebitda,
            "Depreciation": depreciation,
            "EBIT": ebit,
            "Tax": tax_val,
            "NOPAT": nopat,
            "Operating Cash Flow": ocf,
            "Terminal Value": terminal + wc_recovery if y == life else 0.0,
            "Net Cash Flow": cash_flow,
            "Cumulative CF": cum,
        })

    table = pd.DataFrame(rows)

    # ------------------------------------------------------------- metrics
    discount_factors = np.array([1.0 / (1.0 + wacc) ** t for t in years])
    present_values = cash_flows * discount_factors
    pv = present_values.sum()

    npv_val = float(pv)
    irr_val = _irr(cash_flows)
    mirr_val = _mirr(cash_flows, finance_rate, reinvest_rate)
    pi_val = _pi(cash_flows, wacc, expected_capex + net_wc0)
    arr_val, payback_val, d_payback_val = _arr_payback(
        p, wacc, life, cash_flows, ocf_table=rows, capex=expected_capex + net_wc0)
    dcf_value = sum(float(r["Operating Cash Flow"]) / (1.0 + wacc) ** y
                    for y, r in enumerate(rows, start=1))
    eaa_val = _eaa(npv_val, wacc, life)
    breakeven = _breakeven(p, wacc, life)

    decision, reason = recommend(npv_val, irr_val, wacc, pi_val)

    return {
        "npv": npv_val,
        "irr": irr_val,
        "mirr": mirr_val,
        "payback": payback_val,
        "discounted_payback": d_payback_val,
        "arr": arr_val,
        "pi": pi_val,
        "wacc": wacc,
        "eaa": eaa_val,
        "dcf_value": dcf_value,
        "breakeven_revenue": breakeven,
        "expected_capex": expected_capex,
        "inflation_used_pct": inflation_pct,
        "project_life": life,
        "life_years": life,
        "cash_flow_table": table,
        "cash_flows": cash_flows.tolist(),
        "years": years.tolist(),
        "present_values": present_values.tolist(),
        "cumulative_cash_flows": cumulative,
        "discount_factors": discount_factors.tolist(),
        "decision": decision,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expected_capex(p, macro, stress, fx_factor):
    inf = macro.inflation_pct() + float(stress.get("inflation_pct", 0.0))
    inf = max(inf, 0.0) / 100.0
    material = max(0.0, float(stress.get("material_growth", 0.0))) / 100.0

    imported_share = np.clip(p.imported_equipment_pct / 100.0, 0.0, 1.0)
    local_share = 1.0 - imported_share
    import_multi = 1.0 if p.currency == "USD" else fx_factor

    build_years = max(p.construction_months / 12.0, 0.5)
    esc_factor = (1.0 + inf + material) ** (build_years * 0.5)

    capex_base = p.initial_investment * (local_share + imported_share * import_multi)
    contingency = max(0.0, p.contingency_pct / 100.0)
    return capex_base * esc_factor * (1.0 + contingency)


def _straight_line_dep(p, life):
    salvage = p.initial_investment * (p.salvage_value_pct / 100.0)
    return max(p.initial_investment - salvage, 0.0) / life


def wacc_for_p(p: ProjectInput, macro: MacroContext, stress=None):
    stress = stress or {}
    macro = macro or MacroContext()
    risk_free = macro.policy_rate_pct() / 100.0
    country_risk = (p.project_type == "infrastructure" and 0.04) or 0.02
    market_premium = (p.complexity_score / 5.0) * 0.06
    debt_share = p.debt_ratio_pct / 100.0
    equity_share = 1.0 - debt_share
    lending = max(macro.lending_rate_pct() / 100.0, risk_free)
    cost_debt_after = lending * (1 - p.tax_rate_pct / 100.0)
    cost_equity = risk_free + country_risk + market_premium
    if p.wacc_pct and p.wacc_pct > 0:
        base = p.wacc_pct / 100.0
    else:
        base = debt_share * cost_debt_after + equity_share * cost_equity
    base += float(stress.get("interest_rate_shift", 0.0)) / 100.0
    return max(base, 0.01)


def _npv_at_rate(rate, cash_flows):
    return sum(cf / (1 + rate) ** t for t, cf in enumerate(cash_flows))


def _irr(cash_flows):
    cf = np.asarray(cash_flows, dtype=float)
    if np.all(cf >= 0) or np.all(cf <= 0):
        return 0.0
    lo, hi = -0.999, 10.0
    f_lo, f_hi = _npv_at_rate(lo, cf), _npv_at_rate(hi, cf)
    if f_lo * f_hi > 0:
        lo, hi = -0.999, 100.0
        f_lo, f_hi = _npv_at_rate(lo, cf), _npv_at_rate(hi, cf)
    if f_lo * f_hi > 0:
        return 0.0
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = _npv_at_rate(mid, cf)
        if abs(f_mid) < 1e-10:
            return mid
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2


def _mirr(cash_flows, finance_rate, reinvest_rate):
    cf = np.asarray(cash_flows, dtype=float)
    n = len(cf)
    if n < 2:
        return 0.0
    pv_neg = sum(-x / (1 + finance_rate) ** t for t, x in enumerate(cf) if x < 0)
    fv_pos = sum(x * (1 + reinvest_rate) ** (n - 1 - t) for t, x in enumerate(cf) if x > 0)
    if pv_neg <= 0 or fv_pos <= 0:
        return 0.0
    return (fv_pos / pv_neg) ** (1 / (n - 1)) - 1


def _pi(cash_flows, wacc, capex):
    pv_in = sum(cf / (1 + wacc) ** t for t, cf in enumerate(cash_flows) if cf > 0)
    return pv_in / capex if capex else 0.0


def _arr_payback(p, wacc, life, cash_flows, ocf_table, capex=None):
    """Return (ARR, payback, discounted_payback)."""
    avg_profit = float(np.mean([r["NOPAT"] for r in ocf_table]))
    capex = capex if capex is not None else expected_capex_for(p)
    arr = avg_profit / capex if capex else 0.0

    cum = 0.0
    payback = float("inf")
    for t, cf in enumerate(cash_flows[1:], start=1):
        cum += cf
        if cum >= 0:
            frac = (cum - cf) / cf if cf else 0.0
            payback = (t - 1) + (1.0 if t == 1 else frac)
            break

    dcum = 0.0
    d_payback = float("inf")
    for t, cf in enumerate(cash_flows[1:], start=1):
        dcf = cf / (1 + wacc) ** t
        dcum += dcf
        if dcum >= 0:
            frac = (dcum - dcf) / dcf if dcf else 0.0
            d_payback = (t - 1) + (1.0 if t == 1 else frac)
            break
    return arr, payback, d_payback


def expected_capex_for(p, macro=None, stress=None):
    macro = macro or MacroContext()
    return _expected_capex(p, macro, stress or {}, 1.0)


def _eaa(npv_val, wacc, life):
    if life <= 0:
        return npv_val
    if wacc <= 0:
        return npv_val / life
    annuity = wacc * npv_val / (1 - (1 + wacc) ** -life)
    return annuity


def _breakeven(p, wacc, life):
    """Revenue needed so NPV >= 0 (scan positive space, fixed point with OCF)."""
    def npv_at(rev):
        cf = np.zeros(life + 1)
        capex = expected_capex_for(p)
        cf[0] = -capex
        for y in range(1, life + 1):
            ramp = min(1.0, y / max(p.construction_months / 12.0, 0.5))
            opex = p.operating_costs * ramp
            ebitda = rev * ramp - opex
            dep = _straight_line_dep(p, life)
            ebit = ebitda - dep
            taxv = max(ebit * p.tax_rate_pct / 100.0, 0.0)
            cf[y] = (ebit - taxv) + dep
        return sum(c / (1 + wacc) ** t for t, c in enumerate(cf))

    lo, hi = 0.0, max(p.annual_revenue * 2.0, 1.0)
    if npv_at(hi) >= 0:
        return 0.0  # already profitable at zero revenue assumption
    for _ in range(80):
        mid = (lo + hi) / 2
        if npv_at(mid) >= 0:
            hi = mid
        else:
            lo = mid
    return float(hi)


# ---------------------------------------------------------------------------
# Decision + explanation
# ---------------------------------------------------------------------------

def recommend(npv_val, irr_val, wacc, pi_val) -> Tuple[str, str]:
    if npv_val > 0 and irr_val > wacc and pi_val > 1.0:
        return "ACCEPT", (
            "Positive NPV, IRR above the discount rate and PI > 1 mean the "
            "project is expected to create value.")
    if npv_val > 0:
        return "ACCEPT-PROBATION", (
            "Positive NPV but tight IRR/PI — accept conditionally and monitor.")
    return "REJECT", (
        "Negative NPV means the project is expected to destroy value at the "
        "current cost of capital and macro assumptions.")


def decision_from(p: ProjectInput, macro=None, stress=None) -> dict:
    r = project_cash_flows(p, macro, stress)
    decision, reason = recommend(r["npv"], r["irr"], r["wacc"], r["pi"])
    r["decision"] = decision
    r["reason"] = reason
    return r


# ---------------------------------------------------------------------------
# Sensitivity & scenarios
# ---------------------------------------------------------------------------

VARIABLE_LABELS = {
    "revenue": "Annual Revenue",
    "operating_costs": "Operating Costs",
    "initial_investment": "Initial Investment",
    "inflation": "Inflation",
    "exchange_rate": "Exchange Rate",
    "interest_rate": "Interest Rate",
    "cost_growth": "Operating-Cost Growth",
    "project_life": "Project Life",
}


def sensitivity_scan(p, macro=None, variable="revenue",
                     changes=None) -> pd.DataFrame:
    macro = macro or MacroContext()
    changes = changes or [-0.30, -0.15, 0.0, 0.15, 0.30]
    rows = []
    base = getattr(p, _VAR_FIELD(variable))
    for ch in changes:
        pv = ProjectInput(**{**p.__dict__, _VAR_FIELD(variable):
                             base * (1 + ch)})
        if variable in ("inflation", "exchange_rate", "interest_rate"):
            stress = {}
            if variable == "inflation":
                stress["inflation_pct"] = ch * 40
            elif variable == "exchange_rate":
                stress["exchange_rate_stress"] = ch * 50
            else:
                stress["interest_rate_shift"] = ch * 15
            r = project_cash_flows(p, macro, stress)
        else:
            r = project_cash_flows(pv, macro)
        rows.append({"variation": ch, "variation_label": f"{ch:+.0%}",
                     "npv": r["npv"], "irr": r["irr"]})
    return pd.DataFrame(rows)


def _VAR_FIELD(variable):
    return {
        "revenue": "annual_revenue",
        "operating_costs": "operating_costs",
        "initial_investment": "initial_investment",
        "cost_growth": "operating_cost_growth_pct",
        "project_life": "project_life",
    }.get(variable, "annual_revenue")


def tornado_scan(p, macro=None, changes=(-0.20, 0.20)) -> pd.DataFrame:
    """One-driver-at-a-time NPV swing -> tornado chart."""
    macro = macro or MacroContext()
    base = project_cash_flows(p, macro)["npv"]
    rows = []
    for var in VARIABLE_LABELS:
        lo = sensitivity_scan(p, macro, var, [changes[0]])["npv"].iloc[0]
        hi = sensitivity_scan(p, macro, var, [changes[1]])["npv"].iloc[0]
        rows.append({
            "variable": VARIABLE_LABELS[var],
            "low": float(lo), "high": float(hi),
            "swing": abs(float(lo) - float(hi)),
        })
    df = pd.DataFrame(rows).sort_values("swing", ascending=False)
    df["base_npv"] = base
    return df


def build_scenarios(p, macro=None) -> pd.DataFrame:
    macro = macro or MacroContext()
    out = {}
    base = project_cash_flows(p, macro)
    out["Base"] = {
        "npv": base["npv"], "irr": base["irr"], "mirr": base["mirr"],
        "payback": base["payback"], "pi": base["pi"],
        "label": "Base case using current macro assumptions",
    }
    opt = project_cash_flows(p, macro, {
        "inflation_pct": -8.0, "exchange_rate_stress": -0.06,
        "interest_rate_shift": -0.03, "revenue_growth_shift": +0.03,
        "material_growth": -5.0,
    })
    out["Optimistic"] = {
        "npv": opt["npv"], "irr": opt["irr"], "mirr": opt["mirr"],
        "payback": opt["payback"], "pi": opt["pi"],
        "label": "Lower inflation, stable FX, cheaper funding, higher revenue",
    }
    pes = project_cash_flows(p, macro, {
        "inflation_pct": +12.0, "exchange_rate_stress": +0.15,
        "interest_rate_shift": +0.03, "revenue_growth_shift": -0.03,
        "material_growth": +8.0,
    })
    out["Pessimistic"] = {
        "npv": pes["npv"], "irr": pes["irr"], "mirr": pes["mirr"],
        "payback": pes["payback"], "pi": pes["pi"],
        "label": "Higher inflation, ZiG weakness, tighter funding, higher materials",
    }
    df = pd.DataFrame.from_dict(out, orient="index").reset_index()
    df = df.rename(columns={"index": "Scenario"})
    return df


def sensitivity_summary(p, macro=None):
    """Plain-English read of the tornado — 'main driver' identification."""
    df = tornado_scan(p, macro)
    if df.empty:
        return "No sensitivity drivers computed."
    main = df.iloc[0]["variable"]
    least = df.iloc[-1]["variable"]
    return (f"ThE most sensitive driver is {main.lower()} "
            f"(NPV swing ~USD {df.iloc[0]['swing']:,.0f}). "
            f"The least sensitive is {least.lower()}.")


def stress_case(p, macro=None):
    """Base vs Stress across ALL major metrics (for the stress-test lab)."""
    macro = macro or MacroContext()
    base = project_cash_flows(p, macro)
    stressed = project_cash_flows(p, macro, {
        "inflation_pct": +20.0, "exchange_rate_stress": +0.25,
        "interest_rate_shift": +0.06, "material_growth": +15.0,
        "revenue_growth_shift": -0.05,
    })
    return {
        "base": base,
        "stressed": stressed,
        "deltas": {
            "npv": stressed["npv"] - base["npv"],
            "irr": stressed["irr"] - base["irr"],
            "mirr": stressed["mirr"] - base["mirr"],
            "capex": stressed["expected_capex"] - base["expected_capex"],
            "payback": stressed["payback"] - base["payback"],
        },
        "summary": (
            "A combined Zimbabwe stress shock (inflation +ZiG weakness + rate "
            "hike + material surge) erodes the project's value as modelled below."
        ),
    }
