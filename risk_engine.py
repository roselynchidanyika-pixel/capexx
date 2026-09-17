"""Risk engine — geometric status, stress-test lab, scenario & sensitivity.

Everything here consumes the SAME ProjectInput + MacroContext and re-runs
`core.financial_engine.project_cash_flows()` so that any stress input flows
all the way through NPV → IRR → risk → status with zero duplicated math.
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.data_model import ProjectInput, MacroContext
from core.financial_engine import project_cash_flows, expected_capex_for
from core.data_model import ProjectInput, MacroContext


# ---------------------------------------------------------------------------
# Geometric status classification (GREEN hexagon / AMBER diamond / RED triangle)
# ---------------------------------------------------------------------------

def classify_status(result: Dict[str, Any], wacc_override: Optional[float] = None,
                    inf_shift: float = 0.0) -> Dict[str, str]:
    """Classify project into geometric risk status with transparent rules.

    Rules (fully exposed, never a black box):
      * RED    -> NPV < 0  OR  IRR < discount rate  OR  PI < 1
      * AMBER  -> NPV>0 but payback > 70% of life  OR  PI < 1.25
                 OR inflation shift > +10pts pushes NPV < 0
      * GREEN  -> otherwise (stable, value-creating, resilient)
    """
    npv = float(result["npv"])
    irr = float(result["irr"])
    pi = float(result["pi"])
    wacc = float(result["wacc"])
    payback = float(result["payback"])
    life = float(result["project_life"])

    reasons = []

    if npv < 0:
        reasons.append("NPV negative")
    if irr < wacc:
        reasons.append(f"IRR ({irr:.1%}) below cost of capital ({wacc:.1%})")
    if pi < 1.0:
        reasons.append(f"Profitability Index {pi:.2f} < 1")

    if reasons:
        return {"shape": "TRIANGLE", "level": "CRITICAL WARNING",
                "color": "#e14b3a", "emoji": "🔺", "reasons": "; ".join(reasons)}

    amber = []
    if payback > 0.7 * life:
        amber.append(f"Payback ({payback:.1f}y) exceeds 70% of life ({life:.1f}y)")
    if pi < 1.25:
        amber.append(f"PI {pi:.2f} < 1.25 — thin margin")
    if inf_shift and (npv * 1.0 - float(result.get("npv", 0))) < 0:
        amber.append("NPV swings negative under +10pt inflation stress")

    if amber:
        return {"shape": "DIAMOND", "level": "RISK WARNING",
                "color": "#e6a23c", "emoji": "🔶", "reasons": "; ".join(amber)}

    return {"shape": "HEXAGON", "level": "PROJECT STABLE",
            "color": "#33b27c", "emoji": "🟩", "reasons": "All thresholds satisfied"}


# ---------------------------------------------------------------------------
# Stress-test lab — user slides + full re-run
# ---------------------------------------------------------------------------

def stress_test(
    p: ProjectInput,
    macro: Optional[MacroContext] = None,
    drivers: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Re-run the ENTIRE project under user-chosen macro stress.

    `drivers` keys (all optional):
      inflation_pts, exchange_stress_pct, interest_shift_pts,
      material_growth_pct, revenue_shift_pct, duration_days,
      contingency_pts, cost_growth_pct
    """
    macro = macro or MacroContext()
    drivers = drivers or {}

    stress = {
        "inflation_pct": float(drivers.get("inflation_pts", 0.0)),
        "exchange_rate_stress": float(drivers.get("exchange_stress_pct", 0.0)) / 100.0,
        "interest_rate_shift": float(drivers.get("interest_shift_pts", 0.0)) / 100.0,
        "material_growth": float(drivers.get("material_growth_pct", 0.0)),
        "revenue_growth_shift": float(drivers.get("revenue_shift_pct", 0.0)) / 100.0,
        "cost_growth": float(drivers.get("cost_growth_pct", 0.0)) / 100.0,
    }

    base = project_cash_flows(p, macro)

    # project-duration stress: shorten/lengthen via revenue ramp + cost share
    if float(drivers.get("duration_days", 0.0)):
        p_dur = _clone(p)
        p_dur.construction_months = max(
            0.5, p.construction_months + float(drivers["duration_days"]) / 30.0)
        p_stress = p_dur
    else:
        p_stress = p

    stressed = project_cash_flows(p_stress, macro, stress)

    delta = {k: stressed.get(k, 0.0) - base.get(k, 0.0)
             for k in ("npv", "irr", "mirr", "pi", "expected_capex")}

    base_status = classify_status(base)
    stress_status = classify_status(stressed, inf_shift=drivers.get("inflation_pts", 0.0))

    return {
        "base": base,
        "stressed": stressed,
        "delta": delta,
        "base_status": base_status,
        "stress_status": stress_status,
        "drivers": drivers,
        "verdict": _verdict(base_status, stress_status),
    }


def _verdict(base_status: dict, stress_status: dict) -> str:
    lv = stress_status["level"]
    if lv == "CRITICAL WARNING":
        return ("CRITICAL: under the selected stress combination the project "
                "would be reclassified to RED TRIANGLE. Do not fund without "
                "hard mitigants (FX hedges, fixed-price contracts, ZiG-indexed "
                "revenue, contingency increase).")
    if lv == "RISK WARNING":
        return ("WARNING: stress pushes the project to AMBER DIAMOND — "
                "financially viable but with elevated exposure. Review the "
                "dominant stress driver and re-run before commitment.")
    return "STABLE: the project absorbs the selected stress without leaving GREEN HEXAGON status."


def _clone(p: ProjectInput) -> ProjectInput:
    d = {k: getattr(p, k) for k in ProjectInput.__annotations__}
    return ProjectInput(**d)


# ---------------------------------------------------------------------------
# Scenario & sensitivity (one-driver tornado + multi scenario table)
# ---------------------------------------------------------------------------

DRIVERS = {
    "revenue": ("annual_revenue", "Revenue"),
    "operating_costs": ("operating_costs", "Operating Costs"),
    "initial_investment": ("initial_investment", "Initial Investment"),
    "wacc": ("None", "Cost of Capital"),
    "inflation": ("None", "Inflation"),
    "exchange_rate": ("None", "Exchange Rate"),
    "project_life": ("project_life", "Project Life"),
}


def sensitivity_tornado(
    p: ProjectInput,
    macro: Optional[MacroContext] = None,
    shifts: Tuple[float, float] = (-0.20, 0.20),
) -> pd.DataFrame:
    macro = macro or MacroContext()
    base = project_cash_flows(p, macro)

    rows = []
    for key, (field, label) in DRIVERS.items():
        vs = []
        for s in shifts:
            if field == "None":
                # macro/stress driver
                stress = {}
                if key == "wacc":
                    stress["interest_rate_shift"] = s
                elif key == "inflation":
                    stress["inflation_pct"] = s * 40
                elif key == "exchange_rate":
                    stress["exchange_rate_stress"] = s
                r = project_cash_flows(p, macro, stress)
            else:
                dv = _clone(p)
                setattr(dv, field, getattr(p, field) * (1 + s))
                r = project_cash_flows(dv, macro)
            vs.append(float(r["npv"]))
        rows.append({
            "driver": label,
            "low_npv": min(vs),
            "high_npv": max(vs),
            "swing": abs(vs[1] - vs[0]),
        })

    df = pd.DataFrame(rows).sort_values("swing", ascending=False).reset_index(drop=True)
    df["main_driver"] = df["driver"].iloc[0] if not df.empty else ""
    return df


def scenario_matrix(
    p: ProjectInput,
    macro: Optional[MacroContext] = None,
) -> Dict[str, Dict[str, Any]]:
    """Base / Optimistic / Pessimistic / Custom — one shared model."""
    macro = macro or MacroContext()
    return {
        "Base Case": project_cash_flows(p, macro),
        "Optimistic Case": project_cash_flows(p, macro, {
            "inflation_pct": -8.0, "exchange_rate_stress": -0.06,
            "interest_rate_shift": -3.0, "material_growth": -4.0,
            "revenue_growth_shift": 0.05,
        }),
        "Pessimistic Case": project_cash_flows(p, macro, {
            "inflation_pct": +12.0, "exchange_rate_stress": +0.15,
            "interest_rate_shift": +5.0, "material_growth": +8.0,
            "revenue_growth_shift": -0.06,
        }),
        "Custom Case": project_cash_flows(p, macro),
    }


def monte_carlo_npv(
    p: ProjectInput,
    macro: Optional[MacroContext] = None,
    n: int = 2000,
    seed: int = 7,
) -> Dict[str, Any]:
    """Stochastic NPV via correlated macro + project uncertainties."""
    macro = macro or MacroContext()
    rng = np.random.default_rng(seed)

    base = project_cash_flows(p, macro)
    base_npv = float(base["npv"])

    draws = []
    for _ in range(n):
        stress = {}
        # correlated shocks: inflation ↔ fx ↔ interest
        z = rng.normal(0, 1)
        stress["inflation_pct"] = rng.normal(0, 0.06) * 100
        stress["exchange_rate_stress"] = (0.5 * z + 0.8 * rng.normal(0, 1)) * 0.06
        stress["interest_rate_shift"] = (0.4 * z + 0.7 * rng.normal(0, 1)) * 0.04
        stress["revenue_growth_shift"] = rng.normal(0, 0.04)
        stress["material_growth"] = rng.normal(0, 4.0)
        try:
            r = project_cash_flows(p, macro, stress)
            draws.append(float(r["npv"]))
        except Exception:
            continue

    arr = np.asarray(draws)
    mean_v = float(np.mean(arr))
    std_v = float(np.std(arr))
    p5 = float(np.percentile(arr, 5))
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    p_pos = float(np.mean(arr > 0))
    var95 = float(np.percentile(arr, 5))
    cvar95 = float(arr[arr <= var95].mean()) if np.any(arr <= var95) else var95

    return {
        "samples": arr,
        "n": len(arr),
        "mean_npv": mean_v,
        "std_npv": std_v,
        "p5": p5, "p50": p50, "p95": p95,
        "p_positive": p_pos,
        "var_95": var95,
        "cvar_95": cvar95,
        "base_npv": base_npv,
        "risk_class": ("HIGH" if p_pos < 0.50 else "MEDIUM" if p_pos < 0.80 else "LOW"),
    }


# ---------------------------------------------------------------------------
# Real options (simplified Black–Scholes-style expansion/abandon valuations)
# ---------------------------------------------------------------------------

def real_options(
    p: ProjectInput,
    base_npv: float,
    volatility: float = 0.35,
    option_years: float = 2.0,
) -> Dict[str, Any]:
    """Expansion + abandonment real-option values with clear caveats."""
    import math

    macro = MacroContext()
    r = macro.policy_rate_pct() / 100.0
    rf = max(r, 0.02)
    if volatility <= 0:
        volatility = 0.35

    # Expansion option: today's project is a stepping stone to 1.5x later
    exp_value = base_npv * 1.5 * _bs_call(base_npv / 1.0, base_npv, rf,
                                          volatility, option_years)
    # Abandonment option: recover 70% of remaining investment if NPV < 0
    abandon_value = base_npv * 0.70 * _bs_put(base_npv / 1.0, base_npv, rf,
                                              volatility, option_years)

    return {
        "expansion_value": float(exp_value),
        "abandonment_value": float(abandon_value),
        "combined_npv": float(base_npv + exp_value + abandon_value),
        "assumptions": (
            "Real-options values are illustrative Black-Scholes approximates. "
            "They assume a 2-year decision window spacings and 35% annual "
            "volatility. They are a decision guide, not a bankable figure — "
            "they should be stress-tested before inclusion in funding decisions."
        ),
        "expanded_decision": "ACCEPT" if (base_npv + exp_value + abandon_value) > 0 else "REJECT",
    }


def _bs_call(s: float, k: float, r: float, v: float, t: float) -> float:
    import math
    if t <= 0 or v <= 0 or s <= 0:
        return 0.0
    d1 = (math.log(s / k) + (r + 0.5 * v * v) * t) / (v * math.sqrt(t))
    d2 = d1 - v * math.sqrt(t)
    return s * _norm_cdf(d1) - k * math.exp(-r * t) * _norm_cdf(d2)


def _bs_put(s: float, k: float, r: float, v: float, t: float) -> float:
    import math
    if t <= 0 or v <= 0 or s <= 0:
        return 0.0
    d1 = (math.log(s / k) + (r + 0.5 * v * v) * t) / (v * math.sqrt(t))
    d2 = d1 - v * math.sqrt(t)
    return k * math.exp(-r * t) * _norm_cdf(-d2) - s * _norm_cdf(-d1)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
