"""Real-options pricing and Monte-Carlo NPV complement.

These functions live here rather than in financial_engine.py so the core
DCF engine stays single-purpose. They wrap the SAME cash-flow model so any
stochastic or optionality result is always consistent with the base table.

Public API
----------
    monte_carlo_npv(p, macro=None, n=2000, seed=7) -> dict
        Stochastic NPV distribution (mean, std, percentiles, P(NPV>0),
        VaR, CVaR, risk class).

    real_options(p, base_npv, volatility=0.35, option_years=2.0) -> dict
        Expansion + abandonment option value via Black-Scholes-style
        closed forms.

    bs_call(s, k, r, v, t) -> float
    bs_put(s, k, r, v, t) -> float
        European option prices used by real_options.

If a preferred implementation already exists in `core.risk_engine`, we
re-export it so callers never have to know where the math lives.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from core.data_model import MacroContext, ProjectInput

try:
    from core.risk_engine import monte_carlo_npv as _mc_from_risk
    from core.risk_engine import real_options as _ro_from_risk
    HAS_RISK_IMPL = True
except Exception:  # pragma: no cover - fallback
    HAS_RISK_IMPL = False


# ===========================================================================
# Black-Scholes closed forms (kept local so optional_pricing never depends
# on risk_engine's private helpers)
# ===========================================================================

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(s: float, k: float, r: float, v: float, t: float) -> float:
    """Black-Scholes European call price."""
    if t <= 0 or v <= 0 or s <= 0 or k <= 0:
        return 0.0
    d1 = (math.log(s / k) + (r + 0.5 * v * v) * t) / (v * math.sqrt(t))
    d2 = d1 - v * math.sqrt(t)
    return s * _norm_cdf(d1) - k * math.exp(-r * t) * _norm_cdf(d2)


def bs_put(s: float, k: float, r: float, v: float, t: float) -> float:
    """Black-Scholes European put price (via put-call parity)."""
    c = bs_call(s, k, r, v, t)
    return c + k * math.exp(-r * t) - s


# ===========================================================================
# Monte-Carlo NPV
# ===========================================================================

def monte_carlo_npv(
    p: ProjectInput,
    macro: Optional[MacroContext] = None,
    n: int = 2000,
    seed: int = 7,
) -> Dict[str, Any]:
    """Stochastic NPV distribution driven by correlated macro shocks.

    Every draw re-runs the shared cash-flow engine, so results are always
    internally consistent with the deterministic table.
    """
    if HAS_RISK_IMPL:
        try:
            import core.risk_engine as re_mod
            return re_mod.monte_carlo_npv(p, macro, n, seed)
        except Exception:
            pass

    from core.financial_engine import project_cash_flows

    macro = macro or MacroContext()
    rng = np.random.default_rng(seed)

    base = project_cash_flows(p, macro)
    base_npv = float(base["npv"])

    draws = []
    for _ in range(int(n)):
        stress = {}
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

    arr = np.asarray(draws, dtype=float)
    if arr.size == 0:
        arr = np.zeros(1)
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
        "n": int(len(arr)),
        "mean_npv": mean_v,
        "std_npv": std_v,
        "p5": p5,
        "p50": p50,
        "p95": p95,
        "p_positive": p_pos,
        "var_95": var95,
        "cvar_95": cvar95,
        "base_npv": base_npv,
        "risk_class": "HIGH" if p_pos < 0.50 else ("MEDIUM" if p_pos < 0.80 else "LOW"),
    }


# ===========================================================================
# Real options (Black-Scholes style expansion / abandonment)
# ===========================================================================

def real_options(
    p: ProjectInput,
    base_npv: float,
    volatility: float = 0.35,
    option_years: float = 2.0,
) -> Dict[str, Any]:
    """Expansion + abandonment real-option values with clear caveats."""
    if HAS_RISK_IMPL:
        try:
            import core.risk_engine as re_mod
            return re_mod.real_options(p, base_npv, volatility, option_years)
        except Exception:
            pass

    macro = MacroContext()
    r = macro.policy_rate_pct() / 100.0
    rf = max(r, 0.02)
    if volatility <= 0:
        volatility = 0.35

    exp_value = base_npv * 1.5 * bs_call(base_npv, max(base_npv, 1.0), rf, volatility, option_years)
    abandon_value = base_npv * 0.70 * bs_put(base_npv, max(base_npv, 1.0), rf, volatility, option_years)
    combined = base_npv + exp_value + abandon_value

    return {
        "expansion_value": float(exp_value),
        "abandonment_value": float(abandon_value),
        "combined_npv": float(combined),
        "assumptions": (
            "Real-options values are illustrative Black-Scholes approximates. "
            "They assume a %s-year decision window and %s%% annual volatility. "
            "They are a decision guide, not a bankable figure." % (option_years, volatility * 100)
        ),
        "expanded_decision": "ACCEPT" if combined > 0 else "REJECT",
    }


# ===========================================================================
# Convenience helpers used by the UI / report builder
# ===========================================================================

def option_summary_text(opt: Dict[str, Any], base_npv: float) -> str:
    """Plain-English summary of the real-options result."""
    return (
        "Expansion flexibility adds USD %s to the project and abandonment "
        "protection adds USD %s; combined strategic value is USD %s versus "
        "base NPV USD %s."
        % (fmt_usd(opt.get("expansion_value", 0.0)),
           fmt_usd(opt.get("abandonment_value", 0.0)),
           fmt_usd(opt.get("combined_npv", 0.0)),
           fmt_usd(base_npv))
    )


def fmt_usd(v: float) -> str:
    return "${:,.0f}".format(v)