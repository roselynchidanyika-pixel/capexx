"""Portfolio optimisation (MILP integer programming) for capital selection.

Wraps the core PuLP MILP solver from `core.portfolio_engine` with a richer,
UI-friendly API and a deterministic greedy fallback when PuLP is missing.

Public API
----------
    optimize_portfolio(projects, budget, ...) -> dict
    budget_sensitivity(projects, budget_range) -> pd.DataFrame
    efficient_frontier(projects, n_points) -> pd.DataFrame
    build_project_table(projects) -> pd.DataFrame   # ProjectInput[] -> table

Accepted input columns (any of these aliases):
    project_id, sector,
    baseline_capex | capex, expected_npv | npv_usd,
    risk_score | overrun_probability | overrun_risk (0-1),
    strategic_score (optional, defaults 1.0)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.data_model import ProjectInput, MacroContext

try:
    from core.portfolio_engine import (  # noqa: F401
        optimize_portfolio as _engine_optimize,
        budget_sensitivity as _engine_budget_sensitivity,
        efficient_frontier as _engine_efficient_frontier,
    )
    HAS_ENGINE = True
except Exception:  # pragma: no cover
    HAS_ENGINE = False

try:
    from pulp import (  # noqa: F401
        LpMaximize, LpProblem, LpVariable, LpBinary, lpSum, LpStatus, value,
    )
    HAS_PULP = True
except ImportError:
    HAS_PULP = False


# ---------------------------------------------------------------------------
# Build the project table from ProjectInput objects
# ---------------------------------------------------------------------------

def build_project_table(
    projects: List[ProjectInput],
    macro: Optional[MacroContext] = None,
) -> pd.DataFrame:
    """Run every project through the shared engines and return an optimisable table."""
    macro = macro or MacroContext()
    rows = []
    for p in projects:
        try:
            from core.financial_engine import project_cash_flows
            from core.risk_engine import classify_status

            r = project_cash_flows(p, macro)
            npv = float(r["npv"])
            capex = float(r.get("expected_capex", p.initial_investment))
            status = classify_status(r)
            risk = 1.0 if status["shape"] == "TRIANGLE" else 0.5 if status["shape"] == "DIAMOND" else 0.25
            strategic = 1.0
            rows.append({
                "project_id": p.project_id,
                "sector": p.sector,
                "baseline_capex": capex,
                "expected_npv": npv,
                "risk_score": risk,
                "strategic_score": strategic,
                "project_name": p.project_name,
                "irr": float(r.get("irr", 0.0)),
                "payback": float(r.get("payback", float("inf"))),
            })
        except Exception as e:
            rows.append({
                "project_id": p.project_id,
                "sector": p.sector,
                "baseline_capex": float(p.initial_investment),
                "expected_npv": 0.0,
                "risk_score": 0.5,
                "strategic_score": 1.0,
                "project_name": p.project_name,
                "irr": 0.0,
                "payback": float("inf"),
                "error": str(e),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Optimise
# ---------------------------------------------------------------------------

def optimize_portfolio(
    projects: pd.DataFrame,
    budget: float,
    max_risk_exposure: Optional[float] = None,
    min_strategic: Optional[float] = None,
    sector_quotas: Optional[Dict[str, float]] = None,
    strategic_weights: Optional[Dict[str, float]] = None,
    budget_override: Optional[float] = None,
) -> Dict[str, Any]:
    """Select the capital budget's highest-value portfolio.

    Parameters mirror the spec signature plus backward-compatible 'budget'.
    Returns {status, selected_projects, deferred_projects, total_capex,
    total_expected_npv, budget_utilization_pct, n_funded, n_deferred,
    average_risk, ...}.
    """
    if budget_override is not None:
        budget = float(budget_override)
    if budget is None:
        budget = 0.0

    if projects is None or len(projects) == 0:
        return {
            "status": "empty",
            "selected_projects": [],
            "deferred_projects": [],
            "total_capex": 0.0,
            "total_expected_npv": 0.0,
            "budget_utilization_pct": 0.0,
            "n_funded": 0,
            "n_deferred": 0,
            "average_risk": 0.0,
            "notes": "No projects supplied.",
        }

    df = _normalise(projects)
    n = len(df)

    capex = df["baseline_capex"].values.astype(float)
    npvs = df["expected_npv"].values.astype(float)
    risks = df["risk_score"].values.astype(float)
    sectors = df["sector"].values.astype(str)
    strategic = df["strategic_score"].values.astype(float)

    budget = float(budget or 0.0)
    if budget <= 0:
        budget = float(capex.sum())

    min_s = float(min_strategic or 0.0)
    sec_caps = {k: float(v) for k, v in (sector_quotas or {}).items()}
    strat_mult = {k: float(v) for k, v in (strategic_weights or {}).items()}

    # ------------------------------------------------------------------ PuLP
    if HAS_PULP:
        try:
            return _solve_milp(df, capex, npvs, risks, sectors, strategic,
                               budget, max_risk_exposure, min_s, sec_caps, strat_mult)
        except Exception:
            pass

    # ---------------------------------------------- deterministic greedy
    return _solve_greedy(df, capex, npvs, risks, sectors, strategic, budget)


def _solve_milp(df, capex, npvs, risks, sectors, strategic, budget,
                max_risk_exposure, min_s, sec_caps, strat_mult):
    n = len(df)
    prob = LpProblem("CapExPortfolio", LpMaximize)
    x = [LpVariable(f"x_{i}", cat=LpBinary) for i in range(n)]

    driver = [strat_mult.get(sectors[i], 1.0) * float(npvs[i]) for i in range(n)]
    prob += lpSum(driver[i] * x[i] for i in range(n))

    prob += lpSum(capex[i] * x[i] for i in range(n)) <= budget

    if max_risk_exposure is not None and float(max_risk_exposure) > 0:
        prob += lpSum(float(risks[i]) * float(capex[i]) * x[i]
                      for i in range(n)) <= float(max_risk_exposure) * budget

    if min_s > 0:
        prob += lpSum(float(strategic[i]) * x[i] for i in range(n)) >= min_s * n

    for sec in sorted(set(sectors)):
        idx = [i for i in range(n) if sectors[i] == sec]
        cap = sec_caps.get(sec, 1.0)
        if cap < 1.0 and idx:
            prob += lpSum(capex[i] * x[i] for i in idx) <= cap * budget

    status = prob.solve()
    if LpStatus[status] != "Optimal":
        return _solve_greedy(df, capex, npvs, risks, sectors, strategic, budget)

    selected, deferred = [], []
    for i in range(n):
        chosen = value(x[i]) and value(x[i]) > 0.5
        row = {
            "project_id": str(df.get("project_id", f"P{i}").iloc[i]),
            "sector": str(sectors[i]),
            "capex": float(capex[i]),
            "expected_npv": float(npvs[i]),
            "risk_score": float(risks[i]),
            "strategic_score": float(strategic[i]),
            "decision": "FUND" if chosen else "DEFER",
        }
        (selected if chosen else deferred).append(row)

    total_capex = sum(p["capex"] for p in selected)
    total_npv = sum(p["expected_npv"] for p in selected)
    return {
        "status": "optimal",
        "solver": "PuLP + CBC (MILP)",
        "selected_projects": selected,
        "deferred_projects": deferred,
        "total_capex": round(total_capex, 2),
        "total_expected_npv": round(total_npv, 2),
        "budget_utilization_pct": round(total_capex / budget * 100, 1) if budget else 0.0,
        "n_funded": len(selected),
        "n_deferred": len(deferred),
        "average_risk": round(float(np.mean([p["risk_score"] for p in selected])), 3) if selected else 0.0,
        "constraints": {
            "budget": budget,
            "max_risk_exposure": max_risk_exposure,
            "min_strategic": min_s,
            "sector_exposure_caps": sec_caps,
        },
    }


def _solve_greedy(df, capex, npvs, risks, sectors, strategic, budget):
    """Deterministic greedy fallback: sort by NPV/cost ratio."""
    order = np.argsort(-(npvs / np.maximum(capex, 1e-9)))
    selected, deferred, used = [], [], 0.0
    for i in order:
        c = float(capex[i])
        if used + c <= budget:
            selected.append({
                "project_id": str(df["project_id"].iloc[i]),
                "sector": str(sectors[i]),
                "capex": c,
                "expected_npv": float(npvs[i]),
                "risk_score": float(risks[i]),
                "strategic_score": float(strategic[i]),
                "decision": "FUND",
            })
            used += c
        else:
            deferred.append({
                "project_id": str(df["project_id"].iloc[i]),
                "sector": str(sectors[i]),
                "capex": c,
                "expected_npv": float(npvs[i]),
                "risk_score": float(risks[i]),
                "strategic_score": float(strategic[i]),
                "decision": "DEFER",
            })
    total_capex = sum(p["capex"] for p in selected)
    total_npv = sum(p["expected_npv"] for p in selected)
    return {
        "status": "greedy_fallback",
        "solver": "greedy (NPV/capex ratio) - PuLP unavailable",
        "selected_projects": selected,
        "deferred_projects": deferred,
        "total_capex": round(total_capex, 2),
        "total_expected_npv": round(total_npv, 2),
        "budget_utilization_pct": round(total_capex / budget * 100, 1) if budget else 0.0,
        "n_funded": len(selected),
        "n_deferred": len(deferred),
        "average_risk": round(float(np.mean([p["risk_score"] for p in selected])), 3) if selected else 0.0,
        "constraints": {"budget": budget},
    }


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise the DataFrame to the canonical column set."""
    out = df.copy()
    n = len(out)
    if "project_id" not in out:
        out["project_id"] = [f"P{i}" for i in range(n)]
    if "sector" not in out:
        out["sector"] = "other"
    if "baseline_capex" not in out and "capex" in out:
        out["baseline_capex"] = out["capex"]
    if "baseline_capex" not in out:
        out["baseline_capex"] = 0.0
    if "expected_npv" not in out and "npv_usd" in out:
        out["expected_npv"] = out["npv_usd"]
    if "expected_npv" not in out:
        out["expected_npv"] = 0.0
    if "risk_score" not in out and "overrun_probability" in out:
        out["risk_score"] = out["overrun_probability"]
    if "risk_score" not in out and "overrun_risk" in out:
        out["risk_score"] = out["overrun_risk"]
    if "risk_score" not in out:
        out["risk_score"] = 0.5
    if "strategic_score" not in out:
        out["strategic_score"] = 1.0
    for col in ("baseline_capex", "expected_npv", "risk_score", "strategic_score"):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["risk_score"] = out["risk_score"].clip(0.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

def budget_sensitivity(
    projects: pd.DataFrame,
    budget_range: List[float],
    max_risk_exposure: Optional[float] = None,
) -> pd.DataFrame:
    """NPV, funded count and utilisation across a budget sweep."""
    if HAS_ENGINE:
        try:
            return _engine_budget_sensitivity(projects, budget_range,
                                              max_risk_exposure=max_risk_exposure)
        except Exception:
            pass
    rows = []
    for b in budget_range:
        r = optimize_portfolio(projects, b, max_risk_exposure=max_risk_exposure)
        rows.append({
            "budget": float(b),
            "total_expected_npv": r.get("total_expected_npv", 0.0),
            "n_funded": r.get("n_funded", 0),
            "budget_utilization_pct": r.get("budget_utilization_pct", 0.0),
            "average_risk": r.get("average_risk", 0.0),
        })
    return pd.DataFrame(rows)


def efficient_frontier(
    projects: pd.DataFrame,
    n_points: int = 8,
    budget: Optional[float] = None,
    risk_step: float = 0.05,
) -> pd.DataFrame:
    """NPV vs average-risk frontier as the risk ceiling is relaxed."""
    if HAS_ENGINE:
        try:
            return _engine_efficient_frontier(projects, n_points, budget, risk_step)
        except Exception:
            pass
    df = _normalise(projects)
    budget = budget or float(df["baseline_capex"].sum())
    rows = []
    for i in range(max(n_points, 1)):
        ceiling = risk_step * (i + 1)
        r = optimize_portfolio(projects, budget, max_risk_exposure=ceiling)
        rows.append({
            "max_risk_exposure": round(ceiling, 2),
            "total_expected_npv": r.get("total_expected_npv", 0.0),
            "average_risk": r.get("average_risk", 0.0),
            "n_funded": r.get("n_funded", 0),
        })
    return pd.DataFrame(rows)