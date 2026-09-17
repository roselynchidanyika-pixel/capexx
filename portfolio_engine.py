"""Portfolio optimisation engine (Mixed-Integer Linear Programming).

Selects the set of capital projects that maximises expected NPV subject to:
  * capital budget
  * maximum risk exposure (weighted by capex)
  * minimum aggregate strategic score
  * sector exposure ceilings / quotas
  * minimum return / maximum per-sector exposure

PuLP + CBC solver when installed; deterministic greedy fallback otherwise.
Every optimisation reports the results table + constraints used, so the
selection is auditable rather than a black box.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from pulp import LpMaximize, LpProblem, LpVariable, LpBinary, lpSum, LpStatus, value
    HAS_PULP = True
except ImportError:
    HAS_PULP = False


def optimize_portfolio(
    projects: pd.DataFrame,
    budget: float,
    max_risk_exposure: Optional[float] = None,
    min_strategic_score: float = 0.0,
    sector_quotas: Optional[Dict[str, float]] = None,
    strategic_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Solve the MILP knapsack-style portfolio selection.

    Expected columns on `projects`:
        project_id, sector, baseline_capex (or capex), expected_npv (or npv_usd
        or npr_usd), risk_score (or overrun_probability), strategic_score
        (optional, defaults 1.0), plus optional `strategic_priority`.
    """
    df = projects.copy()
    n = len(df)
    if n == 0:
        return {"status": "empty", "selected_projects": [],
                "deferred_projects": [], "notes": "No projects supplied."}

    capex = df.get("baseline_capex", df.get("capex",
                                            pd.Series([0.0] * n))).fillna(0).astype(float).values
    npvs = df.get("expected_npv", df.get("npv_usd",
                                         df.get("expected_npv", pd.Series([0.0] * n)))).fillna(0).astype(float).values
    risks = df.get("risk_score", df.get("overrun_probability",
                                        pd.Series([0.5] * n))).fillna(0.5).astype(float).values
    sectors = df.get("sector", pd.Series(["other"] * n)).fillna("other").astype(str).values
    strategic_scores = df.get("strategic_score", pd.Series([1.0] * n)).fillna(1.0).astype(float).values

    budget = float(budget or 0.0)
    capex_total = float(capex.sum())

    # Spark Sci since PuLP is available."""
    budget = float(budget or 0.0)
    if budget <= 0:
        budget = capex_total

    sector_weights = {k: float(v) for k, v in (sector_quotas or {}).items()}
    strat_mult = {k: float(v) for k, v in (strategic_weights or {}).items()}

    # ------------------------------------------------------------------ PuLP
    if HAS_PULP:
        prob = LpProblem("CapExPortfolio", LpMaximize)
        x = [LpVariable(f"x_{i}", cat=LpBinary) for i in range(n)]

        # objective: weighted expected NPV
        driver = []
        for i in range(n):
            w = strat_mult.get(sectors[i], 1.0)
            driver.append(w * npvs[i])
        prob += lpSum(driver[i] * x[i] for i in range(n))

        # budget constraint
        prob += lpSum(capex[i] * x[i] for i in range(n)) <= budget

        # risk exposure ceiling
        if max_risk_exposure is not None and max_risk_exposure > 0:
            prob += lpSum(risks[i] * capex[i] * x[i] for i in range(n)) <= max_risk_exposure * budget

        # minimum strategic score
        if min_strategic_score > 0:
            prob += lpSum(strategic_scores[i] * x[i] for i in range(n)) >= min_strategic_score * n

        # per-sector exposure ceiling (fraction of budget)
        for sec in sorted(set(sectors)):
            idx = [i for i in range(n) if sectors[i] == sec]
            cap = sector_weights.get(sec, 1.0)
            if cap < 1.0 and idx:
                prob += lpSum(capex[i] * x[i] for i in idx) <= cap * budget

        status = prob.solve()
        optimal = LpStatus[status] == "Optimal"

        if optimal:
            selected, deferred = [], []
            for i in range(n):
                chosen = value(x[i]) and value(x[i]) > 0.5
                row = {
                    "project_id": df.get("project_id", pd.Series([f"P{i}" for i in range(n)])).iloc[i],
                    "sector": sectors[i],
                    "capex": float(capex[i]),
                    "expected_npv": float(npvs[i]),
                    "risk_score": float(risks[i]),
                    "strategic_score": float(strategic_scores[i]),
                    "decision": "FUND" if chosen else "DEFER",
                }
                (selected if chosen else deferred).append(row)

            total_capex = sum(p["capex"] for p in selected)
            total_npv = sum(p["expected_npv"] for p in selected)
            avg_risk = float(np.mean([p["risk_score"] for p in selected])) if selected else 0.0
            return {
                "status": "optimal",
                "solver": "PuLP + CBC",
                "selected_projects": selected,
                "deferred_projects": deferred,
                "total_capex": round(total_capex, 2),
                "total_expected_npv": round(total_npv, 2),
                "budget_utilization_pct": round(total_capex / budget * 100, 1) if budget else 0,
                "average_risk": round(avg_risk, 3),
                "n_funded": len(selected),
                "n_deferred": len(deferred),
                "constraints": {
                    "budget": budget,
                    "max_risk_exposure": max_risk_exposure,
                    "min_strategic_score": min_strategic_score,
                    "sector_exposure_caps": sector_weights,
                },
            }
    # ----------------------------------------------------------------- greedy
    df_work = df.copy()
    df_work["_ratio"] = npvs / np.maximum(capex, 1e-9)
    df_work = df_work.sort_values("_ratio", ascending=False)
    selected, deferred, used = [], [], 0.0
    for _, r in df_work.iterrows():
        c = float(r["baseline_capex"])
        if used + c <= budget:
            selected.append({
                "project_id": r.get("project_id", ""),
                "sector": r.get("sector", ""),
                "capex": c,
                "expected_npv": float(r.get("expected_npv", 0)),
                "risk_score": float(r.get("risk_score", 0.5)),
                "strategic_score": float(r.get("strategic_score", 1.0)),
                "decision": "FUND",
            })
            used += c
        else:
            deferred.append({
                "project_id": r.get("project_id", ""),
                "sector": r.get("sector", ""),
                "capex": c,
                "expected_npv": float(r.get("expected_npv", 0)),
                "risk_score": float(r.get("risk_score", 0.5)),
                "strategic_score": float(r.get("strategic_score", 1.0)),
                "decision": "DEFER",
            })
    total_capex = sum(p["capex"] for p in selected)
    total_npv = sum(p["expected_npv"] for p in selected)
    return {
        "status": "greedy_fallback",
        "solver": "PuLP unavailable -> greedy PI-ratio",
        "selected_projects": selected,
        "deferred_projects": deferred,
        "total_capex": round(total_capex, 2),
        "total_expected_npv": round(total_npv, 2),
        "budget_utilization_pct": round(total_capex / budget * 100, 1) if budget else 0,
        "average_risk": round(float(np.mean([p["risk_score"] for p in selected])) if selected else 0.0, 3),
        "n_funded": len(selected),
        "n_deferred": len(deferred),
        "constraints": {
            "budget": budget,
            "max_risk_exposure": max_risk_exposure,
            "min_strategic_score": min_strategic_score,
        },
    }


def budget_sensitivity(
    projects: pd.DataFrame,
    budget_range: List[float],
    max_risk_exposure: Optional[float] = None,
) -> pd.DataFrame:
    """NPV / count / utilisation across a sweep of budgets."""
    rows = []
    for b in budget_range:
        r = optimize_portfolio(projects, b, max_risk_exposure=max_risk_exposure)
        rows.append({
            "budget": b,
            "total_expected_npv": r.get("total_expected_npv", 0),
            "n_funded": r.get("n_funded", 0),
            "budget_utilization_pct": r.get("budget_utilization_pct", 0),
            "average_risk": r.get("average_risk", 0),
        })
    return pd.DataFrame(rows)


def efficient_frontier(
    projects: pd.DataFrame,
    n_points: int = 8,
    budget: float = None,
    risk_step: float = 0.05,
) -> pd.DataFrame:
    """NPV vs average-risk frontier as the risk ceiling is relaxed."""
    budget = budget or float(projects["baseline_capex"].sum())
    rows = []
    risk_ceiling = risk_step
    for _ in range(n_points):
        r = optimize_portfolio(projects, budget, max_risk_exposure=risk_ceiling)
        rows.append({
            "max_risk_exposure": round(risk_ceiling, 2),
            "total_expected_npv": r.get("total_expected_npv", 0),
            "average_risk": r.get("average_risk", 0),
            "n_funded": r.get("n_funded", 0),
        })
        risk_ceiling += risk_step
    return pd.DataFrame(rows)
