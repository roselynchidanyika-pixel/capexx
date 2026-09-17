"""Canonical data model for the Capital Projects Finance AI Agent.

All project inputs, macro context and portfolio settings ride on these
dataclasses so every engine consumes ONE consistent project definition.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any


@dataclass
class ProjectInput:
    """Project definition as entered by the user (manual or upload)."""

    project_id: str = "ZIM-1001"
    project_name: str = "Unnamed Capital Project"
    sector: str = "roads"
    project_type: str = "infrastructure"
    currency: str = "USD"

    # ------------------------------------------------------------------ capex
    initial_investment: float = 10_000_000.0
    imported_equipment_pct: float = 40.0        # share of capex in hard currency
    contingency_pct: float = 10.0               # contingency reserve on capex
    salvage_value_pct: float = 10.0             # % of investment recovered at end
    construction_months: float = 24.0           # build-out phase

    # -------------------------------------------------------------- operations
    project_life: float = 10.0                  # total operating life (years)
    annual_revenue: float = 4_000_000.0
    revenue_growth_pct: float = 5.0
    operating_costs: float = 2_000_000.0
    operating_cost_growth_pct: float = 30.0     # Zimbabwe cost-escalation default
    working_capital_pct: float = 10.0           # % of annual revenue
    terminal_growth_pct: float = 2.0            # perpetuity growth after life

    # ------------------------------------------------------------- financing
    debt_ratio_pct: float = 60.0
    debt_interest_pct: float = 18.0             # borrowing cost (macro-linked)
    equity_cost_pct: float = 0.0                # 0 -> auto from macro risk-free
    wacc_pct: float = 0.0                       # 0 -> computed from macro
    reinvestment_rate_pct: float = 5.0          # for MIRR
    tax_rate_pct: float = 25.0
    debt_tenor_years: float = 8.0

    # ------------------------------------------------------------- risk / ML
    complexity_score: float = 3.0               # 1..5
    design_completeness: float = 0.60           # 0..1
    procurement_delay_days: float = 60.0
    num_change_orders: float = 3.0
    contractor_type: str = "local_large"
    funding_source: str = "mixed"
    procurement_method: str = "open_competitive"
    province: str = "Harare"

    # ------------------------------------------------------------- metadata
    submitter_email: str = ""
    expected_completion: str = ""
    project_location: str = ""
    contractor_name: str = ""
    notes: str = ""
    remarks: str = ""


@dataclass
class MacroContext:
    """Live Zimbabwe macro context that feeds straight into the model."""

    fetched_at: str = ""
    indicators: Dict[str, Any] = field(default_factory=dict)
    last_updated: str = ""
    observation_dates: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.indicators.get(key, default)

    def inflation_pct(self) -> float:
        v = self.indicators.get("inflation_pct")
        return float(v) if v is not None else 20.0

    def exchange_rate_zig_per_usd(self) -> float:
        return float(self.indicators.get("zig_per_usd", 0) or 13.5)

    def policy_rate_pct(self) -> float:
        return float(self.indicators.get("policy_rate_pct", 0) or 15.0)

    def lending_rate_pct(self) -> float:
        return float(self.indicators.get("lending_rate_pct", 0) or 25.0)

    def construction_price_index(self) -> float:
        return float(self.indicators.get("construction_price_index", 0) or 100.0)

    def us_inflation_pct(self) -> float:
        return float(self.indicators.get("us_inflation_pct", 0) or 2.8)


@dataclass
class PortfolioSettings:
    budget: float = 100_000_000.0
    max_risk_exposure: float = 0.60
    min_strategic_score: float = 0.0
    min_return_pct: float = 10.0
    max_sector_exposure_pct: float = 40.0
    sector_quotas: Dict[str, float] = field(default_factory=dict)
    strategic_weights: Dict[str, float] = field(default_factory=dict)


def project_to_dict(p: ProjectInput) -> Dict[str, Any]:
    return asdict(p)
