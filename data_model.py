"""Canonical data model for the Capital Projects Finance AI Agent.

All project inputs, macro context and portfolio settings ride on these
dataclasses so every engine consumes ONE consistent project definition.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple


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


# ---------------------------------------------------------------------------
# File upload (Option B) parsing + validation
# ---------------------------------------------------------------------------

_NUMERIC_FIELDS = {
    "initial_investment", "imported_equipment_pct", "contingency_pct",
    "salvage_value_pct", "construction_months", "project_life",
    "annual_revenue", "revenue_growth_pct", "operating_costs",
    "operating_cost_growth_pct", "working_capital_pct", "terminal_growth_pct",
    "debt_ratio_pct", "debt_interest_pct", "equity_cost_pct", "wacc_pct",
    "reinvestment_rate_pct", "tax_rate_pct", "debt_tenor_years",
    "complexity_score", "design_completeness", "procurement_delay_days",
    "num_change_orders",
}

_STRING_FIELDS = {
    "project_id", "project_name", "sector", "project_type", "currency",
    "contractor_type", "funding_source", "procurement_method", "province",
    "submitter_email", "expected_completion", "project_location",
    "contractor_name", "notes", "remarks",
}

_COLUMN_ALIASES = {
    "projectname": "project_name",
    "name": "project_name",
    "projectid": "project_id",
    "sector": "sector",
    "projecttype": "project_type",
    "initialinvestment": "initial_investment",
    "initialinvestmentusd": "initial_investment",
    "capex": "initial_investment",
    "projectlife": "project_life",
    "projectduration": "project_life",
    "annualrevenue": "annual_revenue",
    "revenue": "annual_revenue",
    "operatingcosts": "operating_costs",
    "email": "submitter_email",
    "submitteremail": "submitter_email",
    "location": "project_location",
}


def _norm_col(name: Any) -> str:
    """Normalise a spreadsheet column to a field name."""
    s = str(name or "").strip().lower()
    s = "".join(ch if ch.isalnum() else "_" for ch in s)
    s = s.strip("_")
    for k, v in _COLUMN_ALIASES.items():
        if s in (k, k.replace("_", "")):
            return v
    return s


def projects_from_upload(df: Any) -> Tuple[List[ProjectInput], List[str]]:
    """Parse uploaded CSV/Excel rows into `ProjectInput` records.

    Returns (projects, errors). Rows that fail validation are skipped and a
    readable error is appended. Never throws; caller renders `errors` in the UI.
    """
    errors: List[str] = []
    projects: List[ProjectInput] = []

    if df is None or getattr(df, "shape", (0, 0))[0] < 1:
        return [], ["Uploaded file has no data rows."]

    cols = {_norm_col(c): str(c) for c in df.columns}
    if "initial_investment" not in cols:
        errors.append("Initial investment is missing (no 'initial_investment' "
                      "or 'capex' column found).")

    if "submitter_email" in cols:
        bad = df[cols["submitter_email"]].dropna().astype(str)
        invalid = [v for v in bad if "@" not in v or "." not in v.split("@")[-1]]
        if invalid:
            errors.append("Submitter email is invalid in %d row(s)." % len(invalid))

    for idx, row in df.iterrows():
        values: Dict[str, Any] = {}
        for field in _NUMERIC_FIELDS:
            if field not in cols:
                continue
            v = row.get(cols[field])
            if v is None or (isinstance(v, float) and pd_isna(v)):
                continue
            try:
                values[field] = float(v)
            except (TypeError, ValueError):
                errors.append("Invalid numeric value for '%s' in row %d." % (field, idx + 2))
        for field in _STRING_FIELDS:
            if field not in cols:
                continue
            v = row.get(cols[field])
            if v is None or (isinstance(v, float) and pd_isna(v)):
                continue
            values[field] = str(v).strip()

        if "initial_investment" not in values or values.get("initial_investment", 0) is None:
            errors.append("Initial investment is missing in row %d." % (idx + 2))
            continue
        if float(values.get("initial_investment", 0) or 0) <= 0:
            errors.append("Initial investment must be greater than zero in row %d." % (idx + 2))
            continue
        if "project_life" in values and float(values["project_life"] or 0) <= 0:
            errors.append("Project duration must be greater than zero in row %d." % (idx + 2))
            continue

        email = values.get("submitter_email", "")
        if email:
            if "@" not in email or "." not in email.split("@")[-1]:
                errors.append("Submitter email is invalid in row %d." % (idx + 2))
                continue

        defaults = asdict(ProjectInput())
        defaults.update(values)
        defaults.pop("notes_flag", None)
        try:
            projects.append(ProjectInput(**defaults))
        except Exception as e:  # noqa: BLE001
            errors.append("Could not build project from row %d: %s" % (idx + 2, e))

    return projects, errors


def pd_isna(v: Any) -> bool:
    try:
        import math
        return bool(math.isnan(float(v)))
    except (TypeError, ValueError):
        return False
