"""Professional multi-page PDF + plain-text management report builder.

Uses fpdf2 core fonts ONLY (Helvetica) and Latin-1 safe characters so the
report renders identically on every machine with zero font downloads.
Companion generate_text_report() produces a plain-text variant.

Sections (subset selectable via `sections`):
    Executive Summary, Zimbabwe Economic Environment, Live Macro Data,
    Cash Flow, Capital Budgeting, Risk and ML, Stress Test, Sensitivity,
    Scenarios, AI Explanation, Key Assumptions, Data Limitations,
    Final Findings
"""
from __future__ import annotations

import os
import re
import unicodedata
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.data_model import MacroContext, ProjectInput


# ---------------------------------------------------------------------------
# Latin-1 sanitising (no emoji, no unicode arrows -> '->')
# ---------------------------------------------------------------------------

_ARROWS = {"\u2192": "->", "\u2190": "<-", "\u2194": "<->", "\u21d2": "=>"}


def latin1(text: Any) -> str:
    """Return a Latin-1-safe rendering of `text`."""
    s = str(text if text is not None else "")
    for k, v in _ARROWS.items():
        s = s.replace(k, v)
    s = unicodedata.normalize("NFKD", s)
    out = []
    for ch in s:
        try:
            ch.encode("latin-1")
            out.append(ch)
        except UnicodeEncodeError:
            out.append("")
    return "".join(out)


def _usd(v: float) -> str:
    return "{:,.0f}".format(v) if abs(v) >= 0.5 else "{:,.2f}".format(v)


def _pct(v: float) -> str:
    return "{:,.1f}%".format(v)


def _rate(v: float) -> str:
    return "{:,.1%}".format(v)


def _payback(v: float) -> str:
    return "{:.1f}".format(v) if v != float("inf") else "not reached"


# ---------------------------------------------------------------------------
# Section content builders (shared by PDF + text)
# ---------------------------------------------------------------------------

def _exec_summary(project, r, status_text) -> List[str]:
    return [
        "EXECUTIVE SUMMARY",
        "",
        "Project: %s  (ID %s)" % (project.project_name, project.project_id),
        "Sector: %s | Type: %s | Province: %s" % (project.sector, project.project_type, project.province),
        "Currency: %s | Initial investment: USD %s" % (project.currency, _usd(project.initial_investment)),
        "Management decision: %s" % r.get("decision", "N/A"),
        "Risk status: %s" % latin1(status_text),
        "",
        "NPV: USD %s | IRR: %s | MIRR: %s | WACC: %s" % (
            _usd(r.get("npv", 0)), _rate(r.get("irr", 0)), _rate(r.get("mirr", 0)),
            _rate(r.get("wacc", 0))),
        "Payback: %s years | PI: %.2f | EAA: USD %s" % (
            _payback(r.get("payback", float("inf"))), r.get("pi", 0),
            _usd(r.get("eaa", 0))),
        "Reason: %s" % latin1(r.get("reason", "")) if r.get("reason") else "",
    ]


def _macro_env(project, macro) -> List[str]:
    return [
        "ZIMBABWE ECONOMIC ENVIRONMENT",
        "",
        "Inflation (ZiG y/y): %s" % _pct(macro.inflation_pct()),
        "Policy rate: %s" % _pct(macro.policy_rate_pct()),
        "Lending rate: %s" % _pct(macro.lending_rate_pct()),
        "US policy rate: %s" % _pct(float(macro.get("us_policy_rate_pct", 0) or macro.us_inflation_pct() + 2.0)),
        "Exchange rate: ZiG per USD = %s" % ("{:.4f}".format(macro.exchange_rate_zig_per_usd())),
        "Construction price index: %s" % "{:.1f}".format(macro.construction_price_index()),
    ]


def _live_macro(macro) -> List[str]:
    raw = getattr(macro, "raw", {})
    lines = ["LIVE MACRO DATA", "", "Variable | Source | Observed | Retrieved | Status | Value", ""]
    cards = {
        "zig_usd": "ZiG per USD",
        "inflation_zim": "Inflation ZiG y/y",
        "inflation_us": "US inflation y/y",
        "policy_rate_zim": "Policy rate",
        "lending_rate_zim": "Lending rate",
        "policy_rate_us": "US policy rate",
        "construction_index": "Construction price index",
    }
    for key, label in cards.items():
        d = (raw or {}).get(key, {}) or {}
        if not isinstance(d, dict) or "value" in d:
            lines.append("%s | %s | %s | %s | %s | %s" % (
                label,
                latin1(d.get("source", "unavailable")),
                latin1(d.get("observation_date") or "n/a"),
                latin1(d.get("retrieved_at") or "n/a"),
                latin1(d.get("status", "unavailable")),
                _usd(float(d.get("value", 0) or 0)) if isinstance(d.get("value"), (int, float)) else latin1(d.get("value") or "n/a"),
            ))
    lines += [
        "",
        "Status transparency: 'live' = OpenAPI fetch, 'fallback_estimate' = cached",
        "default used because the source was unreachable, 'live_proxy' = proxy series.",
        "Every non-live figure is never presented as official Zimbabwe data.",
    ]
    return lines


def _cash_flow(r) -> List[str]:
    lines = ["CASH FLOW PROJECTION", ""]
    table = r.get("cash_flow_table")
    if table is None or len(table) == 0:
        lines.append("No cash-flow rows were returned.")
        return lines
    head = "Year | Revenue | Op Costs | EBITDA | Depr | EBIT | Tax | NOPAT | OCF | Terminal | Net CF | Cumulative"
    lines.append(head)
    lines.append("-" * len(head))
    for _, row in table.iterrows():
        lines.append("%d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s" % (
            int(row.get("Year", 0)),
            _usd(row.get("Revenue", 0)), _usd(row.get("Operating Costs", 0)),
            _usd(row.get("EBITDA", 0)), _usd(row.get("Depreciation", 0)),
            _usd(row.get("EBIT", 0)), _usd(row.get("Tax", 0)),
            _usd(row.get("NOPAT", 0)), _usd(row.get("Operating Cash Flow", 0)),
            _usd(row.get("Terminal Value", 0)), _usd(row.get("Net Cash Flow", 0)),
            _usd(row.get("Cumulative CF", 0)),
        ))
    return lines


def _cap_budgeting(project, r) -> List[str]:
    return [
        "CAPITAL BUDGETING",
        "",
        "Net Present Value (NPV): USD %s" % _usd(r.get("npv", 0)),
        "Internal Rate of Return (IRR): %s" % _rate(r.get("irr", 0)),
        "Modified IRR (MIRR): %s" % _rate(r.get("mirr", 0)),
        "Payback: %s years | Discounted payback: %s years" % (
            _payback(r.get("payback", float("inf"))), _payback(r.get("discounted_payback", float("inf")))),
        "Accounting Rate of Return (ARR): %s" % _rate(r.get("arr", 0)),
        "Profitability Index (PI): %.2f" % r.get("pi", 0),
        "DCF value: USD %s | EAA: USD %s" % (_usd(r.get("dcf_value", 0)), _usd(r.get("eaa", 0))),
        "Break-even revenue: USD %s (annual revenue today: USD %s)" % (
            _usd(r.get("breakeven_revenue", 0)), _usd(project.annual_revenue)),
        "Weighted-average cost of capital: %s" % _rate(r.get("wacc", 0)),
        "Expected capex (escalated + contingency): USD %s" % _usd(r.get("expected_capex", 0)),
        "",
        "Formulas used:",
        "  NPV = SUM(CF_t / (1+r)^t) - I0, discount rate r = WACC",
        "  IRR  = discount rate with NPV = 0; MIRR = reinvestment-aware IRR",
        "  PI   = PV(inflows) / I0; EAA = NPV converted to annuity",
    ]


def _risk_ml(project, r, status_text, status_detail) -> List[str]:
    return [
        "RISK AND ML",
        "",
        "Complexity score: %s/5 | Design completeness: %s%% | Change orders: %d" % (
            int(project.complexity_score), int(project.design_completeness * 100),
            int(project.num_change_orders)),
        "Procurement delay: %d days | Contractor type: %s | Funding: %s" % (
            int(project.procurement_delay_days), project.contractor_type, project.funding_source),
        "Geometric status: %s (%s)" % (status_text, latin1(status_detail)),
        "",
        "Risk classifier rules (fully transparent):",
        "  RED triangle  = NPV < 0 OR IRR < WACC OR PI < 1",
        "  AMBER diamond = NPV > 0 but payback > 70% of life OR PI < 1.25 OR +10pt inflation flips NPV",
        "  GREEN hexagon = otherwise (stable, value-creating, resilient)",
        "",
        "ML caveat: the ML lab trains on SYNTHETIC demonstration data only",
        "and must never be treated as a bankable credit-risk model.",
    ]


def _stress(project, macro, r) -> List[str]:
    from core.financial_engine import stress_case
    try:
        sc = stress_case(project, macro)
    except Exception:
        sc = None
    lines = ["STRESS TEST", ""]
    if sc is None:
        lines.append("Stress scenario could not be computed.")
        return lines
    b, s, d = sc["base"], sc["stressed"], sc["deltas"]
    lines += [
        "Base NPV: USD %s | Stress NPV: USD %s | Delta: USD %s" % (
            _usd(b.get("npv", 0)), _usd(s.get("npv", 0)), _usd(d.get("npv", 0))),
        "Base IRR: %s | Stress IRR: %s | Delta: %s" % (
            _rate(b.get("irr", 0)), _rate(s.get("irr", 0)), _rate(d.get("irr", 0))),
        "Base MIRR: %s | Stress MIRR: %s" % (
            _rate(b.get("mirr", 0)), _rate(s.get("mirr", 0))),
        "Base capex: USD %s | Stress capex: USD %s | Delta: USD %s" % (
            _usd(b.get("expected_capex", 0)), _usd(s.get("expected_capex", 0)), _usd(d.get("capex", 0))),
        "Base payback: %s y | Stress payback: %s y" % (
            _payback(b.get("payback", float("inf"))), _payback(s.get("payback", float("inf")))),
        "",
        "Shock profile: inflation +20 pts, ZiG depreciation +25%, interest +6 pts,",
        "materials +15%, revenue growth -5pts.",
        "Summary: %s" % latin1(sc.get("summary", "")),
    ]
    return lines


def _sensitivity(project, r) -> List[str]:
    from core.financial_engine import sensitivity_summary
    try:
        summary = latin1(sensitivity_summary(project))
    except Exception:
        summary = "Sensitivity summary unavailable."
    return [
        "SENSITIVITY",
        "",
        "One-driver-at-a-time NPV swings are computed by the tornado scan.",
        summary,
        "",
        "Recommended focus: hedge the highest-swing driver first;",
        "lock prices/fixed-price contracts for the hardest inputs;",
        "re-run after any macro change.",
    ]


def _scenarios(project, r) -> List[str]:
    from core.financial_engine import build_scenarios
    lines = ["SCENARIOS", ""]
    try:
        df = build_scenarios(project)
    except Exception:
        lines.append("Scenario table could not be computed.")
        return lines
    for _, row in df.iterrows():
        lines.append("%s | NPV %s | IRR %s | MIRR %s | Payback %s y | PI %.2f" % (
            latin1(row.get("Scenario", "")), _usd(row.get("npv", 0)),
            _rate(row.get("irr", 0)), _rate(row.get("mirr", 0)),
            _payback(row.get("payback", float("inf"))), row.get("pi", 0)))
    return lines


def _ai_explanation(project, macro, r) -> List[str]:
    from core.ai_explainer import build_explanation
    try:
        ex = build_explanation(project, macro, r)
    except Exception:
        ex = {}
    p = ex.get("plain", {})
    t = ex.get("technical", {})
    n = ex.get("narrative", "")
    return [
        "AI EXPLANATION (deterministic - every figure is computed, not guessed)",
        "",
        "WHAT HAPPENED: %s" % latin1(p.get("what_happened", "")),
        "WHY: %s" % latin1(p.get("why", "")),
        "WHAT DOES IT MEAN: %s" % latin1(p.get("what_does_it_mean", "")),
        "MAIN DRIVER: %s" % latin1(p.get("main_driver", "")),
        "",
        "TECHNICAL VIEW: %s" % latin1(t.get("what_happened", "")),
        "ROBOT NARRATIVE: %s" % latin1(n),
    ]


def _assumptions(project) -> List[str]:
    return [
        "KEY ASSUMPTIONS",
        "",
        "Construction phase: %s months" % _pct(project.construction_months),
        "Operating life: %s years" % _pct(project.project_life),
        "Revenue growth (real): %s%% | Operating-cost growth: %s%%" % (
            project.revenue_growth_pct, project.operating_cost_growth_pct),
        "Imported equipment share (hard currency): %s%%" % project.imported_equipment_pct,
        "Contingency reserve: %s%% | Salvage value: %s%%" % (
            project.contingency_pct, project.salvage_value_pct),
        "Working capital: %s%% of revenue | Terminal growth: %s%%" % (
            project.working_capital_pct, project.terminal_growth_pct),
        "Debt ratio: %s%% | Debt interest: %s%% | Tax: %s%%" % (
            project.debt_ratio_pct, project.debt_interest_pct, project.tax_rate_pct),
        "Reinvestment rate (MIRR): %s%%" % project.reinvestment_rate_pct,
        "WACC override (0 = auto from macro): %s%%" % project.wacc_pct,
    ]


def _limitations() -> List[str]:
    return [
        "DATA LIMITATIONS",
        "",
        "Every macro indicator is tagged live | fallback_estimate | live_proxy.",
        "When a public API is unreachable the model uses clearly-labelled",
        "fallback defaults and NEVER disguises them as official Zimbabwe data.",
        "",
        "Outputs are planning aids, not bankable valuations. Before funding:",
        "  * source primary ZIMSTAT / RBZ data directly,",
        "  * run the Monte-Carlo and stress lab with your own tails,",
        "  * obtain professional QS and legal review.",
        "",
        "The ML lab uses SYNTHETIC demonstration data for training and cannot",
        "be relied on for real credit decisions.",
    ]


def _findings() -> List[str]:
    return ["FINAL FINDINGS", ""]  # filled by caller via project_dict


def _findings_text(project, r, status_text) -> List[str]:
    finding = (
        "The project %s is %s under current Zimbabwe macro assumptions with a "
        "computed NPV of USD %s and IRR of %s. Management should %s."
        % (project.project_name,
           r.get("decision", "N/A"),
           _usd(r.get("npv", 0)), _rate(r.get("irr", 0)),
           "proceed, subject to stress testing and mitigants"
           if r.get("npv", 0) >= 0 else "reconsider, re-scope or defer until the environment improves"))
    return [finding, "", "Generated by the CapEx AI Agent."]


_ALL_SECTIONS = [
    "Executive Summary", "Zimbabwe Economic Environment", "Live Macro Data",
    "Cash Flow", "Capital Budgeting", "Risk and ML", "Stress Test",
    "Sensitivity", "Scenarios", "AI Explanation", "Key Assumptions",
    "Data Limitations", "Final Findings",
]


def _build_items(project, macro, result, sections):
    from core.risk_engine import classify_status
    status = classify_status(result)
    status_text = "%s - %s" % (status.get("shape"), status.get("level"))
    status_detail = status.get("reasons", "")

    builders = {
        "Executive Summary": lambda: _exec_summary(project, result, status_text),
        "Zimbabwe Economic Environment": lambda: _macro_env(project, macro),
        "Live Macro Data": lambda: _live_macro(macro),
        "Cash Flow": lambda: _cash_flow(result),
        "Capital Budgeting": lambda: _cap_budgeting(project, result),
        "Risk and ML": lambda: _risk_ml(project, result, status_text, status_detail),
        "Stress Test": lambda: _stress(project, macro, result),
        "Sensitivity": lambda: _sensitivity(project, result),
        "Scenarios": lambda: _scenarios(project, result),
        "AI Explanation": lambda: _ai_explanation(project, macro, result),
        "Key Assumptions": lambda: _assumptions(project),
        "Data Limitations": lambda: _limitations(),
        "Final Findings": lambda: _findings_text(project, result, status_text),
    }
    if sections is None:
        sections = _ALL_SECTIONS
    out = []
    for sec in sections:
        if sec in builders:
            out.append(builders[sec]())
    return out


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def generate_text_report(project: ProjectInput, macro: MacroContext,
                         result: Dict[str, Any],
                         sections: Optional[List[str]] = None) -> str:
    """Return a plain-text version of the management report."""
    items = _build_items(project, macro, result, sections)
    blocks = ["=" * 78, latin1("CAPEX AI AGENT - MANAGEMENT REPORT"),
              "%s" % datetime.now().strftime("%Y-%m-%d %H:%M"), "=" * 78, ""]
    for block in items:
        for line in block:
            if isinstance(line, str):
                blocks.append(latin1(line))
            else:
                blocks.append(latin1(line))
        blocks.append("")
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# PDF report (fpdf2, core fonts, Latin-1)
# ---------------------------------------------------------------------------

class _ReportPDF:
    def __init__(self) -> None:
        from fpdf import FPDF
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=18)

    def _header_every_page(self):
        self.pdf.set_font("Helvetica", "B", 14)
        self.pdf.set_text_color(10, 22, 40)
        self.pdf.cell(0, 8, latin1("CAPEX AI AGENT - MANAGEMENT REPORT"),
                      new_x="LMARGIN", new_y="NEXT", align="C")
        self.pdf.set_draw_color(46, 139, 255)
        self.pdf.line(10, 18, 200, 18)

    def add_page(self):
        self.pdf.add_page()
        self._header_every_page()

    def title(self, text):
        self.pdf.ln(2)
        self.pdf.set_font("Helvetica", "B", 12)
        self.pdf.set_text_color(10, 40, 80)
        self.pdf.cell(0, 7, latin1(text), new_x="LMARGIN", new_y="NEXT")
        self.pdf.set_draw_color(46, 139, 255)
        self.pdf.line(10, self.pdf.get_y(), 200, self.pdf.get_y())
        self.pdf.ln(2)

    def para(self, text):
        self.pdf.set_font("Helvetica", "", 9)
        self.pdf.set_text_color(20, 20, 20)
        self.pdf.multi_cell(0, 5, latin1(text), new_x="LMARGIN", new_y="NEXT")
        self.pdf.ln(1)

    def code(self, text):
        self.pdf.set_font("Courier", "", 8)
        self.pdf.set_text_color(40, 40, 40)
        self.pdf.multi_cell(0, 4, latin1(text), new_x="LMARGIN", new_y="NEXT")

    def spacer(self):
        self.pdf.ln(3)


def generate_pdf_report(project: ProjectInput, macro: MacroContext,
                        result: Dict[str, Any],
                        sections: Optional[List[str]] = None,
                        out_dir: Optional[str] = None) -> str:
    """Build the PDF management report and return the file path.

    `out_dir` defaults to <project root>/reports (created if missing).
    """
    from core.risk_engine import classify_status
    from fpdf import FPDF  # noqa: F401 - imported for side-effect check

    status = classify_status(result)
    status_text = "%s - %s" % (status.get("shape"), status.get("level"))

    if out_dir is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_dir = os.path.join(base, "reports")
    os.makedirs(out_dir, exist_ok=True)

    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", project.project_id) or "project"
    fname = "%s_%s.pdf" % (safe_id, datetime.now().strftime("%Y%m%d_%H%M%S"))
    path = os.path.join(out_dir, fname)

    doc = _ReportPDF()
    doc.add_page()

    items = _build_items(project, macro, result, sections)
    for block in items:
        first = True
        for line in block:
            text = str(line)
            if not text.strip():
                doc.spacer()
                continue
            if first or _is_title(text, block):
                doc.title(text if first else text)
            else:
                if " | " in text and len(text) < 200:
                    doc.code(text)
                else:
                    doc.para(text)
            first = False
        doc.spacer()

    doc.pdf.output(path)
    return path


def _is_title(text: str, block: List[str]) -> bool:
    stripped = text.strip()
    return stripped.isupper() and not any(c.isdigit() for c in stripped) and len(stripped) < 60


# ---------------------------------------------------------------------------
# Convenience: build once, both outputs
# ---------------------------------------------------------------------------

def build_report_files(project, macro, result,
                       sections: Optional[List[str]] = None) -> Dict[str, str]:
    """Generate PDF + TXT and return {pdf: path, txt: path}."""
    from core.risk_engine import classify_status
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base, "reports")
    os.makedirs(out_dir, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", project.project_id) or "project"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf = generate_pdf_report(project, macro, result, sections, out_dir)
    txt_path = os.path.join(out_dir, "%s_%s.txt" % (safe_id, stamp))
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(generate_text_report(project, macro, result, sections))
    return {"pdf": pdf, "txt": txt_path}