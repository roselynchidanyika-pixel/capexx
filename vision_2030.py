"""Vision 2030 alignment engine for the Capital Projects Finance AI Agent.

Descriptive, deterministic mapping of every project sector onto Zimbabwe's
Vision 2030 ambition and the 14 National Development Goals of the National
Development Strategy (NDS) period to 2030. The alignment is inferential and
labelled as AI judgement for decision support, never as an official government
endorsement.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.data_model import ProjectInput

VISION_2030_TITLE = "Vision 2030"
NDS2_PILLAR = ("National Development Strategy period to 2030 - the 14 National "
               "Development Goals behind an upper-middle-income Zimbabwe")

NDGS: List[Dict[str, str]] = [
    {"code": "NDG1", "name": "Poverty-Free Society"},
    {"code": "NDG2", "name": "Food and Nutrition Security"},
    {"code": "NDG3", "name": "Good Health and Well-Being"},
    {"code": "NDG4", "name": "Quality Education and Skills Development"},
    {"code": "NDG5", "name": "Decent Work and Inclusive Sustainable Economic Growth"},
    {"code": "NDG6", "name": "Sustainable Industrialisation, Innovation and Infrastructure"},
    {"code": "NDG7", "name": "Sustainable Cities, Communities and Housing"},
    {"code": "NDG8", "name": "Climate-Smart, Resilient and Sustainable Environment"},
    {"code": "NDG9", "name": "Water and Sanitation"},
    {"code": "NDG10", "name": "Gender Equality"},
    {"code": "NDG11", "name": "Good Governance and Institutions"},
    {"code": "NDG12", "name": "Sustainable Peace and Security"},
    {"code": "NDG13", "name": "Just and Inclusive Society, Culture and Heritage"},
    {"code": "NDG14", "name": "Sustainable Energy and Digital Economy"},
]

_SECTOR_GOALS: Dict[str, List[Dict[str, Any]]] = {
    "roads": [
        {"code": "NDG6", "weight": 90, "rationale": "road and logistics infrastructure links producers to markets"},
        {"code": "NDG5", "weight": 70, "rationale": "construction labour and mobility support decent work"},
        {"code": "NDG7", "weight": 60, "rationale": "connectivity binds urban and rural communities"},
    ],
    "energy": [
        {"code": "NDG14", "weight": 90, "rationale": "energy capacity and reliability underpin the digital economy"},
        {"code": "NDG8", "weight": 65, "rationale": "renewable additions support climate-smart growth"},
        {"code": "NDG5", "weight": 55, "rationale": "energy access enables enterprise and jobs"},
    ],
    "mining": [
        {"code": "NDG6", "weight": 80, "rationale": "value-chain beneficiation infrastructure"},
        {"code": "NDG5", "weight": 75, "rationale": "exports, royalties and jobs from extractive value chains"},
    ],
    "manufacturing": [
        {"code": "NDG6", "weight": 90, "rationale": "industrialisation and value addition"},
        {"code": "NDG5", "weight": 75, "rationale": "manufacturing jobs and household incomes"},
        {"code": "NDG14", "weight": 50, "rationale": "factory automation and digital processes"},
    ],
    "water": [
        {"code": "NDG9", "weight": 95, "rationale": "water supply and sanitation services"},
        {"code": "NDG2", "weight": 70, "rationale": "irrigation and food security"},
        {"code": "NDG8", "weight": 50, "rationale": "climate-resilient water infrastructure"},
    ],
    "agriculture": [
        {"code": "NDG2", "weight": 90, "rationale": "food and nutrition security"},
        {"code": "NDG5", "weight": 65, "rationale": "agri-value-chain jobs and incomes"},
        {"code": "NDG8", "weight": 50, "rationale": "climate-smart farming and resilience"},
    ],
    "aviation": [
        {"code": "NDG6", "weight": 85, "rationale": "gateway infrastructure for trade and tourism"},
        {"code": "NDG5", "weight": 55, "rationale": "aviation-sector employment"},
        {"code": "NDG7", "weight": 40, "rationale": "regional connectivity"},
    ],
    "education": [
        {"code": "NDG4", "weight": 95, "rationale": "quality education and skills development"},
        {"code": "NDG6", "weight": 55, "rationale": "innovation hubs and research infrastructure"},
        {"code": "NDG5", "weight": 45, "rationale": "future workforce readiness"},
    ],
    "health": [
        {"code": "NDG3", "weight": 95, "rationale": "good health and well-being facilities"},
        {"code": "NDG5", "weight": 40, "rationale": "health-sector employment"},
    ],
    "housing": [
        {"code": "NDG7", "weight": 90, "rationale": "sustainable cities, communities and housing"},
        {"code": "NDG5", "weight": 45, "rationale": "construction jobs and formal employment"},
    ],
    "tourism": [
        {"code": "NDG5", "weight": 80, "rationale": "inclusive growth through tourism"},
        {"code": "NDG7", "weight": 50, "rationale": "destination towns and cities"},
    ],
    "telecoms": [
        {"code": "NDG14", "weight": 95, "rationale": "digital economy backbone"},
        {"code": "NDG4", "weight": 55, "rationale": "connectivity enables e-learning"},
        {"code": "NDG5", "weight": 50, "rationale": "digital jobs and platforms"},
    ],
    "other": [
        {"code": "NDG6", "weight": 55, "rationale": "generic alignment to innovation and infrastructure"},
        {"code": "NDG5", "weight": 45, "rationale": "potential job creation"},
    ],
}

_SECTOR_SYNONYMS = {
    "solar": "energy",
    "renewable": "energy",
    "renewables": "energy",
    "power": "energy",
    "electricity": "energy",
    "roads": "roads",
    "transport": "roads",
    "irrigation": "water",
    "sanitation": "water",
    "sewer": "water",
    "waste": "water",
    "education": "education",
    "ict": "telecoms",
    "digital": "telecoms",
    "telecom": "telecoms",
    "agri": "agriculture",
    "farming": "agriculture",
    "tourism": "tourism",
    "hospital": "health",
    "clinics": "health",
    "housing": "housing",
    "estates": "housing",
    "manufacturing": "manufacturing",
    "mining": "mining",
    "aviation": "aviation",
}

_TYPE_BONUS = [
    ("solar", 5, "NDG14", "solar generation - flagship renewable ambition"),
    ("renewable", 5, "NDG8", "renewable energy - climate-smart signal"),
    ("digital", 5, "NDG14", "digital economy enabler"),
    ("innovation", 5, "NDG6", "innovation and research infrastructure"),
    ("research", 5, "NDG4", "skills and research development"),
]


def normalize_sector(sector: Optional[str], project_type: Optional[str] = None) -> str:
    raw = " ".join([sector or "", project_type or ""]).lower().strip()
    for key, mapped in _SECTOR_SYNONYMS.items():
        if key in raw:
            return mapped
    if sector and sector.strip().lower() in _SECTOR_GOALS:
        return sector.strip().lower()
    return "other"


def ndg_name(want: str) -> Optional[str]:
    for g in NDGS:
        if g["code"] == want:
            return g["name"]
    return None


def _type_bonus(project: ProjectInput) -> List[Dict[str, Any]]:
    bonus = []
    tl = (project.project_type or "").lower()
    for token, weight, code, rationale in _TYPE_BONUS:
        if token in tl:
            bonus.append({"code": code, "weight": weight, "rationale": rationale})
    return bonus


def project_alignment(project: ProjectInput) -> Dict[str, Any]:
    sector = normalize_sector(project.sector, project.project_type)
    goals = [dict(g) for g in _SECTOR_GOALS.get(sector, _SECTOR_GOALS["other"])]
    for b in _type_bonus(project):
        goals.append(b)
    goals.sort(key=lambda g: g["weight"], reverse=True)
    top = goals[:3]
    base = sum(g["weight"] for g in top) / len(top) if top else 50
    score = round(base)
    if getattr(project, "design_completeness", 0) and project.design_completeness >= 0.7:
        score = min(100, score + 5)
    score = max(5, min(100, score))
    band = "High" if score >= 75 else ("Moderate" if score >= 50 else "Low")
    ndg_list = []
    for g in top:
        name = ndg_name(g["code"]) or ""
        ndg_list.append({
            "code": g["code"], "name": name, "weight": g["weight"],
            "rationale": g["rationale"],
        })
    summary = (
        "This %s project in the %s sector maps onto the National Development "
        "Goals most strongly through %s. Inferential alignment is estimated at "
        "%d/100 (%s) under Vision 2030; it is a decision-support judgement and "
        "should not be read as an official development-plan endorsement."
        % (project.project_type or "capital", sector,
           ndg_list[0]["code"] + " (" + ndg_list[0]["name"] + ")" if ndg_list else "n/a",
           score, band))
    return {
        "sector": sector,
        "score": score,
        "band": band,
        "goals": ndg_list,
        "summary": summary,
        "pillar": NDS2_PILLAR,
    }


def alignment_markdown(al: Dict[str, Any]) -> str:
    lines = ["VISION 2030 ALIGNMENT", ""]
    lines.append("Underlying framework: %s" % al["pillar"])
    lines.append("")
    lines.append("Estimated alignment: %d/100 (%s alignment)" % (al["score"], al["band"]))
    lines.append("")
    for i, g in enumerate(al["goals"], 1):
        lines.append("%d. %s - %s (weight %d/100)" % (i, g["code"], g["name"], g["weight"]))
    lines.append("")
    lines.append(al["summary"])
    return "\n".join(lines)


def alignment_html(al: Dict[str, Any]) -> str:
    color = {"High": "#34d399", "Moderate": "#fbbf24", "Low": "#ff6b6b"}.get(al["band"], "#fbbf24")
    bars = "".join(
        "<div style='margin-top:8px;'>"
        "<div style='font-size:.85rem;color:#e6f1ff;'>%s - %s "
        "<span style='float:right;color:#8aa2c8;'>%d/100</span></div>"
        "<div style='background:#0d1b32;border-radius:6px;height:8px;margin-top:3px;'>"
        "<div style='background:%s;height:8px;border-radius:6px;width:%d%%;'></div>"
        "</div></div>" % (g["code"], g["name"], g["weight"], color, g["weight"])
        for g in al["goals"])
    return (
        "<div style='background:linear-gradient(135deg,#112240,#0d1b32);"
        "border:1px solid %s55;border-radius:14px;padding:18px 20px;'>"
        "<div style='display:flex;justify-content:space-between;align-items:center;'>"
        "<div style='font-weight:800;color:#e6f1ff;'>VISION 2030 ALIGNMENT</div>"
        "<div style='background:%s22;color:%s;border:1px solid %s55;border-radius:20px;"
        "padding:2px 12px;font-weight:700;font-size:.8rem;'>%s ALIGNMENT</div></div>"
        "<div style='color:#8aa2c8;font-size:.85rem;margin-top:4px;'>%s</div>"
        "<div style='margin-top:14px;font-size:2.4rem;font-weight:900;color:%s;'>"
        "%d<small style='font-size:.9rem;color:#8aa2c8;'>/100</small></div>"
        "%s<div style='color:#8aa2c8;font-size:.85rem;margin-top:12px;'>%s</div></div>"
        % (color, color, color, color, al["band"], al["pillar"], color, al["score"],
           bars, al["summary"]))