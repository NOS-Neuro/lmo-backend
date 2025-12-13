from __future__ import annotations

from typing import Any, Dict, List

from core.ras.models import ClientReport, ScanResults


def generate_client_report(scan: ScanResults) -> ClientReport:
    cs = scan.component_scores

    category_scores = {
        "discovery": int(round(cs.discovery)),
        "accuracy": int(round(cs.accuracy)),
        "consistency": int(round(cs.consistency)),
        "authority": int(round(cs.authority)),
        "completeness": int(round(cs.completeness)),
    }

    key_findings: List[str] = []
    for f in scan.findings[:3]:
        key_findings.append(f"{f.title}: {f.summary}")

    recommended: List[str] = []
    for a in scan.recommended_actions[:4]:
        recommended.append(f"[{a.priority}] {a.action}")

    improvement_potential = {
        "evidence_confidence": cs.evidence_confidence,
        "hallucination_rate_est": cs.hallucination_rate,
        "biggest_gap": min(category_scores, key=lambda k: category_scores[k]) if category_scores else None,
    }

    return ClientReport(
        overall_score=scan.overall_score,
        score_band=scan.score_band,
        category_scores=category_scores,
        key_findings=key_findings,
        recommended_actions=recommended,
        improvement_potential=improvement_potential,
    )


def generate_operator_report(scan: ScanResults, operator_raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep operator report flexible (dict) to avoid breaking changes.
    """
    return {
        "methodology": operator_raw.get("methodology", {}),
        "raw_results": operator_raw.get("raw_results", {}),
        "gap_analysis": operator_raw.get("gap_analysis", {}),
        "priority_fixes": [
            {
                "priority": a.priority,
                "action": a.action,
                "why": a.why,
                "effort": a.effort,
                "owner_hint": a.owner_hint,
            }
            for a in scan.recommended_actions[:10]
        ],
        "competitive_baseline": operator_raw.get("competitive_baseline"),
        "risk_assessment": operator_raw.get("risk_assessment", {}),
    }

