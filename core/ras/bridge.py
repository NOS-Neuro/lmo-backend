from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.ras.models import ScanMeta, ScanResults, ComponentScores, Finding, RecommendedAction
from core.ras.scoring import calculate_overall, interpret
from core.ras.report import generate_client_report, generate_operator_report


def build_ras_scan_results(
    *,
    business_name: str,
    website: Optional[str],
    component_scores: ComponentScores,
    findings: List[Finding],
    recommended_actions: List[RecommendedAction],
    operator_raw: Dict[str, Any],
    requested_by: Optional[str] = None,
    scan_id: Optional[str] = None,
) -> ScanResults:
    meta = ScanMeta(
        scan_id=scan_id or "",
        business_name=business_name,
        website=website,
        requested_by=requested_by,
        timestamp_utc=datetime.now(timezone.utc),
    )

    overall = calculate_overall(component_scores)
    band = interpret(overall)

    scan = ScanResults(
        meta=meta,
        component_scores=component_scores,
        overall_score=overall,
        score_band=band,
        findings=findings,
        recommended_actions=recommended_actions,
    )

    scan.client_report = generate_client_report(scan)
    scan.operator_report = generate_operator_report(scan, operator_raw)  # dict ok

    return scan

