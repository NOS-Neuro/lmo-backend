from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


Severity = Literal["low", "medium", "high"]
Priority = Literal["P0", "P1", "P2"]
Effort = Literal["low", "medium", "high"]


class ScanMeta(BaseModel):
    scan_id: str
    timestamp_utc: datetime
    business_name: str
    website: Optional[str] = None
    requested_by: Optional[str] = None
    scoring_version: str = "ras-v2.0"


class ComponentScores(BaseModel):
    discovery: float = Field(ge=0, le=100)
    accuracy: float = Field(ge=0, le=100)
    consistency: float = Field(ge=0, le=100)
    authority: float = Field(ge=0, le=100)
    completeness: float = Field(ge=0, le=100)

    # multipliers / diagnostics (not part of the 0..100 categories)
    evidence_confidence: float = Field(default=0.75, ge=0, le=1)
    hallucination_rate: float = Field(default=0.10, ge=0, le=1)


class ScoreBand(BaseModel):
    label: str
    description: str
    confidence: str
    next_steps: str


class Finding(BaseModel):
    severity: Severity
    title: str
    summary: str
    evidence_refs: List[str] = Field(default_factory=list)


class RecommendedAction(BaseModel):
    priority: Priority
    action: str
    why: str
    expected_impact: Dict[str, Any] = Field(default_factory=dict)
    effort: Effort = "medium"
    owner_hint: Optional[str] = None


class ClientReport(BaseModel):
    overall_score: int
    score_band: ScoreBand
    category_scores: Dict[str, int]
    key_findings: List[str]
    recommended_actions: List[str]
    improvement_potential: Dict[str, Any]


class OperatorReport(BaseModel):
    methodology: Dict[str, Any]
    raw_results: Dict[str, Any]
    gap_analysis: Dict[str, Any]
    priority_fixes: List[Dict[str, Any]]
    competitive_baseline: Optional[Dict[str, Any]] = None
    risk_assessment: Dict[str, Any] = Field(default_factory=dict)


class ScanResults(BaseModel):
    meta: ScanMeta
    component_scores: ComponentScores
    overall_score: int
    score_band: ScoreBand

    findings: List[Finding] = Field(default_factory=list)
    recommended_actions: List[RecommendedAction] = Field(default_factory=list)

    # dual outputs (optional but recommended)
    client_report: Optional[ClientReport] = None
    operator_report: Optional[OperatorReport] = None

