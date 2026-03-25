# evidence_signals.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Evidence:
    business_domain: str
    cites_official_domain: bool
    mentions_business_name: bool
    mentions_official_domain: bool
    citation_count: int
    unique_citation_domains: List[str]
    freshest_cited_days: Optional[int]
    has_services: bool
    has_location: bool
    has_contact: bool
    proof_signals: List[str] = field(default_factory=list)
    query_success_count: int = 0
    query_error_count: int = 0
    entity_status: str = "UNCLEAR"
    entity_confidence: int = 0
    unclear_rate: float = 0.0
    validator_supported_count: int = 0
    validator_total_count: int = 0


@dataclass
class Signals:
    # canonical source
    official_source_missing: bool

    # authority diversity / dependency
    low_domain_diversity: bool
    high_domain_diversity: bool
    authority_dependency_risk: bool

    # freshness
    freshness_unknown: bool
    freshness_stale: bool
    freshness_recent: bool

    # coverage
    missing_services: bool
    missing_location: bool
    missing_contact: bool
    proof_points_missing: bool = False
    thin_evidence: bool = False
    partial_results: bool = False
    confidence_limited: bool = False

    # discovery basics
    mentions_business_name: bool = False
    mentions_official_domain: bool = False
    cites_official_domain: bool = False


def build_evidence(metrics: Dict[str, Any]) -> Evidence:
    comp = metrics.get("comprehensiveness") or {}
    return Evidence(
        business_domain=str(metrics.get("business_domain") or ""),
        mentions_business_name=bool(metrics.get("mentions_business_name")),
        mentions_official_domain=bool(metrics.get("mentions_official_domain")),
        cites_official_domain=bool(metrics.get("cites_official_domain")),
        citation_count=int(metrics.get("citation_count") or 0),
        unique_citation_domains=list(metrics.get("unique_citation_domains") or []),
        freshest_cited_days=metrics.get("freshest_cited_days"),
        has_services=bool(comp.get("has_services")),
        has_location=bool(comp.get("has_location")),
        has_contact=bool(comp.get("has_contact")),
        proof_signals=list(metrics.get("proof_signals") or []),
        query_success_count=int((metrics.get("query_health") or {}).get("success_count") or 0),
        query_error_count=int((metrics.get("query_health") or {}).get("error_count") or 0),
        entity_status=str((metrics.get("entity_identity") or {}).get("status") or "UNCLEAR"),
        entity_confidence=int((metrics.get("entity_identity") or {}).get("confidence") or 0),
        unclear_rate=float((metrics.get("entity_identity") or {}).get("unclear_rate") or 0.0),
        validator_supported_count=int((metrics.get("validator_summary") or {}).get("supported_count") or 0),
        validator_total_count=int((metrics.get("validator_summary") or {}).get("total_count") or 0),
    )


def build_signals(
    *,
    evidence: Evidence,
    authority_score: int,
) -> Signals:
    uniq = evidence.unique_citation_domains
    uniq_count = len(uniq)

    freshness_unknown = evidence.freshest_cited_days is None
    freshness_recent = (evidence.freshest_cited_days is not None) and (evidence.freshest_cited_days <= 30)
    freshness_stale = (evidence.freshest_cited_days is not None) and (evidence.freshest_cited_days > 90)
    total_queries = evidence.query_success_count + evidence.query_error_count
    partial_results = evidence.query_error_count > 0
    thin_evidence = (
        evidence.citation_count < 3 or
        uniq_count < 2 or
        evidence.query_success_count < 2 or
        evidence.entity_status in {"UNCLEAR", "MISMATCH"}
    )
    confidence_limited = (
        evidence.unclear_rate > 0.3 or
        evidence.entity_confidence < 60 or
        partial_results or
        thin_evidence or
        (total_queries > 0 and evidence.query_success_count / total_queries < 0.75)
    )

    # “dependency risk”: lots of citations but weak authority OR official not cited
    authority_dependency_risk = (
        (evidence.citation_count >= 8 and authority_score < 50) or
        (not evidence.cites_official_domain and uniq_count >= 8)
    )

    return Signals(
        official_source_missing=not evidence.cites_official_domain,

        low_domain_diversity=uniq_count < 5,
        high_domain_diversity=uniq_count >= 10,
        authority_dependency_risk=bool(authority_dependency_risk),

        freshness_unknown=bool(freshness_unknown),
        freshness_stale=bool(freshness_stale),
        freshness_recent=bool(freshness_recent),

        missing_services=not evidence.has_services,
        missing_location=not evidence.has_location,
        missing_contact=not evidence.has_contact,
        proof_points_missing=len(evidence.proof_signals) == 0,
        thin_evidence=bool(thin_evidence),
        partial_results=bool(partial_results),
        confidence_limited=bool(confidence_limited),

        mentions_business_name=bool(evidence.mentions_business_name),
        mentions_official_domain=bool(evidence.mentions_official_domain),
        cites_official_domain=bool(evidence.cites_official_domain),
    )

