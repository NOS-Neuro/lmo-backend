# evidence_signals.py
from __future__ import annotations

from dataclasses import dataclass
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

    # discovery basics
    mentions_business_name: bool
    mentions_official_domain: bool
    cites_official_domain: bool


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

        mentions_business_name=bool(evidence.mentions_business_name),
        mentions_official_domain=bool(evidence.mentions_official_domain),
        cites_official_domain=bool(evidence.cites_official_domain),
    )

