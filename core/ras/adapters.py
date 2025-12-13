from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.ras.models import ComponentScores
from evidence_signals import Evidence, Signals


def clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def compute_evidence_confidence(e: Evidence) -> float:
    """
    0..1 confidence multiplier.
    Deterministic: based on coverage breadth + whether official domain is present.
    """
    coverage = 0
    coverage += 1 if e.mentions_business_name else 0
    coverage += 1 if e.mentions_official_domain else 0
    coverage += 1 if e.cites_official_domain else 0
    coverage += 1 if e.has_services else 0
    coverage += 1 if e.has_location else 0
    coverage += 1 if e.has_contact else 0

    # Base confidence from coverage (0..6 -> 0.55..0.95)
    base = 0.55 + (coverage / 6.0) * 0.40

    # Bonus for some citation diversity
    uniq = len(e.unique_citation_domains)
    if uniq >= 10:
        base += 0.05
    elif uniq >= 5:
        base += 0.02

    # Penalty if official domain is missing from citations (big deal)
    if not e.cites_official_domain:
        base -= 0.10

    return max(0.50, min(1.0, base))


def compute_hallucination_rate(
    *,
    authority_dependency_risk: bool,
    official_source_missing: bool,
) -> float:
    """
    0..1 fraction. In v0.1 we can only approximate based on risk flags.
    Later: compute from claim-level verification.
    """
    rate = 0.05
    if official_source_missing:
        rate += 0.10
    if authority_dependency_risk:
        rate += 0.10
    return min(0.35, rate)


def scores_from_evidence_and_signals(
    *,
    evidence: Evidence,
    signals: Signals,
    base_discovery: float,
    base_accuracy: float,
    base_authority: float,
    base_completeness: float,
    base_consistency: float,
) -> ComponentScores:
    """
    Takes your existing metric estimates + applies deterministic adjustments from Signals.
    This keeps the system stable (no random swings), and makes reasons explainable.
    """

    discovery = float(base_discovery)
    accuracy = float(base_accuracy)
    authority = float(base_authority)
    completeness = float(base_completeness)
    consistency = float(base_consistency)

    # ---------- Discovery ----------
    if signals.official_source_missing:
        discovery -= 8
    if signals.low_domain_diversity:
        discovery -= 4
    if signals.high_domain_diversity:
        discovery += 3

    # ---------- Completeness ----------
    if signals.missing_services:
        completeness -= 12
    if signals.missing_location:
        completeness -= 10
    if signals.missing_contact:
        completeness -= 10

    # ---------- Authority ----------
    if signals.authority_dependency_risk:
        authority -= 8
    if signals.high_domain_diversity:
        authority += 5

    # ---------- Freshness affects Consistency + Authority ----------
    if signals.freshness_unknown:
        consistency -= 4
    if signals.freshness_stale:
        consistency -= 6
        authority -= 4
    if signals.freshness_recent:
        consistency += 3
        authority += 2

    # ---------- Accuracy + Consistency: official grounding ----------
    if not signals.cites_official_domain:
        accuracy -= 8
        consistency -= 6
    if signals.mentions_official_domain and signals.mentions_business_name:
        accuracy += 2

    # Clamp 0..100
    discovery = clamp(discovery)
    accuracy = clamp(accuracy)
    authority = clamp(authority)
    completeness = clamp(completeness)
    consistency = clamp(consistency)

    evidence_confidence = compute_evidence_confidence(evidence)
    hallucination_rate = compute_hallucination_rate(
        authority_dependency_risk=signals.authority_dependency_risk,
        official_source_missing=signals.official_source_missing,
    )

    return ComponentScores(
        discovery=discovery,
        accuracy=accuracy,
        consistency=consistency,
        authority=authority,
        completeness=completeness,
        evidence_confidence=evidence_confidence,
        hallucination_rate=hallucination_rate,
    )
