# recommendation_rules.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal

from evidence_signals import Evidence, Signals


Priority = Literal["P0", "P1", "P2"]
RecType = Literal["fix", "maintain"]


@dataclass
class Recommendation:
    id: str
    rec_type: RecType
    priority: Priority
    category: str
    title: str
    action_steps: List[str]
    expected_impact: str
    measurable_outcome: str
    why: List[str]


@dataclass
class RecommendationBundle:
    fix_now: List[Recommendation]
    maintain: List[Recommendation]
    next_scan_focus: List[str]


def _rec(
    *,
    id: str,
    rec_type: RecType,
    priority: Priority,
    category: str,
    title: str,
    action_steps: List[str],
    expected_impact: str,
    measurable_outcome: str,
    why: List[str],
) -> Recommendation:
    return Recommendation(
        id=id,
        rec_type=rec_type,
        priority=priority,
        category=category,
        title=title,
        action_steps=action_steps,
        expected_impact=expected_impact,
        measurable_outcome=measurable_outcome,
        why=why,
    )


def build_recommendations(
    *,
    evidence: Evidence,
    signals: Signals,
    scores: Dict[str, int],
) -> RecommendationBundle:
    recs_fix: List[Recommendation] = []
    recs_maintain: List[Recommendation] = []

    discovery = int(scores.get("discovery", 0))
    accuracy = int(scores.get("accuracy", 0))
    authority = int(scores.get("authority", 0))

    # -------------------------
    # P0: Official source missing
    # -------------------------
    if signals.official_source_missing:
        recs_fix.append(_rec(
            id="canonical_official_missing",
            rec_type="fix",
            priority="P0",
            category="Canonical Source",
            title="Make the official site the canonical source AI cites",
            action_steps=[
                "Add/verify clear About + Services + Contact pages on the official site.",
                "Add Organization + LocalBusiness schema.org (name, url, logo, sameAs, address/serviceArea).",
                "Ensure major profiles (LinkedIn, Google Business Profile, industry listings) link to the official domain.",
            ],
            expected_impact="Improves Accuracy + Authority by making the official domain a primary citation source.",
            measurable_outcome="Next scan shows: cites_official_domain = true AND official domain appears in citations.",
            why=[
                "Scan evidence: official site was not cited in citations.",
                f"Business domain: {evidence.business_domain or 'unknown'}",
            ],
        ))

    # -------------------------
    # P1: Coverage gaps (services/location/contact)
    # -------------------------
    if signals.missing_services:
        recs_fix.append(_rec(
            id="coverage_services_missing",
            rec_type="fix",
            priority="P1",
            category="Coverage",
            title="Clarify services in machine-readable and human-readable form",
            action_steps=[
                "Create a ‘Services’ page with 5–10 bullet service lines (plain language).",
                "Add FAQ that answers: what you do, who you serve, where you serve, and constraints.",
                "Add Service schema markup where appropriate.",
            ],
            expected_impact="Improves Accuracy and reduces hallucinated service descriptions.",
            measurable_outcome="Next scan shows has_services = true (comprehensiveness score increases).",
            why=["Scan evidence: services signal was missing/unclear in answers."],
        ))

    if signals.missing_location:
        recs_fix.append(_rec(
            id="coverage_location_missing",
            rec_type="fix",
            priority="P1",
            category="Coverage",
            title="Remove geographic ambiguity",
            action_steps=[
                "Add a ‘Where we operate’ section (regions/cities/countries).",
                "If multiple sites: create Locations page + consistent address formatting.",
                "Add LocalBusiness schema with address/serviceArea fields.",
            ],
            expected_impact="Improves Discovery and Accuracy for location-specific queries.",
            measurable_outcome="Next scan shows has_location = true.",
            why=["Scan evidence: location signal was missing/unclear in answers."],
        ))

    if signals.missing_contact:
        recs_fix.append(_rec(
            id="coverage_contact_missing",
            rec_type="fix",
            priority="P1",
            category="Conversion",
            title="Make the contact path unambiguous for AI and users",
            action_steps=[
                "Ensure Contact page includes phone/email/form and is linked in nav/footer.",
                "Add ‘ContactPoint’ schema (email/telephone/contactType).",
                "Add ‘Request a quote’ page if B2B sales-driven.",
            ],
            expected_impact="Improves conversion and reduces ‘unclear contact’ outputs.",
            measurable_outcome="Next scan shows has_contact = true.",
            why=["Scan evidence: contact path was unclear or missing."],
        ))

    # -------------------------
    # P1: Authority dependency risk
    # -------------------------
    if signals.authority_dependency_risk:
        recs_fix.append(_rec(
            id="authority_dependency_risk",
            rec_type="fix",
            priority="P1",
            category="Authority",
            title="Reduce dependency on weak third-party sources",
            action_steps=[
                "Create 2–3 authoritative profiles that cite the official domain (LinkedIn, Crunchbase, association directory).",
                "Publish 1 authority asset (case study, certification page, capability sheet) on the official site.",
                "Correct or remove inaccurate directory listings that outrank/cite competitors.",
            ],
            expected_impact="Improves Authority score stability and reduces misattribution.",
            measurable_outcome="Next scan shows higher authority score and/or more high-quality domains citing official site.",
            why=[
                "Scan evidence suggests high reliance on non-official domains or weak authority score relative to citations."
            ],
        ))

    # -------------------------
    # P2: Freshness
    # -------------------------
    if signals.freshness_unknown:
        recs_fix.append(_rec(
            id="freshness_unknown",
            rec_type="fix",
            priority="P2",
            category="Freshness",
            title="Increase freshness signals (and make updates visible)",
            action_steps=[
                "Add a News/Updates page or recent posts section (even quarterly).",
                "Ensure key profiles show updated info (hours, services, locations).",
                "Publish a dated page for major updates (new facility, new capabilities, certifications).",
            ],
            expected_impact="Improves freshness confidence and reduces stale summaries.",
            measurable_outcome="Future scans return citation dates or show fresher cited sources.",
            why=["Scan evidence: citation dates were not provided by sources."],
        ))
    elif signals.freshness_stale:
        recs_fix.append(_rec(
            id="freshness_stale",
            rec_type="fix",
            priority="P2",
            category="Freshness",
            title="Update and re-seed sources so citations become recent",
            action_steps=[
                "Update official pages and profiles with a visible timestamp where appropriate.",
                "Publish one new authoritative page and share it on an indexed platform (LinkedIn/company page).",
            ],
            expected_impact="Improves freshness scoring and confidence in current details.",
            measurable_outcome="Next scan shows freshest_cited_days <= 30.",
            why=[f"Scan evidence: freshest cited source ~{evidence.freshest_cited_days} days old."],
        ))

    # -------------------------
    # Maintain rules (if already strong)
    # -------------------------
    if authority >= 70:
        recs_maintain.append(_rec(
            id="maintain_authority",
            rec_type="maintain",
            priority="P2",
            category="Maintain",
            title="Monitor authoritative sources for drift",
            action_steps=[
                "Quarterly re-scan and compare deltas in citation domains + scores.",
                "Watch for incorrect directory listings and stale profiles.",
            ],
            expected_impact="Prevents score regression as sources change.",
            measurable_outcome="Authority score remains stable across scans.",
            why=["Authority score is already strong; focus is stability."],
        ))

    if discovery >= 70 and accuracy >= 70:
        recs_maintain.append(_rec(
            id="maintain_baseline",
            rec_type="maintain",
            priority="P2",
            category="Maintain",
            title="Lock a baseline Truth File to prevent future drift",
            action_steps=[
                "Create a canonical Truth File (Name, services, locations, contact, certifications, keywords).",
                "Re-run scans after major website changes.",
            ],
            expected_impact="Improves long-term consistency and reduces drift risk.",
            measurable_outcome="Reduced variance in scores month to month.",
            why=["Discovery/Accuracy are already healthy; stability is the goal."],
        ))

    # Next scan focus (pick top 1–3)
    focus: List[str] = []
    if signals.official_source_missing:
        focus.append("Get official domain cited")
    if signals.missing_services:
        focus.append("Clarify services")
    if signals.missing_location:
        focus.append("Clarify operating regions")
    if signals.missing_contact:
        focus.append("Clarify contact path")
    if signals.authority_dependency_risk:
        focus.append("Increase authoritative profiles that cite official domain")
    if signals.freshness_stale or signals.freshness_unknown:
        focus.append("Improve freshness signals")

    return RecommendationBundle(
        fix_now=recs_fix[:8],
        maintain=recs_maintain[:6],
        next_scan_focus=focus[:3],
    )
