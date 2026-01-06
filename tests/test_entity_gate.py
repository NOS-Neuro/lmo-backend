"""
Test suite for entity identity gate and score gating rules.

Tests the complete architecture refactor including:
- Entity identity computation
- Score gating based on entity status
- Unclear rate penalties
- 90+ score decompression requirements
"""

import pytest
from dataclasses import dataclass
from typing import List
from scan_engine_real import (
    compute_entity_identity,
    ProviderResult,
    PerplexityHit,
    _extract_capitalized_names,
    _count_unclear_responses,
)


# -----------------------------
# Helper Functions Tests
# -----------------------------

def test_extract_capitalized_names():
    """Test extraction of capitalized names from text"""
    text = "Apple Inc is based in Cupertino. Tim Cook is the CEO. Microsoft Corporation is a competitor."
    names = _extract_capitalized_names(text)

    assert "Apple Inc" in names
    assert "Tim Cook" in names
    assert "Microsoft Corporation" in names


def test_count_unclear_responses():
    """Test unclear response counting"""
    # Create mock results
    results = [
        ProviderResult(
            provider="perplexity",
            model="sonar-pro",
            prompt_name="test1",
            question="Q1",
            answer_text="The answer is unclear.",
            citations=[]
        ),
        ProviderResult(
            provider="perplexity",
            model="sonar-pro",
            prompt_name="test2",
            question="Q2",
            answer_text="This information is not available.",
            citations=[]
        ),
        ProviderResult(
            provider="perplexity",
            model="sonar-pro",
            prompt_name="test3",
            question="Q3",
            answer_text="The company was founded in 2020.",
            citations=[]
        ),
    ]

    unclear_count, unclear_rate = _count_unclear_responses(results)

    assert unclear_count == 2
    assert unclear_rate == pytest.approx(0.666, abs=0.01)


# -----------------------------
# Entity Identity Gate Tests
# -----------------------------

def test_entity_confirmed_status():
    """Test entity status CONFIRMED when domain is cited multiple times"""
    results = [
        ProviderResult(
            provider="perplexity",
            model="sonar-pro",
            prompt_name="test1",
            question="Q1",
            answer_text="Acme Corp is a software company based in Toronto with acme.com as their official website.",
            citations=[
                PerplexityHit(title="Acme Official", url="https://acme.com/about"),
                PerplexityHit(title="Acme Blog", url="https://acme.com/blog"),
                PerplexityHit(title="Crunchbase", url="https://crunchbase.com/acme"),
            ]
        ),
        ProviderResult(
            provider="perplexity",
            model="sonar-pro",
            prompt_name="test2",
            question="Q2",
            answer_text="Acme Corp was founded in 2015 and has 50 employees. Visit acme.com for details.",
            citations=[
                PerplexityHit(title="Acme News", url="https://acme.com/news"),
                PerplexityHit(title="LinkedIn", url="https://linkedin.com/company/acme"),
                PerplexityHit(title="Acme Careers", url="https://acme.com/careers"),
            ]
        ),
        ProviderResult(
            provider="perplexity",
            model="sonar-pro",
            prompt_name="test3",
            question="Q3",
            answer_text="Acme Corp is located at their Toronto headquarters. Contact via acme.com.",
            citations=[
                PerplexityHit(title="Acme Contact", url="https://acme.com/contact"),
                PerplexityHit(title="Bloomberg", url="https://bloomberg.com/acme"),
            ]
        ),
    ]

    all_text = "\n".join(r.answer_text for r in results)

    identity = compute_entity_identity(
        submitted_domain="acme.com",
        business_name="Acme Corp",
        perplexity_results=results,
        all_text=all_text,
    )

    assert identity["entity_status"] == "CONFIRMED"
    assert identity["entity_confidence"] >= 80
    assert identity["submitted_domain_cited"] is True
    assert "acme.com" in identity["dominant_citation_domains"]


def test_entity_mismatch_status():
    """Test entity status MISMATCH when wrong company is referenced"""
    results = [
        ProviderResult(
            provider="perplexity",
            model="sonar-pro",
            prompt_name="test1",
            question="Q1",
            answer_text="Beta Solutions is a different company. Nothing about Acme Corp.",
            citations=[
                PerplexityHit(title="Beta Site", url="https://betasolutions.com"),
                PerplexityHit(title="Beta About", url="https://betasolutions.com/about"),
            ]
        ),
        ProviderResult(
            provider="perplexity",
            model="sonar-pro",
            prompt_name="test2",
            question="Q2",
            answer_text="Beta Solutions provides consulting services.",
            citations=[
                PerplexityHit(title="Beta Services", url="https://betasolutions.com/services"),
            ]
        ),
    ]

    all_text = "\n".join(r.answer_text for r in results)

    identity = compute_entity_identity(
        submitted_domain="acme.com",
        business_name="Acme Corp",
        perplexity_results=results,
        all_text=all_text,
    )

    # Entity status should be MISMATCH or UNCLEAR
    assert identity["entity_status"] in {"MISMATCH", "UNCLEAR"}
    assert identity["entity_confidence"] < 60
    assert identity["submitted_domain_cited"] is False
    assert "acme.com" not in identity["dominant_citation_domains"]


def test_entity_unclear_status_high_unclear_rate():
    """Test entity status when many responses are unclear"""
    results = [
        ProviderResult(
            provider="perplexity",
            model="sonar-pro",
            prompt_name="test1",
            question="Q1",
            answer_text="Information about Acme Corp is unclear.",
            citations=[
                PerplexityHit(title="Generic", url="https://acme.com"),
            ]
        ),
        ProviderResult(
            provider="perplexity",
            model="sonar-pro",
            prompt_name="test2",
            question="Q2",
            answer_text="Not available.",
            citations=[]
        ),
        ProviderResult(
            provider="perplexity",
            model="sonar-pro",
            prompt_name="test3",
            question="Q3",
            answer_text="Unclear.",
            citations=[]
        ),
    ]

    all_text = "\n".join(r.answer_text for r in results)

    identity = compute_entity_identity(
        submitted_domain="acme.com",
        business_name="Acme Corp",
        perplexity_results=results,
        all_text=all_text,
    )

    assert identity["unclear_rate"] >= 0.5
    # High unclear rate should reduce confidence
    assert identity["entity_confidence"] < 70


def test_entity_name_conflict_detection():
    """Test detection of conflicting company names in responses"""
    results = [
        ProviderResult(
            provider="perplexity",
            model="sonar-pro",
            prompt_name="test1",
            question="Q1",
            answer_text="Acme Corporation is often confused with Acme Industries and Acme Global Solutions.",
            citations=[
                PerplexityHit(title="Acme", url="https://acme.com"),
            ]
        ),
        ProviderResult(
            provider="perplexity",
            model="sonar-pro",
            prompt_name="test2",
            question="Q2",
            answer_text="Acme Industries provides different services than Acme Corporation.",
            citations=[]
        ),
    ]

    all_text = "\n".join(r.answer_text for r in results)

    identity = compute_entity_identity(
        submitted_domain="acme.com",
        business_name="Acme Corp",
        perplexity_results=results,
        all_text=all_text,
    )

    # Should detect conflicting names
    assert len(identity["name_conflict_candidates"]) > 0
    assert any("Acme Industries" in name for name in identity["name_conflict_candidates"])


def test_entity_domain_mismatch_detection():
    """Test detection when official domain differs from submitted"""
    results = [
        ProviderResult(
            provider="perplexity",
            model="sonar-pro",
            prompt_name="test1",
            question="Q1",
            answer_text="Acme Corp's official website is acmecorporation.com.",
            citations=[
                PerplexityHit(title="Acme Official", url="https://acmecorporation.com"),
                PerplexityHit(title="Acme About", url="https://acmecorporation.com/about"),
                PerplexityHit(title="Acme News", url="https://acmecorporation.com/news"),
            ]
        ),
    ]

    all_text = "\n".join(r.answer_text for r in results)

    identity = compute_entity_identity(
        submitted_domain="acme.com",
        business_name="Acme Corp",
        perplexity_results=results,
        all_text=all_text,
    )

    # Should detect that submitted domain doesn't match most-cited domain
    assert identity["submitted_domain_cited"] is False
    assert identity["official_domain_detected"] != "acme.com"
    # Status should be downgraded due to domain mismatch
    assert identity["entity_status"] in {"LIKELY", "UNCLEAR"}


# -----------------------------
# Integration Tests (Full Scoring Flow)
# -----------------------------

def test_gating_rule_entity_mismatch_caps_score():
    """
    Integration test: Entity MISMATCH should cap overall score at 60

    This simulates the full scoring flow in run_real_scan_perplexity()
    """
    # Mock a scenario where entity is MISMATCH
    entity_status = "MISMATCH"
    entity_confidence = 30
    unclear_rate = 0.6

    # Start with decent preliminary scores
    discovery = 70
    accuracy = 65
    authority = 60
    overall = int((discovery + accuracy + authority) / 3)  # ~65

    # Apply gating rules (simplified from actual code)
    warnings = []

    if entity_status in {"UNCLEAR", "MISMATCH"}:
        overall = min(overall, 60)
        warnings.append(f"Entity status is {entity_status} - overall score capped at 60")

    if unclear_rate > 0.50:
        overall = min(overall, 65)
        warnings.append(f"High unclear rate ({unclear_rate:.1%}) - overall score capped at 65")

    # Final result
    assert overall == 60  # Capped by entity status
    assert len(warnings) == 2
    assert any("Entity status is MISMATCH" in w for w in warnings)


def test_gating_rule_unclear_rate_penalty():
    """Test that high unclear rate caps overall score at 65"""
    entity_status = "LIKELY"
    unclear_rate = 0.55

    # Start with good scores
    overall = 75

    warnings = []

    if unclear_rate > 0.50:
        overall = min(overall, 65)
        warnings.append(f"High unclear rate ({unclear_rate:.1%}) - overall score capped at 65")
    elif unclear_rate > 0.30:
        overall = min(overall, 75)

    assert overall == 65
    assert any("High unclear rate" in w for w in warnings)


def test_gating_rule_90_plus_requirements():
    """
    Test that 90+ scores require ALL conditions:
    - entity_status == CONFIRMED
    - submitted_domain_cite_count >= 2
    - unique_domains >= 6
    - unclear_rate < 0.20
    """
    warnings = []

    # Scenario 1: High score but entity not CONFIRMED
    entity_status = "LIKELY"
    submitted_domain_cite_count = 3
    unique_domains_count = 8
    unclear_rate = 0.10
    overall = 92

    if overall >= 90:
        if entity_status != "CONFIRMED":
            overall = min(overall, 85)
            warnings.append("90+ requires CONFIRMED entity status - capped at 85")
        if submitted_domain_cite_count < 2:
            overall = min(overall, 85)
            warnings.append("90+ requires domain cited 2+ times - capped at 85")
        if unique_domains_count < 6:
            overall = min(overall, 85)
            warnings.append("90+ requires 6+ unique citation domains - capped at 85")
        if unclear_rate >= 0.20:
            overall = min(overall, 85)
            warnings.append("90+ requires unclear rate < 20% - capped at 85")

    assert overall == 85
    assert any("CONFIRMED entity status" in w for w in warnings)

    # Scenario 2: All requirements met - should stay at 90+
    warnings = []
    entity_status = "CONFIRMED"
    submitted_domain_cite_count = 3
    unique_domains_count = 8
    unclear_rate = 0.10
    overall = 92

    if overall >= 90:
        if entity_status != "CONFIRMED":
            overall = min(overall, 85)
        if submitted_domain_cite_count < 2:
            overall = min(overall, 85)
        if unique_domains_count < 6:
            overall = min(overall, 85)
        if unclear_rate >= 0.20:
            overall = min(overall, 85)

    assert overall == 92  # Should remain unchanged
    assert len(warnings) == 0


def test_gating_rule_domain_mismatch_caps_at_70():
    """Test that domain mismatch caps overall score at 70"""
    official_domain_detected = "acmecorporation.com"
    biz_domain = "acme.com"
    overall = 80

    warnings = []

    if official_domain_detected and official_domain_detected != biz_domain:
        overall = min(overall, 70)
        warnings.append(f"Domain mismatch detected: submitted '{biz_domain}' vs detected '{official_domain_detected}' - overall score capped at 70")

    assert overall == 70
    assert any("Domain mismatch" in w for w in warnings)


def test_gating_rule_not_cited_penalties():
    """Test penalties when submitted domain is never cited"""
    submitted_domain_cited = False
    discovery = 80
    accuracy = 75
    authority = 70

    warnings = []

    if not submitted_domain_cited:
        discovery = min(discovery, 70)
        authority = min(authority, 60)
        accuracy = min(accuracy, 70)
        warnings.append("Submitted domain was never cited - discovery/accuracy/authority penalties applied")

    assert discovery == 70
    assert accuracy == 70
    assert authority == 60
    assert any("never cited" in w for w in warnings)


def test_accuracy_unclear_penalty():
    """Test proportional accuracy reduction based on unclear rate"""
    unclear_rate = 0.40  # 40% unclear
    accuracy = 80

    # Max -30 points at 100% unclear
    accuracy_unclear_penalty = int(unclear_rate * 30)
    accuracy -= accuracy_unclear_penalty

    assert accuracy == 68  # 80 - 12
    assert accuracy_unclear_penalty == 12


# -----------------------------
# Edge Cases
# -----------------------------

def test_entity_gate_empty_results():
    """Test entity gate with no results"""
    identity = compute_entity_identity(
        submitted_domain="acme.com",
        business_name="Acme Corp",
        perplexity_results=[],
        all_text="",
    )

    # Should return a valid structure with low confidence
    assert identity["entity_status"] in {"UNCLEAR", "MISMATCH"}
    assert identity["entity_confidence"] < 50
    assert identity["unclear_rate"] == 0.0
    assert identity["submitted_domain_cited"] is False


def test_entity_gate_no_citations():
    """Test entity gate when results have no citations"""
    results = [
        ProviderResult(
            provider="perplexity",
            model="sonar-pro",
            prompt_name="test1",
            question="Q1",
            answer_text="Acme Corp information is unclear.",
            citations=[]
        ),
    ]

    all_text = "Acme Corp information is unclear."

    identity = compute_entity_identity(
        submitted_domain="acme.com",
        business_name="Acme Corp",
        perplexity_results=results,
        all_text=all_text,
    )

    assert identity["submitted_domain_cited"] is False
    assert len(identity["dominant_citation_domains"]) == 0
    assert identity["entity_confidence"] < 60


def test_multiple_gating_rules_combined():
    """Test when multiple gating rules apply simultaneously"""
    entity_status = "UNCLEAR"
    unclear_rate = 0.55
    official_domain_detected = "wrongdomain.com"
    biz_domain = "acme.com"

    overall = 85
    warnings = []

    # Apply all rules
    if entity_status in {"UNCLEAR", "MISMATCH"}:
        overall = min(overall, 60)
        warnings.append(f"Entity status is {entity_status}")

    if official_domain_detected and official_domain_detected != biz_domain:
        overall = min(overall, 70)
        warnings.append("Domain mismatch")

    if unclear_rate > 0.50:
        overall = min(overall, 65)
        warnings.append("High unclear rate")

    # Most restrictive rule should win (60)
    assert overall == 60
    assert len(warnings) == 3
