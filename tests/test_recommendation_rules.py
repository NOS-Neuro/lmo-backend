"""
Unit tests for recommendation_rules.py

Tests cover:
- Recommendation dataclass
- build_recommendations() function
- P0 recommendations (official source missing)
- P1 recommendations (coverage gaps, authority risk)
- P2 recommendations (freshness)
- Maintain recommendations (high scores)
- Next scan focus
"""
import pytest
from typing import List

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evidence_signals import Evidence, Signals
from recommendation_rules import (
    Recommendation,
    RecommendationBundle,
    build_recommendations,
)


class TestBuildRecommendationsP0:
    """Test P0 (critical) recommendations"""

    def test_official_source_missing_triggers_p0(self):
        """Missing official source triggers P0 recommendation"""
        evidence = Evidence(
            business_domain="example.com",
            cites_official_domain=False,  # NOT cited
            mentions_business_name=True,
            mentions_official_domain=True,
            citation_count=5,
            unique_citation_domains=["domain1.com"],
            freshest_cited_days=20,
            has_services=True,
            has_location=True,
            has_contact=True,
        )

        signals = Signals(
            official_source_missing=True,  # Triggers P0
            low_domain_diversity=False,
            high_domain_diversity=False,
            authority_dependency_risk=False,
            freshness_unknown=False,
            freshness_stale=False,
            freshness_recent=True,
            missing_services=False,
            missing_location=False,
            missing_contact=False,
            mentions_business_name=True,
            mentions_official_domain=True,
            cites_official_domain=False,
        )

        scores = {"discovery": 50, "accuracy": 50, "authority": 50}

        bundle = build_recommendations(evidence=evidence, signals=signals, scores=scores)

        # Should have P0 recommendation
        assert len(bundle.fix_now) > 0
        p0_recs = [r for r in bundle.fix_now if r.priority == "P0"]
        assert len(p0_recs) > 0
        assert p0_recs[0].id == "canonical_official_missing"
        assert "official" in p0_recs[0].title.lower()


class TestBuildRecommendationsP1:
    """Test P1 (high priority) recommendations"""

    def test_missing_services_triggers_p1(self):
        """Missing services triggers P1 recommendation"""
        evidence = Evidence(
            business_domain="example.com",
            cites_official_domain=True,
            mentions_business_name=True,
            mentions_official_domain=True,
            citation_count=5,
            unique_citation_domains=["domain1.com"],
            freshest_cited_days=20,
            has_services=False,  # Missing
            has_location=True,
            has_contact=True,
        )

        signals = Signals(
            official_source_missing=False,
            low_domain_diversity=False,
            high_domain_diversity=False,
            authority_dependency_risk=False,
            freshness_unknown=False,
            freshness_stale=False,
            freshness_recent=True,
            missing_services=True,  # Triggers P1
            missing_location=False,
            missing_contact=False,
            mentions_business_name=True,
            mentions_official_domain=True,
            cites_official_domain=True,
        )

        scores = {"discovery": 50, "accuracy": 50, "authority": 50}

        bundle = build_recommendations(evidence=evidence, signals=signals, scores=scores)

        # Should have P1 recommendation for services
        p1_services = [r for r in bundle.fix_now if r.id == "coverage_services_missing"]
        assert len(p1_services) > 0
        assert p1_services[0].priority == "P1"
        assert "services" in p1_services[0].title.lower()

    def test_missing_location_triggers_p1(self):
        """Missing location triggers P1 recommendation"""
        evidence = Evidence(
            business_domain="example.com",
            cites_official_domain=True,
            mentions_business_name=True,
            mentions_official_domain=True,
            citation_count=5,
            unique_citation_domains=["domain1.com"],
            freshest_cited_days=20,
            has_services=True,
            has_location=False,  # Missing
            has_contact=True,
        )

        signals = Signals(
            official_source_missing=False,
            low_domain_diversity=False,
            high_domain_diversity=False,
            authority_dependency_risk=False,
            freshness_unknown=False,
            freshness_stale=False,
            freshness_recent=True,
            missing_services=False,
            missing_location=True,  # Triggers P1
            missing_contact=False,
            mentions_business_name=True,
            mentions_official_domain=True,
            cites_official_domain=True,
        )

        scores = {"discovery": 50, "accuracy": 50, "authority": 50}

        bundle = build_recommendations(evidence=evidence, signals=signals, scores=scores)

        # Should have P1 recommendation for location
        p1_location = [r for r in bundle.fix_now if r.id == "coverage_location_missing"]
        assert len(p1_location) > 0
        assert p1_location[0].priority == "P1"
        assert "location" in p1_location[0].title.lower() or "geographic" in p1_location[0].title.lower()

    def test_missing_contact_triggers_p1(self):
        """Missing contact triggers P1 recommendation"""
        evidence = Evidence(
            business_domain="example.com",
            cites_official_domain=True,
            mentions_business_name=True,
            mentions_official_domain=True,
            citation_count=5,
            unique_citation_domains=["domain1.com"],
            freshest_cited_days=20,
            has_services=True,
            has_location=True,
            has_contact=False,  # Missing
        )

        signals = Signals(
            official_source_missing=False,
            low_domain_diversity=False,
            high_domain_diversity=False,
            authority_dependency_risk=False,
            freshness_unknown=False,
            freshness_stale=False,
            freshness_recent=True,
            missing_services=False,
            missing_location=False,
            missing_contact=True,  # Triggers P1
            mentions_business_name=True,
            mentions_official_domain=True,
            cites_official_domain=True,
        )

        scores = {"discovery": 50, "accuracy": 50, "authority": 50}

        bundle = build_recommendations(evidence=evidence, signals=signals, scores=scores)

        # Should have P1 recommendation for contact
        p1_contact = [r for r in bundle.fix_now if r.id == "coverage_contact_missing"]
        assert len(p1_contact) > 0
        assert p1_contact[0].priority == "P1"
        assert "contact" in p1_contact[0].title.lower()

    def test_authority_dependency_risk_triggers_p1(self):
        """Authority dependency risk triggers P1 recommendation"""
        evidence = Evidence(
            business_domain="example.com",
            cites_official_domain=True,
            mentions_business_name=True,
            mentions_official_domain=True,
            citation_count=5,
            unique_citation_domains=["domain1.com"],
            freshest_cited_days=20,
            has_services=True,
            has_location=True,
            has_contact=True,
        )

        signals = Signals(
            official_source_missing=False,
            low_domain_diversity=False,
            high_domain_diversity=False,
            authority_dependency_risk=True,  # Triggers P1
            freshness_unknown=False,
            freshness_stale=False,
            freshness_recent=True,
            missing_services=False,
            missing_location=False,
            missing_contact=False,
            mentions_business_name=True,
            mentions_official_domain=True,
            cites_official_domain=True,
        )

        scores = {"discovery": 50, "accuracy": 50, "authority": 50}

        bundle = build_recommendations(evidence=evidence, signals=signals, scores=scores)

        # Should have P1 recommendation for authority risk
        p1_authority = [r for r in bundle.fix_now if r.id == "authority_dependency_risk"]
        assert len(p1_authority) > 0
        assert p1_authority[0].priority == "P1"
        assert "authority" in p1_authority[0].title.lower() or "dependency" in p1_authority[0].title.lower()


class TestBuildRecommendationsP2:
    """Test P2 (medium priority) recommendations"""

    def test_freshness_unknown_triggers_p2(self):
        """Unknown freshness triggers P2 recommendation"""
        evidence = Evidence(
            business_domain="example.com",
            cites_official_domain=True,
            mentions_business_name=True,
            mentions_official_domain=True,
            citation_count=5,
            unique_citation_domains=["domain1.com"],
            freshest_cited_days=None,  # Unknown
            has_services=True,
            has_location=True,
            has_contact=True,
        )

        signals = Signals(
            official_source_missing=False,
            low_domain_diversity=False,
            high_domain_diversity=False,
            authority_dependency_risk=False,
            freshness_unknown=True,  # Triggers P2
            freshness_stale=False,
            freshness_recent=False,
            missing_services=False,
            missing_location=False,
            missing_contact=False,
            mentions_business_name=True,
            mentions_official_domain=True,
            cites_official_domain=True,
        )

        scores = {"discovery": 50, "accuracy": 50, "authority": 50}

        bundle = build_recommendations(evidence=evidence, signals=signals, scores=scores)

        # Should have P2 recommendation for freshness
        p2_freshness = [r for r in bundle.fix_now if r.id == "freshness_unknown"]
        assert len(p2_freshness) > 0
        assert p2_freshness[0].priority == "P2"
        assert "freshness" in p2_freshness[0].title.lower()

    def test_freshness_stale_triggers_p2(self):
        """Stale freshness triggers P2 recommendation"""
        evidence = Evidence(
            business_domain="example.com",
            cites_official_domain=True,
            mentions_business_name=True,
            mentions_official_domain=True,
            citation_count=5,
            unique_citation_domains=["domain1.com"],
            freshest_cited_days=120,  # Stale (> 90)
            has_services=True,
            has_location=True,
            has_contact=True,
        )

        signals = Signals(
            official_source_missing=False,
            low_domain_diversity=False,
            high_domain_diversity=False,
            authority_dependency_risk=False,
            freshness_unknown=False,
            freshness_stale=True,  # Triggers P2
            freshness_recent=False,
            missing_services=False,
            missing_location=False,
            missing_contact=False,
            mentions_business_name=True,
            mentions_official_domain=True,
            cites_official_domain=True,
        )

        scores = {"discovery": 50, "accuracy": 50, "authority": 50}

        bundle = build_recommendations(evidence=evidence, signals=signals, scores=scores)

        # Should have P2 recommendation for stale freshness
        p2_stale = [r for r in bundle.fix_now if r.id == "freshness_stale"]
        assert len(p2_stale) > 0
        assert p2_stale[0].priority == "P2"
        assert "update" in p2_stale[0].title.lower() or "re-seed" in p2_stale[0].title.lower()


class TestBuildRecommendationsMaintain:
    """Test maintain recommendations for high-performing scores"""

    def test_high_authority_triggers_maintain(self):
        """Authority >= 70 triggers maintain recommendation"""
        evidence = Evidence(
            business_domain="example.com",
            cites_official_domain=True,
            mentions_business_name=True,
            mentions_official_domain=True,
            citation_count=10,
            unique_citation_domains=[f"domain{i}.com" for i in range(10)],
            freshest_cited_days=20,
            has_services=True,
            has_location=True,
            has_contact=True,
        )

        signals = Signals(
            official_source_missing=False,
            low_domain_diversity=False,
            high_domain_diversity=True,
            authority_dependency_risk=False,
            freshness_unknown=False,
            freshness_stale=False,
            freshness_recent=True,
            missing_services=False,
            missing_location=False,
            missing_contact=False,
            mentions_business_name=True,
            mentions_official_domain=True,
            cites_official_domain=True,
        )

        scores = {"discovery": 80, "accuracy": 80, "authority": 70}  # Authority >= 70

        bundle = build_recommendations(evidence=evidence, signals=signals, scores=scores)

        # Should have maintain recommendation for authority
        authority_maintain = [r for r in bundle.maintain if r.id == "maintain_authority"]
        assert len(authority_maintain) > 0
        assert authority_maintain[0].rec_type == "maintain"
        assert "monitor" in authority_maintain[0].title.lower() or "authoritative" in authority_maintain[0].title.lower()

    def test_high_discovery_and_accuracy_triggers_maintain(self):
        """Discovery >= 70 and Accuracy >= 70 triggers baseline maintain"""
        evidence = Evidence(
            business_domain="example.com",
            cites_official_domain=True,
            mentions_business_name=True,
            mentions_official_domain=True,
            citation_count=10,
            unique_citation_domains=[f"domain{i}.com" for i in range(10)],
            freshest_cited_days=20,
            has_services=True,
            has_location=True,
            has_contact=True,
        )

        signals = Signals(
            official_source_missing=False,
            low_domain_diversity=False,
            high_domain_diversity=True,
            authority_dependency_risk=False,
            freshness_unknown=False,
            freshness_stale=False,
            freshness_recent=True,
            missing_services=False,
            missing_location=False,
            missing_contact=False,
            mentions_business_name=True,
            mentions_official_domain=True,
            cites_official_domain=True,
        )

        scores = {"discovery": 70, "accuracy": 70, "authority": 60}

        bundle = build_recommendations(evidence=evidence, signals=signals, scores=scores)

        # Should have maintain recommendation for baseline
        baseline_maintain = [r for r in bundle.maintain if r.id == "maintain_baseline"]
        assert len(baseline_maintain) > 0
        assert baseline_maintain[0].rec_type == "maintain"
        assert "Truth File" in baseline_maintain[0].title or "baseline" in baseline_maintain[0].title.lower()


class TestNextScanFocus:
    """Test next_scan_focus generation"""

    def test_next_scan_focus_prioritizes_issues(self):
        """next_scan_focus includes top 3 issues"""
        evidence = Evidence(
            business_domain="example.com",
            cites_official_domain=False,  # Issue 1
            mentions_business_name=True,
            mentions_official_domain=True,
            citation_count=5,
            unique_citation_domains=["domain1.com"],
            freshest_cited_days=None,  # Issue 2
            has_services=False,  # Issue 3
            has_location=False,  # Issue 4
            has_contact=True,
        )

        signals = Signals(
            official_source_missing=True,
            low_domain_diversity=False,
            high_domain_diversity=False,
            authority_dependency_risk=False,
            freshness_unknown=True,
            freshness_stale=False,
            freshness_recent=False,
            missing_services=True,
            missing_location=True,
            missing_contact=False,
            mentions_business_name=True,
            mentions_official_domain=True,
            cites_official_domain=False,
        )

        scores = {"discovery": 40, "accuracy": 40, "authority": 40}

        bundle = build_recommendations(evidence=evidence, signals=signals, scores=scores)

        # Should have next_scan_focus with max 3 items
        assert len(bundle.next_scan_focus) <= 3
        assert len(bundle.next_scan_focus) > 0

        # Should include official domain as top priority
        assert any("official domain" in focus.lower() for focus in bundle.next_scan_focus)

    def test_next_scan_focus_max_three_items(self):
        """next_scan_focus is capped at 3 items"""
        evidence = Evidence(
            business_domain="example.com",
            cites_official_domain=False,
            mentions_business_name=True,
            mentions_official_domain=True,
            citation_count=5,
            unique_citation_domains=["domain1.com"],
            freshest_cited_days=120,  # Stale
            has_services=False,
            has_location=False,
            has_contact=False,
        )

        signals = Signals(
            official_source_missing=True,
            low_domain_diversity=False,
            high_domain_diversity=False,
            authority_dependency_risk=True,
            freshness_unknown=False,
            freshness_stale=True,
            freshness_recent=False,
            missing_services=True,
            missing_location=True,
            missing_contact=True,
            mentions_business_name=True,
            mentions_official_domain=True,
            cites_official_domain=False,
        )

        scores = {"discovery": 20, "accuracy": 20, "authority": 20}

        bundle = build_recommendations(evidence=evidence, signals=signals, scores=scores)

        # Should be capped at 3
        assert len(bundle.next_scan_focus) == 3


class TestRecommendationLimits:
    """Test that recommendations are limited to avoid overwhelming users"""

    def test_fix_now_limited_to_8(self):
        """fix_now recommendations are capped at 8"""
        # This is tested by the slice [:8] in the code
        # Hard to test without triggering all possible recommendations
        # But we can verify the structure is correct
        evidence = Evidence(
            business_domain="example.com",
            cites_official_domain=False,
            mentions_business_name=True,
            mentions_official_domain=True,
            citation_count=5,
            unique_citation_domains=["domain1.com"],
            freshest_cited_days=120,
            has_services=False,
            has_location=False,
            has_contact=False,
        )

        signals = Signals(
            official_source_missing=True,
            low_domain_diversity=False,
            high_domain_diversity=False,
            authority_dependency_risk=True,
            freshness_unknown=False,
            freshness_stale=True,
            freshness_recent=False,
            missing_services=True,
            missing_location=True,
            missing_contact=True,
            mentions_business_name=True,
            mentions_official_domain=True,
            cites_official_domain=False,
        )

        scores = {"discovery": 20, "accuracy": 20, "authority": 20}

        bundle = build_recommendations(evidence=evidence, signals=signals, scores=scores)

        # Should not exceed 8
        assert len(bundle.fix_now) <= 8

    def test_maintain_limited_to_6(self):
        """maintain recommendations are capped at 6"""
        evidence = Evidence(
            business_domain="example.com",
            cites_official_domain=True,
            mentions_business_name=True,
            mentions_official_domain=True,
            citation_count=10,
            unique_citation_domains=[f"domain{i}.com" for i in range(10)],
            freshest_cited_days=20,
            has_services=True,
            has_location=True,
            has_contact=True,
        )

        signals = Signals(
            official_source_missing=False,
            low_domain_diversity=False,
            high_domain_diversity=True,
            authority_dependency_risk=False,
            freshness_unknown=False,
            freshness_stale=False,
            freshness_recent=True,
            missing_services=False,
            missing_location=False,
            missing_contact=False,
            mentions_business_name=True,
            mentions_official_domain=True,
            cites_official_domain=True,
        )

        scores = {"discovery": 80, "accuracy": 80, "authority": 80}

        bundle = build_recommendations(evidence=evidence, signals=signals, scores=scores)

        # Should not exceed 6
        assert len(bundle.maintain) <= 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
