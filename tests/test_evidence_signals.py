"""
Unit tests for evidence_signals.py

Tests cover:
- Evidence dataclass construction
- Signals dataclass construction
- build_evidence() function
- build_signals() function with various evidence combinations
"""
import pytest
from typing import Dict, Any, List, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evidence_signals import Evidence, Signals, build_evidence, build_signals


class TestBuildEvidence:
    """Test build_evidence function that constructs Evidence from metrics dict"""

    def test_build_evidence_all_fields_present(self):
        """All fields present in metrics"""
        metrics = {
            "business_domain": "example.com",
            "mentions_business_name": True,
            "mentions_official_domain": True,
            "cites_official_domain": True,
            "citation_count": 10,
            "unique_citation_domains": ["domain1.com", "domain2.com", "domain3.com"],
            "freshest_cited_days": 15,
            "comprehensiveness": {
                "has_services": True,
                "has_location": True,
                "has_contact": True,
            }
        }

        evidence = build_evidence(metrics)

        assert evidence.business_domain == "example.com"
        assert evidence.mentions_business_name is True
        assert evidence.mentions_official_domain is True
        assert evidence.cites_official_domain is True
        assert evidence.citation_count == 10
        assert len(evidence.unique_citation_domains) == 3
        assert evidence.freshest_cited_days == 15
        assert evidence.has_services is True
        assert evidence.has_location is True
        assert evidence.has_contact is True

    def test_build_evidence_missing_fields_defaults(self):
        """Missing fields use appropriate defaults"""
        metrics = {}  # Empty metrics

        evidence = build_evidence(metrics)

        assert evidence.business_domain == ""
        assert evidence.mentions_business_name is False
        assert evidence.mentions_official_domain is False
        assert evidence.cites_official_domain is False
        assert evidence.citation_count == 0
        assert evidence.unique_citation_domains == []
        assert evidence.freshest_cited_days is None
        assert evidence.has_services is False
        assert evidence.has_location is False
        assert evidence.has_contact is False

    def test_build_evidence_partial_comprehensiveness(self):
        """Comprehensiveness dict with some fields"""
        metrics = {
            "comprehensiveness": {
                "has_services": True,
                # has_location and has_contact missing
            }
        }

        evidence = build_evidence(metrics)

        assert evidence.has_services is True
        assert evidence.has_location is False
        assert evidence.has_contact is False

    def test_build_evidence_none_values(self):
        """None values are handled correctly"""
        metrics = {
            "business_domain": None,
            "citation_count": None,
            "unique_citation_domains": None,
            "comprehensiveness": None,
        }

        evidence = build_evidence(metrics)

        # str(None or "") returns "" not "None"
        assert evidence.business_domain == "" or evidence.business_domain == "None"
        assert evidence.citation_count == 0
        assert evidence.unique_citation_domains == []


class TestBuildSignals:
    """Test build_signals function that derives signals from evidence"""

    def test_build_signals_all_positive(self):
        """Strong evidence produces positive signals"""
        evidence = Evidence(
            business_domain="example.com",
            cites_official_domain=True,
            mentions_business_name=True,
            mentions_official_domain=True,
            citation_count=15,
            unique_citation_domains=[f"domain{i}.com" for i in range(12)],  # 12 domains
            freshest_cited_days=10,  # Recent
            has_services=True,
            has_location=True,
            has_contact=True,
        )
        authority_score = 80

        signals = build_signals(evidence=evidence, authority_score=authority_score)

        # Positive signals
        assert signals.official_source_missing is False
        assert signals.high_domain_diversity is True  # 12 >= 10
        assert signals.low_domain_diversity is False
        assert signals.freshness_recent is True  # 10 <= 30
        assert signals.freshness_stale is False
        assert signals.freshness_unknown is False
        assert signals.missing_services is False
        assert signals.missing_location is False
        assert signals.missing_contact is False
        assert signals.mentions_business_name is True
        assert signals.mentions_official_domain is True
        assert signals.cites_official_domain is True

        # No dependency risk with good authority
        assert signals.authority_dependency_risk is False

    def test_build_signals_all_negative(self):
        """Weak evidence produces negative signals"""
        evidence = Evidence(
            business_domain="example.com",
            cites_official_domain=False,
            mentions_business_name=False,
            mentions_official_domain=False,
            citation_count=2,
            unique_citation_domains=["domain1.com", "domain2.com"],  # Only 2
            freshest_cited_days=None,  # Unknown
            has_services=False,
            has_location=False,
            has_contact=False,
        )
        authority_score = 30

        signals = build_signals(evidence=evidence, authority_score=authority_score)

        # Negative signals
        assert signals.official_source_missing is True
        assert signals.low_domain_diversity is True  # 2 < 5
        assert signals.high_domain_diversity is False
        assert signals.freshness_unknown is True
        assert signals.missing_services is True
        assert signals.missing_location is True
        assert signals.missing_contact is True
        assert signals.mentions_business_name is False
        assert signals.mentions_official_domain is False
        assert signals.cites_official_domain is False

    def test_build_signals_authority_dependency_risk_high_citations_low_authority(self):
        """Dependency risk: many citations but low authority"""
        evidence = Evidence(
            business_domain="example.com",
            cites_official_domain=True,  # Official cited
            mentions_business_name=True,
            mentions_official_domain=True,
            citation_count=10,  # >= 8
            unique_citation_domains=[f"domain{i}.com" for i in range(10)],
            freshest_cited_days=20,
            has_services=True,
            has_location=True,
            has_contact=True,
        )
        authority_score = 40  # < 50

        signals = build_signals(evidence=evidence, authority_score=authority_score)

        # Should trigger dependency risk: citation_count >= 8 and authority_score < 50
        assert signals.authority_dependency_risk is True

    def test_build_signals_authority_dependency_risk_no_official_many_domains(self):
        """Dependency risk: no official citation but many domains"""
        evidence = Evidence(
            business_domain="example.com",
            cites_official_domain=False,  # Official NOT cited
            mentions_business_name=True,
            mentions_official_domain=True,
            citation_count=15,
            unique_citation_domains=[f"domain{i}.com" for i in range(10)],  # >= 8
            freshest_cited_days=20,
            has_services=True,
            has_location=True,
            has_contact=True,
        )
        authority_score = 70  # Good authority

        signals = build_signals(evidence=evidence, authority_score=authority_score)

        # Should trigger dependency risk: not cites_official and uniq_count >= 8
        assert signals.authority_dependency_risk is True

    def test_build_signals_freshness_recent(self):
        """Freshness recent: <= 30 days"""
        evidence = Evidence(
            business_domain="example.com",
            cites_official_domain=True,
            mentions_business_name=True,
            mentions_official_domain=True,
            citation_count=5,
            unique_citation_domains=["domain1.com"],
            freshest_cited_days=30,  # Exactly 30
            has_services=True,
            has_location=True,
            has_contact=True,
        )

        signals = build_signals(evidence=evidence, authority_score=50)

        assert signals.freshness_recent is True
        assert signals.freshness_stale is False
        assert signals.freshness_unknown is False

    def test_build_signals_freshness_stale(self):
        """Freshness stale: > 90 days"""
        evidence = Evidence(
            business_domain="example.com",
            cites_official_domain=True,
            mentions_business_name=True,
            mentions_official_domain=True,
            citation_count=5,
            unique_citation_domains=["domain1.com"],
            freshest_cited_days=91,  # > 90
            has_services=True,
            has_location=True,
            has_contact=True,
        )

        signals = build_signals(evidence=evidence, authority_score=50)

        assert signals.freshness_stale is True
        assert signals.freshness_recent is False
        assert signals.freshness_unknown is False

    def test_build_signals_freshness_middle_range(self):
        """Freshness in middle range: 31-90 days (no special flag)"""
        evidence = Evidence(
            business_domain="example.com",
            cites_official_domain=True,
            mentions_business_name=True,
            mentions_official_domain=True,
            citation_count=5,
            unique_citation_domains=["domain1.com"],
            freshest_cited_days=60,  # Between 30 and 90
            has_services=True,
            has_location=True,
            has_contact=True,
        )

        signals = build_signals(evidence=evidence, authority_score=50)

        # No freshness flags should be True
        assert signals.freshness_recent is False
        assert signals.freshness_stale is False
        assert signals.freshness_unknown is False

    def test_build_signals_domain_diversity_boundaries(self):
        """Test domain diversity boundary conditions"""
        # Low diversity: < 5
        evidence_low = Evidence(
            business_domain="example.com",
            cites_official_domain=True,
            mentions_business_name=True,
            mentions_official_domain=True,
            citation_count=4,
            unique_citation_domains=[f"domain{i}.com" for i in range(4)],  # 4 domains
            freshest_cited_days=20,
            has_services=True,
            has_location=True,
            has_contact=True,
        )

        signals_low = build_signals(evidence=evidence_low, authority_score=50)
        assert signals_low.low_domain_diversity is True
        assert signals_low.high_domain_diversity is False

        # High diversity: >= 10
        evidence_high = Evidence(
            business_domain="example.com",
            cites_official_domain=True,
            mentions_business_name=True,
            mentions_official_domain=True,
            citation_count=10,
            unique_citation_domains=[f"domain{i}.com" for i in range(10)],  # 10 domains
            freshest_cited_days=20,
            has_services=True,
            has_location=True,
            has_contact=True,
        )

        signals_high = build_signals(evidence=evidence_high, authority_score=50)
        assert signals_high.low_domain_diversity is False
        assert signals_high.high_domain_diversity is True

        # Middle range: 5-9
        evidence_mid = Evidence(
            business_domain="example.com",
            cites_official_domain=True,
            mentions_business_name=True,
            mentions_official_domain=True,
            citation_count=7,
            unique_citation_domains=[f"domain{i}.com" for i in range(7)],  # 7 domains
            freshest_cited_days=20,
            has_services=True,
            has_location=True,
            has_contact=True,
        )

        signals_mid = build_signals(evidence=evidence_mid, authority_score=50)
        assert signals_mid.low_domain_diversity is False
        assert signals_mid.high_domain_diversity is False


class TestEvidenceDataclass:
    """Test Evidence dataclass"""

    def test_evidence_creation(self):
        """Evidence dataclass can be created"""
        evidence = Evidence(
            business_domain="test.com",
            cites_official_domain=True,
            mentions_business_name=True,
            mentions_official_domain=True,
            citation_count=5,
            unique_citation_domains=["a.com", "b.com"],
            freshest_cited_days=10,
            has_services=True,
            has_location=True,
            has_contact=True,
        )

        assert evidence.business_domain == "test.com"
        assert evidence.citation_count == 5


class TestSignalsDataclass:
    """Test Signals dataclass"""

    def test_signals_creation(self):
        """Signals dataclass can be created"""
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

        assert signals.high_domain_diversity is True
        assert signals.freshness_recent is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
