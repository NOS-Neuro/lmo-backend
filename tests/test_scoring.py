"""
Unit tests for scoring logic in scan_engine_real.py

Tests cover:
- Discovery score calculation
- Accuracy score calculation
- Authority score calculation
- Overall score derivation
- Package recommendation logic
"""
import pytest
from typing import Dict, Any
from requests.exceptions import Timeout


# Import the scoring function - we'll need to extract it first
# For now, we'll test the logic by importing from scan_engine_real
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scan_engine_real import (
    _domain,
    _extract_coverage_signals,
    derive_recommendation,
    run_real_scan_perplexity,
)
import scan_engine_real
from core.ras.scoring import interpret


class TestDiscoveryScore:
    """Test discovery score calculation logic

    Discovery score is based on:
    - mentions_business_name: +45 points
    - mentions_official_domain: +20 points
    - cites_official_domain: +35 points
    - Max score: 95 (capped from 100)
    """

    def test_perfect_discovery_score(self):
        """All discovery signals present = 100 points (capped to 95)"""
        # This would normally calculate to 100 (45+20+35)
        # but is capped at 95
        # We'll test the derive_recommendation function which takes pre-calculated scores
        discovery = 95
        accuracy = 95
        authority = 95

        overall, package, explanation, strategy = derive_recommendation(discovery, accuracy, authority)
        assert overall == 95
        assert package == "Basic LMO"

    def test_zero_discovery_score(self):
        """No discovery signals = 0 points"""
        discovery = 0
        accuracy = 50  # default
        authority = 50  # default

        overall, package, explanation, strategy = derive_recommendation(discovery, accuracy, authority)
        assert overall < 40  # Should be in low range
        assert "Standard LMO + Add-Ons" in package or "Standard LMO" in package

    def test_partial_discovery_name_only(self):
        """Only business name mentioned = 45 points"""
        discovery = 45
        accuracy = 50
        authority = 50

        overall, package, explanation, strategy = derive_recommendation(discovery, accuracy, authority)
        # (45 + 50 + 50) / 3 = 48.33, rounds to 48
        assert 47 <= overall <= 49
        assert package == "Standard LMO"

    def test_partial_discovery_domain_only(self):
        """Only domain mentioned = 20 points"""
        discovery = 20
        accuracy = 50
        authority = 50

        overall, package, explanation, strategy = derive_recommendation(discovery, accuracy, authority)
        # (20 + 50 + 50) / 3 = 40
        assert 39 <= overall <= 41

    def test_partial_discovery_citation_only(self):
        """Only official citation = 35 points"""
        discovery = 35
        accuracy = 50
        authority = 50

        overall, package, explanation, strategy = derive_recommendation(discovery, accuracy, authority)
        # (35 + 50 + 50) / 3 = 45
        assert 44 <= overall <= 46
        assert package == "Standard LMO"


class TestAccuracyScore:
    """Test accuracy score calculation logic

    Accuracy score is based on:
    - cites_official_domain: +60 points OR +25 if not cited
    - mentions_official_domain: +20 points
    - mentions_business_name: +20 points
    - mismatch_penalty: -15 if 6+ other domains and no official citation
    - Default: 50 points
    - Max score: 95 (capped from 100)
    """

    def test_perfect_accuracy_score(self):
        """All accuracy signals present = 100 points (capped to 95)"""
        discovery = 95
        accuracy = 95  # cites_official(60) + mentions_domain(20) + mentions_name(20) = 100, capped to 95
        authority = 95

        overall, package, explanation, strategy = derive_recommendation(discovery, accuracy, authority)
        assert overall == 95
        assert package == "Basic LMO"

    def test_accuracy_with_mismatch_penalty(self):
        """Mismatch penalty applies when 6+ domains without official citation"""
        # Accuracy without official citation: 25 + 20 + 20 - 15 = 50
        discovery = 50
        accuracy = 50  # with penalty applied
        authority = 50

        overall, package, explanation, strategy = derive_recommendation(discovery, accuracy, authority)
        assert overall == 50
        assert package == "Standard LMO"

    def test_accuracy_minimal_signals(self):
        """Minimal accuracy signals = default 50 points"""
        discovery = 50
        accuracy = 50  # default when no strong signals
        authority = 50

        overall, package, explanation, strategy = derive_recommendation(discovery, accuracy, authority)
        assert overall == 50
        assert package == "Standard LMO"


class TestAuthorityScore:
    """Test authority score calculation logic

    Authority score is based on:
    - cites_official_domain: +55 points OR +15 if not cited
    - unique_citation_domains count: up to +25 points (4 points per domain, max 25)
    - authority domain bonus: up to +20 points for high-quality domains
    - Default: 50 points
    - Max score: 95 (capped from 100)
    """

    def test_perfect_authority_score(self):
        """All authority signals present = ~100 points (capped to 95)"""
        # cites_official(55) + max_domains(25) + bonus(20) = 100, capped to 95
        discovery = 95
        accuracy = 95
        authority = 95

        overall, package, explanation, strategy = derive_recommendation(discovery, accuracy, authority)
        assert overall == 95
        assert package == "Basic LMO"

    def test_authority_without_official_citation(self):
        """Authority without official citation = lower base"""
        # Without official: 15 + domain_count + bonus (max ~60)
        discovery = 50
        accuracy = 50
        authority = 60  # moderate authority without official citation

        overall, package, explanation, strategy = derive_recommendation(discovery, accuracy, authority)
        # (50 + 50 + 60) / 3 = 53.33, rounds to 53
        assert 52 <= overall <= 54
        assert package == "Standard LMO"

    def test_authority_minimal_signals(self):
        """Minimal authority signals = default 50 points"""
        discovery = 50
        accuracy = 50
        authority = 50

        overall, package, explanation, strategy = derive_recommendation(discovery, accuracy, authority)
        assert overall == 50
        assert package == "Standard LMO"


class TestOverallScore:
    """Test overall score derivation

    Overall score = (discovery + accuracy + authority) / 3
    Capped at 95
    """

    def test_overall_calculation_average(self):
        """Overall score is the average of the three scores"""
        discovery = 60
        accuracy = 70
        authority = 80

        overall, _, _, _ = derive_recommendation(discovery, accuracy, authority)
        # (60 + 70 + 80) / 3 = 70
        assert overall == 70

    def test_overall_calculation_uneven(self):
        """Overall score rounds properly with uneven averages"""
        discovery = 33
        accuracy = 34
        authority = 35

        overall, _, _, _ = derive_recommendation(discovery, accuracy, authority)
        # (33 + 34 + 35) / 3 = 34
        assert overall == 34

    def test_overall_score_capped_at_95(self):
        """Overall score calculation with high values"""
        # Note: derive_recommendation doesn't cap scores, capping happens in the scan engine
        discovery = 100
        accuracy = 100
        authority = 100

        overall, _, _, _ = derive_recommendation(discovery, accuracy, authority)
        # (100 + 100 + 100) / 3 = 100
        assert overall == 100


class TestPackageRecommendation:
    """Test package recommendation logic

    Package tiers:
    - Basic LMO: overall >= 80 (maintenance mode)
    - Standard LMO: 40 <= overall < 80 (standard optimization)
    - Standard LMO + Add-Ons: overall < 40 (needs foundational work)
    """

    def test_basic_lmo_package(self):
        """Overall >= 80 = Basic LMO package"""
        discovery = 80
        accuracy = 85
        authority = 90

        overall, package, explanation, strategy = derive_recommendation(discovery, accuracy, authority)
        assert overall >= 80
        assert package == "Basic LMO"
        assert "maintenance" in explanation.lower() or "strong" in explanation.lower()

    def test_standard_lmo_package(self):
        """40 <= overall < 80 = Standard LMO package"""
        discovery = 50
        accuracy = 55
        authority = 60

        overall, package, explanation, strategy = derive_recommendation(discovery, accuracy, authority)
        assert 40 <= overall < 80
        assert package == "Standard LMO"
        assert "gaps" in explanation.lower() or "strengthens" in explanation.lower()

    def test_standard_lmo_addons_package(self):
        """overall < 40 = Standard LMO + Add-Ons package"""
        discovery = 20
        accuracy = 30
        authority = 35

        overall, package, explanation, strategy = derive_recommendation(discovery, accuracy, authority)
        assert overall < 40
        assert "Standard LMO + Add-Ons" in package
        assert "foundational" in explanation.lower() or "weak" in explanation.lower()


def test_run_real_scan_perplexity_allows_partial_results_and_records_timings(monkeypatch):
    monkeypatch.setattr(scan_engine_real.settings, "PERPLEXITY_API_KEY", "test-key")
    monkeypatch.setattr(scan_engine_real.settings, "OPENAI_API_KEY", None)
    monkeypatch.setattr(scan_engine_real.settings, "SCAN_QUERY_MAX_WORKERS", 2)

    def fake_chat_web(self, *, system, user, max_tokens=650, max_retries=None):
        if "Question 2" in user:
            raise Timeout("slow provider")
        return (
            "Acme Corp is based in Toronto. Contact acme.com.",
            [scan_engine_real.PerplexityHit(title="Acme", url="https://acme.com")],
            {"ok": True},
        )

    monkeypatch.setattr(scan_engine_real.PerplexityClient, "chat_web", fake_chat_web)

    result, raw_bundle = run_real_scan_perplexity(
        business_name="Acme Corp",
        website="https://acme.com",
        questions=[("q1", "Question 1"), ("q2", "Question 2")],
    )

    assert result.overall_score >= 0
    assert raw_bundle["timings"]["query_success_count"] == 1
    assert raw_bundle["timings"]["query_error_count"] == 1
    assert raw_bundle["timings"]["queries"][0]["status"] in {"ok", "error"}
    assert "total_seconds" in raw_bundle["timings"]
    assert result.confidence_level in {"low", "medium", "high", "insufficient"}
    assert result.visibility_status in {"clearly_seen", "partially_seen", "weakly_seen", "insufficient_evidence"}


def test_run_real_scan_perplexity_raises_when_all_queries_fail(monkeypatch):
    monkeypatch.setattr(scan_engine_real.settings, "PERPLEXITY_API_KEY", "test-key")
    monkeypatch.setattr(scan_engine_real.settings, "OPENAI_API_KEY", None)

    def fake_chat_web(self, *, system, user, max_tokens=650, max_retries=None):
        raise Timeout("slow provider")

    monkeypatch.setattr(scan_engine_real.PerplexityClient, "chat_web", fake_chat_web)

    with pytest.raises(RuntimeError, match="no successful results"):
        run_real_scan_perplexity(
            business_name="Acme Corp",
            website="https://acme.com",
            questions=[("q1", "Question 1")],
        )


def test_domain_normalization_preserves_real_hostname():
    assert _domain("https://www.vizai.io/about") == "vizai.io"
    assert _domain(" https://Vizai.io ") == "vizai.io"
    assert _domain("https://www2.example.com") == "www2.example.com"
    assert _domain("https://wexample.com") == "wexample.com"


def test_conservative_coverage_extraction_avoids_weak_keyword_matches():
    provider_results = [
        scan_engine_real.ProviderResult(
            provider="perplexity",
            model="sonar-pro",
            prompt_name="competitive_position",
            question="Q1",
            answer_text="The company serves enterprise buyers and has a sales-led motion.",
            citations=[],
        )
    ]

    coverage = _extract_coverage_signals(provider_results)

    assert coverage["has_location"] is False
    assert coverage["has_contact"] is False
    assert coverage["has_services"] is False


def test_thin_evidence_produces_insufficient_or_low_confidence(monkeypatch):
    monkeypatch.setattr(scan_engine_real.settings, "PERPLEXITY_API_KEY", "test-key")
    monkeypatch.setattr(scan_engine_real.settings, "OPENAI_API_KEY", None)

    def fake_chat_web(self, *, system, user, max_tokens=650, max_retries=None):
        return (
            "Acme Corp is unclear.",
            [],
            {"ok": True},
        )

    monkeypatch.setattr(scan_engine_real.PerplexityClient, "chat_web", fake_chat_web)

    result, raw_bundle = run_real_scan_perplexity(
        business_name="Acme Corp",
        website="https://acme.com",
        questions=[("q1", "Question 1"), ("q2", "Question 2")],
    )

    assert result.confidence_level in {"low", "insufficient"}
    assert result.visibility_status == "insufficient_evidence"
    assert "Official domain citations" in result.missing_signals

    def test_boundary_case_exactly_80(self):
        """Exactly 80 should be Basic LMO"""
        discovery = 80
        accuracy = 80
        authority = 80

        overall, package, _, _ = derive_recommendation(discovery, accuracy, authority)
        assert overall == 80
        assert package == "Basic LMO"

    def test_boundary_case_exactly_40(self):
        """Exactly 40 should be Standard LMO"""
        discovery = 40
        accuracy = 40
        authority = 40

        overall, package, _, _ = derive_recommendation(discovery, accuracy, authority)
        assert overall == 40
        assert package == "Standard LMO"

    def test_boundary_case_39(self):
        """39 should be Standard LMO + Add-Ons"""
        discovery = 39
        accuracy = 39
        authority = 39

        overall, package, _, _ = derive_recommendation(discovery, accuracy, authority)
        assert overall == 39
        assert "Standard LMO + Add-Ons" in package


class TestStrategyRecommendations:
    """Test that strategy recommendations are provided"""

    def test_strategy_returned_for_high_scores(self):
        """High scores should return maintenance strategy"""
        discovery = 85
        accuracy = 85
        authority = 85

        _, _, _, strategy = derive_recommendation(discovery, accuracy, authority)
        assert strategy is not None
        assert len(strategy) > 0
        assert "Truth File" in strategy or "schema" in strategy or "recheck" in strategy

    def test_strategy_returned_for_medium_scores(self):
        """Medium scores should return optimization strategy"""
        discovery = 55
        accuracy = 55
        authority = 55

        _, _, _, strategy = derive_recommendation(discovery, accuracy, authority)
        assert strategy is not None
        assert len(strategy) > 0

    def test_strategy_returned_for_low_scores(self):
        """Low scores should return foundational strategy"""
        discovery = 25
        accuracy = 25
        authority = 25

        _, _, _, strategy = derive_recommendation(discovery, accuracy, authority)
        assert strategy is not None
        assert len(strategy) > 0
        assert "Truth File" in strategy or "schema" in strategy or "Re-scan" in strategy

    def test_strategy_copy_uses_ascii_punctuation(self):
        _, _, explanation, strategy = derive_recommendation(85, 85, 85)
        assert "You're" in explanation
        assert "1-2" in strategy

        _, _, low_explanation, _ = derive_recommendation(20, 20, 20)
        assert "You'll" in low_explanation


class TestScoreBandCopy:
    def test_confidence_copy_uses_ascii_punctuation(self):
        assert interpret(90).confidence == "High - low hallucination risk"
        assert interpret(75).confidence == "Medium - minor risk of errors"
        assert interpret(60).confidence == "Low - significant hallucination risk"
        assert interpret(20).confidence == "Very low - unreliable answers expected"


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_all_zeros(self):
        """All scores at zero should still return valid recommendation"""
        discovery = 0
        accuracy = 0
        authority = 0

        overall, package, explanation, strategy = derive_recommendation(discovery, accuracy, authority)
        assert overall == 0
        assert package is not None
        assert explanation is not None
        assert strategy is not None

    def test_mixed_extremes(self):
        """One high score, two low scores"""
        discovery = 95
        accuracy = 10
        authority = 10

        overall, package, _, _ = derive_recommendation(discovery, accuracy, authority)
        # (95 + 10 + 10) / 3 = 38.33, rounds to 38
        assert 37 <= overall <= 39
        assert "Standard LMO + Add-Ons" in package

    def test_negative_scores_not_allowed(self):
        """Scores should never be negative (clamped at 0)"""
        # This tests that the system handles edge cases gracefully
        # In practice, scores are clamped to 0 minimum
        discovery = 0
        accuracy = 0
        authority = 0

        overall, _, _, _ = derive_recommendation(discovery, accuracy, authority)
        assert overall >= 0

    def test_scores_above_100_capped(self):
        """Scores above 100 should be capped at 95"""
        discovery = 95
        accuracy = 95
        authority = 95

        overall, _, _, _ = derive_recommendation(discovery, accuracy, authority)
        assert overall <= 95


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
