"""
scan_engine_real.py — VizAI Real Scan Engine (Perplexity-first, production-safe MVP)

Real + measurable:
- Uses Perplexity web-backed search (search_mode="web")
- Captures answer + citations (search_results)
- Computes deterministic scores: Discovery / Authority (+ Accuracy proxy)
- Adds measurable metrics: Freshness, Comprehensiveness (stored)
- Builds Evidence -> Signals -> Recommendations bundle (stored in raw_bundle + metrics)

Honesty:
- Accuracy is still a proxy until a Truth File compare is implemented.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from requests.exceptions import Timeout, RequestException

from core.prompts import DEFAULT_QUESTIONS, get_identity_questions
from config import settings
import logging
import json

logger = logging.getLogger("vizai")


# -----------------------------
# Types
# -----------------------------

@dataclass
class PerplexityHit:
    title: str
    url: str
    date: Optional[str] = None


@dataclass
class ProviderResult:
    provider: str
    model: str
    prompt_name: str
    question: str
    answer_text: str
    citations: List[PerplexityHit]


@dataclass
class RealScanResult:
    ai_visibility_score: int
    discovery_score: int
    accuracy_score: int
    authority_score: int
    overall_score: int
    visibility_status: str
    confidence_level: str
    evidence_summary: str
    verified_facts: List[str]
    unclear_facts: List[str]
    missing_signals: List[str]
    limitations: List[str]
    proof_signals: List[str]

    package_recommendation: str
    package_explanation: str
    strategy_summary: str

    findings: List[str]
    provider_results: List[ProviderResult]
    metrics: Dict[str, Any]

    # Entity validation fields
    entity_status: Optional[str] = None
    entity_confidence: Optional[int] = None
    warnings: Optional[List[str]] = None


# -----------------------------
# Helpers
# -----------------------------

def _clamp_int(v: Any, default: int = 50, lo: int = 0, hi: int = 100) -> int:
    try:
        n = int(v)
    except Exception:
        n = int(default)
    return max(lo, min(hi, n))


def _domain(url: str) -> str:
    try:
        parsed = urlparse((url or "").strip())
        host = parsed.netloc or parsed.path
        host = host.strip().lower()
        if "/" in host:
            host = host.split("/", 1)[0]
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def _contains_name(answer: str, business_name: str) -> bool:
    a = _norm(answer)
    name = _norm(business_name)
    if not name:
        return False
    if name in a:
        return True
    tokens = [t for t in re.split(r"[^a-z0-9]+", name) if len(t) >= 4]
    if not tokens:
        return False
    return all(t in a for t in tokens[:3])


def _unique(seq: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _parse_date(d: Optional[str]) -> Optional[datetime]:
    if not d:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"):
        try:
            return datetime.strptime(d, fmt)
        except Exception:
            continue
    try:
        return datetime.fromisoformat(d.replace("Z", "+00:00"))
    except Exception:
        return None


def _extract_capitalized_names(text: str) -> List[str]:
    """Extract capitalized name candidates from text (for conflict detection)"""
    # Simple heuristic: find sequences of 2+ capitalized words
    pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
    return re.findall(pattern, text)


def _count_unclear_responses(provider_results: List[ProviderResult]) -> Tuple[int, float]:
    """Count how many responses contain 'unclear' or similar"""
    unclear_count = 0
    for r in provider_results:
        answer_lower = r.answer_text.lower()
        if 'unclear' in answer_lower or 'not available' in answer_lower or 'no information' in answer_lower:
            unclear_count += 1

    unclear_rate = unclear_count / len(provider_results) if provider_results else 0.0
    return unclear_count, unclear_rate


def _extract_coverage_signals(provider_results: List[ProviderResult]) -> Dict[str, bool]:
    texts = [_norm(r.answer_text) for r in provider_results]
    services_patterns = [
        r"\bservices include\b",
        r"\bprovides\b.{0,80}\bservices?\b",
        r"\boffers\b.{0,80}\bservices?\b",
        r"\bspecializes in\b",
        r"\bprimary services\b",
        r"\bservice lines\b",
    ]
    location_patterns = [
        r"\bbased in\s+[a-z]",
        r"\bheadquartered in\s+[a-z]",
        r"\blocated in\s+[a-z]",
        r"\boperates in\s+[a-z]",
        r"\bservice area\b",
        r"\bregions served\b",
    ]
    contact_patterns = [
        r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}",
        r"\+\d[\d\s().-]{7,}",
        r"\bcontact page\b",
        r"\bcontact us\b",
        r"\bphone:\b",
        r"\bemail:\b",
        r"\btelephone\b",
    ]

    has_services = any(any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in services_patterns) for text in texts)
    has_location = any(any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in location_patterns) for text in texts)
    has_contact = any(any(re.search(pattern, r.answer_text or "", flags=re.IGNORECASE) for pattern in contact_patterns) for r in provider_results)

    return {
        "has_services": has_services,
        "has_location": has_location,
        "has_contact": has_contact,
    }


def _extract_proof_signals(provider_results: List[ProviderResult]) -> List[str]:
    proof_patterns = {
        "certifications": [r"\bcertified\b", r"\biso\s?\d{3,5}\b", r"\bsoc\s?2\b", r"\bcompliance\b"],
        "awards": [r"\baward\b", r"\bawarded\b", r"\bwinner\b", r"\brecognized by\b"],
        "partnerships": [r"\bpartnership with\b", r"\bpartnered with\b", r"\bofficial partner\b"],
        "testimonials": [r"\btestimonial\b", r"\bcustomer review\b", r"\bclient review\b"],
        "case_studies": [r"\bcase study\b", r"\bcustomer story\b", r"\bsuccess story\b"],
    }
    all_text = "\n".join(r.answer_text or "" for r in provider_results)
    found: List[str] = []
    for label, patterns in proof_patterns.items():
        if any(re.search(pattern, all_text, flags=re.IGNORECASE) for pattern in patterns):
            found.append(label)
    return found


def _prompt_summary(prompt_name: str, question: str) -> str:
    mapping = {
        "website_validation": "Whether the submitted website is the official site",
        "identity_fingerprint": "Core identity details such as HQ, phone, and primary services",
        "baseline_overview": "A concise description of what the business does",
        "founder_team": "Founder and leadership details",
        "recent_activity": "Recent activity and updates",
        "social_proof": "Scale, customers, or measurable proof",
        "locations_scope": "Where the business operates",
        "competitive_position": "How the business is positioned in its market",
        "proof_points": "Proof points such as certifications, awards, partnerships, or case studies",
    }
    return mapping.get(prompt_name, question.strip())


def _build_visibility_confidence(
    *,
    query_success_count: int,
    query_error_count: int,
    entity_status: str,
    entity_confidence: int,
    unclear_rate: float,
    citation_count: int,
    unique_citation_domain_count: int,
    cites_official_domain: bool,
    validator_supported_count: int,
    validator_total_count: int,
) -> Tuple[int, str]:
    total_queries = query_success_count + query_error_count
    success_rate = (query_success_count / total_queries) if total_queries else 0.0

    confidence_score = 20
    confidence_score += int(success_rate * 25)
    confidence_score += min(int(entity_confidence * 0.25), 25)
    confidence_score += min(citation_count, 10)
    confidence_score += min(unique_citation_domain_count * 2, 10)
    if cites_official_domain:
        confidence_score += 10
    if validator_total_count:
        confidence_score += int((validator_supported_count / validator_total_count) * 10)
    if unclear_rate > 0.5:
        confidence_score -= 20
    elif unclear_rate > 0.3:
        confidence_score -= 10
    if query_error_count:
        confidence_score -= min(query_error_count * 5, 15)
    if entity_status == "MISMATCH":
        confidence_score -= 20
    elif entity_status == "UNCLEAR":
        confidence_score -= 10

    confidence_score = _clamp_int(confidence_score, default=20, lo=0, hi=100)
    if confidence_score >= 75:
        return confidence_score, "high"
    if confidence_score >= 55:
        return confidence_score, "medium"
    if confidence_score >= 35:
        return confidence_score, "low"
    return confidence_score, "insufficient"


def _build_public_summary(
    *,
    business_name: str,
    business_domain: str,
    overall_score: int,
    confidence_level: str,
    query_success_count: int,
    query_error_count: int,
    entity_status: str,
    entity_confidence: int,
    cites_official_domain: bool,
    has_services: bool,
    has_location: bool,
    has_contact: bool,
    proof_signals: List[str],
    provider_results: List[ProviderResult],
    validator_supported_count: int,
    validator_total_count: int,
) -> Dict[str, Any]:
    if confidence_level == "insufficient" or entity_status in {"UNCLEAR", "MISMATCH"}:
        visibility_status = "insufficient_evidence"
    elif overall_score >= 80 and cites_official_domain:
        visibility_status = "clearly_seen"
    elif overall_score >= 55:
        visibility_status = "partially_seen"
    else:
        visibility_status = "weakly_seen"

    verified_facts: List[str] = []
    if _contains_name("\n".join(r.answer_text for r in provider_results), business_name):
        verified_facts.append("AI identified the business name in returned answers.")
    if cites_official_domain:
        verified_facts.append(f"AI cited the official domain: {business_domain}.")
    if has_services:
        verified_facts.append("AI found a clear description of the business's services.")
    if has_location:
        verified_facts.append("AI found an explicit operating location or service area.")
    if has_contact:
        verified_facts.append("AI found a concrete contact path such as email, phone, or contact page.")
    if proof_signals:
        verified_facts.append(f"AI found citeable proof signals: {', '.join(proof_signals)}.")

    unclear_facts = [
        _prompt_summary(r.prompt_name, r.question)
        for r in provider_results
        if "unclear" in (r.answer_text or "").lower() or "not available" in (r.answer_text or "").lower() or "no information" in (r.answer_text or "").lower()
    ]
    unclear_facts = _unique(unclear_facts)

    missing_signals: List[str] = []
    if not cites_official_domain:
        missing_signals.append("Official domain citations")
    if not has_services:
        missing_signals.append("Clear service descriptions")
    if not has_location:
        missing_signals.append("Explicit operating location or service area")
    if not has_contact:
        missing_signals.append("Concrete contact details")
    if not proof_signals:
        missing_signals.append("Proof points such as certifications, awards, partnerships, testimonials, or case studies")

    limitations: List[str] = []
    if query_error_count:
        limitations.append(f"{query_error_count} scan question(s) timed out or failed, so this is a partial read of AI visibility.")
    if confidence_level in {"low", "insufficient"}:
        limitations.append("The available evidence was thin or inconsistent, so confidence is limited.")
    if entity_status in {"UNCLEAR", "MISMATCH"}:
        limitations.append(f"Entity matching was {entity_status.lower()}, which limits how confidently facts can be attributed to this business.")
    if validator_total_count and validator_supported_count < validator_total_count:
        limitations.append("Some answer summaries were not strongly supported by the validator layer.")
    if not limitations:
        limitations.append("No major evidence limitations were detected in this scan.")

    evidence_summary = (
        f"VizAI found {visibility_status.replace('_', ' ')} visibility with {confidence_level} confidence. "
        f"{query_success_count} query result(s) succeeded and {query_error_count} failed. "
        f"Entity matching is {entity_status.lower()} ({entity_confidence}/100). "
        f"{'The official domain was cited in the evidence.' if cites_official_domain else 'The official domain was not cited in the evidence.'}"
    )

    return {
        "visibility_status": visibility_status,
        "verified_facts": verified_facts[:6],
        "unclear_facts": unclear_facts[:6],
        "missing_signals": missing_signals[:6],
        "limitations": limitations[:6],
        "evidence_summary": evidence_summary,
    }


# -----------------------------
# Entity Match Gate (Identity Confidence)
# -----------------------------

def compute_entity_identity(
    *,
    submitted_domain: str,
    business_name: str,
    perplexity_results: List[ProviderResult],
    all_text: str
) -> Dict[str, Any]:
    """
    Compute entity identity confidence from Perplexity evidence ONLY.
    Returns identity gate structure for raw_bundle["identity"].
    """
    # Collect all citation domains
    all_domains: List[str] = []
    for r in perplexity_results:
        for hit in r.citations:
            if hit.url:
                domain = _domain(hit.url)
                if domain:
                    all_domains.append(domain)

    # Count domain frequencies
    domain_counts: Dict[str, int] = {}
    for d in all_domains:
        domain_counts[d] = domain_counts.get(d, 0) + 1

    # Get dominant domains (top 10 by citation count)
    sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
    dominant_citation_domains = [d for d, _ in sorted_domains[:10]]

    # Check if submitted domain appears in citations or answers
    submitted_cited = submitted_domain in all_domains
    submitted_mentioned = submitted_domain in _norm(all_text)

    # Detect official domain from evidence
    official_domain_detected = submitted_domain if (submitted_cited or submitted_mentioned) else ""

    # If submitted domain not found, try to infer from most-cited domain
    if not official_domain_detected and dominant_citation_domains:
        # Check if top domain is described as "official" in answers
        top_domain = dominant_citation_domains[0]
        if "official" in _norm(all_text) and top_domain in _norm(all_text):
            official_domain_detected = top_domain

    # Name conflict detection: look for other capitalized names
    name_candidates = _extract_capitalized_names(all_text)
    name_lower = _norm(business_name)

    # Filter out the actual business name from candidates
    conflict_candidates = [
        name for name in name_candidates
        if _norm(name) != name_lower and len(name) > 3
    ]

    # Count unique conflict candidates (limit to top 5)
    conflict_counts: Dict[str, int] = {}
    for name in conflict_candidates:
        conflict_counts[name] = conflict_counts.get(name, 0) + 1
    top_conflicts = sorted(conflict_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    name_conflict_candidates = [name for name, _ in top_conflicts]

    # Compute unclear rate
    _, unclear_rate = _count_unclear_responses(perplexity_results)

    # Compute entity confidence (0-100)
    confidence = 100

    # Penalties
    if not submitted_cited:
        confidence -= 30  # Major penalty if official domain never cited
    if not submitted_mentioned:
        confidence -= 15  # Penalty if domain not even mentioned
    if not _contains_name(all_text, business_name):
        confidence -= 25  # Major penalty if business name not found
    if len(name_conflict_candidates) > 2:
        confidence -= 20  # Many conflicting names detected
    if unclear_rate > 0.3:
        confidence -= 15  # High unclear rate
    if len(dominant_citation_domains) < 3:
        confidence -= 10  # Very few citation sources

    confidence = _clamp_int(confidence, default=50, lo=0, hi=100)

    # Determine entity status
    if confidence >= 80 and submitted_cited:
        entity_status = "CONFIRMED"
    elif confidence >= 60:
        entity_status = "LIKELY"
    elif confidence >= 40:
        entity_status = "UNCLEAR"
    else:
        entity_status = "MISMATCH"

    # If official domain detected doesn't match submitted, downgrade status
    if official_domain_detected and official_domain_detected != submitted_domain:
        if entity_status == "CONFIRMED":
            entity_status = "LIKELY"
        elif entity_status == "LIKELY":
            entity_status = "UNCLEAR"

    return {
        "submitted_domain": submitted_domain,
        "official_domain_detected": official_domain_detected,
        "entity_confidence": confidence,
        "entity_status": entity_status,
        "dominant_citation_domains": dominant_citation_domains,
        "name_conflict_candidates": name_conflict_candidates,
        "unclear_rate": round(unclear_rate, 3),
        "submitted_domain_cited": submitted_cited,
        "submitted_domain_mentioned": submitted_mentioned,
    }


# -----------------------------
# Package recommendation logic
# -----------------------------

def derive_recommendation(discovery: int, accuracy: int, authority: int) -> Tuple[int, str, str, str]:
    overall = int(round((discovery + accuracy + authority) / 3))

    if overall >= 80:
        package = "Basic LMO"
        explanation = (
            "AI systems can already identify the business and support the core profile with relatively strong evidence. "
            "Basic is for monitoring drift and tightening weaker verification points."
        )
        strategy = (
            "Maintain canonical pages, keep schema and metadata current, and re-check visibility after major updates. "
            "Add 1-2 stronger proof assets only where evidence is still thin."
        )
    elif overall >= 40:
        package = "Standard LMO"
        explanation = (
            "AI can identify part of the business, but some claims are weakly supported, inconsistently described, or hard to verify. "
            "Standard focuses on making the profile clearer and more citeable."
        )
        strategy = (
            "Tighten About, Services, FAQ, contact, and location coverage, add structured data, and strengthen the source mix that points back to the official site. "
            "Then re-scan and compare what AI can verify more confidently."
        )
    else:
        package = "Standard LMO + Add-Ons"
        explanation = (
            "AI visibility is weak, fragmented, or poorly supported by evidence. "
            "This path is for rebuilding the canonical source and adding proof where AI systems currently cannot verify key claims."
        )
        strategy = (
            "Start with canonical source repair, structured data, and proof pages, then add authority seeding and listing cleanup. "
            "Re-scan until AI can identify and verify the business more reliably."
        )

    return overall, package, explanation, strategy


# -----------------------------
# Perplexity client
# -----------------------------

class PerplexityClient:
    BASE_URL = "https://api.perplexity.ai/chat/completions"

    def __init__(self, api_key: str, model: str, timeout: int):
        if not api_key:
            raise ValueError("PERPLEXITY_API_KEY is not set")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def chat_web(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 650,
        max_retries: Optional[int] = None,
    ) -> Tuple[str, List[PerplexityHit], Dict[str, Any]]:
        max_retries = max_retries or settings.PERPLEXITY_MAX_RETRIES
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "search_mode": "web",
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": max_tokens,
            "stream": False,
        }

        last_error = None
        for attempt in range(max_retries):
            try:
                r = requests.post(
                    self.BASE_URL,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                if r.status_code < 200 or r.status_code >= 300:
                    # Don't retry on 4xx errors (client errors)
                    if 400 <= r.status_code < 500:
                        raise RuntimeError(f"Perplexity API error {r.status_code}: {r.text}")
                    # Retry on 5xx errors (server errors)
                    last_error = RuntimeError(f"Perplexity API error {r.status_code}: {r.text}")
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                        time.sleep(wait_time)
                        continue
                    raise last_error

                # Success - break out of retry loop
                break

            except Timeout as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                raise RuntimeError(f"Perplexity API timeout after {max_retries} attempts") from e

            except RequestException as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                raise RuntimeError(f"Perplexity API request failed after {max_retries} attempts") from e

        data = r.json()
        answer = data["choices"][0]["message"]["content"]

        hits: List[PerplexityHit] = []
        for h in (data.get("search_results") or []):
            hits.append(
                PerplexityHit(
                    title=str(h.get("title") or ""),
                    url=str(h.get("url") or ""),
                    date=h.get("date"),
                )
            )

        return answer, hits, data


# -----------------------------
# OpenAI Validator (evidence consistency checker ONLY)
# -----------------------------

VALIDATOR_PROMPT = """You are a fact-checking validator. You will be given:
1. A question that was asked
2. An answer from another AI system (Perplexity)
3. A list of citation URLs/domains that were used as sources

Your job is to evaluate whether the answer is well-supported by the citations provided.

RULES:
- You MUST NOT add new facts, sources, or URLs
- You MUST NOT browse the web or search for information
- You MUST ONLY evaluate the consistency and support of the given answer using the provided citations
- Return ONLY valid JSON (no markdown, no explanation)

Output format (STRICT JSON ONLY):
{
  "confidence_0_to_1": <float between 0 and 1>,
  "consistency_score_0_to_100": <integer 0-100>,
  "red_flags": [<list of string concerns if any>],
  "missing_expected_items": [<list of string items that should be present but aren't>]
}

Evaluation criteria:
- confidence_0_to_1: How confident are you the answer is supported by the citations? (0.0 = no support, 1.0 = fully supported)
- consistency_score_0_to_100: Overall consistency score (0 = inconsistent/unsupported, 100 = fully consistent)
- red_flags: List any concerns (e.g., "answer claims X but no citation supports it", "conflicting information")
- missing_expected_items: What key facts are missing that you'd expect given the question?
"""


class OpenAIValidator:
    """
    OpenAI-based validator for checking Perplexity answer consistency.
    Does NOT discover facts - only validates existing evidence.
    """

    BASE_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", timeout: int = 15):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def validate_answer(
        self,
        *,
        question: str,
        answer: str,
        citations: List[str],
        official_domain: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Validate a Perplexity answer against its citations.
        Returns validation result dict or None if validation fails.
        """
        # Build context for validator
        citations_text = "\n".join([f"- {url}" for url in citations])
        official_context = f"\nOfficial domain (if known): {official_domain}" if official_domain else ""

        user_prompt = f"""Question: {question}

Answer to validate:
{answer}

Citations provided:
{citations_text}{official_context}

Evaluate the answer's support from these citations only. Return JSON only."""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": VALIDATOR_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 300,
            "response_format": {"type": "json_object"},
        }

        try:
            r = requests.post(
                self.BASE_URL,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )

            if r.status_code != 200:
                logger.warning(f"OpenAI validator API error {r.status_code}: {r.text[:200]}")
                return None

            data = r.json()
            response_text = data["choices"][0]["message"]["content"]

            # Parse JSON response
            result = json.loads(response_text)

            # Validate structure
            if not isinstance(result.get("confidence_0_to_1"), (int, float)):
                logger.warning("Validator returned invalid confidence format")
                return None

            if not isinstance(result.get("consistency_score_0_to_100"), int):
                logger.warning("Validator returned invalid consistency_score format")
                return None

            # Clamp values
            result["confidence_0_to_1"] = max(0.0, min(1.0, float(result["confidence_0_to_1"])))
            result["consistency_score_0_to_100"] = max(0, min(100, int(result["consistency_score_0_to_100"])))

            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Validator returned invalid JSON: {e}")
            return None
        except Timeout:
            logger.warning("OpenAI validator timeout")
            return None
        except Exception as e:
            logger.warning(f"OpenAI validator error: {e}")
            return None


# -----------------------------
# Scan prompts
# -----------------------------

AUTHORITY_DOMAIN_BONUS = {
    "wikipedia.org": 10,
    "linkedin.com": 6,
    "crunchbase.com": 6,
    "bloomberg.com": 8,
    "reuters.com": 8,
    "sec.gov": 10,
    "sedarplus.ca": 10,
}


# -----------------------------
# Main entry
# -----------------------------

def run_real_scan_perplexity(
    *,
    business_name: str,
    website: str,
    industry: Optional[str] = None,
    competitors: Optional[List[Dict[str, Any]]] = None,
    questions: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[RealScanResult, Dict[str, Any]]:
    """
    Returns:
      (RealScanResult, raw_bundle)

    competitors:
      Accepted and stored for future use, but not executed here unless you explicitly want it.
    """
    scan_started = time.perf_counter()

    client = PerplexityClient(
        api_key=settings.PERPLEXITY_API_KEY,
        model=settings.PERPLEXITY_MODEL,
        timeout=settings.PERPLEXITY_TIMEOUT,
    )

    biz_domain = _domain(website)

    # Build question list: identity questions FIRST, then discovery questions
    if questions:
        qs = questions
    else:
        identity_qs = get_identity_questions(website)
        qs = identity_qs + list(DEFAULT_QUESTIONS)

    provider_results: List[ProviderResult] = []
    raw_bundle: Dict[str, Any] = {
        "engine": "perplexity_validated_v5",
        "provider": "perplexity",
        "model": settings.PERPLEXITY_MODEL,
        "runs": [],
        "notes": [
            "Discovery/Authority are evidence-based from returned citations.",
            "Accuracy is a proxy until a Truth File compare is implemented.",
            "Entity identity computed from Perplexity evidence only.",
            "Gating rules applied based on entity confidence and unclear rate.",
        ],
        "competitors": competitors or [],
        "identity": {},  # Will be populated after queries complete
        "validation": {  # Optional OpenAI validator results
            "mode": "openai_validator" if settings.OPENAI_API_KEY else "disabled",
            "model": "gpt-4o-mini" if settings.OPENAI_API_KEY else None,
            "runs": [],
            "notes": [],
        },
        "timings": {},
    }

    system = (
        "You are an audit assistant using web search. "
        "Do not guess. If information is missing, say 'unclear'. "
        "Search for official and authoritative sources."
    )

    # Define function to run a single query (for parallel execution)
    def run_single_query(prompt_name: str, question: str) -> Tuple[str, str, str, List[PerplexityHit], Any, int]:
        """Execute a single Perplexity query and return results"""
        query_started = time.perf_counter()
        industry_context = f"\nIndustry: {industry}" if industry else ""
        user = (
            f"Company name: {business_name}{industry_context}\n\n"
            f"Task: {question}\n\n"
            f"Rules:\n"
            f"- Only state facts supported by citations.\n"
            f"- If uncertain, say 'unclear'.\n"
            f"- Keep output concise.\n"
        )

        answer, hits, raw = client.chat_web(system=system, user=user, max_tokens=450)
        duration_ms = int((time.perf_counter() - query_started) * 1000)
        logger.info("Scan query completed: %s in %sms", prompt_name, duration_ms)
        return prompt_name, question, answer, hits, raw, duration_ms

    # Execute all queries in parallel for 5-7x speedup
    logger.info("Starting parallel execution of %d queries", len(qs))
    start_time = time.time()
    query_timings: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=min(len(qs), settings.SCAN_QUERY_MAX_WORKERS)) as executor:
        # Submit all queries
        future_to_query = {
            executor.submit(run_single_query, prompt_name, q): (prompt_name, q)
            for prompt_name, q in qs
        }

        # Collect results as they complete
        for future in as_completed(future_to_query):
            prompt_name, question = future_to_query[future]
            try:
                prompt_name, question, answer, hits, raw, duration_ms = future.result()

                provider_results.append(
                    ProviderResult(
                        provider="perplexity",
                        model=settings.PERPLEXITY_MODEL,
                        prompt_name=prompt_name,
                        question=question,
                        answer_text=answer,
                        citations=hits,
                    )
                )

                raw_bundle["runs"].append(
                    {
                        "prompt_name": prompt_name,
                        "question": question,
                        "answer_text": answer,
                        "search_results": [h.__dict__ for h in hits],
                        "raw": raw,
                    }
                )
                query_timings.append(
                    {
                        "prompt_name": prompt_name,
                        "status": "ok",
                        "duration_ms": duration_ms,
                        "citation_count": len(hits),
                    }
                )
            except Exception as e:
                logger.warning("Query failed for %s: %s", prompt_name, e)
                query_timings.append(
                    {
                        "prompt_name": prompt_name,
                        "status": "error",
                        "duration_ms": 0,
                        "error": str(e),
                    }
                )
                # Continue with other queries even if one fails

    elapsed = time.time() - start_time
    logger.info("Parallel query execution completed in %.2f seconds", elapsed)
    raw_bundle["timings"]["query_execution_seconds"] = round(elapsed, 3)
    raw_bundle["timings"]["queries"] = query_timings
    raw_bundle["timings"]["query_success_count"] = len(provider_results)
    raw_bundle["timings"]["query_error_count"] = len(qs) - len(provider_results)

    if not provider_results:
        raise RuntimeError("Scan provider returned no successful results")

    # -----------------------------
    # Aggregate evidence
    # -----------------------------
    all_text = "\n".join(r.answer_text for r in provider_results)

    all_hits: List[PerplexityHit] = []
    for r in provider_results:
        all_hits.extend(r.citations)

    cite_domains = [_domain(h.url) for h in all_hits if h.url]
    uniq_domains = _unique([d for d in cite_domains if d])

    # -----------------------------
    # Entity Identity Gate (from Perplexity evidence ONLY)
    # -----------------------------
    identity_result = compute_entity_identity(
        submitted_domain=biz_domain,
        business_name=business_name,
        perplexity_results=provider_results,
        all_text=all_text,
    )
    raw_bundle["identity"] = identity_result

    entity_status = identity_result["entity_status"]
    entity_confidence = identity_result["entity_confidence"]
    unclear_rate = identity_result["unclear_rate"]
    official_domain_detected = identity_result["official_domain_detected"]
    submitted_domain_cited = identity_result["submitted_domain_cited"]

    # -----------------------------
    # Optional Validator Layer (OpenAI evaluates Perplexity evidence)
    # -----------------------------
    if settings.OPENAI_API_KEY:
        validation_started = time.perf_counter()
        try:
            validator = OpenAIValidator(
                api_key=settings.OPENAI_API_KEY,
                model="gpt-4o-mini",
                timeout=15,
            )

            for r in provider_results:
                citation_urls = [hit.url for hit in r.citations if hit.url]
                validation = validator.validate_answer(
                    question=r.question,
                    answer=r.answer_text,
                    citations=citation_urls,
                    official_domain=official_domain_detected or None,
                )

                if validation:
                    raw_bundle["validation"]["runs"].append({
                        "prompt_name": r.prompt_name,
                        "validation": validation,
                    })

        except Exception as e:
            raw_bundle["validation"]["notes"].append(f"Validator error (non-fatal): {str(e)}")
            logger.warning(f"OpenAI validator error (non-fatal): {e}")
        finally:
            raw_bundle["timings"]["validation_seconds"] = round(time.perf_counter() - validation_started, 3)
    else:
        raw_bundle["validation"]["notes"].append("Validation skipped: no OPENAI_API_KEY")
        raw_bundle["timings"]["validation_seconds"] = 0.0

    mentions_name = _contains_name(all_text, business_name)
    mentions_domain = bool(biz_domain and biz_domain in _norm(all_text))
    cites_official = bool(biz_domain and any(d == biz_domain for d in cite_domains))

    # Freshness
    parsed_dates = [_parse_date(h.date) for h in all_hits]
    parsed_dates = [d for d in parsed_dates if d is not None]
    freshest_days = None
    if parsed_dates:
        newest = max(parsed_dates)
        now = datetime.now(timezone.utc)
        if newest.tzinfo is None:
            newest = newest.replace(tzinfo=timezone.utc)
        freshest_days = max(0, int((now - newest).total_seconds() // 86400))

    # Conservative coverage + proof extraction
    coverage = _extract_coverage_signals(provider_results)
    has_services = coverage["has_services"]
    has_location = coverage["has_location"]
    has_contact = coverage["has_contact"]
    comprehensiveness_hits = int(has_services) + int(has_location) + int(has_contact)
    proof_signals = _extract_proof_signals(provider_results)

    # -----------------------------
    # Deterministic scoring (then cap to avoid “perfect”)
    # -----------------------------
    discovery = 0
    discovery += 45 if mentions_name else 0
    discovery += 20 if mentions_domain else 0
    discovery += 35 if cites_official else 0
    discovery = _clamp_int(discovery, default=0)

    mismatch_penalty = 0
    if biz_domain:
        other_domains = [d for d in uniq_domains if d != biz_domain]
        if len(other_domains) >= 6 and not cites_official:
            mismatch_penalty = 15

    accuracy = 0
    accuracy += 60 if cites_official else 25
    accuracy += 20 if mentions_domain else 0
    accuracy += 20 if mentions_name else 0
    accuracy -= mismatch_penalty
    accuracy = _clamp_int(accuracy, default=50)

    authority = 0
    authority += 55 if cites_official else 15
    authority += _clamp_int(len(uniq_domains) * 4, default=0, lo=0, hi=25)

    bonus = 0
    for d in uniq_domains:
        bonus += AUTHORITY_DOMAIN_BONUS.get(d, 0)
    authority += _clamp_int(bonus, default=0, lo=0, hi=20)
    authority = _clamp_int(authority, default=50)

    # -----------------------------
    # Hard Gating Rules (score compression and entity-based caps)
    # -----------------------------
    warnings: List[str] = []

    # Count submitted domain citations
    submitted_domain_cite_count = sum(1 for d in cite_domains if d == biz_domain)

    # Entity status gates
    if entity_status in {"UNCLEAR", "MISMATCH"}:
        warnings.append(f"Entity match is {entity_status}; visibility score capped at 60 because attribution is uncertain.")

    if official_domain_detected and official_domain_detected != biz_domain:
        warnings.append(f"Submitted domain '{biz_domain}' does not match the domain most strongly supported by the evidence ('{official_domain_detected}'); visibility score capped at 70.")

    if not submitted_domain_cited:
        warnings.append("The submitted domain was never cited, so visibility and verification strength were reduced.")

    # Unclear rate penalties
    if unclear_rate > 0.50:
        warnings.append(f"A high share of answers were unclear ({unclear_rate:.1%}); visibility score capped at 65.")
    elif unclear_rate > 0.30:
        warnings.append(f"A meaningful share of answers were unclear ({unclear_rate:.1%}); visibility score capped at 75.")

    # Apply proportional accuracy reduction based on unclear rate
    accuracy_unclear_penalty = int(unclear_rate * 30)  # Max -30 points at 100% unclear
    accuracy -= accuracy_unclear_penalty

    # Apply entity-based score penalties
    if not submitted_domain_cited:
        discovery = min(discovery, 70)
        authority = min(authority, 60)
        accuracy = min(accuracy, 70)

    # Clamp all scores 0-100 after penalties
    discovery = _clamp_int(discovery, default=0, lo=0, hi=100)
    accuracy = _clamp_int(accuracy, default=0, lo=0, hi=100)
    authority = _clamp_int(authority, default=0, lo=0, hi=100)

    # Compute preliminary overall score
    overall, package, explanation, strategy = derive_recommendation(discovery, accuracy, authority)
    overall = _clamp_int(overall, default=50, lo=0, hi=100)

    # Apply entity status caps to overall score
    if entity_status in {"UNCLEAR", "MISMATCH"}:
        overall = min(overall, 60)

    if official_domain_detected and official_domain_detected != biz_domain:
        overall = min(overall, 70)

    # Apply unclear rate caps to overall score
    if unclear_rate > 0.50:
        overall = min(overall, 65)
    elif unclear_rate > 0.30:
        overall = min(overall, 75)

    # 90+ Score Decompression (make 90+ rare and earned)
    if overall >= 90:
        # Must meet ALL requirements for 90+
        if entity_status != "CONFIRMED":
            overall = min(overall, 85)
            warnings.append("A 90+ visibility score requires confirmed entity matching; capped at 85.")
        if submitted_domain_cite_count < 2:
            overall = min(overall, 85)
            warnings.append("A 90+ visibility score requires the official domain to be cited at least twice; capped at 85.")
        if len(uniq_domains) < 6:
            overall = min(overall, 85)
            warnings.append("A 90+ visibility score requires at least 6 unique citation domains; capped at 85.")
        if unclear_rate >= 0.20:
            overall = min(overall, 85)
            warnings.append("A 90+ visibility score requires fewer unclear answers; capped at 85.")

    # Final clamp
    overall = _clamp_int(overall, default=50, lo=0, hi=100)

    validator_supported_count = 0
    validator_total_count = 0
    for validation_run in raw_bundle["validation"]["runs"]:
        validation = validation_run.get("validation") or {}
        validator_total_count += 1
        verdict = str(validation.get("verdict") or "").lower()
        if verdict in {"supported", "mostly_supported"}:
            validator_supported_count += 1

    confidence_score, confidence_level = _build_visibility_confidence(
        query_success_count=raw_bundle["timings"]["query_success_count"],
        query_error_count=raw_bundle["timings"]["query_error_count"],
        entity_status=entity_status,
        entity_confidence=entity_confidence,
        unclear_rate=unclear_rate,
        citation_count=len(all_hits),
        unique_citation_domain_count=len(uniq_domains),
        cites_official_domain=bool(cites_official),
        validator_supported_count=validator_supported_count,
        validator_total_count=validator_total_count,
    )

    # -----------------------------
    # Metrics
    # -----------------------------
    scoring_started = time.perf_counter()
    metrics: Dict[str, Any] = {
        "engine": raw_bundle["engine"],
        "provider": "perplexity",
        "model": settings.PERPLEXITY_MODEL,
        "business_domain": biz_domain,
        "mentions_business_name": bool(mentions_name),
        "mentions_official_domain": bool(mentions_domain),
        "cites_official_domain": bool(cites_official),
        "citation_count": len(all_hits),
        "unique_citation_domains": uniq_domains,
        "unique_citation_domain_count": len(uniq_domains),
        "freshest_cited_days": freshest_days,
        "comprehensiveness": {
            "has_services": bool(has_services),
            "has_location": bool(has_location),
            "has_contact": bool(has_contact),
            "score_0_to_3": comprehensiveness_hits,
        },
        "proof_signals": proof_signals,
        "entity_identity": {
            "status": entity_status,
            "confidence": entity_confidence,
            "unclear_rate": unclear_rate,
            "submitted_domain_cite_count": submitted_domain_cite_count,
        },
        "query_health": {
            "success_count": raw_bundle["timings"]["query_success_count"],
            "error_count": raw_bundle["timings"]["query_error_count"],
        },
        "validator_summary": {
            "supported_count": validator_supported_count,
            "total_count": validator_total_count,
        },
        "visibility_confidence": {
            "score": confidence_score,
            "level": confidence_level,
        },
        "gating_warnings": warnings,
    }
    raw_bundle["timings"]["scoring_seconds"] = round(time.perf_counter() - scoring_started, 3)

    # -----------------------------
    # Evidence → Signals → Recommendations
    # -----------------------------
    try:
        from evidence_signals import build_evidence, build_signals
        from recommendation_rules import build_recommendations

        evidence = build_evidence(metrics)
        signals = build_signals(evidence=evidence, authority_score=int(authority))

        rec_bundle = build_recommendations(
            evidence=evidence,
            signals=signals,
            scores={"discovery": int(discovery), "accuracy": int(accuracy), "authority": int(authority)},
        )

        package = rec_bundle.recommended_package
        explanation = rec_bundle.package_explanation
        strategy = rec_bundle.strategy_summary

        metrics["recommendations"] = {
            "fix_now": [r.__dict__ for r in rec_bundle.fix_now],
            "maintain": [r.__dict__ for r in rec_bundle.maintain],
            "next_scan_focus": rec_bundle.next_scan_focus,
            "recommended_package": rec_bundle.recommended_package,
            "package_explanation": rec_bundle.package_explanation,
            "strategy_summary": rec_bundle.strategy_summary,
        }
        raw_bundle["recommendations"] = metrics["recommendations"]

    except Exception as e:
        # Never fail the scan due to rec engine
        metrics["recommendations_error"] = str(e)

    # -----------------------------
    # Findings (specific / variable)
    # -----------------------------
    public_summary = _build_public_summary(
        business_name=business_name,
        business_domain=biz_domain,
        overall_score=int(overall),
        confidence_level=confidence_level,
        query_success_count=raw_bundle["timings"]["query_success_count"],
        query_error_count=raw_bundle["timings"]["query_error_count"],
        entity_status=entity_status,
        entity_confidence=entity_confidence,
        cites_official_domain=bool(cites_official),
        has_services=bool(has_services),
        has_location=bool(has_location),
        has_contact=bool(has_contact),
        proof_signals=proof_signals,
        provider_results=provider_results,
        validator_supported_count=validator_supported_count,
        validator_total_count=validator_total_count,
    )

    findings: List[str] = []
    findings.append("VizAI assessed how AI systems identify this business, what they can verify, and where evidence is weak.")
    findings.append(public_summary["evidence_summary"])

    if cites_official:
        findings.append("AI responses cited the official site, which strengthens attribution and verification.")
    else:
        findings.append("AI responses did not cite the official site, so the model may be relying on weaker third-party sources.")

    findings.append(f"Unique citation domains observed: {len(uniq_domains)}.")

    if freshest_days is None:
        findings.append("Source freshness could not be checked because citation dates were not provided.")
    else:
        if freshest_days <= 30:
            findings.append(f"The freshest cited source appears recent at roughly {freshest_days} days old.")
        elif freshest_days <= 90:
            findings.append(f"The freshest cited source appears moderately current at roughly {freshest_days} days old.")
        else:
            findings.append(f"The freshest cited source appears stale at roughly {freshest_days} days old.")

    coverage_bits: List[str] = []
    if has_services:
        coverage_bits.append("services")
    if has_location:
        coverage_bits.append("location")
    if has_contact:
        coverage_bits.append("contact")
    findings.append(
        f"AI could clearly verify {len(coverage_bits)}/3 core coverage areas "
        f"({('/'.join(coverage_bits)) if coverage_bits else 'none'})."
    )

    recs = (metrics.get("recommendations") or {}).get("next_scan_focus")
    if isinstance(recs, list) and recs:
        findings.append(f"Highest-value next fix: {recs[0]}.")

    metrics["public_summary"] = {
        "visibility_status": public_summary["visibility_status"],
        "confidence_level": confidence_level,
        "evidence_summary": public_summary["evidence_summary"],
        "verified_facts": public_summary["verified_facts"],
        "unclear_facts": public_summary["unclear_facts"],
        "missing_signals": public_summary["missing_signals"],
        "limitations": public_summary["limitations"],
        "proof_signals": proof_signals,
    }

    raw_bundle["metrics"] = metrics
    raw_bundle["public_summary"] = metrics["public_summary"]
    raw_bundle["package"] = {
        "recommendation": package,
        "explanation": explanation,
        "strategy_summary": strategy,
    }
    raw_bundle["scores"] = {
        "discovery": int(discovery),
        "accuracy": int(accuracy),
        "authority": int(authority),
        "overall": int(overall),
    }
    raw_bundle["timings"]["total_seconds"] = round(time.perf_counter() - scan_started, 3)
    logger.info(
        "Scan engine timings: total=%ss query=%ss validation=%ss scoring=%ss",
        raw_bundle["timings"]["total_seconds"],
        raw_bundle["timings"]["query_execution_seconds"],
        raw_bundle["timings"]["validation_seconds"],
        raw_bundle["timings"]["scoring_seconds"],
    )

    result = RealScanResult(
        ai_visibility_score=int(overall),
        discovery_score=int(discovery),
        accuracy_score=int(accuracy),
        authority_score=int(authority),
        overall_score=int(overall),
        visibility_status=public_summary["visibility_status"],
        confidence_level=confidence_level,
        evidence_summary=public_summary["evidence_summary"],
        verified_facts=public_summary["verified_facts"],
        unclear_facts=public_summary["unclear_facts"],
        missing_signals=public_summary["missing_signals"],
        limitations=public_summary["limitations"],
        proof_signals=proof_signals,
        package_recommendation=package,
        package_explanation=explanation,
        strategy_summary=strategy,
        findings=findings,
        provider_results=provider_results,
        metrics=metrics,
        entity_status=entity_status,
        entity_confidence=entity_confidence,
        warnings=warnings if warnings else None,
    )

    return result, raw_bundle
