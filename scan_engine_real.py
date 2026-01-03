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
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from requests.exceptions import Timeout, RequestException

from core.prompts import DEFAULT_QUESTIONS
from config import settings


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
    discovery_score: int
    accuracy_score: int
    authority_score: int
    overall_score: int

    package_recommendation: str
    package_explanation: str
    strategy_summary: str

    findings: List[str]
    provider_results: List[ProviderResult]
    metrics: Dict[str, Any]


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
        host = urlparse(url).netloc.lower().strip()
        return host.lstrip("www.")
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


# -----------------------------
# Package recommendation logic
# -----------------------------

def derive_recommendation(discovery: int, accuracy: int, authority: int) -> Tuple[int, str, str, str]:
    overall = int(round((discovery + accuracy + authority) / 3))

    if overall >= 80:
        package = "Basic LMO"
        explanation = (
            "You’re in a strong baseline position. Basic is about monitoring drift, "
            "tightening a few signals, and keeping answers stable as sources change."
        )
        strategy = (
            "Lock a canonical Truth File, verify schema/metadata, and run scheduled rechecks "
            "to catch drift early. Add 1–2 authority assets if needed."
        )
    elif overall >= 40:
        package = "Standard LMO"
        explanation = (
            "AI can likely find you, but gaps or inconsistencies reduce reliability. "
            "Standard closes gaps and strengthens signals AI uses to describe you correctly."
        )
        strategy = (
            "Improve About/Services/FAQ, deploy schema, and seed a small set of authoritative profiles. "
            "Then re-scan and compare deltas."
        )
    else:
        package = "Standard LMO + Add-Ons"
        explanation = (
            "AI visibility is weak or fragmented. You’ll need foundational correction plus targeted "
            "authority building to correct the record quickly."
        )
        strategy = (
            "Start with a Truth File + schema deployment, then add authority seeding and directory cleanup. "
            "Re-scan weekly until stable."
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
        max_retries: int = 3,
    ) -> Tuple[str, List[PerplexityHit], Dict[str, Any]]:
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
    competitors: Optional[List[Dict[str, Any]]] = None,
    questions: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[RealScanResult, Dict[str, Any]]:
    """
    Returns:
      (RealScanResult, raw_bundle)

    competitors:
      Accepted and stored for future use, but not executed here unless you explicitly want it.
    """

    client = PerplexityClient(
        api_key=settings.PERPLEXITY_API_KEY,
        model=settings.PERPLEXITY_MODEL,
        timeout=settings.PERPLEXITY_TIMEOUT,
    )

    biz_domain = _domain(website)
    qs = questions or DEFAULT_QUESTIONS

    provider_results: List[ProviderResult] = []
    raw_bundle: Dict[str, Any] = {
        "engine": "real_perplexity_mvp_v3",
        "provider": "perplexity",
        "model": settings.PERPLEXITY_MODEL,
        "runs": [],
        "notes": [
            "Discovery/Authority are evidence-based from returned citations.",
            "Accuracy is a proxy until a Truth File compare is implemented.",
        ],
        "competitors": competitors or [],
    }

    system = (
        "You are an audit assistant using web search. "
        "Do not guess. If information is missing, say 'unclear'. "
        "Search for official and authoritative sources."
    )

    for prompt_name, q in qs:
        # Don't mention the website - let Perplexity discover it naturally
        user = (
            f"Company name: {business_name}\n\n"
            f"Task: {q}\n\n"
            f"Rules:\n"
            f"- Only state facts supported by citations.\n"
            f"- If uncertain, say 'unclear'.\n"
            f"- Keep output concise.\n"
        )

        answer, hits, raw = client.chat_web(system=system, user=user, max_tokens=650)

        provider_results.append(
            ProviderResult(
                provider="perplexity",
                model=settings.PERPLEXITY_MODEL,
                prompt_name=prompt_name,
                question=q,
                answer_text=answer,
                citations=hits,
            )
        )

        raw_bundle["runs"].append(
            {
                "prompt_name": prompt_name,
                "question": q,
                "answer_text": answer,
                "search_results": [h.__dict__ for h in hits],
                "raw": raw,
            }
        )

    # -----------------------------
    # Aggregate evidence
    # -----------------------------
    all_text = "\n".join(r.answer_text for r in provider_results)

    all_hits: List[PerplexityHit] = []
    for r in provider_results:
        all_hits.extend(r.citations)

    cite_domains = [_domain(h.url) for h in all_hits if h.url]
    uniq_domains = _unique([d for d in cite_domains if d])

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

    # Comprehensiveness heuristics
    t = _norm(all_text)
    has_services = any(k in t for k in ["services", "solutions", "we provide", "offerings"])
    has_location = any(k in t for k in ["located", "based in", "headquartered", "operations", "serve"])
    has_contact = any(k in t for k in ["contact", "email", "phone", "reach", "sales"])
    comprehensiveness_hits = int(has_services) + int(has_location) + int(has_contact)

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

    # Internal caps (do NOT mention in findings)
    discovery = min(discovery, 95)
    accuracy = min(accuracy, 95)
    authority = min(authority, 95)

    overall, package, explanation, strategy = derive_recommendation(discovery, accuracy, authority)
    overall = min(overall, 95)

    # -----------------------------
    # Metrics
    # -----------------------------
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
    }

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

        metrics["recommendations"] = {
            "fix_now": [r.__dict__ for r in rec_bundle.fix_now],
            "maintain": [r.__dict__ for r in rec_bundle.maintain],
            "next_scan_focus": rec_bundle.next_scan_focus,
        }
        raw_bundle["recommendations"] = metrics["recommendations"]

    except Exception as e:
        # Never fail the scan due to rec engine
        metrics["recommendations_error"] = str(e)

    # -----------------------------
    # Findings (specific / variable)
    # -----------------------------
    findings: List[str] = []
    findings.append("Real scan completed with web-backed answers + captured citations.")

    if cites_official:
        findings.append("Official site is being cited by sources (good canonical signal).")
    else:
        findings.append("Official site was NOT cited — AI may rely on third-party sources.")

    findings.append(f"Unique citation domains observed: {len(uniq_domains)}")

    if freshest_days is None:
        findings.append("Freshness signals: citation dates not provided by sources.")
    else:
        if freshest_days <= 30:
            findings.append(f"Freshness signals: recent (~{freshest_days} days).")
        elif freshest_days <= 90:
            findings.append(f"Freshness signals: moderately current (~{freshest_days} days).")
        else:
            findings.append(f"Freshness signals: stale (~{freshest_days} days).")

    coverage_bits: List[str] = []
    if has_services:
        coverage_bits.append("services")
    if has_location:
        coverage_bits.append("location")
    if has_contact:
        coverage_bits.append("contact")
    findings.append(
        f"Coverage signals detected: {len(coverage_bits)}/3 "
        f"({('/'.join(coverage_bits)) if coverage_bits else 'none'})."
    )

    recs = (metrics.get("recommendations") or {}).get("next_scan_focus")
    if isinstance(recs, list) and recs:
        findings.append(f"Next focus: {recs[0]}.")

    raw_bundle["metrics"] = metrics
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

    result = RealScanResult(
        discovery_score=int(discovery),
        accuracy_score=int(accuracy),
        authority_score=int(authority),
        overall_score=int(overall),
        package_recommendation=package,
        package_explanation=explanation,
        strategy_summary=strategy,
        findings=findings,
        provider_results=provider_results,
        metrics=metrics,
    )

    return result, raw_bundle
