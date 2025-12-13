"""
scan_engine_real.py — VizAI Real Scan Engine (Perplexity-first, production-safe MVP)

Real + measurable:
- Uses Perplexity web-backed search (search_mode="web")
- Captures answer + citations (search_results)
- Computes deterministic scores: Discovery / Accuracy(proxy) / Authority
- Adds measurable metrics: Freshness, Comprehensiveness
- Runs Evidence → Recommendation rules (deterministic)
- Returns (result, raw_bundle) safe to persist in JSONB

Honesty:
- Accuracy is a proxy until you add a ground-truth Truth File compare.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
PERPLEXITY_MODEL = os.getenv("PERPLEXITY_MODEL", "sonar-pro")
PERPLEXITY_TIMEOUT = int(os.getenv("PERPLEXITY_TIMEOUT", "45"))


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
            "tightening a few signals, and keeping answers stable as models and sources change."
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

    def chat_web(self, *, system: str, user: str, max_tokens: int = 600) -> Tuple[str, List[PerplexityHit], Dict[str, Any]]:
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

        r = requests.post(self.BASE_URL, headers=headers, json=payload, timeout=self.timeout)
        if r.status_code < 200 or r.status_code >= 300:
            raise RuntimeError(f"Perplexity API error {r.status_code}: {r.text}")

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

DEFAULT_QUESTIONS: List[Tuple[str, str]] = [
    ("baseline_overview", "In 3–6 bullets: what does this company do? Include official site + main services."),
    ("contact_path", "What is the best contact path (email/form/phone) from sources? If unknown, say unclear."),
    ("locations_scope", "Where does the company operate (regions/countries)? If unclear, say unclear."),
    ("proof_points", "List 3 proof points from sources (certifications, customers, industries, capabilities)."),
]

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
# Scoring helpers (NEW)
# -----------------------------

def _score_diversity(unique_domains: int) -> int:
    # 0..30
    if unique_domains <= 1:
        return 2
    if unique_domains <= 3:
        return 10
    if unique_domains <= 6:
        return 18
    if unique_domains <= 10:
        return 24
    return 30


def _score_volume(citation_count: int) -> int:
    # 0..15
    if citation_count <= 1:
        return 2
    if citation_count <= 4:
        return 7
    if citation_count <= 8:
        return 11
    return 15


def _score_freshness(freshest_days: Optional[int]) -> int:
    # 0..10
    if freshest_days is None:
        return 3
    if freshest_days <= 14:
        return 10
    if freshest_days <= 30:
        return 8
    if freshest_days <= 90:
        return 5
    return 2


def _score_coverage(has_services: bool, has_location: bool, has_contact: bool) -> int:
    # 0..15
    return int((has_services + has_location + has_contact) * 5)


def run_real_scan_perplexity(
    *,
    business_name: str,
    website: str,
    questions: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[RealScanResult, Dict[str, Any]]:

    client = PerplexityClient(
        api_key=PERPLEXITY_API_KEY,
        model=PERPLEXITY_MODEL,
        timeout=PERPLEXITY_TIMEOUT,
    )

    biz_domain = _domain(website)
    qs = questions or DEFAULT_QUESTIONS

    provider_results: List[ProviderResult] = []
    raw_bundle: Dict[str, Any] = {
        "engine": "real_perplexity_mvp_v3",
        "provider": "perplexity",
        "model": PERPLEXITY_MODEL,
        "runs": [],
        "notes": [
            "Discovery/Authority are evidence-based from returned citations.",
            "Accuracy is a proxy until a Truth File compare is implemented.",
        ],
    }

    system = (
        "You are an audit assistant using web search. "
        "Do not guess. If information is missing, say 'unclear'. "
        "Prefer citing the official website when available."
    )

    for prompt_name, q in qs:
        user = (
            f"Company name: {business_name}\n"
            f"Official website (given): {website}\n\n"
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
                model=PERPLEXITY_MODEL,
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
    official_cite_count = int(sum(1 for d in cite_domains if d == biz_domain)) if biz_domain else 0

    parsed_dates = [_parse_date(h.date) for h in all_hits]
    parsed_dates = [d for d in parsed_dates if d is not None]
    freshest_days = None
    if parsed_dates:
        newest = max(parsed_dates)
        now = datetime.now(timezone.utc)
        if newest.tzinfo is None:
            newest = newest.replace(tzinfo=timezone.utc)
        freshest_days = max(0, int((now - newest).total_seconds() // 86400))

    t = _norm(all_text)
    has_services = any(k in t for k in ["services", "solutions", "we provide", "offerings"])
    has_location = any(k in t for k in ["located", "based in", "headquartered", "operations", "serve"])
    has_contact = any(k in t for k in ["contact", "email", "phone", "reach", "sales"])
    comprehensiveness_hits = sum([has_services, has_location, has_contact])  # 0..3

    # -----------------------------
    # Deterministic scoring (FIXED)
    # -----------------------------

    diversity_pts = _score_diversity(len(uniq_domains))
    volume_pts = _score_volume(len(all_hits))
    freshness_pts = _score_freshness(freshest_days)
    coverage_pts = _score_coverage(has_services, has_location, has_contact)

    # Discovery (0..100): identity + official + evidence strength
    discovery = 0
    discovery += 25 if mentions_name else 0
    discovery += 15 if mentions_domain else 0
    discovery += 20 if cites_official else 0
    discovery += 10 if official_cite_count >= 2 else (5 if official_cite_count == 1 else 0)
    discovery += diversity_pts  # up to 30
    discovery = _clamp_int(discovery, default=0)

    # Accuracy (proxy) (0..100): official source + coverage + freshness + penalties
    accuracy = 0
    accuracy += 35 if cites_official else 15
    accuracy += 15 if mentions_domain else 0
    accuracy += 10 if mentions_name else 0
    accuracy += coverage_pts          # up to 15
    accuracy += freshness_pts         # up to 10

    # penalties for missing core fields
    if not has_services:
        accuracy -= 8
    if not has_location:
        accuracy -= 6
    if not has_contact:
        accuracy -= 6

    # penalty if lots of other sources but not official
    if len(uniq_domains) >= 6 and not cites_official:
        accuracy -= 12

    accuracy = _clamp_int(accuracy, default=50)

    # Authority (0..100): official + diversity + bonus domains + freshness
    authority = 0
    authority += 30 if cites_official else 10
    authority += min(25, len(uniq_domains) * 3)  # 0..25
    bonus = 0
    for d in uniq_domains:
        bonus += AUTHORITY_DOMAIN_BONUS.get(d, 0)
    authority += _clamp_int(bonus, default=0, lo=0, hi=25)  # up to 25
    authority += freshness_pts  # up to 10
    authority += 10 if official_cite_count >= 2 else (5 if official_cite_count == 1 else 0)
    authority = _clamp_int(authority, default=50)

    overall, package, explanation, strategy = derive_recommendation(discovery, accuracy, authority)

    findings: List[str] = []
    findings.append("Real scan: web-backed answers + citations captured for auditability.")
    findings.append(f"Official site cited: {'yes' if cites_official else 'no'} (count: {official_cite_count})")
    findings.append(f"Unique citation domains: {len(uniq_domains)}")
    if freshest_days is not None:
        findings.append(f"Freshest cited source: ~{freshest_days} days old")
    else:
        findings.append("Freshness: citation dates not provided by sources")
    findings.append(f"Comprehensiveness signals: {comprehensiveness_hits}/3 (services/location/contact)")

    metrics = {
        "engine": raw_bundle["engine"],
        "provider": "perplexity",
        "model": PERPLEXITY_MODEL,
        "business_domain": biz_domain,
        "mentions_business_name": bool(mentions_name),
        "mentions_official_domain": bool(mentions_domain),
        "cites_official_domain": bool(cites_official),
        "official_citation_count": official_cite_count,
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
        "score_components": {
            "diversity_pts": diversity_pts,
            "volume_pts": volume_pts,
            "freshness_pts": freshness_pts,
            "coverage_pts": coverage_pts,
        }
    }

    # -----------------------------
    # Evidence → Recommendation Engine (deterministic)
    # -----------------------------
    try:
        from evidence_signals import build_evidence, build_signals
        from recommendation_rules import build_recommendations

        evidence = build_evidence(metrics)
        signals = build_signals(evidence=evidence, authority_score=authority)

        rec_bundle = build_recommendations(
            evidence=evidence,
            signals=signals,
            scores={"discovery": discovery, "accuracy": accuracy, "authority": authority},
        )

        metrics["recommendations"] = {
            "fix_now": [r.__dict__ for r in rec_bundle.fix_now],
            "maintain": [r.__dict__ for r in rec_bundle.maintain],
            "next_scan_focus": rec_bundle.next_scan_focus,
        }
        raw_bundle["recommendations"] = metrics["recommendations"]
    except Exception:
        metrics["recommendations"] = {"fix_now": [], "maintain": [], "next_scan_focus": []}
        raw_bundle["recommendations"] = metrics["recommendations"]

    result = RealScanResult(
        discovery_score=discovery,
        accuracy_score=accuracy,
        authority_score=authority,
        overall_score=overall,
        package_recommendation=package,
        package_explanation=explanation,
        strategy_summary=strategy,
        findings=findings,
        provider_results=provider_results,
        metrics=metrics,
    )

    raw_bundle["metrics"] = metrics
    raw_bundle["package"] = {
        "recommendation": package,
        "explanation": explanation,
        "strategy_summary": strategy,
    }
    raw_bundle["scores"] = {
        "discovery": discovery,
        "accuracy": accuracy,
        "authority": authority,
        "overall": overall,
    }

    return result, raw_bundle







