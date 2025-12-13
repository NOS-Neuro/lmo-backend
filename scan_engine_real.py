"""
scan_engine_real.py — VizAI Real Scan Engine (Perplexity-first, production-safe)

Real + measurable:
- Uses Perplexity web-backed search
- Captures answers + citations
- Computes deterministic scores with ceilings (not exposed to UI)
- Builds Evidence → Signals → Recommendations
- Produces tailored findings + strategy per business
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests

from evidence_signals import build_evidence, build_signals
from recommendation_rules import build_recommendations


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
PERPLEXITY_MODEL = os.getenv("PERPLEXITY_MODEL", "sonar-pro")
PERPLEXITY_TIMEOUT = int(os.getenv("PERPLEXITY_TIMEOUT", "45"))


# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def _contains_name(text: str, business_name: str) -> bool:
    t = _norm(text)
    name = _norm(business_name)
    if not name:
        return False
    if name in t:
        return True
    tokens = [x for x in re.split(r"[^a-z0-9]+", name) if len(x) >= 4]
    return all(tok in t for tok in tokens[:3])


def _unique(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _parse_date(d: Optional[str]) -> Optional[datetime]:
    if not d:
        return None
    try:
        return datetime.fromisoformat(d.replace("Z", "+00:00"))
    except Exception:
        return None


# ---------------------------------------------------------------------
# Authority bonus domains
# ---------------------------------------------------------------------

AUTHORITY_DOMAIN_BONUS = {
    "wikipedia.org": 10,
    "linkedin.com": 6,
    "crunchbase.com": 6,
    "bloomberg.com": 8,
    "reuters.com": 8,
    "sec.gov": 10,
    "sedarplus.ca": 10,
}


# ---------------------------------------------------------------------
# Perplexity Client
# ---------------------------------------------------------------------

class PerplexityClient:
    BASE_URL = "https://api.perplexity.ai/chat/completions"

    def __init__(self, api_key: str, model: str, timeout: int):
        if not api_key:
            raise RuntimeError("PERPLEXITY_API_KEY not set")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def chat_web(self, system: str, user: str, max_tokens: int = 650):
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
        }

        r = requests.post(self.BASE_URL, headers=headers, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()

        answer = data["choices"][0]["message"]["content"]
        hits = [
            PerplexityHit(
                title=h.get("title", ""),
                url=h.get("url", ""),
                date=h.get("date"),
            )
            for h in (data.get("search_results") or [])
        ]
        return answer, hits, data


# ---------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------

DEFAULT_QUESTIONS = [
    ("overview", "In 3–6 bullets: what does this company do? Cite sources."),
    ("contact", "What is the best contact path? If unclear, say unclear."),
    ("locations", "Where does the company operate? If unclear, say unclear."),
    ("proof", "List 3 proof points (certifications, industries, capabilities)."),
]


# ---------------------------------------------------------------------
# Main Scan Function
# ---------------------------------------------------------------------

def run_real_scan_perplexity(*, business_name: str, website: str):
    client = PerplexityClient(PERPLEXITY_API_KEY, PERPLEXITY_MODEL, PERPLEXITY_TIMEOUT)
    biz_domain = _domain(website)

    provider_results: List[ProviderResult] = []
    raw_bundle: Dict[str, Any] = {
        "engine": "real_perplexity_v3",
        "provider": "perplexity",
        "model": PERPLEXITY_MODEL,
        "runs": [],
    }

    system_prompt = (
        "You are an audit assistant using web search. "
        "Do not guess. Cite sources. If unclear, say unclear."
    )

    for prompt_name, question in DEFAULT_QUESTIONS:
        user_prompt = (
            f"Company: {business_name}\n"
            f"Official site: {website}\n\n"
            f"Task: {question}"
        )

        answer, hits, raw = client.chat_web(system_prompt, user_prompt)

        provider_results.append(
            ProviderResult(
                provider="perplexity",
                model=PERPLEXITY_MODEL,
                prompt_name=prompt_name,
                question=question,
                answer_text=answer,
                citations=hits,
            )
        )

        raw_bundle["runs"].append(
            {
                "prompt_name": prompt_name,
                "answer_text": answer,
                "search_results": [h.__dict__ for h in hits],
                "raw": raw,
            }
        )

    # -----------------------------------------------------------------
    # Aggregate Evidence
    # -----------------------------------------------------------------

    all_text = "\n".join(r.answer_text for r in provider_results)
    all_hits = [h for r in provider_results for h in r.citations]

    cite_domains = [_domain(h.url) for h in all_hits if h.url]
    uniq_domains = _unique(cite_domains)

    mentions_name = _contains_name(all_text, business_name)
    mentions_domain = biz_domain in _norm(all_text) if biz_domain else False
    cites_official = biz_domain in uniq_domains if biz_domain else False

    dates = [_parse_date(h.date) for h in all_hits if h.date]
    dates = [d for d in dates if d]
    freshest_days = None
    if dates:
        newest = max(dates)
        freshest_days = int((datetime.now(timezone.utc) - newest).days)

    t = _norm(all_text)
    has_services = any(k in t for k in ["services", "solutions", "offer"])
    has_location = any(k in t for k in ["located", "based", "serve"])
    has_contact = any(k in t for k in ["contact", "email", "phone"])

    coverage_hits = sum([has_services, has_location, has_contact])

    # -----------------------------------------------------------------
    # Scoring (with ceilings, not exposed)
    # -----------------------------------------------------------------

    confidence = 0.78
    confidence += 0.07 if cites_official else 0
    confidence += 0.04 if mentions_domain else 0
    confidence += 0.04 if mentions_name else 0
    confidence += 0.03 * (coverage_hits / 3)
    confidence += 0.04 if len(uniq_domains) >= 8 else 0
    confidence = min(confidence, 0.97)

    discovery = round((40 if mentions_name else 0
                       + 20 if mentions_domain else 0
                       + 35 if cites_official else 0
                       + coverage_hits * 3) * confidence)

    accuracy = round((55 if cites_official else 20
                      + 15 if mentions_domain else 0
                      + 10 if mentions_name else 0
                      + 10 if coverage_hits >= 2 else 0) * confidence)

    authority = 45 if cites_official else 10
    authority += min(25, len(uniq_domains) * 3)
    authority += min(15, sum(AUTHORITY_DOMAIN_BONUS.get(d, 0) for d in uniq_domains))
    authority = round(authority * confidence)

    discovery = _clamp(discovery, 0, 95)
    accuracy = _clamp(accuracy, 0, 95)
    authority = _clamp(authority, 0, 95)

    overall = min(round((discovery + accuracy + authority) / 3), 92)

    # -----------------------------------------------------------------
    # Evidence → Signals → Recommendations
    # -----------------------------------------------------------------

    metrics = {
        "business_domain": biz_domain,
        "mentions_business_name": mentions_name,
        "mentions_official_domain": mentions_domain,
        "cites_official_domain": cites_official,
        "citation_count": len(all_hits),
        "unique_citation_domains": uniq_domains,
        "freshest_cited_days": freshest_days,
        "comprehensiveness": {
            "has_services": has_services,
            "has_location": has_location,
            "has_contact": has_contact,
        },
    }

    evidence = build_evidence(metrics)
    signals = build_signals(evidence=evidence, authority_score=authority)
    rec_bundle = build_recommendations(
        evidence=evidence,
        signals=signals,
        scores={"discovery": discovery, "accuracy": accuracy, "authority": authority},
    )

    # -----------------------------------------------------------------
    # Tailored Findings
    # -----------------------------------------------------------------

    findings: List[str] = [
        "Real scan: web-backed answers and citations captured."
    ]

    findings.append(f"Official site cited: {'yes' if cites_official else 'no'}")

    if len(uniq_domains) >= 10:
        findings.append(f"Strong authority footprint: {len(uniq_domains)} unique citation domains.")
    elif len(uniq_domains) >= 5:
        findings.append(f"Moderate authority footprint: {len(uniq_domains)} unique citation domains.")
    else:
        findings.append(f"Limited authority footprint: only {len(uniq_domains)} citation domains.")

    if freshest_days is None:
        findings.append("Freshness could not be confirmed from citation dates.")
    elif freshest_days <= 30:
        findings.append(f"Fresh sources detected (newest ~{freshest_days} days old).")
    else:
        findings.append(f"Sources appear stale (newest ~{freshest_days} days old).")

    if coverage_hits == 3:
        findings.append("Coverage is complete: services, location, and contact surfaced.")
    else:
        missing = [k for k, v in {
            "services": has_services,
            "location": has_location,
            "contact": has_contact,
        }.items() if not v]
        findings.append(f"Coverage gaps detected: {', '.join(missing)}.")

    # -----------------------------------------------------------------
    # Tailored Strategy Summary
    # -----------------------------------------------------------------

    focus = rec_bundle.next_scan_focus or []
    top_fixes = [r.title for r in rec_bundle.fix_now[:2]]

    strategy_parts = []
    if focus:
        strategy_parts.append("Next focus: " + ", ".join(focus) + ".")
    if top_fixes:
        strategy_parts.append("Top fixes: " + " / ".join(top_fixes) + ".")
    strategy_parts.append("Re-scan after updates to confirm stability.")

    strategy = " ".join(strategy_parts)

    # Package text (unchanged semantics)
    if overall >= 80:
        package = "Basic LMO"
        explanation = "Your AI visibility is strong. Focus on monitoring and drift prevention."
    elif overall >= 40:
        package = "Standard LMO"
        explanation = "AI visibility is partial. Address gaps to stabilize answers."
    else:
        package = "Standard LMO + Add-Ons"
        explanation = "AI visibility is weak. Foundational correction is required."

    raw_bundle["metrics"] = metrics
    raw_bundle["recommendations"] = {
        "fix_now": [r.__dict__ for r in rec_bundle.fix_now],
        "maintain": [r.__dict__ for r in rec_bundle.maintain],
        "next_scan_focus": rec_bundle.next_scan_focus,
    }
    raw_bundle["scores"] = {
        "discovery": discovery,
        "accuracy": accuracy,
        "authority": authority,
        "overall": overall,
    }

    return RealScanResult(
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
    ), raw_bundle




