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
        "Do not guess. If in






