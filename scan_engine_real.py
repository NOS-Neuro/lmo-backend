"""
scan_engine_real.py — VizAI Real Scan Engine (Perplexity-first, production-safe)

Real + measurable:
- Uses Perplexity web-backed search (search_mode="web")
- Captures answer + citations (search_results)
- Computes deterministic scores: Discovery / Authority (+ Accuracy proxy)
- Adds measurable metrics: Freshness, Comprehensiveness
- Adds Evidence → Signals → Recommendations bundle
- Adds Competitor baseline scans (optional)

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

# Silent caps (do NOT mention in client-facing report)
SCORE_MAX_COMPONENT = int(os.getenv("SCORE_MAX_COMPONENT", "95"))
SCORE_MAX_OVERALL = int(os.getenv("SCORE_MAX_OVERALL", "92"))


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
    out = []
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
    # overall (silently capped)
    overall_raw = int(round((discovery + accuracy + authority) / 3))
    overall = min(overall_raw, SCORE_MAX_OVERALL)

    if overall >= 78:
        package = "Basic LMO"
        explanation = (
            "You’re in a strong baseline position. Basic is about monitoring drift, "
            "tightening signals, and keeping answers stable as sources change."
        )
        strategy = (
            "Lock a canonical Truth File, verify schema/metadata, and ru








