"""
scan_engine_real.py — VizAI Real Scan Engine (Perplexity-first, RAS v2)
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


# -----------------------------
# Helpers
# -----------------------------

def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().lstrip("www.")
    except Exception:
        return ""


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower().strip())


def _contains_name(text: str, name: str) -> bool:
    t = _norm(text)
    n = _norm(name)
    return n in t if n else False


# -----------------------------
# Perplexity Client
# -----------------------------

class PerplexityClient:
    BASE_URL = "https://api.perplexity.ai/chat/completions"

    def __init__(self, api_key: str):
        if not api_key:
            raise RuntimeError("PERPLEXITY_API_KEY not set")
        self.api_key = api_key

    def chat(self, system: str, user: str) -> Tuple[str, List[PerplexityHit]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": PERPLEXITY_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "search_mode": "web",
            "temperature": 0.2,
        }

        r = requests.post(self.BASE_URL, headers=headers, json=payload, timeout=PERPLEXITY_TIMEOUT)
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

        return answer, hits


# -----------------------------
# Main Scan
# -----------------------------

def run_real_scan_perplexity(*, business_name: str, website: str):
    client = PerplexityClient(PERPLEXITY_API_KEY)
    biz_domain = _domain(website)

    system = (
        "You are an audit assistant using web search. "
        "Do not guess. Cite sources."
    )

    prompts = [
        "What does this company do?",
        "Where does it operate?",
        "What services does it offer?",
        "How can someone contact it?",
    ]

    all_text = ""
    all_hits: List[PerplexityHit] = []

    for p in prompts:
        user = f"Company: {business_name}\nWebsite: {website}\n\n{p}"
        answer, hits = client.chat(system, user)
        all_text += "\n" + answer
        all_hits.extend(hits)

    cite_domains = [_domain(h.url) for h in all_hits if h.url]
    uniq_domains = list(set(cite_domains))

    mentions_name = _contains_name(all_text, business_name)
    cites_official = biz_domain in uniq_domains

    discovery = int(40 + (30 if mentions_name else 0) + (30 if cites_official else 0))
    accuracy = int(40 + (30 if cites_official else 0) + (30 if mentions_name else 0))
    authority = min(100, int(len(uniq_domains) * 8))

    overall = int(round((discovery + accuracy + authority) / 3))

    package = "Standard LMO" if overall < 80 else "Basic LMO"

    findings = [
        f"Official site cited: {'yes' if cites_official else 'no'}",
        f"Unique citation domains: {len(uniq_domains)}",
    ]

    result = RealScanResult(
        discovery_score=discovery,
        accuracy_score=accuracy,
        authority_score=authority,
        overall_score=overall,
        package_recommendation=package,
        package_explanation="Based on evidence alignment and citation quality.",
        strategy_summary="Strengthen canonical signals and authoritative citations.",
        findings=findings,
    )

    raw_bundle = {
        "engine": "real_perplexity_ras_v2",
        "business_domain": biz_domain,
        "citations": [h.__dict__ for h in all_hits],
        "scores": {
            "discovery": discovery,
            "accuracy": accuracy,
            "authority": authority,
            "overall": overall,
        },
    }

    return result, raw_bundle



