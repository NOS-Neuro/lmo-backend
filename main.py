import os
import json
import re
from typing import List, Optional, Dict, Any, Tuple
from urllib.parse import urljoin, urlparse

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, EmailStr

# -------------------------------------------------------------------
# Config / Environment
# -------------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

RESEND_API_KEY = os.getenv("RESEND_API_KEY")
NOTIFY_EMAIL_FROM = os.getenv("NOTIFY_EMAIL_FROM")
NOTIFY_EMAIL_TO = os.getenv("NOTIFY_EMAIL_TO")

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")  # optional but recommended

AUDIT_TIMEOUT_SECONDS = int(os.getenv("AUDIT_TIMEOUT_SECONDS", "12"))
AUDIT_MAX_PAGES = int(os.getenv("AUDIT_MAX_PAGES", "2"))  # homepage + about (best effort)

EMAIL_NOTIFICATIONS_ENABLED = bool(
    RESEND_API_KEY and NOTIFY_EMAIL_FROM and NOTIFY_EMAIL_TO
)

if not OPENAI_API_KEY:
    print("[VizAI] WARNING: OPENAI_API_KEY is not set. /run_scan will fail.")

if EMAIL_NOTIFICATIONS_ENABLED:
    print("[VizAI] Email notifications via Resend are ENABLED.")
else:
    print("[VizAI] Email notifications are DISABLED (missing env vars).")

if SERPAPI_API_KEY:
    print("[VizAI] SERPAPI is ENABLED (AI Search Test will include web retrieval).")
else:
    print("[VizAI] SERPAPI is DISABLED (AI Search Test will be limited to site + Wikipedia).")

# -------------------------------------------------------------------
# FastAPI app setup
# -------------------------------------------------------------------

app = FastAPI(title="VizAI Scan API")

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN] if FRONTEND_ORIGIN != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Models
# -------------------------------------------------------------------

class ScanRequest(BaseModel):
    businessName: str
    website: HttpUrl
    models: List[str] = []  # kept for future use
    contactEmail: Optional[EmailStr] = None


class ScanResponse(BaseModel):
    discovery_score: int
    accuracy_score: int
    authority_score: int
    overall_score: int
    package_recommendation: str
    package_explanation: str
    strategy_summary: str
    findings: List[str]


# -------------------------------------------------------------------
# Simple, explainable scoring helpers
# -------------------------------------------------------------------

def clamp_score(x: int) -> int:
    return max(0, min(100, int(x)))


def derive_recommendation(discovery: int, accuracy: int, authority: int):
    overall = int(round((discovery + accuracy + authority) / 3))

    package = "Standard LMO"
    explanation = ""
    strategy = ""

    if overall >= 80:
        package = "Basic LMO"
        explanation = (
            "Your AI visibility foundation is strong. Basic focuses on monitoring and light adjustments "
            "so your information stays consistent as models and search results evolve."
        )
        strategy = (
            "Lock in a verified Truth File, run monthly drift checks, and patch small discrepancies "
            "before they affect customer-facing answers."
        )
    elif overall >= 40:
        package = "Standard LMO"
        explanation = (
            "Your AI profile is partially correct but has gaps or inconsistencies. Standard is designed to "
            "close those gaps and raise confidence across AI systems."
        )
        strategy = (
            "Fix core facts (who you are, what you do, where you operate), then publish structured data "
            "and authoritative profiles to push scores into the 70–80 range."
        )
    else:
        package = "Standard LMO + Add-Ons"
        explanation = (
            "AI currently has a weak or fragmented view of your business. A deeper correction pass plus targeted "
            "add-ons will be needed to build strong discoverability and accuracy."
        )
        strategy = (
            "Establish a verified Truth File + structured schema, then add external profile cleanup and "
            "industry dataset work to accelerate discoverability."
        )

    return overall, package, explanation, strategy


# -------------------------------------------------------------------
# Module: Website Analyzer (real signals)
# -------------------------------------------------------------------

UA = {"User-Agent": "VizAI-AuditBot/1.0 (+https://vizai.io)"}

def fetch_url(url: str) -> Tuple[str, int]:
    resp = requests.get(url, headers=UA, timeout=AUDIT_TIMEOUT_SECONDS, allow_redirects=True)
    return resp.text, resp.status_code

def extract_jsonld(html: str) -> List[Dict[str, Any]]:
    blocks = []
    for m in re.finditer(r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', html, re.I | re.S):
        raw = m.group(1).strip()
        try:
            data = json.loads(raw)
            # json-ld can be object or list
            if isinstance(data, list):
                blocks.extend([d for d in data if isinstance(d, dict)])
            elif isinstance(data, dict):
                blocks.append(data)
        except Exception:
            continue
    return blocks

def has_schema_microdata(html: str) -> bool:
    # quick check for itemscope/itemtype or vocab schema.org
    return bool(re.search(r'\bitemscope\b|\bitemtype=["\']https?://schema\.org/|\bvocab=["\']https?://schema\.org', html, re.I))

def extract_meta(html: str, name: str) -> Optional[str]:
    # name="description" or property="og:title" etc.
    pattern = rf'<meta[^>]+(?:name|property)=["\']{re.escape(name)}["\'][^>]+content=["\'](.*?)["\']'
    m = re.search(pattern, html, re.I | re.S)
    return m.group(1).strip() if m else None

def extract_title(html: str) -> Optional[str]:
    m = re.search(r"<title>(.*?)</title>", html, re.I | re.S)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else None

def find_about_link(html: str, base_url: str) -> Optional[str]:
    # best-effort: find first href containing "about"
    for m in re.finditer(r'<a[^>]+href=["\'](.*?)["\']', html, re.I | re.S):




