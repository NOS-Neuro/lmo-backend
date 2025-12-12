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
        href = m.group(1)
        if "about" in href.lower():
            return urljoin(base_url, href)
    return None

def website_audit(website: str) -> Dict[str, Any]:
    base = website.rstrip("/") + "/"
    html_home, code = fetch_url(base)
    jsonld = extract_jsonld(html_home)
    schema_microdata = has_schema_microdata(html_home)

    og_title = extract_meta(html_home, "og:title")
    og_desc = extract_meta(html_home, "og:description")
    meta_desc = extract_meta(html_home, "description")
    title = extract_title(html_home)

    about_url = find_about_link(html_home, base)
    about_summary = {"found": False}
    if about_url and AUDIT_MAX_PAGES >= 2:
        try:
            html_about, about_code = fetch_url(about_url)
            text = re.sub(r"<[^>]+>", " ", html_about)
            words = [w for w in re.split(r"\s+", text) if w]
            about_summary = {
                "found": about_code == 200,
                "url": about_url,
                "word_count": len(words),
            }
        except Exception:
            about_summary = {"found": False, "url": about_url}

    # detect key schema types
    schema_types = []
    for b in jsonld:
        t = b.get("@type")
        if isinstance(t, list):
            schema_types.extend([str(x) for x in t])
        elif t:
            schema_types.append(str(t))

    return {
        "homepage_status": code,
        "title": title,
        "meta_description": meta_desc,
        "open_graph": {"og:title": og_title, "og:description": og_desc},
        "jsonld_found": len(jsonld) > 0,
        "jsonld_types": sorted(list(set(schema_types))),
        "schema_microdata_found": schema_microdata,
        "about_page": about_summary,
    }

def structure_score(audit: Dict[str, Any]) -> int:
    score = 0

    if audit.get("jsonld_found"):
        score += 35
        types = audit.get("jsonld_types") or []
        if any(t.lower() in ["organization", "corporation", "localbusiness"] for t in [x.lower() for x in types]):
            score += 15
    if audit.get("schema_microdata_found"):
        score += 10

    meta_desc = audit.get("meta_description")
    if meta_desc and len(meta_desc) >= 50:
        score += 10

    og = audit.get("open_graph") or {}
    if og.get("og:title") and og.get("og:description"):
        score += 10

    about = audit.get("about_page") or {}
    if about.get("found"):
        score += 10
        if about.get("word_count", 0) >= 250:
            score += 10

    return clamp_score(score)


# -------------------------------------------------------------------
# Module: External Presence (safe)
# -------------------------------------------------------------------

def wikipedia_check(company_name: str) -> Dict[str, Any]:
    # Wikipedia search API
    try:
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": company_name,
            "format": "json",
        }
        r = requests.get(url, params=params, headers=UA, timeout=AUDIT_TIMEOUT_SECONDS)
        data = r.json()
        results = (data.get("query", {}).get("search", []) or [])
        if not results:
            return {"found": False}

        # best effort exact-ish match: first result
        top = results[0]
        title = top.get("title")
        return {
            "found": True,
            "title": title,
            "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}" if title else None,
        }
    except Exception as e:
        return {"found": False, "error": str(e)}

def authority_score(presence: Dict[str, Any], has_search: bool) -> int:
    score = 0
    wiki = presence.get("wikipedia") or {}
    if wiki.get("found"):
        score += 45

    # Search-enabled audits can check more third party references;
    # if search is available, we award potential authority signals later.
    if has_search:
        score += 15  # indicates broader ecosystem check is possible

    return clamp_score(score)


# -------------------------------------------------------------------
# Module: Optional Web Search Retrieval (SerpAPI)
# -------------------------------------------------------------------

def serpapi_search(query: str, num: int = 5) -> List[str]:
    if not SERPAPI_API_KEY:
        return []
    try:
        r = requests.get(
            "https://serpapi.com/search.json",
            params={"engine": "google", "q": query, "api_key": SERPAPI_API_KEY, "num": num},
            timeout=AUDIT_TIMEOUT_SECONDS,
        )
        data = r.json()
        links = []
        for item in (data.get("organic_results") or [])[:num]:
            link = item.get("link")
            if link:
                links.append(link)
        return links
    except Exception:
        return []

def fetch_text_snippet(url: str, max_chars: int = 1800) -> str:
    try:
        html, status = fetch_url(url)
        if status != 200:
            return ""
        text = re.sub(r"<script.*?</script>", " ", html, flags=re.S | re.I)
        text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]
    except Exception:
        return ""


# -------------------------------------------------------------------
# Module: AI Summary (grounded, no guessing)
# -------------------------------------------------------------------

def call_openai_json(system: str, user: str) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
    }
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=40,
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {resp.status_code} {resp.text}")
    content = resp.json()["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse LLM JSON: {str(e)} | content={content}")

def ai_visibility_summary(
    business_name: str,
    website: str,
    website_audit_data: Dict[str, Any],
    presence: Dict[str, Any],
    retrieved_sources: List[Dict[str, str]],
) -> Dict[str, Any]:
    # Build a compact evidence block
    evidence_lines = []
    evidence_lines.append(f"Official website: {website}")
    evidence_lines.append(f"Website audit: jsonld_found={website_audit_data.get('jsonld_found')}, jsonld_types={website_audit_data.get('jsonld_types')}, about_found={website_audit_data.get('about_page', {}).get('found')}")
    wiki = presence.get("wikipedia") or {}
    evidence_lines.append(f"Wikipedia: found={wiki.get('found')}, url={wiki.get('url')}")

    # Sources (snippets)
    src_block = []
    for s in retrieved_sources[:8]:
        if s.get("url") and s.get("snippet"):
            src_block.append(f"- {s['url']}\n  snippet: {s['snippet']}")
    src_text = "\n".join(src_block) if src_block else "(no third-party snippets available)"

    user_prompt = f"""
You are producing an AI Visibility Audit summary for a business.
You MUST use ONLY the evidence provided below. If the evidence is insufficient, say so clearly.
Do not invent facts.

Business: {business_name}
Website: {website}

EVIDENCE (facts you may use):
{chr(10).join(evidence_lines)}

THIRD-PARTY SOURCE SNIPPETS (may contain noise):
{src_text}

Return ONE JSON object with exactly:
{{
  "discovery_findings": [ "<short bullet>", ... ],
  "accuracy_findings": [ "<short bullet>", ... ],
  "authority_findings": [ "<short bullet>", ... ],
  "key_gaps": [ "<short bullet>", ... ]
}}

Rules:
- Bullets must be short and readable.
- If you can’t confirm something from evidence, mark it as missing/unclear rather than guessing.
""".strip()

    system = "You only output valid JSON. No extra text."
    return call_openai_json(system=system, user=user_prompt)


# -------------------------------------------------------------------
# Real Scan Orchestrator (v1)
# -------------------------------------------------------------------

def run_real_audit(business_name: str, website: str) -> ScanResponse:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured on the server.")

    # 1) Website audit (deterministic)
    w_audit = website_audit(website)
    acc = structure_score(w_audit)

    # 2) External presence (safe)
    presence = {
        "wikipedia": wikipedia_check(business_name),
    }
    auth = authority_score(presence, has_search=bool(SERPAPI_API_KEY))

    # 3) Optional retrieval for “AI search test”
    retrieved_sources: List[Dict[str, str]] = []

    # Always include homepage snippet (official)
    home_snip = fetch_text_snippet(website.rstrip("/") + "/")
    if home_snip:
        retrieved_sources.append({"url": website.rstrip("/") + "/", "snippet": home_snip})

    # Add wiki snippet if found
    wiki_url = (presence.get("wikipedia") or {}).get("url")
    if wiki_url:
        wiki_snip = fetch_text_snippet(wiki_url)
        if wiki_snip:
            retrieved_sources.append({"url": wiki_url, "snippet": wiki_snip})

    # Add third-party snippets via search (if enabled)
    if SERPAPI_API_KEY:
        # a) “who is” query
        q1 = f"{business_name} {website}"
        # b) company + reviews / profile type
        q2 = f"{business_name} company"
        links = []
        links.extend(serpapi_search(q1, num=5))
        links.extend(serpapi_search(q2, num=5))

        # de-duplicate + avoid obvious repeats
        seen = set()
        clean_links = []
        for u in links:
            if not u:
                continue
            host = urlparse(u).netloc.lower()
            if u in seen:
                continue
            # avoid fetching huge junk; keep a few varied domains
            if host in seen:
                continue
            seen.add(u)
            seen.add(host)
            clean_links.append(u)
            if len(clean_links) >= 6:
                break

        for u in clean_links:
            snip = fetch_text_snippet(u)
            if snip:
                retrieved_sources.append({"url": u, "snippet": snip})

    # 4) Grounded AI summary based on evidence + snippets (no “pretend models”)
    summary = ai_visibility_summary(
        business_name=business_name,
        website=website,
        website_audit_data=w_audit,
        presence=presence,
        retrieved_sources=retrieved_sources,
    )

    # 5) Discovery score (explainable, not random)
    # Simple v1 logic:
    # - if we have at least 2 usable sources (official + one external), discovery is higher
    usable_sources = [s for s in retrieved_sources if s.get("snippet")]
    disc = 30
    if len(usable_sources) >= 2:
        disc += 30
    if len(usable_sources) >= 4:
        disc += 20
    if (presence.get("wikipedia") or {}).get("found"):
        disc += 10
    if SERPAPI_API_KEY:
        disc += 10
    disc = clamp_score(disc)

    # 6) Findings (merge into your existing response model)
    findings: List[str] = []
    for b in (summary.get("discovery_findings") or [])[:2]:
        findings.append(f"Discovery: {b}")
    for b in (summary.get("accuracy_findings") or [])[:2]:
        findings.append(f"Accuracy: {b}")
    for b in (summary.get("authority_findings") or [])[:2]:
        findings.append(f"Authority: {b}")

    # add a couple “key gaps” bullets if present
    for b in (summary.get("key_gaps") or [])[:2]:
        findings.append(f"Gap: {b}")

    overall, package, explanation, strategy = derive_recommendation(disc, acc, auth)

    return ScanResponse(
        discovery_score=disc,
        accuracy_score=acc,
        authority_score=auth,
        overall_score=overall,
        package_recommendation=package,
        package_explanation=explanation,
        strategy_summary=strategy,
        findings=findings if findings else ["Audit completed, but findings were limited by available public sources."],
    )


# -------------------------------------------------------------------
# Helper: send notification via Resend
# -------------------------------------------------------------------

def send_notification(request: ScanRequest, result: ScanResponse) -> bool:
    if not EMAIL_NOTIFICATIONS_ENABLED:
        print("[VizAI] Notifications not configured; skipping email.")
        return False

    subject = f"[VizAI Scan] {request.businessName} ({result.overall_score}/100)"
    models_str = ", ".join(request.models) if request.models else "n/a"

    findings_block = "\n".join(f"- {line}" for line in result.findings)

    body = f"""
New VizAI scan submitted.

Business: {request.businessName}
Website: {request.website}
Models (informational): {models_str}
Contact Email: {request.contactEmail or "n/a"}

Scores
- Discovery: {result.discovery_score}
- Accuracy: {result.accuracy_score}
- Authority: {result.authority_score}
- Overall: {result.overall_score}

Recommended Package: {result.package_recommendation}
Explanation: {result.package_explanation}

Strategy Summary:
{result.strategy_summary}

Findings:
{findings_block}
""".strip()

    headers = {"Authorization": f"Bearer {RESEND_API_KEY}", "Content-Type": "application/json"}
    data = {"from": NOTIFY_EMAIL_FROM, "to": [NOTIFY_EMAIL_TO], "subject": subject, "text": body}

    try:
        resp = requests.post("https://api.resend.com/emails", headers=headers, json=data, timeout=20)
    except Exception as e:
        print(f"[VizAI] Error calling Resend: {e}")
        return False

    if 200 <= resp.status_code < 300:
        print("[VizAI] Notification email sent via Resend.")
        return True

    print(f"[VizAI] Failed to send email via Resend: {resp.status_code} {resp.text}")
    return False


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "service": "VizAI Scan API"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/run_scan", response_model=ScanResponse)
def run_scan(payload: ScanRequest):
    """
    Main endpoint used by the VizAI frontend.
    Now runs a REAL audit based on evidence (website + public sources).
    """
    result = run_real_audit(
        business_name=payload.businessName,
        website=str(payload.website),
    )

    # Notification (errors logged only)
    try:
        _ = send_notification(payload, result)
    except Exception as e:
        print(f"[VizAI] Exception while sending notification: {e}")

    return result

@app.post("/test_email")
def test_email():
    if not EMAIL_NOTIFICATIONS_ENABLED:
        return {"status": "notifications_not_configured"}

    dummy_request = ScanRequest(
        businessName="Test Business",
        website="https://example.com",
        models=["chatgpt"],
        contactEmail="test@example.com",
    )

    dummy_result = ScanResponse(
        discovery_score=70,
        accuracy_score=65,
        authority_score=60,
        overall_score=65,
        package_recommendation="Standard LMO",
        package_explanation="Test email – standard tier.",
        strategy_summary="This is a test email from VizAI backend.",
        findings=["This is a test finding from /test_email."],
    )

    sent = send_notification(dummy_request, dummy_result)
    return {"status": "notification_attempted" if sent else "notification_failed"}
