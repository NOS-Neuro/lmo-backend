import os
import json
import re
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Tuple
from urllib.parse import urljoin, urlparse

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, EmailStr


# -------------------------------------------------------------------
# Config / Environment
# -------------------------------------------------------------------

RESEND_API_KEY = os.getenv("RESEND_API_KEY")
NOTIFY_EMAIL_FROM = os.getenv("NOTIFY_EMAIL_FROM")  # e.g. "scan@vizai.io"
NOTIFY_EMAIL_TO = os.getenv("NOTIFY_EMAIL_TO")      # e.g. "you@yourmail.com"

DATABASE_URL = os.getenv("DATABASE_URL")  # Render Postgres recommended

EMAIL_NOTIFICATIONS_ENABLED = bool(
    RESEND_API_KEY and NOTIFY_EMAIL_FROM and NOTIFY_EMAIL_TO
)

if EMAIL_NOTIFICATIONS_ENABLED:
    print("[VizAI] Email notifications via Resend are ENABLED.")
else:
    print("[VizAI] Email notifications are DISABLED (missing env vars).")

if DATABASE_URL:
    print("[VizAI] DATABASE_URL is set. Scan persistence is ENABLED.")
else:
    print("[VizAI] DATABASE_URL is not set. Scan persistence is DISABLED.")


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
    contactEmail: EmailStr
    pleaseContactMe: bool = False


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
# Helpers: simple HTML checks (no heavy dependencies)
# -------------------------------------------------------------------

USER_AGENT = "VizAI-ScanBot/1.0 (+https://vizai.io)"


def _fetch_html(url: str, timeout: int = 15) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*"},
            timeout=timeout,
            allow_redirects=True,
        )
        ctype = resp.headers.get("Content-Type", "")
        if resp.status_code >= 400:
            return None, resp.status_code, f"HTTP {resp.status_code}"
        if "text/html" not in ctype and "application/xhtml" not in ctype:
            # still allow parsing if it's HTML-ish
            pass
        return resp.text or "", resp.status_code, None
    except requests.RequestException as e:
        return None, None, str(e)


def _find_json_ld_blocks(html: str) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    # Grab script tags for application/ld+json (best-effort regex)
    pattern = re.compile(r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', re.I | re.S)
    for m in pattern.finditer(html):
        raw = m.group(1).strip()
        if not raw:
            continue
        # Some sites include multiple JSON objects/arrays
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        blocks.append(item)
            elif isinstance(data, dict):
                blocks.append(data)
        except Exception:
            # ignore invalid JSON-LD
            continue
    return blocks


def _has_schema_microdata(html: str) -> bool:
    return ("itemscope" in html.lower()) or ("itemtype" in html.lower())


def _get_meta_content(html: str, name: str) -> Optional[str]:
    # matches: <meta name="description" content="...">
    pattern = re.compile(
        rf'<meta[^>]+name=["\']{re.escape(name)}["\'][^>]*content=["\'](.*?)["\']',
        re.I | re.S
    )
    m = pattern.search(html)
    return m.group(1).strip() if m else None


def _get_og_content(html: str, prop: str) -> Optional[str]:
    pattern = re.compile(
        rf'<meta[^>]+property=["\']{re.escape(prop)}["\'][^>]*content=["\'](.*?)["\']',
        re.I | re.S
    )
    m = pattern.search(html)
    return m.group(1).strip() if m else None


def _count_tag(html: str, tag: str) -> int:
    return len(re.findall(rf"<\s*{tag}\b", html, flags=re.I))


def _extract_about_url(base_url: str, html: str) -> Optional[str]:
    # find href containing "/about" or "about-us" etc.
    hrefs = re.findall(r'<a[^>]+href=["\'](.*?)["\']', html, flags=re.I)
    candidates = []
    for h in hrefs:
        lh = h.lower()
        if "about" in lh:
            candidates.append(h)
    if not candidates:
        return None
    # prefer shortest / most canonical
    candidates.sort(key=lambda x: len(x))
    href = candidates[0]
    try:
        return urljoin(base_url, href)
    except Exception:
        return None


def _wikipedia_exists(business_name: str) -> bool:
    # Use Wikipedia Search API (free)
    try:
        api = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": business_name,
            "format": "json",
        }
        r = requests.get(api, params=params, timeout=10, headers={"User-Agent": USER_AGENT})
        if r.status_code != 200:
            return False
        data = r.json()
        results = data.get("query", {}).get("search", [])
        if not results:
            return False
        # basic heuristic: if top result title is close match
        top_title = (results[0].get("title") or "").lower()
        bn = business_name.lower()
        return bn in top_title or top_title in bn
    except Exception:
        return False


# -------------------------------------------------------------------
# Scoring + Recommendations (deterministic)
# -------------------------------------------------------------------

def _clamp(n: int, lo: int = 0, hi: int = 100) -> int:
    return max(lo, min(hi, n))


def derive_recommendation(discovery: int, accuracy: int, authority: int) -> Tuple[int, str, str, str]:
    overall = int(round((discovery + accuracy + authority) / 3))

    package = "Standard LMO"
    explanation = ""
    strategy = ""

    if overall >= 80:
        package = "Basic LMO"
        explanation = (
            "Your visibility signals are strong. Basic focuses on monitoring and small corrections "
            "to prevent drift as AI systems and sources change."
        )
        strategy = (
            "Lock in a canonical Truth File + verify structured data on your site. "
            "Re-scan monthly and correct any drift early."
        )
    elif overall >= 40:
        package = "Standard LMO"
        explanation = (
            "Your signals are mixed. Standard focuses on fixing gaps (schema + About/FAQ) and "
            "strengthening authority sources so AI relies on you more consistently."
        )
        strategy = (
            "Prioritize Organization/LocalBusiness schema + a strong About/FAQ page, then "
            "improve external profiles used for AI discovery."
        )
    else:
        package = "Standard LMO + Add-Ons"
        explanation = (
            "Your signals suggest low discoverability/authority. You likely need a deeper correction pass "
            "plus targeted add-ons to rebuild visibility."
        )
        strategy = (
            "Establish a canonical Truth File, deploy schema, and build/refresh authority sources "
            "(key registries, profiles, citations)."
        )

    return overall, package, explanation, strategy


def run_real_audit(business_name: str, website: str) -> Tuple[int, int, int, List[str], Dict[str, Any]]:
    """
    Real audit signals:
    - Homepage HTML fetch
    - JSON-LD presence + org-ish types
    - Microdata schema presence
    - Meta title + description
    - OG tags
    - H1 count
    - About page discovery (and fetch)
    - Wikipedia presence check
    """
    findings: List[str] = []
    raw: Dict[str, Any] = {
        "website": website,
        "checks": {},
    }

    html, status, err = _fetch_html(website)
    if html is None:
        findings.append(f"Site fetch failed: {err or 'unknown error'}")
        # If website is unreachable, Discovery/Accuracy/Authority are all low
        discovery = 15
        accuracy = 15
        authority = 10
        raw["checks"]["homepage_fetch"] = {"ok": False, "status": status, "error": err}
        return discovery, accuracy, authority, findings, raw

    raw["checks"]["homepage_fetch"] = {"ok": True, "status": status}

    jsonld_blocks = _find_json_ld_blocks(html)
    has_jsonld = len(jsonld_blocks) > 0
    has_microdata = _has_schema_microdata(html)

    meta_desc = _get_meta_content(html, "description")
    og_title = _get_og_content(html, "og:title")
    og_desc = _get_og_content(html, "og:description")
    h1_count = _count_tag(html, "h1")

    about_url = _extract_about_url(website, html)
    about_ok = False
    about_word_count = 0
    if about_url:
        about_html, _, about_err = _fetch_html(about_url, timeout=12)
        if about_html:
            about_ok = True
            about_word_count = len(re.findall(r"\w+", about_html))
        raw["checks"]["about_page"] = {
            "found": True,
            "url": about_url,
            "fetched": about_ok,
            "word_count": about_word_count,
            "error": about_err,
        }
    else:
        raw["checks"]["about_page"] = {"found": False}

    # JSON-LD types check
    jsonld_types = []
    org_like = False
    for block in jsonld_blocks:
        t = block.get("@type")
        if isinstance(t, list):
            jsonld_types.extend([str(x) for x in t])
        elif isinstance(t, str):
            jsonld_types.append(t)
        # detect org-ish schemas
        tset = set([x.lower() for x in (jsonld_types or [])])
        if any(x in tset for x in ["organization", "corporation", "localbusiness"]):
            org_like = True

    wiki_ok = _wikipedia_exists(business_name)

    raw["checks"]["structured_data"] = {
        "jsonld_found": has_jsonld,
        "jsonld_types": list(dict.fromkeys(jsonld_types))[:12],
        "org_like": org_like,
        "microdata_found": has_microdata,
    }
    raw["checks"]["metadata"] = {
        "meta_description": bool(meta_desc),
        "og_title": bool(og_title),
        "og_description": bool(og_desc),
        "h1_count": h1_count,
    }
    raw["checks"]["wikipedia"] = {"found": wiki_ok}

    # -----------------------
    # Discovery score (0–100)
    # -----------------------
    discovery = 35  # base if site reachable
    if wiki_ok:
        discovery += 20
    if og_title or meta_desc:
        discovery += 10
    if about_ok:
        discovery += 10
    if has_jsonld or has_microdata:
        discovery += 10
    discovery = _clamp(discovery)

    # -----------------------
    # Accuracy score (0–100)
    # -----------------------
    accuracy = 20
    if has_jsonld:
        accuracy += 25
    if org_like:
        accuracy += 10
    if meta_desc:
        accuracy += 10
    if h1_count == 1:
        accuracy += 10
    elif h1_count == 0:
        accuracy -= 5
    elif h1_count > 2:
        accuracy -= 5
    if about_ok and about_word_count >= 250:
        accuracy += 15
    elif about_ok:
        accuracy += 8
    accuracy = _clamp(accuracy)

    # -----------------------
    # Authority score (0–100)
    # -----------------------
    authority = 15
    if wiki_ok:
        authority += 35
    if has_jsonld and org_like:
        authority += 15
    elif has_jsonld:
        authority += 8
    if og_title and og_desc:
        authority += 6
    if about_ok and about_word_count >= 400:
        authority += 10
    authority = _clamp(authority)

    # Findings (human readable)
    if has_jsonld:
        findings.append("✓ JSON-LD structured data detected (good machine-readable signal).")
        if org_like:
            findings.append("✓ Organization/LocalBusiness-type schema appears present (strong for AI accuracy).")
        else:
            findings.append("• JSON-LD is present but Organization/LocalBusiness types were not clearly detected.")
    else:
        findings.append("✗ No JSON-LD detected (missed opportunity for AI-readable business facts).")

    if has_microdata:
        findings.append("✓ Schema microdata detected on-page (additional structured signal).")
    else:
        findings.append("• No schema microdata detected on-page.")

    if meta_desc:
        findings.append("✓ Meta description found (helps search + AI summaries).")
    else:
        findings.append("✗ Meta description missing on homepage.")

    if og_title or og_desc:
        findings.append("✓ Open Graph tags found (improves link previews + consistency).")
    else:
        findings.append("• Open Graph tags not detected.")

    if h1_count == 1:
        findings.append("✓ Clean page hierarchy (single H1 detected).")
    else:
        findings.append(f"• Page hierarchy may be unclear (H1 count detected: {h1_count}).")

    if about_ok:
        findings.append("✓ About page detected and readable (helps establish canonical company story).")
        if about_word_count < 250:
            findings.append("• About page looks short — expanding it can improve AI accuracy and citations.")
    else:
        findings.append("✗ About page not clearly detected (or not readable).")

    if wiki_ok:
        findings.append("✓ Wikipedia presence detected (high-authority source for many AI systems).")
    else:
        findings.append("• No clear Wikipedia presence detected (may reduce authority signals).")

    return discovery, accuracy, authority, findings[:10], raw


# -------------------------------------------------------------------
# Email via Resend
# -------------------------------------------------------------------

def _resend_send_email(to_email: str, subject: str, text: str) -> bool:
    if not EMAIL_NOTIFICATIONS_ENABLED:
        return False

    headers = {
        "Authorization": f"Bearer {RESEND_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "from": NOTIFY_EMAIL_FROM,
        "to": [to_email],
        "subject": subject,
        "text": text,
    }

    try:
        resp = requests.post("https://api.resend.com/emails", headers=headers, json=data, timeout=20)
        return 200 <= resp.status_code < 300
    except Exception as e:
        print(f"[VizAI] Resend error: {e}")
        return False


def build_customer_email(request: ScanRequest, result: ScanResponse) -> str:
    findings_block = "\n".join([f"- {x}" for x in result.findings])
    return f"""Your VizAI Scan Report

Business: {request.businessName}
Website: {request.website}

Scores
- Discovery: {result.discovery_score}/100
- Accuracy: {result.accuracy_score}/100
- Authority: {result.authority_score}/100
- Overall: {result.overall_score}/100

Recommended Next Step
- {result.package_recommendation}
{result.strategy_summary}

Key Findings
{findings_block}

Notes
- VizAI uses real visibility signals (site structure + trusted public sources).
- Results are stored so you can compare improvements over time.

If you'd like help improving these scores, reply to this email.
""".strip()


def build_owner_email(request: ScanRequest, result: ScanResponse, raw: Dict[str, Any]) -> str:
    findings_block = "\n".join([f"- {x}" for x in result.findings])
    raw_compact = json.dumps(raw, indent=2)[:6000]  # keep email reasonable
    return f"""New VizAI Scan Submitted

Business: {request.businessName}
Website: {request.website}
Customer Email: {request.contactEmail}
Please contact me: {request.pleaseContactMe}

Scores
- Discovery: {result.discovery_score}
- Accuracy: {result.accuracy_score}
- Authority: {result.authority_score}
- Overall: {result.overall_score}

Recommended
- {result.package_recommendation}

Strategy Summary
{result.strategy_summary}

Findings
{findings_block}

Raw Checks (truncated)
{raw_compact}
""".strip()


# -------------------------------------------------------------------
# Database persistence (Postgres via psycopg2)
# -------------------------------------------------------------------

def _db_enabled() -> bool:
    return bool(DATABASE_URL)


def _db_init() -> None:
    if not _db_enabled():
        return
    try:
        import psycopg2
        conn = psycopg2.connect(DATABASE_URL, sslmode="require")
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS scans (
          id SERIAL PRIMARY KEY,
          business_name TEXT NOT NULL,
          website TEXT NOT NULL,
          contact_email TEXT NOT NULL,
          please_contact_me BOOLEAN NOT NULL DEFAULT FALSE,
          discovery_score INT NOT NULL,
          accuracy_score INT NOT NULL,
          authority_score INT NOT NULL,
          overall_score INT NOT NULL,
          findings JSONB NOT NULL,
          raw_report JSONB NOT NULL,
          created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_scans_email ON scans(contact_email);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_scans_website ON scans(website);")
        cur.close()
        conn.close()
        print("[VizAI] DB init complete.")
    except Exception as e:
        print(f"[VizAI] DB init failed: {e}")


@app.on_event("startup")
def on_startup():
    _db_init()


def _db_store_scan(request: ScanRequest, result: ScanResponse, raw: Dict[str, Any]) -> Optional[int]:
    if not _db_enabled():
        return None
    try:
        import psycopg2
        conn = psycopg2.connect(DATABASE_URL, sslmode="require")
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO scans (
              business_name, website, contact_email, please_contact_me,
              discovery_score, accuracy_score, authority_score, overall_score,
              findings, raw_report
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING id;
            """,
            (
                request.businessName,
                str(request.website),
                str(request.contactEmail),
                bool(request.pleaseContactMe),
                int(result.discovery_score),
                int(result.accuracy_score),
                int(result.authority_score),
                int(result.overall_score),
                json.dumps(result.findings),
                json.dumps(raw),
            )
        )
        scan_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        return int(scan_id)
    except Exception as e:
        print(f"[VizAI] DB store failed: {e}")
        return None


# -------------------------------------------------------------------
# Main scan logic
# -------------------------------------------------------------------

def run_scan_logic(request: ScanRequest) -> Tuple[ScanResponse, Dict[str, Any]]:
    discovery, accuracy, authority, findings, raw = run_real_audit(
        business_name=request.businessName,
        website=str(request.website),
    )

    overall, package, explanation, strategy = derive_recommendation(discovery, accuracy, authority)

    # Use package fields as "next step" copy in your UI
    response = ScanResponse(
        discovery_score=discovery,
        accuracy_score=accuracy,
        authority_score=authority,
        overall_score=overall,
        package_recommendation=package,
        package_explanation=explanation,
        strategy_summary=strategy,
        findings=findings,
    )
    return response, raw


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
    Runs a real site + public-source audit, emails report, and stores results.
    """
    # Run audit
    result, raw = run_scan_logic(payload)

    # Store to DB (best effort)
    scan_id = _db_store_scan(payload, result, raw)
    if scan_id:
        raw["scan_id"] = scan_id

    # Notify owner (best effort)
    try:
        if EMAIL_NOTIFICATIONS_ENABLED:
            lead_prefix = "🔥 Lead" if payload.pleaseContactMe else "VizAI Scan"
            subj = f"[{lead_prefix}] {payload.businessName} ({result.overall_score}/100)"
            owner_body = build_owner_email(payload, result, raw)
            ok = _resend_send_email(NOTIFY_EMAIL_TO, subj, owner_body)
            print(f"[VizAI] Owner email sent: {ok}")
        else:
            print("[VizAI] Owner email skipped (notifications not configured).")
    except Exception as e:
        print(f"[VizAI] Owner email error: {e}")

    # Email customer their report (best effort)
    try:
        if EMAIL_NOTIFICATIONS_ENABLED:
            subj = f"Your VizAI Scan Report: {payload.businessName}"
            customer_body = build_customer_email(payload, result)
            ok = _resend_send_email(str(payload.contactEmail), subj, customer_body)
            print(f"[VizAI] Customer email sent: {ok}")
        else:
            print("[VizAI] Customer email skipped (notifications not configured).")
    except Exception as e:
        print(f"[VizAI] Customer email error: {e}")

    return result


@app.post("/test_email")
def test_email():
    """
    Simple endpoint to confirm email notifications are wired correctly.
    """
    if not EMAIL_NOTIFICATIONS_ENABLED:
        return {"status": "notifications_not_configured"}

    dummy_request = ScanRequest(
        businessName="Test Business",
        website="https://example.com",
        contactEmail="test@example.com",
        pleaseContactMe=True
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

    ok1 = _resend_send_email(
        NOTIFY_EMAIL_TO,
        "[VizAI Test] Owner notification",
        "This is a test notification email from VizAI backend."
    )
    ok2 = _resend_send_email(
        str(dummy_request.contactEmail),
        "[VizAI Test] Customer report",
        "This is a test customer report email from VizAI backend."
    )

    return {"status": "ok", "owner_sent": ok1, "customer_sent": ok2}

