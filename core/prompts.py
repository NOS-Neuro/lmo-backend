# core/prompts.py
from typing import List, Tuple

# Identity anchoring questions - MUST run first for entity verification
def get_identity_questions(website: str) -> List[Tuple[str, str]]:
    """Returns identity-anchoring questions with the actual website URL"""
    return [
        (
            "website_validation",
            f"Validate whether this is the official website for the company: {website}. "
            f"Use citations. If uncertain say 'unclear'."
        ),
        (
            "identity_fingerprint",
            f"Provide an identity fingerprint: official website, HQ city/country, phone (if available), "
            f"primary services. Use citations. If uncertain say 'unclear'."
        ),
    ]


# Core discovery questions - run after identity is anchored
DEFAULT_QUESTIONS: List[Tuple[str, str]] = [
    (
        "baseline_overview",
        "In 3–6 bullets: what does this company do? Include official site + main services. "
        "Use citations. If uncertain say 'unclear'."
    ),
    (
        "founder_team",
        "Who founded this company? Who are the key executives or leadership team? "
        "Use only verified sources with dates. If uncertain say 'unclear'."
    ),
    (
        "recent_activity",
        "What are the 3 most recent news mentions, announcements, or updates about this company? "
        "Include specific dates. If uncertain say 'unclear'."
    ),
    (
        "social_proof",
        "How many employees, customers, users, or other measurable metrics does this company have? "
        "Be specific with numbers. If uncertain say 'unclear'."
    ),
    (
        "locations_scope",
        "Where does the company operate (regions/countries/cities)? "
        "If unclear, say unclear."
    ),
    (
        "competitive_position",
        "How is this company positioned in its market? What makes them different from competitors? "
        "If unclear, say unclear."
    ),
    (
        "proof_points",
        "List 3 concrete proof points from sources (certifications, awards, partnerships, "
        "customer testimonials, case studies). If uncertain say 'unclear'."
    ),
]
