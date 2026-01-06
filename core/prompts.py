# core/prompts.py
from typing import List, Tuple

DEFAULT_QUESTIONS: List[Tuple[str, str]] = [
    ("baseline_overview", "In 3–6 bullets: what does this company do? Include official site + main services."),
    ("founder_team", "Who founded this company? Who are the key executives or leadership team? Use only verified sources with dates."),
    ("recent_activity", "What are the 3 most recent news mentions, announcements, or updates about this company? Include specific dates."),
    ("social_proof", "How many employees, customers, users, or other measurable metrics does this company have? Be specific with numbers."),
    ("locations_scope", "Where does the company operate (regions/countries/cities)? If unclear, say unclear."),
    ("competitive_position", "How is this company positioned in its market? What makes them different from competitors?"),
    ("proof_points", "List 3 concrete proof points from sources (certifications, awards, partnerships, customer testimonials, case studies)."),
]

# Website validation question - added dynamically with the actual URL
def get_website_validation_question(website: str) -> Tuple[str, str]:
    """Returns a question that includes the actual website URL for validation"""
    return (
        "website_validation",
        f"Visit {website} directly. What does this company claim to do on their own website? "
        f"How does their self-description compare to what other sources say?"
    )
