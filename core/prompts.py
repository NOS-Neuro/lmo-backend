# core/prompts.py
from typing import List, Tuple

DEFAULT_QUESTIONS: List[Tuple[str, str]] = [
    ("baseline_overview", "In 3–6 bullets: what does this company do? Include official site + main services."),
    ("contact_path", "What is the best contact path (email/form/phone) from sources? If unknown, say unclear."),
    ("locations_scope", "Where does the company operate (regions/countries)? If unclear, say unclear."),
    ("proof_points", "List 3 proof points from sources (certifications, customers, industries, capabilities)."),
]
