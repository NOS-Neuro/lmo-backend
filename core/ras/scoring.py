from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from core.ras.models import ComponentScores, ScoreBand


MAX_SCORE = 92  # hard ceiling


WEIGHTS: Dict[str, float] = {
    "discovery": 0.25,
    "accuracy": 0.25,
    "consistency": 0.20,
    "authority": 0.15,
    "completeness": 0.15,
}


SCORE_BANDS = [
    (85, 92, ScoreBand(
        label="Operationally Authoritative",
        description="LLMs will reliably provide accurate, consistent information.",
        confidence="High — low hallucination risk",
        next_steps="Maintain and monitor for drift; keep sources fresh."
    )),
    (70, 84, ScoreBand(
        label="Strong Alignment",
        description="Most LLMs get it right, but some inconsistencies exist.",
        confidence="Medium — minor risk of errors",
        next_steps="Focus on consistency, official citations, and coverage gaps."
    )),
    (55, 69, ScoreBand(
        label="Partial Alignment",
        description="LLMs sometimes provide incorrect or conflicting information.",
        confidence="Low — significant hallucination risk",
        next_steps="Address critical accuracy gaps and canonical source issues."
    )),
    (0, 54, ScoreBand(
        label="At Risk",
        description="High probability of hallucinations or omissions.",
        confidence="Very low — unreliable answers expected",
        next_steps="Foundational corrections needed: canonical source + structured data + directory cleanup."
    )),
]


def calculate_overall(scores: ComponentScores) -> int:
    weighted = (
        scores.discovery * WEIGHTS["discovery"] +
        scores.accuracy * WEIGHTS["accuracy"] +
        scores.consistency * WEIGHTS["consistency"] +
        scores.authority * WEIGHTS["authority"] +
        scores.completeness * WEIGHTS["completeness"]
    )

    # Apply a gentle confidence penalty if evidence confidence is low
    # (keeps it deterministic but makes thin evidence show slightly lower scores)
    confidence_multiplier = 0.92 + (scores.evidence_confidence * 0.08)  # 0.92..1.0
    weighted = weighted * confidence_multiplier

    return int(min(round(weighted), MAX_SCORE))


def interpret(overall_score: int) -> ScoreBand:
    for lo, hi, band in SCORE_BANDS:
        if lo <= overall_score <= hi:
            return band
    # fallback
    return SCORE_BANDS[-1][2]

