from __future__ import annotations

from typing import List

from core.ras.models import RecommendedAction
from recommendation_rules import Recommendation


_PRIORITY_MAP = {
    "P0": "P0",
    "P1": "P1",
    "P2": "P2",
}

_EFFORT_DEFAULT = {
    "P0": "high",
    "P1": "medium",
    "P2": "low",
}


def recommendation_to_action(r: Recommendation) -> RecommendedAction:
    priority = _PRIORITY_MAP.get(r.priority, "P2")
    effort = _EFFORT_DEFAULT.get(priority, "medium")

    # We’ll keep expected_impact empty for now unless you want to parse it
    # Later: translate expected_impact string -> estimated deltas.
    return RecommendedAction(
        priority=priority,
        action=r.title,
        why=" | ".join(r.why[:3]) if isinstance(r.why, list) else str(r.why),
        expected_impact={},
        effort=effort,
        owner_hint=r.category,
    )


def bundle_to_actions(fix_now: List[Recommendation], maintain: List[Recommendation]) -> List[RecommendedAction]:
    actions: List[RecommendedAction] = []
    for r in fix_now:
        actions.append(recommendation_to_action(r))
    for r in maintain:
        actions.append(recommendation_to_action(r))
    return actions
