"""
judge.py — Functional adequacy scorer for concept instances.

Scores a simulation on how well it predicts the experience/behavior a concept
produces in a given context toward a given goal (Barrett's functional adequacy).
"""

from __future__ import annotations

import logging
from typing import Optional

from dotenv import load_dotenv

from .watsonx_client import call_watsonx

load_dotenv()

logger = logging.getLogger(__name__)


def score_instance(
    term: str,
    context: str,
    goal: str,
    simulation: str,
) -> Optional[float]:
    """Score a concept simulation for functional adequacy.

    Uses the judge prompt from the rl-feedback-loop skill (Step 3).

    Parameters
    ----------
    term:      The concept being evaluated (e.g. "anger").
    context:   The situational context for this instance.
    goal:      The functional goal the concept is serving.
    simulation: The simulation text to score.

    Returns
    -------
    Float 0–10 representing functional adequacy, or None if scoring failed.
    """
    prompt = (
        f'Concept: "{term}"\n'
        f'Context: "{context}"\n'
        f'Goal: "{goal}"\n'
        f'Simulation: "{simulation}"\n\n'
        "On a scale of 0–10, how functionally adequate is this simulation?\n"
        "(Does it accurately predict what this concept would produce in this context "
        "to serve this goal?)\n"
        "Reply with only a number."
    )

    result = call_watsonx(prompt, call_type="score")
    score = result.get("score")

    if score is None:
        logger.warning(
            "judge: could not score simulation for term=%r context=%r", term, context[:40]
        )

    return score
