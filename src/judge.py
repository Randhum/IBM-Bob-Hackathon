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
    seed_phrase: str = "",
    grammatical_frame: str = "",
) -> Optional[float]:
    """Score a concept simulation for functional adequacy.

    Uses the judge prompt from the rl-feedback-loop skill (Step 3).
    seed_phrase and grammatical_frame are included so the judge evaluates the
    simulation against the specific grammatical construction, not the bare token.

    Parameters
    ----------
    term:             The concept being evaluated (e.g. "anger", "fire").
    context:          The situational context for this instance.
    goal:             The functional goal the concept is serving.
    simulation:       The simulation text to score.
    seed_phrase:      Grammatically framed seed (e.g. "to fire someone"). Optional.
    grammatical_frame: Syntactic role (e.g. "transitive verb"). Optional.

    Returns
    -------
    Float 0–10 representing functional adequacy, or None if scoring failed.
    """
    lines = [f'Concept: "{term}"']
    if seed_phrase:
        lines.append(f'Seed phrase: "{seed_phrase}"')
    if grammatical_frame:
        lines.append(f'Grammatical frame: "{grammatical_frame}"')
    lines += [
        f'Context: "{context}"',
        f'Goal: "{goal}"',
        f'Simulation: "{simulation}"',
        "",
        "On a scale of 0–10, how functionally adequate is this simulation?",
        "(Does it accurately predict what this concept — as grammatically framed — "
        "would produce in this context to serve this goal?)",
        "Reply with only a number.",
    ]
    prompt = "\n".join(lines)

    result = call_watsonx(prompt, call_type="score")
    score = result.get("score")

    if score is None:
        logger.warning(
            "judge: could not score simulation for term=%r context=%r", term, context[:40]
        )

    return score
