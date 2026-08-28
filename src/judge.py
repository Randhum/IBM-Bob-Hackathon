"""
judge.py — Functional adequacy scorer for concept instances.

Scores a simulation on how well it predicts the experience/behavior a concept
produces in a given context toward a given goal (Barrett's functional adequacy).

All linguistic layers (Levels 0–4 per docs/concept-ontology.md §3.4) are injected
into the judge prompt when provided, so the LLM evaluates the simulation against the
correct grammatical construction and sub-lexical grounding.
"""

from __future__ import annotations

import logging
from typing import List, Optional

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
    morphemes: Optional[List[str]] = None,
    phonesthetics_note: str = "",
) -> Optional[float]:
    """Score a concept simulation for functional adequacy.

    Uses the judge prompt from the rl-feedback-loop skill (Step 3).
    seed_phrase and grammatical_frame are included so the judge evaluates the
    simulation against the specific grammatical construction, not the bare token.

    Parameters
    ----------
    term:              The concept being evaluated (e.g. "anger", "fire").
    context:           The situational context for this instance.
    goal:              The functional goal the concept is serving.
    simulation:        The simulation text to score.
    seed_phrase:       Grammatically framed seed. Level 4.
    grammatical_frame: Syntactic role. Level 4.
    morphemes:         Meaningful sub-word units. Level 2. Optional soft context.
    phonesthetics_note: Sound-symbolism annotation. Level 0 signal. Optional.

    Returns
    -------
    Float 0–10 representing functional adequacy, or None if scoring failed.
    """
    lines = [f'Concept: "{term}"']
    if morphemes:
        lines.append(f'Morphemes: {", ".join(morphemes)}')
    if phonesthetics_note:
        lines.append(f'Sound symbolism: {phonesthetics_note}')
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
