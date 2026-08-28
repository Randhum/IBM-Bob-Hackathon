"""
concept_refiner.py — Single-step concept simulation refiner.

Implements the concept-definition-refiner skill as Python code. Produces one improved
simulation given a current simulation, optional score, and optional human hint.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any

from dotenv import load_dotenv

from .watsonx_client import call_watsonx

load_dotenv()

logger = logging.getLogger(__name__)

_MAX_WORDS = 60


def _word_count(text: str) -> int:
    return len(text.split())


def refine_simulation(
    term: str,
    context: str,
    goal: str,
    current_simulation: str,
    score: Optional[float] = None,
    hint: Optional[str] = None,
    aspect: str = "goal-alignment",
) -> Dict[str, Any]:
    """Produce one refined simulation for a concept instance.

    Parameters
    ----------
    term:               Concept term (e.g. "anger").
    context:            Situational context for this instance.
    goal:               Functional goal this concept should serve.
    current_simulation: Simulation text to improve.
    score:              Current adequacy score (optional, improves prompt calibration).
    hint:               Human-supplied correction direction (optional).
    aspect:             Focus of improvement: "goal-alignment" | "specificity" |
                        "contextual-grounding" | "conciseness". Default: "goal-alignment".

    Returns
    -------
    dict with keys: refined_simulation, rationale, hint_used
    """
    def _build_prompt(simulation: str) -> str:
        lines = [
            "You are a concept-simulation refiner grounded in predictive cognition.\n",
            f'Concept: "{term}"',
            f'Context: "{context}"',
            f'Goal: "{goal}"',
            f'Current simulation: "{simulation}"',
        ]
        if score is not None:
            lines.append(f"Current adequacy score: {score}/10 — improve on this.")
        if hint:
            lines.append(f"Human guidance: {hint}")
        lines.append(f"Improvement focus: {aspect}")
        lines.append(
            "\nProduce one improved simulation that better predicts the experience/behavior "
            "this concept produces in this context to serve this goal.\n"
            "Then on a new line write:\n"
            "RATIONALE: <one sentence explaining what was improved and why>"
        )
        return "\n".join(lines)

    def _parse(response: str) -> tuple[str, str]:
        if "RATIONALE:" in response:
            parts = response.split("RATIONALE:", 1)
            sim = parts[0].strip().strip('"')
            rationale = parts[1].strip()
        else:
            sim = response.strip().strip('"')
            rationale = "Not provided"
        return sim, rationale

    prompt = _build_prompt(current_simulation)
    result = call_watsonx(prompt, call_type="generate")
    raw = result.get("response") or ""

    refined_sim, rationale = _parse(raw)

    # Validate: non-empty
    if not refined_sim:
        logger.warning("refine_simulation: empty refined simulation; keeping original.")
        return {
            "refined_simulation": current_simulation,
            "rationale": "Refinement produced empty output; original retained.",
            "hint_used": bool(hint),
        }

    # Validate: not identical to original
    if refined_sim == current_simulation:
        logger.warning("refine_simulation: refined simulation identical to original; keeping.")
        return {
            "refined_simulation": current_simulation,
            "rationale": "Refinement produced identical output; original retained.",
            "hint_used": bool(hint),
        }

    # Validate: word count ≤ 60
    if _word_count(refined_sim) > _MAX_WORDS:
        shorten_prompt = (
            _build_prompt(current_simulation)
            + "\n\nIMPORTANT: Keep the simulation under 60 words."
        )
        result2 = call_watsonx(shorten_prompt, call_type="generate")
        raw2 = result2.get("response") or ""
        refined_sim2, rationale2 = _parse(raw2)
        if refined_sim2 and refined_sim2 != current_simulation:
            refined_sim = refined_sim2
            rationale = rationale2

    return {
        "refined_simulation": refined_sim,
        "rationale": rationale,
        "hint_used": bool(hint),
    }
