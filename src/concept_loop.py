"""
concept_loop.py — RL feedback loop for growing a ConceptPopulation.

Implements the rl-feedback-loop skill as Python code. For each (context, goal) pair,
generates an initial simulation, scores for functional adequacy, and refines until the
adequacy threshold is met or max iterations are reached.

The loop GROWS the population — instances are added, never replaced.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from dotenv import load_dotenv

from .concept_population import ConceptInstance, ConceptPopulation
from .watsonx_client import call_watsonx
from . import judge as judge_module
from . import concept_refiner

load_dotenv()

logger = logging.getLogger(__name__)

# (context, goal, optional seed_phrase, optional grammatical_frame)
ContextGoalPair = Tuple[str, str]
ContextGoalPairFull = Tuple[str, str, str, str]  # (context, goal, seed_phrase, grammatical_frame)


def _build_generation_prompt(
    term: str,
    context: str,
    goal: str,
    seed_phrase: str = "",
    grammatical_frame: str = "",
    morphemes: Optional[List[str]] = None,
    phonesthetics_note: str = "",
    hint: Optional[str] = None,
) -> str:
    """Build the initial simulation generation prompt (rl-feedback-loop SKILL Step 2).

    Injects all available linguistic layers (Level 2–4) so the LLM constructs a
    simulation grounded at the right level of the granularity hierarchy, not from
    an ambiguous bare token (docs/concept-ontology.md §3.4).
    """
    lines = [
        "You are a concept-simulation engine based on predictive cognition.\n",
        f'Concept: "{term}"',
    ]
    if morphemes:
        lines.append(f'Morphemes (meaningful sub-word units): {", ".join(morphemes)}')
    if phonesthetics_note:
        lines.append(f'Sound symbolism note (soft hint): {phonesthetics_note}')
    if seed_phrase:
        lines.append(f'Seed phrase (grammatically framed): "{seed_phrase}"')
    if grammatical_frame:
        lines.append(f'Grammatical frame: "{grammatical_frame}"')
    lines += [
        f'Context: "{context}"',
        f'Goal: "{goal}"\n',
        "Generate a simulation — a prediction of the likely experience, response, or behavior that",
        "this concept, as grammatically framed here, would produce in this context to serve this goal.",
    ]
    if hint:
        lines.append(f'Incorporate this guidance: "{hint}"')
    lines.append("\nBe specific and grounded. 1–3 sentences. ≤60 words.")
    return "\n".join(lines)


def run_rl_loop(
    term: str,
    context_goal_pairs: List[ContextGoalPair],
    seed_phrase: str = "",
    grammatical_frame: str = "",
    morphemes: Optional[List[str]] = None,
    phonesthetics_note: str = "",
    max_iterations: int = 3,
    threshold: float = 7.5,
    hint: Optional[str] = None,
) -> ConceptPopulation:
    """Run the RL auto-scoring loop and return a populated ConceptPopulation.

    For each (context, goal) pair:
    1. Generate an initial simulation (Round 0).
    2. Score for functional adequacy.
    3. If score < threshold and round < max_iterations, refine and re-score.
    4. Add the final instance to the population (always — even if below threshold).

    Parameters
    ----------
    term:               Concept term (raw word/phrase).
    seed_phrase:        Grammatically framed form, e.g. "to fire (someone)". Level 4.
    grammatical_frame:  Syntactic role, e.g. "transitive verb, agent=manager". Level 4.
    morphemes:          Optional list of morphemes, e.g. ["mis-", "trust"]. Level 2.
                        NOT BPE tokens. Injected as soft sub-lexical grounding.
    phonesthetics_note: Optional sound-symbolism annotation. Level 0 signal. Soft hint only.
    context_goal_pairs: List of (context, goal) tuples to process.
    max_iterations:     Maximum refinement rounds per instance (default: 3).
    threshold:          Adequacy score above which refinement stops (default: 7.5).
    hint:               Optional global hint injected into all generation prompts.

    Returns
    -------
    ConceptPopulation with all processed instances.
    """
    population = ConceptPopulation(term=term, seed_phrase=seed_phrase)

    for context, goal in context_goal_pairs:
        logger.info("RL loop — term=%r context=%r goal=%r", term, context[:40], goal[:40])

        # ── Step 2: Generate initial simulation (Round 0) ──────────────────
        gen_prompt = _build_generation_prompt(
            term, context, goal,
            seed_phrase=seed_phrase,
            grammatical_frame=grammatical_frame,
            morphemes=morphemes,
            phonesthetics_note=phonesthetics_note,
            hint=hint,
        )
        gen_result = call_watsonx(gen_prompt, call_type="generate")
        simulation = (gen_result.get("response") or "").strip()

        if not simulation:
            logger.warning("Empty simulation for context=%r; using placeholder.", context[:40])
            simulation = f"[No simulation generated for context: {context}]"

        instance = ConceptInstance(
            context=context,
            goal=goal,
            simulation=simulation,
            morphemes=morphemes or [],
            phonesthetics_note=phonesthetics_note,
            seed_phrase=seed_phrase,
            grammatical_frame=grammatical_frame,
            round=0,
        )

        # ── Step 3: Score for functional adequacy ──────────────────────────
        score = judge_module.score_instance(
            term, context, goal, simulation,
            seed_phrase=seed_phrase,
            grammatical_frame=grammatical_frame,
        )
        instance.adequacy_score = score
        instance.record_round()  # capture Round 0 in history (initial_score property reads [0])

        logger.info("  Round 0 score: %s", score)

        # ── Steps 4–5: Refine loop ──────────────────────────────────────────
        current_round = 0
        while True:
            above_threshold = score is not None and score >= threshold
            at_max_rounds = current_round >= max_iterations

            if above_threshold:
                logger.info("  Threshold met (%.1f >= %.1f) — accepting.", score, threshold)
                break
            if at_max_rounds:
                logger.info("  Max rounds (%d) reached — accepting as-is.", max_iterations)
                break

            # Refine
            current_round += 1
            logger.info("  Refining — starting round %d …", current_round)

            refinement = concept_refiner.refine_simulation(
                term=term,
                context=context,
                goal=goal,
                current_simulation=instance.simulation,
                seed_phrase=seed_phrase,
                grammatical_frame=grammatical_frame,
                morphemes=morphemes,
                phonesthetics_note=phonesthetics_note,
                score=instance.adequacy_score,
                hint=hint,
            )
            instance.simulation = refinement["refined_simulation"]
            instance.round = current_round
            if refinement.get("hint_used") and hint:
                instance.hint = hint

            # Re-score
            score = judge_module.score_instance(
                term, context, goal, instance.simulation,
                seed_phrase=seed_phrase,
                grammatical_frame=grammatical_frame,
            )
            instance.adequacy_score = score
            instance.record_round()  # capture this round in history

            logger.info(
                "  Round %d score: %s (rationale: %s)",
                current_round,
                score,
                refinement.get("rationale", "")[:60],
            )

        # ── Step 6: Add to population ───────────────────────────────────────
        population.add_instance(instance)
        logger.info(
            "  Instance added. Population breadth: %d", population.population_breadth
        )

    return population
