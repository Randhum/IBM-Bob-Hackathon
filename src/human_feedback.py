"""
human_feedback.py — CLI RLHF outer loop for concept instance review.

Implements the rlhf-human-feedback skill as a Python CLI module. For each instance
in a ConceptPopulation the human reviews the simulation in full context and signals:
accept, reject, or refine.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from dotenv import load_dotenv

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    _RICH = True
except ImportError:
    _RICH = False

from .concept_population import ConceptInstance, ConceptPopulation
from .concept_loop import run_rl_loop

load_dotenv()

logger = logging.getLogger(__name__)

_MAX_HUMAN_ROUNDS = 3

if _RICH:
    console = Console()


def _display_instance(term: str, instance: ConceptInstance) -> None:
    """Print a concept instance card to the terminal."""
    score_str = f"{instance.adequacy_score:.1f}/10" if instance.adequacy_score is not None else "N/A"
    if _RICH:
        content = (
            f"[bold]Concept:[/bold]  {term}\n"
            f"[bold]Context:[/bold]  {instance.context}\n"
            f"[bold]Goal:[/bold]     {instance.goal}\n\n"
            f"[italic]Simulation (Round {instance.round}, Adequacy {score_str}):[/italic]\n"
            f"[yellow]\"{instance.simulation}\"[/yellow]"
        )
        console.print(Panel(content, title="CONCEPT INSTANCE REVIEW", border_style="blue"))
    else:
        print("\n" + "━" * 54)
        print("  CONCEPT INSTANCE REVIEW")
        print(f"  Concept:  {term}")
        print(f"  Context:  {instance.context}")
        print(f"  Goal:     {instance.goal}")
        print(f"\n  Simulation (Round {instance.round}, Adequacy {score_str}):")
        print(f'  "{instance.simulation}"')
        print("━" * 54)


def _prompt_signal() -> str:
    """Ask the human for accept/reject/refine. Returns lowercase first char."""
    while True:
        raw = input(
            "\nDoes this simulation accurately capture the concept in this context/goal?\n"
            "  [a] Accept   [r] Reject   [f] Refine\n"
            "Your choice: "
        ).strip().lower()
        if raw in ("a", "accept"):
            return "accept"
        if raw in ("r", "reject"):
            return "reject"
        if raw in ("f", "refine"):
            return "refine"
        print("Please enter 'a', 'r', or 'f'.")


def _prompt_hint(signal: str) -> str:
    """Collect an optional correction hint from the human."""
    prompt_text = (
        "What is wrong with this simulation for this context and goal? "
        if signal == "reject"
        else "What should be improved in this simulation? "
    )
    hint = input(prompt_text + "Provide a short hint (1–2 sentences), or press Enter to skip:\n> ").strip()
    if not hint:
        hint = input("No hint entered. Please provide a hint, or press Enter to use best-so-far:\n> ").strip()
    return hint or ""


def collect_human_feedback(
    population: ConceptPopulation,
    term: str,
) -> ConceptPopulation:
    """Run the human RLHF review loop over all instances in the population.

    For each instance:
    - Display term, context, goal, simulation, adequacy score.
    - Collect accept / reject / refine + optional hint.
    - On reject: re-run RL loop for that (context, goal) pair with hint (reset round=0).
    - On refine: re-run RL loop for that (context, goal) pair with hint (continue round).
    - Max _MAX_HUMAN_ROUNDS feedback rounds per (context, goal) pair.

    Returns the updated ConceptPopulation.
    """
    # Track human feedback round counts per (context, goal)
    feedback_counts: dict[Tuple[str, str], int] = {}

    # Work over a snapshot of current instances; new instances appended by re-runs
    # are not reviewed again in this pass (they become part of the population breadth).
    instances_to_review: List[ConceptInstance] = list(population.instances)

    for instance in instances_to_review:
        pair_key = (instance.context, instance.goal)
        rounds_used = feedback_counts.get(pair_key, 0)

        if rounds_used >= _MAX_HUMAN_ROUNDS:
            logger.info(
                "Max human feedback rounds reached for context=%r; skipping.",
                instance.context[:40],
            )
            instance.human_signal = "accept"  # auto-accept after max rounds
            continue

        _display_instance(term, instance)
        signal = _prompt_signal()
        instance.human_signal = signal
        feedback_counts[pair_key] = rounds_used + 1

        if signal == "accept":
            if _RICH:
                console.print("[green]✅ Accepted — instance added to population.[/green]")
            else:
                print("✅ Accepted — instance added to population.")
            continue

        # Reject or refine: collect hint
        hint = _prompt_hint(signal)
        instance.hint = hint or "No human hint provided."

        if signal == "reject":
            if _RICH:
                console.print("[red]❌ Rejected — re-running RL loop with hint (round reset).[/red]")
            else:
                print("❌ Rejected — re-running RL loop with hint (round reset).")
            # Re-run: reset round, inject hint
            new_pop = run_rl_loop(
                term=term,
                context_goal_pairs=[(instance.context, instance.goal)],
                max_iterations=3,
                threshold=7.5,
                hint=hint or None,
            )
        else:  # refine
            if _RICH:
                console.print("[yellow]✏️  Refinement requested — re-running RL loop with hint.[/yellow]")
            else:
                print("✏️  Refinement requested — re-running RL loop with hint.")
            # Re-run: continue from current round count
            new_pop = run_rl_loop(
                term=term,
                context_goal_pairs=[(instance.context, instance.goal)],
                max_iterations=max(1, 3 - instance.round),
                threshold=7.5,
                hint=hint or None,
            )

        # Append new instance(s) to population — the loop GROWS the population
        for new_instance in new_pop.instances:
            population.add_instance(new_instance)

    return population
