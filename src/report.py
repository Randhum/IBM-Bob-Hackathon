"""
report.py — Concept Population Report generator.

Implements the concept-clarity-report skill as Python code. Computes population
metrics and writes a markdown report following Barrett's conceptual framework.
"""

from __future__ import annotations

import os
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from dotenv import load_dotenv

from .concept_population import ConceptPopulation

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _compute_metrics(population: ConceptPopulation) -> Dict[str, Any]:
    """Compute all population-level metrics for the report."""
    instances = population.instances
    n = len(instances)

    initial_scores = []
    final_scores = []
    deltas = []
    human_interventions = 0

    for inst in instances:
        if inst.human_signal is not None:
            human_interventions += 1

        final = inst.adequacy_score
        initial = inst.initial_score

        if final is not None:
            final_scores.append(final)
        if initial is not None:
            initial_scores.append(initial)
        if inst.score_delta is not None:
            deltas.append(inst.score_delta)

    def _mean(lst):
        return round(sum(lst) / len(lst), 2) if lst else None

    mean_initial = _mean(initial_scores)
    mean_final = _mean(final_scores)
    mean_delta = _mean(deltas)

    # Mean improvement %: delta / initial * 100 per instance (ignoring zero-initial)
    improvements = []
    for inst in instances:
        init = inst.initial_score
        fin = inst.adequacy_score
        if init is not None and fin is not None and init > 0:
            improvements.append((fin - init) / init * 100)
    mean_improvement_pct = round(sum(improvements) / len(improvements), 1) if improvements else 0.0

    return {
        "population_breadth": population.population_breadth,
        "goal_coverage": len(population.goal_coverage),
        "context_coverage": len(population.context_coverage),
        "grammatical_frame_coverage": len(population.grammatical_frames),
        "mean_initial_score": mean_initial,
        "mean_final_score": mean_final,
        "mean_delta": mean_delta,
        "mean_improvement_pct": mean_improvement_pct,
        "human_interventions": human_interventions,
        "n_instances": n,
    }


def _verdict(metrics: Dict[str, Any]) -> str:
    breadth = metrics["population_breadth"]
    improvement = metrics["mean_improvement_pct"]

    if improvement > 25 and breadth >= 3:
        return (
            "**Rich concept population achieved** — strong coverage across goals and contexts."
        )
    if (10 <= improvement <= 25) or breadth == 2:
        return (
            "**Moderate concept population** — consider adding more context-goal pairs "
            "for richer coverage."
        )
    return (
        "**Sparse concept population** — this concept is under-represented. "
        "Add more contexts and goals."
    )


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def _build_report(population: ConceptPopulation, metrics: Dict[str, Any]) -> str:
    """Build the full markdown report string."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ── Header ────────────────────────────────────────────────────────────
    seed_line = f"  \n**Seed Phrase:** {population.seed_phrase}" if population.seed_phrase else ""
    lines = [
        "# Concept Population Report",
        "",
        f"**Concept:** {population.term}{seed_line}  ",
        f"**Generated:** {now}",
        "",
        "---",
        "",
        "## Population Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Population Breadth | {metrics['population_breadth']} instances |",
        f"| Goal Coverage | {metrics['goal_coverage']} distinct goals |",
        f"| Context Coverage | {metrics['context_coverage']} distinct contexts |",
        f"| Grammatical Frame Coverage | {metrics['grammatical_frame_coverage']} distinct frames |",
        f"| Mean Adequacy Score (Initial) | {metrics['mean_initial_score'] or 'N/A'}/10 |",
        f"| Mean Adequacy Score (Final) | {metrics['mean_final_score'] or 'N/A'}/10 |",
        f"| Mean Score Delta | {metrics['mean_delta'] or 'N/A'} |",
        f"| Mean Improvement | {metrics['mean_improvement_pct']}% |",
        f"| Human Interventions | {metrics['human_interventions']} |",
        "",
        "---",
        "",
        "## Theoretical Note",
        "",
        "> A concept, following Barrett's constructionist model, is not a single definition but a",
        "> *population of goal-indexed, grammatically-grounded contextual instances* — predictions",
        "> about what this concept, in its specific grammatical construction, produces in particular",
        "> situations to serve specific goals. The table below shows this population.",
        "",
        "---",
        "",
        "## Instance Population Table",
        "",
        "| Context | Goal | Gram. Frame | Final Simulation | Round | Adequacy | Human Signal |",
        "|---|---|---|---|---|---|---|",
    ]

    for inst in population.instances:
        score_str = f"{inst.adequacy_score:.1f}/10" if inst.adequacy_score is not None else "N/A"
        sig = inst.human_signal or "—"
        # Truncate long simulation text for table readability
        sim_display = inst.simulation[:100].replace("|", "\\|")
        if len(inst.simulation) > 100:
            sim_display += "…"
        ctx_display = inst.context[:50].replace("|", "\\|")
        goal_display = inst.goal[:50].replace("|", "\\|")
        frame_display = (inst.grammatical_frame[:30].replace("|", "\\|") if inst.grammatical_frame else "—")
        lines.append(
            f"| {ctx_display} | {goal_display} | {frame_display} | {sim_display} | {inst.round} | {score_str} | {sig} |"
        )

    # ── Score delta table ─────────────────────────────────────────────────
    lines += [
        "",
        "---",
        "",
        "## Adequacy Improvement by Instance",
        "",
        "| Instance | Context | Initial Score | Final Score | Delta | Rounds |",
        "|---|---|---|---|---|---|",
    ]

    for i, inst in enumerate(population.instances, start=1):
        init_str = f"{inst.initial_score:.1f}" if inst.initial_score is not None else "N/A"
        fin_str = f"{inst.adequacy_score:.1f}" if inst.adequacy_score is not None else "N/A"
        delta_str = f"{inst.score_delta:+.1f}" if inst.score_delta is not None else "N/A"
        ctx_short = inst.context[:50].replace("|", "\\|")
        lines.append(
            f"| {i} | {ctx_short} | {init_str} | {fin_str} | {delta_str} | {inst.round} |"
        )

    # ── Human feedback trace ──────────────────────────────────────────────
    feedback_instances = [i for i in population.instances if i.human_signal]
    if feedback_instances:
        lines += [
            "",
            "---",
            "",
            "## Human Feedback Trace",
            "",
        ]
        for inst in feedback_instances:
            hint_text = inst.hint or "none"
            lines += [
                f"### {inst.context[:60]} → {inst.goal[:60]}",
                f"- Signal: `{inst.human_signal}` | Hint: \"{hint_text}\"",
                "",
            ]

    # ── Verdict ───────────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## Population Verdict",
        "",
        _verdict(metrics),
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_report(
    population: ConceptPopulation,
    output_path: str = "docs/concept_population_report.md",
) -> Dict[str, Any]:
    """Generate a Concept Population Report and write it to disk.

    Parameters
    ----------
    population:   Completed ConceptPopulation (post RL loop + optional RLHF).
    output_path:  Path to write the markdown report to.

    Returns
    -------
    metrics dict (also used by notebook cell output).
    """
    metrics = _compute_metrics(population)
    report_md = _build_report(population, metrics)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(report_md)

    # Console summary (concept-clarity-report SKILL Step 4)
    seed_display = f" ({population.seed_phrase})" if population.seed_phrase else ""
    print(
        f"\n✅ Concept Population Report saved to {output_path}\n"
        f"   Concept: {population.term}{seed_display}\n"
        f"   Population breadth: {metrics['population_breadth']} instances\n"
        f"   Goal coverage: {metrics['goal_coverage']} | "
        f"Context coverage: {metrics['context_coverage']} | "
        f"Grammatical frames: {metrics['grammatical_frame_coverage']}\n"
        f"   Mean adequacy: {metrics['mean_initial_score']} → {metrics['mean_final_score']} "
        f"({metrics['mean_improvement_pct']}% improvement)\n"
    )

    return {
        "report_path": output_path,
        "metrics": metrics,
    }
