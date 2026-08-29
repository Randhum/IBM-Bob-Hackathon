"""
generate_corpus.py — Batch self-generation script for the construction-domain training corpus.

Reads data/corpus_spec.json, runs the RL generation loop over all 10 terms × 5 (context, goal)
pairs using the existing concept_loop.run_rl_loop(), saves raw auto-scored populations to
data/construction_raw_population.json, then runs hybrid human review of borderline instances
(score 6.0–8.0) and saves the final labelled populations to
data/construction_labelled_population.json.

Usage:
    # Full run (real API, with human review of borderline cases):
    python -m src.generate_corpus

    # Skip human review (useful for automated pipelines):
    python -m src.generate_corpus --no-human

    # Dry-run with stub API (no credentials required):
    WATSONX_STUB=true python -m src.generate_corpus --no-human
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent
_DATA_DIR = _REPO_ROOT / "data"
_CORPUS_SPEC = _DATA_DIR / "corpus_spec.json"
_RAW_OUTPUT = _DATA_DIR / "construction_raw_population.json"
_LABELLED_OUTPUT = _DATA_DIR / "construction_labelled_population.json"

# Score boundaries for hybrid labelling
_AUTO_ACCEPT_THRESHOLD = 8.0   # above this → auto-accept, no human review
_AUTO_REJECT_THRESHOLD = 6.0   # below this → auto-reject, excluded from training set
# 6.0 ≤ score ≤ 8.0 → borderline, presented for human review


def _load_corpus_spec() -> Dict[str, Any]:
    if not _CORPUS_SPEC.exists():
        logger.error("corpus_spec.json not found at %s", _CORPUS_SPEC)
        sys.exit(1)
    with open(_CORPUS_SPEC, encoding="utf-8") as f:
        return json.load(f)


def _run_generation(corpus_spec: Dict[str, Any], max_iterations: int, threshold: float) -> List[Dict]:
    """Run RL generation loop for every term in the corpus spec.

    Returns a list of population dicts (one per term) serialised via ConceptPopulation.to_dict().
    """
    from .concept_loop import run_rl_loop

    populations = []
    terms = corpus_spec["terms"]
    logger.info("Starting generation for %d terms…", len(terms))

    for term_spec in terms:
        term = term_spec["term"]
        seed_phrase = term_spec.get("seed_phrase", "")
        grammatical_frame = term_spec.get("grammatical_frame", "")
        morphemes = term_spec.get("morphemes", [])
        context_goal_pairs = [
            (inst["context"], inst["goal"]) for inst in term_spec["instances"]
        ]

        logger.info("Generating population for term: %r (%d pairs)", term, len(context_goal_pairs))

        population = run_rl_loop(
            term=term,
            context_goal_pairs=context_goal_pairs,
            seed_phrase=seed_phrase,
            grammatical_frame=grammatical_frame,
            morphemes=morphemes if morphemes else None,
            max_iterations=max_iterations,
            threshold=threshold,
        )

        pop_dict = population.to_dict()
        populations.append(pop_dict)
        logger.info(
            "  ✓ %r — %d instances, avg score %.2f",
            term,
            pop_dict["population_breadth"],
            _avg_score(pop_dict),
        )

    return populations


def _avg_score(pop_dict: Dict) -> float:
    scores = [
        inst["adequacy_score"]
        for inst in pop_dict.get("instances", [])
        if inst.get("adequacy_score") is not None
    ]
    return sum(scores) / len(scores) if scores else 0.0


def _save_populations(populations: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(populations, f, indent=2)
    logger.info("Saved %d populations → %s", len(populations), path)


def _load_populations(path: Path) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _count_by_band(populations: List[Dict]) -> Dict[str, int]:
    """Return counts of instances in each score band across all populations."""
    auto_accept = borderline = auto_reject = unscored = 0
    for pop in populations:
        for inst in pop.get("instances", []):
            score = inst.get("adequacy_score")
            if score is None:
                unscored += 1
            elif score > _AUTO_ACCEPT_THRESHOLD:
                auto_accept += 1
            elif score >= _AUTO_REJECT_THRESHOLD:
                borderline += 1
            else:
                auto_reject += 1
    return {
        "auto_accept": auto_accept,
        "borderline": borderline,
        "auto_reject": auto_reject,
        "unscored": unscored,
    }


def _run_human_review(populations: List[Dict]) -> List[Dict]:
    """Present borderline instances (6.0 ≤ score ≤ 8.0) for human review.

    Modifies populations in-place, setting human_signal on each reviewed instance.
    Returns the updated populations list.
    """
    from .human_feedback import collect_human_feedback
    from .concept_population import ConceptPopulation

    for pop_dict in populations:
        term = pop_dict["term"]
        population = ConceptPopulation.from_dict(pop_dict)

        # Filter to borderline instances only
        borderline = [
            inst for inst in population.instances
            if inst.adequacy_score is not None
            and _AUTO_REJECT_THRESHOLD <= inst.adequacy_score <= _AUTO_ACCEPT_THRESHOLD
        ]

        if not borderline:
            logger.info("  %r — no borderline instances, skipping human review.", term)
            # Auto-accept all instances above threshold
            for inst in population.instances:
                if inst.human_signal is None:
                    if inst.adequacy_score is not None and inst.adequacy_score > _AUTO_ACCEPT_THRESHOLD:
                        inst.human_signal = "accept"
                    elif inst.adequacy_score is not None and inst.adequacy_score < _AUTO_REJECT_THRESHOLD:
                        inst.human_signal = "reject"
        else:
            logger.info("  %r — %d borderline instance(s) for human review.", term, len(borderline))
            # Build a temporary population with only borderline instances for review
            borderline_pop = ConceptPopulation(term=term, seed_phrase=population.seed_phrase)
            for inst in borderline:
                borderline_pop.add_instance(inst)

            reviewed_pop = collect_human_feedback(borderline_pop, term=term)

            # Merge reviewed signals back into the main population
            reviewed_by_id = {inst.id: inst for inst in reviewed_pop.instances}
            for inst in population.instances:
                if inst.id in reviewed_by_id:
                    inst.human_signal = reviewed_by_id[inst.id].human_signal
                    inst.hint = reviewed_by_id[inst.id].hint
                elif inst.human_signal is None:
                    if inst.adequacy_score is not None and inst.adequacy_score > _AUTO_ACCEPT_THRESHOLD:
                        inst.human_signal = "accept"
                    elif inst.adequacy_score is not None and inst.adequacy_score < _AUTO_REJECT_THRESHOLD:
                        inst.human_signal = "reject"

        # Rebuild the dict from the updated population
        pop_idx = next(
            i for i, p in enumerate(populations) if p["term"] == term
        )
        populations[pop_idx] = population.to_dict()

    return populations


def _print_summary(populations: List[Dict], label: str) -> None:
    bands = _count_by_band(populations)
    total = sum(bands.values())
    print(f"\n{'─' * 56}")
    print(f"  {label}")
    print(f"{'─' * 56}")
    print(f"  Total instances : {total}")
    print(f"  Auto-accept (>8.0)  : {bands['auto_accept']}")
    print(f"  Borderline (6–8)    : {bands['borderline']}")
    print(f"  Auto-reject (<6.0)  : {bands['auto_reject']}")
    print(f"  Unscored            : {bands['unscored']}")
    print(f"{'─' * 56}\n")


def main(args: argparse.Namespace) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load corpus spec ───────────────────────────────────────────
    corpus_spec = _load_corpus_spec()
    logger.info("Loaded corpus spec: %d terms", len(corpus_spec["terms"]))

    # ── Step 2: Generate raw populations ──────────────────────────────────
    populations = _run_generation(
        corpus_spec,
        max_iterations=args.max_iterations,
        threshold=args.threshold,
    )
    _save_populations(populations, _RAW_OUTPUT)
    _print_summary(populations, "RAW GENERATION SUMMARY")

    # ── Step 3: Hybrid human labelling ────────────────────────────────────
    if args.no_human:
        logger.info("--no-human set: auto-labelling all instances by score band.")
        for pop in populations:
            for inst in pop.get("instances", []):
                if inst.get("human_signal") is None:
                    score = inst.get("adequacy_score")
                    if score is None:
                        inst["human_signal"] = "reject"
                    elif score >= _AUTO_REJECT_THRESHOLD:
                        inst["human_signal"] = "accept"
                    else:
                        inst["human_signal"] = "reject"
    else:
        print(
            "\n⚙️  Starting hybrid human review.\n"
            f"  Instances scoring {_AUTO_REJECT_THRESHOLD}–{_AUTO_ACCEPT_THRESHOLD} "
            "will be presented for your review.\n"
            "  All others are auto-labelled by score.\n"
        )
        populations = _run_human_review(populations)

    _save_populations(populations, _LABELLED_OUTPUT)
    _print_summary(populations, "LABELLED CORPUS SUMMARY (after human review)")

    # ── Step 4: Final acceptance counts ───────────────────────────────────
    accepted = sum(
        1 for pop in populations
        for inst in pop.get("instances", [])
        if inst.get("human_signal") == "accept"
    )
    logger.info("Final accepted instances: %d (target ≥ 40)", accepted)
    if accepted < 40:
        logger.warning(
            "Accepted count (%d) is below target of 40. Consider re-running with "
            "--threshold lower or reviewing rejected instances.", accepted
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch construction-domain corpus generation for Barrett LoRA training."
    )
    parser.add_argument(
        "--max-iterations", type=int, default=2,
        help="Max RL refinement rounds per instance (default: 2).",
    )
    parser.add_argument(
        "--threshold", type=float, default=7.5,
        help="Adequacy score threshold; refinement stops above this (default: 7.5).",
    )
    parser.add_argument(
        "--no-human", action="store_true",
        help="Skip human review; auto-label all instances by score band.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(_parse_args())
