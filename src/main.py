"""
main.py — CLI entry point for the concept-learning workflow.

Usage examples:
    python -m src.main --term "anger" --contexts-file contexts.json
    python -m src.main --term "anger" \\
        --context "receiving unfair criticism" --goal "restore social fairness"
    python -m src.main --term "anger" --contexts-file contexts.json \\
        --max-iterations 3 --threshold 7.5 --output docs/report.md --no-human

Orchestrates:
  1. run_rl_loop     — generate and auto-score instances
  2. collect_human_feedback (optional, skipped with --no-human)
  3. generate_report — write Concept Population Report
  4. Save ConceptPopulation JSON to docs/{term}_population.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

load_dotenv()

# Configure logging before importing src modules
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from .concept_loop import run_rl_loop
from .human_feedback import collect_human_feedback
from .report import generate_report


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="src.main",
        description="Concept-learning workflow (Barrett population model + watsonx.ai)",
    )
    parser.add_argument("--term", required=True, help="Concept term (raw word/phrase, e.g. 'fire').")
    parser.add_argument(
        "--seed-phrase", default="", metavar="PHRASE",
        help=(
            "Grammatically framed form of the term, e.g. 'to fire (someone)'. "
            "Prevents tokenization collapse — same word, different frame = different concept. "
            "See docs/concept-ontology.md §3."
        ),
    )
    parser.add_argument(
        "--grammatical-frame", default="", metavar="FRAME",
        help=(
            "Syntactic role, e.g. 'transitive verb, agent=manager, patient=employee'. "
            "Injected into all LLM prompts to anchor concept construction."
        ),
    )

    # Input: either a contexts-file or inline context+goal
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--contexts-file",
        metavar="PATH",
        help=(
            "Path to a JSON file containing a list of {context, goal} objects. "
            "Example: [{\"context\": \"...\", \"goal\": \"...\"}]"
        ),
    )
    input_group.add_argument(
        "--context",
        metavar="TEXT",
        help="Single context string (must also supply --goal).",
    )

    parser.add_argument("--goal", metavar="TEXT", help="Goal for the single --context pair.")
    parser.add_argument("--max-iterations", type=int, default=3, metavar="N",
                        help="Maximum RL refinement rounds per instance (default: 3).")
    parser.add_argument("--threshold", type=float, default=7.5, metavar="SCORE",
                        help="Adequacy score threshold 0–10 (default: 7.5).")
    parser.add_argument("--output", default="docs/concept_population_report.md", metavar="PATH",
                        help="Report output path (default: docs/concept_population_report.md).")
    parser.add_argument("--no-human", action="store_true",
                        help="Skip the interactive RLHF human-feedback step.")
    parser.add_argument("--hint", metavar="TEXT",
                        help="Optional global hint injected into all RL generation prompts.")
    return parser.parse_args(argv)


def _load_context_goal_pairs(args: argparse.Namespace) -> List[Tuple[str, str]]:
    if args.contexts_file:
        path = Path(args.contexts_file)
        if not path.exists():
            print(f"❌ Contexts file not found: {path}", file=sys.stderr)
            sys.exit(1)
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        pairs = []
        for item in data:
            ctx = item.get("context") or item.get("ctx", "")
            goal = item.get("goal", "")
            if ctx and goal:
                pairs.append((ctx, goal))
        if not pairs:
            print("❌ No valid {context, goal} pairs found in contexts file.", file=sys.stderr)
            sys.exit(1)
        return pairs

    # Inline single pair
    if not args.goal:
        print("❌ --goal is required when using --context.", file=sys.stderr)
        sys.exit(1)
    return [(args.context, args.goal)]


def main(argv: List[str] | None = None) -> None:
    args = _parse_args(argv)
    context_goal_pairs = _load_context_goal_pairs(args)

    seed_phrase = args.seed_phrase or ""
    grammatical_frame = args.grammatical_frame or ""

    print(
        f"\n🚀 Starting concept-learning workflow\n"
        f"   Term:             {args.term}\n"
        f"   Seed phrase:      {seed_phrase or '(not set)'}\n"
        f"   Grammatical frame:{grammatical_frame or '(not set)'}\n"
        f"   Pairs:            {len(context_goal_pairs)}\n"
        f"   Max iterations:   {args.max_iterations} | Threshold: {args.threshold}\n"
        f"   Stub mode:        {os.getenv('WATSONX_STUB', 'false')}\n"
    )

    # ── Step 1: RL loop ────────────────────────────────────────────────────
    population = run_rl_loop(
        term=args.term,
        context_goal_pairs=context_goal_pairs,
        seed_phrase=seed_phrase,
        grammatical_frame=grammatical_frame,
        max_iterations=args.max_iterations,
        threshold=args.threshold,
        hint=args.hint,
    )

    print(f"\n📊 RL loop complete. Population breadth: {population.population_breadth}")

    # ── Step 2: Human RLHF (optional) ─────────────────────────────────────
    if not args.no_human:
        print("\n🧑 Starting human feedback collection (--no-human to skip)…\n")
        population = collect_human_feedback(population, term=args.term)
    else:
        print("\n⏭  Skipping human feedback (--no-human flag set).")

    # ── Step 3: Save population JSON ───────────────────────────────────────
    docs_dir = Path(args.output).parent
    docs_dir.mkdir(parents=True, exist_ok=True)
    safe_term = args.term.replace(" ", "_").replace("/", "-")
    json_path = docs_dir / f"{safe_term}_population.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        fh.write(population.to_json())
    print(f"\n💾 ConceptPopulation saved to {json_path}")

    # ── Step 4: Generate report ────────────────────────────────────────────
    generate_report(population, output_path=args.output)


if __name__ == "__main__":
    main()
