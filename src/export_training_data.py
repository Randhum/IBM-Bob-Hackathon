"""
export_training_data.py — Dual-format JSONL exporter for construction-domain LoRA training.

Reads data/construction_labelled_population.json and writes three files:

  data/construction_judge_training.jsonl
      Format A — teaches the model to score functional adequacy.
      input:  term + seed_phrase + grammatical_frame + context + goal + simulation + "Rate…"
      output: adequacy_score as string
      Includes all accepted instances (full score range).

  data/construction_generator_training.jsonl
      Format B — teaches the model to construct goal-indexed simulations.
      input:  term + seed_phrase + grammatical_frame + context + goal + "Generate simulation:"
      output: simulation text
      Includes only accepted instances with adequacy_score >= GENERATOR_THRESHOLD (default 8.0).

  data/construction_eval.jsonl
      Held-out evaluation set — one instance per term (highest adequacy_score).
      Used by the benchmark notebook to compare base vs fine-tuned models.
      Excluded from both training files.

Usage:
    python -m src.export_training_data
    python -m src.export_training_data --generator-threshold 7.5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent
_DATA_DIR = _REPO_ROOT / "data"

_LABELLED_INPUT = _DATA_DIR / "construction_labelled_population.json"
_JUDGE_OUTPUT = _DATA_DIR / "construction_judge_training.jsonl"
_GENERATOR_OUTPUT = _DATA_DIR / "construction_generator_training.jsonl"
_EVAL_OUTPUT = _DATA_DIR / "construction_eval.jsonl"

_DEFAULT_GENERATOR_THRESHOLD = 8.0


# ---------------------------------------------------------------------------
# Prompt builders (imported from src modules; replicated here as thin wrappers
# so this script can be run stand-alone for debugging without circular imports)
# ---------------------------------------------------------------------------

def _build_judge_input(inst: Dict[str, Any]) -> str:
    """Build Format A input prompt for a labelled instance dict."""
    from .judge import build_judge_prompt
    return build_judge_prompt(
        term=inst["term"],
        context=inst["context"],
        goal=inst["goal"],
        simulation=inst["simulation"],
        seed_phrase=inst.get("seed_phrase", ""),
        grammatical_frame=inst.get("grammatical_frame", ""),
        morphemes=inst.get("morphemes") or None,
        phonesthetics_note=inst.get("phonesthetics_note", ""),
    )


def _build_generator_input(inst: Dict[str, Any]) -> str:
    """Build Format B input prompt (no simulation in input — that is the target)."""
    from .concept_loop import build_generator_prompt
    return build_generator_prompt(
        term=inst["term"],
        context=inst["context"],
        goal=inst["goal"],
        seed_phrase=inst.get("seed_phrase", ""),
        grammatical_frame=inst.get("grammatical_frame", ""),
        morphemes=inst.get("morphemes") or None,
        phonesthetics_note=inst.get("phonesthetics_note", ""),
        hint=None,
    )


# ---------------------------------------------------------------------------
# Load + flatten
# ---------------------------------------------------------------------------

def _load_populations() -> List[Dict[str, Any]]:
    if not _LABELLED_INPUT.exists():
        logger.error("Labelled population file not found: %s", _LABELLED_INPUT)
        logger.error("Run src/generate_corpus.py first.")
        sys.exit(1)
    with open(_LABELLED_INPUT, encoding="utf-8") as f:
        return json.load(f)


def _flatten_instances(populations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten all instances across populations, injecting term and population-level fields."""
    flat = []
    for pop in populations:
        term = pop["term"]
        seed_phrase = pop.get("seed_phrase", "")
        for inst in pop.get("instances", []):
            enriched = dict(inst)
            enriched["term"] = term
            # Inherit population-level seed_phrase if instance doesn't have one
            if not enriched.get("seed_phrase"):
                enriched["seed_phrase"] = seed_phrase
            flat.append(enriched)
    return flat


def _is_accepted(inst: Dict[str, Any]) -> bool:
    """An instance is accepted if human_signal == 'accept', or score > 8.0 with no rejection."""
    signal = inst.get("human_signal")
    if signal == "reject":
        return False
    if signal == "accept":
        return True
    # Fallback for instances without explicit human signal (auto-accepted by score)
    score = inst.get("adequacy_score")
    return score is not None and score > 8.0


# ---------------------------------------------------------------------------
# Held-out eval selection
# ---------------------------------------------------------------------------

def _select_eval_instances(flat: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Select the highest-scoring accepted instance per term for the eval set.

    Returns a dict mapping term → instance dict.
    """
    best: Dict[str, Dict[str, Any]] = {}
    for inst in flat:
        if not _is_accepted(inst):
            continue
        term = inst["term"]
        score = inst.get("adequacy_score") or 0.0
        if term not in best or score > (best[term].get("adequacy_score") or 0.0):
            best[term] = inst
    return best


# ---------------------------------------------------------------------------
# JSONL writers
# ---------------------------------------------------------------------------

def _write_jsonl(records: List[Dict[str, Any]], path: Path) -> int:
    """Write a list of dicts as newline-delimited JSON. Returns line count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            line = json.dumps(rec, ensure_ascii=False)
            f.write(line + "\n")
            count += 1
    return count


def _validate_jsonl(path: Path, output_is_score: bool = False) -> List[str]:
    """Validate a JSONL file. Returns a list of error strings (empty = OK)."""
    errors = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"Line {i}: invalid JSON — {exc}")
                continue
            if not obj.get("input"):
                errors.append(f"Line {i}: empty 'input' field")
            if not obj.get("output"):
                errors.append(f"Line {i}: empty 'output' field")
            if output_is_score:
                try:
                    val = float(obj["output"])
                    if not (0.0 <= val <= 10.0):
                        errors.append(f"Line {i}: score {val} out of range [0, 10]")
                except (ValueError, KeyError):
                    errors.append(f"Line {i}: output is not a valid float: {obj.get('output')!r}")
    return errors


# ---------------------------------------------------------------------------
# Main export logic
# ---------------------------------------------------------------------------

def export(generator_threshold: float = _DEFAULT_GENERATOR_THRESHOLD) -> None:
    populations = _load_populations()
    flat = _flatten_instances(populations)

    total = len(flat)
    accepted_count = sum(1 for inst in flat if _is_accepted(inst))
    logger.info("Loaded %d total instances, %d accepted.", total, accepted_count)

    # ── Select held-out eval set (one per term, best score) ────────────────
    eval_by_term = _select_eval_instances(flat)
    eval_ids = {inst["id"] for inst in eval_by_term.values()}
    logger.info("Held-out eval set: %d instances (one per term).", len(eval_by_term))

    # ── Write eval JSONL ────────────────────────────────────────────────────
    eval_records = [
        {
            "term": inst["term"],
            "seed_phrase": inst.get("seed_phrase", ""),
            "grammatical_frame": inst.get("grammatical_frame", ""),
            "context": inst["context"],
            "goal": inst["goal"],
            "simulation": inst["simulation"],
            "adequacy_score": inst.get("adequacy_score"),
            "judge_input": _build_judge_input(inst),
            "generator_input": _build_generator_input(inst),
        }
        for inst in eval_by_term.values()
    ]
    eval_count = _write_jsonl(eval_records, _EVAL_OUTPUT)
    logger.info("Wrote eval set: %d lines → %s", eval_count, _EVAL_OUTPUT)

    # ── Build training sets (exclude held-out eval instances) ──────────────
    training_instances = [
        inst for inst in flat
        if _is_accepted(inst) and inst["id"] not in eval_ids
    ]

    # Format A — Judge training (all accepted, full score range)
    judge_records = []
    for inst in training_instances:
        score = inst.get("adequacy_score")
        if score is None:
            continue
        judge_records.append({
            "input": _build_judge_input(inst),
            "output": str(round(score, 2)),
        })

    judge_count = _write_jsonl(judge_records, _JUDGE_OUTPUT)
    logger.info("Wrote judge training set: %d lines → %s", judge_count, _JUDGE_OUTPUT)

    # Format B — Generator training (high-quality only, score >= threshold)
    generator_records = []
    for inst in training_instances:
        score = inst.get("adequacy_score")
        if score is None or score < generator_threshold:
            continue
        simulation = inst.get("simulation", "").strip()
        if not simulation:
            continue
        generator_records.append({
            "input": _build_generator_input(inst),
            "output": simulation,
        })

    generator_count = _write_jsonl(generator_records, _GENERATOR_OUTPUT)
    logger.info(
        "Wrote generator training set (score ≥ %.1f): %d lines → %s",
        generator_threshold, generator_count, _GENERATOR_OUTPUT,
    )

    # ── Validate both JSONL files ───────────────────────────────────────────
    for path, is_score in [(_JUDGE_OUTPUT, True), (_GENERATOR_OUTPUT, False)]:
        errors = _validate_jsonl(path, output_is_score=is_score)
        if errors:
            logger.error("Validation errors in %s:", path)
            for err in errors:
                logger.error("  %s", err)
        else:
            logger.info("✓ %s validated OK.", path.name)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  EXPORT SUMMARY")
    print(f"{'─' * 60}")
    print(f"  Total instances loaded        : {total}")
    print(f"  Accepted instances            : {accepted_count}")
    print(f"  Held-out eval (one/term)      : {eval_count}")
    print(f"  Judge training lines (Format A): {judge_count}")
    print(f"  Generator training lines (B)  : {generator_count}  (threshold ≥ {generator_threshold})")
    print(f"{'─' * 60}\n")

    if judge_count < 40:
        logger.warning(
            "Judge training set (%d lines) is below target of 40. "
            "Re-run generate_corpus.py or lower --generator-threshold.", judge_count
        )
    if generator_count < 25:
        logger.warning(
            "Generator training set (%d lines) is below target of 25. "
            "Consider lowering --generator-threshold (currently %.1f).", generator_count, generator_threshold
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export labelled construction-domain corpus to dual-format JSONL for LoRA fine-tuning."
    )
    parser.add_argument(
        "--generator-threshold", type=float, default=_DEFAULT_GENERATOR_THRESHOLD,
        help=f"Minimum adequacy score for generator training examples (default: {_DEFAULT_GENERATOR_THRESHOLD}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    export(generator_threshold=args.generator_threshold)
