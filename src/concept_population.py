"""
concept_population.py — Barrett-aligned data model for the concept-learning workflow.

A concept is represented as a *population of variable instances*, each indexed by a
context and a goal — not a single fixed definition (Barrett, 2017).

Five-level linguistic granularity (docs/concept-ontology.md §3.4):
  Level 0  letter/phoneme  — substrate; phonesthetics_note is an optional signal here
  Level 1  BPE token       — REJECTED; arbitrary, not meaningful
  Level 2  morpheme        — optional annotation; carries bounded sub-lexical meaning
  Level 3  word            — polysemous seed
  Level 4  seed_phrase     — grammatically framed; minimum required input
  Level 5  instance        — context + goal + simulation; primary output unit

Key fields on ConceptInstance:
  morphemes          — optional list of morphemes, e.g. ["mis-", "trust"]
                       NOT BPE tokens. Meaningful sub-word units only.
  phonesthetics_note — optional free-text sound-symbolism annotation
  seed_phrase        — grammatically framed seed form (required, Level 4)
  grammatical_frame  — syntactic role (required, Level 4)
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass
class ConceptInstance:
    """A single contextual simulation within a concept population.

    Corresponds to one (context, goal) pair and the simulation produced for it.
    Levels 2–5 of the linguistic granularity model are represented as fields
    (see docs/concept-ontology.md §3.4 and §4).
    """

    context: str
    goal: str
    simulation: str
    # Level 2 — Sub-lexical (optional; enriches prompts when provided)
    morphemes: List[str] = field(default_factory=list)   # e.g. ["mis-", "trust"]
    phonesthetics_note: str = ""                          # e.g. "sl- cluster: smooth unpleasantness"
    # Level 4 — Grammatical construction (prevents tokenization collapse)
    seed_phrase: str = ""          # grammatically framed seed, e.g. "to fire (someone)"
    grammatical_frame: str = ""    # syntactic role, e.g. "transitive verb, agent=X, patient=Y"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    adequacy_score: Optional[float] = None
    human_signal: Optional[str] = None   # "accept" | "reject" | "refine" | None
    hint: Optional[str] = None
    round: int = 0
    # history tracks each round's simulation + score for score-delta reporting
    history: List[Dict[str, Any]] = field(default_factory=list)

    def record_round(self) -> None:
        """Append the current (round, simulation, adequacy_score) to history."""
        self.history.append({
            "round": self.round,
            "simulation": self.simulation,
            "adequacy_score": self.adequacy_score,
        })

    @property
    def initial_score(self) -> Optional[float]:
        """Score at round 0 (first history entry)."""
        if self.history:
            return self.history[0].get("adequacy_score")
        return None

    @property
    def score_delta(self) -> Optional[float]:
        """Final score minus initial score. None if either is unavailable."""
        initial = self.initial_score
        final = self.adequacy_score
        if initial is not None and final is not None:
            return round(final - initial, 2)
        return None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConceptInstance":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            context=data["context"],
            goal=data["goal"],
            simulation=data["simulation"],
            morphemes=data.get("morphemes", []),
            phonesthetics_note=data.get("phonesthetics_note", ""),
            seed_phrase=data.get("seed_phrase", ""),
            grammatical_frame=data.get("grammatical_frame", ""),
            adequacy_score=data.get("adequacy_score"),
            human_signal=data.get("human_signal"),
            hint=data.get("hint"),
            round=data.get("round", 0),
            history=data.get("history", []),
        )


@dataclass
class ConceptPopulation:
    """A Barrett-aligned population of contextual instances for a single concept term.

    The population GROWS through RL iterations — instances are never replaced,
    only added and refined.

    grammatical_frames tracks all distinct syntactic constructions represented in
    the population. Same word, different frame = potentially different concept.
    See docs/concept-ontology.md §3.
    """

    term: str
    seed_phrase: str = ""                  # canonical grammatically framed form for this population
    instances: List[ConceptInstance] = field(default_factory=list)
    goal_coverage: List[str] = field(default_factory=list)
    context_coverage: List[str] = field(default_factory=list)
    grammatical_frames: List[str] = field(default_factory=list)
    population_breadth: int = 0

    def add_instance(self, instance: ConceptInstance) -> None:
        """Add an instance and update coverage metadata."""
        self.instances.append(instance)
        if instance.goal not in self.goal_coverage:
            self.goal_coverage.append(instance.goal)
        if instance.context not in self.context_coverage:
            self.context_coverage.append(instance.context)
        if instance.grammatical_frame and instance.grammatical_frame not in self.grammatical_frames:
            self.grammatical_frames.append(instance.grammatical_frame)
        self.population_breadth = len(self.instances)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "term": self.term,
            "seed_phrase": self.seed_phrase,
            "instances": [i.to_dict() for i in self.instances],
            "goal_coverage": self.goal_coverage,
            "context_coverage": self.context_coverage,
            "grammatical_frames": self.grammatical_frames,
            "population_breadth": self.population_breadth,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConceptPopulation":
        pop = cls(term=data["term"], seed_phrase=data.get("seed_phrase", ""))
        pop.instances = [ConceptInstance.from_dict(i) for i in data.get("instances", [])]
        pop.goal_coverage = data.get("goal_coverage", [])
        pop.context_coverage = data.get("context_coverage", [])
        pop.grammatical_frames = data.get("grammatical_frames", [])
        pop.population_breadth = data.get("population_breadth", len(pop.instances))
        return pop

    @classmethod
    def from_json(cls, json_str: str) -> "ConceptPopulation":
        return cls.from_dict(json.loads(json_str))
