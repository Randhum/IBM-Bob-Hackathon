# Problem & Solution Statement

## The Problem

Large language models are trained on token co-occurrence statistics. A token is a sub-word
fragment — the word `"fire"` may become `["fi", "##re"]`; the phrase `"firing an employee"`
shares token weight space with `"firing a gun"` because both reduce to the same fragments
with similar co-occurrence statistics. This is the tokenization problem: an LLM has no
concept of *fire* at all — it has a high-probability generator calibrated to contexts where
that token cluster appears in training text.

Below the tokenization problem lies a deeper one. Even if we move to whole words, the same
word constructs entirely different concepts depending on grammatical role: `"fire the
employee"` (termination), `"fire the gun"` (discharge), `"the fire burns"` (combustion),
`"fire in her eyes"` (intensity). A token-weight system collapses these into a single
distribution. A concept-construction system must treat them as distinct, because grammar
is the machinery of concept construction — not decoration around a token.

Grounded in Lisa Feldman Barrett's constructionist theory of concepts, the correct framing
is: an LLM lacks a *population* of goal-indexed, grammatically-grounded conceptual
instances. It cannot select the contextually adequate simulation of a concept for a given
context and goal, because it does not represent concepts as populations at all.

## The Solution

This project builds a Bob-orchestrated workflow that constructs a `ConceptPopulation` for
any target concept — not as a definition, but as Barrett describes: a family of variable
instances, each anchored to a specific grammatical frame, context, and functional goal,
evaluated by how well its simulation serves that goal.

Input is a seed phrase (grammatically framed — `"to fire someone"`, not `"fire"`), a
grammatical frame (e.g. `"transitive verb, agent=manager, patient=employee"`), and a set
of (context, goal) pairs. The workflow then runs two nested loops.

The inner RL loop calls watsonx.ai to generate a candidate simulation for each
(context, goal) pair, then scores it via a context-aware judge prompt for *functional
adequacy* — how well the simulation predicts the experience or behavior this concept
produces here, toward this goal. Low-scoring instances are refined iteratively until the
adequacy threshold is met or max iterations are reached.

The outer RLHF loop presents each instance to a human evaluator in its full context and
goal frame, collecting a contextual fit signal — accept, reject, or refine with a hint.
Rejected or refined instances re-enter the RL loop; new instances are added to the
population rather than replacing the original, growing population breadth.

The output is a **Concept Population Report**: population breadth, grammatical frame
coverage, goal and context coverage, per-instance table, and adequacy score deltas from
round zero to final.

## Target Users & Interaction

The primary users are ML researchers, NLP engineers, and cognitive-AI practitioners who
need to inspect and improve the contextual grounding of a model's vocabulary without
retraining weights. No ML infrastructure is required beyond a watsonx.ai API key.

Users interact through Bob chat to run the full orchestrated workflow, or through a Python
CLI (`python -m src.main --term "fire" --seed-phrase "to fire someone" ...`) and Jupyter
notebook for reproducible experimentation. Output is the Concept Population Report with
before-and-after adequacy metrics showing where the population improved across successive
rounds.

## Why Creative & Unique

This is the first project to apply Barrett's constructionist theory — augmented with a
precise account of the tokenization and grammar problems — as a concrete engineering
framework for LLM vocabulary improvement. Concepts-as-populations, grammatical frames as
concept constructors, and functional adequacy as the scoring criterion move from cognitive
science into a data structure, a prompt design, and a feedback loop.

Bob is a runtime orchestrator — not only generating code but driving the concept-learning
loop through purpose-built skills. Auto-scoring RL and human RLHF are combined in a live,
inspectable loop where every instance, goal, grammatical frame, and context is visible and
adjustable.

## Closing

A ConceptPopulation is not a fine-tuned model — it is a structured, inspectable record of
what a concept means across the grammatical constructions, contexts, and goals that matter.
Long-term, this framework offers a path toward LLMs that reason with goal-indexed,
grammatically-grounded, context-sensitive populations rather than token averages — closer
to how human cognition actually works.
