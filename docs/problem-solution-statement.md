# Problem & Solution Statement

## The Problem

Large language models are trained on token co-occurrence statistics. They do not maintain
goal-indexed conceptual populations — they have no internal representation that says "this
instance of *safety* is adequate for this context and this goal, but not for that one."
When asked to reason about a concept, an LLM draws on average token distributions rather
than selecting the contextually adequate simulation. The result is familiar: hallucinated
instances that ignore context, inconsistent vocabulary across similar prompts, and outputs
that are statistically plausible yet functionally inadequate. The model does not know which
instance of a concept to activate — because it does not represent concepts as populations at all.

## The Solution

Grounded in Lisa Feldman Barrett's constructionist theory of concepts, this project
represents each concept as a *population of contextual instances*, not a single definition.
Each instance is anchored to a specific context and goal, evaluated by how well its
simulation serves that goal — its *functional adequacy*.

A Bob-orchestrated workflow builds this population in two nested loops. The inner RL loop
calls watsonx.ai to generate candidate simulations for each (context, goal) pair, then
scores each via a context-aware judge prompt. Low-scoring instances are refined iteratively
until adequacy thresholds are met. The outer RLHF loop presents each instance to a human
evaluator in its full context and goal frame, collecting a contextual fit signal — accept,
reject, or refine with a hint — before the instance is committed to the population.

The output is a **Concept Population Report**: population breadth, goal-context coverage,
per-instance table, and adequacy score deltas from round zero to final.

## Target Users & Interaction

The primary users are ML researchers, NLP engineers, and LLM evaluators who need to
inspect and improve the contextual grounding of a model's vocabulary without retraining
weights. No ML infrastructure is required beyond a watsonx.ai API key.

Users interact through Bob chat to run the full orchestrated workflow, or through a Python
CLI and Jupyter notebook for reproducible experimentation. Input is a term plus a set of
(context, goal) pairs. Output is the Concept Population Report with before-and-after
adequacy metrics showing where the population improved across successive rounds.

## Why Creative & Unique

This is the first project to apply Barrett's constructionist theory as a concrete
engineering framework for LLM vocabulary improvement. Concepts-as-populations moves from
cognitive science into a data structure, a scoring criterion, and a feedback loop.

Bob is a runtime orchestrator — not only generating code but driving the concept-learning
loop through purpose-built skills. Auto-scoring RL and human RLHF are combined in a live,
inspectable loop where every instance, goal, and context is visible and adjustable.

## Closing

A ConceptPopulation is not a fine-tuned model — it is a structured, inspectable record of
what a concept means across the contexts that matter. Long-term, this framework offers a
path toward LLMs that reason with goal-indexed, context-sensitive populations rather than
token averages — closer to how human cognition actually works.
