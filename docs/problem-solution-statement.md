# Problem & Solution Statement

## The Problem

LLMs store token co-occurrence statistics, not concepts. The word *fire* becomes the
fragments `fi` + `##re` — and every use of that token, whether dismissing an employee,
discharging a weapon, or describing intensity, collapses into the same weight distribution.
The model has no mechanism to select the meaning that is right *here*, toward *this goal*.

Grounded in Lisa Feldman Barrett's constructionist theory, the correct framing is sharper:
an LLM lacks a *population* of goal-indexed, grammatically-grounded conceptual instances.
It cannot construct the contextually adequate simulation of a concept because it does not
represent concepts as populations at all.

## The Solution

We built that missing layer. A **Generator LLM** (Granite 13B on watsonx.ai) constructs a
*simulation* — a specific prediction of experience — for a given seed phrase, context, and
goal. A **Judge LLM** (Granite 3B, fine-tuned) scores its *functional adequacy* from 0–10.
Low scores trigger refinement. Human RLHF provides a final contextual fit signal. Every
accepted instance joins a growing **ConceptPopulation** — never replacing earlier instances,
always expanding coverage across contexts, goals, and grammatical frames.

The output is a **Concept Population Report**: breadth, goal coverage, frame coverage, and
per-instance adequacy deltas from round zero to final. No weight retraining required.
Orchestrated end-to-end by IBM Bob.

## Why It Matters

This is the first project to apply Barrett's concept-as-population model as a concrete
engineering framework — turning cognitive science into a data structure, a prompt design,
and a two-LLM feedback loop. The result is an inspectable, auditable vocabulary layer that
moves AI one step closer to how human cognition actually works.
