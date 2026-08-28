# Demo Video Script

**Project:** Optimizing LLM Vocabulary via Concept-Learning Workflow
**Target duration:** ≤ 3 minutes (180 seconds)
**Format:** Screen recording with live narration

---

## Pre-Recording Checklist

- [ ] Terminal open with `src/` directory in view
- [ ] `notebooks/demo.ipynb` open in Jupyter and pre-run through Cell 2
- [ ] Bob session open in a browser tab showing the `rl-feedback-loop` skill ready
- [ ] `.env` file closed and not visible in any open editor tab
- [ ] Microphone tested; background noise minimised
- [ ] Screen resolution set for legibility (1280×720 minimum)

---

## Segment 1 — Hook & Problem (0:00–0:30)

**Duration:** 30 seconds
**Screen:** Title slide or blank Bob chat window (no terminal yet)

### Narration

> "Your brain doesn't store definitions. It stores predictions. When you think of 'anger',
> you don't recall a dictionary entry — your brain simulates what anger looks, feels, and
> sounds like in *this specific situation*, toward *this specific goal*.
>
> Current LLMs do the opposite: they store token statistics. That's the problem we're
> solving."

### Speaker Notes

- Speak slowly for the first two sentences — this is the conceptual hook for judges.
- Pause briefly after "toward *this specific goal*" before transitioning to "Current LLMs…"
- Keep the screen clean here; no need to show code yet.

---

## Segment 2 — Barrett Framework (0:30–0:45)

**Duration:** 15 seconds
**Screen:** Bob chat showing the kickoff plan or a prepared slide with the ConceptPopulation
schema visible

### Narration

> "Lisa Feldman Barrett's constructionist theory tells us a concept is not a fixed
> definition — it's a *population of variable instances*, each grounded in a context and
> a goal. We built that missing layer on top of an LLM."

### Speaker Notes

- Emphasise "population of variable instances" — this is the core theoretical claim.
- The screen can show the `ConceptPopulation` JSON schema from `src/concept_population.py`
  or the plan file — either reinforces the data-model vocabulary.
- This segment is intentionally short; transition quickly into the demo.

---

## Segment 3 — Live Demo (0:45–2:30)

**Duration:** 105 seconds
**Screen:** Terminal / Jupyter Notebook / Bob UI — switch between as narrated below

---

### 3a — Setup (0:45–1:00)

**Screen:** Terminal or Notebook Cell 3 showing the seed term and context-goal pairs

#### Narration

> "We start with a term: 'anger'. We give it three context-goal pairs — for example:
> receiving unfair criticism at work with the goal of restoring social fairness; feeling
> ignored during a conflict with the goal of being heard; and witnessing injustice with
> the goal of protecting others."

#### Screen Action Cues

- Show `main.py` being invoked, or Notebook Cell 3 with the three `(context, goal)` pairs
  visible.
- Scroll slowly so the three pairs are all readable.

---

### 3b — Simulation Generation (1:00–1:20)

**Screen:** Terminal output showing watsonx.ai generating candidate simulations, or Bob
running the `rl-feedback-loop` skill

#### Narration

> "Bob's `rl-feedback-loop` skill generates an initial simulation for each pair — a
> predicted experience of anger that is specific to that context and that goal."

#### Screen Action Cues

- Show the skill activation in Bob, or the terminal output with three simulation strings
  appearing.
- If running from notebook, show Cell 4 output appearing line by line.

---

### 3c — Adequacy Scoring (1:20–1:40)

**Screen:** Terminal or notebook output showing numerical adequacy scores per instance

#### Narration

> "The judge LLM scores each simulation for functional adequacy in context. A score of
> eight or above means the simulation is predictively useful for that goal in that
> situation."

#### Screen Action Cues

- Highlight the score column in the output table (e.g. `adequacy_score: 6.5`).
- If one score is below threshold, pause on it briefly to set up the next segment.

---

### 3d — Refinement (1:40–2:00)

**Screen:** Terminal or Bob showing the `concept-definition-refiner` skill in action

#### Narration

> "Low-scoring instances are refined by the `concept-definition-refiner` skill — the LLM
> receives the original simulation plus its score and a refinement rationale, and produces
> an improved version."

#### Screen Action Cues

- Show the before/after simulation text side by side, or the refinement output with the
  rationale paragraph visible.
- Scroll to show the score improving from round 0 to round 1.

---

### 3e — Human RLHF (2:00–2:15)

**Screen:** CLI prompt from `human_feedback.py` or Bob RLHF display

#### Narration

> "After auto-scoring, we collect human feedback on contextual fit. The reviewer sees the
> full context and goal alongside the simulation — then accepts, rejects, or provides a
> refinement hint."

#### Screen Action Cues

- Show the RLHF prompt: term, context, goal, simulation, adequacy score, and the
  accept/reject/refine menu.
- Demonstrate selecting "accept" for one instance.

---

### 3f — Population Growth (2:15–2:30)

**Screen:** The growing ConceptPopulation JSON or the formatted population table

#### Narration

> "The population grows — we never replace instances, we add them. Each accepted instance
> expands the concept's coverage across contexts and goals, just as Barrett describes."

#### Screen Action Cues

- Show the `ConceptPopulation` JSON with two or three instances, each with a distinct
  `context` and `goal` field.
- Briefly show `population_breadth` incrementing.

---

## Segment 4 — Report & Closing (2:30–3:00)

**Duration:** 30 seconds
**Screen:** `report.py` output or `notebooks/demo.ipynb` final cell showing the Concept
Population Report markdown

### Narration

> "Finally, the Concept Population Report shows us population breadth, goal and context
> coverage, and per-instance adequacy improvements.
>
> This is Barrett's concept-as-population made operational — a proof-of-concept that LLM
> vocabulary can be improved without weight retraining, using Bob as the orchestrator."

### Screen Action Cues

- Show the full report: header (term, breadth, coverage counts), the per-instance table
  with round-0 vs final scores, and the verdict line.
- End on the Bob UI or repo README to close with the project name on screen.

### Speaker Notes

- The closing line "without weight retraining, using Bob as the orchestrator" is the key
  differentiator for judges — say it clearly and let it land before cutting.
- No need to rush; this segment has slack time.

---

## Timing Summary

| Segment | Time | Duration | Key Point |
|---------|------|----------|-----------|
| 1 — Hook & Problem | 0:00–0:30 | 30 sec | Brain predicts; LLMs store statistics |
| 2 — Barrett Framework | 0:30–0:45 | 15 sec | Concept = population of instances |
| 3a — Setup | 0:45–1:00 | 15 sec | Term + 3 context-goal pairs |
| 3b — Simulation | 1:00–1:20 | 20 sec | watsonx.ai generates instances |
| 3c — Scoring | 1:20–1:40 | 20 sec | Judge LLM rates adequacy |
| 3d — Refinement | 1:40–2:00 | 20 sec | Low scores → refiner skill |
| 3e — RLHF | 2:00–2:15 | 15 sec | Human accept/reject/refine |
| 3f — Population | 2:15–2:30 | 15 sec | Population grows, never replaces |
| 4 — Report & Close | 2:30–3:00 | 30 sec | Concept Population Report + wrap |

**Total: 3:00 exactly.** Cut 5–10 seconds from Segment 3b or 3c if the live demo runs long.

---

## Post-Recording

1. Trim the recording to ≤ 3:00.
2. Upload to YouTube (unlisted) or Loom; verify public access.
3. Add the URL to `docs/video-url.md` and `README.md`.
4. Commit both files.
