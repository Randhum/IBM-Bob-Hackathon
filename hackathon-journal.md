# Hackathon Journal
## Project: Optimizing LLM Vocabulary via Concept-Population Workflow

*Running log of decisions, insights, and progress. Most recent entry first.*

---

## Entry 5 — Construction-Domain Training: Two-Track LoRA Fine-Tuning

**Trigger:** Strategic pivot — move from a pure runtime demonstration (concept population
built at inference time) to a full training demonstration: use the Barrett-structured
self-generated corpus as *labelled training data* to LoRA fine-tune Granite 3B on
watsonx.ai Tuning Studio, then compare trained vs base models on construction-domain
concept tasks.

**The core insight that unlocked the pivot:**

The same 50 `(context, goal, simulation, adequacy_score)` tuples we generate to build a
ConceptPopulation can be reformatted into *two distinct training signals* without any
additional generation:

- **Format A — Judge training:** prompt = full instance, completion = adequacy score string.
  Model learns to score functional adequacy in the construction domain.
- **Format B — Generator training:** prompt = context + goal, completion = high-quality
  simulation text (score ≥ 8.0 instances only). Model learns to construct goal-indexed
  concept simulations for construction-domain terms.

This means the self-generation step is not just a demo — it is a *data pipeline*. Barrett's
concept-as-population framework becomes the data generation strategy, not just the framing.

**Architecture decision: two LoRA jobs, same base model:**

Both jobs target `ibm/granite-3b-code-instruct` on watsonx.ai Tuning Studio (cloud infra —
no local GPU). Both run from the same base model with the same hyperparameters (5 epochs,
batch 8, lr 2e-4). The only difference is the training file and the learned behaviour.
Both jobs can run in parallel overnight.

**Evaluation design: three-way comparison:**

The benchmark notebook (`notebooks/construction_benchmark.ipynb`) compares three models:
1. **Base** — `ibm/granite-3b-code-instruct` cold, no domain training
2. **Generator-tuned** — trained on Format B; should produce better goal-anchored simulations
3. **Judge-tuned** — trained on Format A; should score construction-domain adequacy more accurately

Cross-product scoring (all three models generate, both base+tuned judge evaluate) produces
two independent deltas per term — making Barrett's claim empirically testable from two angles.

**Construction domain rationale:**

Narrow enough (10 terms) that 50 labelled examples produce a visible domain shift. Specific
enough (scaffolding, load-bearing, site induction, liability, etc.) that the base model's
generic priors are a meaningful and detectable comparison baseline.

**All compute runs on IBM Cloud:**

- Self-generation: watsonx.ai inference API (Granite 13B)
- LoRA training: watsonx.ai Tuning Studio (both jobs)
- Evaluation: watsonx.ai inference API (three model endpoints)
- Local machine: only lightweight Python scripts and the Jupyter notebook

**Files introduced by this entry:**
- `construction-domain-training-plan.md` — full 6-sub-task plan
- `data/corpus_spec.json` — human-authored seed (Sub-Task 1)
- `src/generate_corpus.py` — batch generation script (Sub-Task 2)
- `src/export_training_data.py` — dual-format JSONL exporter (Sub-Task 3)
- `src/launch_tuning_job.py` — SDK script for both LoRA jobs (Sub-Task 4)
- `notebooks/construction_benchmark.ipynb` — three-way evaluation notebook (Sub-Task 5)
- `docs/construction-training-report.md` — training report template (Sub-Task 6)

**Standing decisions updated:**

| Decision | Rationale |
|---|---|
| Generator is also fine-tuned, not just the judge | Same corpus, Format B export — no extra generation cost |
| Score threshold for generator training data: ≥ 8.0 | Model must learn from high-quality targets only; low-scoring simulations would teach the wrong pattern |
| Held-out eval: one instance per term, highest-scoring | Clean per-term delta; ensures eval set is not contaminated by training data |

---

## Entry 4 — The Sub-Lexical Layer: Morphemes and Phonesthetics

**Trigger:** Team question: should we add a letter/phoneme level below the word, to be fully
respectful of how concepts are constructed from language?

**The answer — precise and grounded:**

Letters themselves carry no conceptual content (`"f"`, `"i"`, `"r"`, `"e"` → no concept).
But the question revealed a genuine gap: between the *arbitrary* sub-word BPE token level
(which LLMs use) and the *meaningful* sub-word morpheme level (which linguistics tells us
matters for concept construction).

**Two distinct layers introduced:**

1. **Level 2 — Morphemes** (structural, optional field `morphemes: List[str]`)
   - The right sub-word unit: bounded, semantically meaningful.
   - `"mis-trust"` → `["mis-", "trust"]`; `"un-employment"` → `["un-", "employ", "-ment"]`
   - The `"mis-"` prefix shifts the concept; the `"-ment"` suffix nominalizes it.
   - This is exactly the level BPE tokenization destroys — and the level we now surface.
   - **NOT BPE tokens.** The distinction is enforced in code, docstrings, and the ontology.

2. **Level 0 annotation — Phonesthetics** (optional free-text `phonesthetics_note: str`)
   - Sound symbolism: `"sl-"` words cluster around smooth/unpleasant movement.
   - Injected as a **soft hint** only — does not affect scoring logic.
   - Optional and free-text: the system degrades gracefully when absent.
   - Makes the demo richer and the theoretical story more complete.

**Full five-level granularity model now locked:**
```
Level 0  Letter/phoneme      substrate; phonesthetics note is optional signal
Level 1  BPE token           REJECTED — arbitrary, destroys meaning
Level 2  Morpheme            optional annotation; bounded sub-lexical meaning
Level 3  Word                polysemous seed
Level 4  Seed phrase         grammatically framed; minimum required input
Level 5  Instance            context + goal + simulation; primary output
```

**What changed:**

- `ConceptInstance` gained:
  - `morphemes: List[str]` (default `[]`) — Level 2
  - `phonesthetics_note: str` (default `""`) — Level 0 annotation
- All prompt builders (`_build_generation_prompt`, `score_instance`, `refine_simulation`)
  inject both fields when present, skip them cleanly when absent.
- `main.py` CLI gained `--morphemes` (comma-separated) and `--phonesthetics-note`.
- `docs/concept-ontology.md` updated with:
  - §3.4 revised to full 5-level table
  - §3.5 new section: Phonesthetics — Sound Symbolism as Optional Annotation
  - §4 data model annotated with new fields
  - §5 vocabulary table gains `morpheme` and `phonesthetics note` rows
  - §7 new: Five-Level Summary table for video narration
  - North star sentence updated: "morpheme-aware words (not arbitrary tokens)"

**Design decision logged:** Morphemes are optional enrichments, not required fields.
The system operates at Level 4 → Level 5 by default. Levels 2 and 0 enrich the LLM's
construction when provided — they do not block construction when absent.

**Files changed:**
- `src/concept_population.py`
- `src/concept_loop.py`
- `src/judge.py`
- `src/concept_refiner.py`
- `src/main.py`
- `docs/concept-ontology.md`

---

## Entry 3 — The Tokenization Layer

**Trigger:** Team member raised a critical architectural gap: the system was treating concept
construction as a word-level problem, but LLMs operate at the *token* level — and the same
token can participate in entirely different concept constructions depending on grammatical role.

**The insight, precisely stated:**

LLMs encode `"fire"` → `["fi", "##re"]` and learn weights over all co-occurrences of that
token cluster, regardless of whether `"fire"` is dismissing an employee, discharging a weapon,
a combustion event, or a metaphor for passion. The model's "knowledge" of `"fire"` is the
statistical average of all these — a concept of nothing in particular.

Barrett's framework demands that we operate above this level: concepts are constructed from
**words + grammar**, not tokens. The grammatical frame (subject, verb, object, tense, modality,
aspect) is the disambiguation machinery. Without it, we're just building a slightly smarter
lookup table — not a concept-construction engine.

**What changed:**

1. `docs/concept-ontology.md` created — the team's shared theoretical reference document.
   Defines all canonical vocabulary. Introduces the 4 granularity levels:
   - Level 1: Token — `"ang"`, `"##er"` — NO concept
   - Level 2: Word — `"anger"` — POTENTIAL seed (polysemous)
   - Level 3: Phrase — `"anger at injustice"` — CONCEPT FRAME (grammatically grounded)
   - Level 4: Instance — phrase + context + goal — CONCEPT INSTANCE (fully specified)

2. `ConceptInstance` gained two new fields:
   - `seed_phrase: str` — grammatically framed form; minimum concept construction unit
   - `grammatical_frame: str` — syntactic role; prevents tokenization collapse

3. `ConceptPopulation` gained:
   - `seed_phrase: str` — canonical framed form for this population
   - `grammatical_frames: List[str]` — all distinct constructions in the population
   - `add_instance()` updated to track `grammatical_frames` coverage

4. All prompts (generation, scoring, refinement) now include `seed_phrase` and
   `grammatical_frame` so the LLM never constructs a simulation for a bare token.

5. `main.py` CLI gained `--seed-phrase` and `--grammatical-frame` arguments.

6. `report.py` now shows `Grammatical Frame Coverage` in the Population Summary table
   and the Instance Population Table includes a `Gram. Frame` column.

**Files changed:**
- `src/concept_population.py`
- `src/concept_loop.py`
- `src/judge.py`
- `src/concept_refiner.py`
- `src/report.py`
- `src/main.py`
- `.bob/skills/rl-feedback-loop/SKILL.md`
- `hackathon-kickoff-plan.md` (data model section updated)

**New files:**
- `docs/concept-ontology.md`

**Design decision logged:** The workflow explicitly rejects bare token input. `--seed-phrase`
is optional in the CLI (defaults to `""`) to maintain backwards compatibility, but all prompt
builders check for it and inject it when present. The ontology document enforces this as a
team convention: "Never accept a bare token. The seed phrase is the minimum concept
construction unit."

---

## Entry 2 — The Barrett Reframe

**Trigger:** After initial planning, the team grounded the project in Lisa Feldman Barrett's
*Theory of Constructed Emotion* ("How Emotions Are Made", 2017). This triggered a fundamental
redesign of the core model.

**What the old model said:**
- Input: one term
- Output: N definitions scored for "clarity"
- Best definition wins

**What Barrett's model demands:**
- Input: term + list of (context, goal) pairs
- Output: a *population* of contextual instances — not definitions, *simulations*
- Scoring: *functional adequacy* in each context toward each goal — not abstract clarity
- Human signal: *contextual fit* — not "is this definition good?"

**Why this matters for LLMs:** LLMs store word statistics. Barrett says concepts are
goal-indexed populations of past experience used to predict future input. The gap between
these is the exact problem this project addresses.

**What changed:**

1. `ConceptPopulation` data model introduced — term, instances, goal_coverage,
   context_coverage, population_breadth.
2. `ConceptInstance` introduced — context, goal, simulation, adequacy_score, human_signal.
3. All prompts rewritten around "simulation" (not "definition"), "functional adequacy"
   (not "clarity"), "contextual fit" (not "correctness").
4. All 4 affected skills rewritten:
   - `rl-feedback-loop` — now generates instances per (context, goal) pair, not N definitions
   - `concept-definition-refiner` — now `concept-simulation-refiner`; context + goal required
   - `rlhf-human-feedback` — presents instance with full context + goal; judges contextual fit
   - `concept-clarity-report` — renamed Concept Population Report; shows breadth + coverage
5. `hackathon-kickoff-plan.md` — problem statement, solution, and all sub-tasks rewritten
   around Barrett vocabulary.

**Canonical vocabulary established (full list in `docs/concept-ontology.md` §5):**
- simulation ≠ definition
- instance ≠ example
- functional adequacy ≠ correctness or clarity
- concept construction ≠ concept retrieval
- prediction error ≠ mistake

**Key design rule:** no concept instance exists without a context and a goal. Presenting
an instance to the human without its context frame would contradict Barrett's entire claim.

---

## Entry 1 — Kickoff and Initial Plan

**Date:** Hackathon day 1.

**Problem framing:** LLMs lack the meta-skill of concept formation. Their vocabulary is a
flat statistical distribution over tokens — not a structured population of goal-indexed
instances. The project demonstrates a workflow that introduces exactly this missing layer.

**Initial architecture decided:**
- Inner RL loop: LLM auto-scores candidate simulations → refinement on low scores
- Outer RLHF loop: human thumbs-up/down + correction hint
- Output: Concept Population Report with before/after metrics

**Deliverables confirmed:**
1. Video demo ≤3 min (publicly accessible URL)
2. Written problem-solution statement ≤500 words
3. Written Bob usage statement
4. GitHub repo with code + Bob session screenshots

**IBM Bob usage decided:**
- Plan mode for kickoff planning and skill design
- Agent mode for all code generation
- Bob as runtime orchestrator: the concept-learning loop itself is driven by Bob skills

**6 skills created:**
1. `watsonx-api-caller` — standardized LLM call entry point; stub mode for demos
2. `rl-feedback-loop` — inner RL auto-scoring loop
3. `concept-definition-refiner` — single-step simulation mutation
4. `rlhf-human-feedback` — outer RLHF human signal collection
5. `concept-clarity-report` — Concept Population Report generator
6. `hackathon-deliverable-writer` — judge-facing written statement drafter

**Sub-tasks 1–4 and 6 completed** (per external plan file updates):
- [x] Repository bootstrap
- [x] Problem-solution statement
- [x] Bob usage statement
- [x] Core Python workflow (concept_loop, judge, human_feedback, report, main)
- [x] Bob session screenshots

**Remaining:**
- [ ] Sub-Task 5: README
- [ ] Sub-Task 7: Video demo

---

## Standing Decisions (immutable unless journal records a change)

| Decision | Rationale |
|---|---|
| No neural weight updates | Out of scope; project is a prompt-level RL simulation |
| No vector database | ConceptPopulation lives as structured JSON + markdown report |
| No bare token input | Seed phrase required; tokens cannot construct concepts |
| Grammar is an input, not inferred | System requires grammatical frame; doesn't auto-detect it |
| `WATSONX_STUB=true` for rehearsal | API quota; allows full workflow demo without live calls |
| Barrett vocabulary enforced | All files, skills, prompts, narration use ontology §5 vocabulary |

---

## Open Questions

| Question | Status |
|---|---|
| watsonx.ai account / API key | Not yet obtained; stub mode covers demo until then |
| Demo seed term for video | Candidate: "fire" (multiple frames, dramatic contrast); "anger" (emotionally resonant) |
| Video platform | YouTube unlisted vs Loom — TBD |
| README (Sub-Task 5) | Pending — can be written directly from concept-ontology.md |

---

*This journal is append-only. Add new entries at the top under a new ## Entry N heading.*
