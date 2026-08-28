# Hackathon Journal
## Project: Optimizing LLM Vocabulary via Concept-Population Workflow

*Running log of decisions, insights, and progress. Most recent entry first.*

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
