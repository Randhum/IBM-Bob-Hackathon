# Bob Usage Statement

**Project:** Optimizing LLM Vocabulary via Concept-Learning Workflow (Barrett-Aligned)
**Hackathon:** IBM TechXchange Hackathon — Build with Bob

---

## 1. Planning with Bob (Plan Mode)

The entire project was architected in a single Bob **Plan mode** session. Before a single line of
code was written, the team opened Bob, switched to Plan mode, and ran the kickoff session that
produced `hackathon-kickoff-plan.md`.

In that session, Bob was used to:

- **Reframe the problem theoretically.** The Barrett constructionist model (concepts as
  goal-indexed populations of contextual instances, not fixed definitions) was identified as the
  correct theoretical lens for diagnosing why LLMs fail at contextual concept use. Bob helped
  articulate the four key Barrett claims (concepts are not fixed; concepts are predictive;
  concepts are context-dependent; concepts are learned through prediction error) and mapped each
  one to a concrete engineering decision in the workflow.

- **Identify the tokenization and grammar layer.** A second planning pass extended the Barrett
  framing to include the tokenization problem (LLMs operate on sub-word fragments, not words) and
  the grammar problem (the same word in different grammatical constructions builds different
  concepts). This produced the `seed_phrase` / `grammatical_frame` fields and the
  `docs/concept-ontology.md` vocabulary specification. Bob drafted all four sections of that
  document.

- **Break the project into 7 sub-tasks.** Bob decomposed the full submission into sequenced,
  dependency-ordered sub-tasks: repository bootstrap, problem statement, Bob usage statement,
  core Python workflow, README, screenshots, and video. The dependency graph (Sub-Task 1 →
  Sub-Tasks 2/3/4 in parallel → Sub-Task 5 → 6 → 7) was produced in this session.

- **Define the core data model.** Bob produced the `ConceptPopulation` / `ConceptInstance`
  schema — including field names (`seed_phrase`, `grammatical_frame`, `context`, `goal`,
  `simulation`, `adequacy_score`, `human_signal`, `hint`, `round`) — directly in the plan
  session, before any Python file existed.

- **Design all 6 skills.** Bob identified which skills were needed, what each must do, and which
  Bob mode (Plan vs Agent) would be used to build them.

The full output of this session is committed as `hackathon-kickoff-plan.md` in the repo root.

---

## 2. Skill Creation with Bob

All 6 project skills were designed in Bob Plan mode and scaffolded in Bob Agent mode. Each skill
lives under `.bob/skills/<skill-name>/SKILL.md`.

| Skill | File | Purpose |
|---|---|---|
| `watsonx-api-caller` | `.bob/skills/watsonx-api-caller/SKILL.md` | Standardizes every IBM watsonx.ai LLM call: prompt construction, API authentication via `.env`, parameter defaults (`model_id`, `max_new_tokens`, `temperature`), response parsing, and error handling. Single entry point for all watsonx.ai calls in the project. |
| `rl-feedback-loop` | `.bob/skills/rl-feedback-loop/SKILL.md` | Inner RL loop that grows the `ConceptPopulation`. Iteratively generates contextual concept instances using a judge LLM, scores each for functional adequacy per (context, goal) pair, and runs until an adequacy threshold is reached or max iterations are exhausted. |
| `rlhf-human-feedback` | `.bob/skills/rlhf-human-feedback/SKILL.md` | Outer RLHF loop for human contextual fit feedback. Presents each instance in its full context/goal/grammatical-frame frame, collects accept / reject / refine signal from the human, and routes the result: accepted instances are committed to the population; rejected or refined instances re-enter the RL loop with the user's hint, adding new instances to the population. |
| `concept-definition-refiner` | `.bob/skills/concept-definition-refiner/SKILL.md` | Single-step simulation refinement. Takes a term, seed phrase, grammatical frame, context, goal, current simulation, and an optional hint, calls watsonx.ai to produce an improved simulation, and returns the result with a rationale. Used inside the RL loop when an instance scores below threshold. |
| `concept-clarity-report` | `.bob/skills/concept-clarity-report/SKILL.md` | Renders the Concept Population Report from a completed `ConceptPopulation`. Formats population breadth, grammatical frame coverage, goal-context coverage, per-instance table, adequacy score deltas (round 0 → final), and a verdict on population completeness. |
| `hackathon-deliverable-writer` | `.bob/skills/hackathon-deliverable-writer/SKILL.md` | Drafts judge-facing written deliverables: the problem-solution statement (`docs/problem-solution-statement.md`, ≤500 words) and this Bob usage statement (`docs/bob-usage-statement.md`). Enforces section order, word counts, and specificity requirements. |

Skills were activated during both development (for Bob to follow when writing code) and at runtime
(for Bob to drive the live concept-learning session with a user).

---

## 3. Code Generation with Bob (Agent Mode)

All core Python source files were generated by Bob in **Agent mode**, guided by the plan and the
data model defined in the planning session.

| File | What Bob Generated |
|---|---|
| `src/concept_population.py` | `ConceptInstance` and `ConceptPopulation` dataclasses with all Barrett-aligned and tokenization-layer fields (`seed_phrase`, `grammatical_frame`, `grammatical_frames`); `to_json()` / `from_json()` serialization; computed properties `initial_score` and `score_delta` for delta reporting. |
| `src/watsonx_client.py` | IBM watsonx.ai API client wrapping `ibm_watsonx_ai`; loads `WATSONX_API_KEY`, `WATSONX_PROJECT_ID`, and `WATSONX_URL` from `.env`; supports `WATSONX_STUB=true` for dry-run demos. |
| `src/concept_loop.py` | Main RL loop: accepts a seed term, `seed_phrase`, `grammatical_frame`, and list of (context, goal) pairs; injects grammatical fields into all generation and scoring prompts; scores each via `judge.py`; runs K refinement iterations on low-scoring instances; accumulates accepted instances into the `ConceptPopulation`. |
| `src/judge.py` | LLM-based adequacy scorer; constructs context-aware judge prompt including `seed_phrase` and `grammatical_frame` so the judge evaluates the simulation against the specific grammatical construction; calls watsonx.ai; parses numeric response (0–10). |
| `src/concept_refiner.py` | Single-step refinement; injects `seed_phrase` and `grammatical_frame` to keep the refiner anchored to the correct grammatical construction; validates for empty output, identical-to-original, and word-count overflow. |
| `src/human_feedback.py` | CLI RLHF collector; displays term, seed phrase, context, goal, simulation, and adequacy score; collects accept / reject / refine signal + optional hint; on reject/refine re-runs the RL loop with the stored hint and appends new instances to the population. |
| `src/report.py` | Concept Population Report generator; renders markdown with population breadth, grammatical frame coverage, goal coverage, context coverage, per-instance table with frame column, and score delta from round 0 to final round. |
| `src/main.py` | CLI entry point; exposes `--term`, `--seed-phrase`, `--grammatical-frame`, `--context`, `--goal`, `--contexts-file`, `--max-iterations`, `--threshold`, `--no-human`, `--hint`; orchestrates RL loop → RLHF → JSON save → report generation. |
| `notebooks/demo.ipynb` | End-to-end demo notebook wiring all modules with narrative markdown cells; runs the full loop on the concept `"fire"` (dismissal frame) with three (context, goal) pairs as a live demo. |
| `docs/concept-ontology.md` | Shared vocabulary specification: defines concept, instance, simulation, context, goal, functional adequacy, prediction error, construction, and the tokenization/grammar layer in precise terms. Every code file and deliverable is traceable to this document. |

The workflow was **iterative**: Bob generated a first draft of each file, the team reviewed it for
Barrett-vocabulary alignment, grammatical-frame correctness, and data-model integrity, and Bob
applied targeted edits in response to feedback. No file was accepted in its first draft without at
least one review-and-revise cycle. Bob's `apply_diff` and `search_and_replace` tools were used for
surgical edits rather than wholesale rewrites, keeping each change traceable to a specific review
comment.

---

## 4. Bob as Runtime Orchestrator

Bob is not merely the tool that *wrote* the code — Bob *runs* the workflow at runtime.

When a user wants to build a Concept Population for a new term, the session proceeds entirely
inside Bob:

1. The user activates the **`rl-feedback-loop`** skill. Bob reads the skill instructions and
   orchestrates generation + scoring: calling `src/concept_loop.py` logic via the
   `watsonx-api-caller` skill, presenting scored instances to the user, and growing the population.
   All prompts include the `seed_phrase` and `grammatical_frame` to anchor the LLM to the correct
   concept construction.

2. At each checkpoint, the **`rlhf-human-feedback`** skill takes over. Bob presents the instance
   in its full context/goal/grammatical-frame, prompts the user for an accept / reject / refine
   signal, and routes the result: accepted instances are committed to the population; rejected or
   refined instances re-enter the RL loop with the user's hint fed to the
   `concept-definition-refiner` skill. New instances are appended to the population — the
   population grows through each feedback round.

3. When the population reaches the adequacy threshold (or the user ends the session), the
   **`concept-clarity-report`** skill renders the final Concept Population Report as a formatted
   markdown document, including breadth score, grammatical frame coverage, goal and context tables,
   and per-instance adequacy deltas.

This design means the **concept-learning loop is a Bob skill pipeline**, not a standalone Python
script. Bob holds the session state, drives the interaction, and decides when to escalate from
auto-scoring to human feedback and back.

---

## 5. watsonx.ai Integration

All LLM calls in the project go through IBM watsonx.ai using the `ibm_watsonx_ai` Python SDK.

- **`watsonx-api-caller` skill** — activated whenever any code or skill needs to call watsonx.ai.
  It defines the standard prompt structure, required parameters (`model_id`, `project_id`,
  `max_new_tokens`, `temperature`, `decoding_method`), response parsing logic, and the error
  handling pattern. Every module (`src/concept_loop.py`, `src/judge.py`, `src/concept_refiner.py`)
  follows this single pattern.

- **Credentials management** — `WATSONX_API_KEY`, `WATSONX_PROJECT_ID`, and `WATSONX_URL` are
  stored exclusively in `.env` (never committed; excluded by `.gitignore` and `.bobignore`).
  An `.env.example` file with placeholder values is committed to the repo for setup guidance.

- **`WATSONX_STUB=true`** — setting this environment variable switches `src/watsonx_client.py`
  into dry-run mode, returning deterministic stub responses. This allows end-to-end demo runs
  without consuming API quota or requiring live credentials — used for initial testing.

---

## 6. Session Evidence

Bob session summary screenshots are committed to `assets/screenshots/`. These capture the
Plan mode kickoff session (showing the sub-task breakdown and Barrett theoretical framing) and
the Agent mode coding sessions (showing file generation and iterative code review).

See `assets/screenshots/README.md` for a description of each screenshot file.

---

*This document was drafted and updated using the `hackathon-deliverable-writer` Bob skill, following
the section schema and specificity requirements defined in
`.bob/skills/hackathon-deliverable-writer/SKILL.md`.*
