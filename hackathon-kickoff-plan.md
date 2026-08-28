# Hackathon Kickoff Plan — Optimizing LLM Vocabulary via Concept-Learning Workflow

## Theoretical Foundation — Barrett's Constructionist Framework

This project is grounded in Lisa Feldman Barrett's **Theory of Constructed Emotion** (and
Conceptual Act Theory) as described in *How Emotions Are Made* (2017).

**Key claims that drive the design:**

1. **Concepts are not fixed definitions.** A concept (e.g. "anger", "red", "safety") is a
   *population* of variable instances — past experiences clustered by a shared *functional goal*,
   not by perceptual similarity.

2. **Concepts are predictive.** The brain uses concepts to *simulate* future sensory input before
   it arrives (predictive coding / allostasis). A concept is a forward model, not a lookup table.

3. **Concepts are context-dependent.** The "right" instance of a concept is determined by the
   current goal and context. There is no context-free correct definition — only contextually
   adequate predictions.

4. **Concepts are learned through experience and feedback.** The brain refines its conceptual
   population through prediction error — when a simulation fails, the population is updated.
   This is the biological analogue of the RL feedback loop we are building.

**Implication for LLMs:** Current LLMs store word statistics, not goal-indexed conceptual
populations. They cannot select the contextually adequate instance of a concept because they have
no notion of *goal* or *predictive adequacy*. This project demonstrates a workflow that introduces
exactly this missing layer.

---

## Top-Level Overview

**Problem:** LLMs produce inconsistent, context-blind outputs because their "vocabulary" is a
flat statistical distribution over tokens — not a structured population of goal-indexed conceptual
instances. They lack the meta-skill of concept formation as described by Barrett: the ability to
simulate the functionally adequate instance of a concept for a given context and goal.

**Solution:** A Bob-driven agent workflow that represents each concept as a *population of
contextual instances* (not a single definition), runs an RL inner loop that scores each instance
for *functional adequacy in a specific context*, collects human RLHF signals on *contextual fit*
(not abstract clarity), and produces a **Concept Population Report** showing population breadth,
goal-context coverage, and per-context best instances.

**Target users:** ML researchers, NLP engineers, and cognitive-AI practitioners who want to
inspect and improve the contextual grounding of a deployed LLM's conceptual vocabulary.

**Key technology:** Python + watsonx.ai LLM (IBM SDK) + Bob agent workflow + structured
population representation (JSON) + scoring heuristics (functional adequacy per context).

**Deliverables to submit:**
1. Video demo (≤3 min, publicly accessible URL)
2. Written problem-and-solution statement (≤500 words)
3. Written statement on Bob usage
4. GitHub repo with code, README, and Bob session screenshots

---

## Core Data Model (Barrett-aligned)

```
ConceptPopulation {
  term: str                          # e.g. "anger"
  instances: [
    {
      id: str,
      context: str,                  # e.g. "receiving unfair criticism at work"
      goal: str,                     # e.g. "restore social fairness"
      simulation: str,               # the predicted experience/response
      adequacy_score: float (0-10),  # how well this instance serves the goal in context
      human_signal: accept|reject|refine,
      hint: str|null,
      round: int
    }
  ]
  goal_coverage: [str]               # list of distinct goals covered
  context_coverage: [str]            # list of distinct contexts covered
  population_breadth: int            # count of distinct accepted instances
}
```

The RL loop **adds and refines instances** in this population. It does not replace one definition
with another — it grows and diversifies the population while improving per-instance adequacy.

---

## Sub-Tasks

---

### Sub-Task 1 — Repository Bootstrap & Project Skeleton

**Intent:** Create the GitHub repository with the IBM hackathon template structure, `.gitignore`,
`.bobignore`, and top-level folder layout so every subsequent sub-task has a home.

**Expected Outcomes:**
- Public GitHub repo exists with correct ignore files (no credentials ever committed).
- Directory structure: `src/`, `notebooks/`, `docs/`, `assets/screenshots/`, `README.md`.
- Local workspace mirrors the repo.

**Todo List:**
1. Create a new GitHub repo (use the IBM Hackathon template if available, else manual).
2. Add `.gitignore` (Python standard + `.env` + `__pycache__`).
3. Add `.bobignore` (exclude `.env`, any `*.key` files).
4. Create empty placeholder files: `README.md`, `src/.gitkeep`, `notebooks/.gitkeep`,
   `docs/.gitkeep`, `assets/screenshots/.gitkeep`.
5. Commit and push; verify repo is **public**.

**Relevant Context:** IBM security monitoring — never commit API keys. Use `.env` for all secrets.

**Status:** [x] done

---

### Sub-Task 2 — Problem & Solution Written Statement

**Intent:** Draft the ≤500-word written statement for judges, grounded in Barrett's framework.

**Expected Outcomes:**
- `docs/problem-solution-statement.md` committed to repo.
- Stays within 500 words.
- Introduces Barrett's concept-as-population model accessibly for non-cognitive-science judges.
- Explains why this is a novel lens on LLM vocabulary improvement.

**Todo List:**
1. Create `docs/problem-solution-statement.md`.
2. Write section: **Problem** — LLMs have token statistics, not goal-indexed conceptual
   populations; they cannot select the contextually adequate instance of a concept.
3. Write section: **Solution** — Bob workflow builds a ConceptPopulation per term through
   RL (functional adequacy scoring per context) + RLHF (contextual fit signal), producing
   a Concept Population Report.
4. Write section: **Target Users & Interaction** — ML researchers and cognitive-AI practitioners;
   interact via Bob chat + Python CLI/notebook; input is a term + a set of contexts/goals.
5. Write section: **Why Creative & Unique** — first project to apply Barrett's constructionist
   theory of concepts as a concrete engineering framework for LLM vocabulary improvement.
6. Word-count check; trim to ≤500 words.
7. Commit to repo.

**Status:** [x] done

---

### Sub-Task 3 — Bob Usage Written Statement

**Intent:** Draft the written statement detailing exactly how IBM Bob was used to build the project.

**Expected Outcomes:**
- `docs/bob-usage-statement.md` committed to repo.
- Specific, names actual files and skills — no vague statements.

**Todo List:**
1. Create `docs/bob-usage-statement.md`.
2. Document Bob Plan mode: kickoff plan, theoretical reframe, skill design.
3. Document Bob Agent mode: generating all `src/` Python files, notebook, README.
4. Document Bob as runtime orchestrator: the concept-learning loop is driven by Bob skills.
5. List all 6 skills by name and purpose.
6. Describe watsonx.ai integration via `watsonx-api-caller` skill.
7. Reference `assets/screenshots/` for session evidence.
8. Commit to repo.

**Status:** [x] done

---

### Sub-Task 4 — Core Python Workflow Implementation (Barrett-aligned)

**Intent:** Build the concept-learning simulation using the population-of-instances model.
This is the centerpiece of the demo.

**Expected Outcomes:**
- `src/concept_population.py` — ConceptPopulation dataclass and JSON serialization.
- `src/concept_loop.py` — main RL loop: generates contextual instances, scores for
  functional adequacy per context, grows the population.
- `src/judge.py` — LLM-based adequacy scorer (context + goal aware).
- `src/human_feedback.py` — CLI RLHF: presents instance in its context, collects
  contextual fit signal.
- `src/report.py` — generates the Concept Population Report.
- `notebooks/demo.ipynb` — end-to-end runnable notebook.
- All secrets loaded from `.env` via `python-dotenv`.

**Todo List:**
1. Create `src/concept_population.py`:
   - Define `ConceptInstance` dataclass: `id, context, goal, simulation, adequacy_score,
     human_signal, hint, round`.
   - Define `ConceptPopulation` dataclass: `term, instances, goal_coverage,
     context_coverage, population_breadth`.
   - Add `to_json()` / `from_json()` methods.

2. Create `src/concept_loop.py`:
   - Accept: seed term + list of (context, goal) pairs.
   - For each (context, goal) pair, call watsonx.ai to generate a candidate simulation
     (the predicted experience/response for that context toward that goal).
   - Score each candidate via `judge.py` for functional adequacy.
   - Run K refinement iterations on low-scoring instances.
   - Add accepted instances to the ConceptPopulation.

3. Create `src/judge.py`:
   - Build context-aware judge prompt:
     ```
     Given the concept "{term}", the context "{context}", and the goal "{goal}":
     Rate how functionally adequate this simulation is (0-10):
     "{simulation}"
     Reply with only a number.
     ```
   - Call watsonx.ai; parse numeric response.

4. Create `src/human_feedback.py`:
   - Display: term, context, goal, simulation, adequacy score.
   - Collect: accept / reject / refine + optional hint.
   - On refine/reject: re-enter loop with hint.

5. Create `src/report.py`:
   - Render ConceptPopulation as markdown report (see `concept-clarity-report` skill).
   - Show: population breadth, goal coverage, context coverage, per-instance table,
     score delta from round 0 to final.

6. Create `notebooks/demo.ipynb` wiring all modules with narrative markdown cells.
7. Create `requirements.txt` (ibm-watsonx-ai, python-dotenv, notebook, rich, dataclasses-json).
8. Create `.env.example`.
9. Smoke-test; commit all files.

**Relevant Context:** The RL loop does NOT replace instances — it grows and refines the population.
A concept is complete when it has adequate coverage across the supplied (context, goal) pairs.

**Status:** [x] done

---

### Sub-Task 5 — README & Repo Documentation

**Intent:** Write a professional README grounded in Barrett's framework that judges can follow.

**Expected Outcomes:**
- `README.md` with: project title, one-liner, Barrett framework summary, architecture flow,
  setup, run instructions, screenshot location.

**Todo List:**
1. Write header: project name, one-liner, hackathon badge.
2. Write **Theoretical Foundation** section: 3-sentence Barrett summary accessible to
   non-cognitive-science readers.
3. Write **Architecture** section with ASCII flow:
   `Term + Contexts/Goals → Bob Orchestrator → watsonx.ai Simulator → Adequacy Judge → RLHF → Concept Population Report`
4. Write **Setup** section: clone, install, `.env` setup.
5. Write **Run** section: CLI and notebook instructions.
6. Write **Bob Session Screenshots** section.
7. Write **Deliverables** links.
8. Commit.

**Status:** [ ] pending

---

### Sub-Task 6 — Bob Session Screenshots

**Intent:** Capture and commit required Bob task session summary screenshots.

**Expected Outcomes:**
- Screenshots in `assets/screenshots/` covering Plan and Agent sessions.

**Todo List:**
1. Capture screenshot of this planning session summary.
2. Capture screenshot of Agent coding session (Sub-Task 4).
3. Name: `<name>-plan-session.png`, `<name>-agent-session.png`.
4. Add `assets/screenshots/README.md` explaining each file.
5. Commit.

**Status:** [x] done

---

### Sub-Task 6 — Bob Session Screenshots

**Intent:** Capture and commit required Bob task session summary screenshots.

**Status:** [x] done — `assets/screenshots/README.md` created with instructions for 4 screenshots. Actual screenshot PNG files require manual capture by the team during the recording session.

---

### Sub-Task 7 — Video Demo Production

**Intent:** Record the ≤3-minute demo video meeting all judge requirements.

**Expected Outcomes:**
- Video ≤3 minutes, publicly accessible URL.
- Barrett framework introduced in first 30 seconds.
- Live demo of population-building loop on screen for ≥90 seconds.
- Bob UI visible; narration throughout.

**Todo List:**
1. Script:
   - 0:00–0:30 — Hook: "Your brain doesn't store definitions. It stores predictions.
     LLMs do the opposite — and that's the problem."
   - 0:30–0:45 — Barrett framework: concept-as-population, goal-indexed, context-dependent.
   - 0:45–2:30 — Live demo: run concept loop on term "anger" with 3 (context, goal) pairs;
     show instances being generated, scored, human feedback given, population report rendered.
   - 2:30–3:00 — Show Concept Population Report; Bob session summary; closing statement.
2. Record screen + audio.
3. Upload to YouTube (unlisted-but-public) or Loom.
4. Verify public access.
5. Add URL to `README.md` and `docs/video-url.md`.
6. Commit.

**Status:** [ ] pending

---

## Dependency Order

```
Sub-Task 1 (repo bootstrap)
    ├── Sub-Task 2 (problem statement)   ← parallel with 3 and 4
    ├── Sub-Task 3 (Bob usage statement) ← parallel with 2 and 4
    └── Sub-Task 4 (core Python workflow)
            └── Sub-Task 5 (README)
                    └── Sub-Task 6 (screenshots)
                            └── Sub-Task 7 (video)
```

---

## Non-Goals

- No actual neural network weight updates.
- No production deployment.
- No vector database — the ConceptPopulation lives as structured JSON + markdown report.
- No full implementation of Barrett's interoception/allostasis model — the goal-indexed
  contextual instance structure is the conceptual analogue, not a neuroscientific simulation.

---

## Key Constraints

- **Credentials:** Never commit API keys. `.env` + `.env.example` from day one.
- **Video:** Strictly ≤3 minutes.
- **Written statements:** ≤500 words for the problem-solution statement.
- **Bob evidence:** Session summary screenshots mandatory.
- **Barrett alignment:** Every skill and module must use the vocabulary of the framework —
  "instance", "population", "goal", "context", "functional adequacy", "simulation" —
  not generic "definition" / "clarity" language.
