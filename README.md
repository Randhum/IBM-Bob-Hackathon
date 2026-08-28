# LLM Vocabulary via Barrett's Concept-Population Model

A Bob-orchestrated RL+RLHF workflow that builds goal-indexed, context-sensitive concept populations for LLMs — grounded in Lisa Feldman Barrett's constructionist theory of concepts.

> IBM TechXchange Hackathon 2024 — Bob + watsonx.ai

---

## Theoretical Foundation

Lisa Feldman Barrett's constructionist model holds that a concept is not a fixed definition but a *population of variable instances*, each anchored to a specific context and a specific goal — the brain predicts forward by selecting the instance whose simulation is functionally adequate for the situation at hand. Current LLMs store token co-occurrence statistics, not goal-indexed conceptual populations; they have no mechanism to select the contextually adequate instance, producing outputs that are statistically plausible but functionally misaligned. This project introduces the missing layer: a Bob-orchestrated RL+RLHF workflow that builds and evaluates a `ConceptPopulation` per term, scoring each instance for *functional adequacy* in its (context, goal) frame and refining low-scoring instances until the population reaches sufficient coverage.

---

## Architecture

```
Term + (Context, Goal) Pairs
        │
        ▼
Bob Orchestrator (rl-feedback-loop skill)
        │
        ▼
watsonx.ai Simulator ──► Adequacy Judge (judge.py)
        │
        ▼ [if score < threshold]
Concept Refiner (concept_refiner.py)
        │
        ▼
RLHF Human Feedback (human_feedback.py)
        │
        ▼
Concept Population Report (report.py)
        │
        ▼
docs/concept_population_report.md
```

---

## Project Structure

```
.
├── src/
│   ├── main.py                  # CLI entry point; orchestrates the full workflow
│   ├── concept_population.py    # ConceptInstance + ConceptPopulation dataclasses; JSON serialization
│   ├── concept_loop.py          # RL inner loop: generates, scores, and refines instances
│   ├── judge.py                 # LLM-based adequacy scorer (context + goal aware, 0–10)
│   ├── concept_refiner.py       # Single-step simulation refinement with optional hint
│   ├── human_feedback.py        # CLI RLHF: presents instances, collects accept/reject/refine signal
│   ├── report.py                # Renders Concept Population Report as markdown
│   └── watsonx_client.py        # IBM watsonx.ai API client; supports WATSONX_STUB=true dry-run
│
├── notebooks/
│   └── demo.ipynb               # End-to-end demo notebook (concept "anger", 3 context/goal pairs)
│
├── docs/
│   ├── problem-solution-statement.md   # ≤500-word judge-facing problem & solution statement
│   ├── bob-usage-statement.md          # Detailed statement on how Bob was used throughout
│   ├── concept_population_report.md    # Sample Concept Population Report output
│   └── anger_population.json           # Sample ConceptPopulation JSON (term: "anger")
│
├── assets/
│   └── screenshots/             # Bob session summary screenshots (Plan + Agent mode)
│
├── .bob/skills/
│   ├── watsonx-api-caller/      # Standardized watsonx.ai LLM call pattern
│   ├── rl-feedback-loop/        # Inner RL auto-scoring loop skill
│   ├── rlhf-human-feedback/     # Human contextual fit feedback skill
│   ├── concept-definition-refiner/   # Single-step simulation refinement skill
│   ├── concept-clarity-report/  # Concept Population Report renderer skill
│   └── hackathon-deliverable-writer/ # Drafts judge-facing written deliverables
│
├── requirements.txt             # Python dependencies
├── .env.example                 # Credential template (copy to .env; never commit .env)
└── hackathon-kickoff-plan.md    # Full project plan produced in Bob Plan mode session
```

---

## Setup

```bash
git clone <repo-url>
cd <repo>
pip install -r requirements.txt
cp .env.example .env
# Fill in WATSONX_API_KEY and WATSONX_PROJECT_ID in .env
```

The `.env` file is excluded from version control by `.gitignore` and `.bobignore`. See `.env.example` for all available configuration options including `WATSONX_MODEL_ID`, `WATSONX_URL`, and `LOG_LEVEL`.

---

## Quick Start (Stub / Dry-Run — no API keys required)

`WATSONX_STUB=true` switches `src/watsonx_client.py` into dry-run mode: generation calls return deterministic stub text and scoring calls return a random float in the 5.0–9.5 range. No API quota is consumed.

```bash
# Set WATSONX_STUB=true in .env for demo without API access
python -m src.main --term "anger" \
  --context "receiving unfair criticism at work" \
  --goal "restore social fairness" \
  --max-iterations 3 --threshold 7.5 --no-human
```

The report is written to `docs/concept_population_report.md` by default. Use `--output <path>` to override.

**Using a contexts file (multiple context/goal pairs):**

```bash
python -m src.main --term "safety" \
  --contexts-file contexts.json \
  --max-iterations 5 --threshold 8.0
```

Where `contexts.json` is a list of `{"context": "...", "goal": "..."}` objects.

---

## Notebook Demo

```bash
jupyter notebook notebooks/demo.ipynb
```

The notebook runs the full concept-learning loop on the term `"anger"` with three (context, goal) pairs, shows each instance being generated and scored, simulates human RLHF feedback, and renders the final Concept Population Report inline. All cells can be executed without API access when `WATSONX_STUB=true` is set in `.env`.

---

## Deliverables

| Deliverable | Path |
|---|---|
| Problem & Solution Statement | [docs/problem-solution-statement.md](docs/problem-solution-statement.md) |
| Bob Usage Statement | [docs/bob-usage-statement.md](docs/bob-usage-statement.md) |
| Bob Session Screenshots | [assets/screenshots/](assets/screenshots/) |

---

## Key Vocabulary (Barrett Alignment)

| Barrett Term | Project Equivalent |
|---|---|
| concept instance | `ConceptInstance` dataclass (`src/concept_population.py`) |
| population | `ConceptPopulation` dataclass; the full set of instances for a term |
| simulation | the generated text prediction for a given (context, goal) pair |
| functional adequacy | `adequacy_score` (float 0–10); how well the instance serves the goal in context |
| goal | the purpose the concept serves in context; drives adequacy scoring |

Every module, field name, prompt, and skill in this project uses Barrett vocabulary — "instance", "population", "simulation", "goal", "context", "functional adequacy" — rather than generic "definition" or "clarity" language.

---

Built with IBM Bob + watsonx.ai for IBM TechXchange Hackathon 2024
