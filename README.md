# LLM Vocabulary via Barrett's Concept-Population Model

A Bob-orchestrated RL+RLHF workflow that builds goal-indexed, context-sensitive concept populations for LLMs — grounded in Lisa Feldman Barrett's constructionist theory of concepts.

> IBM TechXchange Hackathon 2026 — Bob + watsonx.ai

---

## Theoretical Foundation

Lisa Feldman Barrett's constructionist model holds that a concept is not a fixed definition but a *population of variable instances*, each anchored to a specific context and a specific goal — the brain predicts forward by selecting the instance whose simulation is functionally adequate for the situation at hand. Current LLMs store token co-occurrence statistics, not goal-indexed conceptual populations; they have no mechanism to select the contextually adequate instance, producing outputs that are statistically plausible but functionally misaligned. This project introduces the missing layer: a Bob-orchestrated RL+RLHF workflow that builds and evaluates a `ConceptPopulation` per term, scoring each instance for *functional adequacy* in its (context, goal) frame and refining low-scoring instances until the population reaches sufficient coverage.

### Linguistic Granularity Model

The system operates across five levels of linguistic structure — explicitly rejecting the BPE token level that LLMs use, and instead grounding concept construction in morpheme-aware, grammatically-framed inputs:

| Level | Unit | LLM uses? | This system | Conceptual content |
|---|---|---|---|---|
| 0 | Letter / phoneme | as character tokens | `phonesthetics_note` (optional soft hint) | substrate only |
| 1 | BPE token | ✅ primary | ❌ rejected | none — arbitrary frequency split |
| 2 | Morpheme | ❌ no | `morphemes` field (optional) | bounded sub-lexical meaning |
| 3 | Word | implicit in tokens | `term` — polysemous seed | potential, ungrounded |
| 4 | Seed phrase | via attention | `seed_phrase` + `grammatical_frame` — **required** | grammatically grounded frame |
| 5 | Instance | never | `ConceptInstance` — primary output | fully specified concept |

The gap between row 1 and row 5 is the gap this project fills. See [`docs/concept-ontology.md`](docs/concept-ontology.md) for the full theoretical grounding.

---

## Architecture

```
Term + seed_phrase + grammatical_frame + [morphemes] + [phonesthetics_note]
     + (Context, Goal) Pairs
        │
        ▼
Bob Orchestrator (rl-feedback-loop skill)
        │
        ▼
watsonx.ai Simulator ──► Adequacy Judge (judge.py)
[Level 2–4 injected     [Level 2–4 injected into
 into all prompts]       scoring prompt]
        │
        ▼ [if adequacy_score < threshold]
Concept Refiner (concept_refiner.py)
        │
        ▼
RLHF Human Feedback (human_feedback.py)
[contextual fit signal — accept / reject / refine + hint]
        │
        ▼
Concept Population Report (report.py)
[breadth · goal coverage · context coverage · grammatical frame coverage]
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
│   ├── concept-ontology.md             # Shared theoretical grounding — all canonical vocabulary
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
# Minimal — bare term, single context/goal pair
python -m src.main --term "anger" \
  --context "receiving unfair criticism at work" \
  --goal "restore social fairness" \
  --max-iterations 3 --threshold 7.5 --no-human
```

```bash
# Full — with all linguistic layers specified (recommended)
python -m src.main \
  --term "fire" \
  --seed-phrase "to fire (someone)" \
  --grammatical-frame "transitive verb, agent=manager, patient=employee" \
  --morphemes "fire" \
  --phonesthetics-note "fi- cluster: forceful action" \
  --context "the manager fired her in front of the team" \
  --goal "restore power balance" \
  --max-iterations 3 --threshold 7.5
```

The report is written to `docs/concept_population_report.md` by default. Use `--output <path>` to override.

**Using a contexts file (multiple context/goal pairs):**

```bash
python -m src.main --term "safety" \
  --seed-phrase "a sense of safety" \
  --grammatical-frame "noun phrase, experiencer subject" \
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

| Barrett / Linguistic Term | Project Equivalent | Level |
|---|---|---|
| concept instance | `ConceptInstance` dataclass (`src/concept_population.py`) | 5 |
| population | `ConceptPopulation` — the full set of instances for a term | 5 |
| simulation | generated text prediction for a (context, goal) pair | 5 |
| functional adequacy | `adequacy_score` float 0–10; how well the instance serves the goal | 5 |
| goal | purpose the concept serves in context; drives adequacy scoring | 5 |
| grammatical frame | `grammatical_frame` field; syntactic role of the seed phrase | 4 |
| seed phrase | `seed_phrase` field; grammatically framed minimum input unit | 4 |
| morpheme | `morphemes: List[str]`; meaningful sub-word units (NOT BPE tokens) | 2 |
| phonesthetics | `phonesthetics_note: str`; optional sound-symbolism soft hint | 0 |
| prediction error | low `adequacy_score` + human rejection signal → triggers refinement | — |
| concept construction | the RL+RLHF loop itself; builds instances, never retrieves them | — |

Every module, field name, prompt, and skill uses Barrett + linguistic vocabulary. See [`docs/concept-ontology.md`](docs/concept-ontology.md) for the full enforced vocabulary table.

---

## Explain It Like I'm Nine

New to the project or sharing it with someone who is? [`docs/eli9-explainer.md`](docs/eli9-explainer.md) walks through the full idea without jargon — what the problem is, how the brain does it differently, what this system builds, and how the RL loop works. It also includes a section on **estimated training effort** with the current base model (Granite 13B Instruct v2): per-call token counts, scaling tables across session sizes, and a plain-English rundown of where the model fits well and where its limits are.

---

## Hackathon Journal

[`hackathon-journal.md`](hackathon-journal.md) — running log of all design decisions, theoretical insights, and architectural evolution (4 entries). Start here to understand *why* the system is built the way it is.

---

Built with IBM Bob + watsonx.ai · IBM TechXchange Hackathon 2026
