# Construction-Domain Training Plan
## Barrett-Aligned Fine-Tuning of Granite 3B via Self-Generated Labelled Data

---

## Top-Level Overview

**Goal:** Demonstrate that Barrett's concept-as-population model can be used as a *data
generation strategy* — not just a runtime workflow. We self-generate a labelled training
corpus of `(context, goal, simulation, adequacy_score)` tuples for 10 construction-domain
concept terms, then use that same corpus to produce **two separate LoRA fine-tuning jobs**
on `ibm/granite-3b-code-instruct`:

1. **Generator fine-tuning** — trains the model to produce high-quality, goal-indexed
   construction-domain simulations given a (context, goal) input. Only high-scoring
   instances (adequacy_score ≥ 8.0) are used as training targets.
2. **Judge fine-tuning** — trains the model to accurately score functional adequacy in
   the construction domain. All scored instances across the full 0–10 range are used.

A before-vs-after Jupyter notebook then compares three models — base, generator-tuned, and
judge-tuned — on both generation quality and adequacy scoring, with a summary delta table
per term.

**The claim we are demonstrating:** A model trained on Barrett-structured labelled data —
where every training example is anchored to a (context, goal) pair — learns both to
*construct* better domain-specific concept simulations and to *evaluate* them, without any
domain prompting at inference time. This is Barrett's claim made empirically testable.

**Domain:** Construction industry (terms like "load-bearing", "tolerances", "site safety",
"scaffolding", "sub-contractor", etc.). Narrow enough that 50 training examples produce a
visible domain shift. Domain-specific enough that the base model's generic priors are a
meaningful comparison baseline.

**Time constraint:** Self-generation + labelling must complete in ~1 hour so both training
jobs can start tonight. The notebook can be written in parallel while training runs.

**What is NOT in scope:**
- Production deployment of any fine-tuned model.
- Vector database or retrieval-augmented generation.
- Prompt-tuning or soft-prompt methods — LoRA only for both jobs.
- Any prompting of the 3B model during the self-generation step (generation uses the
  existing Granite 13B base; the 3B model is only trained, not used for generation).

---

## Architecture Flow

```
Sub-Task 1: Define construction domain corpus
  10 terms × 5 (context, goal) pairs → corpus_spec.json

Sub-Task 2: Self-generate + auto-score simulations (Granite 13B)
  concept_loop.py (existing) → 50 raw (context, goal, simulation, score) instances
  Borderline instances (score 6–8) → human review CLI

Sub-Task 3: Export TWO labelled training JSONL files from same corpus
  ┌─ Format A — Judge JSONL ──────────────────────────────────────────────┐
  │  input:  term + seed_phrase + context + goal + simulation             │
  │  output: "8.2"   (adequacy score as string)                           │
  │  instances: all accepted, full score range                            │
  └───────────────────────────────────────────────────────────────────────┘
  ┌─ Format B — Generator JSONL ──────────────────────────────────────────┐
  │  input:  term + seed_phrase + context + goal + "Generate simulation:" │
  │  output: <accepted simulation text>                                   │
  │  instances: only adequacy_score ≥ 8.0  (high-quality targets only)   │
  └───────────────────────────────────────────────────────────────────────┘

Sub-Task 4: Upload + launch TWO LoRA jobs on watsonx Tuning Studio
  Job A: ibm/granite-3b-code-instruct + judge JSONL    → judge-tuned model
  Job B: ibm/granite-3b-code-instruct + generator JSONL → generator-tuned model

Sub-Task 5: Write evaluation notebook
  notebooks/construction_benchmark.ipynb
  3-way comparison: base | generator-tuned | judge-tuned
  Sections: generation quality + adequacy scoring + summary delta table

Sub-Task 6: Update docs and README
  docs/construction-training-report.md
  README.md new section: Construction Domain Training
```

---

## Sub-Tasks

---

### Sub-Task 1 — Define the Construction Domain Corpus Spec

**Intent:** Produce a machine-readable spec (`corpus_spec.json`) that lists the 10 construction
terms and their 5 (context, goal) pairs each. This is the seed file that drives all subsequent
generation — it must be written by a human, not generated, to ensure the contexts are realistic
construction-domain scenarios and the goals are genuine functional purposes, not generic ones.

**Expected Outcomes:**
- `data/corpus_spec.json` exists with exactly 10 terms, each with exactly 5 (context, goal)
  pairs.
- Every context is a full sentence describing a real construction-site or engineering scenario.
- Every goal is a functional purpose (what the concept is serving in that context), not an
  abstract definition.
- Terms cover a spread: physical objects (scaffolding, formwork), processes (curing, tendering),
  relational concepts (tolerances, liability), and safety concepts (PPE, site induction).
- Grammatical frame and seed phrase are specified for each term (prevents tokenization collapse).

**Todo List:**
1. Create `data/` directory.
2. Write `data/corpus_spec.json` with this schema:
   ```json
   {
     "domain": "construction",
     "terms": [
       {
         "term": "scaffolding",
         "seed_phrase": "to erect scaffolding",
         "grammatical_frame": "transitive verb, agent=crew, patient=structure",
         "morphemes": ["scaffold", "-ing"],
         "instances": [
           { "context": "...", "goal": "..." },
           ...
         ]
       },
       ...
     ]
   }
   ```
3. Target 10 terms:
   - `scaffolding`, `load-bearing`, `tolerances`, `site induction`,
     `curing`, `formwork`, `sub-contractor`, `PPE`, `practical completion`, `liability`
4. Write all 50 (context, goal) pairs — 5 per term.
5. Review: every context must mention a concrete actor, action, or physical element.
   No abstract contexts ("a general construction scenario").
6. Review: every goal must start with a verb ("protect", "ensure", "assign", "verify", etc.).

**Relevant Context:**
- The existing `main.py` CLI already reads `--contexts-file` as a JSON list of
  `{"context": "...", "goal": "..."}` objects. Our `corpus_spec.json` wraps that format
  per term — the generation script (Sub-Task 2) will unpack it.
- See `docs/concept-ontology.md` §2.4 and §2.5 for the rules on what makes a valid context
  and goal.

**Status:** [x] done

---

### Sub-Task 2 — Self-Generate Simulations + Hybrid Labelling

**Intent:** Run the existing `concept_loop.py` RL generation workflow over all 50 (context,
goal) pairs to produce candidate simulations, auto-score every instance, then surface only
the borderline cases (score 6.0–8.0) for human review. The output is a list of 50 labelled
instances ready for JSONL export.

**Expected Outcomes:**
- `data/construction_raw_population.json` exists — a serialized `ConceptPopulation`-per-term
  for all 10 terms, all 50 instances, each with an `adequacy_score`.
- A short CLI session reviews borderline instances and records `human_signal`
  (`accept` / `reject` / `refine`) plus optional `hint`.
- Final accepted instance count ≥ 40 (aim for all 50; human review only touches the ~15
  borderline ones).
- No instance enters the training set without an `adequacy_score`.

**Todo List:**
1. Write `src/generate_corpus.py` — a new script (not modifying `main.py`) that:
   - Reads `data/corpus_spec.json`.
   - For each term, calls `run_rl_loop()` with `max_iterations=2`, `threshold=7.5`.
   - Accumulates one `ConceptPopulation` per term.
   - Saves all populations to `data/construction_raw_population.json`.
2. After generation, run `human_feedback.py` (existing) filtered to instances where
   `6.0 ≤ adequacy_score ≤ 8.0` — add a `--score-range` filter flag to the existing CLI.
3. Re-save the reviewed populations to `data/construction_labelled_population.json`.
4. Log total counts: auto-accepted (>8.0), human-reviewed (6–8), auto-rejected (<6.0, excluded).

**Relevant Context:**
- `src/concept_loop.py` → `run_rl_loop()` is the core function. It already returns a
  `ConceptPopulation` and handles all prompt-building and scoring. No changes needed.
- `src/human_feedback.py` exists but currently processes all instances. The `--score-range`
  filter is a minimal addition: skip instances outside the range.
- `WATSONX_STUB=false` must be set for this run (real API needed).
- Base model for generation: keep `ibm/granite-13b-instruct-v2` (as currently configured).
  The 3B model is only used for fine-tuning, not generation.
- `src/watsonx_client.py` → `call_watsonx()` is the entry point for all LLM calls.

**Status:** [x] done

---

### Sub-Task 3 — Export Two Labelled Training JSONL Files

**Intent:** Transform the accepted labelled instances into two JSONL files — one for each
fine-tuning job — from the same underlying corpus. The split happens purely at export time
by changing how each instance is formatted: Format A teaches scoring (judge), Format B
teaches generation (generator). No additional generation is needed.

**Expected Outcomes:**
- `data/construction_judge_training.jsonl` — Format A (judge):
  ```json
  {
    "input": "Concept: \"scaffolding\"\nSeed phrase: \"to erect scaffolding\"\nGrammatical frame: \"...\"\nContext: \"...\"\nGoal: \"...\"\nSimulation: \"...\"\n\nRate the functional adequacy of this simulation (0–10):",
    "output": "8.2"
  }
  ```
  Contains all accepted instances, full score range. Target: ≥ 40 lines.

- `data/construction_generator_training.jsonl` — Format B (generator):
  ```json
  {
    "input": "Concept: \"scaffolding\"\nSeed phrase: \"to erect scaffolding\"\nGrammatical frame: \"...\"\nContext: \"...\"\nGoal: \"...\"\n\nGenerate a simulation — a prediction of the experience, response, or behavior this concept would produce in this context to serve this goal. Be specific. 1–3 sentences. ≤60 words:",
    "output": "The site crew erects tubular steel scaffolding around the south facade, distributing load through base plates to the ground slab, enabling safe overhead work at height while the cladding is fixed."
  }
  ```
  Contains only instances where `adequacy_score ≥ 8.0` — high-quality simulations only,
  so the model learns from correct examples. Target: ≥ 25 lines.

- `data/construction_eval.jsonl` — held-out set (one per term, highest-scoring):
  Used in the benchmark notebook for both judge and generator evaluation.
  Contains the full instance fields so the notebook can run both formats against it.

**Todo List:**
1. Write `src/export_training_data.py`:
   - Reads `data/construction_labelled_population.json`.
   - Holds out the highest-scoring instance per term → `construction_eval.jsonl` first
     (so held-out instances are excluded from both training files).
   - **Format A (judge):** for each remaining accepted instance, build the judge prompt
     using a new `build_judge_prompt()` helper extracted from `judge.py`; write
     `{"input": <prompt>, "output": <str(score)>}` → `construction_judge_training.jsonl`.
   - **Format B (generator):** for each remaining accepted instance where
     `adequacy_score ≥ 8.0`, build the generator prompt using a new
     `build_generator_prompt()` helper extracted from `concept_loop.py`; write
     `{"input": <prompt>, "output": <simulation_text>}` →
     `construction_generator_training.jsonl`.
   - Print summary: judge train lines / generator train lines / eval lines.
2. Extract `build_judge_prompt()` from `src/judge.py` — same logic as the internal prompt
   construction in `score_instance()`, exposed as a standalone function.
3. Extract `build_generator_prompt()` from `src/concept_loop.py` — same logic as
   `_build_generation_prompt()`, exposed as a public function.
4. Validate both JSONL files — every line valid JSON, `input` non-empty, `output` non-empty.
5. Commit all three data files.

**Relevant Context:**
- `src/judge.py` → `score_instance()` — extract its internal prompt assembly as
  `build_judge_prompt(term, context, goal, simulation, seed_phrase, grammatical_frame, ...)`.
- `src/concept_loop.py` → `_build_generation_prompt()` — rename to `build_generator_prompt()`
  (public) with no logic change; `run_rl_loop()` continues to call it internally.
- watsonx Tuning Studio accepts JSONL with `input`/`output` keys by default.
- Both `output` fields must be strings — Tuning Studio treats all completions as text.
- The generator JSONL must NOT include the simulation in the input — that is the target the
  model learns to produce.

**Status:** [x] done

---

### Sub-Task 4 — Upload + Launch Two LoRA Jobs on watsonx Tuning Studio

**Intent:** Upload both JSONL files to watsonx.ai and start two LoRA fine-tuning jobs — one
for the judge, one for the generator — both targeting `ibm/granite-3b-code-instruct`. Both
jobs can run in parallel overnight. The output is two deployed model IDs recorded in a single
config file that the evaluation notebook reads.

**Expected Outcomes:**
- Both JSONL files uploaded as separate watsonx.ai dataset assets.
- Two LoRA fine-tuning jobs started on `ibm/granite-3b-code-instruct`.
- Both job IDs and (after completion) both tuned model IDs recorded in
  `data/tuning_job_config.json`:
  ```json
  {
    "base_model": "ibm/granite-3b-code-instruct",
    "method": "lora",
    "epochs": 5,
    "batch_size": 8,
    "learning_rate": 2e-4,
    "judge_job": {
      "training_file": "construction_judge_training.jsonl",
      "job_id": "<watsonx job id>",
      "tuned_model_id": "<fill in after training completes>"
    },
    "generator_job": {
      "training_file": "construction_generator_training.jsonl",
      "job_id": "<watsonx job id>",
      "tuned_model_id": "<fill in after training completes>"
    }
  }
  ```
- `docs/tuning-setup.md` written with step-by-step instructions for reproducing both
  uploads and job launches (so hackathon judges can verify training was real).

**Todo List:**
1. Write `src/launch_tuning_job.py` — Python SDK script that:
   - Accepts a `--job` argument: `judge` or `generator` (or `both` to launch sequentially).
   - For each job: loads the corresponding JSONL, uploads as a watsonx.ai data asset,
     creates a TuningExperiment targeting `ibm/granite-3b-code-instruct` with LoRA,
     sets `num_epochs=5`, `batch_size=8`, `learning_rate=2e-4`.
   - Prints the `job_id` for each and polls status every 60 s until `COMPLETED` or `FAILED`.
   - Writes both job IDs into `data/tuning_job_config.json`.
2. Write `docs/tuning-setup.md` documenting the manual fallback (Tuning Studio UI path)
   for both jobs in case the SDK script hits a permissions issue.
3. After training completes: fill `tuned_model_id` for both jobs in
   `data/tuning_job_config.json`.
4. Add `WATSONX_JUDGE_MODEL_ID` and `WATSONX_GENERATOR_MODEL_ID` to `.env.example`.

**Relevant Context:**
- `src/watsonx_client.py` → `_get_env()` — reuse for credential loading.
- watsonx.ai Python SDK: `ibm_watsonx_ai.foundation_models.finetuning` namespace
  (may need `ibm-watsonx-ai>=1.1.0` — update `requirements.txt` if needed).
- Both jobs use the same base model and same hyperparameters — the only difference is the
  training file and thus the learned behaviour.
- `data/tuning_job_config.json` is the single handoff artifact Sub-Task 5 reads to get
  both model IDs.

**Status:** [x] done

---

### Sub-Task 5 — Three-Way Evaluation Notebook

**Intent:** Write `notebooks/construction_benchmark.ipynb` — the primary demo artifact. It
runs three models (base, generator-tuned, judge-tuned) against the held-out eval set,
measuring both generation quality and adequacy scoring. The notebook tells the full Barrett
story: base model is context-blind; generator-tuned model produces goal-anchored simulations;
judge-tuned model evaluates them accurately in the construction domain.

**Expected Outcomes:**
- Notebook runs end-to-end with real API keys, or fully in stub mode for rehearsal.
- **Section 1 — Theory:** two markdown cells: Barrett framework summary + construction
  domain scope. Sets up *why* we expect a delta.
- **Section 2 — Setup:** loads `data/construction_eval.jsonl` and
  `data/tuning_job_config.json`; resolves all three model IDs.
- **Section 3 — Generation Comparison:** for 3 selected eval terms, generates one
  simulation per model (base / generator-tuned / judge-tuned) with `max_iterations=1`.
  Renders a side-by-side table: `term | base_simulation | generator_simulation | judge_simulation`.
- **Section 4 — Adequacy Scoring Comparison:** for all 10 eval instances, scores each
  simulation (from Section 3) using the base judge and the judge-tuned model independently.
  Records 4 scores per instance: base-generates+base-judges, base-generates+tuned-judges,
  tuned-generates+base-judges, tuned-generates+tuned-judges.
- **Section 5 — Summary Delta Table:** one row per term, columns:
  `term | base_gen_score | tuned_gen_score | gen_delta | base_judge_score | tuned_judge_score | judge_delta`.
  Rendered as a `pandas.DataFrame`.
- **Section 6 — Barrett Alignment Commentary:** markdown cell interpreting the results
  through Barrett's lens — what the deltas mean for the concept-as-population claim.
- The notebook never imports `main.py` — it calls `judge.score_instance()` and
  `concept_loop.run_rl_loop()` directly with explicit `model_id` overrides.

**Todo List:**
1. Create `notebooks/construction_benchmark.ipynb` with the 6 sections above.
2. Helper cell `generate_simulation(model_id, term_spec)` — calls `run_rl_loop()` with
   `max_iterations=1`, returns the single generated simulation text.
3. Helper cell `score_simulation(judge_model_id, term_spec, simulation)` — calls
   `judge.score_instance()` with explicit `model_id` override, returns float score.
4. Section 4 loop: for each eval instance, call both helpers with all model_id combinations
   and collect results into a dict keyed by term.
5. Section 5: build `pandas.DataFrame` from collected results; render with `display()`.
   Add `pandas` to `requirements.txt` if not already present.
6. Add narrative markdown cells before each section grounding the measurement in Barrett's
   framework — specifically explaining what the generator delta and the judge delta each
   demonstrate independently.
7. Test all cells in stub mode (`WATSONX_STUB=true`) — verify no errors before training
   completes.

**Relevant Context:**
- `src/judge.py` → `score_instance()` — needs `model_id` threaded through to
  `call_watsonx()`. The `call_watsonx()` signature already accepts `model_id` as an
  override; `score_instance()` needs a `model_id` parameter added (defaults to `None` →
  falls back to env var as today).
- `src/concept_loop.py` → `run_rl_loop()` — same: add optional `model_id` parameter
  threaded through to `call_watsonx()`.
- `data/construction_eval.jsonl` (Sub-Task 3) — input eval set.
- `data/tuning_job_config.json` (Sub-Task 4) — source of `judge_job.tuned_model_id` and
  `generator_job.tuned_model_id`.

**Status:** [x] done

---

### Sub-Task 6 — Documentation and README Update

**Intent:** Record the full two-track training approach, results, and Barrett alignment in
the project docs so judges can follow the complete chain: theory → self-generated corpus →
two LoRA jobs → three-way evaluation.

**Expected Outcomes:**
- `docs/construction-training-report.md` — structured report covering: domain choice
  rationale, corpus spec summary, generation + labelling stats, both tuning configs,
  full benchmark results table (copied from notebook), Barrett alignment commentary.
- `README.md` gains a new **Construction Domain Training** section linking to the report
  and the notebook, with a one-paragraph summary of the two-track training approach.
- `hackathon-journal.md` gains Entry 5 recording the pivot decision and its rationale.

**Todo List:**
1. Write `docs/construction-training-report.md` as a template with TODOs for the results
   sections that will be filled in after Sub-Task 5 runs. Pre-populate: domain rationale,
   corpus spec table (10 terms), both JSONL file stats, both tuning configs.
2. Add **Construction Domain Training** section to `README.md` after the Architecture
   section. Include: one-paragraph summary of the two-track approach, links to
   `docs/construction-training-report.md` and `notebooks/construction_benchmark.ipynb`,
   and a mini results table placeholder.
3. Add Entry 5 to `hackathon-journal.md` (at the top — most-recent-first):
   - Trigger: question of whether fine-tuning the generator is possible, not just the judge.
   - Decision: same self-generated corpus → two JSONL formats → two LoRA jobs.
   - Generator track uses only high-scoring (≥8.0) instances as targets.
   - Judge track uses full score range to learn the adequacy distribution.
   - Three-way evaluation (base / generator-tuned / judge-tuned) makes the Barrett claim
     empirically testable from two independent angles.
4. Commit all.

**Relevant Context:**
- `hackathon-journal.md` — append Entry 5 at the top (most-recent-first convention).
- `README.md` Architecture section ends at line ~57 — insert new section after it.
- Results table template for the report (fill after Sub-Task 5):
  ```
  | Term | Base Gen Score | Tuned Gen Score | Gen Δ | Base Judge Score | Tuned Judge Score | Judge Δ |
  |---|---|---|---|---|---|---|
  | scaffolding | ? | ? | ? | ? | ? | ? |
  ```

**Status:** [x] done

---

## Dependency Order

```
Sub-Task 1  corpus spec  ──►  Sub-Task 2  self-generate + label
                                    │
                                    ▼
                              Sub-Task 3  export JSONL
                                    │
                          ┌─────────┴──────────┐
                          ▼                    ▼
                    Sub-Task 4           Sub-Task 5
                    launch tuning        write notebook
                    job (tonight)        (parallel with training)
                          │                    │
                          └─────────┬──────────┘
                                    ▼
                              Sub-Task 6  docs + README
```

Sub-Tasks 1–3 must complete **today** (within ~1 hour) so training can start tonight.
Sub-Tasks 4–5 run in parallel (training runs overnight; notebook is written while it trains).
Sub-Task 6 completes after Sub-Task 5 benchmark results are in.

---

## Key Files Produced

| File | Sub-Task | Purpose |
|---|---|---|
| `data/corpus_spec.json` | 1 | 10 terms × 5 (context, goal) pairs — human-authored seed |
| `src/generate_corpus.py` | 2 | Batch generation script over all terms |
| `data/construction_raw_population.json` | 2 | Raw generated + auto-scored instances |
| `data/construction_labelled_population.json` | 2 | After human review of borderline cases |
| `src/export_training_data.py` | 3 | Dual-format JSONL exporter |
| `data/construction_judge_training.jsonl` | 3 | Judge fine-tuning set — full score range (≥40 lines) |
| `data/construction_generator_training.jsonl` | 3 | Generator fine-tuning set — score ≥8.0 only (≥25 lines) |
| `data/construction_eval.jsonl` | 3 | Held-out eval set — one per term, highest scoring |
| `src/launch_tuning_job.py` | 4 | SDK script to upload + launch both LoRA jobs |
| `data/tuning_job_config.json` | 4 | Both job IDs + both tuned model IDs (handoff artifact) |
| `docs/tuning-setup.md` | 4 | Reproducibility instructions for judges (both jobs) |
| `notebooks/construction_benchmark.ipynb` | 5 | Three-way evaluation notebook |
| `docs/construction-training-report.md` | 6 | Full two-track training report |

---

## Non-Goals

- Generating `corpus_spec.json` automatically — contexts and goals must be human-authored
  to ensure they are realistic construction-domain scenarios, not generic hallucinations.
- Prompt-tuning or soft-prompt methods — LoRA only for both jobs.
- Multi-domain corpus — construction only, this hackathon.
- Changing `watsonx_client.py` interfaces — credential and call logic is unchanged.
- Adding `model_id` parameters to `score_instance()` and `run_rl_loop()` beyond what is
  needed to thread the override through to `call_watsonx()` — minimal, targeted change only.
