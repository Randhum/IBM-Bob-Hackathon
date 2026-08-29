# Construction Domain Training Report
## Barrett-Aligned LoRA Fine-Tuning — Two-Track Demonstration

> **IBM TechXchange Hackathon 2026**
> Status: Template — fill in benchmark results after running `notebooks/construction_benchmark.ipynb`

---

## 1. Overview

This report documents the construction-domain fine-tuning demonstration built on top of the
Barrett-aligned concept-population workflow. The same self-generated corpus of labelled
`(context, goal, simulation, adequacy_score)` tuples is used as training data for two
independent LoRA fine-tuning jobs on watsonx.ai Tuning Studio (IBM Cloud).

**The central claim:** Barrett's concept-as-population framework is not only a useful runtime
architecture — it is a viable *data generation strategy*. Training on goal-indexed,
context-anchored, adequacy-labelled instances produces models that both construct better
domain-specific concept simulations (generator track) and evaluate them more accurately
(judge track), without any domain prompting at inference time.

---

## 2. Domain Choice Rationale

**Domain:** Construction industry

| Criterion | Assessment |
|---|---|
| Narrow enough for 50 examples | ✅ 10 terms × 5 pairs = 50 instances covers the core vocabulary |
| Strong base-model prior gap | ✅ Construction-specific concepts (tolerances, practical completion, CDM) are underrepresented in general LLM training data |
| Realistic (context, goal) variation | ✅ Same term (e.g. "liability") has genuinely different functional meanings in different construction contexts |
| Barrett alignment | ✅ Construction concepts are heavily goal-indexed — "scaffolding" means different things to a safety manager vs a quantity surveyor vs a structural engineer |

---

## 3. Corpus Specification

**File:** `data/corpus_spec.json`

| Term | Seed Phrase | Grammatical Frame | # Pairs |
|---|---|---|---|
| scaffolding | to erect scaffolding | transitive verb, agent=crew, patient=structure | 5 |
| load-bearing | a load-bearing wall | attributive adjective modifying noun | 5 |
| tolerances | construction tolerances | plural noun phrase | 5 |
| site induction | to deliver a site induction | transitive verb phrase, agent=site-manager | 5 |
| curing | curing of concrete | gerund noun phrase | 5 |
| formwork | to strike the formwork | transitive verb phrase, agent=carpenter | 5 |
| sub-contractor | to appoint a sub-contractor | transitive verb phrase, agent=main-contractor | 5 |
| PPE | wearing appropriate PPE | gerund phrase | 5 |
| practical completion | to certify practical completion | transitive verb phrase, agent=contract-administrator | 5 |
| liability | contractual liability | noun phrase, attributive modifier=contractual | 5 |
| **Total** | | | **50** |

All (context, goal) pairs were human-authored to ensure:
- Every context mentions a concrete actor, action, or physical element
- Every goal starts with a functional verb ("protect", "ensure", "assign", etc.)
- No abstract contexts ("a general construction scenario")

---

## 4. Self-Generation and Labelling

**Script:** `src/generate_corpus.py`
**Base model for generation:** `ibm/granite-13b-instruct-v2` (Granite 13B — hosted on watsonx.ai)

### Generation Configuration

| Parameter | Value |
|---|---|
| Max RL refinement rounds | 2 |
| Adequacy threshold | 7.5 |
| Auto-accept threshold | > 8.0 |
| Auto-reject threshold | < 6.0 |
| Borderline range (human review) | 6.0 – 8.0 |

### Labelling Statistics

> **TODO:** Fill in after running `python -m src.generate_corpus`

| Metric | Value |
|---|---|
| Total instances generated | ___ |
| Auto-accepted (score > 8.0) | ___ |
| Borderline (reviewed by human) | ___ |
| Auto-rejected (score < 6.0) | ___ |
| Final accepted instances | ___ |

**Output files:**
- `data/construction_raw_population.json` — all generated instances with auto-scores
- `data/construction_labelled_population.json` — after hybrid human review

---

## 5. Training Data Export

**Script:** `src/export_training_data.py`

### Format A — Judge Training JSONL

**File:** `data/construction_judge_training.jsonl`

```
input:  Concept: "scaffolding"
        Seed phrase: "to erect scaffolding"
        Grammatical frame: "transitive verb, agent=crew, patient=structure"
        Context: "..."
        Goal: "..."
        Simulation: "..."

        On a scale of 0–10, how functionally adequate is this simulation?
        ...
        Reply with only a number.

output: "8.4"
```

Includes all accepted instances (full score range). Generator learns the adequacy distribution.

> **TODO:** Fill in after running `python -m src.export_training_data`

| Metric | Value |
|---|---|
| Judge training lines | ___ |
| Score range | ___ – ___ |
| Mean score | ___ |

### Format B — Generator Training JSONL

**File:** `data/construction_generator_training.jsonl`

```
input:  Concept: "scaffolding"
        Seed phrase: "to erect scaffolding"
        Grammatical frame: "transitive verb, agent=crew, patient=structure"
        Context: "..."
        Goal: "..."

        Generate a simulation — a prediction of the experience, response, or behavior
        this concept would produce in this context to serve this goal. ...

output: "The site crew erects tubular steel scaffolding around the south facade, ..."
```

Includes only instances with adequacy_score ≥ 8.0. Model learns from high-quality targets only.

> **TODO:** Fill in after running `python -m src.export_training_data`

| Metric | Value |
|---|---|
| Generator training lines | ___ |
| Score range (all ≥ 8.0) | ___ – ___ |
| Mean score | ___ |

### Held-Out Eval Set

**File:** `data/construction_eval.jsonl`
One instance per term (highest adequacy_score). Used by the benchmark notebook.

> **TODO:** Fill in after export

| Metric | Value |
|---|---|
| Eval instances | ___ (target: 10) |
| Mean eval score | ___ |

---

## 6. Fine-Tuning Configuration

**Script:** `src/launch_tuning_job.py`
**Infrastructure:** IBM Cloud — watsonx.ai Tuning Studio (no local GPU)

| Parameter | Value |
|---|---|
| Base model | `ibm/granite-3b-code-instruct` |
| Method | LoRA (Low-Rank Adaptation) |
| Epochs | 5 |
| Batch size | 8 |
| Learning rate | 2e-4 (0.0002) |

### Judge Job

| Field | Value |
|---|---|
| Training file | `construction_judge_training.jsonl` |
| Job ID | ___ *(fill after launch)* |
| Tuned model ID | ___ *(fill after training completes)* |

### Generator Job

| Field | Value |
|---|---|
| Training file | `construction_generator_training.jsonl` |
| Job ID | ___ *(fill after launch)* |
| Tuned model ID | ___ *(fill after training completes)* |

Reproducibility instructions: [`docs/tuning-setup.md`](tuning-setup.md)

---

## 7. Benchmark Results

> **TODO:** Fill in after running `notebooks/construction_benchmark.ipynb`

**Notebook:** [`notebooks/construction_benchmark.ipynb`](../notebooks/construction_benchmark.ipynb)

### Generation Comparison (3 selected terms)

| Term | Base Simulation | Generator-Tuned Simulation | Judge-Tuned Simulation |
|---|---|---|---|
| scaffolding | *(paste from notebook)* | *(paste from notebook)* | *(paste from notebook)* |
| practical completion | | | |
| liability | | | |

### Summary Delta Table (all 10 terms)

| Term | A: base/base | B: base/tuned-judge | C: tuned-gen/base | D: tuned-gen/tuned-judge | Gen Δ (C−A) | Judge Δ (B−A) |
|---|---|---|---|---|---|---|
| scaffolding | ? | ? | ? | ? | ? | ? |
| load-bearing | ? | ? | ? | ? | ? | ? |
| tolerances | ? | ? | ? | ? | ? | ? |
| site induction | ? | ? | ? | ? | ? | ? |
| curing | ? | ? | ? | ? | ? | ? |
| formwork | ? | ? | ? | ? | ? | ? |
| sub-contractor | ? | ? | ? | ? | ? | ? |
| PPE | ? | ? | ? | ? | ? | ? |
| practical completion | ? | ? | ? | ? | ? | ? |
| liability | ? | ? | ? | ? | ? | ? |
| **MEAN** | ? | ? | ? | ? | ? | ? |

### Barrett Alignment Interpretation

> *(Fill in after results are available)*

---

## 8. Barrett Alignment Summary

| Barrett Concept | Project Implementation |
|---|---|
| Concept as population | 50 labelled instances across 10 terms — the training corpus *is* the population |
| Goal-indexed instances | Every training example is anchored to a specific (context, goal) pair |
| Functional adequacy | `adequacy_score` 0–10 is the training label for the judge; the scoring criterion for generator data selection |
| Prediction error | Low-scoring instances excluded from generator training; borderline instances reviewed by human |
| Concept construction | Generator-tuned model learns to construct goal-indexed simulations — not retrieve definitions |
| Self-generation as data pipeline | The same RL workflow used at runtime is used to produce the training data — theory and engineering are aligned |

---

*Built with IBM Bob + watsonx.ai · IBM TechXchange Hackathon 2026*
