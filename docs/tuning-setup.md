# Tuning Studio Setup Guide
## Reproducing the Construction-Domain LoRA Fine-Tuning Jobs

> This document allows hackathon judges and collaborators to reproduce both LoRA fine-tuning
> jobs that were run as part of the construction-domain training demonstration.
> All compute runs on **IBM Cloud (watsonx.ai Tuning Studio)** — no local GPU required.

---

## Prerequisites

1. IBM Cloud account with access to watsonx.ai.
2. A watsonx.ai project with Tuning Studio enabled.
3. API key and Project ID (see `.env.example`).
4. Python environment with dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```
5. `.env` file populated (copy from `.env.example`):
   ```bash
   cp .env.example .env
   # Fill in WATSONX_API_KEY and WATSONX_PROJECT_ID
   ```
6. Training JSONL files generated (see pipeline below).

---

## Full Pipeline (Start to Finish)

Run these steps in order. Steps 1–3 complete in ~1 hour. Steps 4–5 run overnight on IBM Cloud.

### Step 1 — Generate the Construction-Domain Corpus

```bash
# With real API (requires credentials in .env):
python -m src.generate_corpus

# Dry-run / stub mode (no API key needed, for testing):
WATSONX_STUB=true python -m src.generate_corpus --no-human
```

**Output:**
- `data/construction_raw_population.json` — auto-scored instances
- `data/construction_labelled_population.json` — after hybrid human review

### Step 2 — Export Training JSONL Files

```bash
python -m src.export_training_data
```

**Output:**
- `data/construction_judge_training.jsonl` — Format A, ≥40 lines
- `data/construction_generator_training.jsonl` — Format B, ≥25 lines
- `data/construction_eval.jsonl` — held-out eval set, one per term

### Step 3 — Launch Both LoRA Jobs

#### Option A — Python SDK Script (recommended)

```bash
python -m src.launch_tuning_job --job both
```

The script will:
1. Upload both JSONL files as watsonx.ai data assets.
2. Start two LoRA fine-tuning experiments on `ibm/granite-3b-code-instruct`.
3. Poll for completion every 60 seconds (expected: 30–90 min per job).
4. Write both job IDs and tuned model IDs to `data/tuning_job_config.json`.

**Fire-and-forget (start jobs, check status later):**
```bash
python -m src.launch_tuning_job --job both --no-poll
```
Then fill `tuned_model_id` manually in `data/tuning_job_config.json` after training completes.

#### Option B — Tuning Studio UI (manual fallback)

If the SDK script encounters a permissions issue, follow these steps in the UI:

1. Go to [watsonx.ai](https://dataplatform.cloud.ibm.com) → your project → **Assets**.
2. Upload `data/construction_judge_training.jsonl` as a **Data asset**.
3. Navigate to **Tuning Studio** → **New tuning experiment**.
4. Configure the judge job:
   - **Foundation model:** `ibm/granite-3b-code-instruct`
   - **Task:** Text generation
   - **Method:** LoRA
   - **Training data:** select the uploaded `construction_judge_training.jsonl` asset
   - **Epochs:** 5  |  **Batch size:** 8  |  **Learning rate:** 0.0002
   - Name: `construction-judge-lora`
5. Click **Start training**.
6. Repeat steps 2–5 for `construction_generator_training.jsonl`:
   - Name: `construction-generator-lora`
7. After both jobs complete, copy the **Tuned model IDs** from the Tuning Studio UI
   into `data/tuning_job_config.json`:
   ```json
   {
     "judge_job":     { "tuned_model_id": "<paste here>" },
     "generator_job": { "tuned_model_id": "<paste here>" }
   }
   ```

### Step 4 — Add Tuned Model IDs to `.env`

After training completes, add the two tuned model IDs to your `.env` file:

```env
WATSONX_JUDGE_MODEL_ID=<judge tuned model id from Tuning Studio>
WATSONX_GENERATOR_MODEL_ID=<generator tuned model id from Tuning Studio>
```

### Step 5 — Run the Benchmark Notebook

```bash
jupyter notebook notebooks/construction_benchmark.ipynb
```

All notebook cells can also be run in stub mode for rehearsal:
```bash
WATSONX_STUB=true jupyter notebook notebooks/construction_benchmark.ipynb
```

---

## Training Configuration Reference

| Parameter | Value |
|---|---|
| Base model | `ibm/granite-3b-code-instruct` |
| Method | LoRA (Low-Rank Adaptation) |
| Epochs | 5 |
| Batch size | 8 |
| Learning rate | 2e-4 (0.0002) |
| Judge training set | `construction_judge_training.jsonl` (≥40 examples, full score range) |
| Generator training set | `construction_generator_training.jsonl` (≥25 examples, score ≥ 8.0 only) |
| Infrastructure | IBM Cloud — watsonx.ai Tuning Studio |

---

## Job Status Reference

`data/tuning_job_config.json` tracks both jobs:

```json
{
  "base_model": "ibm/granite-3b-code-instruct",
  "method": "lora",
  "epochs": 5,
  "batch_size": 8,
  "learning_rate": 0.0002,
  "judge_job": {
    "training_file": "construction_judge_training.jsonl",
    "job_id": "<watsonx job id — filled by launch_tuning_job.py>",
    "tuned_model_id": "<filled after training completes>"
  },
  "generator_job": {
    "training_file": "construction_generator_training.jsonl",
    "job_id": "<watsonx job id — filled by launch_tuning_job.py>",
    "tuned_model_id": "<filled after training completes>"
  }
}
```

---

## Troubleshooting

| Issue | Resolution |
|---|---|
| `Missing env vars: WATSONX_API_KEY` | Copy `.env.example` to `.env` and fill in credentials |
| `ibm-watsonx-ai SDK not found` | Run `pip install 'ibm-watsonx-ai>=1.1.0'` |
| `Training file not found` | Run `python -m src.export_training_data` first |
| Job ends with `FAILED` state | Check Tuning Studio UI for error details; common cause is malformed JSONL |
| `tuned_model_id` not auto-extracted | Extract from Tuning Studio UI and fill in `tuning_job_config.json` manually |
| Benchmark notebook can't find model | Ensure `WATSONX_JUDGE_MODEL_ID` / `WATSONX_GENERATOR_MODEL_ID` set in `.env` |
