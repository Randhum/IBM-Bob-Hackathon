# How-To: watsonx.ai Tuning with the Python SDK (v1.4.11+)

> Based on the [official IBM watsonx.ai Python SDK docs — Tuning](https://ibm.github.io/watsonx-ai-python-sdk/v1.4.11/prompt_tuner.html).
> All tuning in this project uses the **`TuneExperiment`** API — **not** the legacy `FineTuning` class.

---

## Prerequisites

- Python **3.12** (enforced by `.python-version` and `pyproject.toml`)
- A clean venv based on Python 3.12:
  ```bash
  python3.12 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- `.env` populated with credentials (copy from `.env.example`):
  ```env
  WATSONX_API_KEY=<your IBM Cloud API key>
  WATSONX_PROJECT_ID=<your watsonx.ai project id>
  WATSONX_URL=https://us-south.ml.cloud.ibm.com
  ```
  > **Tip:** Copy the `project_id` from **Project → Manage → General → Details** in the watsonx.ai UI.

---

## 1. Connect to watsonx.ai

```python
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.experiment import TuneExperiment
import os
from dotenv import load_dotenv

load_dotenv()

credentials = Credentials(
    url=os.environ["WATSONX_URL"],
    api_key=os.environ["WATSONX_API_KEY"],
)

experiment = TuneExperiment(
    credentials,
    project_id=os.environ["WATSONX_PROJECT_ID"],
)
```

---

## 2. LoRA / Full Fine-Tuning with `FineTuner`

### 2a. Configure the tuner

```python
fine_tuner = experiment.fine_tuner(
    name="construction-judge-lora",
    description="Construction-domain judge LoRA fine-tuning",
    base_model="ibm/granite-3b-code-instruct",
    task_id="generation",
    num_epochs=5,
    learning_rate=0.0002,
    batch_size=8,
    max_seq_length=1024,
    accumulate_steps=4,
    verbalizer="### Input: {{input}}\n\n### Response: {{output}}",
    response_template="\n### Response:\n",
    gpu={"num": 1},
    auto_update_model=True,
)
```

#### With explicit LoRA / QLoRA parameters

```python
from ibm_watsonx_ai.foundation_models.schema import PeftParameters

fine_tuner = experiment.fine_tuner(
    name="construction-judge-lora",
    base_model="ibm/granite-3b-code-instruct",
    task_id="generation",
    num_epochs=5,
    learning_rate=0.0002,
    batch_size=8,
    peft_parameters=PeftParameters(
        type="lora",
        rank=8,
        lora_alpha=2,
        lora_dropout=0.05,
        target_modules=["all-linear"],
    ),
    gpu={"num": 1},
    auto_update_model=True,
)
```

### 2b. Inspect configuration

```python
print(fine_tuner.get_params())
```

### 2c. Launch the run

```python
from ibm_watsonx_ai.helpers import DataConnection

# Option A — data asset already uploaded to the project
tuning_details = fine_tuner.run(
    training_data_references=[
        DataConnection(data_asset_id="<asset-id-from-watsonx-ui>")
    ],
    background_mode=True,   # fire-and-forget; poll manually below
)

# Option B — file in project container storage
from ibm_watsonx_ai.helpers import ContainerLocation
tuning_details = fine_tuner.run(
    training_data_references=[
        DataConnection(location=ContainerLocation("data/construction_judge_training.jsonl"))
    ],
    background_mode=False,  # block until complete
)
```

### 2d. Monitor progress

```python
status = fine_tuner.get_run_status()
# "running"  |  "completed"  |  "failed"

run_details = fine_tuner.get_run_details()
```

### 2e. Get results

```python
# Summary table (pandas DataFrame)
print(fine_tuner.summary())

# Learning curve (Jupyter only)
fine_tuner.plot_learning_curve()

# Training logs
fine_tuner.get_logs()  # writes training.log

# Tuned model ID — use this in .env as WATSONX_JUDGE_MODEL_ID
model_id = fine_tuner.get_model_id()
print(model_id)
```

> `get_model_id()` only works when `auto_update_model=True` (the default).

---

## 3. Prompt Tuning with `PromptTuner`

> **Note:** Prompt Tuning is deprecated for IBM Cloud Pak for Data since v5.2 and will be removed in a future release.
> For new work, prefer `FineTuner` (section 2 above).

### 3a. Configure the tuner

```python
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes

prompt_tuner = experiment.prompt_tuner(
    name="construction-prompt-tuner",
    task_id=experiment.Tasks.GENERATION,
    base_model=ModelTypes.FLAN_T5_XL,
    accumulate_steps=32,
    batch_size=16,
    learning_rate=0.2,
    max_input_tokens=256,
    max_output_tokens=128,
    num_epochs=6,
    tuning_type=experiment.PromptTuningTypes.PT,
    verbalizer="### Input: {{input}}\n\n### Response: {{output}}",
    auto_update_model=True,
)
```

### 3b. Launch and monitor (same API as FineTuner)

```python
tuning_details = prompt_tuner.run(
    training_data_references=[
        DataConnection(data_asset_id="<asset-id>")
    ],
    background_mode=True,
)

print(prompt_tuner.get_run_status())
model_id = prompt_tuner.get_model_id()
```

---

## 4. After Training — Use the Tuned Model

```python
# Add to .env:
# WATSONX_JUDGE_MODEL_ID=<model_id from fine_tuner.get_model_id()>
# WATSONX_GENERATOR_MODEL_ID=<model_id from fine_tuner.get_model_id()>

from ibm_watsonx_ai.foundation_models import ModelInference

model = ModelInference(
    model_id=os.environ["WATSONX_JUDGE_MODEL_ID"],
    credentials=credentials,
    project_id=os.environ["WATSONX_PROJECT_ID"],
)

response = model.generate("Your prompt here")
print(response)
```

---

## 5. Quick Reference — Key Classes

| Class | Import | Purpose |
|---|---|---|
| `TuneExperiment` | `ibm_watsonx_ai.experiment` | Entry point — creates tuners and lists runs |
| `FineTuner` | returned by `experiment.fine_tuner()` | LoRA / full fine-tuning |
| `PromptTuner` | returned by `experiment.prompt_tuner()` | Prompt tuning (deprecated on CP4D ≥5.2) |
| `PeftParameters` | `ibm_watsonx_ai.foundation_models.schema` | LoRA/QLoRA hyperparameters |
| `DataConnection` | `ibm_watsonx_ai.helpers` | Points to training data (asset, S3, container) |
| `ModelInference` | `ibm_watsonx_ai.foundation_models` | Run inference against a tuned model |

---

## 6. Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError: ibm_watsonx_ai` | SDK not installed in active venv — run `pip install 'ibm-watsonx-ai>=1.1.0'` |
| `RuntimeError: Python 3.12+ required` | Wrong venv/interpreter — activate `.venv` with `source .venv/bin/activate` |
| `Missing env vars: WATSONX_API_KEY` | Copy `.env.example` → `.env` and fill in credentials |
| `get_model_id()` returns `None` | Training not yet complete, or `auto_update_model=False` — wait and retry |
| Job state `FAILED` | Check `fine_tuner.get_run_details()` — most common cause is malformed JSONL |
| `Training file not found` | Run `python -m src.export_training_data` first to generate JSONL files |
