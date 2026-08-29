"""
launch_tuning_job.py — Upload JSONL training data and start LoRA fine-tuning jobs on
watsonx.ai Tuning Studio.

Supports launching either the judge job, the generator job, or both sequentially.
Polls for completion and writes job IDs + tuned model IDs to data/tuning_job_config.json.

All compute runs on IBM Cloud (watsonx.ai Tuning Studio) — no local GPU required.

Usage:
    # Launch both jobs (recommended — they run in parallel on IBM Cloud):
    python -m src.launch_tuning_job --job both

    # Launch judge job only:
    python -m src.launch_tuning_job --job judge

    # Launch generator job only:
    python -m src.launch_tuning_job --job generator

    # Launch without polling (fire-and-forget; fill tuned_model_id manually later):
    python -m src.launch_tuning_job --job both --no-poll

Credentials required in .env:
    WATSONX_API_KEY
    WATSONX_PROJECT_ID
    WATSONX_URL  (default: https://us-south.ml.cloud.ibm.com)
"""

from __future__ import annotations

import sys

if sys.version_info < (3, 12):
    raise RuntimeError(
        f"Python 3.12+ required (running {sys.version}). "
        "Recreate your venv: python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    )

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent
_DATA_DIR = _REPO_ROOT / "data"

_JUDGE_JSONL = _DATA_DIR / "construction_judge_training.jsonl"
_GENERATOR_JSONL = _DATA_DIR / "construction_generator_training.jsonl"
_JOB_CONFIG = _DATA_DIR / "tuning_job_config.json"

_BASE_MODEL = "ibm/granite-3b-code-instruct"
_LORA_CONFIG = {
    "method": "lora",
    "epochs": 5,
    "batch_size": 8,
    "learning_rate": 2e-4,
}
_POLL_INTERVAL_S = 60
_MAX_POLL_ATTEMPTS = 90  # 90 min maximum poll window


# ---------------------------------------------------------------------------
# Credential helpers
# ---------------------------------------------------------------------------

def _get_credentials() -> Dict[str, str]:
    """Load and validate required watsonx.ai credentials from environment."""
    import os
    missing = []
    creds = {}
    for key in ("WATSONX_API_KEY", "WATSONX_PROJECT_ID"):
        val = os.getenv(key)
        if not val:
            missing.append(key)
        else:
            creds[key] = val
    creds["WATSONX_URL"] = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    if missing:
        raise EnvironmentError(
            f"Missing required env vars: {', '.join(missing)}. "
            "Copy .env.example to .env and fill in your credentials."
        )
    return creds


# ---------------------------------------------------------------------------
# SDK helpers
# ---------------------------------------------------------------------------

def _get_sdk():
    """Import and return watsonx.ai SDK components, with a helpful error if missing."""
    try:
        from ibm_watsonx_ai import Credentials, APIClient
        try:
            # SDK >= 1.1.0 moved the module from finetuning -> tuning
            from ibm_watsonx_ai.foundation_models.tuning import FineTuning
        except ImportError:
            from ibm_watsonx_ai.foundation_models.finetuning import FineTuning
        return Credentials, APIClient, FineTuning
    except ImportError as exc:
        raise ImportError(
            "ibm-watsonx-ai SDK not found or version too old. "
            "Run: pip install 'ibm-watsonx-ai>=1.1.0'"
        ) from exc


def _build_client(creds: Dict[str, str]) -> Any:
    Credentials, APIClient, _ = _get_sdk()
    credentials = Credentials(url=creds["WATSONX_URL"], api_key=creds["WATSONX_API_KEY"])
    return APIClient(credentials=credentials, project_id=creds["WATSONX_PROJECT_ID"])


# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------

def _upload_training_file(client: Any, jsonl_path: Path, label: str) -> str:
    """Upload a JSONL file as a watsonx.ai data asset. Returns the asset ID."""
    logger.info("Uploading %s training file: %s", label, jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Training file not found: {jsonl_path}\n"
            "Run src/export_training_data.py first."
        )

    asset_details = client.data_assets.create(
        name=jsonl_path.name,
        file_path=str(jsonl_path),
    )
    asset_id = asset_details["metadata"]["asset_id"]
    logger.info("  ✓ Uploaded %s → asset_id: %s", jsonl_path.name, asset_id)
    return asset_id


# ---------------------------------------------------------------------------
# Fine-tuning job
# ---------------------------------------------------------------------------

def _start_tuning_job(
    client: Any,
    project_id: str,
    asset_id: str,
    label: str,
) -> str:
    """Create and start a LoRA fine-tuning experiment. Returns the job/experiment ID."""
    _, _, FineTuning = _get_sdk()

    logger.info("Starting LoRA fine-tuning job: %s", label)
    logger.info(
        "  Base model: %s | epochs: %d | batch: %d | lr: %s",
        _BASE_MODEL, _LORA_CONFIG["epochs"], _LORA_CONFIG["batch_size"], _LORA_CONFIG["learning_rate"],
    )

    ft = FineTuning(
        api_client=client,
        project_id=project_id,
    )

    response = ft.run(
        model_id=_BASE_MODEL,
        training_data_references=[{
            "type": "data_asset",
            "location": {"href": f"/v2/assets/{asset_id}?project_id={project_id}"},
        }],
        parameters={
            "method": _LORA_CONFIG["method"],
            "num_epochs": _LORA_CONFIG["epochs"],
            "batch_size": _LORA_CONFIG["batch_size"],
            "learning_rate": _LORA_CONFIG["learning_rate"],
        },
        name=f"construction-{label}-lora",
        description=f"Barrett construction-domain LoRA fine-tuning — {label} track",
    )

    job_id = response.get("metadata", {}).get("id") or response.get("entity", {}).get("training_id")
    if not job_id:
        raise RuntimeError(f"Could not extract job ID from tuning response: {response}")

    logger.info("  ✓ Job started: job_id = %s", job_id)
    return job_id


# ---------------------------------------------------------------------------
# Polling
# ---------------------------------------------------------------------------

def _poll_job(client: Any, project_id: str, job_id: str, label: str) -> Dict[str, Any]:
    """Poll a fine-tuning job until completion. Returns the final job details dict."""
    _, _, FineTuning = _get_sdk()
    ft = FineTuning(api_client=client, project_id=project_id)

    logger.info("Polling job %s (%s) every %ds…", job_id, label, _POLL_INTERVAL_S)

    for attempt in range(1, _MAX_POLL_ATTEMPTS + 1):
        details = ft.get_details(training_id=job_id)
        state = (
            details.get("entity", {}).get("status", {}).get("state")
            or details.get("metadata", {}).get("state")
            or "UNKNOWN"
        ).upper()

        logger.info("  [%d/%d] %s — state: %s", attempt, _MAX_POLL_ATTEMPTS, label, state)

        if state == "COMPLETED":
            logger.info("  ✓ %s job completed.", label)
            return details
        if state in ("FAILED", "CANCELED", "ERROR"):
            logger.error("  ✗ %s job ended with state: %s", label, state)
            logger.error("  Details: %s", json.dumps(details, indent=2))
            raise RuntimeError(f"Tuning job {job_id} ({label}) ended with state {state}")

        time.sleep(_POLL_INTERVAL_S)

    raise TimeoutError(
        f"Tuning job {job_id} ({label}) did not complete within "
        f"{_MAX_POLL_ATTEMPTS * _POLL_INTERVAL_S / 60:.0f} minutes."
    )


def _extract_tuned_model_id(job_details: Dict[str, Any]) -> Optional[str]:
    """Extract the deployed/tuned model ID from completed job details."""
    # Try common SDK response paths
    entity = job_details.get("entity", {})
    model_id = (
        entity.get("model_id")
        or entity.get("results", {}).get("model_id")
        or entity.get("status", {}).get("message")
        or job_details.get("metadata", {}).get("tuned_model_id")
    )
    return model_id or None


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------

def _load_config() -> Dict[str, Any]:
    if _JOB_CONFIG.exists():
        with open(_JOB_CONFIG, encoding="utf-8") as f:
            return json.load(f)
    return {
        "base_model": _BASE_MODEL,
        "method": _LORA_CONFIG["method"],
        "epochs": _LORA_CONFIG["epochs"],
        "batch_size": _LORA_CONFIG["batch_size"],
        "learning_rate": _LORA_CONFIG["learning_rate"],
        "judge_job": {
            "training_file": "construction_judge_training.jsonl",
            "job_id": None,
            "tuned_model_id": None,
        },
        "generator_job": {
            "training_file": "construction_generator_training.jsonl",
            "job_id": None,
            "tuned_model_id": None,
        },
    }


def _save_config(config: Dict[str, Any]) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_JOB_CONFIG, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    logger.info("Config saved → %s", _JOB_CONFIG)


# ---------------------------------------------------------------------------
# Per-job orchestration
# ---------------------------------------------------------------------------

def _run_job(
    client: Any,
    project_id: str,
    label: str,
    jsonl_path: Path,
    config: Dict[str, Any],
    poll: bool,
) -> None:
    """Upload, start, optionally poll, and record one fine-tuning job."""
    job_key = f"{label}_job"

    # Upload training file
    asset_id = _upload_training_file(client, jsonl_path, label)

    # Start tuning job
    job_id = _start_tuning_job(client, project_id, asset_id, label)
    config[job_key]["job_id"] = job_id
    _save_config(config)

    if not poll:
        logger.info(
            "  --no-poll set. Job %s submitted. Fill in tuned_model_id manually in %s "
            "after training completes.", job_id, _JOB_CONFIG
        )
        return

    # Poll to completion
    job_details = _poll_job(client, project_id, job_id, label)
    tuned_model_id = _extract_tuned_model_id(job_details)
    config[job_key]["tuned_model_id"] = tuned_model_id or "<extract-from-tuning-studio-ui>"
    _save_config(config)

    if tuned_model_id:
        logger.info("  ✓ Tuned model ID: %s", tuned_model_id)
    else:
        logger.warning(
            "  Could not auto-extract tuned_model_id from job details. "
            "Check Tuning Studio UI and fill in %s manually.", _JOB_CONFIG
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    creds = _get_credentials()
    client = _build_client(creds)
    project_id = creds["WATSONX_PROJECT_ID"]

    config = _load_config()

    jobs_to_run = []
    if args.job in ("judge", "both"):
        jobs_to_run.append(("judge", _JUDGE_JSONL))
    if args.job in ("generator", "both"):
        jobs_to_run.append(("generator", _GENERATOR_JSONL))

    for label, jsonl_path in jobs_to_run:
        _run_job(client, project_id, label, jsonl_path, config, poll=not args.no_poll)

    print(f"\n{'─' * 60}")
    print("  TUNING JOB SUMMARY")
    print(f"{'─' * 60}")
    for job_key in ("judge_job", "generator_job"):
        j = config.get(job_key, {})
        print(f"  {job_key}")
        print(f"    job_id         : {j.get('job_id') or 'not started'}")
        print(f"    tuned_model_id : {j.get('tuned_model_id') or 'pending'}")
    print(f"  Config file: {_JOB_CONFIG}")
    print(f"{'─' * 60}\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload JSONL and start LoRA fine-tuning jobs on watsonx.ai Tuning Studio."
    )
    parser.add_argument(
        "--job", choices=["judge", "generator", "both"], default="both",
        help="Which fine-tuning job to launch (default: both).",
    )
    parser.add_argument(
        "--no-poll", action="store_true",
        help="Submit jobs and exit without polling for completion.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(_parse_args())
