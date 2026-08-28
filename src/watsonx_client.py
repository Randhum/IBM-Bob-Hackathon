"""
watsonx_client.py — Standardized IBM watsonx.ai API caller.

All LLM calls in the concept-learning workflow route through call_watsonx().
Supports WATSONX_STUB=true for dry-run demos without API credentials.
"""

from __future__ import annotations

import os
import re
import random
import time
import logging
from typing import List, Optional, Dict, Any

_STUB_RNG = random.Random(42)  # seeded for reproducible stub-mode runs

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_env(var: str) -> str:
    """Return env var value or raise a descriptive error."""
    value = os.getenv(var)
    if not value:
        raise EnvironmentError(
            f"❌ Missing env var: {var}. Check .env.example for required variables."
        )
    return value


def _is_stub_mode() -> bool:
    return os.getenv("WATSONX_STUB", "false").lower() == "true"


def _stub_response(call_type: str, prompt: str, model_id: str) -> Dict[str, Any]:
    """Return a deterministic-enough stub response for dry-run mode."""
    if call_type == "score":
        score = round(_STUB_RNG.uniform(5.0, 9.5), 2)
        return {
            "response": str(score),
            "score": score,
            "model_id": model_id,
            "call_type": call_type,
            "prompt_preview": prompt[:100],
        }
    # generate
    term_match = re.search(r'Concept:\s*["\']?([^"\'\n]+)["\']?', prompt)
    term = term_match.group(1).strip() if term_match else "unknown"
    stub_text = (
        f"Stub simulation for '{term}': a placeholder concept instance used in dry-run mode."
    )
    return {
        "response": stub_text,
        "score": None,
        "model_id": model_id,
        "call_type": call_type,
        "prompt_preview": prompt[:100],
    }


def _extract_score(text: str) -> Optional[float]:
    """Extract the first float/int from text; return None if invalid or out of range."""
    match = re.search(r"\d+(\.\d+)?", text)
    if not match:
        return None
    value = float(match.group())
    if not (0.0 <= value <= 10.0):
        logger.warning("⚠️ Score %s out of range [0, 10]; discarding.", value)
        return None
    return value


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def call_watsonx(
    prompt: str,
    call_type: str = "generate",
    max_tokens: int = 256,
    temperature: float = 0.7,
    stop_sequences: Optional[List[str]] = None,
    model_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Make a single call to watsonx.ai and return a structured result dict.

    Parameters
    ----------
    prompt:
        The full prompt string to send.
    call_type:
        ``"generate"`` (default) or ``"score"``.  When ``"score"``, overrides
        max_tokens=32, temperature=0.2, stop_sequences=["\n"].
    max_tokens:
        Maximum new tokens for generation calls.
    temperature:
        Sampling temperature for generation calls.
    stop_sequences:
        List of stop strings (generation calls only).
    model_id:
        Model override; falls back to WATSONX_MODEL_ID env var, then granite default.

    Returns
    -------
    dict with keys: response, score, model_id, call_type, prompt_preview
    """
    if stop_sequences is None:
        stop_sequences = []

    resolved_model = model_id or os.getenv(
        "WATSONX_MODEL_ID", "ibm/granite-13b-instruct-v2"
    )

    # Override params for score calls
    if call_type == "score":
        max_tokens = 32
        temperature = 0.2
        stop_sequences = ["\n"]

    # --- Stub mode ---
    if _is_stub_mode():
        return _stub_response(call_type, prompt, resolved_model)

    # --- Live mode: validate credentials ---
    api_key = _get_env("WATSONX_API_KEY")
    project_id = _get_env("WATSONX_PROJECT_ID")
    url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

    logger.debug("watsonx call [%s] key=%s**** model=%s", call_type, api_key[:4], resolved_model)

    try:
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
    except ImportError as exc:
        raise ImportError(
            "ibm-watsonx-ai is not installed. Run: pip install ibm-watsonx-ai"
        ) from exc

    credentials = Credentials(url=url, api_key=api_key)

    params: Dict[str, Any] = {
        GenParams.MAX_NEW_TOKENS: max_tokens,
        GenParams.TEMPERATURE: temperature,
    }
    if stop_sequences:
        params[GenParams.STOP_SEQUENCES] = stop_sequences

    def _do_call() -> str:
        model = ModelInference(
            model_id=resolved_model,
            credentials=credentials,
            project_id=project_id,
            params=params,
        )
        return model.generate_text(prompt=prompt)

    # --- Execute with error handling ---
    try:
        raw = _do_call()
    except Exception as exc:
        exc_str = str(exc)
        if "401" in exc_str or "403" in exc_str:
            print("❌ watsonx.ai auth failed. Check WATSONX_API_KEY.")
            raise
        if "429" in exc_str:
            logger.warning("Rate limited (429). Waiting 5s before retry…")
            time.sleep(5)
            raw = _do_call()  # retry once — let any subsequent 429 propagate
        else:
            logger.error("watsonx call failed: %s", exc)
            raise

    if not raw or not raw.strip():
        logger.warning("⚠️ Empty response from watsonx.ai for prompt: %s...", prompt[:80])
        return {
            "response": None,
            "score": None,
            "model_id": resolved_model,
            "call_type": call_type,
            "prompt_preview": prompt[:100],
        }

    # --- Post-process ---
    score = None
    if call_type == "score":
        score = _extract_score(raw)
        if score is None:
            logger.warning("⚠️ Could not parse score from: %r", raw[:60])

    return {
        "response": raw.strip(),
        "score": score,
        "model_id": resolved_model,
        "call_type": call_type,
        "prompt_preview": prompt[:100],
    }
