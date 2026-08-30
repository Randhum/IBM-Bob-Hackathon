#!/usr/bin/env bash
# =============================================================================
# record_demo.sh — Scene 5 terminal run for the hackathon video
#
# Usage:
#   cd <repo-root>
#   bash assets/record_demo.sh
#
# What it does:
#   1. Verifies the .venv / requirements are in order
#   2. Prints a visible countdown so you can start your screen recorder
#   3. Runs src.main with WATSONX_STUB=true (no API key needed)
#   4. Pauses 3 s at the end so the final output is fully visible on screen
#
# Recommended screen recorder settings:
#   Resolution : 1920×1080   FPS : 30   Font size : 16 pt
#   Trim the recording to start at "RECORDING" and end 3 s after the report.
# =============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── 1. Quick env check ────────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  echo ""
  echo "  .venv not found — creating and installing requirements..."
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi

# ── 2. Countdown ─────────────────────────────────────────────────────────────
echo ""
echo "  ┌─────────────────────────────────────────────────────┐"
echo "  │   Barrett Concept Construction — watsonx.ai Demo    │"
echo "  │   IBM TechXchange Hackathon 2026 · WATSONX_STUB=true │"
echo "  └─────────────────────────────────────────────────────┘"
echo ""
echo "  Start your screen recorder NOW."
echo ""
for i in 3 2 1; do
  echo "  Recording in $i..."
  sleep 1
done
echo ""
echo "  ┌─── RECORDING ───────────────────────────────────────┐"
echo ""

# ── 3. Run the concept loop ───────────────────────────────────────────────────
WATSONX_STUB=true .venv/bin/python -m src.main \
  --term "fire" \
  --seed-phrase "to fire someone" \
  --grammatical-frame "transitive verb, agent=manager, patient=employee" \
  --morphemes "fire" \
  --phonesthetics-note "fi- cluster: forceful, abrupt action" \
  --context "the manager fired her in front of the team" \
  --goal "restore power balance" \
  --max-iterations 3 \
  --threshold 7.5 \
  --no-human \
  --output docs/concept_population_report.md

# ── 4. Hold final screen ─────────────────────────────────────────────────────
echo ""
echo "  ┌─── END OF RECORDING ────────────────────────────────┐"
echo "  │  Report written → docs/concept_population_report.md │"
echo "  └─────────────────────────────────────────────────────┘"
echo ""
sleep 3
