# Bob Session Screenshots

This directory holds the required Bob session screenshots for hackathon submission evidence.
Each screenshot documents a specific Bob session used to build the project.

---

## Purpose

Hackathon judges require visual evidence that IBM Bob was used throughout the development
process. Screenshots must show the Bob UI, the chat history, and the relevant output
(files generated, skills invoked, plan content, etc.).

---

## Required Screenshots

### `plan-session.png`
**Session type:** Bob Plan mode
**What to show:**
- The Bob Plan mode chat session in which `hackathon-kickoff-plan.md` was produced
- The sub-task breakdown visible in the output or chat
- Ideally: the Barrett framework discussion (constructionist theory, concept-as-population)
- The generated plan file title visible in the response

---

### `agent-coding-session.png`
**Session type:** Bob Agent mode — Sub-Task 4 (Python implementation)
**What to show:**
- Bob Agent mode generating the core Python source files (`src/concept_population.py`,
  `src/concept_loop.py`, `src/judge.py`, `src/human_feedback.py`, `src/report.py`)
- File generation in progress or the completed file contents shown in chat
- Ideally: the `concept_loop.py` or `concept_population.py` file being written

---

### `agent-deliverables-session.png`
**Session type:** Bob Agent mode — Sub-Tasks 2 and 3 (written deliverables)
**What to show:**
- Bob Agent mode generating `docs/problem-solution-statement.md` and/or
  `docs/bob-usage-statement.md`
- The written content visible in the chat response
- Ideally: the Barrett framework language ("conceptual instance", "population breadth",
  "functional adequacy") visible in the generated text

---

### `rl-loop-demo.png`
**Session type:** Bob session running the `rl-feedback-loop` skill
**What to show:**
- The `rl-feedback-loop` skill activated in Bob on the concept "anger"
- A concept instance being reviewed with its context, goal, and simulation
- The adequacy scoring output (numerical score 0–10)
- Ideally: a refinement step visible showing score improvement from round N to round N+1

---

## Naming Convention

All screenshot files must use **exactly** the filenames listed above. Do not add suffixes,
prefixes, numbers, or timestamps. The filenames are referenced by `README.md`, the Bob usage
statement, and the video script.

```
assets/screenshots/plan-session.png
assets/screenshots/agent-coding-session.png
assets/screenshots/agent-deliverables-session.png
assets/screenshots/rl-loop-demo.png
```

---

## How to Capture

- **Browser (Chrome/Firefox/Edge):** Press `F12` → no; use `Ctrl+Shift+S` (Firefox) or the
  browser's built-in screenshot shortcut, or use the OS shortcut below.
- **Linux:** `PrtSc` for full screen; `Shift+PrtSc` for region select (GNOME).
- **macOS:** `Cmd+Shift+3` for full screen; `Cmd+Shift+4` for region select.
- **Windows:** `Win+Shift+S` for Snipping Tool region select; `PrtSc` for full screen.
- Crop the screenshot to show the Bob UI clearly, removing unrelated browser tabs or toolbars.
- Save as `.png` (not `.jpg` or `.webp`).

---

## What NOT to Capture

> ⚠️ **Never include screenshots that show any of the following:**
>
> - The `.env` file or its contents
> - Any API key, token, or credential in any form
> - The `.bob/` configuration directory if it contains secrets
> - Any terminal output showing a live API key or secret
>
> If a terminal window with credentials is visible in the background, crop it out or
> re-take the screenshot with that window closed.
