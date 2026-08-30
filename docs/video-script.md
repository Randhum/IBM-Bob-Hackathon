# invideo.ai — Full Video Generation Prompt

**Project:** Barrett Concept Construction — Two LLMs on watsonx.ai  
**Platform:** invideo.ai (AI video generator)  
**Target runtime:** 120 seconds  
**Aspect ratio:** 16:9 (1920×1080)  
**Tone:** Professional tech demo — clear, confident, slightly cinematic  
**Music mood:** Minimal ambient electronic, low volume, no drops or vocals  

---

## HOW TO USE THIS FILE

1. Go to **invideo.ai → AI Video** (or "Text to Video")
2. Paste the **Master Prompt** (section below) into the script/prompt box
3. Select style: **Tech / Corporate / Modern Dark**
4. Select voice: **English (US), Male, calm & authoritative** — e.g. "James" or "Michael"
5. Let invideo generate the first draft, then use the per-scene instructions in
   **Scene Details** to manually adjust individual clips, overlays, and transitions.

---

## MASTER PROMPT
*(paste this entire block into invideo.ai's prompt or script field)*

```
Create a 120-second professional tech demo video about a software project called
"Barrett Concept Construction" built with IBM Bob and watsonx.ai.

Tone: confident, clear, slightly cinematic. Dark background theme throughout.
Music: minimal ambient electronic, low volume, no vocals, fade out at end.
Voice: calm authoritative male narrator, US English, moderate pace.

The video has 6 scenes:

SCENE 1 (0–10s): HOOK
Show the word "fire" on a black screen. Then show three phrases appearing one by one:
"to fire an employee", "fire in a gun", "fire in her eyes".
Text overlay at the bottom: "Same word. Three different concepts. AI gets it wrong every time."
Narrator says: "The word fire means three completely different things — and an AI gets it
wrong every time. Here's why — and how we fixed it."

SCENE 2 (10–28s): THE PROBLEM
Show the word "fire" splitting into two code-style token chips: "fi" and "##re" on a dark
tech background. Then show a blurred pile of sentence fragments all merging together.
Text overlay: "Token co-occurrence ≠ Concept"
Show a flat bar chart labelled "Average meaning — no context, no goal."
Narrator says: "LLMs don't understand words — they count token fragments. The word fire
becomes fi plus hash-hash-re. Every use of that token blurs together into one flat
distribution. The model has no idea what fire means here, toward this goal.
It's a statistical average — a concept of nothing."

SCENE 3 (28–58s): THE SCIENCE — BARRETT'S MODEL
Split screen. Left side: a human brain icon with three memory cards branching out, each
card showing a different situation and goal. Right side: code editor showing a Python
dataclass called ConceptPopulation.
Animate the three memory cards appearing one by one.
Show a quote card: "A concept is a population of variable instances — Lisa Feldman Barrett,
How Emotions Are Made, 2017"
Then introduce two glowing LLM icons side by side: "Generator LLM (Granite 13B)" in blue
and "Judge LLM (Granite 3B)" in purple.
Text overlay: "We built this missing layer for AI."
Narrator says: "Neuroscientist Lisa Feldman Barrett proved that your brain doesn't store
definitions — it stores a population of instances, each tied to a specific situation and
goal. When you hear fire the employee, your brain picks the right memory. We built exactly
this layer for AI — using two LLMs on watsonx.ai: one to construct simulations, one to
judge them."

SCENE 4 (58–80s): THE ARCHITECTURE
Show an animated flow diagram building step by step on a dark background.
Steps appear in order as narrator speaks:
1. Input box (yellow): "Seed Phrase + Context + Goal"
2. Arrow down to Generator LLM box (blue): "Simulation"
3. Arrow right to Judge LLM box (purple): "Score 0–10 — Functional Adequacy"
4. Arrow with red label: "score too low → refine and re-score (RL inner loop)"
5. Arrow down to Human Feedback box (green): "accept / reject / refine with hint"
6. Arrow right to Concept Population box (gold): "Breadth · Goal Coverage · Frame Coverage"
Small badge at bottom: "Orchestrated end-to-end by IBM Bob"
Narrator says: "A Generator LLM constructs a simulation — a specific prediction of
experience — for a given context and goal. A Judge LLM scores its functional adequacy
from zero to ten. Too low: refined and scored again. Then a human gives a thumbs-up or a
hint. Every accepted instance joins the Concept Population."

SCENE 5 (80–110s): LIVE DEMO
Show a dark terminal window. Text types out line by line as if running live.
Display these lines appearing sequentially:
  $ python -m src.main --term "fire" --seed-phrase "to fire someone"
  Term: fire  |  Seed: "to fire someone"  |  Frame: transitive verb
  Context: "the manager fired her in front of the team"
  Goal: "restore power balance"
  --- Round 0 ---
  Generating simulation...
  Simulation: "The manager terminated her contract publicly, asserting authority..."
  Judge scoring... adequacy_score: 6.2  [below threshold 7.5 — refining]
  --- Round 1 ---
  Refining with hint: "focus on social power dynamics and visibility..."
  Simulation: "By dismissing her publicly, the manager signalled dominance..."
  Judge scoring... adequacy_score: 8.4  [accepted]
  Human review: ACCEPTED
  Instance added to ConceptPopulation.
  Population breadth: 1  |  Goal coverage: 1/1  |  Frame coverage: 1
Callout arrows highlight: "6.2 → below threshold" and "8.4 → accepted"
Narrator says: "Watch it run. Term: fire. Seed phrase: to fire someone. Context: a manager
dismisses an employee in front of the team. Goal: restore power balance. The Generator
produces a simulation. The Judge scores it — six point two. Too low. A hint is injected.
Round two: eight point four. The human accepts. One perfect instance, ready for the
population."

SCENE 6 (110–120s): CLOSING
Dark background. Show three elements centred, fading in one by one:
Line 1: "Generator LLM  +  Judge LLM  =  Concept Population"
Line 2 (smaller, grey): "Not token averages. Goal-indexed, context-grounded instances."
Line 3: IBM Bob logo badge (blue) and watsonx.ai badge (purple) side by side.
Line 4 (small, monospace): "github.com/your-org/barrett-concept-construction"
Narrator says: "Two LLMs. One theory. A population of concepts — not averages.
Built with IBM Bob and watsonx.ai."
Fade music out. Hold last frame for 3 seconds.
```

---

## SCENE-BY-SCENE INVIDEO EDITOR INSTRUCTIONS

Use these after the AI generates the first draft to fine-tune individual scenes.

---

### Scene 1 — Hook (0–10 s)

| Setting | Value |
|---|---|
| Background | Pure black (`#000000`) |
| Stock footage | None — text only |
| Font | Large, white, bold sans-serif, centred |
| Animation | Each phrase fades in and slides up, 0.8 s apart |
| Transition out | Slow fade to Scene 2 |

**On-screen text sequence:**
1. `"to fire an employee"` — white
2. `"to fire a gun"` — white
3. `"fire in her eyes"` — white, word **fire** in `#e07b1e` (amber)
4. Subtitle fade-in: `Same word. Three different concepts.` — grey, smaller

**Voiceover (exact):**
> "The word fire means three completely different things — and an AI gets it wrong every time. Here's why — and how we fixed it."

---

### Scene 2 — The Problem (10–28 s)

| Setting | Value |
|---|---|
| Background | Very dark navy (`#0d0d18`) |
| Stock footage | Abstract data/network particles or circuit board (dark) |
| Overlay opacity | 20% — background should stay dark |
| Transition out | Quick cut to Scene 3 |

**On-screen text sequence:**
1. Large word `fire` in amber — centre screen
2. Arrow down: `LLM tokenises ↓`
3. Two code chips side by side: `fi` (blue) and `##re` (purple)
4. Small text block below: `"to fire someone" · "campfire" · "fired up" · "firing range"` — blurred / faded
5. Bold verdict bar: `Token averaging ≠ Concept construction` — red border, red text

**Voiceover (exact):**
> "LLMs don't understand words — they count token fragments. The word fire becomes fi plus double-hash-re. Every use of that token blurs together into one flat distribution. The model has no idea what fire means here, toward this goal. It's a statistical average — a concept of nothing."

---

### Scene 3 — Barrett's Model (28–58 s)

| Setting | Value |
|---|---|
| Background | Dark blue-black (`#0a0f1a`) |
| Stock footage | Neuroscience / brain scan or abstract neural network — left half |
| Right half | Code / Python editor screenshot |
| Transition out | Crossfade to Scene 4 |

**On-screen text sequence:**
1. Left panel title: `Your brain stores a population` — small caps, grey
2. Three memory cards animate in (left side):
   - Card 1: **Context:** manager + employee · **Goal:** restore power balance
   - Card 2: **Context:** weapons range · **Goal:** discharge safely
   - Card 3: **Context:** emotional intensity · **Goal:** signal passion
3. Quote card (full width): *"A concept is not a definition — it is a population of variable instances, each anchored to a context and a goal."* — italic, purple left border · Attribution: Barrett, 2017
4. Right panel: Two LLM badges appear:
   - `Generator LLM · Granite 13B Instruct` — blue badge
   - `Judge LLM · Granite 3B (fine-tuned)` — purple badge
5. Overlay text: `We built this missing layer for AI.`

**Voiceover (exact):**
> "Neuroscientist Lisa Feldman Barrett proved that your brain doesn't store definitions — it stores a population of instances, each tied to a specific situation and goal. When you hear 'fire the employee', your brain picks the right memory. We built exactly this layer for AI — using two LLMs on watsonx.ai: one to construct simulations, one to judge them."

---

### Scene 4 — Architecture (58–80 s)

| Setting | Value |
|---|---|
| Background | Very dark green-black (`#0a0f0a`) |
| Stock footage | None — diagram only |
| Animation | Each node and arrow builds in as narrator speaks, 1 s delays |
| Transition out | Cut to Scene 5 |

**Flow diagram nodes (build in order):**

```
[Seed Phrase + Context + Goal]  ← yellow box
          ↓
  [Generator LLM · Simulation]  ← blue box
          →
  [Judge LLM · Score 0–10]     ← purple box
          ↓
  score < threshold → REFINE    ← red annotation
          ↓
  [Human Feedback: ✓/✗/hint]   ← green box
          →
  [Concept Population]          ← gold box
```

**Badge at bottom:** `Orchestrated end-to-end by IBM Bob` — blue pill

**Voiceover (exact):**
> "A Generator LLM constructs a simulation — a specific prediction of experience — for a given context and goal. A Judge LLM scores its functional adequacy from zero to ten. Too low: refined and scored again. Then a human gives a thumbs-up or a hint. Every accepted instance joins the Concept Population."

---

### Scene 5 — Live Demo (80–110 s)

| Setting | Value |
|---|---|
| Background | Terminal dark (`#0d1117`) — GitHub-style dark theme |
| Stock footage | None — terminal text only |
| Font | Monospace, white/green on dark, 15 pt |
| Animation | Lines type in one at a time, ~0.5 s between lines |
| Callout arrows | Highlight `6.2` (red) and `8.4` (green) with animated arrows |
| Transition out | Slow fade to Scene 6 |

**Terminal text to display (type-in animation, line by line):**

```
$ python -m src.main --term "fire" \
    --seed-phrase "to fire someone" \
    --grammatical-frame "transitive verb, agent=manager, patient=employee" \
    --context "the manager fired her in front of the team" \
    --goal "restore power balance" \
    --max-iterations 3 --threshold 7.5

Term         : fire
Seed phrase  : to fire someone
Frame        : transitive verb, agent=manager, patient=employee
Context      : the manager fired her in front of the team
Goal         : restore power balance

──── Round 0 ────────────────────────────────────────────
Generating simulation via watsonx.ai (Granite 13B)...
Simulation : "The manager terminated her contract publicly,
             asserting authority over the team dynamic..."
Judge score: 6.2  ◀ below threshold (7.5) — refining

──── Round 1 ────────────────────────────────────────────
Refining simulation with hint: focus on social power dynamics...
Simulation : "By dismissing her in front of colleagues, the
             manager publicly reasserted dominance, restoring
             the chain of command visibly..."
Judge score: 8.4  ✓ accepted

Human review: ACCEPTED
Instance added to ConceptPopulation.
Population breadth : 1   Goal coverage : 1/1   Frame coverage : 1
Report written → docs/concept_population_report.md
```

**Callout overlays:**
- Red arrow + label on `6.2`: `"Below threshold — RL refines"`
- Green arrow + label on `8.4`: `"Accepted — joins population"`

**Voiceover (exact):**
> "Watch it run. Term: fire. Seed phrase: to fire someone. Context: a manager dismisses an employee in front of the team. Goal: restore power balance. The Generator produces a simulation. The Judge scores it — six point two. Too low. A hint is injected. Round two: eight point four. The human accepts. One perfect instance, ready for the population."

---

### Scene 6 — Closing (110–120 s)

| Setting | Value |
|---|---|
| Background | Near-black (`#08080f`) |
| Stock footage | None — text and badges only |
| Animation | Each element fades up with 0.6 s delay between |
| Music | Fade to silence over last 5 s |
| Hold | Freeze final frame for 3 s before end |

**On-screen text sequence:**
1. `Generator LLM  +  Judge LLM  =  Concept Population`
   — Generator in blue `#60a5fa`, Judge in purple `#c084fc`, result in gold `#fcd34d`
2. Subtitle (grey, smaller): `Not token averages. Goal-indexed, context-grounded instances.`
3. Two badges side by side: `IBM Bob` (blue) · `watsonx.ai` (purple)
4. Repo line (small monospace, dim): `github.com/your-org/barrett-concept-construction`

**Voiceover (exact):**
> "Two LLMs. One theory. A population of concepts — not averages. Built with IBM Bob and watsonx.ai."

---

## FULL NARRATION — CLEAN READ-ALOUD VERSION

*Copy this into invideo.ai's voiceover text field if using its built-in TTS:*

Scene 1: The word fire means three completely different things — and an AI gets it wrong every time. Here's why — and how we fixed it.

Scene 2: LLMs don't understand words — they count token fragments. The word fire becomes fi plus double-hash-re. Every use of that token blurs together into one flat distribution. The model has no idea what fire means here, toward this goal. It's a statistical average — a concept of nothing.

Scene 3: Neuroscientist Lisa Feldman Barrett proved that your brain doesn't store definitions — it stores a population of instances, each tied to a specific situation and goal. When you hear fire the employee, your brain picks the right memory. We built exactly this layer for AI — using two LLMs on watsonx.ai: one to construct simulations, one to judge them.

Scene 4: A Generator LLM constructs a simulation — a specific prediction of experience — for a given context and goal. A Judge LLM scores its functional adequacy from zero to ten. Too low: refined and scored again. Then a human gives a thumbs-up or a hint. Every accepted instance joins the Concept Population.

Scene 5: Watch it run. Term: fire. Seed phrase: to fire someone. Context: a manager dismisses an employee in front of the team. Goal: restore power balance. The Generator produces a simulation. The Judge scores it — six point two. Too low. A hint is injected. Round two: eight point four. The human accepts. One perfect instance, ready for the population.

Scene 6: Two LLMs. One theory. A population of concepts — not averages. Built with IBM Bob and watsonx.ai.

---

## TIMING SUMMARY

| Scene | Time | Duration | Voiceover words |
|---|---|---|---|
| 1 — Hook | 0–10 s | 10 s | 25 |
| 2 — Problem | 10–28 s | 18 s | 50 |
| 3 — Barrett | 28–58 s | 30 s | 60 |
| 4 — Architecture | 58–80 s | 22 s | 55 |
| 5 — Live Demo | 80–110 s | 30 s | 70 |
| 6 — Closing | 110–120 s | 10 s | 18 |
| **Total** | | **120 s** | **~278 words** |

Narration pace: ~140 words/min. Remaining time in each scene is visual animation.

---

## STOCK MEDIA KEYWORDS (for invideo.ai media search)

| Scene | Search terms |
|---|---|
| 2 | `data network dark`, `neural network abstract`, `circuit board dark background` |
| 3 | `human brain neurons`, `memory recall abstract`, `cognitive science visualization` |
| 4 | `workflow diagram`, `AI pipeline`, `system architecture dark` |
| 5 | `terminal coding dark`, `developer typing`, `code screen night` |

---

## POST-GENERATION CHECKLIST

- [ ] Replace `github.com/your-org/barrett-concept-construction` with real repo URL in Scene 6
- [ ] Review auto-selected stock clips — replace any that show chatbots or generic "AI robot" imagery
- [ ] Check TTS pronunciation of: `Granite` (GRAN-ite), `Barrett` (BARE-ett), `watsonx` (watson-ex)
- [ ] Verify terminal font in Scene 5 is monospace and readable at full screen
- [ ] Export at 1920×1080, H.264, for YouTube upload
- [ ] Upload as unlisted YouTube video and paste URL into `docs/video-url.md`
