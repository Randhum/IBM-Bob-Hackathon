# Concept Ontology
## Shared Language for the Concept-Population Workflow

*This document is the team's shared brain. Every line of code, every sentence of narration,
every judge-facing statement must be traceable to a term defined here.*

*Theoretical anchor: Lisa Feldman Barrett, "How Emotions Are Made" (2017) — Theory of
Constructed Emotion / Conceptual Act Theory.*

---

## 1. The Core Problem — Two Levels of Failure

### Level 1 — The Tokenization Problem

Current LLMs do not process language at the word level. They process **tokens** — sub-word
fragments produced by algorithms like BPE (Byte-Pair Encoding) or WordPiece. The word
`"anger"` may become `["ang", "##er"]`. The word `"misunderstanding"` becomes
`["mis", "##under", "##stand", "##ing"]`. The phrase `"fire the employee"` becomes a
sequence of token IDs that share no structural relationship with `"fire the gun"` except
statistical co-occurrence in training data.

**What is stored in the weights:** conditional probability distributions over token
sequences. Not meanings. Not concepts. Pattern frequencies.

**The consequence:** an LLM can produce a fluent sentence about "anger" without having
a concept of anger at all — it has a high-probability token sequence generator calibrated
to contexts where the token cluster `["ang", "##er"]` appears in training text.

### Level 2 — The Grammar Problem

Even if we move from tokens to whole words, we face a second, deeper failure: the same
word constructs entirely different concepts depending on its **grammatical role and
combinatorial context**.

| Surface form | Grammatical role | Concept constructed |
|---|---|---|
| `"fire the employee"` | verb + object | termination of employment |
| `"fire the gun"` | verb + object | discharge of a weapon |
| `"the fire burns"` | noun + verb | combustion event |
| `"under fire"` | prepositional phrase | under attack / intense scrutiny |
| `"fire in her eyes"` | metaphorical noun | intense emotional energy |

A token-weight system treats these as variants of the same high-frequency token. A
concept-construction system must treat them as **five distinct concept instances**, each
assembled through a different grammatical construction applied to a shared phonological form.

**Grammar is not decoration.** It is the machinery of concept construction. Subject-verb-
object structure, tense, modality (`"might"`, `"should"`, `"could"`, `"used to"`), aspect
(`"is firing"` vs `"fired"` vs `"will fire"`), and prepositional framing all participate
in determining which concept is being constructed in any given utterance.

---

## 2. Barrett's Framework — Precise Definitions

### 2.1 Concept

> A concept is a **population of variable instances** — not a single definition, not a
> prototype, not a necessary-and-sufficient feature list.

A concept is the brain's learned statistical model of a category of experience. It is:
- **Population-based:** represented as a family of past instances, not an essence.
- **Goal-indexed:** instances are clustered by *what they were useful for*, not by
  perceptual similarity. The concept "chair" groups together bar stools, thrones, and
  tree stumps because all were used for the goal of *sitting*, not because they look alike.
- **Predictive:** the brain deploys a concept *before* sensory input arrives, as a
  forward prediction. Concepts are used to simulate the world, not merely label it.
- **Constructed on the fly:** each use of a concept is a fresh construction from the
  available population, calibrated to the current context and goal.

**What a concept is NOT:**
- A dictionary entry.
- A fixed semantic vector in an embedding space.
- A token or token cluster.
- A prototype or average instance.

### 2.2 Instance

> A **concept instance** is one specific realization of a concept — a particular
> simulation generated for a particular context toward a particular goal.

An instance is what the brain actually uses. It is always specific:
- `"anger"` as concept → *abstract, undefined*
- `"anger"` instance in context `"receiving unfair criticism at work"` toward goal
  `"restore social fairness"` → *a specific predicted experience: chest tightening,
  sharp verbal response, urge to challenge the critic publicly*

Instances are the granular units of the ConceptPopulation.

### 2.3 Simulation

> A **simulation** is the brain's active prediction of what will happen — sensorily,
> motorically, emotionally — if a particular concept instance is deployed in the
> current context.

A simulation is forward-looking and embodied. It is not a description of the past.
It is the brain running a model: *"if this is anger in this context, what does that
predict about my next moment?"*

In our system, a simulation is the LLM-generated prediction of the experience, behavior,
or response that a concept instance would produce in a given context toward a given goal.

**Vocabulary rule:** we always say *simulation*, never *definition* or *description*.

### 2.4 Context

> **Context** is the specific situational frame that makes a particular concept instance
> appropriate.

Context is not vague background. It is a precise situational specification:
- ✅ `"receiving unfair criticism from a manager in a performance review"`
- ❌ `"a work situation"`

Context determines which instance in the population is the right prediction.

### 2.5 Goal

> **Goal** is the functional purpose the concept instance is serving — what the organism
> is trying to achieve or maintain in this context.

Goals are what cluster instances into a concept (Barrett's key insight). Two experiences
that look nothing alike — a clenched jaw and a passive-aggressive email — belong to the
same concept instance of "anger" because both serve the same goal: *restoring a perceived
violation of fairness*.

**In our system:** goal is an explicit field, not inferred. Every instance has one.

### 2.6 Functional Adequacy

> **Functional adequacy** is the measure of how well a simulation serves the goal in the
> given context.

This replaces "correctness" and "clarity" entirely. A simulation is not correct or
incorrect in the abstract — it is more or less adequate for the goal in this context.

Scored 0–10. High adequacy = the simulation accurately predicts the experience/behavior
this concept would produce here, in a way that serves this goal.

### 2.7 Prediction Error

> **Prediction error** is what drives concept learning — when a simulation fails (the
> predicted experience does not match the actual one), the concept population is updated.

In Barrett's brain: prediction error propagates upward through the cortical hierarchy,
triggering a revision of the concept that generated the bad prediction.

In our system: prediction error is operationalized as **low adequacy score + human
rejection signal**. These trigger refinement of the instance (mutation) or addition of
a new instance (population growth).

### 2.8 Construction

> **Concept construction** is the active, on-the-fly process of selecting and assembling
> a concept instance appropriate to the current context and goal.

This is Barrett's most radical claim: the brain does not retrieve a stored emotion/concept.
It *builds* one, freshly, each time, from the available population of past instances.

**In our system:** construction is the moment the workflow selects which concept frame
applies to a given input — *before* generating a simulation. This is the step that most
LLM pipelines skip entirely.

---

## 3. The Vocabulary / Tokenization Layer

### 3.1 Why Tokenization is the Root Failure

The mismatch between tokens and concepts is not a minor technical detail — it is the
**architectural source** of why LLMs hallucinate, produce context-blind outputs, and
fail at nuanced reasoning.

Token-level encoding means:
- `"fire"` (combustion), `"fire"` (dismiss), `"fire"` (discharge), `"fire"` (inspiration)
  share weight space because they share a token ID.
- The model's "knowledge" of "fire" is the statistical average of all these uses —
  a concept of nothing in particular.
- Grammatical structure (which disambiguates these) is encoded implicitly through
  attention patterns, but never as an explicit **concept construction rule**.

### 3.2 Concept Construction from Words, Not Tokens

Our workflow operates at the **word + grammar** level. Concept construction requires:

1. **Whole-word units** — not sub-word tokens. The concept vehicle is the word, not the fragment.

2. **Grammatical role tagging** — the same word constructs different concepts depending on:
   - **Part of speech:** noun vs verb vs adjective (`"fire"` as noun vs verb)
   - **Argument structure:** what is the subject, verb, object? (`"he fired her"` vs
     `"she fired him"` — same tokens, different concept instances if power dynamics matter)
   - **Modality:** `"he fired"` vs `"he might fire"` vs `"he should have fired"` —
     different temporal and epistemic framings, potentially different concept instances
   - **Aspect:** `"firing"` (ongoing) vs `"fired"` (completed) vs `"will fire"` (anticipated)

3. **Combinatorial sensitivity** — concept construction is compositional but not
   strictly compositional. `"cold shoulder"` does not construct from `"cold"` + `"shoulder"`.
   The system must handle idiomatic and metaphorical constructions as atomic units.

### 3.3 Implications for the Workflow

| Requirement | What it means in practice |
|---|---|
| Input is words, not tokens | The seed term must be provided as a grammatically framed phrase, not a raw word. `"to fire (someone)"` not `"fire"`. |
| Context specifies grammatical frame | The context field must include enough grammatical structure to disambiguate. `"the manager fired her"` not `"a workplace event"`. |
| Goal disambiguates construction | The goal field resolves ambiguity where grammar alone cannot. Same words, different goals → different concept instances. |
| Same word ≠ same concept | The workflow must treat `"fire"` in two different (context, goal) pairs as potentially different concepts, not as instances of the same concept. |
| Grammar is an input, not an output | The system does not infer grammatical role — it requires it to be specified in the context. This is a design choice, not a limitation. |

### 3.4 Vocabulary Granularity Levels

Our system recognizes three levels at which language constructs concepts:

```
Level 1 — Token      "ang", "##er"         NO concept
Level 2 — Word       "anger"               POTENTIAL concept seed (polysemous)
Level 3 — Phrase     "anger at injustice"  CONCEPT FRAME (grammatically grounded)
Level 4 — Instance   "anger at injustice   CONCEPT INSTANCE (fully specified)
                      in context X
                      toward goal Y"
```

Our workflow operates at **Level 3 → Level 4**. It explicitly rejects Level 1 and Level 2
as insufficient for concept construction.

---

## 4. The ConceptPopulation Data Model — Annotated

```python
@dataclass
class ConceptInstance:
    id: str                    # UUID — each instance is unique
    # --- Grammatical / construction fields ---
    seed_phrase: str           # grammatically framed seed, e.g. "to fire (someone)"
    grammatical_frame: str     # e.g. "transitive verb, agent=manager, patient=employee"
    # --- Barrett fields ---
    context: str               # specific situational frame (full sentence minimum)
    goal: str                  # functional purpose being served
    simulation: str            # predicted experience/behavior/response (≤60 words)
    # --- Learning fields ---
    adequacy_score: float      # 0-10, functional adequacy in this context/goal
    initial_score: float       # score at round 0, for delta computation
    human_signal: str          # "accept" | "reject" | "refine"
    hint: str | None           # human correction hint if rejected/refined
    round: int                 # which iteration produced this simulation
    # --- Metadata ---
    timestamp: str             # ISO 8601

@dataclass
class ConceptPopulation:
    term: str                  # the word/phrase being conceptualized
    seed_phrase: str           # the grammatically framed seed form
    instances: list[ConceptInstance]
    goal_coverage: list[str]   # distinct goals covered by accepted instances
    context_coverage: list[str]# distinct contexts covered
    grammatical_frames: list[str] # distinct grammatical constructions covered
    population_breadth: int    # count of accepted instances
```

**New field: `grammatical_frame`** — this is what enables the workflow to distinguish
`"fire"` (dismiss) from `"fire"` (discharge) from `"fire"` (inspire). Without it, the
population collapses the tokenization problem back into the concept level.

---

## 5. Canonical Vocabulary — Enforced Across All Files

| Use this | Never use this | Why |
|---|---|---|
| simulation | definition | a definition is static; a simulation is predictive |
| instance | example | instances are population members; examples are pedagogical |
| functional adequacy | correctness, accuracy, clarity | adequacy is relational (to goal+context); correctness implies ground truth |
| concept construction | concept retrieval, concept lookup | Barrett: concepts are built, not fetched |
| prediction error | mistake, wrong answer | prediction error is a learning signal, not a failure |
| grammatical frame | syntax, grammar tag | frame captures both structure and semantic role |
| seed phrase | term, keyword, query | seed phrase is grammatically grounded from the start |
| population breadth | coverage, completeness | breadth is a population property, not a checklist |
| contextual fit | relevance, accuracy | fit is a relational adequacy judgment |

---

## 6. The One-Sentence Project Description

> We build a workflow that teaches an LLM what a concept is — not as a dictionary entry,
> but as Barrett describes: a population of goal-indexed, grammatically-grounded simulations,
> constructed on the fly from words (not tokens), refined through prediction error and human
> feedback.

This sentence is the north star. Every design decision should be traceable to it.
