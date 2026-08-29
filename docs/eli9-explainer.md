# The Project Explained Like You're Nine

> A friendly, jargon-free guide to what this project does and why it matters.

---

## Words Don't Mean Just One Thing

Imagine the word **"fire"**. 🔥

- "My dad had to **fire** someone at work." *(kick them out)*
- "Look at the **fire** in the fireplace!" *(burning flames)*
- "That song **fires me up!**" *(excitement)*

Same word. Totally different meanings. **You** understand which one is right because you know what's happening *around* the word.

---

## The Problem: Computers Are Bad at This

A computer program called an **LLM** (like a really smart autocomplete) has read millions of books. But it doesn't *really understand* words — it just remembers **which words appear next to other words a lot**.

So when you ask it about "fire", it might pick the wrong meaning for the moment — because it has no idea **what you are trying to do** or **what is going on around you**.

> Think of it like a robot that memorised every page of every dictionary, but has never actually seen a campfire, been to a job interview, or heard a hype song. It *knows* the words — but not the *feeling* behind them.

---

## The Science Part: A Brain Scientist Named Barrett

A scientist named **Lisa Feldman Barrett** studied how *your brain* actually understands words. She found something cool:

> Your brain doesn't store one single definition for a word. It stores **a whole collection of memories** of that word being used — each one in a different situation, for a different reason.

When you hear "fire" at a campfire, your brain doesn't look up a definition — it **picks the right memory** from its collection that best fits what is happening right now.

That is what this project tries to teach a computer to do!

---

## The Solution: Building a Word's "Family Album"

Instead of giving the computer one definition for a word, this project builds a **"family album" of examples** — called a **Concept Population**.

Each photo in the album shows:
- 📍 **The situation** (where/when the word is used)
- 🎯 **The goal** (what you are trying to do)
- 💭 **What it means in that exact moment**
- ⭐ **A score** (how good is this example for this situation?)

---

## The Loop: Learning Like You Do in School

The computer goes through a cycle — like practising spelling until you get it right:

```
1. 🤖 Computer makes a guess about what the word means in a situation
2. ⚖️  A "judge" gives it a score (0–10) — "how good was that guess?"
3. ❌  If the score is low, it tries again with hints
4. 🙋 A human also says: "Yes, that's right!" or "Nope, try again!"
5. ✅  When it gets enough good examples, it's done!
```

This keep-trying-until-you-get-it-right process is called **Reinforcement Learning** — the same idea used to train robots and chess computers!

---

## The Tools

| Tool | What it does |
|---|---|
| 🧰 **IBM Bob** | The smart helper that runs everything and writes the code |
| 🌐 **watsonx.ai** | IBM's AI brain that generates and scores the word examples |
| 🐍 **Python** | The programming language all the code is written in |

---

## What Comes Out at the End?

A **Concept Population Report** — like a report card for a word! It shows:

- How many different situations the word now understands 📊
- Which goals it covers 🎯
- Its best examples per situation 🏆

---

## In One Sentence

> This project teaches an AI to understand words **the way your brain does** — not as dictionary definitions, but as a collection of real situations, each with a purpose — using IBM Bob and watsonx.ai to build and score that collection automatically.

---

## Estimated Training Effort with the Current Base Model

The project runs on **IBM Granite 13B Instruct v2** (`ibm/granite-13b-instruct-v2`) — a 13-billion-parameter instruction-tuned language model. Here is what "building a Concept Population" actually costs in practice.

### Per-Term Estimate (a single word with N context/goal pairs)

Each (context, goal) pair drives two types of LLM calls:

| Call type | Purpose | Tokens per call (approx.) |
|---|---|---|
| **Generate** | Produce a candidate simulation | ~300 prompt + ~80 output = ~380 |
| **Score** | Judge functional adequacy (0–10) | ~250 prompt + ~10 output = ~260 |
| **Refine** (if score < threshold) | Produce an improved simulation | ~380 prompt + ~80 output = ~460 |

With `--max-iterations 3` and `--threshold 7.5`, a single (context, goal) pair costs roughly:

```
Round 0: 1 generate + 1 score         ≈  640 tokens
Round 1: 1 refine   + 1 score         ≈  720 tokens   (if score < 7.5)
Round 2: 1 refine   + 1 score         ≈  720 tokens   (if still < 7.5)
─────────────────────────────────────────────────────
Worst-case per pair                   ≈ 2 080 tokens
Typical (1–2 refinements on average)  ≈ 1 350 tokens
```

### Scaling to a Full Session

| Scenario | Context/goal pairs | Est. total tokens | Est. API time* |
|---|---|---|---|
| Quick demo (e.g. stub smoke-test) | 1 | ~1 400 | < 1 s (stub) |
| Notebook demo ("anger", 3 pairs) | 3 | ~4 000 | ~15–25 s |
| Standard term audit (5 pairs) | 5 | ~6 750 | ~25–40 s |
| Broad coverage run (10 pairs) | 10 | ~13 500 | ~50–80 s |
| Vocabulary study (50 terms × 5 pairs) | 250 | ~337 500 | ~20–35 min |

\* *Live API latency for Granite 13B on watsonx.ai us-south, assuming no rate-limit back-off. Stub mode (`WATSONX_STUB=true`) is instant and consumes zero tokens.*

### Why Granite 13B Is a Good Fit Here

- **Instruction-following**: the judge and refiner prompts use direct imperative framing ("Rate…", "Rewrite…") — a format Granite 13B was fine-tuned to follow reliably.
- **Short-generation quality**: simulations are capped at ~60 words; 13B-scale models produce coherent short text at low latency without the overhead of larger models.
- **Scoring consistency**: score calls are capped at 32 tokens with `temperature=0.2` — Granite 13B returns stable numeric outputs in this regime.
- **Cost efficiency**: at 13B parameters, token cost is significantly lower than 70B+ alternatives while remaining adequate for concept-population breadth (not deep reasoning).

### Limitations to Keep in Mind

- **Not fine-tuned on Barrett vocabulary**: the model has no awareness of "concept population" or "functional adequacy" as technical terms — the prompts must carry all grounding context, adding ~80–120 tokens per call.
- **Score variance**: even at low temperature, score calls can drift ±0.5 between identical prompts; the threshold should be set with a ~0.5 tolerance margin (e.g. target 7.5, accept ≥ 7.0).
- **Context window (4 096 tokens)**: for very long contexts or large hint chains, the cumulative prompt can approach the limit. Keep `simulation` fields under 80 words and `hint` under 40 words.
- **No actual weight updates**: this workflow builds the population at the *application layer* — it does **not** fine-tune or retrain Granite 13B. All learning is stored in the `ConceptPopulation` JSON, not in the model weights.
