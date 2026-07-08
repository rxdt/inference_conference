# Intent → Model: matching a plain-English goal to the right AI model

**In one sentence:** you type what you want to build ("I need something to detect
tumors in X-rays"), and this project tries to hand you the best-fitting model from a
catalog of ~13,000 — using only classical [NLP](https://en.wikipedia.org/wiki/Natural_language_processing)
(text-processing) with [spaCy](https://spacy.io/), and **no** [large language models](https://en.wikipedia.org/wiki/Large_language_model)
(no ChatGPT-style AI).

This README is written for a newcomer. It explains what we built, how well it works
(honestly — including where it *doesn't*), the mistakes we hit, and exactly how to
run it yourself.

> **📚 For a reviewer — what to read, in order:**
> 1. **`README.md`** (this file) — the plain-English overview + results.
> 2. **`EXPERIMENTS_V2.md`**, **`EXPERIMENTS_V3.md`** — the real lab notebooks (every
>    number regenerated from code; every failure logged).
> 3. **`RESEARCH_NOTES_V2.md`** / **`_V3.md`** — problem framing + verified data truths.
> 4. **`STATE.md`** — one-page current status. **`PRESENTATION.md`** / `slides/` — the talk.
>
> **Archived / do-not-cite (clearly banner-marked at their tops):** `EXPERIMENTS.md`,
> `RESEARCH_NOTES.md` (V1, pre-reframe), and `HANDOFF.md` (internal agent log). `PROMPT.md`
> and `config.py` are owner-authored inputs, not our results.

> ✅ **HOLDOUT REFRESHED (2026-07-03):** the held-out set was updated to Hugging-Face-native
> task labels (which are essentially *our* labels), so we corrected the mapping (now
> near-lossless: 109/111 rows scored) and re-ran once. Current holdout: **modality-match
> 0.50, task-in-list@1 0.284** (see §3). The old 0.41 (on a hand-filtered 66-row subset)
> is retired.

---

## 1. What problem is this?

Imagine an app store, but for AI models (the ~13k catalog is
[Hugging Face](https://huggingface.co/models)–style, each entry with a
[model card](https://huggingface.co/docs/hub/model-cards) — a free-text description). A
user describes a goal in everyday words. We want to return the model that best serves
that goal.

The important insight: **this is not classification.** The user isn't giving us a
label like `robotics`; they're describing a *need*. "Something to build robots with"
could mean control code, perception, reasoning, or all three. So the real task is
**[semantic retrieval](https://en.wikipedia.org/wiki/Information_retrieval)** — searching
by *meaning* rather than exact words: reading the model descriptions and finding the one
whose capabilities *match the intent*.

**Self-imposed rule:** spaCy's medium English model ([`en_core_web_md`](https://spacy.io/models/en#en_core_web_md))
+ Python builtins only. No GPT, no [embedding APIs](https://en.wikipedia.org/wiki/Word_embedding)
(no paid "turn text into vectors" services), no internet at runtime. Partly a deployment
constraint, partly an honest scientific question: *how far does classical NLP get you?*

---

## The story, start to finish (how we got here)

1. **The lab decided to** treat this as *retrieval*, not classification — match the
   user's goal against each model's description by meaning, spaCy-only.
2. **We set up proto-experiments** to see what signal even exists: baseline task
   predictors (majority-class, random, keyword/[IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
   overlap) and a whole-document word-vector matcher.
3. **We tried ~13 approaches** across two rounds: whole-doc vectors (centroid &
   [nearest-neighbor](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)),
   [fuzzy matching](https://en.wikipedia.org/wiki/Approximate_string_matching),
   [NER](https://en.wikipedia.org/wiki/Named-entity_recognition) gazetteers, a trained
   [text classifier](https://en.wikipedia.org/wiki/Document_classification),
   [dependency-parse](https://en.wikipedia.org/wiki/Dependency_grammar) subject-verb-object,
   clause splitting, part-of-speech weighting, and — the big bet — **injecting extra
   vocabulary** into spaCy's word vectors. Plus the hybrid retriever itself.
4. **We found real issues:** whole-doc vectors collapse across the prompt↔card style gap
   (0.69 → 0.07); a [data-leak](https://en.wikipedia.org/wiki/Leakage_(machine_learning))
   where a setting had been tuned on test prompts (removed); a tokenizer trap that made
   our first injection silently do nothing; and — once fixed — injection actually *hurt*
   (0.98 → 0.775). None of the alternatives beat the hybrid retriever.
5. **We settled on Z = the `clean_pick_v2` hybrid retriever** as the tolerable result,
   with these five numbers: **(1)** home satisfy@1 = **0.1184**, **(2)** vs a fair
   baseline of **0.0695** = **(3) ~1.70× lift**, **(4)** held-out modality-match =
   **0.50**, **(5)** held-out task-in-list = **0.284** (n=109). Modest but real, honestly bounded,
   independently verified — and a clear map of what does *not* work.

---

## 2. How it works (the approach, plainly)

The shipped system is called **`clean_pick_v2`**. It's a two-stage retriever:

**In one sentence:** understand the prompt, cheaply shortlist ~200 models by keyword
overlap, then re-rank that shortlist by phrase-level meaning similarity (plus soft task
and constraint checks) and return the top few.

```mermaid
flowchart LR
    P([Your prompt]) --> U["1 - Understand<br/>lemmas, verb+object,<br/>hard constraints"]
    U --> S["2 - Sparse retrieve<br/>IDF keyword overlap<br/>over all 13k models"]
    S --> C["Top ~200 candidates"]
    C --> D["3 - Dense rerank<br/>phrase-vs-phrase similarity<br/>+ soft task bonus<br/>+ constraint fit"]
    D --> R([Top model(s)])
    U -.->|no clear intent| A([Abstain])
```

1. **Understand the prompt.** Pull out the meaningful words (the
   [lemmas](https://en.wikipedia.org/wiki/Lemma_(morphology)) — dictionary forms, so
   "building" and "built" both count as "build"), the "verb + object" (what you want to
   *do* to *what*), and any hard constraints ("open source", "runs on a laptop",
   "long context").
2. **Narrow down (["sparse retrieval"](https://en.wikipedia.org/wiki/Information_retrieval)).**
   Cheaply score all 13k models by keyword overlap and keep the top ~200 — using
   [TF-IDF/IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf), which just means rare,
   informative words count more than common ones. Fast, catches obvious matches.
3. **Rerank ("dense").** Compare the *phrases* of your prompt to the *phrases* of each
   model card using [word-vector](https://en.wikipedia.org/wiki/Word_embedding)
   similarity — word vectors turn words into points in space so that similar meanings
   sit close together, and we measure closeness with
   [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity). Crucially we do
   this **phrase by phrase, never averaging a whole document** (see Issues). Add a small
   bonus if the predicted task matches, and check the hard constraints. Popularity is
   deliberately **not** used — a model being popular doesn't mean it fits *you*.

> **Do we use [RRF](https://plg.uwaterloo.ca/~gvcormack/cormacksigir09-rrf.pdf) (Reciprocal Rank Fusion)?**
> Only *inside* the small "predicted task" helper (it fuses three weak task guessers,
> `k=60`). The **main** retrieve → rerank pipeline above does **not** use RRF — it scores
> and sorts directly. So: RRF is a sub-component, not the top-level algorithm.

If your request is too vague to act on, it **abstains** instead of guessing.

---

## 3. Results (honest, with confidence intervals)

There is **no "correct answer key"** — nobody labeled the single best model per
prompt. So we can't claim we return *the* best model. Instead we measure **coverage**:
does the returned model's task / domain / specialty / constraints match what the
prompt asked for? We always report this against a **fair baseline** (a random model
from the plausible set) so you know if we're actually beating luck. Every score comes
with a [95% confidence interval](https://en.wikipedia.org/wiki/Confidence_interval)
(a range we're fairly sure the true value falls in), computed by
[bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)) — re-sampling
the data many times to see how much the number wobbles.

### Home test (1,537 prompts)

> **What "home" means:** our *development* set — the prompts we built and checked the
> system against (like practicing on questions you've seen). Numbers here can flatter
> you, so we also run a **held-out** test below (fresh questions the system never saw) —
> the real measure of whether it [generalizes](https://en.wikipedia.org/wiki/Generalization_(machine_learning)).

| what we measured | score | 95% confidence | plain meaning |
|---|---|---|---|
| **satisfy@1** | **0.1184** | [0.103, 0.135] | top pick fully matches ~12% of the time |
| satisfy@k (top 5) | 0.1809 | [0.162, 0.201] | right answer in top 5 ~18% |
| fair baseline | 0.0695 | — | random-in-plausible gets ~7% |
| **lift over baseline** | **~1.70×** | — | we roughly *1.7× beat* luck |

**Is 12% good?** It's low in absolute terms — but the bar is strict (the top pick must
match task **and** domain **and** specialty **and** constraints, all at once, with no
answer key and noisy labels). The honest headline is the **1.7× lift over a fair
baseline**, independently reproduced.

### Held-out test (111 real human-written project descriptions, never seen during development)

These use Hugging-Face-native task labels — essentially *our* labels — so the mapping is
near-lossless (109 of 111 rows scored; 2 dropped as tasks with no model type in the
catalog). Mapping was **frozen before scoring** (no peeking, single run, no tuning).

| what we measured | score | 95% CI | note |
|---|---|---|---|
| **modality-match@1** | **0.50** | [0.41, 0.59] | right *modality* (image vs text vs audio…) on top, ~half the time |
| task-in-list@1 | **0.284** | [0.20, 0.37] | top pick's task is acceptable ~28% (31 of 109) |
| task-in-list@5 | 0.358 | — | acceptable task in top 5 ~36% |
| unservable rows | 2 of 111 | — | anomaly-detection, recommendation — no such model type; counted, not hidden |

**How to read this:** the holdout checks the **task only**, not the full 4-part match
that home's 0.12 requires — so 0.284 vs home's ~0.20 task facet is a *reasonable, honest*
match, **not** a contradiction and **not** an improvement. Modality 0.50 says the same
thing bluntly: even the coarse type is right only half the time. **This problem is hard,
and the fresh holdout confirms it rather than inflating it.**

---

## 4. What we tried that *didn't* work (the negatives are the point)

Good science reports failures. Every one below was implemented fully and **rejected**
because none beat the hybrid retriever:

- **Whole-document word vectors** (averaging a whole model card into one vector):
  scored 0.69 inside the catalog but **0.07 on real prompts** — it doesn't survive the
  gap between long formal model cards and short casual prompts. This is *why* we rerank
  phrase-by-phrase instead.
- **[Fuzzy matching](https://en.wikipedia.org/wiki/Approximate_string_matching)
  (tolerating typos/variants), [named-entity](https://en.wikipedia.org/wiki/Named-entity_recognition)
  gazetteers (spotting known names/terms), a trained
  [text classifier](https://en.wikipedia.org/wiki/Document_classification),
  [dependency parsing](https://en.wikipedia.org/wiki/Dependency_grammar) (grammar
  structure), clause splitting, [part-of-speech](https://en.wikipedia.org/wiki/Part-of-speech_tagging)
  weighting** — all rejected.
- **Teaching spaCy new words** (our big "maybe this fixes it" idea). Some terms, like
  `text-generation`, are not in spaCy's dictionary, so spaCy has no
  [word vector](https://en.wikipedia.org/wiki/Word_embedding) (no numeric "meaning") for
  them. We guessed *that* was why matching failed, so we built vectors for 1,943 missing
  terms and added them. It made things **worse**, not better. Here is the plain story:

  - We ran a simple check: *give a model its own description as the search — does it find
    itself first?* With normal spaCy it does, almost always: **98 out of 100**.
  - After adding our home-made word vectors, that dropped to **about 77 out of 100**.
  - Why? Our home-made vectors were lower quality than spaCy's real ones, so instead of
    helping, they *dragged the matches in the wrong direction*.
  - **The real lesson:** the missing words were never the problem. The problem is that
    **users and model descriptions are written in totally different styles** — a short
    casual request ("something to spot tumors") vs. a long technical write-up. Adding
    words cannot fix a style mismatch. No amount of new vocabulary bridges that gap.

---

## 5. Issues & mistakes we hit (and caught)

We kept a paper trail of our own errors, because catching them is the work:

- **A tuning leak (V1):** a setting had been secretly tuned on the *test* prompts —
  like grading your own exam after seeing the answers ([data leakage](https://en.wikipedia.org/wiki/Leakage_(machine_learning))).
  Found via adversarial review, removed; V2 never looks at test prompts while tuning
  ("prompt-blind").
- **A [tokenizer](https://en.wikipedia.org/wiki/Lexical_analysis#Tokenization) trap:**
  the tokenizer (the step that splits text into words) breaks `text-generation` into
  `text` / `-` / `generation`, so our first injection round silently did nothing. We
  caught it with a sanity check and fixed it by merging those pieces back into one token.
- **A file-clobbering bug** in a batch script overwrote some baseline result files,
  briefly making a score look like 0.1041 when the true value was 0.1184. Diagnosed and
  corrected; the bug is fixed.
- **[Determinism](https://en.wikipedia.org/wiki/Deterministic_algorithm) & verification:**
  results are identical every run (no randomness sneaking in), and an independent agent
  plus an adversarial code reviewer re-checked the numbers and the code.

None of these were swept under the rug — they're all in `EXPERIMENTS_V2.md` /
`EXPERIMENTS_V3.md`.

---

## 6. How to recreate it yourself

**Prerequisites:** Python 3.11, the [`uv`](https://github.com/astral-sh/uv) package
manager (or plain [`pip`](https://pip.pypa.io/)), and the
[SQLite](https://www.sqlite.org/) catalog `tempjune13.db` (a single-file database) in
the repo root.

```bash
# 1. Environment + dependencies (spaCy + the medium English model)
uv venv                                   # or: python3.11 -m venv .venv
source .venv/bin/activate
uv pip install spacy numpy pytest ruff    # core deps
python -m spacy download en_core_web_md   # the ONLY model used
# (A pristine copy of spaCy + the model is also vendored under vendor/.)

# 2. Sanity-check the finalist (fast, ~2s)
python -m pytest tests/test_clean_pick_v2.py -q     # 19 tests

# 3. Run the full home evaluation (rebuilds the index; ~28 min)
python clean_pick_metric.py run
#   -> writes results/clean_pick_v2.json  (satisfy@1 ~0.1184)

# 4. (Optional) reproduce the external holdout probe
#    needs holdout/scoping_instructions.json + the frozen holdout_mapping.py
python holdout_eval.py
#   -> writes results/holdout_eval.json   (modality 0.50, task-in-list 0.284, n=109)
```

Everything is deterministic: two runs give identical numbers. If yours differ, that's
a bug worth reporting.

### Where things live

| file | what it is |
|---|---|
| `clean_pick_v2.py` | the retriever (the shipped method) |
| `clean_pick_metric.py` | the scoring + evaluation runner |
| `tests/test_clean_pick_v2.py` | 19 honest tests |
| `results/clean_pick_v2.json` | home result |
| `results/holdout_eval.json` | held-out result |
| `EXPERIMENTS_V2.md` / `_V3.md` | full lab notebooks (every number + every failure) |
| `PRESENTATION.md`, `slides/` | the talk (Reveal.js HTML, Beamer, spoken script) |
| `STATE.md` | one-page "where things stand" |

---

## 7. Honest bottom line

A simple, transparent, classical-NLP retriever beats a fair random baseline by **~1.7×**
at matching plain-English intent to models — the best of everything we tried, and the
only method measured on the real endpoint. It is **not** a solved problem: the ceiling
is low because prompts and model cards are written in different registers, and — proven
three separate ways — throwing more vectors or vocabulary at it does not help. The value
here is a rigorously-verified honest baseline and a clear map of what *doesn't* work.
