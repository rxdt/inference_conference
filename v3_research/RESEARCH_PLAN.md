# RESEARCH PLAN — Infer user intent from a prompt (spaCy-only)

> ⚠️ **DO NOT TRUST THIS DOCUMENT. VERIFY WITH FURTHER RESEARCH.** ⚠️
> This plan is **decisive about method and integrity, and deliberately silent about the
> answer.** It does not tell you which approach wins, and it contains no predicted
> outcomes to anchor you — those would only invite you to confirm the lead's bias. Your
> job is to run a clean study and let the evidence decide. Every API fact and paper claim
> here is second-hand; re-check it against the live source. (Reminder recurs per section.)

## 0. HOW TO USE THIS PLAN
Do **not** treat any experiment as favored. **Depth over breadth:**
run few experiments fully and honestly; a half-run experiment is a failed experiment.

## 1. GOAL
Infer a user's intent from a free-text prompt using **spaCy + Python builtins only**.

Intent vaguely maps to HF `task_type` (`pipeline_tag`); and possibly domain/specialty where it helps model-pick. Report both **top-1**, **in-list@3**, and **in-list@12**.

## 2. RULES (hard — any violation invalidates the result)
- **spaCy ONLY.** No external/embedding/LLM model, **no regex**, no non-spaCy fuzzymatch,
  no external API, no downloads unless owner approves. (`Matcher`'s built-in `FUZZY` is spaCy-native → allowed.)
- **`en_core_web_md` only.** Not `sm`; do not load whole `lg`. In-repo custom injection / vector of our vocabulary into spaCy is allowed and expected.
- **`config.py` is OFF-LIMITS** — do not read, edit, import, or use it as a cue source.
- **`prompts.py`: never read/edit.** The harness may *import* it at runtime to score, and
  must emit **aggregate metrics only** — no prompt text ever enters an agent's context.
- **Clean vocabulary injection, no overfitting** — every cue traces to a prompt-blind source; prove it.
- **Profile every run** (wall-time + hotspots).
- **You will get a holdout set when you are fully and honestly done** Fabricating a holdout by splitting the prompts is **forbidden**.
- Request an adversarial and honest review from independent source periodically `codex exec …`.

## 3. OPEN QUESTIONS (no favored answer — resolve empirically)
- **Q-ARCH:** two-stage (prompt→`task_type` gate → retrieve within task) vs direct
  (prompt→model)? Run head-to-head; let the clean witness decide. Neither is presumed.
- **Q-SIGNAL:** which signal to generalize (structural vs lexical vs vector vs
  trained-on-DB)? Unknown — measure, don't assume structural or vectors are better.
- **Q-CEILING:** is there an informational ceiling because intent is unstated? Quantify it
  (e.g., label ambiguity / disagreement), don't assert it.
- **Q-TARGET:** is a top-N shortlist + abstain the right deliverable, or top-1? Decide from
  the ambiguity you measure in Q-CEILING.

## 5. INTEGRITY PROTOCOL (the decisive core — follow exactly)
1. **Train/tune only on prompt-blind data.** For any trained component or tuned weight,
   fit on a **DB-internal split** (hold out model rows by id) — never on the eval prompts.
2. **Pre-register before measuring.** Freeze an approach's full config *before* you look at
   its home score. No iterating weights/cues to raise home. (Iterating on the DB-internal
   split is fine.)
3. **Home is a DIAGNOSTIC, not a target.** Report it, but do not optimize toward it and do
   not select on it. Treat a large **home ≫ holdout gap as an overfitting/leak alarm**, not
   a success.
4. **Adversarial leak-audit (every experiment; run by a SECOND agent, not the author).**
   For the approach's cue set, compute the fraction of home-correct predictions that
   coincide with a cue co-occurring (substring + lemma) uniquely with that prompt. Define a
   hard threshold up front; **exceeding it FAILS the experiment** regardless of accuracy.
   This is runnable without config.py (cues come from DB/HF; prompts read only in aggregate).
5. **The holdout set will be supplied ONCE, on your "final" pre-committed shortlist of ≤3 finalists** — the
   final generalization witness. Never per-variant (that would overfit). **absent from the repo** — do not invent one, it's likely you have not met the goal.
6. **Blind reproduction.** A second agent re-runs each carried result without being told the
   number. Numbers that don't reproduce don't count.
7. **Everything regenerated from code.** No hand-typed metrics; assert code↔doc consistency
   in tests.

## 6. TOOLBOX (in-bounds; verify each)
- **Rule/structural (label-free):** `Matcher`, `PhraseMatcher`, `DependencyMatcher`,
  `EntityRuler`, `noun_chunks`, subtrees, ClausIE-style clause decomposition.
- **Vectors:** static `md` + clean injection (`vocab.set_vector`, `spacy init vectors` from
  an in-repo corpus, no download). Note spaCy's caveat that `Doc.vector` is a token-vector
  mean; test per-phrase/head-token alternatives rather than assuming one is better.
- **Trainable on DB labels (clean):** `TextCategorizer`, `TextCatEnsemble`, `tok2vec` on
  `(model text → task_type)`. Prompt-side transfer is a shift bet — the holdout judges it.
- **Unsupervised:** `spacy pretrain` (no labels).
- **Out of bounds (verify):** `sense2vec`, `scispacy`, `trf`, whole `lg` (downloads/
  external). Coref is download-gated + pins old spaCy → blocked unless a spaCy-builtin
  substitute exists.

## 7. EVALUATION
- Metrics: **top-1**, **in-list@#**, **far-error** (cross-modality confusion), **abstain
  rate**. Always print baselines: random (1/#labels), majority-class.
- Home (all prompts) + holdout reported side by side **only for finalists** (§5.5);
  home reported for all, as diagnostic.
- Profile: total wall-time + ms/prompt + `nlp.pipe` batching; pin BLAS threads and sort
  sets for determinism.

## 8. EXPERIMENT PLAN
Run phases in order. **Each phase has a GATE; do not proceed if the gate fails.** Do **not**
run later phases speculatively. Every row records: Hypothesis · Method · Clean-label src ·
Leak-audit result · (finalists only) holdout · Profile · Reproduced?  Decision rules are
integrity-based, never "beat a home number."

**Phase 0 — floor & feasibility.** GATE: baselines computed and the harness proven
(aggregate-only, deterministic, profiled).
| ID | Hypothesis it tests | Method | Clean src |
|----|--------------------|--------|-----------|
| P0-a | what is the floor? | random + majority-class | — |
| P0-b | does lexical overlap beat the floor at all? | lemma/IDF overlap vs DB per-task docs | DB |

**Phase 1 — the architecture question (Q-ARCH), head-to-head.** GATE: both arms leak-audit
clean; pick the direction to expand by holdout in-list@2 on these two finalists (one read).
| ID | Hypothesis | Method | Clean src |
|----|-----------|--------|-----------|
| P1-A | two-stage is viable | prompt→`task_type[0:2]` via DB-kNN/centroid gate → rank models within task | DB |
| P1-B | direct is viable | prompt vector → nearest model row across DB | DB |

**Phase 2 — expand ONLY the surviving direction (Q-SIGNAL).** Add at most a few signal
variants and one blend; each pre-registered and leak-audited. Examples to draw from (choose,
don't run all): object-noun/head-verb, DependencyMatcher SVO, atomic-clause decompose,
TextCat-on-DB, injected phrase centroids, soft-POS weighting. GATE: a blend only survives if
it beats its own components under leak-audit on the DB-internal split.

**Phase 3 — finalize.** Pre-commit ≤3 finalists → will only get holdout **once** when you can prove completion of best-effort experiments (owner can tell if you ar faking it) →
choose the approach with best holdout in-list@2 that is leak-clean and fast. Then: clean
code + **independent** tests + docs **with profiling** + adversarial review (`codex exec …`).

> ⚠️ Verify before trusting. If evidence contradicts a phase's assumption, change the plan.

## 9. PROTO-EXPERIMENT MENU (atomic signals to compose from — a menu, not an order)
grammatical-rule extraction · lexical overlap (lemma/IDF) · `PhraseMatcher` alias ·
`Matcher` token patterns · spaCy `FUZZY` alias · `DependencyMatcher` SVO · object-noun/
head-verb (`noun_chunks`,`ROOT`) · parse-subtree features · atomic-clause decomposition
(relcl/advcl/conj/ccomp → score each → aggregate) · `EntityRuler` domain tags · NER signal ·
soft-POS-weighted features · clarity/OOS gate · whole-prompt vector cosine to task centroid ·
per-phrase injected centroids · head-token/grammar-weighted vector · spaCy sentence-
similarity · `TextCategorizer`(BOW) on DB · `TextCatEnsemble`(BOW+tok2vec) on DB ·
`spacy pretrain` tok2vec (unsupervised) · DB-kNN vote · DB task-centroid · per-task
concatenated card-doc IDF · collapsed ~N-centroid index (speed) · domain/specialty extraction
for model-pick. *(All cues from DB/HF, never config.py or prompts.)*

## 10. DEFINITION OF DONE
The approach that is (1) **leak-audit clean** (hard gate), (2) best **holdout in-list@2**
among the pre-committed finalists, (3) fast (profiled, ms/prompt), (4) **reproduced blind**,
and (5) shipped as clean code + independent tests + docs with profiling. If no approach
clears the leak-audit gate, the honest deliverable is that finding plus the measured ceiling
(Q-CEILING) — not a contaminated number.

## 11. GLOSSARY
**Infer** - To infer a human user's intent.
**Blending** — combine signals inside one approach. **Ensembling** — chain separate
experiments for a result. **Cascade** — strict chain; each experiment feeds the next.
**Deferral** — cascade invoking the next stage only when low-confidence (can fail under
specialist stages / label noise / distribution shift). **Signal / Approach** — one scorer /
a full predictor. **Home / Mini-holdout** — dev eval / subset allowed-read-at-end witness.
**DB-internal split** — held-out model rows used for clean tuning. **Contamination** — any
cue/label/param derived from the eval (incl. shared provenance). **Provenance** — prompt-
blind origin of a cue. **Injection** — adding vectors/phrases to spaCy vocab.
**Overfitting** — fitting the eval distribution. **top-1 / in-list@2** — pred==gold[0] /
gold ∩ top-2 ≠ ∅. **Centroid** — mean vector of a group. **IDF** — down-weights common cues.
**Fusion** — how signals combine (sum/RRF/weighted; justify per experiment). **OOS** — too
vague to classify. **Object-noun/head** — `dobj`/`pobj` target and `ROOT` verb. **Static
vectors vs tok2vec** — frozen vs trained contextual. **Distribution shift** — train source
(cards) ≠ target (prompts). **Leak-audit** — adversarial check that accuracy isn't cue-echo.

## 12. THE 5 PAPERS → ACTIONABLES (read fully)
- **[P-A] LaCy** [2602.12005](https://arxiv.org/abs/2602.12005): augment a decision with
  lightweight spaCy grammar, not confidence alone → a **clarity/OOS gate** (ROOT/`dobj`
  presence, clause count) as a routing input to test.
- **[P-B] Grammatically-Guided Sparse Attention** [2605.24518](https://arxiv.org/abs/2605.24518):
  soft POS masks → **soft-POS weighting** of features (upweight ROOT verb / object nouns) to test.
- **[P-C] Rule-Based Atomic Sentence Extraction** [2601.00506](https://arxiv.org/abs/2601.00506):
  dep-rule clause splitting; watch dropped objects → **atomic-clause decompose** to test.
- **[P-D] Blended RAG** [2404.07220](https://arxiv.org/abs/2404.07220): sparse+dense hybrid,
  multiple indexes, **no fusion formula given** → **multi-representation blend** + adaptive
  sparse/dense weighting to test (you choose & justify the fusion).
- **[P-E] Intent Recognition + OOS** [2507.22289](https://arxiv.org/abs/2507.22289):
  label-space reduction to a shortlist + OOS detection → informs **Q-TARGET** (shortlist +
  abstain). Note: it's neural-only — a caution that trained stages must survive the holdout.
- Context: [2307.02764](https://arxiv.org/abs/2307.02764) (confidence-deferral fails under
  specialist/label-noise/shift — governs any deferral design) ·
  [2402.15610](https://arxiv.org/abs/2402.15610) (reduce over-abstention via cheap evidence).


## documentation (read)
Matchers: [rule-based-matching](https://spacy.io/usage/rule-based-matching) · [matcher](https://spacy.io/api/matcher) · [phrasematcher](https://spacy.io/api/phrasematcher) · [dependencymatcher](https://spacy.io/api/dependencymatcher) · [entityruler](https://spacy.io/api/entityruler)
Textcat/vectors: [textcategorizer](https://spacy.io/api/textcategorizer) · [architectures](https://spacy.io/api/architectures) · [training](https://spacy.io/usage/training) · [tok2vec](https://spacy.io/api/tok2vec) · [vectors](https://spacy.io/api/vectors) · [vectors-similarity](https://spacy.io/usage/linguistic-features#vectors-similarity) · [models](https://spacy.io/models)
NER/coref: [named-entities](https://spacy.io/usage/linguistic-features#named-entities) · [coref](https://spacy.io/api/coref) · [issue 13111](https://github.com/explosion/spaCy/issues/13111) · [coreferee](https://github.com/richardpaulhudson/coreferee)
Deferral/fusion: [2307.02764](https://arxiv.org/abs/2307.02764) · [2404.07220](https://arxiv.org/abs/2404.07220) · [2402.15610](https://arxiv.org/abs/2402.15610)
Projects/clausie: [explosion/projects](https://github.com/explosion/projects) · [spacy-clausie](https://github.com/mmxgn/spacy-clausie)
Vectors/bounds: [en_core_web_lg](https://huggingface.co/spacy/en_core_web_lg) · [universe/pipeline](https://spacy.io/universe/category/pipeline) · [scispacy](https://allenai.github.io/scispacy/) · [sense2vec](https://github.com/explosion/sense2vec) · [init-vectors](https://spacy.io/api/cli#init-vectors)
HF metadata: [model-cards](https://huggingface.co/docs/hub/model-cards) · [models-tags](https://huggingface.co/docs/hub/models-tags) · [tasks](https://huggingface.co/tasks)
Perf: [thinc backends](https://thinc.ai/docs/api-backends) · [spaCy GPU](https://spacy.io/usage#gpu)
Papers: [P-A](https://arxiv.org/abs/2602.12005) · [P-B](https://arxiv.org/abs/2605.24518) · [P-C](https://arxiv.org/abs/2601.00506) · [P-D](https://arxiv.org/abs/2404.07220) · [P-E](https://arxiv.org/abs/2507.22289)

> ⚠️ **REMINDER: DO NOT TRUST THIS DOCUMENT. VERIFY EVERYTHING. The lead fixed the method,
> not the answer — the answer is yours to discover and prove clean.** ⚠️

---

## User prompt shape
```py
{
    "prompt": <USER PROMPT>,        REQUIRED
    "expected_task_type": [0:2],    REQUIRED
    "expected_domains": [0:],       REQUIRED
    "expected_specialties": [0:],   REQUIRED
    "model": <MODEL>,               OPTIONAL!
    "url": <URL>                    OPTIONAL!
},
```
---

## DB Inventory — `tempjune13.db`, table `models`

Read-only characterization of the supervision substrate. **13,329 rows**, primary key `id` (TEXT).

`task_type` is stored as a JSON list; the **normalized label** used throughout is the *first element* of that list (131 rows carry a multi-element list, e.g. `["image-to-3d","text-to-image"]`).

---

## task_type

Fill-rate = fraction of the task's rows with a nonempty value (`''`, `[]`, `{}`, `null` count as empty).
`card_text` is read from `payload_json.card_text`.

| task_type | rows | short_desc | card_text | tags | specialties | domains |
|---|--:|--:|--:|--:|--:|--:|
| text-generation | 4025 | 100% | 78% | 99% | 67% | 5% |
| image-text-to-text | 946 | 100% | 94% | 100% | 100% | 4% |
| token-classification | 754 | 100% | 97% | 100% | 7% | 72% |
| sentence-similarity | 621 | 100% | 88% | 100% | 100% | 5% |
| text-to-image | 550 | 100% | 51% | 99% | 3% | 0% |
| text-classification | 547 | 100% | 89% | 100% | 15% | 12% |
| image-classification | 508 | 100% | 99% | 100% | 23% | 2% |
| automatic-speech-recognition | 423 | 100% | 76% | 97% | 8% | 0% |
| fill-mask | 419 | 100% | 93% | 100% | 12% | 18% |
| question-answering | 415 | 100% | 16% | 100% | 82% | 1% |
| feature-extraction | 385 | 100% | 69% | 95% | 19% | 6% |
| text2text-generation | 349 | 100% | 89% | 100% | 22% | 3% |
| text-to-speech | 292 | 100% | 54% | 95% | 8% | 0% |
| graph-ml | 249 | 100% | 8% | 100% | 3% | 12% |
| multimodal-chat-completion `*` | 208 | 100% | 36% | 100% | 47% | 0% |
| text-to-video | 201 | 100% | 26% | 96% | 1% | 0% |
| translation | 188 | 100% | 93% | 100% | 90% | 1% |
| image-to-image | 181 | 100% | 36% | 100% | 7% | 0% |
| zero-shot-image-classification | 169 | 100% | 99% | 100% | 44% | 6% |
| image-feature-extraction | 135 | 100% | 98% | 100% | 24% | 9% |
| robotics | 128 | 100% | 75% | 100% | 25% | 0% |
| image-segmentation | 125 | 100% | 95% | 100% | 59% | 5% |
| image-to-video | 123 | 100% | 39% | 99% | 7% | 0% |
| image-to-text | 102 | 100% | 92% | 99% | 89% | 2% |
| text-ranking | 102 | 100% | 93% | 97% | 95% | 4% |
| text-to-audio | 85 | 100% | 48% | 100% | 8% | 0% |
| time-series-forecasting | 83 | 100% | 90% | 100% | 11% | 4% |
| object-detection | 79 | 100% | 86% | 100% | 46% | 4% |
| visual-question-answering | 76 | 100% | 30% | 100% | 82% | 1% |
| audio-to-audio | 67 | 100% | 57% | 100% | 4% | 0% |
| audio-classification | 65 | 100% | 100% | 100% | 8% | 2% |
| image-to-3d | 60 | 100% | 72% | 100% | 8% | 0% |
| zero-shot-classification | 58 | 100% | 98% | 100% | 16% | 7% |
| summarization | 57 | 100% | 86% | 100% | 23% | 11% |
| any-to-any | 56 | 100% | 98% | 100% | 100% | 0% |
| depth-estimation | 53 | 100% | 94% | 100% | 45% | 0% |
| video-to-video `*` | 51 | 100% | 24% | 100% | 0% | 0% |
| audio-text-to-text | 51 | 100% | 55% | 84% | 61% | 0% |
| multiple-choice `*` | 46 | 100% | 100% | 100% | 7% | 0% |
| image-text-to-video | 45 | 100% | 87% | 100% | 2% | 2% |
| video-classification | 35 | 100% | 94% | 100% | 60% | 0% |
| video-text-to-text | 33 | 100% | 82% | 100% | 79% | 0% |
| mask-generation | 24 | 100% | 100% | 100% | 29% | 4% |
| **reinforcement-learning** | 21 | 100% | 81% | 100% | 43% | 29% |
| **text-to-3d** | 20 | 100% | 30% | 100% | 5% | 0% |
| visual-document-retrieval | 19 | 100% | 100% | 100% | 42% | 0% |
| zero-shot-object-detection | 17 | 100% | 100% | 100% | 94% | 0% |
| unconditional-image-generation | 15 | 100% | 100% | 100% | 0% | 7% |
| text-retrieval | 13 | 100% | 46% | 100% | 15% | 0% |
| table-question-answering | 12 | 100% | 100% | 100% | 0% | 0% |
| voice-activity-detection | 11 | 100% | 100% | 100% | 18% | 0% |
| keypoint-detection | 10 | 100% | 100% | 100% | 10% | 0% |
| tabular-classification | 9 | 100% | 100% | 100% | 11% | 33% |
| image-text-to-image | 6 | 100% | 100% | 100% | 0% | 0% |
| document-question-answering | 4 | 100% | 100% | 100% | 50% | 25% |
| tabular-regression | 3 | 100% | 100% | 100% | 0% | 0% |

`*` = non-HF task_type string

`pipeline_tag` is the original HF task_type.

Note: `short_description` is 100% filled for every task — it is the only universally-present text field. It is sometimes duplicative of other rows and not informative. There are no duplicate ids, no null `task_type`. `domains` is sparse across the board (mostly 0–18%). **`proprietary` column is mixed-type.**

---

## imbalance

- **Distinct labels:** 56. **Total rows:** 13,329.
- **Top:** `text-generation` = 4,025 (30.2% of all rows). **Bottom:** `tabular-regression` = 3.
- **Top:bottom ratio:** ~1,342×. **Gini (over per-label counts):** 0.694 — heavily skewed.
- **90% coverage:** the top **24** labels account for ≥90% of rows; the remaining 32 labels share <10%.
- Practically: one head class (`text-generation`) plus a long tail.

---

## Domain / specialties

Both columns are JSON lists; a row may carry multiple values, so per-value counts sum above the
fill-row count.

**`domains` — 14 distinct values; filled on 1,089 rows (8.2%).**

| domain | rows | | domain | rows |
|---|--:|---|---|--:|
| medical | 811 | | climate | 25 |
| biology | 600 | | manufacturing | 21 |
| chemistry | 281 | | education | 15 |
| finance | 63 | | marketing | 15 |
| government | 38 | | agriculture | 5 |
| cybersecurity | 32 | | insurance | 3 |
| | | | energy | 3 |
| | | | legal | 1 |

**`specialties` — 19 distinct values; filled on 6,176 rows (46.3%).**

| specialty | rows | | specialty | rows |
|---|--:|---|---|--:|
| conversational | 2640 | | long-context | 94 |
| instruction-following | 1544 | | function-calling | 45 |
| multimodal | 1402 | | edge-deployment | 44 |
| multilingual | 840 | | reading-comprehension | 24 |
| semantic-search | 738 | | rag | 24 |
| vision | 692 | | creative | 14 |
| reasoning | 689 | | chart-analysis | 11 |
| coding | 580 | | structured output | 6 |
| math | 292 | | citations | 1 |
| ocr | 118 | | | |

Both vocabularies are small and closed. `domains` is very sparse (dominated by medical/biology/
chemistry — a life-sciences skew) and several values (`legal`, `energy`, `insurance`) have ≤3 rows.
`specialties` is denser and headed by `conversational`/`instruction-following`/`multimodal`, with a
tail (`citations`, `structured output`, `chart-analysis`).
