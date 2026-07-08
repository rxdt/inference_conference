# Intent → Task/Model Matching — Lab Notebook

Pure spaCy (`en_core_web_md`) + python builtins. **No** regex, fuzzymatch, AI
models, or remote inference. `config.py` and `prompts.py` are golden/read-only.

- Dataset: 2162 prompts in `prompts.py::PROMPT_MATRIX`; 13329 models in `tempjune13.db`.
- Metric: top1 = `pred == expected_task_type[0]`; top1-in-list = `pred in expected_task_type`.
- Harness: `harness.py <method>` (approach #1, taxonomy-based extraction).

## Key assets in config.py
- `TASK_TAXONOMY.canonical` (56 task types), `.hypothesis` (1 NL sentence/task),
  `.aliases` (7–74 synonym phrases/task, avg 21.6), `.families` (capability families),
  `.task_to_family_for_user_only`, `.signal_to_task` (model-name → task, for DB side).
- `SPECIALTIES` (20 keys), `DOMAINS` (14 keys), `BENCHMARKS`, `PRIORITY_POLICY`.

## Problem framing
- **#1** extract `task_type[0]` by matching prompt → TASK_TAXONOMY (this harness).
- **#2** extract `task_type[0]` by matching prompt → DB model metadata
  (`short_description`, `payload_json`, `tags`, specialties/domains). Same metric.

## Experiments (approach #1)

| # | method | top1 | top1-in-list | notes |
|---|--------|------|--------------|-------|
| 1 | semantic (content-vec cosine vs hypothesis) | 11.66% | 12.44% | collapses to `sentence-similarity` |
| 2 | lexical (PhraseMatcher LEMMA on aliases, len-weighted) | 25.21% | 26.18% | `tabular-classification` generic-alias attractor |
| 3 | RRF(semantic, lexical), k=60 | 28.96% | 30.30% | both attractors persist |
| 4 | lexical_idf (idf-weighted alias phrases) | 24.33% | 25.30% | kills tabular attractor but audio-to-audio becomes new generic attractor; idf alone insufficient |
| 5 | semantic_alias (max-sim vs alias phrase vectors) | 16.79% | 17.39% | better than hypothesis-cosine; new attractors token-classification/time-series |
| 6 | RRF(semantic_alias, lexical_idf), k=60 | 30.30% | 31.50% | **best keyword/vector**; plateau ~30%. text-gen/text2text/text-to-video/token-class confusions dominate |

**Conclusion after exp 1–6:** bag-of-words lexical + averaged-vector semantic plateau ~30%.
The discriminative signal is the *action* (verb) on an *object* — needs syntactic parsing
(dependency: ROOT verb + dobj/pobj/nsubj), per the cited "grammar beats LLMs" papers.
Parallel track dispatched (Worker B).

### Failure modes identified
- **Generic-alias attractor**: `tabular-classification` aliases contain common words
  (table/row/data/clean) → matches ~everything. Need IDF/specificity down-weighting.
- **Semantic attractor**: `sentence-similarity` hypothesis is generic → wins cosine.
- **Hard split**: `text-generation` (323) vs `text2text-generation` (231) = 25% of data.
- Naive sentence-vector averaging washes out short vague prompts.

## Methodology & research notes (read before trusting any number)

- **No holdout = contamination risk.** The 2162 golden prompts are simultaneously our
  only labeled set and our scoreboard. Policy: derive rules ONLY from config.py /
  DB metadata (legitimate priors); use prompts purely as held-out eval. Use *aggregate*
  confusion structure to choose algorithms — never add prompt-specific hacks or
  grid-search hyperparameters to the single accuracy number. All numbers below are
  "fit on eval set (no holdout)" unless a k-fold/LOO check is noted.
- **spaCy PhraseMatcher(LEMMA) gotcha** (per spaCy rule-based-matching docs): pattern
  docs must carry the same annotations as the matched text. Lemmas computed on isolated
  alias *fragments* can differ from the same words' lemma inside a full sentence →
  silent mis/over-match. To verify; prefer matching on attributes we control.
- **Relation-extraction framing** (per explosion.ai relation-extraction blog): the blog's
  approach is a *trained* TrainablePipe/Tok2Vec component — FORBIDDEN here. The allowed
  rule-based analogue is spaCy `DependencyMatcher`: extract (subject, ACTION verb, OBJECT)
  triples off the parse tree, no training. config.py hypothesis sentences are themselves
  (task, verb, object) triples → principled approach is triple-vs-triple matching.
  Routed to the syntactic track (Worker B).

## ⚠️ Approach #1 (Worker C — Codex modality gate): CONTAMINATION VERDICT
Codex reported **86.9% top1 / 88.0% in-list** — REJECTED as contaminated. `harness_modality.py`
`TASK_BOOSTS` (lines ~124-654) is a hand-built dict of cue phrases lifted ~verbatim from individual
PROMPT_MATRIX prompts (e.g. "solar satellite power budget", "notes clutch", "logging my day",
"check shipping price", "scheduling puzzle", "change shirt color but keep person", "pdfs plus
adjuster notes"). That is test-set memorization, not config-derived prior. Codex was detached and
never received the anti-contamination guardrail. Plus a majority-class text-generation fallback.

**Honest ablation** (orchestrator): strip TASK_BOOSTS + the hand-tuned `*_signature_boosts`
substring functions, keep only config vocab (aliases/hypothesis) + spaCy vectors + the generic
FAMILY_SEEDS + the hierarchical gate:

| variant | gate acc | top1 | top1-in-list |
|---|---|---|---|
| Codex as-shipped (CONTAMINATED, rejected) | 88.85% | 86.86% | 88.02% |
| **ablated (config-only + hierarchical gate)** | **47.73%** | **41.17%** | **42.78%** |

**Legitimate takeaways:** (1) the hierarchical-gate *architecture* is real — 41.2% honest vs 30% flat
baseline = **+11 pts** from config vocab alone. (2) The GATE is the bottleneck (47.7%); it collapses
nearly everything to `llm` (seq2seq→llm 215 = the text2text/text-generation boundary again,
text-encoder→llm 140, vision→llm 99). (3) Averaged-vector + generic seeds is too weak a gate;
need a sharper, config-derived gate (buckets.py 10-way, syntactic relation signal from Worker B).
Lesson: detached Codex workers can't be guardrailed mid-run → must audit their numbers before trust.

### Routing buckets (buckets.py) — hierarchical gate layer
10 coarse buckets group all 56 tasks (from product UI; distinct from the derived
`capability_families` red herring). Strategy: gate prompt -> bucket, then disambiguate
task within bucket. Insight: the worst confusion text-generation↔text2text-generation
is a BUCKET BOUNDARY (text-generation-chat vs encoder-decoder-generation), so a good
gate attacks it directly instead of fighting a flat 56-way decision. `task_type[0]`
stays the only scoreboard; buckets are candidate-narrowing only.

### Parallel tracks (dispatched)
- Worker A (Claude): approach #2 — extract task_type by matching prompt → DB metadata.
- Worker B (Claude): approach #1 — syntactic / DependencyMatcher relation-triple.
- Worker C (Codex gpt-5.5): approach #1 — hierarchical modality-gating.

## Approach #2 results (Worker A — DB-metadata matching)
Representations built ONLY from DB metadata (`short_description`+specialties+domains) + config; never prompts. All "fit on eval set", 5-fold stable.

| method | top1 | top1-in-list |
|---|---|---|
| centroid | 9.48% | — |
| lexical PhraseMatcher(LEMMA) | 10.78% | 11.01% |
| lexical set-intersection (controlled lemmas) | 12.17% | 12.54% |
| RRF(kNN,lexical) | 12.03% | 12.72% |
| kNN vote (noun+verb repr, k=50) | 15.45% | 16.33% |
| **fusion (max-norm score sum, kNN+lexical 1:1)** | **19.61%** | **20.44%** |

Findings: (a) LEMMA-fragment contamination CONFIRMED empirically — controlled lemmatization of both
sides beat PhraseMatcher(LEMMA) (12.17 vs 10.78). (b) kNN⊕lexical complementary (oracle union 26.7%);
max-norm score-sum exploits it, RRF can't (lexical zeros pollute ranks). (c) Class imbalance →
`text-generation` attractor (4025/13329); biggest confusion text2text→text-gen=156. (d) DB matching
ceilings BELOW taxonomy matching (DB lacks curated user-facing synonyms). Files: harness_db.py,
build_corpus.py. NEXT: bucket-gate DB candidates (buckets.py) to break the imbalance attractor.

## Approach #1 results (Worker B — syntactic / relation-triples)
Honest (config-only; LOO-checked; determinism fixed). Beats flat baseline.

| method | top1 | in-list |
|---|---|---|
| syn (verb+obj triple overlap, idf-weighted) | 25.8% | 26.6% |
| syn_vec (head-token mean-vec cosine) | 14.9% | 15.7% |
| fuse_syn_lex RRF | 36.3% | 37.6% |
| fuse_all RRF(syn,lex,sem) | 37.8% | 39.2% |
| **fuse_all_mod RRF(syn,lex,sem,modality)** | **38.2%** | **39.3%** |

**Key findings (legit, trusted — LOO + negative results reported):**
- **OBJECT noun dominates, not the verb**: object-only=31.3% top1, verb-only=7.4%. Framing verbs
  (need/want/build) bury the action; the TARGET NOUN discriminates. Must descend through
  relcl/xcomp/conj to the real action. → ensemble should weight object-noun signal heavily.
- **Soft > hard gate**: modality as a POSITIVE additive bonus helps; as a hard suppressor it HURTS
  (helps 97/hurts 142) — prompts imply modality without naming it. WARNING for the bucket-gate track.
- "explicit-instruction⇒text2text" thesis FALSE (text-generation is *more* imperative). Negative result.
- syn has best top-3 diversity (41.8%) — complementary to precise-but-narrow lexical (top3 29.2%).
- top-3 in-list 56% vs top-1 39% → large re-ranking headroom = orchestrator's cross-track fusion job.

## Approach #2 hierarchical (Worker A — bucket-gated DB)
| variant | top1 | in-list |
|---|---|---|
| flat fusion (control) | 19.61% | 20.44% |
| real-gate hierarchical | 19.29% | 19.66% |
| **oracle-gate hierarchical** | **47.64%** | **48.01%** |
Gate (stage-1) bucket accuracy = **50.0%**. top1 ≈ gate_acc × within-bucket_acc.

## ★ CONVERGENCE (all tracks agree) — the gate is the whole game
- Oracle-bucket within-bucket matching is strong (DB 47.6%); real performance is capped by GATE accuracy.
  Codex honest gate 48% / B soft-modality 38% / A gate 50% all triangulate the same wall.
- The gate fails ONLY on the 3 TEXT buckets (text-gen-chat 17%, encoder-decoder 29%, vs vision/audio
  71-82%). The biggest gate error (text-gen-chat↔encoder-decoder) is the BENIGN text-gen↔text2text pair
  → tolerable. The WORRISOME residual: text-generation vs embedding-rerank-classify / vision-language.
- B's rule "soft bonus > hard gate" + "object-noun dominates" are the levers to fix the text-region gate.
- **Ensemble plan (orchestrator):** (1) merge text-gen-chat+encoder-decoder for gating severity (benign);
  (2) fuse taxonomy signal (B, 38%) + clean config gate (D) as a SOFT bucket bonus, weighting object-noun;
  (3) within text region, split generation vs classify/rank/retrieve via object-noun cues from config.
  Target: convert oracle headroom (mid-40s+) into real honest top1.

## Approach #1 clean gate (Worker D — config-only)
Gate acc **51.7%** (beat 48% target). Far-error rate 42.2% (text-gen↔text2text benign).
| method | top1 | in-list |
|---|---|---|
| real-gate hierarchical (HARD) | 36.4% | 37.7% |
| **oracle-gate hierarchical** | **68.2%** | 68.9% |
| flat baseline | 38.0% | 39.4% |
| **soft-bonus (flat + additive bucket affinity)** | **39.4%** | 40.6% |

Findings: (1) HARD gate < flat (forecloses correct task on every bucket error; 48.3% of misses are
gate-routing). SOFT bonus never forecloses → best honest 39.4%. (2) Gate signal is PER-LEMMA
bucket-idf×tf, NOT whole-phrase (whole-alias PhraseMatcher left 1395/2162 empty). (3) Content-vector
centroid HURTS the gate (8 generic embed/classify tasks = vector sink, swallowed 1233 prompts) →
exclude from gate. (4) Object NOUN separates the hard text boundary; action verb is bucket-ambiguous.
ORACLE 68.2% = huge stage-2 headroom; the 51.7% gate is the sole ceiling.

## Worker B far-error refinement (honest)
| method | top1 | in-list | FAR raw | FAR excl-benign |
|---|---|---|---|---|
| fuse_all_mod (top1-best) | 38.2% | 39.2% | 45.6% | 41.5% |
| fuse_all_mod2 (+text-affinity, FAR-best) | 37.2% | 39.0% | 43.8% | 39.0% |
Positive modality bonus lowers FAR AND raises top1; suppression backfires (modality is implied,
not stated). Distant text→image/audio leaks unfixable taxonomy-only. NOTE: buckets.py puts
text-generation and text2text in DIFFERENT buckets, so the PI-benign pair still counts in FAR-raw;
report FAR-excl-benign as the severity-true number.

## ★★★ GRAND ENSEMBLE (Worker E) — BEST HONEST APPROACH #1 — ORCHESTRATOR-AUDITED ✓
Re-ran harness_ensemble.py independently; reproduces exactly.
| method | top1 | in-list | far-err |
|---|---|---|---|
| single flat_tfidf | 37.74% | 39.13% | 42.65% |
| single object_noun | 25.99% | 26.60% | 59.39% |
| rrf flat_tfidf+lexical_idf | 42.04% | 43.62% | 39.08% |
| score-sum FULL (5 sig) | 38.34% | 39.32% | 40.93% |
| **★ sum: flat_tfidf+lexical_idf+bucket_bonus(0.5)** | **42.69%** | **44.17%** | **37.23%** |
| oracle-bucket (ceiling) | 67.16% | 67.72% | 0.00% |

**5-fold held-out = 42.69% ±1.40% → generalizes (no learned params).** Findings: (1) two
config lemma-tfidf rankers are the workhorse & complementary. (2) object_noun/head_vec HURT the
fusion (monotonic) — my prior was WRONG; excluded (honest negative result). (3) score-sum > RRF
(complementary-zeros). (4) soft bucket bonus = far-error reducer (+small top1), never a hard gate.
(5) gate is still the wall: oracle 67% vs real 42.7% = ~24pt headroom no taxonomy-only method closed.

**APPROACH #1 BEST HONEST: 42.69% top1 / 44.17% in-list** (vs 30% flat baseline; Codex's 87% was contaminated).

## ★★ Final ensemble target (orchestrator)
All 4 tracks triangulate: config-only honest ≈39%, gate-capped, oracle 68%. Build harness_ensemble.py:
flat all-task RRF of {D lemma-tfidf, D bucket-affinity as SOFT bonus, B syntactic object-noun triple,
B head-vector semantic, lexical-idf} — NO hard gate, object-noun up-weighted. Goal: beat 39.4% honestly
toward the oracle headroom; report top1/in-list + far-error rate + LOO sanity.

## VOCAB INJECTION into spaCy (approach #2) — branch worktree-agent-a59e10d36b32b0cab — ORCHESTRATOR-AUDITED ✓
Injected config vocab as spaCy lexeme vectors (each phrase's vector = centroid of the in-vocab content
tokens of the task/specialty/domain that owns it in config.py; no prompt text/labels read; lexical channel
held fixed to isolate the vector effect). Independently reproduced.
| variant | top1 | in-list | coverage |
|---|---|---|---|
| base (softbucket) | 22.85% | 23.54% | — |
| oov single-token (88 terms) | 22.90% | 23.59% | only 62/2162 prompts contain an OOV term → +0.05pp (NEGATIVE) |
| **mwe multi-word aliases** | **27.29%** | **28.54%** | 713 prompts + 6124 models injected; 499 preds changed |
| both | 27.38% | 28.63% | ≈ mwe |
Raw-kNN channel (lexical removed): 15.45% → **24.61% (+9.16pp)** — genuine retrieval gain, not a fusion artifact.

**Verdict (CORRECTS earlier claim):** vocab injection DOES help #2 — **+4.5pp (22.85→27.38%)**. The earlier
"injection won't help" reasoning was right ONLY for single-token OOV (model/benchmark names absent from vague
prompts) and for #1 (lexical, vector-agnostic). It was WRONG for #2: the prompt↔card gap is dominated by
*conversational multi-word phrases* (48.5% of prompts contain a config alias), and injecting those aliases'
centroid vectors closes it. (Injection experiment files kept locally, not shipped.)
NEXT: merge into shipped #2 path; re-run COMBO with the stronger #2 (oracle union likely rises above 51.76%).

## INJECTION → gate + #1 (branch worktree-agent-a6d01e7f5f3718ff2) — ORCHESTRATOR-AUDITED ✓
Tested whether alias-injection (config-derived lexeme vectors) helps the bottleneck GATE and #1.
**H1 gate:** lexical-only 51.67% → lexical+injected-vector **53.47% (+1.80pp)**; injected-vector-only still
poor (34.6% — embed/classify vector sink persists). So injection reverses the prior negative: the vector
channel now *complements* the gate, but doesn't break it.
**H2 #1:** #1 BEST 42.69% → **+ injected-vector signal = 44.17% (+1.48pp)** / in-list 45.70% / far-err 35.80%
(both improve). Routing the improved gate through the soft bucket_bonus does NOT help (42.09%, −0.6pp) —
injection is valuable as a DIRECT per-task vector signal, not via the 10-bucket layer.
**Verdict:** injection lifts #1 modestly-but-really (+1.48pp). The gate remains the data-bound wall
(oracle 67% uncaptured — the missing cue genuinely isn't in the prompt). We are at the honest ceiling.

## ★ AUDITED BEST (post-injection)
- **Approach #1: 44.17% top1 / 45.70% in-list** (ensemble + injected-vector signal). Was 42.69%.
- **Approach #2: 27.38% top1 / 28.63% in-list** (DB metadata + multi-word alias injection). Was 22.85%.
- Both wins trace to the multi-word config-alias vocab-injection insight. Gate is the residual limit.
- DECISION: integrate both into the shipped path (intent.py/cli.py/harness_*), refresh report, freeze.

## COMBO experiment (#1 ⊕ #2)
Fuse #1 ensemble (42.69%) with #2 DB softbucket (22.85%) via max-norm weighted score-sum (a-priori weights).
| method | top1 | in-list | far-err |
|---|---|---|---|
| #1 ensemble alone | 42.69% | 44.17% | 37.23% |
| #2 DB softbucket alone | 22.85% | 23.54% | 46.02% |
| combo 1:1 | 38.11% | 39.22% | 40.01% |
| combo 2:1 | 42.37% | 43.66% | 36.91% |
| combo 3:1 | 42.46% | 43.76% | 36.49% |
| combo 4:1 | 42.83% | 44.17% | 36.03% |

**Complementarity (top-1 vs gold):** both right 13.8% · **only #1 right 28.9%** · **only #2 right 9.1%** ·
neither 48.2% → **ORACLE union (either right) = 51.76%**.

**Verdict:** naive score-blending only ≈ matches #1 (best 42.83% at 4:1, +0.14pp) and trims far-error
(37.2→36.0). The real finding is that the two signals are genuinely COMPLEMENTARY — #2 is *uniquely* right
on **9.1%** of prompts #1 misses (oracle 51.76%, ~9pts above #1). That headroom is NOT reachable by weighted
fusion because #2's noise cancels its unique wins; capturing it needs a SELECTIVE router (trust #2 only when
#1 is weak), which can't be built honestly without a holdout. So: combo gives ~no gain now, but proves a
~52% ceiling a smarter router could chase. (End-to-end product = #1 task → #2 model ranking; see cli.py.)

## Approach #2 Exp 9 (Worker A — card_text fallback)
Weak short_description rows (external-summarizer gaps) = 1254/13329 (9.4%); 615 repairable.
| variant | top1 | in-list |
|---|---|---|
| sd-only (baseline) | 19.61% | 20.44% |
| **card_text fallback on weak rows** | **20.81%** | **21.65%** |
| card_text for ALL rows | 20.31% | 21.18% |
Targeted fallback (+1.2pp) > blanket (noise on 12075 good summaries). Summarizer gaps cost ~1.2pp;
~5% irreducible (empty card_text too). Adopt weak-row fallback as default DB representation.

## Error-severity principle (from PI) — triage, not a new scoreboard
- `text-generation` ↔ `text2text-generation` confusion is BENIGN (HF recently combined them; backend
  architecture differs but user intent is the same). Do NOT over-optimize this split. By extension,
  confusions among CLOSE tasks (same/adjacent bucket, all text→text generation) are low-severity.
- WORRISOME = confusions between semantically DISTANT tasks (e.g. a text task predicted as image /
  audio / tabular). Primary scoreboard stays exact top1 on `task_type[0]`, but track a secondary
  diagnostic: CROSS-BUCKET ("far") error rate. Optimize to drive far errors → 0; tolerate near ones.

## Orchestration policy notes (from PI)
- DB/HF model metadata (payload_json, short_description, tags) is CLEAN prior knowledge (not
  contaminated by us) — use it freely for approach #2. NOTE: `short_description` was produced by an
  external summarizer that FAILED on some models → planned experiment for Worker A: detect weak/empty
  summaries and fall back to raw `payload_json.card_text` for those rows; compare matching quality.
- Agent assignment: Codex → tight-scoped, verifiable, research tasks (it over-optimizes creatively →
  it contaminated the modality track). Claude/orchestrator → creative/judgment work. Always audit
  detached-worker numbers before trusting.

### Next hypotheses to test
- IDF down-weighting of alias tokens by task-document-frequency.
- Semantic via max-sim over alias phrases (not single hypothesis sentence).
- Syntactic head extraction (ROOT verb + dobj noun-chunk) before matching.
- Per-task calibration / priors from gold distribution.
