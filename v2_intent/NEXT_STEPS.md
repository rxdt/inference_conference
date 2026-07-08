# Next Steps — build-on-V2 plan (from the conference verdict)

**Verdict:** hard-constrained, evidence-rich **shortlist/abstain** product — NOT an autonomous
rank-1 recommender. Build on **V2 (`~/intent`) retrieval substrate**; add hard constraints +
abstain; keep autonomous rank-1 disabled until human model-level relevance judgments exist.
Source of truth: `CONFERENCE_intent_inference_presentation_and_debate_july_3.json` →
`final_debate.debate_to_settle_fields_in_end_of_debate_FINAL_decision`.

**Honest ceiling (do not violate):** no gold model IDs → satisfy@1 is a *coverage* proxy, not
endpoint correctness. Deliverable is a shortlist + abstain, validated later by human raters.

---

## 🧭 TAKEOVER CONTEXT (read this first, then STATE.md, then this plan)

**What this repo is:** `~/intent` = the **V2** implementation (`clean_pick_v2`), an intent→model
RETRIEVAL system (not task classification). It was judged the **build-on base** at a 3-panelist
conference. You are picking up to implement the P0–P5 plan below.

**Read order:** (1) this TAKEOVER + plan; (2) `STATE.md` (live status); (3) `README.md` (plain
overview); (4) `EXPERIMENTS_V2.md`/`_V3.md` (lab notebooks, every number + failure).
`EXPERIMENTS.md`/`RESEARCH_NOTES.md`/`HANDOFF.md` are banner-marked SUPERSEDED/internal — do not
cite as current.

**Key paths / facts (verified):**
- Env: `~/intent/.venv` (python 3.11, spaCy 3.8.14, en_core_web_md 3.8.0). A pristine
  copy is vendored at `vendor/`. DB: `tempjune13.db` (read-only, `?mode=ro`), ~13,329 models.
- Ship code: `clean_pick_v2.py` (retriever), `clean_pick_metric.py` (metric + `run` CLI),
  `cli.py` (single-prompt V2 CLI — this REPLACED an old V1 cli.py), `tests/test_clean_pick_v2.py`
  (19 tests, pass in ~2s). Results: `results/clean_pick_v2.json` (home), `results/holdout_eval.json`.
- **CLI cache now EXISTS**: `.cli_cache_docs.pkl` (~21MB, built Jul 3 04:53) → cli.py runs ~8s.
  The cache is docs-only (a known limitation — see P2; it does NOT persist the phrase-vector norms).
- Conference record: `~/conference/CONFERENCE_intent_inference_presentation_and_debate_july_3.json`
  — my identity there is `scientist_v2`; my entries are keyed `scientist_v2__clean_pick_v2__*`.

**Headline numbers (authoritative, from results JSONs):** home satisfy@1 **0.1184**
[0.1028,0.1347] vs fair baseline 0.0695 (~1.70×), n=1537; holdout task-in-list@1 **0.2844**,
modality-match **0.50** (near-lossless map, 109/111). Index build ~28 min (full eval config).

**HARD INVARIANTS (do not break — these are the lab's fixed rules):**
- spaCy `en_core_web_md` + Python builtins ONLY. No LLM / embedding APIs / network / downloads.
  (Note: spaCy's own trainable TextCategorizer IS in scope — the panel confirmed this.)
- DB read-only. Determinism required (harness imported first for BLAS pins; ORDER BY id; seeded).
- **Prompt-blindness:** no cue/weight/threshold chosen from any prompt metric; tune on DB-dev;
  a prompt holdout is duplicate/source-controlled and used once. `config.py`/`prompts.py` text
  are NOT read in the scoring path. Run V3 `leak_audit.py` as a CI gate on any trained component.
- `config.py` is currently shown MODIFIED in git — that is the OWNER's edit; do NOT touch/stage it.

**What's DONE vs NOT:**
- DONE: V2 retriever built/verified/committed; independent repro within CI; v3 injection rejected
  (hurts, 0.98→0.775); holdout run + honestly framed; conference debate closed (V2 = build-on base).
- NOT DONE (this plan): the hard constraint/modality filter (P0 — the decisive P3 fix), abstain
  branch (P1), CLI/eval parity + real cache (P2), prompt-trained task signal (P3), gates (P4),
  human endpoint validation (P5). **Nothing below is implemented yet** — plan only.

**Process rule I followed and you should too:** do not edit the shipped artifact to chase a
metric mid-review; implement a step, add its failing→passing test, keep P1/P2 wins from regressing.

---

## P0 — the central risk: hard constraint + modality filter (fixes P3)
The only shipped detector (V2 lexicon) fired only `open_source` on P3 and missed size/local, so a
120B image generator won. This is the #1 build item.

1. **Broaden the constraint lexicon** (`CONSTRAINT_LEXICON`, prompt-blind, DB-justified surfaces):
   `little / small / tiny / SLM / small language model / local / on device / runs local / offline /
   CPU / free / oss / open weight / permissive`. → `edge` (size) and `open_source`.
2. **Hard DB-column filter, applied BEFORE rerank** (columns already on `ModelDoc`): drop
   `parameters > _SMALL_PARAM_MAX` on a size constraint; drop `open_source==0` on OSS. REMOVE
   candidates, don't down-weight. (The `edge` size predicate already exists — wire it as a filter.)
3. **Hard modality gate** via `harness.modality_of(task_type)`: a text request removes
   image-to-image / image-segmentation candidates entirely (would have killed FLUX + Prithvi on P3).
4. **Metadata-trust states**: `parameters`/`license` null → `UNKNOWN`, not a false OSS/small claim.
   (Note: `proprietary==0` does NOT prove OSS.)
5. Apply the hard filters ONLY when the constraint is explicitly stated with high-precision surface
   forms (avoid over-pruning); modality gate always on for single-modality requests.

**Tests (must fail before the fix, pass after):** P3 hard-exclusion (no image-gen / no >size for
"little SLM"); constraint-router units ("a little SLM", "run local", "free and OSS" fire
size+license); metadata-trust (null → clarify, not false claim).

## P1 — abstain / clarify as a first-class output
1. Abstain-to-clarify branch in `pick()` + `cli.py` when: task-inference margin < DB-frozen
   threshold, OR regime=constrained but a required facet is UNKNOWN in the DB, OR the hard filter
   empties the candidate set.
2. Never return a wrong-modality fallback instead of abstaining.
**Test:** the 3 judge prompts return correct-family OR abstain — never confident-wrong.

## P2 — CLI / release parity (close the packaging defect)
1. Default `cli.py` path == the evaluated/recommended path (or an explicit banner when it differs).
2. Ship a **warm cache** (and fix it to persist L2-normalized model-norm matrices, not just docs).
3. Add the abstain/clarify branch to the CLI.
**Test:** default-path ablation (retrieval-only vs +soft-task vs +hard-constraints vs full) all
through the shipped CLI, not an evaluator-only path; determinism two-run bit-identical.

## P3 — task signal upgrade (HOLDOUT-FIRST, not train-now)
1. Freeze a **source/near-duplicate-controlled prompt holdout** + OOS calibration set FIRST.
2. Train a spaCy TextCategorizer AND a vectors-as-features SVM/DIET **on the prompt matrix**
   (NOT card text — my card→prompt shift gave a 0.59 memorization gap). Proper train/dev/holdout.
3. Wrap the winner as `harness.Predictor.predict_task_types` → drop into the **existing**
   `task_predictor` soft-bonus seam (integration is already wired; keep it SOFT, never a gate, or
   P1/P2 text-match wins regress).
**Bar (else diagnostic-only):** beat majority task baseline 0.2453 on held-out top1 OR materially
lift endpoint satisfy@1 over 0.1184; card-text-trained ~0.18 is NOT sufficient.

## P4 — gates & guardrails (borrow from V3)
1. Import V3 `lib/leak_audit.py` as a **required pre-release CI gate** on any trained/labeled
   component (positive control: a pure prompt→gold memorizer MUST fail it).
2. Parser features allowed ONLY behind a slice-specific ablation (constraints, negation,
   infinitival complements, size/local cues) — bare verb+dobj is measured-dead (0.038 top1 / 91%
   abstain), keep it out of primary ranking.
3. Domain/specialty tagger enters ranking ONLY on a CI-gated paired ablation (domain 0.912P/1.5%FP
   is keep-eligible; specialty 0.695P/9.8%FP is diagnostic-only until repaired).

## P5 — honest endpoint validation (unblocks any rank-1 claim)
1. Pool candidates from V1/V2/V3/hybrid/HF-search/popularity; **human/domain raters** judge
   prompt satisfaction, constraint honoring, harm. Until this exists, task holdout does NOT
   validate model recommendation — ship shortlist mode only.
2. **Cost-weighted safety eval** (wrong modality / violated hard constraint / unsafe mismatch /
   unverifiable license all cost more than a benign near-miss or an abstain).
3. **Shortlist-mode metrics** reported separately: recall@k, diversity/coverage, explanation
   usefulness, human decision time.

---

## Sequence & non-regression
Do **P0 → P1 → P2** first (they fix the decisive P3 failure + make the product honest and
runnable). P3–P5 are gated experiments. **Every step must pass P1/P2 non-regression:** adding
constraints/task-signal must NOT destroy the satellite/geospatial (P1) and mitochondria/biomedical
(P2) retrieval wins. Nothing ships as "validated"; the artifact is a shortlist+abstain tool.
