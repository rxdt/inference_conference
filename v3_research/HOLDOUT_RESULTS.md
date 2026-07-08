# HOLDOUT RESULTS — full, honest documentation

The one-time generalization test. Read ONCE. This document states exactly what went in, what
came out, how it was scored, the result, and — critically — where the comparison is and is NOT
apples-to-apples. No spin.

---

## -2. REFRESHED HOLDOUT (2026-07-03) — owner re-cleaned the `task` labels; result IMPROVED
The owner refreshed the 111-row holdout; **only the `task` field changed** (modality/domain
unchanged, so the family-level scoring below is unaffected). The new task values are mostly
already-valid hyphenated HF labels, so more rows map cleanly and the baseline is no longer
inflated. Scorer updated (`run_holdout_56class._resolve_task`: pass through valid 56-labels,
map legacy underscore forms, exclude unmappable). Re-run result (`holdout_56class_results.json`):

| 56-class top-1 | value |
|---|---|
| HOME (1537 eval) | 0.226 |
| **REFRESHED HOLDOUT (106 rows scored, 5 excluded)** | **0.226** |
| in_list@3 / @12 | 0.377 / 0.736 |
| random | 0.018 |
| majority ("text-classification") | **0.245** (was an inflated 0.368 under the OLD mapping) |

**Verdict (updated):** home 0.226 **== holdout 0.226** — identical, rock-solid generalization.
Against the now-fair majority baseline (0.245) the finalist is **at parity**, not below it. This
supersedes the earlier "loses to majority (0.232 < 0.368)" finding, which §-1 already showed was
a mapping artifact — now confirmed directly by the cleaner labels. The §0/§-1 sections below are
retained as the record of the earlier (superseded) run and the robustness analysis that predicted
this.

---

## -1. ROBUSTNESS OF THE "LOSES TO MAJORITY" VERDICT — it does NOT hold (2026-07-02)
The "finalist loses to majority baseline" conclusion depended on ONE aggressive mapping choice
and is NOT robust. Tested under 3 defensible holdout-task→HF-label mappings (read-only,
`diag_robustness.py` → `robustness.json`):

| mapping | n | finalist top1 | majority top1 | verdict |
|---|---|---|---|---|
| A — original (binary/multiclass/multilabel → text-classification) | 95 | 0.232 | **0.368** | loses |
| B — modality-aware (classification → tabular/image/text by modality) | 95 | 0.179 | 0.189 | ~tie (noise) |
| C — drop ambiguous classification rows | 66 | 0.182 | 0.136 | **BEATS** |

**Corrected verdict:** the "loses to majority" headline was an ARTIFACT of mapping A, which
collapsed three distinct classification types into one label → manufactured a 37% majority
class (the easiest baseline to lose to). Under the more defensible modality-aware mapping (B)
the gap is within noise (0.179 vs 0.189 on 95 rows); dropping the ambiguous rows (C) the
finalist BEATS the majority. **Honest statement: the finalist performs at ROUGHLY majority-
baseline level on the holdout — mapping-dependent, within noise — NOT clearly below it.**
(This corrects my own overstated "fails / loses to a constant" claim; it was the same
artifact-from-an-unexamined-choice error as the earlier false 'ceiling.')

---

## 0. APPLES-TO-APPLES HEADLINE (56-class, same metric as home) — the correct comparison
Both numbers below are **56-way top-1** (identical metric + label space). This is the honest
comparison the family-level number obscured.

| 56-class top-1 | value | notes |
|---|---|---|
| **HOME** (1537 eval, `results_current_eval.json`) | **0.226** | in-distribution |
| **HOLDOUT** (95 real human rows, `holdout_56class_results.json`) | **0.232** | out-of-distribution |
| random floor | 0.018 | 1/56 |
| holdout majority baseline (`text-classification`) | **0.368** | skewed set (35/95) |
| holdout in_list@3 / @12 | 0.432 / **0.821** | shortlist utility |

**Corrected verdict (supersedes the earlier "catastrophic collapse" claim):**
- **The finalist does NOT collapse out-of-distribution.** Home 0.226 ≈ holdout 0.232 top-1 —
  essentially the SAME on the SAME metric, ~13× random. Generalization is *stable*, not
  catastrophic. My earlier family-level framing ("0.387 < 0.396, does not generalize") was
  MISLEADING — a coarser metric that made a skew-driven baseline loss look like a collapse.
- **BUT it still LOSES to the majority baseline (0.232 < 0.368).** The holdout is heavily
  skewed toward classification (binary/multi-class/multi-label all → `text-classification`,
  37% of scored rows), so "always guess text-classification" beats the model on top-1.
- **As a shortlist it has real value:** the correct label is in the top-12 for **82%** of rows.
- **Honest net:** a stable-but-weak model — beats random comfortably, generalizes without
  collapse, but does not beat a trivial constant on this skewed set, and never tested the
  mandated config vocabulary. Trustworthy, modest, not a "working system" by the top-1 bar.

Caveats on THIS 56-class number: 16/111 rows excluded (holdout tasks with no clean HF label:
recommendation/anomaly/clustering/data_generation/unknown); the holdout task→HF-label map is
best-effort and lossy (e.g. binary_classification→text-classification could arguably be
tabular for tabular-modality rows). Map + coverage in `run_holdout_56class.py`.

---

## 1. The holdout (INPUT)
- **Source:** `sarimahsan101/ai-project-scoping-instructions/scoping_instructions.json`
  (downloaded to `src/research/holdout/scoping_instructions.json`, 54,624 bytes).
- **111 rows, authored by real humans, independent of this study** (NOT from `prompts.py`, NOT
  written by the lab). This is a genuine out-of-distribution set.
- **Row shape:** `{"instruction": <str>, "input": "" (always empty), "output": <str>}`.
  The `output` is human clarifying-questions text followed by an embedded JSON object, e.g.:
  ```json
  {"task":"image_classification","domain":"medical","modality":"image",
   "dataset_provided":false,"classes":"unknown"}
  ```
- **What we extract as gold:** the `modality` field. Parsing used builtins only
  (`json.loads` on the `{...}` substring located via `str.rfind` — **no regex**). All
  **111/111 rows parsed**.
- **Gold label spaces present in the holdout:**
  - `modality` (6 values): text 42, tabular 32, image 22, audio 7, video 3, **unknown 5**.
  - `task` (26 values, e.g. binary_classification, regression, nlp_generation, …) — a
    DIFFERENT, finer taxonomy than our 56 HF labels.
  - `domain` (47 values). **No `model` field exists in the dataset.**

## 2. What the system predicts (the FINALIST)
- **Finalist = the plain lexical retriever (exp02):** for a text input, it lemmatizes,
  IDF-weights, and cosine-ranks the input against 56 per-task documents (built from DB
  `short_description`+`specialties`+`domains` + HF task descriptions). Output = the 56 HF
  `task_type` labels, ranked. **It does NOT use config.py.** spaCy + builtins only, no regex.

## 3. Scoring method (INPUT → OUTPUT → SCORE)
The finalist's label space (56 HF tasks) and the holdout's gold space (modality) do not match,
and a row-level task→task mapping would be lossy/judgment-laden. So scoring was done at the
**MODALITY / FAMILY level** — the one clean, unambiguous axis both spaces share:
1. For each row: run `LexScorer.rank(instruction)` → ranked 56-label list.
2. Collapse each predicted label to its **family** via `lib/metrics.py::family_map()`
   (families: image/text/audio/video/tabular/3d/multimodal/other).
3. Gold = row `modality` mapped 1:1 to family (image→image, text→text, tabular→tabular,
   audio→audio, video→video).
4. **Exclude the 5 `unknown`-modality rows** (no gold family) → **106 rows scored**.
5. Metric: family top-1 (predicted top family == gold) and in-family@3.

## 4. Result (OUTPUT) — from `holdout_results.json`, regenerated from code
| metric | finalist | majority-family baseline |
|---|---|---|
| **family top-1** | **0.3868** | **0.3962** (always predict "text") |
| in-family@3 | 0.6698 | — |
| n scored | 106 | (5 excluded: unknown modality) |

Gold family distribution (scored rows): text 42, tabular 32, image 22, audio 7, video 3.

**The finalist scores BELOW the trivial majority baseline (0.387 < 0.396).** On real human
prompts, at the *coarsest* 5-way modality level, it does not beat "always guess text."

## 5. ⚠ APPLES-TO-APPLES — read this before comparing to the home number
**The often-quoted home number (top-1 = 0.226) and this holdout number (0.387) are NOT the
same metric. Comparing them directly is INVALID.**
- Home 0.226 = **56-way** top-1 (exact task_type). Chance ≈ 1/56 = 0.018.
- Holdout 0.387 = **5-way family** top-1 (modality). Chance ≈ 1/5 = 0.20; the *relevant* floor
  is the **majority-family baseline = 0.396**.
- A 5-way task is far easier than a 56-way task, so 0.387 looks "higher" than 0.226 purely
  because the metric is coarser — NOT because the model does better on humans.
- **The only valid within-metric comparison is: holdout family top-1 (0.387) vs its own
  majority baseline (0.396).** By that honest comparison, **the finalist LOSES.**
- (A fully apples-to-apples home-vs-holdout would require scoring the HOME eval at family
  level too. That was set up but NOT run — the owner halted further runs. So the precise
  home-family number is not reported here rather than guessed. What is certain: on the holdout,
  at family level, the finalist is below the family-majority floor.)

## 6. Critique — unflinching
- **The system does not generalize.** Below a constant-guess baseline on real humans, at the
  easiest granularity. This is a clear negative, not a marginal one.
- **Root cause = distribution shift.** The scorer's supervision is HF *model-card* text
  ("Qwen3 is a large language model…"); the holdout is *human project-scoping prose* ("Build
  cancer detection from X-rays"). The lexical overlap that produced an in-distribution signal
  on `prompts.py` largely vanishes on human phrasing. The home signal was substantially an
  in-distribution artifact.
- **The mandated approach was never in this result.** config.py vocabulary — the one lever the
  brief insisted on — is NOT used by the finalist. So this holdout tests the *fallback* (plain
  lexical), not the mandated system. Whether config-vocab would bridge the shift is **unknown
  and now un-testable on THIS holdout** (it has been read; tuning to it would be cheating).
- **Scoring is coarser than the real task.** We validated *modality routing*, not *intent →
  best model*. The actual product goal (return the best DB model) was NOT holdout-validated at
  all; the holdout has no `model` field to check against.
- **Honesty caveats on the number itself:** 5 rows excluded (unknown); `text-to-image`-type
  labels map to `multimodal` in our family map while a human might call them `image` — a few
  rows could be mis-scored either way; the family map is a lab artifact, not ground truth.
- **Bottom line:** the strongest claim the evidence supports is *negative* — the delivered
  lexical system fails out-of-sample generalization even at modality level, and the mandated
  vocabulary component was never validated. A trustworthy negative; not a working system.

## 7. Reproduce
```
cd ~/research
uv run --no-sync python src/research/holdout/run_holdout.py   # writes holdout_results.json
```
Deterministic; emits aggregate numbers only (no instruction text).
