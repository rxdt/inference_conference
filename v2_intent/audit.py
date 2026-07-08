"""Adversarial leak-audit for the spaCy-only intent predictors.

Integrity question: is a CORRECT top-1 real intent matching, or a *single-cue
echo* -- one rare lemma/token shared near-uniquely between a prompt and its
gold task's DB material that mechanically decides the winner?

Aggregate-only. Prompt text never leaves this process; we emit counts/rates.
No regex, no external APIs; NLP is spaCy en_core_web_md only (via the existing
harness + predictor code, which we import rather than reimplement).

Definitions declared UP FRONT (see LEAK_AUDIT.md-style docstrings below and the
constants), computed only after they are fixed:

Lexical arm (p0b):
  - Winning score of a correct top-1 = IDF-weighted overlap between prompt
    lemmas and the predicted task's lemma set (exactly the predictor's score).
  - Per-lemma contribution = idf(lemma) for each overlapping lemma.
  - "Dominated" (single-cue) = the single largest per-lemma contribution is
    >= DOMINATION_FRAC of the total winning score.
  - "Distinctive term of the predicted task's DB doc" = the dominating lemma is
    rare across task docs: its IDF is in the high-IDF regime, idf >= HIGH_IDF
    (== ln(N/df) with df <= RARE_DF_MAX task docs). By construction the lemma is
    already in the predicted task's lemma set (it is in the overlap).
  - leak-echo rate = fraction of CORRECT top-1s that are BOTH dominated AND whose
    dominating lemma is distinctive (high-IDF). We also report the plain
    domination rate (>=50% by any single lemma, regardless of IDF) for context.
  - concentration = mean number of contributing lemmas per correct prediction.

Vector arms (p1a/p1b):
  - Degenerate-leak proxy among CORRECT top-1s: fraction where the matched
    row/centroid cosine is NEAR_DUP_COS or higher AND the prompt shares at least
    one exact RARE token with the matched text (rare = token appears in
    <= RARE_TOKEN_DOC_MAX deduped row texts). "Matched text" is the winning
    centroid's nearest member row (p1a) or the nearest neighbor row (p1b).

Threshold (declared before computing): FAIL an approach if its leak-echo /
degenerate-leak rate > FAIL_THRESHOLD.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import harness  # determinism + BLAS pins + loaders (aggregate-only).
import lexical_floor
import phase1

if TYPE_CHECKING:
    from spacy.language import Language

# ---- Declared thresholds / definitions (fixed BEFORE any measurement) ----
DOMINATION_FRAC = 0.50      # single lemma >= 50% of winning score == "dominated"
RARE_DF_MAX = 4             # a task-doc lemma is "rare" if it appears in <=4 task docs
HIGH_IDF = 0.0              # set at runtime from RARE_DF_MAX and N task docs
NEAR_DUP_COS = 0.98         # cosine >= this == "trivial near-duplicate" match
RARE_TOKEN_DOC_MAX = 4      # exact token rare if it appears in <=4 deduped row texts
FAIL_THRESHOLD = 0.60       # FAIL if leak-echo / degenerate rate exceeds this

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _lemmatize(nlp: "Language", text: str) -> list[str]:
    return lexical_floor._lemmatize_batch(nlp, [text])[0]  # noqa: SLF001


def audit_lexical() -> dict[str, object]:
    """Leak-echo audit of the p0b lexical/IDF-overlap predictor."""
    nlp = lexical_floor.load_nlp()
    models = harness.load_models()
    predictor = lexical_floor.LexicalOverlapPredictor.build(models, nlp=nlp)

    n_task_docs = predictor.n_tasks
    high_idf = float(np.log(n_task_docs / RARE_DF_MAX))

    task_lemmas = predictor._task_lemmas   # noqa: SLF001
    idf = predictor._idf                   # noqa: SLF001

    prompts = harness.load_prompts()

    correct = 0
    evaluated = 0
    dominated = 0                 # top lemma >= DOMINATION_FRAC of score
    dominated_and_distinctive = 0  # dominated AND top lemma is high-IDF (rare)
    contrib_counts: list[int] = []

    for row in prompts:
        gold = row.expected_task_type
        if not gold:
            continue
        evaluated += 1
        pred = predictor.predict_task_types(row.prompt)
        if not pred or pred[0] != gold[0]:
            continue
        correct += 1

        # Recompute the winning-score decomposition for the predicted label.
        plemmas = set(_lemmatize(nlp, row.prompt))
        overlap = plemmas & task_lemmas[pred[0]]
        contribs = sorted(
            ((idf.get(lem, 0.0), lem) for lem in overlap), reverse=True
        )
        total = sum(c for c, _ in contribs)
        contrib_counts.append(len(contribs))
        if total <= 0.0 or not contribs:
            continue
        top_c, top_lemma = contribs[0]
        if top_c >= DOMINATION_FRAC * total:
            dominated += 1
            if top_c >= high_idf:  # dominating lemma is rare/distinctive
                dominated_and_distinctive += 1

    leak_echo_rate = dominated_and_distinctive / correct if correct else 0.0
    plain_dom_rate = dominated / correct if correct else 0.0
    mean_contrib = (sum(contrib_counts) / len(contrib_counts)) if contrib_counts else 0.0

    return {
        "approach": "p0b_lexical_overlap",
        "n_evaluated": evaluated,
        "n_correct_top1": correct,
        "domination_frac": DOMINATION_FRAC,
        "high_idf_threshold": round(high_idf, 4),
        "rare_df_max": RARE_DF_MAX,
        "n_task_docs": n_task_docs,
        "n_dominated": dominated,
        "n_dominated_and_distinctive": dominated_and_distinctive,
        "plain_domination_rate": round(plain_dom_rate, 4),
        "leak_echo_rate": round(leak_echo_rate, 4),
        "mean_contributing_lemmas": round(mean_contrib, 4),
        "median_contributing_lemmas": (
            int(np.median(contrib_counts)) if contrib_counts else 0
        ),
        "fail_threshold": FAIL_THRESHOLD,
        "fail": leak_echo_rate > FAIL_THRESHOLD,
    }


def _rare_token_sets(row_texts: list[str]) -> tuple[dict[str, int], list[set[str]]]:
    """Exact whitespace-lowered token doc-frequency across deduped row texts."""
    df: Counter[str] = Counter()
    per_row: list[set[str]] = []
    for t in row_texts:
        toks = set(t.lower().split())
        per_row.append(toks)
        for tok in toks:
            df[tok] += 1
    return dict(df), per_row


def audit_vector(arm: str) -> dict[str, object]:
    """Degenerate near-duplicate leak proxy for p1a (centroid) / p1b (1-NN)."""
    nlp = phase1.load_nlp()
    models = harness.load_models()
    rows, vecs, has, _ = phase1._build_index(nlp, models)  # noqa: SLF001

    row_texts = [t for (t, _), h in zip(rows, has) if h]
    row_labels = [lab for (_, lab), h in zip(rows, has) if h]
    row_vecs = vecs[has]

    token_df, row_tokens = _rare_token_sets(row_texts)

    if arm == "p1a":
        predictor = phase1.CentroidPredictor.build(rows, nlp, vecs, has)
        # Map each label -> indices of its member rows (for nearest-member cosine).
        label_rows: dict[str, list[int]] = defaultdict(list)
        for i, lab in enumerate(row_labels):
            label_rows[lab].append(i)
    else:
        predictor = phase1.NearestNeighborPredictor.build(rows, nlp, vecs, has, k=1)

    prompts = harness.load_prompts()
    correct = 0
    evaluated = 0
    near_dup = 0            # matched cosine >= NEAR_DUP_COS
    degenerate = 0         # near_dup AND shares an exact rare token

    for row in prompts:
        gold = row.expected_task_type
        if not gold:
            continue
        evaluated += 1
        pred = predictor.predict_task_types(row.prompt)
        if not pred or pred[0] != gold[0]:
            continue
        correct += 1

        pvec = phase1._prompt_vector(nlp, row.prompt)  # noqa: SLF001
        if pvec is None:
            continue
        sims = row_vecs @ pvec
        if arm == "p1a":
            # nearest member row of the winning label
            members = label_rows.get(pred[0], [])
            if not members:
                continue
            m_arr = np.asarray(members)
            best_local = int(m_arr[int(np.argmax(sims[m_arr]))])
            best_cos = float(sims[best_local])
            matched_idx = best_local
        else:
            best_idx = int(np.argmax(sims))
            best_cos = float(sims[best_idx])
            matched_idx = best_idx

        if best_cos >= NEAR_DUP_COS:
            near_dup += 1
            ptoks = set(row.prompt.lower().split())
            shared = ptoks & row_tokens[matched_idx]
            if any(token_df.get(t, 0) <= RARE_TOKEN_DOC_MAX for t in shared):
                degenerate += 1

    near_dup_rate = near_dup / correct if correct else 0.0
    degenerate_rate = degenerate / correct if correct else 0.0
    return {
        "approach": arm,
        "n_evaluated": evaluated,
        "n_correct_top1": correct,
        "near_dup_cos_threshold": NEAR_DUP_COS,
        "rare_token_doc_max": RARE_TOKEN_DOC_MAX,
        "n_near_dup": near_dup,
        "n_degenerate_leak": degenerate,
        "near_dup_rate": round(near_dup_rate, 4),
        "degenerate_leak_rate": round(degenerate_rate, 4),
        "fail_threshold": FAIL_THRESHOLD,
        "fail": degenerate_rate > FAIL_THRESHOLD,
    }


def main() -> int:
    lexical = audit_lexical()
    print("LEXICAL (p0b):", json.dumps(lexical, indent=2))
    p1a = audit_vector("p1a")
    print("VECTOR p1a:", json.dumps(p1a, indent=2))
    p1b = audit_vector("p1b")
    print("VECTOR p1b:", json.dumps(p1b, indent=2))

    out = {
        "declared_thresholds": {
            "domination_frac": DOMINATION_FRAC,
            "rare_df_max": RARE_DF_MAX,
            "near_dup_cos": NEAR_DUP_COS,
            "rare_token_doc_max": RARE_TOKEN_DOC_MAX,
            "fail_threshold": FAIL_THRESHOLD,
        },
        "lexical_p0b": lexical,
        "vector_p1a": p1a,
        "vector_p1b": p1b,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "leak_audit.json").write_text(
        json.dumps(out, indent=2, sort_keys=True)
    )
    print("wrote", RESULTS_DIR / "leak_audit.json")
    return 0


# ===========================================================================
# PHASE-2 leak-audit (S1 alias / S2 structural / S3 RRF blend).
#
# Declared UP FRONT (fixed before measuring), reusing FAIL_THRESHOLD = 0.60:
#
# S1 alias:
#   - Winning score of a correct top-1 = sum of matched-alias weights the
#     predictor assigns to the predicted label (recomputed via the same
#     PhraseMatcher hits).
#   - Contribution of an alias = its weight per match span (summed if the same
#     alias matches multiple spans).
#   - "Dominated" = the single largest alias contribution >= DOMINATION_FRAC of
#     the predicted label's total score.
#   - "Rare/distinctive alias" = that alias phrase appears in the alias table of
#     at most ALIAS_RARE_TASK_MAX tasks (a near-unique cue, not a shared word).
#   - leak-echo rate = fraction of correct top-1s BOTH dominated AND whose
#     dominating alias is rare. Also report mean # contributing aliases.
#
# S2 structural:
#   - Winning score = sum over matched cues of weight*(count_in_task/total) for
#     the predicted label (same formula as the predictor).
#   - Contribution of a cue = its weight*(count_in_task/total) toward the label.
#   - "Dominated" = single cue >= DOMINATION_FRAC of the label's score.
#   - "Rare cue" = the cue's association Counter maps to <= CUE_RARE_TASK_MAX
#     tasks (a near-unique (verb,noun)/noun/verb -> task link).
#   - leak-echo rate = correct top-1s BOTH dominated AND dominating cue rare.
#     Also report mean # contributing cues.
#
# S3 blend:
#   - Winning RRF score = sum over the 3 components of 1/(k+rank) for the
#     predicted label.
#   - "Degenerate" = a single component contributes >= DOMINATION_FRAC of the
#     predicted label's RRF score (top-1 driven by one ranker alone).
#   - degenerate rate = fraction of correct top-1s that are degenerate.
# ===========================================================================
ALIAS_RARE_TASK_MAX = 2   # alias phrase in <=2 tasks' tables == near-unique cue
CUE_RARE_TASK_MAX = 2     # assoc cue -> <=2 tasks == near-unique structural link
HEARTBEAT_EVERY = 500     # flush a progress line every N prompts


def _alias_task_counts(alias_table: dict[str, dict[str, float]]) -> Counter[str]:
    """How many task tables each alias phrase appears in (distinctiveness)."""
    c: Counter[str] = Counter()
    for label in alias_table:
        for phrase in alias_table[label]:
            c[phrase] += 1
    return c


def audit_s1() -> dict[str, object]:
    """Leak-echo audit of the S1 alias-vote predictor (p2_s1)."""
    import phase2

    nlp = phase2.load_nlp()
    models = harness.load_models()
    train = phase2._dedup_by_short_desc(phase2._train_models(models))  # noqa: SLF001
    alias_table = phase2.build_alias_table(train, nlp)
    predictor = phase2.AliasVotePredictor(alias_table, nlp)
    alias_tasks = _alias_task_counts(alias_table)
    id_meta = predictor._id_meta  # noqa: SLF001  match_id -> (label, weight)
    matcher = predictor._matcher  # noqa: SLF001
    # Exact per-match provenance: each match_id was registered under the key
    # f"{label} {phrase}", so recover the alias phrase from the string store.
    strings = nlp.vocab.strings
    id_phrase: dict[int, str] = {}
    # Per-label set of label-SURFACE aliases (the intended task-name signal, as
    # opposed to distinctive desc/tag lemmas which are closer to cue-echo).
    label_surface: dict[str, set[str]] = {
        label: set(phase2.label_surface_aliases(label)) for label in alias_table
    }
    for label in alias_table:
        for phrase in alias_table[label]:
            mid = strings[f"{label} {phrase}"]
            id_phrase[mid] = phrase
    prompts = harness.load_prompts()

    correct = evaluated = dominated = dom_and_rare = 0
    # Split the dominating rare alias by provenance: label-surface (intended)
    # vs desc/tag lemma (the genuine echo concern).
    dom_rare_label = dom_rare_nonlabel = 0
    contrib_counts: list[int] = []
    for i, row in enumerate(prompts):
        if i % HEARTBEAT_EVERY == 0:
            print(f"S1 audit {i}/{len(prompts)}", flush=True)
        gold = row.expected_task_type
        if not gold:
            continue
        evaluated += 1
        pred = predictor.predict_task_types(row.prompt)
        if not pred or pred[0] != gold[0]:
            continue
        correct += 1
        doc = nlp(row.prompt)
        # Per-alias contribution to the winning label, keyed on the exact stored
        # alias phrase that fired (via match_id), not the surface span text.
        contrib: dict[str, float] = defaultdict(float)  # alias phrase -> weight
        for match_id, _start, _end in matcher(doc):
            label, weight = id_meta[match_id]
            if label != pred[0]:
                continue
            contrib[id_phrase[match_id]] += weight
        total = sum(contrib.values())
        contrib_counts.append(len(contrib))
        if total <= 0 or not contrib:
            continue
        top_phrase, top_c = max(contrib.items(), key=lambda kv: kv[1])
        if top_c >= DOMINATION_FRAC * total:
            dominated += 1
            # Rarity: the dominating alias phrase is in <= N tasks' alias tables.
            if 0 < alias_tasks.get(top_phrase, 0) <= ALIAS_RARE_TASK_MAX:
                dom_and_rare += 1
                if top_phrase in label_surface.get(pred[0], ()):
                    dom_rare_label += 1  # intended task-name match
                else:
                    dom_rare_nonlabel += 1  # desc/tag lemma echo

    leak = dom_and_rare / correct if correct else 0.0
    leak_nonlabel = dom_rare_nonlabel / correct if correct else 0.0
    plain = dominated / correct if correct else 0.0
    mean_c = sum(contrib_counts) / len(contrib_counts) if contrib_counts else 0.0
    return {
        "approach": "p2_s1_alias",
        "n_evaluated": evaluated,
        "n_correct_top1": correct,
        "domination_frac": DOMINATION_FRAC,
        "alias_rare_task_max": ALIAS_RARE_TASK_MAX,
        "n_dominated": dominated,
        "n_dominated_and_rare": dom_and_rare,
        "n_dom_rare_label_surface": dom_rare_label,
        "n_dom_rare_desc_or_tag": dom_rare_nonlabel,
        "plain_domination_rate": round(plain, 4),
        "leak_echo_rate": round(leak, 4),
        "leak_echo_rate_nonlabel": round(leak_nonlabel, 4),
        "mean_contributing_aliases": round(mean_c, 4),
        "fail_threshold": FAIL_THRESHOLD,
        "fail": leak > FAIL_THRESHOLD,
    }


def audit_s2() -> dict[str, object]:
    """Leak-echo audit of the S2 structural predictor (p2_s2)."""
    import phase2

    nlp = phase2.load_nlp()
    models = harness.load_models()
    train = phase2._dedup_by_short_desc(phase2._train_models(models))  # noqa: SLF001
    predictor = phase2.StructuralPredictor.build(train, nlp)
    pair_t = predictor._pair_table  # noqa: SLF001
    noun_t = predictor._noun_table  # noqa: SLF001
    verb_t = predictor._verb_table  # noqa: SLF001

    prompts = harness.load_prompts()
    correct = evaluated = dominated = dom_and_rare = 0
    contrib_counts: list[int] = []
    for i, row in enumerate(prompts):
        if i % HEARTBEAT_EVERY == 0:
            print(f"S2 audit {i}/{len(prompts)}", flush=True)
        gold = row.expected_task_type
        if not gold:
            continue
        evaluated += 1
        pred = predictor.predict_task_types(row.prompt)
        if not pred or pred[0] != gold[0]:
            continue
        correct += 1
        verbs, nouns, pairs = phase2.extract_verb_object_cues(nlp(row.prompt))
        # Decompose the predicted label's score into per-cue contributions.
        contribs: list[tuple[float, tuple[str, str, object]]] = []
        win = pred[0]

        def add(cue_key, counts, weight):
            total = sum(counts.values())
            if total <= 0 or win not in counts:
                return
            contribs.append((weight * (counts[win] / total), cue_key))

        for pair in sorted(pairs):
            if pair in pair_t:
                add(("pair", pair, pair_t[pair]), pair_t[pair], phase2._S2_W_PAIR)  # noqa: SLF001
        for noun in sorted(nouns):
            if noun in noun_t:
                add(("noun", noun, noun_t[noun]), noun_t[noun], phase2._S2_W_NOUN)  # noqa: SLF001
        for verb in sorted(verbs):
            if verb in verb_t:
                add(("verb", verb, verb_t[verb]), verb_t[verb], phase2._S2_W_VERB)  # noqa: SLF001
        total = sum(c for c, _ in contribs)
        contrib_counts.append(len(contribs))
        if total <= 0 or not contribs:
            continue
        top_c, top_key = max(contribs, key=lambda ck: ck[0])
        if top_c >= DOMINATION_FRAC * total:
            dominated += 1
            n_tasks_for_cue = len(top_key[2])  # the Counter's task span
            if n_tasks_for_cue <= CUE_RARE_TASK_MAX:
                dom_and_rare += 1

    leak = dom_and_rare / correct if correct else 0.0
    plain = dominated / correct if correct else 0.0
    mean_c = sum(contrib_counts) / len(contrib_counts) if contrib_counts else 0.0
    return {
        "approach": "p2_s2_struct",
        "n_evaluated": evaluated,
        "n_correct_top1": correct,
        "domination_frac": DOMINATION_FRAC,
        "cue_rare_task_max": CUE_RARE_TASK_MAX,
        "n_dominated": dominated,
        "n_dominated_and_rare": dom_and_rare,
        "plain_domination_rate": round(plain, 4),
        "leak_echo_rate": round(leak, 4),
        "mean_contributing_cues": round(mean_c, 4),
        "fail_threshold": FAIL_THRESHOLD,
        "fail": leak > FAIL_THRESHOLD,
    }


def audit_s3() -> dict[str, object]:
    """Degenerate-blend audit of S3 (p2_s3): correct top-1s driven by one ranker."""
    import phase2

    nlp = phase2.load_nlp()
    models = harness.load_models()
    train = phase2._dedup_by_short_desc(phase2._train_models(models))  # noqa: SLF001
    print("S3 audit: building S1...", flush=True)
    s1 = phase2.build_s1(train, nlp)
    print("S3 audit: building S2...", flush=True)
    s2 = phase2.StructuralPredictor.build(train, nlp)
    print("S3 audit: building lexical corpus (slow ~5min)...", flush=True)
    lexical = lexical_floor.LexicalOverlapPredictor.build(
        models, nlp=lexical_floor.load_nlp()
    )
    predictor = phase2.BlendPredictor(lexical, s1, s2)
    k = phase2._RRF_K  # noqa: SLF001

    prompts = harness.load_prompts()
    correct = evaluated = degenerate = 0
    n_components_used: list[int] = []
    for i, row in enumerate(prompts):
        if i % HEARTBEAT_EVERY == 0:
            print(f"S3 audit {i}/{len(prompts)}", flush=True)
        gold = row.expected_task_type
        if not gold:
            continue
        evaluated += 1
        pred = predictor.predict_task_types(row.prompt)
        if not pred or pred[0] != gold[0]:
            continue
        correct += 1
        win = pred[0]
        rankings = [
            lexical.predict_task_types(row.prompt),
            s1.predict_task_types(row.prompt),
            s2.predict_task_types(row.prompt),
        ]
        comp_contribs = []
        used = 0
        for r in rankings:
            if win in r:
                comp_contribs.append(1.0 / (k + r.index(win)))
                used += 1
            else:
                comp_contribs.append(0.0)
        n_components_used.append(used)
        total = sum(comp_contribs)
        if total <= 0:
            continue
        if max(comp_contribs) >= DOMINATION_FRAC * total:
            degenerate += 1

    deg_rate = degenerate / correct if correct else 0.0
    mean_used = sum(n_components_used) / len(n_components_used) if n_components_used else 0.0
    return {
        "approach": "p2_s3_blend",
        "n_evaluated": evaluated,
        "n_correct_top1": correct,
        "domination_frac": DOMINATION_FRAC,
        "n_degenerate": degenerate,
        "degenerate_rate": round(deg_rate, 4),
        "mean_components_supporting_win": round(mean_used, 4),
        "fail_threshold": FAIL_THRESHOLD,
        "fail": deg_rate > FAIL_THRESHOLD,
    }


def main_p2(which: str) -> int:
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "leak_audit_p2.json"
    existing = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text())
    if which in ("s1", "all"):
        existing["s1_alias"] = audit_s1()
        print("S1:", json.dumps(existing["s1_alias"], indent=2))
    if which in ("s2", "all"):
        existing["s2_struct"] = audit_s2()
        print("S2:", json.dumps(existing["s2_struct"], indent=2))
    if which in ("s3", "all"):
        existing["s3_blend"] = audit_s3()
        print("S3:", json.dumps(existing["s3_blend"], indent=2))
    existing["declared_thresholds"] = {
        "domination_frac": DOMINATION_FRAC,
        "alias_rare_task_max": ALIAS_RARE_TASK_MAX,
        "cue_rare_task_max": CUE_RARE_TASK_MAX,
        "fail_threshold": FAIL_THRESHOLD,
    }
    out_path.write_text(json.dumps(existing, indent=2, sort_keys=True))
    print("wrote", out_path, flush=True)
    return 0


if __name__ == "__main__":
    import sys

    argv = sys.argv[1:]
    if argv and argv[0] in ("s1", "s2", "s3", "all"):
        raise SystemExit(main_p2(argv[0]))
    raise SystemExit(main())
