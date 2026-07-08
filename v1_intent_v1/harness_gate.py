"""Hierarchical bucket-gate harness for task_type[0] (approach #1).

Pure spaCy (en_core_web_md) + python builtins. NO regex, NO fuzzymatch, NO AI
models, NO remote inference. All cues derived PROGRAMMATICALLY from config.py
(TASK_TAXONOMY.aliases/hypothesis/signal_to_task) and buckets.py. No hand-written
phrase lists, no prompt-specific special cases, no grid-searched weights/
thresholds (flat justified defaults only). All numbers are fit on the eval set
(no holdout); rules are config-derived only.

Method (what worked, after ablation):
  * The discriminative gate signal is LEXICAL, at the LEMMA level (not whole
    multi-word alias phrases: ~65% of prompts contain no alias verbatim, so a
    full-phrase PhraseMatcher gives zero signal). For each prompt token lemma
    that occurs in some task's config text, we add idf(lemma) * tf(lemma in
    bucket), where idf is over buckets (log(B/df)) so cross-bucket-generic
    lemmas are down-weighted and bucket-discriminative cues (rank/translate/
    embed/transcribe/...) dominate. This raised gate acc from 46% (whole-phrase
    + centroid) to ~52%.
  * The spaCy content-vector bucket-CENTROID cosine HURTS the gate: averaging
    8 generic tasks makes the text-embedding-rerank-classify centroid a sink
    that swallows most prompts. So semantic centroid is excluded from the gate.
  * Within-bucket (stage 2) uses the same lemma-tfidf scheme restricted to the
    bucket's tasks, with idf over tasks (log(T/df)).

Variants reported:
  (a) real-gate hierarchical  - hard gate (stage1 argmax) then stage2 within it
  (b) oracle-gate hierarchical- stage2 restricted to the GOLD bucket (ceiling)
  (c) flat baseline           - lemma-tfidf over all 56 tasks, no gate
  (d) soft-bonus              - flat task ranker + additive bucket-affinity bonus
                                (no hard restriction; per syntactic-track finding
                                that hard suppression hurts vague prompts)

Run:  python harness_gate.py
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from functools import lru_cache

import spacy
from spacy.tokens import Doc

import config as cfg
from buckets import BUCKETS, TASK_TO_BUCKET
from prompts import PROMPT_MATRIX

TT = cfg.TASK_TAXONOMY
CANON: set[str] = set(TT["canonical"])
HYP: dict[str, str] = TT["hypothesis"]
ALIASES: dict[str, set[str]] = TT["aliases"]
SIGNAL_TO_TASK: dict[str, str] = TT["signal_to_task"]
NOUNISH = {"NOUN", "PROPN"}

# The two text-generation buckets are mutually benign: HF combined
# text-generation and text2text-generation (same user intent). A gate error
# between these two buckets is NOT a far (cross-modality) error.
BENIGN_BUCKET_PAIRS = {
    frozenset({"text-generation-chat", "encoder-decoder-generation"})
}


@lru_cache(maxsize=1)
def nlp():
    return spacy.load("en_core_web_md", exclude=["ner"])


# --------------------------- config-derived text ----------------------------
def task_text(task: str) -> list[str]:
    """All config-derived phrases describing a task: aliases + hypothesis +
    any signal_to_task keys mapping to it. No hand-written strings."""
    phrases = list(ALIASES.get(task, ()))
    phrases.append(HYP[task])
    phrases += [sig for sig, t in SIGNAL_TO_TASK.items() if t == task]
    return phrases


@lru_cache(maxsize=1)
def alias_lemmas() -> dict[tuple[str, str], list[str]]:
    """(task, phrase) -> content lemmas, for every config phrase."""
    n = nlp()
    out = {}
    for task in CANON:
        phrases = task_text(task)
        for p, pdoc in zip(phrases, n.pipe(phrases)):
            out[(task, p)] = [
                t.lemma_.lower()
                for t in pdoc
                if not t.is_punct and not t.is_space and not t.is_stop
            ]
    return out


# --------------------- lemma -> bucket / task statistics --------------------
@lru_cache(maxsize=1)
def lemma_stats():
    """Per lemma: tf in each bucket / task, and bucket/task idf.

    tf(lemma, bucket) = #(task, phrase) in that bucket whose phrase uses lemma.
    idf(lemma) over buckets = log(B / #buckets using it); over tasks = log(T/df).
    """
    pl = alias_lemmas()
    lem_bucket_tf: dict[str, Counter] = defaultdict(Counter)
    lem_task_tf: dict[str, Counter] = defaultdict(Counter)
    lem_buckets: dict[str, set] = defaultdict(set)
    lem_tasks: dict[str, set] = defaultdict(set)
    for (task, _p), lemmas in pl.items():
        bucket = TASK_TO_BUCKET[task]
        for lm in set(lemmas):  # presence per phrase
            lem_bucket_tf[lm][bucket] += 1
            lem_task_tf[lm][task] += 1
            lem_buckets[lm].add(bucket)
            lem_tasks[lm].add(task)
    B, T = len(BUCKETS), len(CANON)
    bidf = {lm: math.log(B / len(bs)) for lm, bs in lem_buckets.items()}
    tidf = {lm: math.log(T / len(ts)) for lm, ts in lem_tasks.items()}
    return lem_bucket_tf, lem_task_tf, bidf, tidf


def _prompt_lemmas(doc: Doc):
    return [
        t.lemma_.lower()
        for t in doc
        if not t.is_stop and not t.is_punct and not t.is_space
    ]


# ----------------------------- stage 1: gate --------------------------------
def gate_scores(doc: Doc) -> dict[str, float]:
    """Lemma-tfidf bucket affinity. score(bucket) = sum over prompt lemmas of
    bidf(lemma) * tf(lemma in bucket)."""
    lem_bucket_tf, _lt, bidf, _ti = lemma_stats()
    s: Counter = Counter()
    for lm in _prompt_lemmas(doc):
        w = bidf.get(lm)
        if w is None:
            continue
        for bucket, tf in lem_bucket_tf[lm].items():
            s[bucket] += w * tf
    return s


def gate(doc: Doc) -> str:
    s = gate_scores(doc)
    return max(BUCKETS, key=lambda b: s.get(b, 0.0))


# ------------------------ stage 2 / flat: task score ------------------------
def task_scores(doc: Doc, tasks=None) -> dict[str, float]:
    """Lemma-tfidf task affinity (idf over tasks), optionally restricted."""
    cand = tasks if tasks is not None else CANON
    _lb, lem_task_tf, _bi, tidf = lemma_stats()
    s: Counter = Counter()
    for lm in _prompt_lemmas(doc):
        w = tidf.get(lm)
        if w is None:
            continue
        for task, tf in lem_task_tf[lm].items():
            if task in cand:
                s[task] += w * tf
    return s


def within_bucket(doc: Doc, bucket: str) -> str:
    s = task_scores(doc, BUCKETS[bucket])
    return max(BUCKETS[bucket], key=lambda t: s.get(t, 0.0))


def flat(doc: Doc) -> str:
    s = task_scores(doc)
    return max(CANON, key=lambda t: s.get(t, 0.0))


def _norm(d: dict[str, float]) -> dict[str, float]:
    mx = max(d.values()) if d else 0.0
    return {k: (v / mx if mx else 0.0) for k, v in d.items()}


def soft_bonus(doc: Doc, alpha: float = 1.0) -> str:
    """Flat task ranker + additive bucket-affinity bonus (no hard restriction).

    Per syntactic-track finding that hard candidate-suppression hurts vague
    prompts. alpha=1.0: max-normalized task and bucket scores weighted equally.
    """
    ts = _norm(task_scores(doc))
    bs = _norm(gate_scores(doc))
    combined = {
        t: ts.get(t, 0.0) + alpha * bs.get(TASK_TO_BUCKET[t], 0.0) for t in CANON
    }
    return max(CANON, key=lambda t: combined[t])


# ----------------------------- evaluation -----------------------------------
def evaluate():
    n = nlp()
    docs = list(n.pipe([p["prompt"] for p in PROMPT_MATRIX]))
    total = len(PROMPT_MATRIX)

    gate_hit = far_err = 0
    gate_conf: Counter = Counter()
    real_top1 = real_inlist = 0
    oracle_top1 = oracle_inlist = 0
    flat_top1 = flat_inlist = 0
    soft_top1 = soft_inlist = 0
    per_task_total: Counter = Counter()
    per_task_hit: Counter = Counter()

    for p, doc in zip(PROMPT_MATRIX, docs):
        gold = p["expected_task_type"]
        g0 = gold[0]
        gb = TASK_TO_BUCKET[g0]
        per_task_total[g0] += 1

        pb = gate(doc)
        if pb == gb:
            gate_hit += 1
        else:
            gate_conf[(gb, pb)] += 1
            if frozenset({gb, pb}) not in BENIGN_BUCKET_PAIRS:
                far_err += 1

        pred = within_bucket(doc, pb)
        if pred == g0:
            real_top1 += 1
            per_task_hit[g0] += 1
        if pred in gold:
            real_inlist += 1

        opred = within_bucket(doc, gb)
        if opred == g0:
            oracle_top1 += 1
        if opred in gold:
            oracle_inlist += 1

        fpred = flat(doc)
        if fpred == g0:
            flat_top1 += 1
        if fpred in gold:
            flat_inlist += 1

        spred = soft_bonus(doc)
        if spred == g0:
            soft_top1 += 1
        if spred in gold:
            soft_inlist += 1

    print(f"N={total}  (fit on eval set, no holdout; config-derived rules only)")
    print(f"\nSTAGE-1 gate accuracy: {gate_hit}/{total} = {gate_hit / total:.3%}")
    print(
        f"CROSS-BUCKET far-error rate (text-gen<->enc-dec benign): "
        f"{far_err}/{total} = {far_err / total:.3%}"
    )

    print("\n--- top1 / in-list ---")
    print(f"{'method':28s} {'top1':>10s} {'in-list':>10s}")
    for name, t1, il in [
        ("(a) real-gate hier", real_top1, real_inlist),
        ("(b) oracle-gate hier", oracle_top1, oracle_inlist),
        ("(c) flat baseline", flat_top1, flat_inlist),
        ("(d) soft-bonus", soft_top1, soft_inlist),
    ]:
        print(f"{name:28s} {t1 / total:>9.2%} {il / total:>10.2%}")

    print("\nworst bucket confusions (gold_bucket -> pred_bucket : count):")
    for (g, pr), c in gate_conf.most_common(12):
        tag = "  (benign)" if frozenset({g, pr}) in BENIGN_BUCKET_PAIRS else ""
        print(f"  {g:32s} -> {pr:32s} {c}{tag}")

    print("\nworst tasks by real-gate recall (task: hit/total):")
    recalls = sorted(
        ((t, per_task_hit[t], per_task_total[t]) for t in per_task_total),
        key=lambda x: x[1] / x[2] if x[2] else 1,
    )
    for t, h, tot in recalls[:15]:
        print(f"  {t:34s} {h}/{tot} = {h / tot:.0%}")


if __name__ == "__main__":
    evaluate()
