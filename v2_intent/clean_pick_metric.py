"""Pre-registered evaluation metric + CLI for the clean V2 retriever.

Split out of ``clean_pick_v2`` (which holds the model representation + retriever)
to keep each module focused. Endpoint = intent-MATCH vs the owner-fixed golden
labels: for each prompt with a nonempty ``expected_task_type``, the top-1
returned model "satisfies" iff its task_type is IN the expected list (not
gold[0]-exact), its domains/specialties intersect the (possibly empty) expected
sets, and every hard constraint detected in the prompt is honored. Reports
satisfy@1, satisfy@k, per-facet rates, constrained-subpopulation rate + size,
coverage, ms/prompt, bootstrap 95% CIs, and a FAIR random-in-plausible baseline.

This measures intent COVERAGE, not provably "best" -- there is no gold model id.
Aggregate metrics only; no prompt text is emitted. Determinism: fixed bootstrap
seed; all iteration order-stable; imports harness first for BLAS pins.
"""

from __future__ import annotations

# harness first so its BLAS thread pins take effect before numpy/spacy.
# pylint: disable=wrong-import-order,wrong-import-position
import harness  # noqa: F401  (re-exported determinism pins)

import json  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from dataclasses import asdict, dataclass, field  # noqa: E402
from typing import TYPE_CHECKING  # noqa: E402

from clean_pick_v2 import (  # noqa: E402
    _DEFAULT_K,
    _LONG_CONTEXT_MIN,
    _SMALL_PARAM_MAX,
    _SOFT_CONSTRAINT_BONUS,
    _SOFT_TASK_BONUS,
    _SPACY_MODEL,
    _SPARSE_TOP_N,
    _TASK_TOPN,
    _CARD_TEXT_CAP_SENTS,
    _DENSE_MAX_PHRASES,
    CONSTRAINT_LEXICON,
    Candidate,
    CleanPickV2,
    ModelDoc,
    ModelLookup,
    build_model_docs,
    constraint_satisfied,
    extract_intent,
    load_build_nlp,
    load_nlp,
    load_raw_models,
)

if TYPE_CHECKING:
    from spacy.language import Language

# ===========================================================================
# PRE-REGISTERED METRIC (intent-coverage vs golden labels; owner-fixed).
# ===========================================================================
# The facet rules below are the SINGLE definition of the coverage endpoint;
# both satisfies() and facet_hits() build on them so the two never drift.
def _task_in_list(cand: Candidate, gold: harness.GoldRow) -> bool:
    """task: model.task_type IN expected_task_type (in-list, NOT gold[0]-exact)."""
    return cand.task_type in set(gold.expected_task_type)


def _facet_covered(cand_labels: tuple[str, ...], expected: list[str]) -> bool:
    """A label facet passes when expected is empty OR the sets intersect."""
    return (not expected) or bool(set(cand_labels) & set(expected))


def satisfies(cand: Candidate, gold: harness.GoldRow, hard: frozenset[str],
               model_lookup: ModelLookup) -> bool:
    """Whether a single candidate satisfies a prompt's golden intent facets.

    task/domain/specialty per the shared facet rules above; constraints: every
    hard constraint detected in the prompt is honored by the model's attributes.
    """
    if not _task_in_list(cand, gold):
        return False
    if not _facet_covered(cand.domains, gold.expected_domains):
        return False
    if not _facet_covered(cand.specialties, gold.expected_specialties):
        return False
    doc = model_lookup(cand.row_id)
    return all(constraint_satisfied(c, doc) for c in hard)


@dataclass
class SatisfyMetrics:
    """Aggregate intent-COVERAGE metrics (not provably 'best'; no gold model)."""

    satisfy_at_1: float = 0.0
    satisfy_at_k: float = 0.0
    task_rate: float = 0.0
    domain_rate: float = 0.0
    specialty_rate: float = 0.0
    constraint_rate: float = 0.0
    constrained_subpop_rate: float = 0.0
    constrained_subpop_n: int = 0
    coverage: float = 0.0
    n: int = 0
    n_skipped: int = 0
    k: int = _DEFAULT_K
    ms_per_prompt: float = 0.0
    satisfy_at_1_ci: tuple[float, float] = (0.0, 0.0)
    satisfy_at_k_ci: tuple[float, float] = (0.0, 0.0)
    baseline_random_in_plausible: float = 0.0
    note: str = (
        "measures intent COVERAGE (top-1 model's task/domain/specialty/"
        "constraint facets vs golden labels), NOT provably 'best' -- no gold "
        "model id exists."
    )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def bootstrap_ci(
    per_prompt_hits: list[int], n_resamples: int, seed: int
) -> tuple[float, float]:
    """95% bootstrap CI for a mean of 0/1 outcomes (resample prompts)."""
    if not per_prompt_hits:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(per_prompt_hits)
    means: list[float] = []
    for _ in range(n_resamples):
        total = 0
        for _ in range(n):
            total += per_prompt_hits[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    lo = means[int(0.025 * n_resamples)]
    hi = means[min(int(0.975 * n_resamples), n_resamples - 1)]
    return (round(lo, 4), round(hi, 4))


def facet_hits(cand: Candidate, gold: harness.GoldRow) -> tuple[bool, bool, bool]:
    """Per-facet (task, domain, specialty) satisfaction for one candidate."""
    return (
        _task_in_list(cand, gold),
        _facet_covered(cand.domains, gold.expected_domains),
        _facet_covered(cand.specialties, gold.expected_specialties),
    )


@dataclass
class _Fold:
    """Mutable per-prompt tallies folded during evaluation."""

    sat1: list[int] = field(default_factory=list)
    satk: list[int] = field(default_factory=list)
    task: int = 0
    domain: int = 0
    specialty: int = 0
    constraint: int = 0
    covered: int = 0
    con_sub_hits: int = 0
    con_sub_n: int = 0
    n: int = 0
    n_skipped: int = 0
    wall_time_s: float = 0.0


def _random_in_plausible_baseline(
    retriever: CleanPickV2,
    docs_by_task: dict[str, list[ModelDoc]],
    prompts_rows: list[harness.GoldRow],
    model_lookup: ModelLookup,
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> float:
    """FAIR baseline: for each prompt draw a random model from the plausible set
    (models whose task_type is in the prompt's PREDICTED task top-k), check
    satisfy@1, average over seeds. NOT a popularity strawman.
    """
    # Prompt-deterministic work (predicted task set -> plausible pool, and the
    # prompt's hard constraints) is computed ONCE per scored prompt; only the
    # random draw depends on the seed. This avoids re-running the task predictor
    # and re-parsing every prompt once per seed.
    @dataclass(frozen=True)
    class _Plausible:
        gold: harness.GoldRow
        pool: list[ModelDoc]
        hard: frozenset[str]

    scorable: list[_Plausible] = []
    for row in prompts_rows:
        if not row.expected_task_type:
            continue
        pool: list[ModelDoc] = []
        for task in sorted(retriever.predicted_tasks(row.prompt)):
            pool.extend(docs_by_task.get(task, []))
        hard = extract_intent(row.prompt, retriever.nlp).hard_constraints
        scorable.append(_Plausible(row, pool, hard))

    if not scorable:
        return 0.0
    rates: list[float] = []
    for seed in seeds:
        rng = random.Random(seed)
        hits = 0
        for entry in scorable:
            if not entry.pool:
                continue
            pick = entry.pool[rng.randrange(len(entry.pool))]
            cand = Candidate(
                row_id=pick.row_id,
                task_type=pick.task_type,
                domains=pick.domains,
                specialties=pick.specialties,
                score=0.0,
            )
            if satisfies(cand, entry.gold, entry.hard, model_lookup):
                hits += 1
        rates.append(hits / len(scorable))
    return round(sum(rates) / len(rates), 4)


def evaluate_retriever(
    retriever: CleanPickV2,
    k: int = _DEFAULT_K,
    n_bootstrap: int = 1000,
    bootstrap_seed: int = 1234,
) -> SatisfyMetrics:
    """Evaluate the retriever over all prompts (aggregate only, no prompt text).

    Skips prompts with empty expected_task_type (no task endpoint to score).
    """
    prompts_rows = harness.load_prompts()
    model_lookup: ModelLookup = retriever.index.doc

    fold = _Fold()
    start = time.perf_counter()
    for row in prompts_rows:
        if not row.expected_task_type:
            fold.n_skipped += 1
            continue
        fold.n += 1
        intent = extract_intent(row.prompt, retriever.nlp)
        picks = retriever.pick(row.prompt, k=k, intent=intent)
        if picks:
            fold.covered += 1
        sat1 = 0
        satk = 0
        if picks:
            top = picks[0]
            task, domain, specialty = facet_hits(top, row)
            fold.task += int(task)
            fold.domain += int(domain)
            fold.specialty += int(specialty)
            con_ok = all(
                constraint_satisfied(c, model_lookup(top.row_id))
                for c in intent.hard_constraints
            )
            fold.constraint += int(con_ok)
            if satisfies(top, row, intent.hard_constraints, model_lookup):
                sat1 = 1
            if any(
                satisfies(c, row, intent.hard_constraints, model_lookup)
                for c in picks
            ):
                satk = 1
        fold.sat1.append(sat1)
        fold.satk.append(satk)
        if intent.hard_constraints:
            fold.con_sub_n += 1
            fold.con_sub_hits += sat1
    fold.wall_time_s = time.perf_counter() - start

    return _finalize_metrics(
        retriever, fold, k, n_bootstrap, bootstrap_seed,
        prompts_rows, model_lookup,
    )


def _finalize_metrics(  # noqa: PLR0913 - aggregation needs the tallies + inputs
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    retriever: CleanPickV2,
    fold: _Fold,
    k: int,
    n_bootstrap: int,
    bootstrap_seed: int,
    prompts_rows: list[harness.GoldRow],
    model_lookup: ModelLookup,
) -> SatisfyMetrics:
    """Fold raw tallies into a SatisfyMetrics (incl. CIs + fair baseline)."""
    n = fold.n

    def rate(hits: int, denom: int) -> float:
        return round(hits / denom, 4) if denom else 0.0

    docs_by_task: dict[str, list[ModelDoc]] = defaultdict(list)
    for doc in retriever.index.all_docs:
        docs_by_task[doc.task_type].append(doc)

    baseline = _random_in_plausible_baseline(
        retriever, docs_by_task, prompts_rows, model_lookup
    )
    return SatisfyMetrics(
        satisfy_at_1=rate(sum(fold.sat1), n),
        satisfy_at_k=rate(sum(fold.satk), n),
        task_rate=rate(fold.task, fold.covered),
        domain_rate=rate(fold.domain, fold.covered),
        specialty_rate=rate(fold.specialty, fold.covered),
        constraint_rate=rate(fold.constraint, fold.covered),
        constrained_subpop_rate=rate(fold.con_sub_hits, fold.con_sub_n),
        constrained_subpop_n=fold.con_sub_n,
        coverage=rate(fold.covered, n),
        n=n,
        n_skipped=fold.n_skipped,
        k=k,
        ms_per_prompt=round(fold.wall_time_s / n * 1000.0, 4) if n else 0.0,
        satisfy_at_1_ci=bootstrap_ci(fold.sat1, n_bootstrap, bootstrap_seed),
        satisfy_at_k_ci=bootstrap_ci(fold.satk, n_bootstrap, bootstrap_seed),
        baseline_random_in_plausible=baseline,
    )


# ===========================================================================
# CLI: build once, evaluate once, write results/clean_pick_v2.json.
# ===========================================================================
def _build_retriever(
    nlp: "Language", popularity_tiebreak: bool = False
) -> CleanPickV2:
    """Build the full retriever with the SOFT phase2.S3 task predictor."""
    import lexical_floor
    import phase2

    raw = load_raw_models()
    # Cheap parser-disabled pass for the full-DB build (lemmas + POS phrases).
    # The retriever precomputes phrase vectors from those phrases once; the full
    # `nlp` (parser on) is needed by phase2's S2 and by prompt intent extraction.
    docs = build_model_docs(raw, load_build_nlp())

    # SOFT task feature: phase2 S3 blend (reuse; used only as an additive bonus).
    # Cues built on phase2's frozen TRAIN split (prompt-blind); the blend is
    # consumed only as an additive bonus, never a gate.
    hmodels = harness.load_models()
    train = [m for m in hmodels if phase2.split_of(m.row_id) == "train"]
    # Reuse phase2's frozen, prompt-blind dedup on its TRAIN split.
    # pylint: disable-next=protected-access
    train = phase2._dedup_by_short_desc(train)  # noqa: SLF001 - frozen build path
    s1 = phase2.build_s1(train, nlp)
    s2 = phase2.StructuralPredictor.build(train, nlp)
    lexical = lexical_floor.LexicalOverlapPredictor.build(
        hmodels, nlp=lexical_floor.load_nlp()
    )
    blend = phase2.BlendPredictor(lexical, s1, s2)
    return CleanPickV2(
        docs, nlp, task_predictor=blend, popularity_tiebreak=popularity_tiebreak
    )


def run() -> None:
    """Evaluate ONCE; report popularity-tiebreak on/off delta; write results."""
    nlp = load_nlp()

    t0 = time.perf_counter()
    retr_off = _build_retriever(nlp, popularity_tiebreak=False)
    build_s = time.perf_counter() - t0

    metrics_off = evaluate_retriever(retr_off)

    # Popularity tie-break ON delta (separate, reported; OFF is the headline).
    retr_on = CleanPickV2(
        retr_off.index.all_docs,  # reuse the built docs
        nlp,
        task_predictor=retr_off.task_predictor,
        popularity_tiebreak=True,
    )
    metrics_on = evaluate_retriever(retr_on)

    extra = {
        "spacy_model": _SPACY_MODEL,
        "build_time_s": round(build_s, 4),
        "constants": {
            "card_text_cap_sents": _CARD_TEXT_CAP_SENTS,
            "sparse_top_n": _SPARSE_TOP_N,
            "dense_max_phrases": _DENSE_MAX_PHRASES,
            "k": _DEFAULT_K,
            "soft_task_bonus": _SOFT_TASK_BONUS,
            "task_topn": _TASK_TOPN,
            "soft_constraint_bonus": _SOFT_CONSTRAINT_BONUS,
            "small_param_max": _SMALL_PARAM_MAX,
            "long_context_min": _LONG_CONTEXT_MIN,
        },
        "constraint_lexicon": {
            name: {"surfaces": list(s), "hardness": h}
            for name, (s, h) in CONSTRAINT_LEXICON.items()
        },
        "popularity_tiebreak_delta": {
            "satisfy_at_1_off": metrics_off.satisfy_at_1,
            "satisfy_at_1_on": metrics_on.satisfy_at_1,
            "satisfy_at_k_off": metrics_off.satisfy_at_k,
            "satisfy_at_k_on": metrics_on.satisfy_at_k,
        },
    }
    harness.RESULTS_DIR.mkdir(exist_ok=True)
    out = harness.RESULTS_DIR / "clean_pick_v2.json"
    out.write_text(
        json.dumps(
            {
                "exp_id": "clean_pick_v2",
                "predictor": CleanPickV2.name,
                "metrics": metrics_off.to_dict(),
                "extra": extra,
            },
            indent=2,
            sort_keys=True,
            default=list,
        )
    )
    print(f"build_time_s={round(build_s, 4)}")
    print(json.dumps(metrics_off.to_dict(), indent=2, default=list))
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> int:
    import sys

    args = argv if argv is not None else sys.argv[1:]
    if not args or args[0] != "run":
        print("usage: python clean_pick_v2.py run")
        return 1
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
