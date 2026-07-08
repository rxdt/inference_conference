"""Leakage audit for a spaCy-only NLI approach.

The threat this module must catch is MEMORIZATION: an approach whose correctness
rides on prompt text (or ``config.py``) rather than on clean, task-describing
supervision. The airtight defense is PROVENANCE: for a spaCy-only approach, if
EVERY cue an approach relies on is a token derivable from DB metadata + the HF
task descriptions, then it cannot have memorized prompts by construction.

Design (owner-decided)
----------------------
* PRIMARY gate  -> :func:`provenance_ok`: every cue must be a member of the
  allowed vocabulary (:func:`db_hf_vocab`). A cue outside it (e.g. a whole prompt
  or a prompt-only token) fails the gate.
* SECONDARY diagnostic -> :func:`cue_echo_fraction` (the former ``audit``): the
  fraction of home-correct predictions coinciding with a prompt-UNIQUE cue. It is
  a narrow cue-echo signal and, on its own, does NOT catch broad/whole-prompt
  memorization (see its docstring).
* Positive control -> :func:`memorization_probe`: shows a pure ``{prompt: gold}``
  memorizer scores ~1.0 on home, so reviewers can see that PROVENANCE (not the
  cue-echo fraction) is what must reject it.
* :func:`leak_verdict` bundles them; the verdict is FAIL when provenance fails.

Nothing here emits prompt or config text: outputs are aggregate numbers and
(for provenance) the offending cue tokens the caller itself supplied.
"""

import json
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from lib import db, harness, metrics, spacy_env
from spacy.language import Language

CUE_ECHO_THRESHOLD = 0.20

_SRC_RESEARCH = Path(__file__).resolve().parents[1]
_HF_DESC_PATH = _SRC_RESEARCH / "docs" / "hf_task_descriptions.json"

# DB text fields feeding the allowed vocabulary. specialties/domains are JSON
# lists; card_text lives inside payload_json (see lib.db.card_text).
_DB_TEXT_FIELDS = ("short_description",)
_DB_LIST_FIELDS = ("specialties", "domains")


def _lemmas(nlp: Language, text: str) -> set[str]:
    """Lower-cased content lemmas of ``text`` (stop/punct/space removed)."""
    out: set[str] = set()
    for tok in nlp(text):
        if tok.is_stop or tok.is_punct or tok.is_space:
            continue
        lemma = tok.lemma_.lower().strip()
        if lemma:
            out.add(lemma)
    return out


def _json_list_words(raw: str | None) -> str:
    """Join a JSON-list TEXT cell into a space-separated string ("" if bad)."""
    if not raw:
        return ""
    try:
        val = json.loads(raw)
    except (ValueError, TypeError):
        return ""
    if isinstance(val, list):
        return " ".join(x for x in val if isinstance(x, str))
    return ""


def _hf_texts() -> list[str]:
    """HF task label+description strings (skips ``_meta`` and non-dict values)."""
    with _HF_DESC_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    texts: list[str] = []
    for key, val in data.items():
        if key == "_meta" or not isinstance(val, dict):
            continue
        label = val.get("label", "")
        desc = val.get("description", "")
        text = f"{label} {desc}".strip()
        if text:
            texts.append(text)
    return texts


def _db_texts(
    row_ids: set[str] | None, include_card_text: bool
) -> list[str]:
    """Clean-supervision text chunks from DB rows (deterministic, ORDER BY id).

    Each row contributes ``short_description`` + ``specialties`` + ``domains``
    words, plus ``card_text`` when ``include_card_text``. If ``row_ids`` is given
    only those rows contribute (used to keep tests tractable). No prompt text.
    """
    fields = ("id", *_DB_TEXT_FIELDS, *_DB_LIST_FIELDS)
    if include_card_text:
        fields = (*fields, "payload_json")
    texts: list[str] = []
    for row in db.iter_models(fields):
        if row_ids is not None and row["id"] not in row_ids:
            continue
        parts = [row["short_description"] or ""]
        parts.extend(_json_list_words(row[lf]) for lf in _DB_LIST_FIELDS)
        if include_card_text:
            parts.append(db.card_text(row["payload_json"]))
        chunk = " ".join(p for p in parts if p)
        if chunk:
            texts.append(chunk)
    return texts


def db_hf_vocab(
    nlp: Language,
    row_ids: set[str] | None = None,
    include_card_text: bool = True,
) -> set[str]:
    """Build the ALLOWED vocabulary: lemmas derivable from clean supervision.

    Sources (the same clean-supervision surface the experiments draw on): DB
    ``short_description`` + ``specialties`` + ``domains`` + ``card_text`` across
    model rows, plus ``docs/hf_task_descriptions.json`` (label + description).
    Text is lemmatized with spaCy (lowercase content lemmas; stop/punct/space
    dropped). Returns a deterministic ``set[str]``.

    Prompt text and ``config.py`` are NEVER a source, so any cue in this set is,
    by construction, non-prompt-derived — which is what makes provenance the
    airtight gate for a spaCy-only approach.

    ``row_ids`` restricts the DB rows that contribute (deterministic subset; used
    to keep tests tractable — card_text over the full corpus is large). A smaller
    subset can only SHRINK the allowed set, i.e. make provenance STRICTER, never
    laxer. ``include_card_text=False`` skips the (large) card_text source.
    """
    vocab: set[str] = set()
    texts = _db_texts(row_ids, include_card_text) + _hf_texts()
    for doc in nlp.pipe(texts, batch_size=64):
        for tok in doc:
            if tok.is_stop or tok.is_punct or tok.is_space:
                continue
            lemma = tok.lemma_.lower().strip()
            if lemma:
                vocab.add(lemma)
    return vocab


def provenance_ok(
    cues: Iterable[str],
    allowed_vocab: set[str],
    nlp: Language | None = None,
) -> tuple[bool, dict[str, object]]:
    """PRIMARY gate: assert every cue originates from clean supervision.

    A cue is provenanced when all of its content lemmas are members of
    ``allowed_vocab`` (the DB/HF surface from :func:`db_hf_vocab`). A cue whose
    lemmas are not all in the vocab (e.g. a whole prompt or a prompt-only token)
    is "offending". Returns ``(ok, details)`` where ``ok`` is True iff there are
    no offending cues, and ``details`` reports counts and the offending cue
    tokens (which the CALLER supplied — never prompt text from the matrix).

    ``details`` keys: ``n_cues``, ``n_offending``, ``offending`` (sorted list of
    the caller's own offending cue strings).
    """
    if nlp is None:
        spacy_env.pin_determinism()
        nlp = spacy_env.load_nlp(exclude=["parser", "ner"])

    offending: list[str] = []
    n_cues = 0
    for cue in cues:
        n_cues += 1
        cue_lemmas = _lemmas(nlp, cue)
        # A cue with no content lemmas (pure stopwords/punct) carries no signal;
        # treat it as provenanced (it cannot encode a memorized prompt token).
        if cue_lemmas and not cue_lemmas <= allowed_vocab:
            offending.append(cue)
    ok = not offending
    return ok, {
        "n_cues": n_cues,
        "n_offending": len(offending),
        "offending": sorted(offending),
    }


def cue_echo_fraction(
    predict_fn: Callable[[str], Sequence[str]],
    cues: Iterable[str],
    nlp: Language | None = None,
) -> dict[str, object]:
    """SECONDARY diagnostic: fraction of home-correct predictions coinciding
    with a prompt-UNIQUE cue.

    A cue is "unique" to a prompt when it co-occurs with exactly one prompt in
    the matrix (co-occurrence = the cue is a substring of the prompt OR shares a
    content lemma with it). For each prompt whose top-1 prediction is correct, we
    mark it cue-coincident when at least one unique cue binds to it. The returned
    ``fraction`` is (cue-coincident correct) / (correct).

    What this does NOT catch (why it is only secondary): it detects NARROW
    cue-echo, not broad or whole-prompt MEMORIZATION. A predictor that memorizes
    entire prompts needs no UNIQUELY-co-occurring cue, so it scores ~0 here. The
    metric is further weakened by EXACT-DUPLICATE prompts, which can never carry a
    prompt-unique cue. Memorization is instead rejected by :func:`provenance_ok`.

    Returns aggregates only: ``fraction``, ``n_prompts``, ``n_correct``,
    ``n_cue_coincident``, ``n_unique_cues``. No prompt/cue text is emitted.
    """
    if nlp is None:
        spacy_env.pin_determinism()
        nlp = spacy_env.load_nlp(exclude=["parser", "ner"])

    matrix = harness.load_prompt_matrix()
    texts: list[str] = [row["prompt"] for row in matrix]  # type: ignore[misc]
    lemmas = [_lemmas(nlp, t) for t in texts]
    unique_prompts = _unique_cue_prompts(list(cues), texts, lemmas, nlp)

    n_correct = 0
    n_cue_coincident = 0
    for i, row in enumerate(matrix):
        golds: Sequence[str] = row.get("expected_task_type") or []  # type: ignore[assignment]
        if metrics.top1(list(predict_fn(texts[i])), golds):
            n_correct += 1
            if i in unique_prompts:
                n_cue_coincident += 1

    return {
        "fraction": (n_cue_coincident / n_correct) if n_correct else 0.0,
        "n_prompts": len(matrix),
        "n_correct": n_correct,
        "n_cue_coincident": n_cue_coincident,
        "n_unique_cues": len(unique_prompts),
    }


def _unique_cue_prompts(
    cues: list[str],
    texts: list[str],
    lemmas: list[set[str]],
    nlp: Language,
) -> set[int]:
    """Prompt indices to which at least one cue binds UNIQUELY (exactly one prompt).

    A cue binds to prompt ``i`` when it is a substring of the text OR shares a
    content lemma with it. Cues binding to exactly one prompt mark that index.
    """
    unique: set[int] = set()
    for cue in cues:
        cue_low = cue.lower()
        cue_lemmas = _lemmas(nlp, cue)
        bound = {
            i
            for i, text in enumerate(texts)
            if cue_low in text.lower() or (cue_lemmas & lemmas[i])
        }
        if len(bound) == 1:
            unique.add(next(iter(bound)))
    return unique


def memorization_probe(
    predict_fn: Callable[[str], Sequence[str]] | None = None,
) -> float:
    """POSITIVE CONTROL: home-correct fraction of a pure ``{prompt: gold}`` memorizer.

    Builds an exact-lookup predictor internally (``prompt -> its gold labels``)
    and returns the fraction of the home set it gets top-1 correct. This exists to
    DEMONSTRATE that a memorizer scores ~1.0 on home, so reviewers can see that
    provenance — not :func:`cue_echo_fraction` — is what rejects it. If a
    ``predict_fn`` is supplied it is scored instead (for symmetry); otherwise the
    internal memorizer is used. Emits ONLY the aggregate fraction — no prompt text.
    """
    matrix = harness.load_prompt_matrix()
    lookup: dict[str, list[str]] = {}
    for row in matrix:
        prompt: str = row["prompt"]  # type: ignore[assignment]
        golds: Sequence[str] = row.get("expected_task_type") or []  # type: ignore[assignment]
        lookup[prompt] = list(golds)

    fn = predict_fn if predict_fn is not None else (lambda p: lookup.get(p, []))
    n = len(matrix)
    if not n:
        return 0.0
    correct = 0
    for row in matrix:
        prompt = row["prompt"]  # type: ignore[assignment]
        golds = row.get("expected_task_type") or []  # type: ignore[assignment]
        if metrics.top1(list(fn(prompt)), golds):
            correct += 1
    return correct / n


def leak_verdict(
    cues: Iterable[str],
    allowed_vocab: set[str],
    predict_fn: Callable[[str], Sequence[str]] | None = None,
    nlp: Language | None = None,
) -> dict[str, object]:
    """Top-level verdict. Provenance is the HARD gate; cue-echo is secondary.

    Returns ``{provenance_ok, offending_cue_count, cue_echo_fraction, threshold,
    verdict}``. ``verdict`` is ``"FAIL"`` when ``provenance_ok`` is False
    (provenance decides), regardless of the cue-echo fraction. When a
    ``predict_fn`` is given, ``cue_echo_fraction`` is computed as a secondary
    signal; otherwise it is reported as ``None``. No prompt text in the output.
    """
    if nlp is None:
        spacy_env.pin_determinism()
        nlp = spacy_env.load_nlp(exclude=["parser", "ner"])

    cue_list = list(cues)
    ok, details = provenance_ok(cue_list, allowed_vocab, nlp=nlp)

    echo: float | None = None
    if predict_fn is not None:
        frac = cue_echo_fraction(predict_fn, cue_list, nlp=nlp)["fraction"]
        echo = float(frac)  # type: ignore[arg-type]  # fraction is numeric by contract

    return {
        "provenance_ok": ok,
        "offending_cue_count": int(details["n_offending"]),  # type: ignore[arg-type]
        "cue_echo_fraction": echo,
        "threshold": CUE_ECHO_THRESHOLD,
        "verdict": "PASS" if ok else "FAIL",
    }
