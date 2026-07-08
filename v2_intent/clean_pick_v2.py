"""Clean V2 intent->model retriever + its pre-registered evaluation metric.

This module REPLACES a voided v1 (``pick.py``) that ranked by popularity and
hard-gated on ``task_type``. The framing here is NLI-style RETRIEVAL: the user
prompt is a *hypothesis* about a need; each DB model's descriptive text is a
*premise*; we return the model(s) whose capability best entails (satisfies) the
intent. It is NOT a task_type classifier and NOT a popularity ranker.

Pipeline (all prompt-blind; see PRE-REGISTERED CONSTANTS below):

1. Intent extraction -- content lemmas, noun-chunk / ROOT verb+object phrases,
   and explicit CONSTRAINTS via a small documented general-English lexicon
   (free/open, edge/small, long-context, multilingual, fast) mapped to model
   attributes (open_source, parameters, context_window, specialties, deployable).
2. SPARSE retrieve -- IDF-weighted content-lemma overlap (prompt vs. per-model
   text) over an inverted index -> top-N candidates.
3. DENSE rerank -- max/mean spaCy vector similarity between prompt intent phrases
   and each candidate's granular key-phrase vectors (NOT a degenerate full-doc
   mean) (+) a SOFT task bonus (predicted task_type from phase2.S3 matching the
   candidate's task adds a small bonus, never a gate) (+) constraint fit (hard
   constraints filter; soft constraints add). Popularity is NOT in the fit score
   (allowed only as an OFF-by-default final tie-break, reported separately).
4. Return ranked top-k model ids + their task_type/domains/specialties, or
   abstain when the intent is underspecified (no content lemmas / no candidates).

INTEGRITY (strict prompt-blindness): no cue, weight, threshold, or constant was
chosen by looking at any prompt or the eval metric. Every constant below carries
a DB-only justification. ``config.py`` / ``prompts.py`` text are never read; only
aggregate metrics are emitted. NLP is spaCy ``en_core_web_md`` only; no regex, no
external/LLM/embedding APIs, no network.

Determinism: importing ``harness`` FIRST pins BLAS threads before numpy/spacy;
every DB read uses ``ORDER BY id``; dedup is order-independent (id-sorted); all
score ties break on model-id ascending. Two runs give bit-identical metrics.
"""

from __future__ import annotations

# harness MUST import before numpy/spacy so its BLAS thread pins take effect;
# this deliberately places a first-party import ahead of stdlib ones.
# pylint: disable=wrong-import-order,wrong-import-position
import harness  # MUST be first: pins BLAS threads before numpy/spacy import.

import json  # noqa: E402
import math  # noqa: E402
import sqlite3  # noqa: E402
from collections import defaultdict  # noqa: E402
from collections.abc import Callable  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import TYPE_CHECKING  # noqa: E402

if TYPE_CHECKING:
    import numpy as np  # annotations only; runtime import stays lazy (BLAS pins)
    from spacy.language import Language
    from spacy.tokens import Doc, Span

# Maps a model row_id to its ModelDoc (the index's public lookup).
ModelLookup = Callable[[str], "ModelDoc"]

_SPACY_MODEL = "en_core_web_md"
_PIPE_BATCH = 128
_MIN_TOKEN_LEN = 3

# ===========================================================================
# PRE-REGISTERED CONSTANTS (DB-only justification; NEVER prompt-tuned).
# ===========================================================================

# card_text is capped to its first N SENTENCES before representation. DB fact
# (DB_INVENTORY sec 4): card_text median 5,126 chars, p90 17,474, max 363,382 -
# raw model-card markdown, long and noisy. HF card task-indicative content
# (title / one-line summary / "this model ..." lead) sits in the opening
# sentences, so a sentence prefix bounds cost while keeping the task signal.
# Sentence (not char) cap so we never split a phrase mid-sentence.
_CARD_TEXT_CAP_SENTS = 30

# HARD char pre-cap applied BEFORE spaCy parsing, purely to bound parse cost on
# the pathological tail (max card_text is 363,382 chars; parsing that in full is
# ~0.5s/model and dominates the build). DB fact: card_text p90 is ~17,474 chars,
# so a 20,000-char prefix fully preserves >=90% of cards and only trims the long
# tail -- and the first _CARD_TEXT_CAP_SENTS sentences (the task signal) live far
# inside that budget. This is a cost bound, not a data filter or a tuned knob.
_CARD_TEXT_PRECAP_CHARS = 20_000

# SPARSE candidate pool size handed to the (expensive) dense reranker. Chosen
# from DB scale only: 13,329 rows; the smallest trainable HF task has ~80 train
# rows, and a broad head class (text-generation) has ~4k, so 200 comfortably
# covers a single task's worth of plausible models plus lexical neighbours
# without scanning the whole DB per prompt. Not tuned on prompts.
_SPARSE_TOP_N = 200

# Max granular key-phrase vectors kept per model for dense rerank. DB fact:
# capped card_text yields tens of noun-chunks; 40 covers the informative early
# phrases (title + lead sentences) with a bounded per-model vector budget.
_DENSE_MAX_PHRASES = 40

# Default top-k returned / scored.
_DEFAULT_K = 5

# SOFT task bonus: added to a candidate's dense score when its task_type is in
# the prompt's predicted task_type top list. Small relative to the [0,1] cosine
# scale so strong text can still outrank a task match (NEVER a gate). Value is a
# fixed fraction of the cosine range, not a tuned knob.
_SOFT_TASK_BONUS = 0.15
# How many predicted task_types (from phase2.S3) count as "the predicted set".
_TASK_TOPN = 2

# SOFT constraint bonus per satisfied soft constraint (same [0,1] scale idea).
_SOFT_CONSTRAINT_BONUS = 0.05

# ---- Constraint lexicon thresholds (DB parameter/context distributions) ----
# DB fact (parameters, n=7072): p10=33M, p25=135M, median=1.1B. A model is
# "small / edge-deployable-by-size" when parameters <= ~1e9 (about the DB
# median); we use 1e9 as a round, distribution-anchored cutoff, not a tuned one.
_SMALL_PARAM_MAX = 1_000_000_000
# DB fact (context_window, n=6231): p75=40,960, p90=131,072. "Long-context"
# means the upper quartile; we anchor at 32,768 (>= p75) as a round threshold.
_LONG_CONTEXT_MIN = 32_768

DB_PATH = harness.DB_PATH


# ===========================================================================
# CONSTRAINT LEXICON (small, documented, general-English; prompt-blind).
# Each entry maps free-English surface forms to a model-attribute predicate.
# Surface forms are ordinary English, chosen for meaning, NOT from any prompt.
# ===========================================================================
# name -> (lemma surface forms, hardness). "hard" constraints filter the
# candidate set; "soft" constraints only add a small bonus.
CONSTRAINT_LEXICON: dict[str, tuple[tuple[str, ...], str]] = {
    # free / open-source -> open_source == 1
    "open_source": (
        ("open source", "open-source", "opensource", "open weight",
         "open-weight", "free", "self host", "self-host", "on premise",
         "on-premise", "on prem"),
        "hard",
    ),
    # small / edge / on-device -> small parameter count AND deployable
    "edge": (
        ("edge", "on device", "on-device", "mobile", "small", "tiny",
         "lightweight", "laptop", "raspberry pi", "low resource",
         "low-resource"),
        "hard",
    ),
    # long context -> large context_window
    "long_context": (
        ("long context", "long-context", "large context", "long document",
         "long-document", "big context"),
        "hard",
    ),
    # multilingual -> "multilingual" specialty
    "multilingual": (
        ("multilingual", "many languages", "multiple languages",
         "cross lingual", "cross-lingual"),
        "soft",
    ),
    # fast / low-latency -> deployable (proxy: deployable == 1) + soft
    "fast": (
        ("fast", "low latency", "low-latency", "realtime", "real time",
         "real-time", "quick", "speedy", "responsive"),
        "soft",
    ),
}


# ===========================================================================
# Model representation (prompt-blind, built once).
# ===========================================================================
@dataclass(frozen=True)
class ModelDoc:
    """A DB model's scoreable representation.

    ``lemmas`` feeds the sparse inverted index. ``phrases`` are the granular
    key phrases (POS-based noun-phrase chunks) whose spaCy vectors drive the
    dense rerank -- both are extracted in the one-time cheap build pass (POS
    tagger only, no dependency parser), so no per-query parsing is needed.
    Attribute fields back the constraint predicates. Descriptive text is
    card_text (capped) else short_description; tags and capability_families are
    excluded by design.
    """

    row_id: str
    task_type: str
    domains: tuple[str, ...]
    specialties: tuple[str, ...]
    open_source: bool
    deployable: bool
    parameters: int | None
    context_window: int | None
    likes: int
    lemmas: frozenset[str]
    phrases: tuple[str, ...]


def _parse_json_list(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return ()
    if isinstance(parsed, list):
        return tuple(str(x) for x in parsed if str(x).strip())
    return ()


@dataclass(frozen=True)
class RawModel:
    """Raw DB fields for one model before spaCy processing (ORDER BY id)."""

    row_id: str
    task_type: str
    short_description: str
    card_text: str
    domains: tuple[str, ...]
    specialties: tuple[str, ...]
    open_source: bool
    deployable: bool
    parameters: int | None
    context_window: int | None
    likes: int

    def descriptive_text(self) -> str:
        """card_text when non-empty, else short_description (per spec)."""
        return self.card_text if self.card_text.strip() else self.short_description


def load_raw_models(db_path: Path | str = DB_PATH) -> list[RawModel]:
    """Read-only load of all model rows, ORDER BY id (determinism fix)."""
    uri = f"file:{Path(db_path)}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT id, task_type, short_description, payload_json, domains, "
            "specialties, open_source, deployable, parameters, context_window, "
            "likes FROM models ORDER BY id"
        )
        rows: list[RawModel] = []
        for r in cur:
            # Reuse harness's identical card_text extractor (single source).
            # pylint: disable-next=protected-access
            card = harness._extract_card_text(r["payload_json"])  # noqa: SLF001
            rows.append(
                RawModel(
                    row_id=str(r["id"]),
                    task_type=harness.normalize_task_type(r["task_type"]),
                    short_description=r["short_description"] or "",
                    card_text=card,
                    domains=_parse_json_list(r["domains"]),
                    specialties=_parse_json_list(r["specialties"]),
                    open_source=bool(r["open_source"]),
                    deployable=bool(r["deployable"]),
                    parameters=(
                        int(r["parameters"]) if r["parameters"] is not None else None
                    ),
                    context_window=(
                        int(r["context_window"])
                        if r["context_window"] is not None
                        else None
                    ),
                    likes=int(r["likes"]) if r["likes"] is not None else 0,
                )
            )
        return rows
    finally:
        conn.close()


def load_nlp() -> "Language":
    """Load full spaCy en_core_web_md (parser needed for sentences + chunks).

    Used for prompt intent extraction and lazy per-candidate phrase parsing.
    """
    import spacy

    return spacy.load(_SPACY_MODEL)


def load_build_nlp() -> "Language":
    """Cheap pipeline for the one-time sparse-index build.

    Parser + NER are disabled (the sparse side needs only lemmas); a rule-based
    ``sentencizer`` supplies sentence boundaries for the sentence cap without the
    parser's per-token cost. Granular phrases are parsed lazily at rerank time.
    """
    import spacy

    nlp = spacy.load(_SPACY_MODEL, disable=["parser", "ner"])
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


def _capped_span(doc: "Doc") -> "Doc | Span":
    """Return the first _CARD_TEXT_CAP_SENTS sentences of a doc as a Span.

    A Span iterates tokens just like a Doc, so downstream lemma / POS-phrase
    extraction works unchanged. Sentence boundaries come from the parser (full
    nlp) or the sentencizer (cheap build nlp) -- either works.
    """
    sents = list(doc.sents)[:_CARD_TEXT_CAP_SENTS]
    if not sents:
        return doc
    return doc[sents[0].start : sents[-1].end]


def _content_lemmas(span: "Doc | Span") -> set[str]:
    """Lowercased alphabetic content lemmas (drop stop/punct/space/short)."""
    return {
        tok.lemma_.lower()
        for tok in span
        if not tok.is_stop
        and not tok.is_punct
        and not tok.is_space
        and tok.is_alpha
        and len(tok.lemma_) >= _MIN_TOKEN_LEN
    }


def _key_phrases(span: "Doc | Span") -> list[str]:
    """Granular key phrases via POS-based noun-phrase chunking (tagger only).

    A key phrase is a maximal contiguous run of ADJ/NOUN/PROPN content tokens
    (e.g. "neural machine translation", "transformer architecture"). This needs
    only the POS tagger -- NOT the dependency parser -- so model phrase vectors
    can be precomputed in the one-time cheap build pass. These granular phrase
    vectors (NOT a full-doc mean, which is degenerate: proven .69 dev -> .07
    prompt) drive the dense rerank. De-duplicated (order-preserving), capped.
    """
    phrases: list[str] = []
    seen: set[str] = set()
    current: list[str] = []

    def flush() -> None:
        if current:
            phrase = " ".join(current)
            key = phrase.lower()
            if key not in seen:
                seen.add(key)
                phrases.append(phrase)
            current.clear()

    for tok in span:
        if tok.pos_ in ("NOUN", "PROPN", "ADJ") and tok.is_alpha:
            current.append(tok.text)
        else:
            flush()
    flush()
    return phrases[:_DENSE_MAX_PHRASES]


def _dedup_descriptive(models: list[RawModel]) -> list[RawModel]:
    """Drop rows whose descriptive text duplicates an earlier (id-sorted) row.

    Order-independent by construction: we sort by id first, so the surviving
    representative for each identical text is deterministic regardless of the
    input order. Prevents boilerplate/unit-test stubs from dominating the index.
    """
    seen: set[str] = set()
    kept: list[RawModel] = []
    for model in sorted(models, key=lambda m: m.row_id):
        key = model.descriptive_text().strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        kept.append(model)
    return kept


def build_model_docs(
    models: list[RawModel], nlp: "Language | None" = None
) -> list[ModelDoc]:
    """Build ModelDocs (lemmas + sentence-capped text) for all rows.

    Uses the cheap build pipeline (parser disabled) so the one-time full-DB pass
    avoids the parser. Dedup is on descriptive text across the WHOLE input
    (id-sorted, so order-independent). Granular phrases are derived lazily later.
    """
    nlp = nlp if nlp is not None else load_build_nlp()
    kept = _dedup_descriptive(models)
    # Char pre-cap bounds cost on the pathological long tail; the sentence cap
    # (via _capped_span) does the real trimming and defines capped_text.
    texts = [m.descriptive_text()[:_CARD_TEXT_PRECAP_CHARS] for m in kept]
    docs: list[ModelDoc] = []
    for model, doc in zip(kept, nlp.pipe(texts, batch_size=_PIPE_BATCH)):
        span = _capped_span(doc)
        docs.append(
            ModelDoc(
                row_id=model.row_id,
                task_type=model.task_type,
                domains=model.domains,
                specialties=model.specialties,
                open_source=model.open_source,
                deployable=model.deployable,
                parameters=model.parameters,
                context_window=model.context_window,
                likes=model.likes,
                lemmas=frozenset(_content_lemmas(span)),
                phrases=tuple(_key_phrases(span)),
            )
        )
    return docs


# ===========================================================================
# Prompt intent extraction (prompt-blind machinery).
# ===========================================================================
@dataclass(frozen=True)
class PromptIntent:
    """Extracted intent from one prompt (no raw prompt text is stored)."""

    lemmas: frozenset[str]
    phrases: tuple[str, ...]
    hard_constraints: frozenset[str]
    soft_constraints: frozenset[str]


def _detect_constraints(doc: "Doc") -> tuple[set[str], set[str]]:
    """Detect (hard, soft) constraint names via the documented lexicon.

    Matching is lemma/lowercase substring over the prompt's lemma string and its
    lowercased text (multi-word surface forms). No regex: plain ``in`` checks on
    a spaCy-lemmatized token stream.
    """
    lemma_text = " " + " ".join(t.lemma_.lower() for t in doc) + " "
    lower_text = " " + doc.text.lower() + " "
    hard: set[str] = set()
    soft: set[str] = set()
    for name, (surfaces, hardness) in CONSTRAINT_LEXICON.items():
        hit = False
        for surface in surfaces:
            if " " in surface:
                if f" {surface} " in lower_text:
                    hit = True
                    break
            elif f" {surface} " in lemma_text:
                hit = True
                break
        if hit:
            (hard if hardness == "hard" else soft).add(name)
    return hard, soft


def extract_intent(prompt: str, nlp: "Language") -> PromptIntent:
    """Extract content lemmas, key phrases, and constraints from a prompt."""
    doc = nlp(prompt)
    hard, soft = _detect_constraints(doc)
    return PromptIntent(
        lemmas=frozenset(_content_lemmas(doc)),
        phrases=tuple(_key_phrases(doc)),
        hard_constraints=frozenset(hard),
        soft_constraints=frozenset(soft),
    )


# ===========================================================================
# Constraint predicates (model attribute checks).
# ===========================================================================
def constraint_satisfied(name: str, model: ModelDoc) -> bool:
    """Whether a model's attributes honor the named constraint."""
    if name == "open_source":
        return model.open_source
    if name == "edge":
        return model.deployable and (
            model.parameters is not None and model.parameters <= _SMALL_PARAM_MAX
        )
    if name == "long_context":
        return (
            model.context_window is not None
            and model.context_window >= _LONG_CONTEXT_MIN
        )
    if name == "multilingual":
        return "multilingual" in model.specialties
    if name == "fast":
        return model.deployable
    return True


# ===========================================================================
# SPARSE retrieval (IDF-weighted inverted index).
# ===========================================================================
class SparseIndex:
    """IDF-weighted content-lemma inverted index over ModelDocs."""

    def __init__(self, docs: list[ModelDoc]) -> None:
        self._docs = docs
        self._by_id = {d.row_id: d for d in docs}
        # lemma -> list of doc row_ids containing it (postings).
        self._postings: dict[str, list[str]] = defaultdict(list)
        doc_freq: dict[str, int] = defaultdict(int)
        for doc in sorted(docs, key=lambda d: d.row_id):
            for lemma in sorted(doc.lemmas):
                self._postings[lemma].append(doc.row_id)
                doc_freq[lemma] += 1
        n_docs = len(docs)
        self._idf = {
            lemma: math.log(n_docs / df) if df else 0.0
            for lemma, df in doc_freq.items()
        }

    def idf_of(self, lemma: str) -> float:
        return self._idf.get(lemma, 0.0)

    def retrieve(self, lemmas: frozenset[str], top_n: int) -> list[str]:
        """Return up to top_n candidate row_ids by IDF-weighted overlap.

        Ties break on row_id ascending for determinism.
        """
        scores: dict[str, float] = defaultdict(float)
        for lemma in sorted(lemmas):
            idf = self._idf.get(lemma, 0.0)
            if idf <= 0.0:
                continue
            for row_id in self._postings.get(lemma, ()):
                scores[row_id] += idf
        if not scores:
            return []
        ranked = sorted(scores, key=lambda rid: (-scores[rid], rid))
        return ranked[:top_n]

    def doc(self, row_id: str) -> ModelDoc:
        return self._by_id[row_id]

    @property
    def all_docs(self) -> list[ModelDoc]:
        return self._docs


# ===========================================================================
# DENSE rerank (granular phrase-vector similarity + soft bonuses).
# ===========================================================================
def _l2_normalize(vecs: list[object]) -> "np.ndarray":
    """Stack phrase vectors into an L2-normalized (n, dim) float32 matrix.

    Returns an empty (0, 0) array when there are no usable vectors. Model-side
    matrices are normalized ONCE at build (their vectors are fixed); only the
    per-prompt matrix is normalized at query time.
    """
    import numpy as np

    if not vecs:
        return np.empty((0, 0), dtype="float32")
    mat = np.asarray(vecs, dtype="float32")
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)


def _phrase_vectors(phrases: tuple[str, ...], nlp: "Language") -> list[object]:
    """Vectorize phrases; keep only those with a non-zero vector.

    All pipeline components are disabled: ``doc.vector`` is the mean of the
    md model's static word vectors, which needs only the tokenizer + vectors,
    so vectorization is cheap even over the whole DB's phrases.
    """
    import numpy as np

    vecs: list[object] = []
    disabled = list(nlp.pipe_names)
    for doc in nlp.pipe(phrases, batch_size=_PIPE_BATCH, disable=disabled):
        vec = np.asarray(doc.vector, dtype="float32")
        if doc.has_vector and float(np.linalg.norm(vec)) > 0.0:
            vecs.append(vec)
    return vecs


def _max_mean_similarity(prompt_norm: "np.ndarray", model_norm: "np.ndarray") -> float:
    """Mean over prompt phrases of the max cosine to any model phrase.

    Both inputs are already-L2-normalized (n, dim) matrices (see
    ``_l2_normalize``). Granular (phrase-to-phrase), not a single full-doc mean
    vector. Returns 0.0 if either side has no usable vectors.
    """
    if prompt_norm.shape[0] == 0 or model_norm.shape[0] == 0:
        return 0.0
    sims = prompt_norm @ model_norm.T  # (n_prompt, n_model) cosine matrix
    return float(sims.max(axis=1).mean())


@dataclass(frozen=True)
class Candidate:
    """A scored candidate model returned by the retriever."""

    row_id: str
    task_type: str
    domains: tuple[str, ...]
    specialties: tuple[str, ...]
    score: float


class CleanPickV2:
    """The V2 intent->model retriever (sparse retrieve + dense rerank)."""

    name = "clean_pick_v2"

    def __init__(
        self,
        docs: list[ModelDoc],
        nlp: "Language",
        task_predictor: harness.Predictor | None = None,
        popularity_tiebreak: bool = False,
    ) -> None:
        self._nlp = nlp
        self._index = SparseIndex(docs)
        self._task_predictor = task_predictor
        self._popularity_tiebreak = popularity_tiebreak
        # Precompute granular phrase-vectors once (dense side), already
        # L2-normalized: model vectors are fixed, so their norms never change
        # and are computed ONCE here rather than per prompt x candidate. Phrases
        # were extracted in the cheap build; vectorizing short phrases is fast
        # and needs no parser, so every query is then pure vector math.
        self._model_norms: dict[str, "np.ndarray"] = {
            d.row_id: _l2_normalize(_phrase_vectors(d.phrases, nlp)) for d in docs
        }

    @property
    def nlp(self) -> "Language":
        return self._nlp

    @property
    def index(self) -> SparseIndex:
        return self._index

    @property
    def task_predictor(self) -> harness.Predictor | None:
        return self._task_predictor

    def predicted_tasks(self, prompt: str) -> set[str]:
        """Prompt's predicted task_type top set (SOFT feature; never a gate)."""
        if self._task_predictor is None:
            return set()
        preds = self._task_predictor.predict_task_types(prompt)
        return set(preds[:_TASK_TOPN])

    def pick(
        self, prompt: str, k: int = _DEFAULT_K, intent: PromptIntent | None = None
    ) -> list[Candidate]:
        """Return ranked top-k candidates, or [] to abstain/clarify.

        Abstains when the intent has no content lemmas or no candidate survives
        the sparse stage / hard constraints. ``intent`` may be passed in when the
        caller already parsed the prompt (avoids a second full spaCy parse).
        """
        if intent is None:
            intent = extract_intent(prompt, self._nlp)
        if not intent.lemmas:
            return []
        cand_ids = self._index.retrieve(intent.lemmas, _SPARSE_TOP_N)
        if not cand_ids:
            return []

        predicted_tasks = self.predicted_tasks(prompt)
        prompt_norm = _l2_normalize(_phrase_vectors(intent.phrases, self._nlp))

        scored: list[Candidate] = []
        for row_id in cand_ids:
            doc = self._index.doc(row_id)
            # Hard constraints filter.
            if any(
                not constraint_satisfied(c, doc) for c in intent.hard_constraints
            ):
                continue
            dense = _max_mean_similarity(prompt_norm, self._model_norms[row_id])
            score = dense
            # SOFT task bonus (never a gate).
            if doc.task_type in predicted_tasks:
                score += _SOFT_TASK_BONUS
            # SOFT constraint bonuses.
            for c in intent.soft_constraints:
                if constraint_satisfied(c, doc):
                    score += _SOFT_CONSTRAINT_BONUS
            scored.append(
                Candidate(
                    row_id=row_id,
                    task_type=doc.task_type,
                    domains=doc.domains,
                    specialties=doc.specialties,
                    score=score,
                )
            )
        if not scored:
            return []

        # Deterministic ranking. Popularity is NOT in the fit score; it is only
        # an OFF-by-default final tie-break (middle key element). Default
        # tie-break is row_id asc.
        def key(c: Candidate) -> tuple[float, int, str]:
            likes = -self._index.doc(c.row_id).likes if self._popularity_tiebreak else 0
            return (-c.score, likes, c.row_id)

        scored.sort(key=key)
        return scored[:k]
