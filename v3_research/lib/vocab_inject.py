"""Inject a vocabulary from ``config`` into a spaCy pipeline (config never revealed).

``config`` is imported at RUNTIME and treated as radioactive: we iterate its
vocabulary term objects and inject them into the pipeline as (a) static vectors
on ``nlp.vocab`` and (b) a ``PhraseMatcher`` keyed by the terms. We compute over
the terms but NEVER print, return-as-text, or persist any config string. The
only artifacts returned are the built matcher and an integer count of injected
keys.

Approach (no config text quoted): the config module is expected to expose one or
more iterables of short term strings (a domain lexicon). We flatten every string
we find at the top level of the module — walking plain ``str``, and the members
of any ``list``/``tuple``/``set``/``dict`` (dict keys and string values) — into a
de-duplicated term set. Each term is added to a ``PhraseMatcher`` (on the shared
vocab) and given a zeroed-then-averaged static vector so downstream similarity
can see it. Counts only; contents stay internal.
"""

import importlib
import sys
from collections.abc import Iterator
from pathlib import Path

from spacy.language import Language
from spacy.matcher import PhraseMatcher

_REPO_ROOT = str(Path(__file__).resolve().parents[3])


def _dict_strings(obj: dict[object, object]) -> Iterator[str]:
    """Yield string keys and string values of a mapping."""
    for k, v in obj.items():
        if isinstance(k, str):
            yield k
        if isinstance(v, str):
            yield v


def _iter_strings(obj: object) -> Iterator[str]:
    """Yield term strings from a config value without echoing them elsewhere.

    Handles a top-level ``str`` and one level of container (list/tuple/set/dict).
    Dict keys and string values are both harvested. Non-strings are ignored.
    """
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        yield from _dict_strings(obj)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        yield from (item for item in obj if isinstance(item, str))


def _collect_terms() -> list[str]:
    """Import ``config`` at runtime and collect a de-duplicated term list.

    The returned list stays internal to :func:`build`; it is never emitted.
    """
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    config = importlib.import_module("config")

    seen: set[str] = set()
    ordered: list[str] = []
    for name in dir(config):
        if name.startswith("__"):
            continue
        for raw in _iter_strings(getattr(config, name)):
            term = raw.strip()
            if term and term not in seen:
                seen.add(term)
                ordered.append(term)
    return ordered


def build(nlp: Language) -> tuple[PhraseMatcher, int]:
    """Inject config vocabulary into ``nlp``; return (matcher, injected_count).

    Builds a ``PhraseMatcher`` over the terms and sets a static vector on
    ``nlp.vocab`` for each (using the mean of its constituent token vectors, or a
    zero vector when out-of-vocabulary). Returns ONLY the matcher object and an
    integer count of injected keys — never any term string.
    """
    import numpy as np  # noqa: PLC0415  lazy: numpy only needed when building

    terms = _collect_terms()
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    width = nlp.vocab.vectors_length or 300
    patterns = []
    for term in terms:
        doc = nlp.make_doc(term)
        patterns.append(doc)
        vec = doc.vector if len(doc) and doc.has_vector else np.zeros(width, dtype="float32")
        nlp.vocab.set_vector(term, vec)

    if patterns:
        matcher.add("CONFIG_VOCAB", patterns)
    return matcher, len(patterns)
