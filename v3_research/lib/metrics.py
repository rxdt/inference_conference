"""Label-only evaluation metrics (pure functions, deterministic).

All functions operate on label STRINGS and ranked prediction lists — never on
prompt or config text. The label->family map is built ONCE over the full set of
first-labels present in the DB (see ``family_map``).

Family taxonomy (8 families):
    text, image, audio, video, 3d, multimodal, tabular, other

Family assignment rule (documented, regex-free)
-----------------------------------------------
Each label is a hyphenated HF pipeline tag (e.g. ``image-text-to-text``). We
tokenize on ``-`` and inspect the tokens with plain substring/membership checks:

  * MODALITY words are text / image / audio / video, with ``visual`` and
    ``question``/``answering`` treated as image- and text-modality signals
    respectively. If the label mixes two or more DISTINCT modalities it is
    ``multimodal`` — e.g. ``image-text-to-text``, ``audio-text-to-text``,
    ``visual-question-answering`` (image + text).
  * Else a single dominant modality maps to that family, checked in order:
      - ``audio`` / ``speech`` / ``voice``        -> audio
      - ``video``                                  -> video
      - ``3d``                                     -> 3d
      - ``image`` / ``visual`` (strong modality)   -> image
      - ``tabular`` / ``time-series`` / ``graph``  -> tabular
      - ``text`` / ``token`` / ``translation`` /
        ``summarization`` / ``sentence`` / ``question`` / ``fill`` /
        ``ranking`` / ``retrieval`` / ...          -> text
      - ``depth`` / ``mask`` / ``keypoint`` /
        ``object`` / ``segmentation`` (weak image) -> image
  * Anything unmatched (e.g. ``reinforcement-learning``, ``robotics``,
    ``feature-extraction``, ``multiple-choice``) -> ``other``.

Ordering matters: strong image words (``image``/``visual``) win before text, so
``image-classification`` -> image; but weak image words (``mask`` etc.) are
checked AFTER text, so ``fill-mask`` -> text (``fill``) while ``mask-generation``
-> image. The mapping is precomputed as a plain dict; lookups are O(1) and
deterministic.
"""

import sqlite3
from collections.abc import Sequence

from lib import db

FAMILIES = ("text", "image", "audio", "video", "3d", "multimodal", "tabular", "other")

# Modality detectors: each maps a canonical modality to the needles that signal
# it. Two+ DISTINCT canonical modalities present in one label force ``multimodal``.
# ``visual`` counts as the image modality; ``question``/``answering`` as text
# (so ``visual-question-answering`` = image + text = multimodal).
_MODALITY_NEEDLES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("text", ("text", "question", "answering")),
    ("image", ("image", "visual")),
    ("audio", ("audio",)),
    ("video", ("video",)),
)

# Ordered single-modality routing (checked top-to-bottom after the multimodal
# test); first family whose needles match wins. Strong image words
# (``image``/``visual``) precede text; weak image words (``mask`` etc.) follow it
# so ``fill-mask`` -> text via ``fill`` while ``mask-generation`` -> image.
_SINGLE_MODALITY: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("audio", ("audio", "speech", "voice")),
    ("video", ("video",)),
    ("3d", ("3d",)),
    ("image", ("image", "visual")),
    ("tabular", ("tabular", "series", "graph")),
    (
        "text",
        (
            "text", "token", "translation", "summarization", "sentence",
            "question", "fill", "ranking", "retrieval", "classification",
            "choice", "answering",
        ),
    ),
    ("image", ("depth", "mask", "keypoint", "object", "segmentation")),
)


def _has_any(tokens: Sequence[str], needles: Sequence[str]) -> bool:
    """True if any needle is a token or a substring of any token."""
    for tok in tokens:
        for needle in needles:
            if needle in tok:
                return True
    return False


def label_family(label: str) -> str:
    """Classify one label into a family (regex-free, deterministic)."""
    tokens = label.split("-")
    # Count distinct canonical modalities present -> multimodal when >= 2.
    present = {mod for mod, needles in _MODALITY_NEEDLES if _has_any(tokens, needles)}
    if len(present) >= 2:
        return "multimodal"
    for family, needles in _SINGLE_MODALITY:
        if _has_any(tokens, needles):
            return family
    return "other"


def family_map(con: sqlite3.Connection | None = None) -> dict[str, str]:
    """Build the TOTAL label->family map over every first-label in the DB.

    Queries distinct ``task_type`` values, normalizes each to its first label,
    and assigns a family via :func:`label_family`. Deterministic.
    """
    owns = con is None
    if owns:
        con = db.connect()
    try:
        labels: set[str] = set()
        for row in con.execute("SELECT DISTINCT task_type FROM models"):
            lab = db.norm_task(row[0])
            if lab is not None:
                labels.add(lab)
    finally:
        if owns:
            con.close()
    return {lab: label_family(lab) for lab in sorted(labels)}


def top1(pred: Sequence[str], golds: Sequence[str]) -> bool:
    """True if the top-1 prediction is in the gold set."""
    if not pred:
        return False
    return pred[0] in set(golds)


def in_list_at_k(pred: Sequence[str], golds: Sequence[str], k: int) -> bool:
    """True if any gold label appears within the top-k predictions."""
    if not pred or k <= 0:
        return False
    goldset = set(golds)
    return any(p in goldset for p in pred[:k])


def far_error(
    pred: Sequence[str], golds: Sequence[str], fam: dict[str, str]
) -> bool:
    """True if the top-1 prediction's family differs from EVERY gold family.

    A "far" mistake: not merely the wrong label, but the wrong modality family.
    Requires a non-empty prediction and at least one gold; unknown labels map to
    family ``"other"``.
    """
    if not pred or not golds:
        return False
    pred_fam = fam.get(pred[0], "other")
    gold_fams = {fam.get(g, "other") for g in golds}
    return pred_fam not in gold_fams


def baselines(
    golds: Sequence[Sequence[str]],
    n_labels: int,
    majority_label: str,
) -> dict[str, object]:
    """Compute trivial baselines against a sequence of per-item gold lists.

    Returns ``random_top1`` (=1/n_labels), ``majority_top1`` (fraction of items
    whose gold set contains ``majority_label``), ``majority_label``, ``n_labels``.
    Deterministic.
    """
    n = len(golds)
    hits = sum(1 for g in golds if majority_label in set(g))
    return {
        "random_top1": (1.0 / n_labels) if n_labels else 0.0,
        "majority_top1": (hits / n) if n else 0.0,
        "majority_label": majority_label,
        "n_labels": n_labels,
    }
