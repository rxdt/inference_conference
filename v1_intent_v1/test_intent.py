"""Executable tests for the intent extractors.

Run as a script (prints accuracy) or under pytest:
    .venv/bin/python test_intent.py
    .venv/bin/python -m pytest test_intent.py

Thresholds are set comfortably below the documented honest results (approach #1
42.69% top1 / 44.17% in-list; approach #2 22.85% / 23.54%) so the suite asserts
"did not regress" rather than pinning an exact number.
"""

from __future__ import annotations

from config import TASK_TAXONOMY
from intent import extract_intent, extract_intent_by_matching_to_db_metadata
from prompts import PROMPT_MATRIX

CANON = set(TASK_TAXONOMY["canonical"])


def _accuracy(fn) -> tuple[float, float]:
    hit = inlist = 0
    for p in PROMPT_MATRIX:
        ranked = fn(p["prompt"])["task_type"]
        assert ranked, "extractor returned empty ranking"
        assert ranked[0] in CANON, f"predicted non-canonical task {ranked[0]!r}"
        hit += ranked[0] == p["expected_task_type"][0]
        inlist += ranked[0] in p["expected_task_type"]
    n = len(PROMPT_MATRIX)
    return hit / n, inlist / n


def test_extract_intent_shape():
    r = extract_intent("a math tester for children in school")
    assert isinstance(r, dict) and r["task_type"]
    assert r["task_type"][0] in CANON
    assert len(r["task_type"]) == len(CANON)  # full ranking


def test_extract_intent_deterministic():
    a = extract_intent("summarize this long report into a few bullet points")[
        "task_type"
    ]
    b = extract_intent("summarize this long report into a few bullet points")[
        "task_type"
    ]
    assert a == b  # tie-broken by task name -> stable


def test_extract_intent_accuracy():
    top1, inlist = _accuracy(extract_intent)
    assert top1 >= 0.40, f"approach #1 top1 regressed: {top1:.3%}"
    assert inlist >= 0.42, f"approach #1 in-list regressed: {inlist:.3%}"


def test_db_matching_shape():
    r = extract_intent_by_matching_to_db_metadata(
        "transcribe my voicemail recordings to text"
    )
    assert isinstance(r, dict) and r["task_type"][0] in CANON


def test_db_matching_accuracy():
    top1, inlist = _accuracy(extract_intent_by_matching_to_db_metadata)
    assert top1 >= 0.20, f"approach #2 top1 regressed: {top1:.3%}"
    assert inlist >= 0.21, f"approach #2 in-list regressed: {inlist:.3%}"


if __name__ == "__main__":
    t1, l1 = _accuracy(extract_intent)
    print(
        f"approach #1 extract_intent:                  top1={t1:.2%} in-list={l1:.2%}"
    )
    t2, l2 = _accuracy(extract_intent_by_matching_to_db_metadata)
    print(
        f"approach #2 db-metadata matching:            top1={t2:.2%} in-list={l2:.2%}"
    )
    print("OK" if t1 >= 0.40 and t2 >= 0.20 else "REGRESSION")
