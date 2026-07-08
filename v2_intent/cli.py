"""V2 CLI — free-text prompt -> best-matching DB models (the clean_pick_v2 retriever).

This is the scientist_v2 system: SEMANTIC RETRIEVAL, not task-type classification.
It runs the shipped hybrid retriever `clean_pick_v2.CleanPickV2.pick`:
    sparse IDF retrieve -> granular phrase-vector dense rerank + constraint fit.

It uses spaCy en_core_web_md + builtins only; DB is read-only; NO config.py, NO
network, NO LLM. Deterministic (ties break on model id).

  NOTE (honesty): the previous cli.py in this repo was a V1 artifact (task
  classifier importing config/harness_db/buckets); this file REPLACES it and
  invokes the actual V2 retriever.

Usage:
  python cli.py "your free-text goal here"
  python cli.py            # then type a prompt at the 'prompt>' stdin prompt

First run builds the per-model representation over ~13k models and CACHES it to
.cli_cache_docs.pkl (a few minutes); later runs load the cache and are fast.
The soft phase2 task bonus is OFF in the CLI (kept out for speed); the core
sparse+dense+constraint scoring is identical to the evaluated system. Pass
--with-task-bonus to include it (slower first build).
"""

from __future__ import annotations

import harness  # noqa: F401  # FIRST: pins BLAS threads before numpy/spacy.

import pickle  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import clean_pick_v2 as cp  # noqa: E402

_CACHE = Path(__file__).with_name(".cli_cache_docs.pkl")


def _load_or_build_docs() -> list[cp.ModelDoc]:
    """Load cached ModelDocs, else build once (parser-disabled) and cache them."""
    if _CACHE.exists():
        with _CACHE.open("rb") as fh:
            return pickle.load(fh)  # noqa: S301  # our own cache, local only
    print("building model index over the catalog (one-time, a few minutes)...",
          file=sys.stderr, flush=True)
    docs = cp.build_model_docs(cp.load_raw_models(), cp.load_build_nlp())
    with _CACHE.open("wb") as fh:
        pickle.dump(docs, fh)
    return docs


def _run(prompt: str, with_task_bonus: bool, k: int = 5) -> int:
    nlp = cp.load_nlp()
    docs = _load_or_build_docs()
    task_predictor = None
    if with_task_bonus:
        import clean_pick_metric as cm  # builds the full phase2 blend (slower)
        retr = cm._build_retriever(nlp)  # noqa: SLF001
    else:
        retr = cp.CleanPickV2(docs, nlp, task_predictor=task_predictor)

    picks = retr.pick(prompt, k=k)
    if not picks:
        print("(abstain) — the prompt has no clear intent or no candidate matched.")
        return 0
    print(f"top {len(picks)} models for the intent:\n")
    for i, c in enumerate(picks, 1):
        dom = ", ".join(c.domains) or "-"
        spec = ", ".join(c.specialties) or "-"
        print(f"{i}. id={c.row_id}  task={c.task_type}  score={c.score:.4f}")
        print(f"     domains=[{dom}]  specialties=[{spec}]")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    with_bonus = "--with-task-bonus" in args
    args = [a for a in args if a != "--with-task-bonus"]
    prompt = " ".join(args).strip()
    if not prompt:
        try:
            prompt = input("prompt> ").strip()
        except EOFError:
            prompt = ""
    if not prompt:
        print('usage: python cli.py "your free-text goal"  [--with-task-bonus]')
        return 1
    return _run(prompt, with_task_bonus=with_bonus)


if __name__ == "__main__":
    raise SystemExit(main())
