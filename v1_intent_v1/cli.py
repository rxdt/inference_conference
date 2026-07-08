"""Tiny CLI: free-text prompt -> inferred task type, bucket, capabilities, and the
best concrete DB model.

End-to-end product path:
  prompt --(approach #1: extract_intent)--> task_type ranking
         --(buckets.py)----------------->  routing bucket
         --(task_to_family map)---------->  capability families (derived)
         --(approach #2: DB retrieval)---->  best-matching model of that task_type

Usage:
  python cli.py "transcribe my voicemail recordings to text"
  python cli.py            # then type a prompt at the prompt (stdin)
"""

from __future__ import annotations

import os
import sqlite3
import sys

import numpy as np

import config as cfg
import harness_db as _DB
from buckets import TASK_TO_BUCKET
from intent import extract_intent

_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tempjune13.db")
_FAMILIES = cfg.TASK_TAXONOMY["task_to_family_for_user_only"]


def _top_models(prompt: str, task: str, n: int = 3) -> list[dict]:
    """Top-`n` DB models by prompt↔model cosine whose task_type[0] matches `task`
    (falls back to global nearest if no model carries that task). Each is enriched
    with metadata from the DB. Deterministic: ties broken by descending similarity
    then model id."""
    z = np.load(_DB.CACHE, allow_pickle=True)
    vecs = _DB.l2norm(z["vecs"].astype("float32"))
    tasks = z["tasks"].astype(str)
    ids = z["ids"].astype(str)

    pv = _DB.l2norm(_DB.task_vec(_DB.nlp()(prompt))[None, :])[0]
    sims = pv @ vecs.T
    cand = np.where(tasks == task)[0]
    if cand.size == 0:
        cand = np.arange(ids.size)
    # stable top-n: sort by (-similarity, id)
    order = sorted(cand, key=lambda i: (-float(sims[i]), str(ids[i])))[:n]

    con = None
    try:
        con = sqlite3.connect(_DB_PATH)
    except sqlite3.Error:
        con = None
    out: list[dict] = []
    for i in order:
        mid = str(ids[i])
        info = {
            "id": mid,
            "model_task": str(tasks[i]),
            "similarity": round(float(sims[i]), 4),
        }
        if con is not None:
            row = con.execute(
                "SELECT pipeline_tag, short_description, likes, downloads, "
                "capability_families, proprietary FROM models WHERE id=?",
                (mid,),
            ).fetchone()
            if row:
                info.update(
                    pipeline_tag=row[0],
                    description=row[1],
                    likes=row[2],
                    downloads=row[3],
                    capability_families=row[4],
                    proprietary=bool(row[5]),
                )
        # Hugging Face model page URL (proprietary models are not on the Hub).
        info["url"] = (
            None if info.get("proprietary") else f"https://huggingface.co/{mid}"
        )
        out.append(info)
    if con is not None:
        con.close()
    return out


def cli(prompt: str | None = None) -> dict:
    """Parse one prompt and print the inferred intent + best model to stdout."""
    if prompt is None:
        prompt = " ".join(sys.argv[1:]).strip() or input("prompt> ").strip()
    if not prompt:
        print("no prompt given")
        return {}

    res = extract_intent(prompt)
    ranked, scores = res["task_type"], res["scores"]
    top = ranked[0]
    top3_tasks = [(t, round(scores[t], 4)) for t in ranked[:3]]
    models = _top_models(prompt, top, n=3)

    # ---- inferred attributes ----
    print(f'\nprompt           : "{prompt}"')
    print("--- inferred from prompt ---")
    print(f"task_type        : {top}")
    print(f"  alt task_types : {top3_tasks}")
    print(f"bucket           : {TASK_TO_BUCKET[top]}")
    print(f"capabilities     : {_FAMILIES.get(top, [])}")

    # ---- top-3 recommended models ----
    print("--- top 3 recommended models ---")
    for rank, m in enumerate(models, 1):
        tag = " [PROPRIETARY]" if m.get("proprietary") else ""
        url = m.get("url")
        print(f"  {rank}. {m['id']}{tag}" + (f"  {url}" if url else ""))
        print(
            f"       task={m['model_task']}  similarity={m['similarity']}"
            f"  likes/dl={m.get('likes')}/{m.get('downloads')}"
            f"  pipeline={m.get('pipeline_tag')}"
        )
        desc = (m.get("description") or "").strip()
        if desc:
            print(f"       {desc[:160]}")
    print()
    return {
        "prompt": prompt,
        "task_type": top,
        "alt_task_types": top3_tasks,
        "bucket": TASK_TO_BUCKET[top],
        "capabilities": _FAMILIES.get(top, []),
        "top_models": models,
    }


if __name__ == "__main__":
    cli()
